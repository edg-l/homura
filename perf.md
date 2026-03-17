# Performance

## Current: GPT-2 small (124M params) decode

- **4.3 tok/s** (233ms/token) single-threaded on Ryzen 9 7900X3D
- Reference: llama.cpp does ~100+ tok/s on similar hardware (~25x faster)
- 232ms/token breakdown: **99% JIT execution**, <1% setup, <1% argmax/KV update

## Critical issue: two compilation paths

There are two separate compilation pipelines:

1. **`Compiler::compile`** (tensor trace API) — has the transform schedule
   with tiling (32×32×32 cache + 8×8×1 register), padding, vectorization,
   and OpenMP support. Used by unit tests.

2. **`GraphBuilder::compile_with_cache`** (graph_builder) — used by ONNX
   model inference (GPT-2). Has **none of the above**: no transform
   schedule, no tiling, no vectorization, no OpenMP. Goes straight from
   `linalg → loops → LLVM` with naive scalar code.

**GPT-2 runs through path 2.** The 4.3 tok/s measurement is on completely
unoptimized scalar loops — not "tiled but naive" as previously assumed.
This means the actual performance gap is much larger than expected, and
adding the transform schedule to graph_builder should give a massive
improvement even before threading.

### Phase 0: Unify compilation pipelines (prerequisite for everything)

Add the transform schedule and optimized pass pipeline to graph_builder's
`compile_with_cache`. This is a prerequisite for all other phases.

**Changes needed in `graph_builder.rs`**:
1. Call `build_transform_schedule` before the pass pipeline (requires
   adding `transform.with_named_sequence` module attribute).
2. Add `transform-interpreter` + `canonicalize,cse` + `symbol-dce` to the
   pipeline, before bufferization.
3. Add `func.func(lower-vector-multi-reduction, lower-vector-mask)` for
   vector lowering (the transform schedule vectorizes contractions).
4. Add `scf-forall-to-parallel`, `convert-scf-to-openmp`,
   `convert-openmp-to-llvm` in the correct positions (see Compiler::compile
   pipeline for the working order).
5. Add `convert-vector-to-scf` and `convert-vector-to-llvm` passes.

**Expected impact**: Going from scalar loops to tiled+vectorized code
should give a **10-50x improvement on its own**, before any threading.

## Bottlenecks

### 1. Single-threaded execution

All `scf.for` loops are sequential. GPT-2 decode has ~24 matmuls per
token. Each runs on one core out of 12 available.

**Fix**: Use `tile_using_forall` in the transform schedule to tile outer
parallel dims into `scf.forall`, which maps to parallel threads.
Requires a threading runtime (OpenMP or pthreads via MLIR's async
dialect). Expected ~6-10x speedup.

### 2. Naive GEMM kernels

After Phase 0, generated code will be tiled (32x32x32 cache + 8x8x1
register) and vectorized (`vector.contract` → AVX2), but still far from
hand-tuned BLAS. Missing:
- Register blocking with explicit accumulator registers
- Software pipelining (prefetch next tile while computing current)
- Optimal loop ordering for the micro-kernel
- AVX-512 utilization (CPU supports it)

**Fix** (MLIR-native, no external BLAS):
- `transform.structured.pack` for packed GEMM data layout (eliminates
  TLB misses, enables streaming — this is how BLIS/GotoBLAS get speed)
- Tune tile sizes and loop ordering for the micro-kernel
- Explicit unroll-and-jam via transform schedule
- AVX-512 targeting via wider vector_sizes

### 3. No memory prefetching

Cache tiles load data on demand. For large matmuls (768x3072), the
working set exceeds L1. Prefetching the next tile while computing the
current would hide memory latency.

### 4. KV cache memory traffic

Each decode step reads/writes the full KV cache ([1, 12, seq_len, 64]
× 24 layers). As seq_len grows, this becomes memory-bandwidth bound.

**Fix**: Quantize KV cache to FP16/INT8, or use paged attention.

## Improvement roadmap

All MLIR-native — no external BLAS. Each phase builds on the previous.

### Phase 0: Unify pipelines (10-50x → ~40-200 tok/s)

Add the transform schedule (tiling + vectorization) to graph_builder.
This is the single biggest win because GPT-2 is currently running
completely unoptimized scalar loops.

See "Critical issue" section above for implementation details.

### Phase 1: Multi-threading (6-10x → ~25-40 tok/s)

Parallelize outer tile loops across CPU cores. This is the single
biggest win because we have 12 cores doing nothing.

**Transform schedule changes**:
1. Replace `tile_using_for` (cache-level) with `tile_using_forall` for
   the M and N parallel dims. K (reduction) stays sequential.
2. The tiled contraction loops become `scf.forall` instead of `scf.for`.
3. Inner register tile + pad + vectorize stay the same (run per-thread).

**Pass pipeline changes**:
1. After bufferization: `scf-forall-to-parallel` converts
   `scf.forall` → `scf.parallel`.
2. After `lower-affine`: `convert-scf-to-openmp` converts
   `scf.parallel` → OpenMP worksharing constructs.
3. `convert-openmp-to-llvm` **before** `convert-scf-to-cf` (critical
   ordering: omp ops must be lowered before scf-to-cf creates
   multi-block regions inside `memref.alloca_scope`).
4. Link against `libomp.so` at link time (derived from
   `MLIR_SYS_210_PREFIX`).

**Key details**:
- `tile_using_forall` requires `operandSegmentSizes` attribute
  (`array<i32: 1, 0, 0, 0, 0>`) and `static_num_threads` (empty array).
- `tile_using_forall` uses `static_tile_sizes` (not `static_sizes`).
- Only parallelize the OUTER (cache) tile level. The inner register
  tile runs sequentially per thread — this is the standard approach.

**Status**: Implemented in `Compiler::compile` path (commit 558756e).
Needs to be carried over to graph_builder as part of Phase 0.

**Verification**: measure with `perf stat` to confirm all cores active.
Target: 6-10x speedup → **25-40 tok/s**.

### Phase 2: Packed GEMM layout (2-4x → ~50-120 tok/s)

Rearrange matrix data into a packed, cache-friendly layout before the
tiled computation. This is the key technique from GotoBLAS/BLIS that
eliminates TLB misses on large matrices.

**Why it matters**: For a 768×3072 matmul tiled at 32×32, each tile
access strides across the full row (768 or 3072 elements apart). This
causes TLB misses because non-adjacent cache lines are touched. Packing
copies each tile's data into a contiguous buffer so the micro-kernel
streams sequentially through memory.

**Transform schedule changes**:
1. After matching the contraction, use `transform.structured.pack` to
   pack the LHS and RHS operands into tiled layout:
   ```
   // Pack A[M, K] into A_packed[M/mc, K/kc, mc, kc]
   // Pack B[K, N] into B_packed[K/kc, N/nc, kc, nc]
   transform.structured.pack %op
       packed_sizes [32, 32, 32]  // mc, nc, kc matching cache tile
   ```
2. Use `transform.structured.pack_transpose` to reorder the inner tile
   dims for optimal micro-kernel access (column-major for B panel).
3. Tile the packed op (the outer dims are the tile-loop iterators).
4. The inner `[mc, nc, kc]` block is the micro-kernel — tile to
   register level [8, 8, 1], pad, vectorize as we do now.

**Pass pipeline**: No new passes needed. `tensor.pack`/`tensor.unpack`
lower through `one-shot-bufferize` → memref copies, which the existing
pipeline handles.

**Key details**:
- Packing adds an O(N²) copy cost, but it's a streaming write (fast)
  and the packed data is reused across all tiles in the inner loop.
- For small matrices (e.g., [1, 768] × [768, 3072] in GPT-2 decode),
  the B matrix packing can be hoisted out of the decode loop since
  weights don't change between tokens.
- `pack_greedily` can auto-detect optimal packing for all contractions.

**Verification**: measure L1/TLB misses with `perf stat -e dTLB-load-misses`.
Target: 2-4x on top of threading → **50-120 tok/s**.

### Phase 3: Micro-kernel tuning (1.5-2x → ~80-200 tok/s)

Squeeze more out of each core's compute throughput.

**3a. AVX-512 vectors**

The Ryzen 7900X3D supports AVX-512. Current register tile is 8×8×1
targeting AVX2 (`vector<8xf32>`). With AVX-512:
- Widen to `vector<16xf32>` (512-bit)
- Register tile becomes 16×16×1 or 8×16×1
- Change `vector_sizes` and `pad_to_multiple_of` accordingly
- 2x throughput per vector op vs AVX2

**Note**: Zen 4 executes AVX-512 as two 256-bit uops. Real speedup is
~1.3-1.5x, not 2x. But the reduced loop overhead still helps.

**3b. Unroll-and-jam**

Unroll the K (reduction) loop by 4 and interleave FMAs from different
K iterations. This hides FMA latency (5 cycles on Zen 4) by keeping
multiple independent accumulations in flight.

```mlir
// Instead of K=1 per inner iteration:
// Change register tile to [8, 8, 4] and unroll K
transform.structured.tile_using_for %op tile_sizes [8, 8, 4]
// The vectorizer produces 4 independent FMAs per iteration
```

**3c. Loop permutation**

Reorder the register-tile loops for optimal register reuse:
- Current: M → N → K (K innermost)
- Better: K → M → N (broadcast B row, accumulate across M)

Use `tile_using_for` `interchange` attribute:
```
tile_sizes [8, 8, 1] interchange = [2, 0, 1]  // K, M, N order
```

**3d. Prefetch intrinsics**

Insert `llvm.prefetch` before each cache tile to overlap data fetch
with compute. MLIR doesn't have a prefetch op, but we can emit it
via `llvm.intr.prefetch` in a custom lowering or post-LLVM-lowering
pass.

### Phase 4: Inference-specific optimizations

**4a. Weight prepacking at load time**

Pack weight matrices once when the model loads, store the packed layout
in the compilation cache. Each decode step uses the pre-packed weights
directly — zero packing overhead at inference time.

**4b. Operator fusion**

Fuse elementwise ops (bias add, ReLU, layer norm) into the matmul
epilogue. Instead of writing the matmul result to memory and reading
it back for the next op, compute the chain in-register.

MLIR approach: use `transform.structured.fuse_into_containing_op` to
pull consumer ops into the tiled matmul loop body.

**4c. KV cache quantization**

As sequence length grows, KV cache becomes memory-bandwidth bound.
Quantize to FP16 or INT8 to halve/quarter the memory traffic.
Requires `arith.truncf`/`arith.extf` at the KV cache boundary.

## Summary

| Phase | Change | Target tok/s | Cumulative |
|---|---|---|---|
| Current | Unoptimized scalar loops (!) | 4.3 | 4.3 |
| 0 | Tiling + vectorization in graph_builder | ×10-50 | 40-200 |
| 1 | Multi-threading (OpenMP) | ×6-10 | 250-2000 |
| 2 | Packed GEMM layout | ×2-4 | 500-8000 |
| 3 | AVX-512 + unroll + loop order | ×1.5-2 | 750-16000 |
| 4 | Fusion + weight prepack + KV quant | ×1.3-2 | 1000-32000 |

Note: Phase 0 targets are speculative — the actual improvement from
adding tiling+vectorization to the graph_builder path hasn't been
measured yet. Upper estimates are likely unrealistic due to memory
bandwidth limits. Real-world target after all phases: ~100-300 tok/s
(matching llama.cpp).

All phases use MLIR transform dialect + standard passes. No external
BLAS, no hand-written assembly. The compiler generates everything.
