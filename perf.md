# Performance

## Current: GPT-2 small (124M params) decode

- **4.3 tok/s** (233ms/token) single-threaded on Ryzen 9 7900X3D
- Reference: llama.cpp does ~100+ tok/s on similar hardware (~25x faster)
- 232ms/token breakdown: **99% JIT execution**, <1% setup, <1% argmax/KV update

## Pipeline unification (done)

Both compilation paths now share the same transform schedule and pass
pipeline. Graph_builder emits `linalg.generic` for matmuls (not named
ops), calls `build_transform_schedule`, and runs the full pipeline:
tiling → vectorization → OpenMP → LLVM.

**Current blocker**: LLVM backend compilation (O3 + instruction selection)
takes ~10 minutes on the tiled+vectorized GPT-2 IR. The MLIR passes
complete successfully. The `.so` would be cached after first compile.

Options to fix compilation time:
- Parallel codegen: split MLIR module into per-function .o files, compile
  on separate threads, link together
- Size threshold: skip tiling for small contractions that don't benefit
- Accept one-time cost: cached .so makes subsequent runs instant

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

### Phase 0: Unify pipelines — DONE

Pipelines unified. Graph_builder now emits `linalg.generic` for matmuls
and shares `build_transform_schedule` with the compiler path. Both paths
have tiling, vectorization, and OpenMP.

**Blocker**: LLVM backend O3 compilation takes ~10 minutes on the
tiled+vectorized GPT-2 IR (stuck in X86 instruction selection on the
large vectorized module). The `.so` is cached after first compile.

**Options to fix**:
- Split module into per-function compilation units, compile in parallel
- Add size threshold to matcher — skip tiling tiny contractions
- Accept one-time cost (cached)

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

**Status**: Implemented in both paths. `tile_using_forall` + OpenMP passes
are in the shared `build_transform_schedule` and unified pipeline.

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

| Phase | Change | Status | Target tok/s |
|---|---|---|---|
| 0 | Tiling + vectorization + OpenMP in graph_builder | **Done** (blocked by slow LLVM compile) | unmeasured |
| 1 | Multi-threading (OpenMP) | Done (part of Phase 0) | ×6-10 |
| 2 | Packed GEMM layout | Not started | ×2-4 |
| 3 | AVX-512 + unroll + loop order | Not started | ×1.5-2 |
| 4 | Fusion + weight prepack + KV quant | Not started | ×1.3-2 |

Immediate next step: fix LLVM backend compilation time so we can
measure actual tok/s with the new pipeline. Until then, performance
impact of Phase 0+1 is unknown.

All phases use MLIR transform dialect + standard passes. No external
BLAS, no hand-written assembly. The compiler generates everything.
