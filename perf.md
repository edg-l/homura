# Performance

## Compilation performance

Per-kernel compilation (IREE-style): each heavy op gets its own MLIR
context + module, compiled in parallel via rayon.

| Model | Kernels | Compile time | Notes |
|---|---|---|---|
| MNIST | 6 | 97ms | 2 Conv + MatMul + lightweight |
| ResNet-18 | 41 | 467ms | 20 Conv + Gemm + BatchNorm groups |
| GPT-2 | ~150+ | untested | Per-kernel approach, should scale well |

Compilation is cached per-kernel. Subsequent runs with same model skip
compilation entirely.

### What makes it fast

1. **Per-kernel modules**: Each kernel is a small MLIR module (~20-50 ops).
   MLIR passes run in milliseconds per kernel vs 40+ seconds monolithic.

2. **Parallel compilation**: All kernels compile simultaneously via rayon
   (12 threads on Ryzen 9 7900X3D). Wall clock = slowest kernel.

3. **Targeted vectorization**: Only the tiled micro-kernel gets vectorized
   (via outline → vectorize_children_and_apply_patterns → LLVM inline).
   Untiled ops (BatchNorm, Relu) stay as scalar loops with compact IR.

4. **Conv tiling**: Conv2D matched by operation name, tiled at cache level
   [0,32,4,4,0,0,0] and register level [0,8,1,1,1,0,0]. Produces 7KB
   LLVM bitcode vs 200-500KB untiled.

## Runtime performance

### GPT-2 small (124M params) decode

- **4.3 tok/s** (233ms/token) single-threaded on Ryzen 9 7900X3D
- Reference: llama.cpp does ~100+ tok/s on similar hardware (~25x faster)
- 232ms/token breakdown: **99% JIT execution**, <1% setup, <1% argmax/KV update

## Bottlenecks

### 1. Single-threaded execution

All `scf.for` loops are sequential. GPT-2 decode has ~24 matmuls per
token. Each runs on one core out of 12 available.

**Fix**: The transform schedule tiles with `tile_using_forall` which
produces `scf.forall` → `scf.parallel` → OpenMP. The pass pipeline
includes `convert-scf-to-openmp`. Need to verify it's actually producing
multi-threaded code at runtime.

### 2. Naive GEMM kernels

Generated code is tiled (32x32 cache + 8x8x1 register) and vectorized
(`vector.contract` → AVX2), but still far from hand-tuned BLAS. Missing:
- Register blocking with explicit accumulator registers
- Software pipelining (prefetch next tile while computing current)
- Optimal loop ordering for the micro-kernel
- AVX-512 utilization (CPU supports it)

**Fix** (MLIR-native, no external BLAS):
- `transform.structured.pack` for packed GEMM data layout
- Tune tile sizes and loop ordering for the micro-kernel
- Explicit unroll-and-jam via transform schedule
- AVX-512 targeting via wider vector_sizes

### 3. KV cache memory traffic

Each decode step reads/writes the full KV cache ([1, 12, seq_len, 64]
× 24 layers). As seq_len grows, this becomes memory-bandwidth bound.

**Fix**: Quantize KV cache to FP16/INT8, or use paged attention.

## Improvement roadmap

All MLIR-native — no external BLAS. Each phase builds on the previous.

### Phase 0: Per-kernel compilation — DONE

Each heavy op compiled independently with its own MLIR context. Parallel
compilation via rayon. Targeted vectorization via outline pattern.
Compilation times: MNIST 97ms, ResNet-18 467ms.

### Phase 1: Verify multi-threading — TODO

Confirm OpenMP parallelization is working at runtime. Measure with
`perf stat` to verify all cores active during inference. Expected
~6-10x speedup → **25-40 tok/s**.

### Phase 2: Packed GEMM layout (2-4x → ~50-120 tok/s)

`transform.structured.pack` for cache-friendly data layout.
See previous perf.md for detailed plan.

### Phase 3: Micro-kernel tuning (1.5-2x → ~80-200 tok/s)

AVX-512, unroll-and-jam, loop permutation, prefetch intrinsics.

### Phase 4: Inference-specific optimizations

Weight prepacking, operator fusion, KV cache quantization.

## Summary

| Phase | Change | Status | Target tok/s |
|---|---|---|---|
| 0 | Per-kernel compilation + targeted vectorize | **Done** | compile: <500ms |
| 1 | Multi-threading (OpenMP) | Verify | ×6-10 |
| 2 | Packed GEMM layout | Not started | ×2-4 |
| 3 | AVX-512 + unroll + loop order | Not started | ×1.5-2 |
| 4 | Fusion + weight prepack + KV quant | Not started | ×1.3-2 |

All phases use MLIR transform dialect + standard passes. No external
BLAS, no hand-written assembly. The compiler generates everything.
