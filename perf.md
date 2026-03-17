# Performance

## Current: GPT-2 small (124M params) decode

- **4.3 tok/s** (233ms/token) single-threaded on Ryzen 9 7900X3D
- Reference: llama.cpp does ~100+ tok/s on similar hardware (~25x faster)
- 232ms/token breakdown: **99% JIT execution**, <1% setup, <1% argmax/KV update

## Bottlenecks

### 1. Single-threaded execution

All `scf.for` loops are sequential. GPT-2 decode has ~24 matmuls per
token. Each runs on one core out of 12 available.

**Fix**: Use `tile_using_forall` in the transform schedule to tile outer
parallel dims into `scf.forall`, which maps to parallel threads.
Requires a threading runtime (OpenMP or pthreads via MLIR's async
dialect). Expected ~6-10x speedup.

### 2. Naive GEMM kernels

Generated code is tiled (32x32x32 cache + 8x8x1 register) and
vectorized (`vector.contract` → AVX2), but still far from hand-tuned
BLAS. Missing:
- Register blocking with explicit accumulator registers
- Software pipelining (prefetch next tile while computing current)
- Optimal loop ordering for the micro-kernel
- AVX-512 utilization (CPU supports it)

**Fix options**:
- Link against OpenBLAS/MKL for matmuls (bypass generated code)
- Improve the generated micro-kernel (better tile sizes, unroll, prefetch)
- Use MLIR's `transform.structured.pack` for packed GEMM layout

### 3. No memory prefetching

Cache tiles load data on demand. For large matmuls (768x3072), the
working set exceeds L1. Prefetching the next tile while computing the
current would hide memory latency.

### 4. KV cache memory traffic

Each decode step reads/writes the full KV cache ([1, 12, seq_len, 64]
× 24 layers). As seq_len grows, this becomes memory-bandwidth bound.

**Fix**: Quantize KV cache to FP16/INT8, or use paged attention.

## Improvement roadmap (estimated impact)

| Change | Expected speedup | Complexity |
|---|---|---|
| Multi-threading (forall + OpenMP) | 6-10x | Medium |
| Link OpenBLAS for matmul | 5-10x | Low |
| Both above | 20-50x → ~80-200 tok/s | Medium |
| AVX-512 micro-kernel | 1.5-2x | High |
| KV cache quantization | 1.3-2x (long sequences) | Medium |
| Software prefetching | 1.2-1.5x | Medium |
