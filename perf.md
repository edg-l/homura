# Performance

## Compilation performance

Per-kernel compilation (IREE-style): each heavy op gets its own MLIR
context + module, compiled in parallel via rayon.

| Model | Kernels | Compile time | Notes |
|---|---|---|---|
| MNIST | 6 | 97ms | 2 Conv + MatMul + lightweight |
| ResNet-18 | 41 | 467ms | 20 Conv + Gemm + BatchNorm groups |
| GPT-2 decode | 146 | ~1.2s | 12 layers × MatMul/Gemm/attention/MLP |

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

- **~3 tok/s** (~150-200ms/token) on Ryzen 9 7900X3D
- FMA enabled: 2689 `vfmadd` instructions across 72/146 decode kernels
- Prefill (5 tokens): ~1.8s (runs all 12 layers on full prompt)
- Reference: llama.cpp does ~100+ tok/s on similar hardware (~30x faster)

### What's been done

- **FMA fusion**: `vector-contract-lowering=outerproduct` + `fastmath<contract>`
  on all float arith ops + LLVM-level `AllowContract` flag. Produces `vfmadd`
  instead of separate `vmulps + vaddps`.
- **Symbolic shape tracking**: `SymDim` expressions propagate through the op
  graph. Models with dynamic dims compile once, shapes resolve at runtime.
  Eliminates the earlier two-probe dim resolution hack.

## Bottlenecks

### 1. GEMM performance

Generated code is tiled (32x32 cache + 8x8x1 register) and vectorized with
FMA (outerproduct → `vfmadd`), but still far from hand-tuned BLAS. Missing:
- Register blocking with explicit accumulator registers
- Software pipelining (prefetch next tile while computing current)
- Optimal loop ordering for the micro-kernel

**Fix** (MLIR-native, no external BLAS):
- `transform.structured.pack` for packed GEMM data layout
- Tune tile sizes and loop ordering for the micro-kernel
- Explicit unroll-and-jam via transform schedule
- Wider vector_sizes for AVX-512

### 3. KV cache memory traffic

Each decode step reads/writes the full KV cache ([1, 12, seq_len, 64]
× 24 layers). As seq_len grows, this becomes memory-bandwidth bound.

**Fix**: Quantize KV cache to FP16/INT8, or use paged attention.

## Improvement roadmap

All MLIR-native — no external BLAS. Each phase builds on the previous.

### Phase 0: Per-kernel compilation — DONE

Each heavy op compiled independently with its own MLIR context. Parallel
compilation via rayon. Targeted vectorization via outline pattern.
Compilation times: MNIST 97ms, ResNet-18 467ms, GPT-2 ~1.2s.

### Phase 0.5: FMA + symbolic shapes — DONE

- Outerproduct vector contract lowering → `vfmadd` instructions
- `fastmath<contract>` on MLIR arith ops + LLVM-level `AllowContract`
- Symbolic dim tracking (`SymDim`) replaces two-probe hack
- GPT-2 decode: ~3 tok/s (from ~4.3 tok/s baseline, with FMA)

### Phase 1: Packed GEMM layout (2-4x)

`transform.structured.pack` for cache-friendly data layout.

### Phase 2: Micro-kernel tuning (1.5-2x)

Wider vector_sizes for AVX-512, unroll-and-jam, loop permutation, prefetch.

### Phase 3: Inference-specific optimizations

Weight prepacking, operator fusion, KV cache quantization, unified
single-model inference (eliminate separate prefill/decode models).

### Phase 4: Modern model support

SmolLM2/Llama-family architecture: SiLU, RMSNorm, RoPE, GQA.

## Summary

| Phase | Change | Status | Target tok/s |
|---|---|---|---|
| 0 | Per-kernel compilation + targeted vectorize | **Done** | compile: <1.5s |
| 0.5 | FMA (outerproduct) + symbolic shapes | **Done** | ~3 tok/s |
| 1 | Packed GEMM layout | Not started | ×2-4 |
| 2 | AVX-512 + unroll + loop order | Not started | ×1.5-2 |
| 3 | Fusion + weight prepack + KV quant | Not started | ×1.3-2 |
| 4 | Modern models (Llama-family) | Not started | — |

All phases use MLIR transform dialect + standard passes. No external
BLAS, no hand-written assembly. The compiler generates everything.
