# Vectorization Plan: Two-Level Tiling

## Problem

Single-level tiling at 32×32×32 + vectorize creates `vector<32x32x32xf32>` (128KB) which gets fully unrolled into ~200k vector ops. This is wrong — it conflates cache tiles with register tiles.

## Architecture

Two-level tiling separates concerns:

```
Cache tile (32×32×32)     — fits working set in L1/L2
  Register tile (8×8×1)   — maps to vector instructions
    vectorize              — small vectors, no unroll explosion
```

The outer loop iterates over 32×32×32 blocks for cache locality. The inner loop iterates over 8×8×1 micro-tiles that vectorize cleanly into `vector<8xf32>` (AVX2) operations.

## Why 8×8×1 for the register tile

- M=8: one vector width of output rows (8 floats = AVX2 `<8 x float>`)
- N=8: 8 output columns — each is one FMA of a `vector<8xf32>`
- K=1: process one K element at a time — accumulate 8×8 outer product

This produces: 8 `vector.transfer_read` of `vector<8xf32>`, 8 FMAs, 8 `vector.transfer_write`. Small, predictable, maps to hardware.

## Transform schedule

```mlir
// Match contraction generics
transform.named_sequence @match_contraction(...)

// Two-level tile + vectorize
transform.named_sequence @tile_and_vectorize(%op) {
  // Level 1: cache tile — 32×32×32
  %tiled_l1, %l0, %l1, %l2 = transform.structured.tile_using_for %op
    tile_sizes [32, 32, 32]

  // Level 2: register tile — 8×8×1
  %tiled_l2, %l3, %l4 = transform.structured.tile_using_for %tiled_l1
    tile_sizes [8, 8, 1]

  // Vectorize the 8×8×1 micro-kernel
  transform.structured.vectorize %tiled_l2 : !transform.any_op
}
```

## Pipeline changes

```
transform-interpreter → canonicalize → cse → symbol-dce
→ func.func(lower-vector-multi-reduction, lower-vector-mask)
→ one-shot-bufferize{...}
→ func.func(buffer-hoisting, promote-buffers-to-stack{...})
→ fold-memref-alias-ops
→ convert-vector-to-scf
→ convert-linalg-to-loops
→ fold-memref-alias-ops → lower-affine → convert-scf-to-cf
→ canonicalize → cse → sccp
→ convert-vector-to-llvm → convert-ub-to-llvm → convert-math-to-llvm
→ expand-strided-metadata → lower-affine → finalize-memref-to-llvm
→ convert-arith-to-llvm → convert-index-to-llvm → convert-cf-to-llvm
→ convert-func-to-llvm → reconcile-unrealized-casts
```

New passes vs current:
- `lower-vector-multi-reduction` — before bufferize
- `lower-vector-mask` — before bufferize
- `convert-vector-to-scf` — after bufferize, before linalg-to-loops
- `convert-vector-to-llvm` — after scf-to-cf
- `convert-ub-to-llvm` — after vector-to-llvm

## Implementation steps

1. Modify `build_transform_schedule` — add second `tile_using_for` + `vectorize` in the action
2. Add vector lowering passes to the pipeline string
3. Test: all 394 tests pass, MNIST/ResNet-18 e2e, GPT-2 correct output + measure perf

## Expected vector IR

For the 8×8×1 micro-kernel:
- `vector.transfer_read` of `vector<8xf32>` (one row of A tile)
- `vector.broadcast` + `arith.mulf` + `arith.addf` on `vector<8xf32>`
- `vector.transfer_write` of `vector<8xf32>`

These map directly to AVX2 `vmovups` + `vfmadd231ps` instructions.

## Verified IR size

Final LLVM dialect output for 128×768 × 768×3072 matmul:
- 313× `vector<8xf32>` — AVX2-sized, maps to `vmovups`/`vfmadd231ps`
- 711× `vector<1xf32>` — scalar ops from K=1 reduction dimension
- Total: ~1k vector ops — vs ~200k with single-level vectorize. 200× less IR.

## Blocking issue: rank-aware vector sizes

`structured.vectorize` requires `vector_sizes` to exactly match the op's iterator count. Unlike `tile_sizes` (which pads with leading zeros), `vector_sizes` does NOT pad — providing 3 sizes for a 4-dim op is an error.

GPT-2 contractions have different ranks:
- 3-dim: M, N, K (2D matmul)
- 4-dim: B, M, N, K (batched matmul)
- 5-dim: B1, B2, M, N, K (4D attention)

A single `vector_sizes [8, 8, 1]` only works for 3-dim ops. 4-dim needs `[1, 8, 8, 1]`, 5-dim needs `[1, 1, 8, 8, 1]`.

### Possible solutions

1. **Multiple action sequences** per rank (match by dim count). Pragmatic but not general.
2. **Dynamic vector sizes via `classify_contraction_dims`**. General but complex.
3. **Skip vectorize, rely on LLVM -O3**. Current approach — two-level tiling gives cache locality, LLVM handles SIMD.

### Current status

Two-level tiling (32×32×32 cache + 8×8×1 register) is implemented without vectorize. Vectorization deferred until rank-awareness is solved.
