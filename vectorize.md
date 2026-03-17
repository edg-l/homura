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

## Resolved: vectorization approach

### Problems encountered

1. **Rank-aware vector sizes**: `structured.vectorize` with explicit `vector_sizes` requires them to exactly match the op's iterator count. GPT-2 contractions have 3, 4, or 5 dims.

2. **Masked vectorization**: After tiling, shapes become dynamic (via `affine.min` boundary handling). Using explicit `vector_sizes` triggers masked vectorization, which creates `vector.mask` ops containing `tensor.cast` — illegal IR (`vector.mask expects only one operation`).

3. **Regular vectorization**: Without `vector_sizes`, `structured.vectorize` requires static shapes — but dynamic shapes from tiling prevent this.

### Solution: `vectorize_children_and_apply_patterns`

Applied at function level in `@__transform_main` after all tiling via `foreach_match`:

```mlir
%func = transform.structured.match ops{["func.func"]} in %updated
transform.structured.vectorize_children_and_apply_patterns %func
```

This pattern-based approach:
- Handles any rank (no per-rank matchers needed)
- Gracefully skips ops with dynamic shapes (no failure)
- Vectorizes ops with static shapes (e.g., `linalg.fill`)
- Works for all test sizes including small tensors with uneven tiling

### Solved: pad after tiling for static shapes

The key insight: `structured.pad` with `pad_to_multiple_of` converts
dynamic shapes from `affine.min` back to static tile-sized shapes.
Applied AFTER register tiling, BEFORE vectorization:

```mlir
// In @tile_contraction action:
%tiled2, ... = tile_using_for %tiled1 tile_sizes [8, 8, 1]
%padded, %pad, %copy = transform.structured.pad %tiled2
    pad_to_multiple_of [8, 8, 1]
    {padding_dimensions = [0, 1, 2], copy_back_op = "none"}

// In @__transform_main, after foreach_match:
%func = transform.structured.match ops{["func.func"]} in %updated
transform.structured.vectorize_children_and_apply_patterns %func
```

Padding values auto-inferred as 0 for the mulf/addf contraction ring.
The padded op has static shapes (8×8×1), so `vectorize_children_and_apply_patterns`
now vectorizes contractions into `vector.contract` ops:

```
vector.transfer_read  → vector<8x1xf32>  (LHS tile row)
vector.transfer_read  → vector<1x8xf32>  (RHS tile col)
vector.contract       → vector<8x8xf32>  (outer product accumulate)
vector.transfer_write → vector<8x8xf32>  (write result tile)
```

No `vector.mask` ops — padding eliminates the need for masking.
Works for any rank without per-rank dispatch.
GPT-2 decode: ~0.23s/token (same ballpark, LLVM -O3 was already
auto-vectorizing the scalar loops effectively).
