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

### Limitation

`vectorize_children_and_apply_patterns` skips the tiled contraction ops
because they have dynamic shapes after tiling. Only simple ops like
`linalg.fill` get vectorized. The contractions still lower through
`convert-linalg-to-loops` → LLVM `-O3`, so LLVM's auto-vectorizer
handles SIMD. This means GPT-2 decode stays at ~0.22s/token — no
measurable improvement from this change.

## Next: true micro-kernel vectorization

The goal is to vectorize the 8×8×1 register-tiled contraction ops
directly, producing explicit `vector.transfer_read/write` +
`vector.contract` or FMA ops. This requires solving the masked
vectorization issue.

### Root cause

`tile_using_for` always emits `affine.min(tile_size, remaining)` for
boundary handling, producing dynamic shapes even when the data divides
evenly. `structured.vectorize vector_sizes [8, 8, 1]` then creates
masked ops (`vector.mask`) to handle the dynamic-sized boundary tiles.
The bug: the vectorizer wraps both `vector.transfer_write` AND a
`tensor.cast` inside `vector.mask`, violating the "exactly one maskable
op" constraint.

```
// Illegal IR produced by masked vectorization after tiling:
%r = "vector.mask"(%mask) ({
  %w = vector.transfer_write ...  // maskable op
  %c = tensor.cast %w             // non-maskable — violates constraint
  vector.yield %c
}) : (vector<8x8xi1>) -> tensor<?x?xf32>
```

### Approach A: loop peeling (most promising)

Use `transform.loop.peel` (available in LLVM 21 as `loop.peel` in SCF
transform ops) to split each tiling loop into:
- **Main loop**: trip count divisible by step → static shapes inside
- **Remainder loop**: last partial iteration → dynamic shapes (scalar)

After peeling, the main loop body has a linalg.generic with **static**
shapes (exactly tile_size), so `structured.vectorize` works without
masking.

```mlir
// Pseudocode schedule:
%tiled, %l0, %l1, %l2 = tile_using_for %op [8, 8, 1]
%main0, %rem0 = loop.peel %l0   // peel M loop
%main1, %rem1 = loop.peel %l1   // peel N loop
vectorize %tiled vector_sizes [8, 8, 1]  // static shapes in main body
```

**Complication**: handle invalidation in the transform dialect. Peeling
`%l0` invalidates `%l1` (nested inside). Workarounds:
- Peel innermost-first (l2 → l1 → l0), re-matching inner loops after
  each peel via `structured.match` inside the peeled loop body.
- Use `transform.foreach` to iterate over peeled main loops and apply
  inner peeling + vectorization.
- Alternatively, peel at the cache-tile level (outer loops) and only
  vectorize the full-tile path.

**Key MLIR ops**:
- `transform.loop.peel` — takes `!transform.op<"scf.for">`, returns
  peeled main + remainder loop handles
- `fail_if_already_divisible = false` — silently no-ops if already even

### Approach B: pad before tiling

Use `transform.structured.pad` (available in LLVM 21) to pad operands
to multiples of tile sizes before tiling. This ensures all tiles are
full-sized (no boundary handling), so shapes are always static.

```mlir
%padded = transform.structured.pad %op {
  padding_values = [0.0 : f32, 0.0 : f32, 0.0 : f32],
  padding_dimensions = [0, 1, 2],
  pack_paddings = [1, 1, 1]
}
%tiled, ... = tile_using_for %padded [8, 8, 1]
vectorize %tiled vector_sizes [8, 8, 1]  // always static
```

**Pros**: no masking, no peeling, clean static shapes everywhere.
**Cons**: memory overhead from padding, extra computation on pad values,
need to slice the result back to original size.

### Approach C: per-rank vectorization with peeled cache tiles

Combine rank-aware matching (via `match.structured.rank` +
`match.param.cmpi`) with loop peeling at the cache tile level only:

1. Match contraction ops by rank (3d, 4d, 5d)
2. Cache-tile with batch dims tiled to 1: `[32,32,32]` / `[1,32,32,32]`
3. Peel the cache-tile loops (outer level — fewer loops, simpler)
4. Register-tile the peeled main body: `[8,8,1]` / `[1,8,8,1]`
5. Vectorize with rank-appropriate sizes: `[8,8,1]` / `[1,8,8,1]`

This avoids peeling the register-tile loops (which cause the handle
invalidation cascade) and instead ensures the register tile always
operates on full 32×32×32 blocks with no boundary.

### MLIR references

- `transform.loop.peel`: `/usr/lib/llvm/21/include/mlir/Dialect/SCF/TransformOps/SCFTransformOps.td`
- `transform.structured.pad`: `/usr/lib/llvm/21/include/mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.td`
- `match.structured.rank` + `match.param.cmpi`: for per-rank dispatch
- `vector.mask` constraint: `/usr/lib/llvm/21/include/mlir/Dialect/Vector/IR/VectorOps.td`
- LLVM issue #78787: pattern rewriting for maskable ops
- MLIR discourse: "Linalg and Masking" thread, "RFC: Vector Masking" thread
