# Runtime Shape Ops & Dynamic Reshape — Status & Fix Plan

## Current Status (2026-03-15)

### Completed
- ShapeOf, ConstantOfShape, Range ops: implemented and tested
- Decode model compiles through full MLIR pipeline (TOSA → linalg → SCF → LLVM → .so)
- All 391 tests pass, static models (MNIST, ResNet, GPT-2 prefill) work correctly
- Codegen fixes: 0-D tensor.extract, expand_shape output_shape, operandSegmentSizes,
  i1 tensor kDynamic, convert-linalg-to-loops, matmul dynamic dispatch

### Blocker: JIT Segfault in Decode Model

The compiled decode model segfaults during execution. Root cause identified via
gdb + MLIR IR analysis.

## Root Cause: `one-shot-bufferize` Shape Buffer Reuse

### The problem

All 48 `tensor.reshape` ops in the decode model use `tensor.from_elements` to build
their shape tensor. After `one-shot-bufferize`, ALL of these share a single
`alloc_28: memref<2xi64>` shape buffer. The sequence becomes:

```
write [1, 768] → alloc_28
memref.reshape tensorA(alloc_28) → viewA    // viewA's descriptor built from [1, 768]
write [1, 12, 1, 64] → alloc_28            // overwrites!
memref.reshape tensorB(alloc_28) → viewB
use viewA                                   // viewA's descriptor references alloc_28
                                            // but alloc_28 now has [1, 12, 1, 64]
```

`memref.reshape` creates a view whose strides are computed from the shape buffer.
When `finalize-memref-to-llvm` lowers it, the shape buffer is read to compute strides.
If the buffer was overwritten between the reshape and the use of the reshaped tensor,
the strides are wrong → invalid memory access → segfault.

### GDB evidence

- Crash at `vmovups -0xc0(%rax,%rdx,4),%zmm0` in a vectorized memcpy loop
- `%rax` = 192 (a stride value, not a valid pointer)
- Stack struct at `%rbx` has NULL data pointers and a pointer value in the offset field
- This matches corrupted memref descriptor from wrong shape buffer contents

### How `tensor.reshape` lowers through MLIR

```
tensor.reshape   (tensor dialect, pre-bufferize)
    ↓ one-shot-bufferize
memref.reshape   (memref dialect, with shape buffer operand)
    ↓ finalize-memref-to-llvm
llvm.load from shape buffer → compute strides → build descriptor struct
```

The LLVM lowering of `memref.reshape` reads the shape buffer at **execution time**,
not at "reshape construction time". So if the shape buffer was reused and overwritten,
the reads get stale values.

## Industry Standard: expand_shape / collapse_shape

No production ML compiler uses `tensor.reshape` with runtime shape tensors:

- **IREE (Google):** Resolves all reshapes to `tensor.expand_shape`/`tensor.collapse_shape`.
  These encode the dim grouping as a static reassociation attribute and pass dynamic sizes
  as SSA values — no shape buffer needed.

- **ONNX-MLIR (IBM):** Uses `memref.reinterpret_cast` with explicit SSA size/stride/offset
  values computed eagerly at the reshape site.

- **TVM / XLA / TensorRT:** Either compile per-shape or use their own runtime shape metadata
  separate from the compute kernel.

In a transformer, **all reshapes are structurally predictable**:

| Pattern | Example | MLIR op |
|---|---|---|
| Merge dims | `[B, S, H]` → `[B*S, H]` | `tensor.collapse_shape [[0,1], [2]]` |
| Split dims | `[B*S, H]` → `[B, S, H]` | `tensor.expand_shape [[0,1], [2]]` |
| Split head dim | `[B, S, H]` → `[B, S, heads, head_dim]` | `tensor.expand_shape [[0], [1], [2,3]]` |
| Merge head dim | `[B, heads, S, D]` → `[B, S, heads*D]` | needs transpose first |

These are all **static reassociation patterns** with dynamic sizes as SSA operands.

## Fix Plan: Replace `tensor.reshape` with expand/collapse_shape

### Approach

In the compiler's `Op::Reshape` codegen (the `shape_tensor: Some(...)` path), instead of
emitting `tensor.reshape %input(%shape_tensor)`, analyze the input and output shapes to
determine the reassociation pattern, then emit the appropriate `tensor.expand_shape` or
`tensor.collapse_shape`.

### Implementation

**Step 1: Classify the reshape pattern**

Given input shape `[d0, d1, ..., dM]` and output shape `[e0, e1, ..., eN]`:
- If M < N → expand (splitting dims) → `tensor.expand_shape`
- If M > N → collapse (merging dims) → `tensor.collapse_shape`
- If M == N → same rank → `tensor.cast` (just refines types, no layout change)

**Step 2: Compute the reassociation map**

For collapse `[B, S, H] → [B*S, H]`:
- Match dims left-to-right: `B*S` consumes input dims 0,1; `H` consumes dim 2
- Reassociation: `[[0, 1], [2]]`

For expand `[B*S, H] → [B, S, H]`:
- Match dims: input dim 0 splits into output dims 0,1; input dim 1 maps to output dim 2
- Reassociation: `[[0, 1], [2]]`
- Need `output_shape` SSA values for the dynamic dims (from the shape tensor or tensor.dim)

The matching algorithm:
- Walk input and output dims together
- A group of consecutive output dims maps to one input dim when their product matches
- For dynamic dims, use the runtime values from the shape tensor via `tensor.extract`

**Step 3: Emit the op**

For `tensor.expand_shape`:
- Build reassociation attribute
- For each dynamic output dim, emit `tensor.extract %shape_tensor[%ci]` → `arith.index_cast`
- Pass as `output_shape` operands (already supported — we fixed this in `promote_rank_with_reshape`)

For `tensor.collapse_shape`:
- Build reassociation attribute
- No dynamic operands needed (output sizes are computed from input sizes automatically)

### Edge cases

- **Identity reshape** (same shape): emit `tensor.cast` or no-op
- **Flatten** `[B, S, H, D] → [B*S*H*D]`: single group `[[0,1,2,3]]`
- **Reshape with size-1 dims**: `[1, S, 768] → [S, 768]` — collapse removing the 1
- **Cannot determine reassociation**: Fall back to `tensor.reshape` (shouldn't happen for
  transformer models but keeps correctness)

### Where to modify

| File | Change |
|---|---|
| `src/compiler.rs` | `Op::Reshape` match arm: when `shape_tensor.is_some()`, classify and emit expand/collapse instead of `tensor.reshape` |
| `src/compiler.rs` | New helper `fn emit_dynamic_reshape(...)` that does the pattern analysis |
| No mapper changes | The mapper already produces the right `Op::Reshape` with `shape_tensor` |
| No runtime changes | expand/collapse_shape don't need runtime shape buffers |

### Testing

1. Existing `run_dynamic_reshape_3d_to_2d` test — should still pass (collapse_shape path)
2. New test: expand_shape path (2D → 3D with dynamic dim)
3. New test: same-rank reshape (tensor.cast path)
4. End-to-end: `cargo run --release -- run tests/fixtures/ --prompt "Hello" --max-tokens 5`

### Acceptance criteria

- All 391 tests pass
- Decode model runs without segfault
- GPT-2 generates tokens via KV cache decode path
- No `tensor.reshape` ops in the emitted MLIR (all replaced by expand/collapse/cast)
