# Runtime Shape Ops & Dynamic Reshape — Status

## Status: RESOLVED (2026-03-16)

All runtime shape ops work. Decode model runs end-to-end.

### Completed
- ShapeOf, ConstantOfShape, Range ops: implemented and tested
- Decode model compiles and runs through full MLIR pipeline
- All 401 tests pass, static models (MNIST, ResNet, GPT-2 prefill) work correctly
- Dynamic reshape with ONNX -1 dim inference via tensor.reshape

## Root Cause: ONNX -1 in Shape Tensors (NOT buffer reuse)

The original analysis blamed `one-shot-bufferize` shape buffer reuse for the
segfault. This was **wrong**. The actual root cause:

ONNX shape tensors contain `-1` (meaning "infer this dimension"). These raw
values were passed to `tensor.reshape`, producing garbage strides in the
lowered `memref.reshape`. The value `192 = 3 × 64` in the crash was from a
stride computation involving `-1`.

### Fix
Resolve ONNX -1 dims at the MLIR level before passing to `tensor.reshape`:
- `arith.cmpi` to detect -1
- `arith.divui` to compute `total_elements / product_of_known_dims`
- `arith.select` to replace -1 with the inferred value
- `tensor.from_elements` to build corrected shape tensor

### Why not expand/collapse_shape?
`tensor.expand_shape`/`tensor.collapse_shape` with dynamic dims crashes in
MLIR's `expand-strided-metadata` pass (LLVM bug #61158, IREE #17760). This
is not fixable from our side. `tensor.reshape` works correctly.

## Industry Context

- **IREE**: `flow.tensor.reshape` (custom op, avoids upstream reshape entirely)
- **ONNX-MLIR**: `memref.reinterpret_cast` with manually computed strides
- **torch-mlir**: `tensor.reshape` as fallback, notes dynamic dims are problematic

Our approach (tensor.reshape with corrected shape tensor) is simpler than all
of these and works for transformer models.
