# Dynamic Dimensions — Status

## Status: DONE (decode model runs end-to-end)

GPT-2 with KV cache decode path compiles and executes correctly.
Warm decode at **0.30s/token** (beats the 0.85s target).

**Remaining issue:** Output quality is poor (mostly commas). This is likely
an attention mask / KV cache / position ID issue, not a codegen bug.

## What works

- [x] DIM_DYNAMIC sentinel in Shape (u64::MAX → i64::MIN = MLIR's kDynamic)
- [x] dim_to_mlir / dim_to_mlir_i64 conversion functions
- [x] Shape::broadcast handles DIM_DYNAMIC
- [x] Shape::num_elements panics on dynamic, concrete_num_elements returns Option
- [x] emit_tensor_dim / emit_tensor_empty_dynamic helpers
- [x] promote_rank_with_reshape handles DIM_DYNAMIC (tensor.expand_shape with output_shape operands)
- [x] emit_tosa_reshape falls back to tensor.cast for same-rank dynamic shapes
- [x] emit_binary_elementwise handles dynamic output shapes
- [x] emit_batched_matmul handles dynamic dims
- [x] emit_reduction handles dynamic dims (falls back to linalg.generic)
- [x] emit_strided_slice handles dynamic dims
- [x] Concat with dynamic dims uses tensor.insert_slice (with operandSegmentSizes)
- [x] Where i1 conversion handles dynamic dims (dim_to_mlir for i1 tensor type)
- [x] Gather handles dynamic dims
- [x] Cast handles dynamic dims
- [x] Op::ShapeOf + codegen (tensor.dim + tensor.from_elements)
- [x] Op::ConstantOfShape + codegen (tensor.extract + tensor.empty + linalg.fill)
- [x] Op::Range + codegen (handles 0-D and 1-D scalar tensor.extract)
- [x] Op::DynamicSlice + codegen (tensor.extract_slice with operandSegmentSizes)
- [x] Op::Reshape with shape_tensor: ONNX -1 dim inference + tensor.reshape
- [x] Mapper conditional tracing: Shape, ConstantOfShape, Range, Reshape, Slice
- [x] Sentinel guard in eval_gather_constant
- [x] Model::load_with_dynamic_dims + partial symbolic dim resolution
- [x] shapes_changed check skips DIM_DYNAMIC positions
- [x] Cache key uses DIM_DYNAMIC for dynamic positions
- [x] CompiledGraph::run_dynamic with caller-provided output shapes
- [x] Tensor::matmul allows DIM_DYNAMIC inner dims
- [x] Tensor::concat handles DIM_DYNAMIC in non-axis dims
- [x] Tensor::slice skips normalization for dynamic dims
- [x] KvGenerator uses load_with_dynamic_dims for decode model
- [x] Prefill works (bucket-padded prompt → KV cache)
- [x] Decode model compiles (MLIR verification passes)
- [x] MLIR pass pipeline succeeds (LLVM IR generated, .so emitted)
- [x] Decode model runs at runtime (segfault fixed)
- [x] dlti.dl_spec + LLVM data layout for correct struct layouts
- [x] 0-element memref sentinel (EMPTY_BUF) for shape tensors

## Root cause of the segfault: ONNX -1 in shape tensors

The decode model segfaulted because ONNX shape tensors contain `-1` (meaning
"infer this dimension"). These raw values were passed to `tensor.reshape` in
the compiled code. After lowering:

1. `tensor.reshape` → `memref.reshape` (reads shape buffer for strides)
2. Shape buffer contains `-1` → stride computed as `product_of_remaining / -1`
3. Wrong strides → invalid memory access → segfault

The value `192 = 3 × 64` seen in the original crash was a stride computed from
a shape tensor containing `-1` for one dimension and `3` for another, producing
garbage stride values.

### Fix

In the compiler's `Op::Reshape` codegen (shape_tensor path):
1. Extract raw i64 values from the shape tensor
2. Detect `-1` via `arith.cmpi eq`
3. Compute `inferred_dim = total_input_elements / product_of_known_dims`
4. Replace `-1` with inferred value via `arith.select`
5. Build corrected shape tensor via `tensor.from_elements`
6. Pass corrected tensor to `tensor.reshape`

### Approach history

**tensor.reshape** was the correct approach all along. The intermediate
`linalg.generic` copy-reshape was unnecessary — it was introduced to work
around what we thought was a `memref.reshape` buffer reuse bug, but the
real issue was always the ONNX `-1` values.

`tensor.expand_shape`/`tensor.collapse_shape` with dynamic dims still crashes
in MLIR's `expand-strided-metadata` pass (LLVM bug #61158). This is not
relevant since `tensor.reshape` works correctly with proper shape values.

### Items from original TODO not yet hit (may be fine)

- [ ] **emit_tosa_reduce keepdim=false reshape**: Rank-changing with dynamic dims.
      Not triggered yet — may need tensor.collapse_shape if encountered.
- [ ] **ReduceMean scaling**: Reciprocal constant reshape with dynamic dims.
- [ ] **Softmax Div**: Broadcast map with dynamic reduced dim.
- [ ] **BatchNorm reshapes**: Likely fine (dynamic is seq dim, not channel).

## Performance (measured)

| Phase | Time | Notes |
|-------|------|-------|
| Load models | ~2.3s | Parse ONNX (2 models) + tokenizer |
| Prefill (cold) | ~23s | First-ever compile for this bucket |
| Prefill (warm) | ~7.5s | Cache hit: dlopen + inference |
| Decode (cold) | ~87s | First-ever compile for decode model |
| Decode (warm) | ~0.30s/token | Cache hit: dlopen + inference |

The decode model compiles ONCE (dynamic past_sequence_length). After that,
every token is just inference — no recompilation. Warm decode at 0.30s/token
beats the 0.85s target.

## Next: output quality

The model generates correctly but output is low quality (mostly commas).
Possible causes:
- Attention mask not properly passed/computed for decode steps
- KV cache concatenation producing wrong values
- Position IDs not incrementing correctly
- Causal mask handling in the with-past model
