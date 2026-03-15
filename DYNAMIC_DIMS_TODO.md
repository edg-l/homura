# Dynamic Dimensions — Remaining Work

## Status

The dynamic dimension infrastructure is in place. The GPT-2 decode model (with-past)
gets through parsing, tracing, and most of MLIR emission, but hits remaining edge cases
in the compiler codegen. Each fix is small and follows the same pattern.

**Last error:** `'tensor.extract' op incorrect number of indices for extract_element`
during decode model compilation.

## What works

- [x] DIM_DYNAMIC sentinel in Shape (u64::MAX → i64::MIN = MLIR's kDynamic)
- [x] dim_to_mlir / dim_to_mlir_i64 conversion functions
- [x] Shape::broadcast handles DIM_DYNAMIC
- [x] Shape::num_elements panics on dynamic, concrete_num_elements returns Option
- [x] emit_tensor_dim / emit_tensor_empty_dynamic helpers
- [x] promote_rank_with_reshape handles DIM_DYNAMIC
- [x] emit_tosa_reshape falls back to tensor.cast for dynamic shapes
- [x] emit_binary_elementwise handles dynamic output shapes
- [x] emit_batched_matmul handles dynamic dims
- [x] emit_reduction handles dynamic dims (falls back to linalg.generic)
- [x] emit_strided_slice handles dynamic dims
- [x] Concat with dynamic dims uses tensor.insert_slice
- [x] Where i1 conversion handles dynamic dims
- [x] Gather handles dynamic dims
- [x] Cast handles dynamic dims
- [x] Op::ShapeOf + codegen (tensor.dim + tensor.from_elements)
- [x] Op::ConstantOfShape + codegen (tensor.extract + tensor.empty + linalg.fill)
- [x] Op::Range + codegen (tensor.generate + arith.ceildivsi)
- [x] Op::DynamicSlice + codegen (tensor.extract_slice)
- [x] Op::Reshape extended with shape_tensor (tensor.reshape)
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

## Remaining: compiler DIM_DYNAMIC edge cases

Each of these follows the same pattern: a codegen helper emits an MLIR op that
assumes static shapes. When a tensor has DIM_DYNAMIC dims, the op either:
1. Produces an invalid attribute (u64::MAX in a dense<> literal)
2. Uses wrong index count for tensor.extract on reshaped tensors
3. Passes static shape where MLIR expects dynamic operands

### To debug

- [ ] Enable IR dump on verification failure (add eprintln of module.to_string()
      before the verify call in compiler.rs) to see exactly which op fails
- [ ] Search the dumped IR for the first verification error
- [ ] Fix the specific codegen site

### Known patterns that may need fixes

- [ ] **tensor.extract on 0-D vs 1-D tensors**: Range/ConstantOfShape extract
      scalars from 1-element tensors. If the tensor is 0-D (`tensor<i64>`), extract
      uses `tensor.extract %t[] : tensor<i64>` (no indices). If 1-D (`tensor<1xi64>`),
      extract uses `tensor.extract %t[%c0] : tensor<1xi64>`. The mapper may produce
      either representation — codegen must handle both.

- [ ] **emit_tosa_matmul (2D float path)**: Lines 1498-1523 call emit_tosa_reshape
      to add/remove batch dim. When M, K, or N is DIM_DYNAMIC, the reshape produces
      a tensor.cast which may have wrong rank. The 2D matmul path should fall through
      to emit_batched_matmul for dynamic shapes.

- [ ] **emit_tosa_reduce keepdim=false reshape**: Line 2333 calls emit_tosa_reshape
      after reduction. When output has dynamic dims, this now falls back to tensor.cast,
      but the rank may differ (reduce removes a dim). May need tensor.collapse_shape.

- [ ] **ReduceMean scaling**: Lines 2874, 2899 call emit_tosa_reshape for the
      reciprocal constant. If any dim is dynamic, tensor.cast may not suffice.

- [ ] **Gemm path**: The Gemm codegen calls emit_tosa_matmul_2d_values which
      internally calls emit_tosa_reshape. If Gemm inputs have dynamic dims, this
      path may fail.

- [ ] **Softmax Div**: The softmax's division step emits a linalg.generic with
      broadcast map. If the reduced dim is dynamic, the broadcast may not work.

- [ ] **BatchNorm reshapes**: Lines 4170-4194 reshape parameters [C] → [1,C,1,1].
      Shouldn't be affected by dynamic seq dims but verify.

- [ ] **tosa.const_scalar**: Used in matmul zero-points, negate zero-points,
      mul shift. These are always static scalars — should be fine.

### Fix strategy

For each failure:
1. Add IR dump before verify (`eprintln!("{}", module.as_operation().to_string())`)
2. Find the failing op in the dump
3. Trace back to which emit function produced it
4. Add DIM_DYNAMIC guard (same pattern as emit_tosa_reshape)
5. Remove IR dump, run tests

### Testing

After all fixes:
```sh
# Clean cache and run with fresh compilation
cargo run --release -- clean-cache
cargo run --release -- run tests/fixtures/ --prompt "Hello" --max-tokens 5

# Expected: prefill ~7s (cached), decode ~X s per token, NO recompilation per token
```

## Performance targets

| Phase | Time | Notes |
|-------|------|-------|
| Load models | ~2.3s | Parse ONNX (2 models) + tokenizer |
| Prefill (cold) | ~28s | First-ever compile for this bucket |
| Prefill (warm) | ~7.5s | Cache hit: dlopen + inference |
| Decode (cold) | ~?s | First-ever compile for decode model |
| Decode (warm) | ~0.85s/token | Cache hit: dlopen + inference |

The decode model compiles ONCE (dynamic past_sequence_length). After that,
every token is just inference — no recompilation.
