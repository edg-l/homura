# Dynamic Dimensions — Remaining Work

## Status

The decode model (GPT-2 with-past) now **compiles successfully** through the entire
MLIR pipeline: trace → emission → verification → TOSA → linalg → SCF → LLVM IR → .so.

**Current blocker:** Segfault during JIT execution of the compiled decode model.
The crash is inside the generated machine code. All MLIR passes complete, the LLVM IR
is clean (no leftover unrealized_conversion_cast, affine.apply, etc.), but at runtime
something accesses invalid memory.

**Commit:** `be5338c` — all codegen fixes landed, tests pass (388/388).

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
- [x] Decode model compiles (MLIR verification passes)
- [x] MLIR pass pipeline succeeds (LLVM IR generated, .so emitted)

## Fixes applied in be5338c

1. **Range codegen 0-D extract**: `tensor.extract %t[]` (no indices) for rank-0 tensors,
   `tensor.extract %t[%c0]` for rank-1. Uses `trace.ops()[id].shape().rank()` to decide.

2. **tensor.expand_shape dynamic output_shape**: `promote_rank_with_reshape` now emits
   `tensor.dim` SSA values for each DIM_DYNAMIC position and passes them as operands.

3. **2D matmul with dynamic dims**: `emit_tosa_matmul_2d_values` uses
   `promote_rank_with_reshape` (expand_shape) for 2D→3D and `tensor.collapse_shape`
   for 3D→2D when dims are dynamic. Static case still uses `tosa.reshape`.

4. **Op::Matmul dispatch**: Dynamic-dim 2D matmuls route through `emit_batched_matmul`
   (linalg.generic) instead of the tosa.matmul 2D→3D→2D path.

5. **operandSegmentSizes**: `tensor.insert_slice` and `tensor.extract_slice` now emit
   the required `operandSegmentSizes` attribute for proper operand segmentation.

6. **i1 tensor type**: `RankedTensorType::new` for Where's i1 condition tensor now uses
   `dim_to_mlir()` instead of raw `shape.0` (was producing `-1` instead of `kDynamic`).

7. **Pass pipeline**: `convert-linalg-to-affine-loops` → `convert-linalg-to-loops`
   (affine loops can't handle dynamic bounds). Extra `lower-affine` after
   `expand-strided-metadata` to catch late-introduced affine.apply ops.

8. **Model API**: Added `Model::run_with_output_shapes()` and `Model::output_descs()`.
   KvGenerator decode step uses explicit output shapes instead of auto-inference.

## Remaining: JIT runtime segfault

### Symptoms

- Decode model .so loads and `_mlir__mlir_ciface_compute` is called
- Crash happens inside the generated machine code (no Rust backtrace)
- Prefill (static shapes) works fine with the same pass pipeline
- All 388 unit tests pass

### Likely root causes

1. **tensor.reshape lowering**: The decode model has ~145 `tensor.reshape` ops with
   dynamic shape tensors. These lower to memref reinterpretation (no copy, just new
   strides/offsets). If any shape tensor value is wrong at runtime (e.g., still
   contains DIM_DYNAMIC sentinel), the memref descriptor gets absurd dimensions.

2. **tensor.cast for same-rank dynamic reshapes**: `emit_tosa_reshape` falls back to
   `tensor.cast` which is a no-op at runtime — just reinterprets the type. If the
   actual runtime shape doesn't match the declared tensor type, subsequent code
   computes wrong offsets.

3. **tensor.collapse_shape strides**: The matmul 3D→2D collapse may produce wrong
   strides when the batch dimension is 1 but M/N are dynamic.

### Debug plan

- [ ] Run under gdb to get faulting instruction address
- [ ] Cross-reference with LLVM IR dump (`HOMURA_DUMP_IR=1` → `/tmp/homura_post_passes.mlir`)
- [ ] Trace back to which tensor.reshape / tensor.cast / collapse_shape produced the bad descriptor
- [ ] Optionally: write a minimal test with a small dynamic-dim model to isolate the crash

### IR dump

Set `HOMURA_DUMP_IR=1` to dump:
- `/tmp/homura_pre_passes.mlir` — MLIR before lowering passes
- `/tmp/homura_post_passes.mlir` — LLVM dialect IR after all passes

### Items from original TODO not yet hit (may be fine)

- [ ] **emit_tosa_reduce keepdim=false reshape**: Rank-changing with dynamic dims.
      Not triggered yet — may need tensor.collapse_shape if encountered.
- [ ] **ReduceMean scaling**: Reciprocal constant reshape with dynamic dims.
- [ ] **Softmax Div**: Broadcast map with dynamic reduced dim.
- [ ] **BatchNorm reshapes**: Likely fine (dynamic is seq dim, not channel).

## Performance targets

| Phase | Time | Notes |
|-------|------|-------|
| Load models | ~2.3s | Parse ONNX (2 models) + tokenizer |
| Prefill (cold) | ~28-30s | First-ever compile for this bucket |
| Prefill (warm) | ~7.5s | Cache hit: dlopen + inference |
| Decode (cold) | ~? | First-ever compile for decode model (compiles now!) |
| Decode (warm) | ~0.85s/token | Cache hit: dlopen + inference (blocked by segfault) |

The decode model compiles ONCE (dynamic past_sequence_length). After that,
every token is just inference — no recompilation.
