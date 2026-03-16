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

### Approach history

**Approach 1: `tensor.reshape` (current)**
- Compiles successfully, `mlir-opt` runs full pipeline without errors
- Segfaults at runtime in generated code
- Crash at `compute+13072`, `vmovups` loading from address 192 (0xC0)

**Approach 2: `tensor.expand_shape` / `tensor.collapse_shape` (abandoned)**
- MLIR itself crashes in `expand-strided-metadata` pass
- `mlir::IntegerAttr::getValue()` in `dispatchIndexOpFoldResult`
- LLVM bug, related to #61158, not fixed in LLVM 21

**Industry approach:** IREE uses `flow.tensor.reshape` (custom), ONNX-MLIR uses
`memref.reinterpret_cast` (post-bufferize). Neither uses upstream `tensor.reshape`
or `expand_shape`/`collapse_shape` with dynamic dims in production.

### gdb findings (detailed)

- Crash: `vmovups -0xc0(%rax,%rdx,4),%zmm0` at compute+13072
- `rax = 0xC0 = 192`, `rdx = 0` → load from address 192-192 = 0 → segfault
- `%rbx` points to a stack struct; the code reads fields:
  - `rbx+0x80 = 0` (used as stride multiplier → 0)
  - `rbx+0xC0 = 192` (added as base → becomes the "pointer")
  - `rbx+0x130 = 9216` (another stride/size: 12*768 or 12*64*12)
- The struct at `rbx` looks like loop state with multiple packed descriptors
- First entries at `rbx+0x00..0x10` are NULL (not valid memref pointers)
- `rbx+0x10` contains a valid heap pointer (0x555595...) in the offset field

### Crash traced to specific MLIR ops (line ~589 of pre-passes IR)

The crash is in the **QKV attention head reshape + transpose**:

```mlir
// line 589: reshape QKV: [batch, seq, 2304] → [batch, seq, 12, 64]
%reshape_84 = tensor.reshape %312(%337)
    : (tensor<?x?x?xf32>, tensor<4xi64>) -> tensor<?x?x?x?xf32>

// line 590: transpose to [batch, heads, seq, head_dim]
%338 = tosa.transpose %reshape_84 {perms = array<i32: 0, 2, 1, 3>}
    : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
```

The lowered code's memcpy loop (from the transpose) has wrong stride values.
The value `192 = 3 × 64 = 3 × head_dim` appears where a full stride should be.
This suggests `tensor.reshape`'s memref view computes `stride[2] = head_dim = 64`
correctly but stride[1] comes out wrong.

For reshape `[?, ?, 2304] → [?, ?, 12, 64]`, the correct strides should be
`[seq*2304, 2304, 64, 1]`. If the reshape lowering produces `[seq*2304, 192, 64, 1]`
(where 192 = 3*64 instead of 12*64=768), that would explain 192 appearing as a
base address in the copy loop.

Wait: 192 = 3*64. But the reshape splits dim 2 (size 2304) into [12, 64].
The correct stride for the "12" dimension should be 64 (since elements for
different heads are 64 apart). And stride for the "seq" dimension should be
2304. So the view strides are `[seq*2304, 2304, 64, 1]`. This is what
`memref.reshape` should produce.

The value 192 doesn't match any expected stride for this reshape. It DOES match
`3 * head_dim` which is a stride from a DIFFERENT reshape (the one that splits
QKV: `[?, ?, 768] → [?, ?, 3, 12, 64]` or similar). The buffer reuse hypothesis
may be correct after all: the shape buffer `alloc_28` was overwritten with values
from a DIFFERENT reshape by the time THIS reshape's copy loop executes.

### tensor.reshape bug FIXED (commit 9932181)

Replaced all `tensor.reshape` with `linalg.generic` copy-reshape.
Zero `tensor.reshape` ops in the decode model IR. The previous crash
at `compute+13072` (vmovups from address 192) is gone.

### NEW crash: NULL pointer dereference (rdx=0)

After the copy-reshape fix, a NEW crash appears:
- `compute+13347`: `vmovss (%rdx,%rsi,4),%xmm0` with `rdx=0`
- This is a scalar float load from a NULL pointer
- Different from the previous reshape crash — new bug to investigate
- Likely another codegen issue with dynamic dims (wrong pointer in
  a linalg.generic body or tensor.extract)

### Ruled out causes

- **memref<0xi64> dangling pointers**: Fixed (use sentinel), crash unchanged.
  0-element memrefs have valid descriptors now but crash is at the same location.
- **Input/output descriptor construction**: Confirmed correct via HOMURA_DUMP_MEMREFS.
  All 1408 inputs have valid pointers, correct shapes. 25 outputs correct.
- **matmul codegen**: Tested `run_dynamic_matmul_2d` — works correctly at runtime.
- **Buffer reuse in one-shot-bufferize**: Initially suspected but NOT confirmed.
  `finalize-memref-to-llvm` lowers `memref.reshape` by reading shape buffer values
  at the reshape site (immediate, not lazy). Reuse should be safe.

### Next: map crash to MLIR source op

The crash at `compute+13072` is in a vectorized memcpy loop. The struct at `%rbx`
has wrong field values (0 stride, 192 as "base pointer"). Need to:

1. Dump LLVM IR (`HOMURA_DUMP_IR=1`)
2. Find which LLVM IR block/function corresponds to compute+13072
3. Trace back through the LLVM IR to find which memref op produced the bad descriptor
4. The `0xC0 = 192` value is suspicious: 192 = 48*4 = 12*16 or 3*64 — possibly
   a stride from a [1, 12, ?, 64] tensor (stride for dim 3 = 64, stride for dim 2 = 64,
   stride for dim 1 = ?*64)

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
| Decode (warm) | ~0.85s/token | Target: cache hit, dlopen + inference |

The decode model compiles ONCE (dynamic past_sequence_length). After that,
every token is just inference — no recompilation.
