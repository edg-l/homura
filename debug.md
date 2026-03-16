# GPT-2 Emitter Debug Notes

**IMPORTANT:** Always `source env-llvm21-dev.sh` before any cargo command.

## Current status

- **Prefill model: WORKING.** Both `gpt2_prefill_emitter_only` and
  `gpt2_prefill_emitter_matches_mapper_output` pass. All 457 unit tests pass.
- **Decode model: WORKING.** `gpt2_decode_emitter_dynamic` passes.
  Compiles and runs correctly with dynamic `past_sequence_length`.
- **"double free or corruption (!prev)" on exit** is a pre-existing libLLVM cleanup
  issue — also happens with the working mapper path. Ignore it.

## Completed: Eliminate `tensor.reshape` via collapse/expand_shape

### The problem (solved)

The emitter used `tensor.reshape` (with a runtime shape tensor) for ONNX Reshape
ops. After bufferization, this became `memref.reshape` which reinterprets memory
with new strides — when the resolved `-1` dimension was wrong, strides became
garbage → SEGFAULT.

### The fix: `compute_reassociation()` + collapse/expand decomposition

Replaced all `tensor.reshape` usage with `tensor.collapse_shape`/`tensor.expand_shape`
which compute strides correctly via reassociation indices. Key components:

1. **`compute_reassociation(in_shape, out_shape)`** — Analyzes shapes to find the
   optimal reassociation using bidirectional matching:
   - Match static dims from the END (they anchor the grouping unambiguously)
   - Distribute remaining input dims from the FRONT (ONNX contiguous merge: leading
     dims merge before trailing dims)
   - Returns `Collapse`, `Expand`, `CollapseExpand`, or `Identity`

2. **`emit_reshape()` / `emit_reshape_from_index_dims()`** — Try collapse/expand
   first, fall back to `tensor.reshape` only when constraints aren't met.

3. **Partial const_i64 propagation** — Concat now propagates known values even when
   some inputs are unknown (using `CONST_I64_UNKNOWN` sentinel), so downstream
   Reshape ops get partial static shape info for better reassociation matching.

4. **Cast-to-dynamic fallback** — For expand where input dims are static but output
   group has dynamic dims (MLIR requires collapsed dim to be dynamic), cast input
   dims to dynamic via `tensor.cast` first, then expand.

### Key discoveries

- **`tensor.collapse_shape` supports multiple dynamic dims per group.** The previous
  code had a false constraint (≤1 dynamic per group). MLIR computes the collapsed
  dim as the product at runtime. Tested and confirmed with `mlir-opt`.

- **`tensor.expand_shape` requires collapsed dim to be dynamic when expanded group
  has any dynamic dim.** Trying to expand a static dim into a dynamic group fails
  the MLIR verifier. Workaround: `tensor.cast` the input to make the dim dynamic.

- **`tensor.cast` for type refinement (dynamic→static) is UNSAFE for dynamic shapes.**
  The static values from const_i64 may come from a specific input shape at compile
  time, not the actual runtime shape. Using `tensor.cast` to refine `?→1` when the
  runtime dim might be 5 causes segfaults.

- **Reassociation ambiguity with matching static values:** `[1, ?, 768] → [1, ?]`
  could be `[[0], [1, 2]]` (keep batch, merge seq*hidden) or `[[0, 1], [2]]` (merge
  batch*seq, keep hidden). Only the latter is correct for ONNX Gemm flatten. The
  end-matching algorithm resolves this by matching 768↔768 from the right first,
  then assigning extra dims to the first group.

### Remaining tensor.reshape (1 instance)

One same-rank reshape `[?, ?, 768] → [1, 1, ?]` still uses `tensor.reshape` as
fallback. This is a type refinement case that can't be safely handled with
`tensor.cast` (runtime shape may differ from compile-time constants). It doesn't
cause issues because the runtime -1 resolution produces correct strides for this
simple case.

### What's already done (don't redo)

- **const_i64 propagation** through Shape, Gather, Unsqueeze, Squeeze, Concat,
  Cast, Reshape, Slice, Sub, Add, Mul, Div — makes most shape subgraph values
  available at compile time
- **Partial const_i64 for Concat** — uses `CONST_I64_UNKNOWN` sentinel so known
  values propagate even when some inputs are dynamic
- **Static -1 inference** in `emit_reshape` — when all input dims are known,
  the `-1` dim is resolved statically in the output type
- **DIM_DYNAMIC handling** — `Dim::Fixed(DIM_DYNAMIC)` treated as `None` in emitter
- **emit_where** uses broadcast-aware indexing maps + improved dyn_source selection
- **emit_collapse_shape_nd_to_3d** always uses native collapse (no multi-dynamic fallback)
- **emit_expand_shape_3d_to_nd** uses batch reference tensor for dynamic batch dims
- **compute_concrete_output_shapes** resolves keep_dynamic symbols from inputs

### MLIR notes

- `arith.cmpi` predicate: eq=0, ne=1, slt=2, sle=3, sgt=4, sge=5, ult=6, ule=7, ugt=8, uge=9
  Use `Attribute::parse(ctx, "0 : i64")` — the `#arith.cmpipredicate<eq>` syntax doesn't work
- `tensor.collapse_shape` infers its result type from input + reassociation — you
  cannot override it with a different result type via OperationBuilder. The result
  type passed to the builder MUST match what MLIR infers.
- `tensor.collapse_shape` DOES support multiple dynamic dims per group (confirmed
  with mlir-opt and full pass pipeline).
- `tensor.expand_shape` with dynamic output dims: if ANY dim in a group is dynamic,
  the corresponding input dim MUST also be dynamic (MLIR verifier enforces this).
  Workaround: `tensor.cast` the input to make dims dynamic first.
- `tensor.expand_shape` needs `output_shape` operands for dynamic dims +
  `static_output_shape` attribute with `i64::MIN` sentinel for dynamic positions.
- `tensor.cast` can refine dynamic→static or static→dynamic, but is UNSAFE when
  the runtime shape doesn't match the cast target.
- `memref.reshape` reinterprets memory (no copy) with contiguous row-major strides —
  source must have identity layout. Strides computed from new shape at runtime.

### Commit history

```
f3de340 eliminate tensor.reshape via collapse/expand decomposition, fix decode segfault
16729c5 debug.md: add concrete reshape patterns, mapper comparison, commit history, constraint details
f0de921 debug.md: full plan for eliminating tensor.reshape via collapse/expand decomposition
31f50af smarter collapse/expand: use native ops when safe, emit_reshape as fallback
f76e0c6 update debug.md: prefill fixed, decode model runtime crash documented
4a2f210 fix emit_where broadcasting + expand_shape for dynamic dims + output shape resolution
```

## How to reproduce

```bash
source env-llvm21-dev.sh
rm -rf ~/.cache/homura/

# Prefill (works):
cargo test --test gpt2_e2e gpt2_prefill_emitter_only -- --ignored --nocapture

# Decode (works):
cargo test --test gpt2_e2e gpt2_decode_emitter_dynamic -- --ignored --nocapture

# Full generation:
cargo run -- run tests/fixtures --prompt "The meaning of life is" --max-tokens 20

# All unit tests:
cargo test
```

## Useful commands

```bash
# Dump IR
HOMURA_DUMP_IR=1 cargo test --test gpt2_e2e gpt2_decode_emitter_dynamic -- --ignored --nocapture

# Verify IR manually
/home/edgar/data/llvm-21/bin/mlir-opt /tmp/homura_gb_pre_passes.mlir -o /dev/null

# Run passes step by step
/home/edgar/data/llvm-21/bin/mlir-opt /tmp/homura_gb_pre_passes.mlir \
  --pass-pipeline="builtin.module(func.func(canonicalize,cse))" -o /tmp/step1.mlir

# Count reshape types in IR
grep -c 'tensor.reshape' /tmp/homura_gb_pre_passes.mlir
grep -c 'tensor.collapse_shape' /tmp/homura_gb_pre_passes.mlir
grep -c 'tensor.expand_shape' /tmp/homura_gb_pre_passes.mlir
```

## Key files

- `src/onnx/emitter.rs` — ONNX→MLIR emitter (Reshape handler ~line 742)
- `src/graph_builder.rs` — MLIR op emission:
  - `compute_reassociation` (~line 183) — bidirectional shape matching algorithm
  - `emit_reshape` (~line 2932) — tries collapse/expand, falls back to tensor.reshape
  - `emit_reshape_with_tensor` (~line 3148) — tensor.reshape fallback path
  - `emit_reshape_from_index_dims` (~line 3360) — dynamic path with collapse/expand
  - `emit_resolve_reshape_dims` (~line 3170) — runtime -1/0 resolution
  - `emit_collapse_shape_nd_to_3d` (~line 4842) — matmul batch collapse (always native)
  - `emit_expand_shape_3d_to_nd` (~line 4858) — matmul batch expand with batch ref
  - `emit_expand_shape_impl` (~line 4693) — core expand_shape with output_shape
  - `emit_expand_shape_impl_with_dyn_vals` (~line 4747) — expand with pre-computed dyn vals
  - `emit_tensor_cast` (~line 4958) — tensor.cast for type refinement
  - `emit_where` (~line 4349) — broadcast with improved dyn_source selection
- `src/onnx/mod.rs` — Model load/run, compile_model_emitter, compute_concrete_output_shapes
- `tests/gpt2_e2e.rs` — GPT-2 integration tests
- `debug_memref4.py` — gdb Python script for inspecting memref descriptors
- `env-llvm21-dev.sh` — env vars for patched LLVM 21
