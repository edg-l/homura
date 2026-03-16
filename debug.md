# GPT-2 Emitter Debug Notes

**IMPORTANT:** Always `source env-llvm21-dev.sh` before any cargo command.

## Current status

- **Prefill model: WORKING.** Both `gpt2_prefill_emitter_only` and
  `gpt2_prefill_emitter_matches_mapper_output` pass. All 457 unit tests pass.
- **Decode model: SEGFAULT at runtime.** Compiles through the full MLIR pipeline
  but crashes during JIT execution (`vmovss` load at wrong offset inside `compute()`).
- **"double free or corruption (!prev)" on exit** is a pre-existing libLLVM cleanup
  issue — also happens with the working mapper path. Ignore it.

## Next task: Eliminate `tensor.reshape` — use collapse/expand_shape everywhere

### The problem

The emitter uses `tensor.reshape` (with a runtime shape tensor) for ONNX Reshape
ops that go through the dynamic path. After bufferization, this becomes
`memref.reshape` which reinterprets memory with new strides computed from the shape
tensor at runtime. When the resolved `-1` dimension is wrong (off by one, wrong
divui input), the strides become garbage → SEGFAULT.

Evidence from gdb: the first 3 `memrefCopy` calls in the decode model have garbage
strides like `[2304, 1, 140735984714131, 1]` for a `[1, 1, 768, 2304]` tensor.

The mapper path avoids this entirely — it uses `tosa.reshape` for known shapes and
`collapse_shape`/`expand_shape` for rank changes. It NEVER uses `tensor.reshape`.

### The fix: decompose every tensor.reshape into collapse→expand

Replace all `tensor.reshape` usage with a two-step pattern:
1. `tensor.collapse_shape` to flatten to 1D (or the appropriate lower rank)
2. `tensor.expand_shape` to expand to the target shape

This avoids `memref.reshape` entirely. The collapse→expand pattern is what MLIR
natively supports with correct stride computation.

### Implementation plan

**Where the changes go:**
- `src/graph_builder.rs`: `emit_reshape()` (line ~2494) and
  `emit_reshape_from_index_dims()` / `emit_reshape_with_tensor()` (line ~2674+)
- The `emit_resolve_reshape_dims()` runtime -1 resolution code can be REMOVED
  once all reshapes go through collapse/expand (which don't need runtime shape
  tensors for known reassociation patterns)

**Key constraint for collapse/expand_shape:**
- Each reassociation group can have AT MOST ONE dynamic dim
- `emit_expand_shape_impl` uses `tensor.dim` on the input for dynamic output dims
  in a group — this works correctly when there's only one dynamic dim per group
  (MLIR can infer its value from the input dim and the static dims in the group)
- When multiple dynamic dims share a group, fall back to... what? Options:
  - Use `tosa.reshape` with `tosa.const_shape` (requires static target shape)
  - Use a `linalg.generic` copy with appropriate indexing maps
  - Accept the limitation and use `tensor.reshape` only for that rare case

**The decomposition algorithm:**
Given ONNX Reshape from `[a, b, c, d]` to `[x, y, z]`:
1. Compute which source dims map to which target dims (contiguous grouping)
2. For dims that merge: use `tensor.collapse_shape` with appropriate reassociation
3. For dims that split: use `tensor.expand_shape` with appropriate reassociation
4. For dims that just pass through: identity in the reassociation

The hard part: inferring the reassociation from two shapes when some dims are dynamic.
When both shapes are fully known (from const_i64), this is trivial.

**Simpler alternative decomposition (flatten-to-1D):**
1. `tensor.collapse_shape %input [[0,1,...,n-1]]` → flatten to 1D
2. `tensor.expand_shape %flat [[0,1,...,m-1]]` → expand to target

**Constraints on this approach:**
- Step 1 (collapse to 1D): the input can have at most ONE dynamic dim, because
  `collapse_shape` requires each group to have ≤1 dynamic dim. If the input has
  multiple dynamic dims, we must first collapse groups of contiguous dims that
  each have ≤1 dynamic, then collapse those intermediate results further.
- Step 2 (expand from 1D): same constraint — the target must have ≤1 dynamic dim
  in the single reassociation group `[[0,1,...,m-1]]`. Since all target dims are in
  one group, this means the target can have AT MOST ONE dynamic dim total.

For the decode model, most reshapes have exactly one dynamic dim
(`past_sequence_length`), so this covers nearly all cases.

**Fallback for multiple dynamic dims:** Use `tosa.reshape` with `tosa.const_shape`
when all target dims are statically known (via const_i64). Otherwise, use
`tensor.reshape` as a last resort (only for truly unknowable shapes).

### What's already done (don't redo)

- **const_i64 propagation** through Shape, Gather, Unsqueeze, Squeeze, Concat,
  Cast, Reshape, Slice, Sub, Add, Mul, Div — this makes most shape subgraph
  values available at compile time
- **Static -1 inference** in `emit_reshape` — when all input dims are known,
  the `-1` dim is resolved statically in the output type
- **DIM_DYNAMIC handling** — `Dim::Fixed(DIM_DYNAMIC)` treated as `None` in emitter
- **emit_where** uses broadcast-aware indexing maps (not pre-broadcast + identity)
- **emit_collapse_shape_nd_to_3d** and **emit_expand_shape_3d_to_nd** use native
  collapse/expand when safe (≤1 dynamic dim in batch group), emit_reshape as fallback
- **compute_concrete_output_shapes** resolves keep_dynamic symbols from inputs

### MLIR notes

- `arith.cmpi` predicate: eq=0, ne=1, slt=2, sle=3, sgt=4, sge=5, ult=6, ule=7, ugt=8, uge=9
  Use `Attribute::parse(ctx, "0 : i64")` — the `#arith.cmpipredicate<eq>` syntax doesn't work
- `tensor.collapse_shape` infers its result type from input + reassociation — you
  cannot override it with a different result type via OperationBuilder
- `tensor.expand_shape` with dynamic output dims: uses `tensor.dim` on the input
  for ALL output dims in the same reassociation group — only correct when the group
  has at most one dynamic dim
- `memref.reshape` reinterprets memory (no copy) with contiguous row-major strides —
  source must have identity layout. Strides computed from new shape at runtime.

### Concrete reshape patterns in the decode model

From the decode model's pre-pass IR (172 `tensor.reshape` ops), the patterns are:

1. **Identity**: `[1,1] → [1,1]` — no-op, can skip
2. **Flatten for Gemm**: `[?,1,768] → [?,?]` (batch*seq × hidden) — collapse_shape [[0,1],[2]]
3. **Unflatten after Gemm**: `[?,2304] → [?,?,?]` (batch × seq × qkv) — expand_shape [[0,1],[2]]
4. **QKV split reshape**: `[?,?,768] → [?,?,12,64]` — expand_shape [[0],[1],[2,3]]
5. **Attention head merge**: `[?,?,12,64] → [?,?,768]` — collapse_shape [[0],[1],[2,3]]
6. **Matmul batch collapse**: `[1,12,?,64] → [12,?,64]` — collapse_shape [[0,1],[2],[3]]
7. **Matmul batch expand**: `[12,?,64] → [1,12,?,64]` — expand_shape [[0,1],[2],[3]]

Key: patterns 2-3 have the Gemm flatten where batch*seq merges. For decode,
batch=1 and seq=1, so this is `[1,1,768] → [1,768]`. The `?` dims come from
the emitter not const-propagating far enough — if const_i64 covers the Concat
that produces the shape tensor, the static path handles it.

Patterns 4-5 split/merge the head dimension (768 = 12 × 64). These are always
static in the last dims.

Patterns 6-7 are already handled by `emit_collapse_shape_nd_to_3d` /
`emit_expand_shape_3d_to_nd` (use native collapse/expand when ≤1 dynamic in batch).

### How the mapper avoids this

The mapper path uses `Tensor::reshape(&target_shape)` which records an `Op::Reshape`
in the trace. The `Compiler` then emits `tosa.reshape` with `tosa.const_shape` —
the target shape is always fully static because symbolic dims were resolved before
tracing. The TOSA reshape gets lowered by `tosa-to-linalg` into proper
`linalg.generic` ops with correct indexing maps. No `tensor.reshape` or
`memref.reshape` ever appears.

### Commit history for this session

```
a4bfcbb fix emitter dynamic reshape: resolve ONNX -1 and 0 shape values
4d66648 propagate static dims through emitter shape subgraph ops
c4d1224 propagate const_i64 through Cast, Range, Reshape; fix matmul collapse shapes
8a00c89 propagate const_i64 through Slice; use emit_reshape for static collapses
e6cf297 fix emitter: propagate const_i64 through Sub/Add, complete GPT-2 prefill
53e00b9 fix emitter: treat DIM_DYNAMIC as None in input shapes
4a2f210 fix emit_where broadcasting + expand_shape for dynamic dims + output shape resolution
31f50af smarter collapse/expand: use native ops when safe, emit_reshape as fallback
f0de921 debug.md: full plan for eliminating tensor.reshape via collapse/expand decomposition
```

## How to reproduce

```bash
source env-llvm21-dev.sh
rm -rf ~/.cache/homura/

# Prefill (works):
cargo test --test gpt2_e2e gpt2_prefill_emitter_only -- --ignored --nocapture

# Decode (crashes):
cargo test --test gpt2_e2e gpt2_decode_emitter_dynamic -- --ignored --nocapture

# Full generation (prefill works, decode crashes):
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

# gdb with memref inspection
gdb -batch -ex "source debug_memref4.py" \
  -ex "run gpt2_decode_emitter_dynamic --ignored --nocapture" \
  -ex "bt 5" --args target/debug/deps/gpt2_e2e-*

# Count reshape types in IR
grep -c 'tensor.reshape' /tmp/homura_gb_pre_passes.mlir
grep -c 'tensor.collapse_shape' /tmp/homura_gb_pre_passes.mlir
grep -c 'tensor.expand_shape' /tmp/homura_gb_pre_passes.mlir
```

## Key files

- `src/onnx/emitter.rs` — ONNX→MLIR emitter (Reshape handler ~line 682)
- `src/graph_builder.rs` — MLIR op emission:
  - `emit_reshape` (~line 2494) — static path, uses tensor.reshape with index shape tensor
  - `emit_reshape_with_tensor` (~line 2674) — dynamic path
  - `emit_reshape_from_index_dims` (~line 2886) — builds shape tensor from index values
  - `emit_resolve_reshape_dims` (~line 2700) — runtime -1/0 resolution
  - `emit_collapse_shape_nd_to_3d` (~line 4412) — matmul batch collapse
  - `emit_expand_shape_3d_to_nd` (~line 4457) — matmul batch expand
  - `emit_expand_shape_impl` (~line 4533) — core expand_shape with output_shape
  - `emit_where` (~line 3788) — uses broadcast maps
- `src/onnx/mod.rs` — Model load/run, compile_model_emitter, compute_concrete_output_shapes
- `tests/gpt2_e2e.rs` — GPT-2 integration tests
- `debug_memref4.py` — gdb Python script for inspecting memref descriptors
- `env-llvm21-dev.sh` — env vars for patched LLVM 21
