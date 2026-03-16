# GPT-2 Emitter Runtime Debug Notes

**IMPORTANT:** Always `source env-llvm21-dev.sh` before any cargo command.
This sets `LLVM_SYS_211_PREFIX`, `MLIR_SYS_210_PREFIX`, `MLIR_SYS_LINK_SHARED`,
and `RUSTFLAGS` with rpath to the patched LLVM 21 build. Without it, you'll link
against the system LLVM which has the expand-strided-metadata crash.

## Status

**SEGFAULT FIXED.** The null-pointer crash in `memrefCopy` has been resolved.
The emitter now runs GPT-2 prefill to completion, but produces **NaN/Inf logits**.

## Root cause of the SEGFAULT (fixed)

The emitter's dynamic Reshape path passed raw ONNX shape tensors (containing `-1`
for "infer this dimension") directly to `tensor.reshape`, which does NOT handle
`-1`. The `-1` was interpreted as a literal dimension size (0xFFFFFFFFFFFFFFFF),
causing massive `memref.alloc` calls that returned null pointers.

### gdb evidence

Using `debug_memref4.py`, the first `memrefCopy` call had:
```
src: alloc=0x0 aligned=0x0 *** NULL *** offset=0 sizes=[-1, 4, 768, 9216] strides=[2304, 1, 0, 0]
dst: alloc=0x0 aligned=0x0 *** NULL *** offset=0 sizes=[-1, 4, 768, 3072] strides=[768, 1, 0, 0]
```
- `sizes[0] = -1` = unresolved dimension (should be batch_size=1)
- 9216 = 768 × 12 (attention QKV projection)
- 3072 = 768 × 4 (FFN intermediate)
- Both alloc/aligned pointers are NULL because allocating `-1` elements fails

### The fix

Added `emit_resolve_reshape_dims()` in `graph_builder.rs` which emits MLIR ops to:
1. Detect `-1` entries via `arith.cmpi eq` + `arith.select`
2. Detect `0` entries (ONNX "copy from input dim") via same mechanism
3. Compute `total_input_elems / product_of_known_dims` for the inferred dim
4. Replace `-1` with the computed value at runtime

The emitter's `Reshape` handler now extracts individual dims from the runtime
shape tensor, resolves special values, then builds a corrected shape tensor.

### MLIR `arith.cmpi` predicate encoding (LLVM 21)

From `ArithBase.td`:
- `eq = 0`, `ne = 1`, `slt = 2`, `sle = 3`, `sgt = 4`, `sge = 5`
- `ult = 6`, `ule = 7`, `ugt = 8`, `uge = 9`

Use `Attribute::parse(ctx, "0 : i64")` for eq, `"1 : i64"` for ne, etc.
The `#arith.cmpipredicate<eq>` syntax does NOT work in LLVM 21.

## Current issue: NaN/Inf logits

After the reshape fix, the test runs to completion but:
```
thread 'gpt2_prefill_emitter_only' panicked at tests/gpt2_e2e.rs:151:5:
logits contain NaN/Inf
```

This is a numerical correctness issue, not a crash. Possible causes:
- Attention score computation (QK^T / sqrt(d)) may produce Inf before softmax
- Missing or incorrect attention mask application
- LayerNorm epsilon handling
- Accumulation of errors from dynamic-dim overhead in the emitter path

## How to reproduce

```bash
source env-llvm21-dev.sh
rm -rf ~/.cache/homura/
cargo test --test gpt2_e2e gpt2_prefill_emitter_only -- --ignored --nocapture
```

Requires patched LLVM 21 (fix for expand-strided-metadata, PR #186834).

## Useful commands

### Dump IR at each stage
```bash
HOMURA_DUMP_IR=1 cargo test --test gpt2_e2e gpt2_prefill_emitter_only -- --ignored --nocapture
# Pre-pass: /tmp/homura_gb_pre_passes.mlir
# Post-pass: /tmp/homura_gb_post_passes.mlir
```

### Run under gdb with memref descriptor inspection
```bash
source env-llvm21-dev.sh
gdb -batch \
  -ex "source debug_memref4.py" \
  -ex "run gpt2_prefill_emitter_only --ignored --nocapture" \
  -ex "bt 5" \
  --args target/debug/deps/gpt2_e2e-*
```

### Dump memref descriptors
```bash
HOMURA_DUMP_MEMREFS=1 cargo test --test gpt2_e2e gpt2_prefill_emitter_only -- --ignored --nocapture
```

### Run passes step by step with mlir-opt
```bash
/home/edgar/data/llvm-21/bin/mlir-opt /tmp/homura_gb_pre_passes.mlir \
  --pass-pipeline="builtin.module(func.func(canonicalize,cse))" -o /tmp/step1.mlir

/home/edgar/data/llvm-21/bin/mlir-opt /tmp/step1.mlir \
  --pass-pipeline="builtin.module(one-shot-bufferize{bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map unknown-type-conversion=identity-layout-map})" \
  -o /tmp/step_bufferized.mlir
```

### Compare emitter vs mapper output
```bash
# Run the comparison test:
cargo test --test gpt2_e2e gpt2_prefill_emitter_matches_mapper_output -- --ignored --nocapture
```

## Key files

- `src/onnx/emitter.rs` — ONNX→MLIR emitter (Reshape handler at ~line 682)
- `src/graph_builder.rs` — MLIR op emission (`emit_resolve_reshape_dims` at ~line 2700)
- `tests/gpt2_e2e.rs` — GPT-2 integration tests
- `debug_memref4.py` — gdb Python script for inspecting memref descriptors at `memrefCopy`
- `/tmp/homura_gb_pre_passes.mlir` — emitter pre-pass IR (regenerate with HOMURA_DUMP_IR=1)
- `env-llvm21-dev.sh` — env vars for building against patched LLVM 21
