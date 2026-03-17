# Homura

Rust ML inference framework built on MLIR via melior.

## Architecture

Traces tensor operations into a computation graph, emits MLIR IR, lowers to LLVM, JIT-executes.

```
User code (Tensor ops)  →  Trace (Vec<Op>)  →  MLIR (TOSA + linalg)  →  LLVM  →  JIT
```

### Design process

- **Never choose the path of least resistance without confirming with the user.** When multiple approaches exist (e.g., quick hack vs. clean architecture), present the options and let the user decide. The "simplest thing that works" is not always the right choice — the user may prefer a more general or elegant approach even if it takes more effort.

### Key design decisions

- **Deferred tracing**: Operations record into a thread-local `Trace`, not executed eagerly. `.compile()` freezes the trace and emits MLIR.
- **TOSA-primary codegen**: Emits TOSA dialect ops (add, sub, mul, matmul, conv2d, max_pool2d, reshape, etc.) with `linalg.generic` fallback for ops TOSA lacks (float div, integer matmul). `tosa-to-linalg` passes handle lowering.
- **NCHW internally, NHWC at TOSA boundary**: Spatial ops (conv2d, max_pool2d) transpose NCHW→NHWC before the TOSA op, NHWC→NCHW after. MLIR can fuse adjacent transposes.
- **Backend-agnostic graph**: The `Trace`/`Op` layer has no CPU assumptions. GPU support can be added by swapping the lowering backend without refactoring the frontend.
- **ONNX graph replay**: The ONNX mapper replays the graph through the Tensor API, reusing shape inference, broadcast, and the entire compilation pipeline.

### Source layout

| Path | Purpose |
|---|---|
| `src/dtype.rs` | `DType` enum (F32, F64, I32, I64) with MLIR type conversion |
| `src/shape.rs` | `Shape` wrapper over `Vec<u64>` with broadcast |
| `src/op.rs` | `NodeId` and `Op` enum (Input, Add, Sub, Mul, Div, Neg, Relu, Exp, Tanh, Matmul, ReduceSum, ReduceMax, Reshape, Gemm, Conv2d, MaxPool2d, GlobalAvgPool, BatchNorm) |
| `src/trace.rs` | Thread-local `Trace` context, `begin_trace()`/`take_trace()`/`record()` |
| `src/tensor.rs` | `Tensor` handle with operator overloads, matmul, gemm, conv2d, max_pool2d, global_avg_pool, batch_norm, reshape, reductions, softmax, eval sugar |
| `src/compiler.rs` | Trace → MLIR IR emission (TOSA + linalg), pass pipeline, ExecutionEngine |
| `src/runtime.rs` | N-D `MemRefDescriptor`, `Buffer`, `CompiledGraph::run()` with JIT marshalling |
| `src/main.rs` | CLI: `homura info` (model inspector) and `homura run` (inference) |
| `src/onnx/mod.rs` | Public API: `Model` struct (`load`/`run`) |
| `src/onnx/proto.rs` | Re-export prost-generated protobuf types |
| `src/onnx/parser.rs` | ONNX ModelProto → internal `OnnxModel` representation |
| `src/onnx/mapper.rs` | Walk ONNX graph, replay through Tensor API |

### MLIR pass pipeline

```
func.func(tosa-make-broadcastable, tosa-to-linalg-named, tosa-to-linalg, tosa-to-arith, tosa-to-tensor)
→ func.func(canonicalize, cse)
→ transform-interpreter → func.func(canonicalize, cse) → symbol-dce
→ one-shot-bufferize → func.func(buffer-hoisting, promote-buffers-to-stack)
→ convert-linalg-to-loops
→ lower-affine → convert-scf-to-cf → canonicalize → cse → sccp
→ convert-math-to-llvm → expand-strided-metadata → finalize-memref-to-llvm
→ convert-arith-to-llvm → convert-index-to-llvm → convert-cf-to-llvm
→ convert-func-to-llvm → reconcile-unrealized-casts
```

TOSA passes are function-level. `transform-interpreter` applies the tiling schedule (tiles contraction `linalg.generic` ops via `foreach_match` + body pattern matching); `symbol-dce` removes the schedule after application. `canonicalize`+`cse` run after TOSA lowering and after tiling to simplify IR. `buffer-hoisting`+`promote-buffers-to-stack` optimize buffer placement. `sccp` does constant propagation. JIT uses LLVM `-O3`. `lower-affine` is needed for conv2d pad lowering. `expand-strided-metadata` handles `tensor.expand_shape` from broadcast rank promotion.

### JIT ABI

Functions use `llvm.emit_c_interface` (set as `Attribute::unit`). Arguments are N-D `MemRefDescriptor` structs passed via `invoke_packed` with double indirection. Conv2d/padded ops require `libmlir_c_runner_utils.so` loaded via dlopen (configurable with `MLIR_RUNNER_UTILS_PATH` env var).

### LLVM 21 TOSA API specifics

| Op | LLVM 21 requirement |
|---|---|
| `tosa.mul` | shift is 3rd operand `tensor<1xi8>`, not an attribute |
| `tosa.negate` | 3 operands: input, input1_zp, output_zp (zero-point tensors) |
| `tosa.clamp` | unified `min_val`/`max_val` attributes |
| `tosa.reshape` | target shape is operand via `tosa.const_shape`, not attribute |
| `tosa.slice` | 3 operands: input, start, size (all via `tosa.const_shape`); no stride support — stride>1 needs `linalg.generic` |
| `tosa.conv2d` | 5 operands: input, weight, bias, input_zp, weight_zp; requires `acc_type` attribute |
| `tosa.max_pool2d` | requires `acc_type` attribute; strict divisibility (pad+slice workaround) |
| `tosa.avg_pool2d` | 3 operands: input, input_zp, output_zp; requires `acc_type` attribute |
| `tosa.conv2d` stride | strict divisibility like max_pool2d (pad+slice workaround for non-divisible cases) |
| `tosa.cast` | NO I64 support — max integer width is int32; use arith dialect casts via `linalg.generic` for I64 paths |
| `tosa.gather` | 3D values + 2D I32 indices ONLY; not usable for general ONNX Gather (arbitrary axis/rank/I64 indices) |
| `tosa.transpose` | `perms` is `DenseI32ArrayAttr` (compile-time attribute), not a tensor operand |
| `tosa.pow` | native TOSA op; FP-only, two operands with same element type |
| `tosa.select` | native Where equivalent: i1 condition, X, Y — condition must be i1 (cast Bool/I64 first) |
| `tosa.concat` | variadic inputs (not binary), I32Attr axis |
| no `tosa.sqrt` | use `tosa.rsqrt` + `tosa.reciprocal`, or `linalg.generic` + `math.sqrt` |

### MLIR verification workflow

Always verify TOSA op signatures against the actual `.td` files before writing codegen:
```
/usr/lib/llvm/21/include/mlir/Dialect/Tosa/IR/TosaOps.td
/usr/lib/llvm/21/include/mlir/Dialect/Tosa/IR/TosaOpBase.td
```

## Dependencies

- **melior** (git dep) — MLIR Rust bindings with `ods-dialects` feature for TOSA
- **mlir-sys** (patched branch `support-shared-libs`) — shared LLVM/MLIR lib support
- **prost** / **prost-build** / **protobuf-src** — ONNX protobuf parsing
- **clap** — CLI argument parsing
- **libc** — dlopen for runner utils
- Requires LLVM 21 with `libMLIR-C.so` (Gentoo: `mlir-21.1.8-r1` from `edgar_repo` overlay)

## Build environment

Before building or running tests, **always** source the LLVM 21 dev environment:
```sh
source env-llvm21-dev.sh
```
This sets `LLVM_SYS_211_PREFIX`, `MLIR_SYS_210_PREFIX`, and linker rpath to use the patched LLVM 21 from `/home/edgar/data/llvm-21`.

Before running tests or benchmarks, **always** clear the compilation cache:
```sh
rm -rf ~/.cache/homura
```
Stale cached `.so` files from previous compilations will mask changes to the compiler pipeline.

## Commands

```sh
cargo test                                        # unit + integration tests
cargo run -- info tests/fixtures/mnist-12.onnx    # inspect model
cargo run -- run tests/fixtures/mnist-12.onnx     # run MNIST inference
cargo run -- run tests/fixtures/resnet18-v1-7.onnx # run ResNet-18 inference
cargo run -- run tests/fixtures --prompt "The meaning of life is" --max-tokens 50  # GPT-2 text generation
cargo run --example onnx_mnist -- digit.png       # classify digit image
cargo run --example add                           # element-wise add demo
cargo run --example ops                           # all ops demo
cargo run --example mlp                           # hand-coded MLP
```

## Current status

- N-D tensors, F32/F64/I32/I64 with broadcasting
- Element-wise ops: Add, Sub, Mul, Div, Neg, Relu, Exp, Tanh
- Matmul, Gemm (general matrix multiply with transpose/scaling)
- Conv2d (NCHW, with padding/stride/dilation, auto_pad=SAME_UPPER, pad+slice for TOSA divisibility)
- MaxPool2d (NCHW, with padding/stride, tosa.slice for ONNX floor-division compat)
- GlobalAvgPool (NCHW, via tosa.avg_pool2d with kernel=spatial dims)
- BatchNorm (composed from TOSA primitives: sub, rsqrt, mul, add)
- Reductions: ReduceSum, ReduceMax (with keepdim)
- Reshape (with -1 dimension inference), Flatten
- Softmax (composed from reductions + exp)
- TOSA-based codegen (with linalg.generic fallback for Div, integer Matmul/reductions)
- ONNX model loading and inference (`Model::load` / `Model::run`)
- CLI: `homura info` and `homura run` (with `--prompt`/`--max-tokens` for text generation)
- MNIST CNN runs end-to-end (mnist-12.onnx)
- ResNet-18 runs end-to-end (resnet18-v1-7.onnx)
- GPT-2 small runs end-to-end (decoder_model.onnx): `homura run <dir> --prompt "..." --max-tokens N`
- BPE tokenizer (vocab.json + merges.txt), greedy generation loop
- Compilation cache (hash model → cached LLVM bitcode) to avoid recompiling on every load
- CPU JIT via MLIR ExecutionEngine
- Dynamic dimensions: DIM_DYNAMIC sentinel, runtime shape ops (ShapeOf, ConstantOfShape, Range)
- Dynamic reshape: ONNX -1 dim inference at MLIR level, tensor.reshape with corrected shape tensors
- KV cache two-model architecture: prefill + decode with `past_sequence_length` as dynamic dim
- Decode model compiles once (dynamic dims), warm inference at ~0.30s/token
- dlti.dl_spec data layout on MLIR module for correct struct lowering

