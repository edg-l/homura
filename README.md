# Homura

A Rust ML inference framework that traces tensor operations, compiles them through MLIR, and JIT-executes native machine code.

```rust
use homura::{Tensor, DType, Buffer, begin_trace};

begin_trace();
let a = Tensor::new(&[2, 3], DType::F32);
let b = Tensor::new(&[2, 3], DType::F32);
let c = (&a + &b).relu();

let a_buf = Buffer::from_slice::<f32>(&[1.0, -2.0, 3.0, -4.0, 5.0, -6.0], &[2, 3], DType::F32);
let b_buf = Buffer::from_slice::<f32>(&[0.5, 2.5, -1.0, 4.5, -3.0, 7.0], &[2, 3], DType::F32);
let result = c.eval(&[a_buf, b_buf]);
// [1.5, 0.5, 2.0, 0.5, 2.0, 1.0]
```

### ONNX model inference

```rust
use homura::Model;

let model = Model::load("model.onnx").unwrap();
let input = Buffer::from_slice::<f32>(&input_data, &[1, 1, 28, 28], DType::F32);
let output = model.run(&[&input]).unwrap();
```

### CLI

```sh
homura info model.onnx                     # inspect model graph
homura run model.onnx                      # run with zero input
homura run model.onnx --input data.bin --shape 1,1,28,28  # run with data
```

## How it works

Operations aren't executed eagerly. They're recorded into a trace — a flat list of ops — then compiled all at once into optimized machine code via MLIR.

```
Tensor ops  -->  Trace (Vec<Op>)  -->  MLIR (TOSA + linalg)  -->  LLVM  -->  JIT
```

The compiler emits [TOSA](https://mlir.llvm.org/docs/Dialects/TOSA/) dialect ops as the primary IR (add, sub, mul, matmul, reshape, etc.), with `linalg.generic` fallback for ops TOSA doesn't cover (float div, integer matmul). TOSA's well-tested lowering passes handle conversion to linalg, then bufferization and LLVM lowering produce the final machine code.

For ONNX models, the `Model` API parses the protobuf, replays the graph through the tracing system, compiles, and provides a simple `load`/`run` interface.

See [docs/design.md](docs/design.md) for a detailed walkthrough of the architecture, MLIR lowering pipeline, and JIT ABI.

## Building

Requires LLVM 21 with MLIR C API support (`libMLIR-C.so`).

```sh
cargo build
```

## Running

```sh
cargo run -- info tests/fixtures/mnist-12.onnx    # inspect ONNX model
cargo run -- run tests/fixtures/mnist-12.onnx     # run inference
cargo run --example onnx_mnist -- digit.png       # classify a digit image
cargo run --example add                           # element-wise add demo
cargo run --example ops                           # all ops demo
cargo run --example mlp                           # hand-coded MLP
cargo test                                        # 241 tests
```

## Current status

- N-D tensors, F32/F64/I32/I64 with broadcasting
- Element-wise ops: Add, Sub, Mul, Div, Neg, Relu, Exp, Tanh
- Matmul, Gemm (general matrix multiply with transpose/scaling)
- Conv2d (NCHW layout, padding, stride, dilation)
- MaxPool2d (NCHW layout, padding, stride)
- Reductions: ReduceSum, ReduceMax (with keepdim)
- Reshape (with -1 dimension inference), Flatten
- Softmax (composed from reductions + exp)
- TOSA-based codegen (with linalg.generic fallback)
- ONNX model loading and inference (`Model::load` / `Model::run`)
- CLI: `homura info` / `homura run`
- MNIST CNN end-to-end (mnist-12.onnx classifies digits correctly)
- CPU JIT via MLIR ExecutionEngine

## Roadmap

**Milestone 1** (complete) — N-D tensors, matmul, broadcast, softmax, eval sugar. Runs a hand-coded MLP.

**Milestone 2** (in progress) — TOSA backend, ONNX loading, Conv2d, MaxPool2d, MNIST CNN (done). BatchNorm, GlobalAvgPool, ResNet-18 (remaining).

**Milestone 3** — GPU backend (swap linalg-to-loops for GPU tiling passes)

**Milestone 4** — Graph optimizations, dynamic shapes, autograd, memory planning

See [docs/design.md](docs/design.md) for details.
