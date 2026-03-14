# Homura

A Rust ML inference framework that traces tensor operations, compiles them through MLIR, and JIT-executes native machine code.

```rust
use homura::{Tensor, DType, begin_trace, take_trace, Compiler};

begin_trace();
let a = Tensor::new(&[4], DType::F32);
let b = Tensor::new(&[4], DType::F32);
let c = (&a + &b).relu();
let trace = take_trace();

let compiled = Compiler::compile(&trace, &[c.id()]).unwrap();
let result = compiled.run(&[&[1.0, -2.0, 3.0, -4.0], &[5.0, 6.0, -7.0, 8.0]]);
assert_eq!(result, vec![6.0, 4.0, 0.0, 4.0]);
```

## How it works

Operations aren't executed eagerly. They're recorded into a trace — a flat list of ops — then compiled all at once into optimized machine code via MLIR.

```
Tensor ops  -->  Trace (Vec<Op>)  -->  MLIR (linalg.generic)  -->  LLVM  -->  JIT
```

The compiler emits [linalg.generic](https://mlir.llvm.org/docs/Dialects/Linalg/) operations on tensor types, then runs MLIR's bufferization and lowering passes to produce LLVM IR. This approach was chosen over manual loop emission because `linalg.generic` can lower to both CPU loops and GPU kernels — the same IR supports multiple backends.

See [docs/design.md](docs/design.md) for a detailed walkthrough of the architecture, MLIR lowering pipeline, and JIT ABI.

## Building

Requires LLVM 21 with MLIR C API support (`libMLIR-C.so`).

```sh
cargo build
```

## Running

```sh
cargo run --example add    # element-wise add demo
cargo run --example ops    # all ops demo (sub, mul, div, neg, relu)
cargo test                 # 30 tests
```

## Current status

- Rank-1 tensors, F32/F64/I32/I64
- Element-wise ops: Add, Sub, Mul, Div, Neg, Relu
- CPU JIT via MLIR ExecutionEngine
- `linalg.generic`-based codegen (GPU-ready IR)

## Roadmap

**Milestone 1** — N-D tensors, matmul, broadcast, softmax, eval sugar → run a hand-coded MLP

**Milestone 2** — ONNX/safetensors loading, conv2d, layer norm, GPU → run a real pre-trained model

**Milestone 3** — graph optimizations, dynamic shapes, autograd, memory planning → production-grade

See [docs/design.md](docs/design.md) for details.
