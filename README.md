# Homura

A Rust ML inference framework that compiles ONNX models through MLIR/LLVM to native shared libraries and executes them. Per-kernel compilation (IREE-style) with parallel codegen via rayon. Runs MNIST, ResNet-18, GPT-2.

### ONNX model inference

```rust
use homura::{Model, Buffer, DType};

let model = Model::load("model.onnx").unwrap();
let input = Buffer::from_slice::<f32>(&input_data, &[1, 1, 28, 28], DType::F32);
let outputs = model.run(&[&input]).unwrap();  // Vec<Buffer>
```

### CLI

```sh
homura info model.onnx                          # inspect model graph
homura run model.onnx                           # run with zero input
homura run model.onnx --input data.bin --shape 1,1,28,28
homura run tests/fixtures/ --prompt "Hello world" --max-tokens 50  # GPT-2 text generation
homura clean-cache                              # clear compiled .so cache
```

## How it works

Each heavy ONNX op (Conv, MatMul, Gemm) is compiled as an independent kernel with its own MLIR context. Lightweight ops (BatchNorm, Relu, Add) are grouped between heavy ops. All kernels compile in parallel via rayon. A Rust-side `ExecutionPlan` routes buffers between kernels at runtime.

```
ONNX model
  → partition into kernel groups
  → per kernel: MLIR emission → transform schedule (tile + vectorize) → bufferize → LLVM IR → .so
  → link all .so files
  → ExecutionPlan routes buffers between kernels
```

The transform schedule tiles matmuls (3D, 4D contractions) and Conv2D into cache-level + register-level tiles, outlines the tiled region into a temporary function, and vectorizes it with `vectorize_children_and_apply_patterns`. Vector contractions lower via outerproduct for FMA fusion (`vfmadd`). Untiled ops stay as scalar loops — LLVM's O2 handles them. This keeps LLVM IR compact and compilation fast.

Models with dynamic dimensions (e.g., KV cache sequence length) are compiled once with symbolic shape tracking — a parallel `SymDim` expression system resolves buffer shapes at runtime without recompilation.

Compiled kernels are cached on disk per-kernel. Subsequent runs with the same model load instantly.

## Building

Requires patched LLVM 21 with MLIR C API support.

```sh
source env-llvm21-dev.sh
cargo build
```

GPT-2 model files (not in repo due to size):
```sh
./scripts/download_gpt2.sh
```

## Running

```sh
cargo run --example onnx_mnist                    # MNIST digit classification
cargo run --example onnx_resnet                   # ResNet-18 image classification
cargo run -- run tests/fixtures/ --prompt "The meaning of life is" --max-tokens 20
cargo run -- clean-cache                          # clear compilation cache
cargo test                                        # ~150 tests
```

## Performance

| Model | Kernels | Compile time | Inference |
|---|---|---|---|
| MNIST | 6 | 97ms | — |
| ResNet-18 | 41 | 467ms | — |
| GPT-2 (124M) | 146 | ~1.2s | ~3 tok/s decode |

## Current status

- **ONNX ops**: Conv2d, MatMul, Gemm, BatchNorm, Add, Sub, Mul, Div, Relu, Sigmoid, Tanh, Softmax, MaxPool, GlobalAvgPool, Reshape, Flatten, Gather, Slice, Concat, Split, Transpose, Squeeze, Unsqueeze, Where, Cast, Shape, ConstantOfShape, Range, and more
- **Compilation**: Per-kernel MLIR → LLVM, parallel via rayon, cached on disk
- **Tiling**: Transform dialect schedule for matmul (3D, 4D) and Conv2D (7D NCHW)
- **Vectorization**: Outerproduct lowering with FMA fusion (`vfmadd` on AVX-512/AVX2)
- **Dynamic shapes**: Symbolic dim tracking (`SymDim` expressions) — compile once, resolve at runtime
- **Dtype**: F32, F64, I32, I64
- **Tokenizer**: Byte-level BPE (GPT-2 compatible)
- **Generation**: Autoregressive text generation with KV-cache
- **Models**: MNIST CNN, ResNet-18, GPT-2 (124M) all run end-to-end

## Roadmap

See [perf.md](perf.md) for the performance improvement roadmap.

Next priorities:
- Unified single-model inference (eliminate separate prefill/decode models)
- Packed GEMM layout for better cache utilization
- AVX-512 micro-kernel tuning
- Operator fusion (matmul + bias + relu as one kernel)
- Support for modern models (SmolLM2, Llama-family architecture)
