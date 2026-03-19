# Homura

A Rust ML inference framework that compiles models through MLIR/LLVM to native shared libraries. Per-kernel compilation with parallel codegen via rayon. Supports ONNX and HuggingFace/safetensors models.

### Quick start

```sh
homura run Qwen/Qwen2.5-0.5B --prompt "The capital of France is" --max-tokens 20
```

Models are downloaded and cached automatically from HuggingFace Hub.

### Interactive chat

```sh
homura chat Qwen/Qwen2.5-0.5B
homura chat Qwen/Qwen2.5-0.5B --system "You are a pirate." --max-tokens 500
homura chat Qwen/Qwen3-0.6B --think              # enable thinking/reasoning output
```

Multi-turn conversation with persistent KV cache across turns. Chat templates are loaded from `tokenizer_config.json` via minijinja, so any HF model's chat format works out of the box. Type `/help` for REPL commands.

The `--think` flag enables styled thinking output for reasoning models like Qwen3 -- think blocks are rendered in gray italic. Without the flag, think blocks are suppressed via `/no_think` in the system prompt.

### ONNX model inference

```rust
use homura::{Model, Buffer, DType};

let model = Model::load("model.onnx").unwrap();
let input = Buffer::from_slice::<f32>(&input_data, &[1, 1, 28, 28], DType::F32);
let outputs = model.run(&[&input]).unwrap();  // Vec<Buffer>
```

### CLI

```sh
homura run Qwen/Qwen2.5-0.5B --prompt "Hello" --max-tokens 50   # HF model by name
homura run ./my-model/ --prompt "Hello" --max-tokens 50          # local HF model
homura chat Qwen/Qwen2.5-0.5B                                   # interactive chat
homura chat Qwen/Qwen3-0.6B --think                             # chat with thinking
homura run model.onnx                                            # ONNX inference
homura run model.onnx --input data.bin --shape 1,1,28,28
homura info model.onnx                                           # inspect model graph
homura clean-cache                                               # clear compiled .so cache
```

Sampling options: `--temperature`, `--top-k`, `--top-p`, `--min-p`, `--seed`, `--repetition-penalty`, `--frequency-penalty`, `--presence-penalty`, `--stop`.

## How it works

Each heavy op (Conv, MatMul, Gemm) is compiled as an independent kernel with its own MLIR context. Lightweight ops (BatchNorm, Relu, Add) are grouped between heavy ops. All kernels compile in parallel via rayon. A Rust-side `ExecutionPlan` routes buffers between kernels at runtime.

```
model (ONNX, safetensors/HF)
  -> partition into kernel groups
  -> per kernel: MLIR emission (linalg ops) -> transform schedule (tile + vectorize) -> bufferize -> LLVM IR -> .o
  -> link all .o into unified .so
  -> ExecutionPlan routes buffers between kernels
```

The transform schedule tiles matmuls (3D, 4D contractions) and Conv2D with adaptive tile sizes based on available parallelism. Large matmuls use OpenMP via `tile_using_forall` for multi-threaded execution, with inner `tile_using_for` loops that LLVM auto-vectorizes to AVX-512 FMA. Untiled ops stay as scalar loops.

Models with dynamic dimensions (e.g., KV cache sequence length) are compiled once with symbolic shape tracking -- a parallel `SymDim` expression system resolves buffer shapes at runtime without recompilation.

Compiled kernels are cached on disk per-kernel. Subsequent runs with the same model load instantly.

## Building

Requires patched LLVM 21 with MLIR C API support.

```sh
source env-llvm21-dev.sh
cargo build
```

## Running

```sh
cargo run -- run Qwen/Qwen2.5-0.5B --prompt "Hello" --max-tokens 20
cargo run -- chat Qwen/Qwen2.5-0.5B
cargo run -- chat Qwen/Qwen3-0.6B --think
cargo run --example onnx_mnist                                   # MNIST digit classification
cargo run --example onnx_resnet                                  # ResNet-18 image classification
cargo run -- clean-cache                                         # clear compilation cache
cargo test
```

Use `--verbose` / `-v` to see compilation progress (MLIR passes, kernel timing).

## Performance

| Model | Params | Kernels | Compile time | Decode |
|---|---|---|---|---|
| MNIST | -- | 6 | 97ms | -- |
| ResNet-18 | 11M | 41 | 467ms | -- |
| GPT-2 | 124M | 158 compiled + 24 native | ~650ms | ~50 tok/s |
| Qwen2.5-0.5B | 494M | 74 compiled + 48 native | ~950ms | ~10 tok/s |

## Current status

- **Model formats**: ONNX, HuggingFace safetensors (auto-detected)
- **HuggingFace Hub**: Download models by name (`Qwen/Qwen2.5-0.5B`), cached locally
- **ONNX ops**: Conv2d, MatMul, Gemm, BatchNorm, Add, Sub, Mul, Div, Relu, Sigmoid, Tanh, Softmax, MaxPool, GlobalAvgPool, Reshape, Flatten, Gather, Slice, Concat, Split, Transpose, Squeeze, Unsqueeze, Where, Cast, Shape, ConstantOfShape, Range, and more
- **HF architectures**: Decoder-only transformers with RoPE (Qwen2, Qwen3, and compatible architectures). QK-norm support for Qwen3. Config auto-detection from `config.json`.
- **Chat mode**: Interactive multi-turn REPL with persistent KV cache, Jinja2 chat template rendering via minijinja, think block support for reasoning models (--think flag)
- **Compilation**: Per-kernel MLIR (linalg dialect) -> LLVM, parallel via rayon, cached on disk
- **Tiling**: Adaptive OpenMP-parallel tiling for matmuls, scaled to available cores
- **Vectorization**: LLVM auto-vectorization of tiled scalar loops to AVX-512 FMA
- **Dynamic shapes**: Symbolic dim tracking (`SymDim` expressions) -- compile once, resolve at runtime
- **Dtype**: F32, F64, I32, I64
- **Generation**: Streaming text output, persistent KV cache, configurable sampling (temperature, top-k, top-p, min-p, repetition/frequency/presence penalties, seed)
- **Models tested**: MNIST CNN, ResNet-18, GPT-2 (124M), Qwen2.5-0.5B (494M)

## Roadmap

- Multimodal models (vision-language: image encoder + text decoder)
- More architectures (Llama, Mistral, Phi, Gemma)
- Packed GEMM layout for better cache utilization
- Flash Attention (fused softmax(QK^T)V for O(1) memory)
- Operator fusion (matmul + bias + activation as one kernel)
- Weight quantization (INT8/INT4)
- GPU backend (CUDA via `gpu-to-nvvm`)
