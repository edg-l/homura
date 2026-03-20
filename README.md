# Homura

A Rust ML inference framework that compiles models through MLIR/LLVM to native shared libraries. Per-kernel compilation with parallel codegen via rayon. BF16 mixed-precision matmul on AVX-512. Supports ONNX and HuggingFace/safetensors models.

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

The transform schedule tiles matmuls (3D, 4D, 5D contractions) and Conv2D with adaptive tile sizes based on available parallelism. Large matmuls use OpenMP via `tile_using_forall` for multi-threaded execution, with inner `tile_using_for` loops that LLVM auto-vectorizes to AVX-512 FMA/BF16. GQA attention uses zero-copy floordiv indexing to avoid materializing head expansions, with dedicated 5D tiling. Untiled ops stay as scalar loops.

Models with dynamic dimensions (e.g., KV cache sequence length) are compiled once with symbolic shape tracking -- a parallel `SymDim` expression system resolves buffer shapes at runtime without recompilation.

Compiled kernels are cached on disk per-kernel. Subsequent runs with the same model load instantly.

## Building

Requires a [patched LLVM 21](https://github.com/edg-l/llvm-project/tree/edgl/llvm-21.1.8-patched) with MLIR C API support, bug fixes, and added C bindings (e.g. `LLVMSplitModule`).

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

Benchmarked on AMD Ryzen 9 7900X3D (single CCD, auto-pinned).

| Model | Params | Kernels | Compile time | Decode |
|---|---|---|---|---|
| MNIST | -- | 6 | 96ms | -- |
| ResNet-18 | 11M | 41 | 422ms | -- |
| GPT-2 | 124M | 158 compiled + 24 native | ~650ms | ~50 tok/s |
| Qwen2.5-0.5B | 494M | -- | -- | ~36 tok/s (bf16) |
| Qwen3-0.6B | 600M | -- | -- | ~24 tok/s (bf16) |
| TinyLlama-1.1B | 1.1B | 68 | ~1s | ~20 tok/s (bf16) |
| Qwen2.5-1.5B | 1.5B | 86 | ~1.6s | ~13 tok/s (bf16) |
| SmolLM2-1.7B | 1.7B | 74 | ~1s | ~11 tok/s (bf16) |

Qwen3-0.6B decode breakdown (42ms/tok):

| Kernel | Time | % | Per layer | BW | % peak |
|---|---|---|---|---|---|
| MLP (28L) | 16.7ms | 40% | 0.60ms | 32 GB/s | 41% |
| QKV (28L) | 7.0ms | 17% | 0.25ms | 34 GB/s | 44% |
| Attn (28L) | 7.6ms | 18% | 0.27ms | 16 GB/s | 20% |
| LMHead | 7.9ms | 19% | -- | 40 GB/s | 51% |

Total weights: 1192 MB. Effective bandwidth: 28 GB/s (37% of 77 GB/s peak). Theory minimum: 15ms/tok.

## Current status

- **Model formats**: ONNX, HuggingFace safetensors (auto-detected)
- **HuggingFace Hub**: Download models by name (`Qwen/Qwen2.5-0.5B`), cached locally
- **ONNX ops**: Conv2d, MatMul, Gemm, BatchNorm, Add, Sub, Mul, Div, Relu, Sigmoid, Tanh, Softmax, MaxPool, GlobalAvgPool, Reshape, Flatten, Gather, Slice, Concat, Split, Transpose, Squeeze, Unsqueeze, Where, Cast, Shape, ConstantOfShape, Range, and more
- **HF architectures**: Decoder-only transformers with RoPE (Qwen2, Qwen3, Llama, and compatible architectures). QK-norm support for Qwen3. Config auto-detection from `config.json`. Instruct models auto-detected via chat template.
- **Chat mode**: Interactive multi-turn REPL with persistent KV cache, Jinja2 chat template rendering via minijinja, think block support for reasoning models (--think flag)
- **Compilation**: Per-kernel MLIR (linalg dialect) -> LLVM, parallel via rayon, cached on disk
- **Tiling**: Adaptive OpenMP-parallel tiling for 3D/4D/5D contractions (matmul, batched matmul, GQA attention), scaled to available cores
- **Vectorization**: LLVM auto-vectorization of tiled scalar loops to AVX-512 FMA/BF16
- **Mixed precision**: BF16 weight storage with F32 accumulation for matmul kernels on AVX-512 BF16
- **Zero-copy GQA**: Floordiv head indexing avoids materializing repeat_kv expansions
- **CPU affinity**: Auto-pins to single CCD on multi-chiplet AMD CPUs (Zen 3D/4) for consistent L3 cache behavior. Disable with `HOMURA_NO_PIN=1`
- **Dynamic shapes**: Symbolic dim tracking (`SymDim` expressions) -- compile once, resolve at runtime
- **Dtype**: F32, F64, I32, I64, BF16
- **Generation**: Streaming text output, persistent KV cache, configurable sampling (temperature, top-k, top-p, min-p, repetition/frequency/presence penalties, seed)
- **Profiling**: `make profile` runs decode profiling with per-kernel bandwidth analysis via `scripts/profile.py`
- **Models tested**: MNIST CNN, ResNet-18, GPT-2 (124M), Qwen2.5-0.5B (494M), Qwen2.5-1.5B (1.5B), Qwen3-0.6B (600M), SmolLM2-1.7B (1.7B), TinyLlama-1.1B (1.1B)

## Roadmap

### Phase 1: Quantized model support
- Load pre-quantized INT4/INT8 weights (GGUF, AWQ, GPTQ)
- Mixed-precision dequant-during-matmul kernels
- Packed GEMM layout for better cache utilization

### Phase 2: More architectures
- Llama 3/3.1 (8B quantized, 1B/3B native)
- Phi-4-mini (3.8B quantized)
- Gemma 3n (E2B/E4B -- efficient mobile-first models)
- Operator fusion (matmul + bias + activation as one kernel)

### Phase 3: GPU backend
- GPU dialect emission (`gpu-to-nvvm` for CUDA, `gpu-to-rocdl` for ROCm)
- Device memory management and host/device transfers
- GPU-specific tiling and scheduling
- Flash Attention (fused softmax(QK^T)V for O(n) memory)
