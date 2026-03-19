# Homura Design Document

Homura is a Rust ML inference framework that compiles models through MLIR/LLVM to native shared libraries. Per-kernel compilation (IREE-style) with parallel codegen via rayon. Supports ONNX and HuggingFace safetensors models.

## The Big Picture

```
model (ONNX or HF safetensors)
  -> partition into kernel groups
  -> per-kernel MLIR emission (linalg ops)
  -> transform schedule (tile + vectorize)
  -> bufferize -> LLVM IR -> .o -> link .so
  -> ExecutionPlan (Rust-side buffer routing)
```

Each heavy op becomes its own compiled kernel. Lightweight ops are grouped between heavy ops. All kernels compile in parallel. A Rust-side ExecutionPlan routes buffers between kernels at runtime.

## How It Works: Step by Step

### 1. Model Loading and Partitioning

**ONNX path:** `parser.rs` loads the ONNX protobuf into `OnnxModel` (nodes, weights, dynamic inputs). `emitter.rs` partitions nodes into kernel groups.

**HF path:** `hf/config.rs` loads `config.json` into `TransformerConfig`. `hf/safetensors.rs` loads weights. `hf/emitter.rs` emits a fixed transformer architecture (embedding -> N x attention+FFN layers -> output) as kernel groups.

In both cases:
- Each heavy op (Conv, MatMul, Gemm) becomes its own kernel
- Lightweight ops (Reshape, Transpose, Add, Relu) are grouped between heavy ops
- KV Concat groups are split via `split_kv_concat_groups` into pre-concat, concat, and post-concat sub-groups

### 2. Per-Kernel MLIR Emission

Each kernel group gets its own MLIR context and module. `GraphBuilder` emits **linalg dialect** ops directly:
- `linalg.matmul`, `linalg.batch_matmul`, `linalg.conv_2d_nchw_fchw` for heavy ops
- `linalg.generic` for elementwise, reductions, gather, etc.
- `tensor.empty` + `linalg.fill` for output allocation
- Fused Gemm+residual Add via accumulator trick

### 3. Transform Schedule (tile + vectorize)

`build_transform_schedule` in compiler.rs emits transform dialect sequences:
- Match 3D contractions (matmul), 4D contractions (batched matmul), Conv2D NCHW
- Tile with `tile_using_forall` (cache level) + `tile_using_for` (register level) + `pad`
- Outline the forall loop into a temporary function via `transform.loop.outline`
- Vectorize the outlined function with `vectorize_children_and_apply_patterns`
- LLVM inlines the outlined function during O2 codegen

Kernels without tileable ops (BatchNorm, Relu, elementwise) don't match the schedule -- the transform-interpreter is a no-op, they go straight to scalar lowering with compact LLVM IR.

### 4. Bufferize and Lower to LLVM

```
linalg ops on tensors
  -> transform-interpreter (tile + vectorize)
  -> one-shot-bufferize (tensor -> memref)
  -> convert-linalg-to-loops
  -> lower-affine -> scf -> cf
  -> convert to LLVM dialect
  -> LLVM IR -> .o
```

### 5. Parallel Compilation and Linking

All kernels compile in parallel via rayon. Each produces a `.o` object file. All objects are linked into a single `.so` via `link_shared_lib`. The `.so` is loaded once via dlopen; each kernel's function is resolved via dlsym.

### 6. Execution -- ExecutionPlan

`ExecutionPlan` holds the compiled kernels and a sequence of `KernelStep`s. Each step specifies input/output buffer slot indices and either a compiled kernel index or a `NativeOp` (e.g., Concat for KV cache ops).

The buffer pool routes data between kernels. Dead intermediates are recycled via a free-list. Model inputs are borrowed (zero-copy), outputs are extracted.

### 7. KV Cache (Autoregressive Decode)

For models with KV Concat ops (axis=-2, 2 inputs), `KvPlanInfo` tracks which buffer slots are past_kv inputs and present_kv outputs.

Two execution paths share the same plan:
- `run()` -- treats KV Concats as normal concat (prefill path)
- `run_kv()` -- intercepts KV Concats via `KvPlanInfo`, routes through `KvCache.append_key/value` + `view_key/value` (decode path)

`KvCache` pre-allocates `[1, heads, max_seq_len, head_dim]` per layer. Generation flow: `prefill (run) -> init_kv_cache -> decode loop (run_kv)`.

### 8. Chat Mode

Interactive multi-turn chat with persistent KV cache across turns. Chat templates are loaded from `tokenizer_config.json` and rendered via minijinja (Jinja2), so any HF model's chat format works automatically.

Each turn:
1. Render the full conversation with the chat template
2. Compute delta tokens (only new tokens since last turn)
3. Feed delta through `run_kv` to extend the existing KV cache (incremental prefill)
4. Decode loop generates the assistant response
5. KV cache persists for the next turn

This avoids re-processing the entire conversation history each turn.

### 9. Native ABI

The compiled function ABI uses **N-D memref descriptors** -- C structs with allocated_ptr, aligned_ptr, offset, sizes[N], strides[N]. The `llvm.emit_c_interface` attribute generates a C-compatible wrapper `_mlir__mlir_ciface_compute` that accepts a packed pointer array.

## HF Model Support

```
config.json + model.safetensors
  -> TransformerConfig + TransformerWeights
  -> emit_transformer_plan (fixed architecture)
  -> ExecutionPlan + weight buffers

HfModel
  |-- run()     -> full prefill (empty KV cache)
  |-- run_kv()  -> KV-cached decode step
  |-- generate()  -> single-prompt generation
  \-- chat (via HfGenerationContext)
       |-- incremental prefill across turns
       \-- persistent KV cache
```

Supported features:
- Grouped-query attention (GQA) via `num_key_value_heads`
- Explicit `head_dim` (Qwen3-style, where head_dim != hidden_size/num_heads)
- QK-norm (per-head RMSNorm on Q/K after projection, auto-detected from weights)
- RoPE with configurable theta
- Tied/untied word embeddings

Architectures tested: Qwen2.5, Qwen3 (any decoder-only transformer with standard HF config fields should work).

## ONNX Support

```
.onnx file --prost--> ModelProto --parser.rs--> OnnxModel
   |                                              |
   |                              emitter.rs: partition + emit per-kernel MLIR
   |                                              |
   v                                              v
Model { plan, weights } <-- compile <-- ExecutionPlan + kernels
   |
   |-- model.run(&[input])    -> Vec<Buffer>  (prefill / non-KV)
   \-- model.run_kv(&[input]) -> Vec<Buffer>  (KV-cached decode)
```

The emitter walks the ONNX graph per kernel group, emitting linalg ops into each kernel's MLIR module. Shape-prep ops (Shape, Gather, Concat for shapes) are constant-folded at emit time.

**Supported ONNX ops:** Add, Sub, Mul, Div, Neg, Relu, Exp, Tanh, Pow, Sqrt, Cast, MatMul, Gemm, Softmax, Clip, Reshape, Flatten, Gather, Slice, Concat, Split, Transpose, Where, Conv, MaxPool, BatchNormalization, GlobalAveragePool, ReduceMean, ReduceSum, ReduceMax, Constant, Shape, ConstantOfShape, Range, Squeeze, Unsqueeze, and more.

**Symbolic dimensions:** Models with dynamic dimensions (like KV cache `past_sequence_length`) use `SymDim` expression tracking. Compiled once with dynamic dims, resolved at runtime from actual input shapes.

**Multiple outputs:** `Model::run()` returns `Vec<Buffer>`. GPT-2 produces logits + 24 KV cache tensors.

## Compilation Cache

Compiled `.so` files are cached on disk at `~/.cache/homura/` (or `HOMURA_CACHE_DIR`). The cache key is a hash of:
- Model bytes (any model change invalidates)
- Input shapes (different seq_len = different compilation)
- Compiler fingerprint: homura version, LLVM version, host CPU name + features

On cache hit, compilation is skipped entirely -- the `.so` is loaded via dlopen in milliseconds. Power-of-2 bucket padding for sequence lengths limits the number of unique compilations to at most 6 (32, 64, 128, 256, 512, 1024).

`homura clean-cache` removes all cached files.

## Text Generation

Two model backends share the same `GenerativeModel` trait and `generate_streaming` loop:

```rust
// HF model (Qwen, etc.)
let model = HfModel::load(Path::new("model_dir"))?;
let tokenizer = HfTokenizer::from_file(&path.join("tokenizer.json"))?;
let text = model.generate(&tokenizer, "Hello", 50, 1024, &SamplingConfig::default())?;

// ONNX model (GPT-2) with unified KV cache
let mut gen = UnifiedKvGenerator::load("model_dir", 1024, 50256)?;
let text = gen.generate("Hello", 50);
```

Sampling: temperature, top-k, top-p (nucleus), min-p, repetition penalty, frequency penalty, presence penalty, seeded RNG.

## Architecture Decisions

### AOT compilation, not JIT

The original design used MLIR's ExecutionEngine (JIT). This was replaced with ahead-of-time compilation to native `.so` files because:
- The JIT took ~33s for GPT-2 with no way to cache the result
- `dump_to_object_file` failed silently for models using `memrefCopy`
- AOT produces a standard `.so` that can be cached on disk and loaded instantly

The AOT path: MLIR -> `mlirTranslateModuleToLLVMIR` -> `LLVMRunPasses("default<O3>")` -> `LLVMTargetMachineEmitToFile` -> `cc -shared` -> dlopen. Uses `llvm-sys` for LLVM C API bindings.

### Linalg as the primary dialect

All ops are emitted directly as `linalg` dialect operations (matmul, conv, generic). No TOSA. This gives full control over tiling, vectorization, and lowering without depending on TOSA's pass pipeline.

### Per-kernel compilation (IREE-style)

Each heavy op gets its own MLIR module, context, and compilation pipeline. This enables parallel compilation via rayon and fine-grained disk caching. Lightweight ops are grouped between heavy ops to avoid excessive kernel launch overhead.

### Deferred tracing, not eager execution

Operations record to a trace instead of executing immediately. This lets the compiler see the entire computation graph before generating code, enabling global optimizations. The same pattern as JAX and `torch.compile`.

## Source Layout

```
src/
|-- lib.rs              Public API re-exports
|-- dtype.rs            DType enum (F32, F64, I32, I64)
|-- shape.rs            Shape wrapper over Vec<u64> with broadcast
|-- main.rs             CLI: homura run / chat / info / clean-cache
|-- compiler.rs         Transform schedule, emit_object_files, link_shared_lib
|-- runtime.rs          Buffer, CompiledGraph, ExecutionPlan, KvCache
|-- cache.rs            Disk-based per-kernel compilation cache
|-- generate.rs         GenerativeModel trait, generate_streaming, sampling
|-- kv_generate.rs      UnifiedKvGenerator (ONNX KV-cached generation)
|-- tokenizer.rs        Byte-level BPE tokenizer (GPT-2)
|-- llvm_ffi.rs         LLVM C API FFI helpers
|-- log.rs              Logging macros
|-- op.rs               Op enum definitions
|-- graph_builder/
|   |-- mod.rs          MLIR emission (GraphBuilder, GraphContext)
|   |-- emit_linalg.rs  Linalg op emission (matmul, conv, etc.)
|   |-- emit_reshape.rs Reshape/transpose/slice emission
|   \-- emit_arithmetic.rs  Elementwise and reduction emission
|-- hf/
|   |-- mod.rs          Module root
|   |-- config.rs       TransformerConfig (generic HF config.json)
|   |-- model.rs        HfModel (load, run, run_kv, generate, chat)
|   |-- emitter.rs      Transformer plan emission (attention + FFN kernels)
|   |-- weights.rs      TransformerWeights (safetensors -> weight buffers)
|   |-- safetensors.rs  Safetensors file loading
|   |-- tokenizer.rs    HfTokenizer (wraps tokenizers crate)
|   |-- precompute.rs   RoPE cos/sin tables, causal masks
|   \-- chat.rs         Chat template rendering (minijinja), stop token detection
\-- onnx/
    |-- mod.rs          Model struct (load/run/run_kv), lazy compilation
    |-- proto.rs        Prost-generated protobuf types
    |-- parser.rs       ONNX ModelProto -> OnnxModel
    |-- emitter.rs      Per-kernel MLIR emission, partition_nodes, split_kv_concat_groups
    \-- sym_shapes.rs   Symbolic dimension expression tracking (SymDim)

examples/
|-- onnx_mnist.rs       MNIST digit classification
\-- onnx_resnet.rs      ResNet-18 image classification

scripts/
|-- download_gpt2.sh    Download GPT-2 ONNX models + tokenizer
\-- download_mnist.sh   Download MNIST ONNX model
```

## Dependencies

- **melior** -- Rust bindings for MLIR's C API. IR construction and pass management.
- **mlir-sys** -- Low-level FFI to `libMLIR-C.so`.
- **llvm-sys** -- LLVM C API bindings for AOT compilation (target machine, pass builder, object emission).
- **prost** / **prost-build** -- Protobuf compilation for ONNX model parsing.
- **protobuf-src** -- Vendored protoc compiler.
- **clap** -- CLI argument parsing.
- **libc** -- dlopen/dlsym for loading compiled `.so` files.
- **serde** / **serde_json** -- Config and tokenizer JSON parsing.
- **tokenizers** -- HuggingFace tokenizer library (BPE, WordPiece, etc.).
- **minijinja** -- Jinja2 template engine for HF chat templates.
- **hf-hub** -- HuggingFace Hub API for model downloads.
- **rayon** -- Parallel kernel compilation.
- **memmap2** -- Memory-mapped file I/O for safetensors.
- Requires **LLVM 21** with `libMLIR-C.so` and `libmlir_c_runner_utils.so`.

## Current Limitations

- **CPU only** -- no GPU backend
- **No autograd** -- forward pass only
- **F32 only for inference** -- no quantization (int8/int4)

## Performance

| Model | Params | Kernels | Compile time | Decode |
|---|---|---|---|---|
| MNIST | -- | 6 | 97ms | -- |
| ResNet-18 | 11M | 41 | 467ms | -- |
| GPT-2 | 124M | 158 compiled + 24 native | ~650ms | ~50 tok/s |
| Qwen2.5-0.5B | 494M | 74 compiled + 48 native | ~950ms | ~10 tok/s |

## Roadmap

### Completed

- Per-kernel MLIR compilation with parallel codegen via rayon
- Transform dialect schedule (tile + vectorize) for matmul/conv
- Symbolic dimension tracking (compile once, resolve at runtime)
- KV cache with persistent pre-allocated buffers
- HuggingFace model support (safetensors, auto-download from Hub)
- QK-norm for Qwen3-style models
- Interactive multi-turn chat mode with incremental KV cache
- Jinja2 chat template rendering (any HF model's format)
- GPT-2 at ~50 tok/s, Qwen2.5-0.5B at ~10 tok/s on CPU

### Next

- Packed GEMM layout for better cache utilization
- AVX-512 micro-kernel tuning
- Operator fusion (matmul + bias + relu as one kernel)
- GPU backend (CUDA via `gpu-to-nvvm`)
- More architectures (Llama, Mistral, Phi, Gemma)
- Weight quantization (INT8/INT4)
