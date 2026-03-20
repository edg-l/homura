# Homura Design Document

Homura is a Rust ML inference framework that compiles models through MLIR/LLVM to native shared libraries. Per-kernel compilation (IREE-style) with parallel codegen via rayon. BF16 mixed-precision matmul on AVX-512. Supports ONNX and HuggingFace safetensors models.

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

Two transform schedule builders in compiler.rs:

**`build_transform_schedule`** (full tile + outline + vectorize):
- Match 3D contractions (matmul), 4D contractions (batched matmul), Conv2D NCHW
- Tile with `tile_using_forall` (cache level) + `tile_using_for` (register level) + `pad`
- Outline the forall loop into a temporary function via `transform.loop.outline`
- Vectorize the outlined function with `vectorize_children_and_apply_patterns`
- LLVM inlines the outlined function during O2 codegen

**`build_tile_parallel_schedule`** (tile for parallelism only):
- Match 3D, 4D, and 5D contractions (including GQA attention with floordiv head mapping)
- Tile with `tile_using_forall` for OpenMP parallelism + `tile_using_for` for register blocking
- No outline/vectorize -- relies on LLVM auto-vectorization to AVX-512 FMA/BF16
- Used for QKV projections and attention kernels where the full schedule is not needed

Kernels without tileable ops (BatchNorm, Relu, elementwise) don't match either schedule -- the transform-interpreter is a no-op, they go straight to scalar lowering with compact LLVM IR.

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

Multi-token KV cache append is supported: `run_kv` accepts input sequences longer than 1 token, enabling incremental prefill for chat turns without reprocessing the entire history. See [chat.md](chat.md) for details.

### 8. Native ABI

The compiled function ABI uses **N-D memref descriptors** -- C structs with allocated_ptr, aligned_ptr, offset, sizes[N], strides[N]. The `llvm.emit_c_interface` attribute generates a C-compatible wrapper `_mlir__mlir_ciface_compute` that accepts a packed pointer array.

## HF Model Support

```
config.json + model.safetensors
  -> TransformerConfig + TransformerWeights
  -> emit_transformer_plan (fixed architecture)
  -> ExecutionPlan + weight buffers

HfModel
  |-- run()     -> full prefill (empty KV cache)
  |-- run_kv()  -> KV-cached decode step (single or multi-token)
  |-- generate()  -> single-prompt generation
  \-- chat (via HfGenerationContext)
       |-- incremental prefill across turns
       \-- persistent KV cache
```

Supported features:
- Grouped-query attention (GQA) via `num_key_value_heads`, zero-copy floordiv head indexing
- Explicit `head_dim` (Qwen3-style, where head_dim != hidden_size/num_heads)
- QK-norm (per-head RMSNorm on Q/K after projection, auto-detected from weights)
- RoPE with configurable theta
- BF16 mixed-precision: bf16 weight storage with f32 accumulation (auto-detected from safetensors dtype)
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

Sampling: temperature, top-k, top-p (nucleus), min-p, repetition penalty, frequency penalty, presence penalty, seeded RNG (xorshift64).

### generate_streaming_from_ids

For chat mode, `generate_streaming_from_ids` accepts pre-tokenized token IDs directly, avoiding a decode-then-re-encode round-trip that would lose special tokens (`<|im_start|>`, `<|im_end|>`, etc.). This is critical for correct incremental prefill with chat templates.

### ThinkConfig

The `ThinkConfig` struct controls think-block handling during generation:
- `token_ids: Option<(u32, u32)>` -- the `<think>` and `</think>` token IDs
- `style_content: bool` -- when true, think content is shown in gray italic; when false, it is hidden entirely

Think tags are always hidden from output. The `--think` CLI flag sets `style_content = true`, so reasoning content is visible but visually distinct.

## Compilation Cache

Compiled `.so` files are cached on disk at `~/.cache/homura/` (or `HOMURA_CACHE_DIR`). The cache key is a hash of:
- Model bytes (any model change invalidates)
- Input shapes (different seq_len = different compilation)
- Compiler fingerprint: homura version, LLVM version, host CPU name + features

On cache hit, compilation is skipped entirely -- the `.so` is loaded via dlopen in milliseconds. Power-of-2 bucket padding for sequence lengths limits the number of unique compilations to at most 6 (32, 64, 128, 256, 512, 1024).

`homura clean-cache` removes all cached files.

## Architecture Decisions

### AOT compilation, not JIT

The original design used MLIR's ExecutionEngine (JIT). This was replaced with ahead-of-time compilation to native `.so` files because:
- The JIT took ~33s for GPT-2 with no way to cache the result
- `dump_to_object_file` failed silently for models using `memrefCopy`
- AOT produces a standard `.so` that can be cached on disk and loaded instantly

The AOT path: MLIR -> `mlirTranslateModuleToLLVMIR` -> `LLVMRunPasses("default<O3>")` -> `LLVMTargetMachineEmitToFile` -> `cc -shared` -> dlopen. Uses `llvm-sys` for LLVM C API bindings.

Requires a [patched LLVM 21](https://github.com/edg-l/llvm-project/tree/edgl/llvm-21.1.8-patched) with bug fixes and added C bindings (`LLVMSplitModule`, `tile_using_forall` crash fix, etc.).

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
|-- dtype.rs            DType enum (F32, F64, I32, I64, BF16)
|-- shape.rs            Shape wrapper over Vec<u64> with broadcast
|-- main.rs             CLI: homura run / chat / info / clean-cache
|-- compiler.rs         Transform schedule, emit_object_files, link_shared_lib
|-- runtime.rs          Buffer, CompiledGraph, ExecutionPlan, KvCache
|-- cache.rs            Disk-based per-kernel compilation cache
|-- cpu_affinity.rs     Auto-pin to single CCD on multi-chiplet AMD CPUs
|-- cpu_caps.rs         CPU feature detection (AVX-512 BF16, etc.)
|-- generate.rs         GenerativeModel trait, generate_streaming, sampling, ThinkConfig
|-- progress.rs         CLI progress bars for compilation and generation
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
|   |-- model.rs        HfModel (load, run, run_kv, generate), HfGenerationContext
|   |-- emitter.rs      Transformer plan emission (attention + FFN kernels)
|   |-- weights.rs      TransformerWeights (safetensors -> weight buffers)
|   |-- safetensors.rs  Safetensors file loading
|   |-- tokenizer.rs    HfTokenizer (wraps tokenizers crate)
|   |-- precompute.rs   RoPE cos/sin tables, causal masks
|   \-- chat.rs         ChatTemplate (minijinja), ChatMessage, stop/think token detection
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
- **minijinja** / **minijinja-contrib** -- Jinja2 template engine for HF chat templates, with Python string method compatibility (pycompat).
- **hf-hub** -- HuggingFace Hub API for model downloads.
- **rayon** -- Parallel kernel compilation.
- **memmap2** -- Memory-mapped file I/O for safetensors.
- **atty** -- TTY detection for streaming output vs. piped mode.
- Requires [**patched LLVM 21**](https://github.com/edg-l/llvm-project/tree/edgl/llvm-21.1.8-patched) with `libMLIR-C.so` and `libmlir_c_runner_utils.so`.

## CPU Affinity (Multi-Chiplet AMD)

`cpu_affinity.rs` auto-pins the process to a single CCD on AMD Zen 3D / Zen 4 chiplet CPUs. When OS threads migrate between CCDs, L3 contents are invalidated and bandwidth-bound inference slows down 2-3x. Pinning to one CCD eliminates this variance.

Reads Linux sysfs topology to discover CCDs (L3 cache domains). No-op on single-CCD CPUs. Disable with `HOMURA_NO_PIN=1`.

## BF16 Mixed Precision

On CPUs with AVX-512 BF16 (detected via `cpu_caps.rs`), matmul kernels use bf16 weight inputs with f32 accumulation. Weights are stored as bf16 in safetensors and loaded directly -- no F32 upcast at load time. The MLIR emission path generates `linalg.generic` with bf16 inputs and f32 output, which LLVM lowers to `vdpbf16ps` (dot product of bf16 pairs with f32 accumulate).

This halves memory bandwidth for weight-bound decode kernels, giving ~49% speedup on AVX-512 BF16 hardware.

## Current Limitations

- **CPU only** -- no GPU backend
- **No autograd** -- forward pass only
- **No quantization** -- bf16 and f32 only, no int8/int4

## Performance

Benchmarked on AMD Ryzen 9 7900X3D (single CCD, auto-pinned).

| Model | Params | Kernels | Compile time | Decode |
|---|---|---|---|---|
| MNIST | -- | 6 | 96ms | -- |
| ResNet-18 | 11M | 41 | 422ms | -- |
| GPT-2 | 124M | 158 compiled + 24 native | ~650ms | ~50 tok/s |
| Qwen2.5-0.5B | 494M | -- | -- | ~36 tok/s (bf16) |
| Qwen3-0.6B | 600M | -- | -- | ~24 tok/s (bf16) |

Qwen3-0.6B decode breakdown (42ms/tok, 1192 MB weights):

| Kernel | Time | % | Per layer | BW | % peak |
|---|---|---|---|---|---|
| MLP (28L) | 16.7ms | 40% | 0.60ms | 32 GB/s | 41% |
| QKV (28L) | 7.0ms | 17% | 0.25ms | 34 GB/s | 44% |
| Attn (28L) | 7.6ms | 18% | 0.27ms | 16 GB/s | 20% |
| LMHead | 7.9ms | 19% | -- | 40 GB/s | 51% |

Effective bandwidth: 28 GB/s (37% of 77 GB/s peak). Theory minimum: 15ms/tok at peak BW.

Profile with: `make profile MODEL=Qwen/Qwen3-0.6B TOKENS=300`

## Roadmap

### Completed

- Per-kernel MLIR compilation with parallel codegen via rayon
- Transform dialect schedule (tile + vectorize) for matmul/conv
- TileParallel schedule for QKV/attention with 5D GQA contraction tiling
- Symbolic dimension tracking (compile once, resolve at runtime)
- KV cache with persistent pre-allocated buffers
- HuggingFace model support (safetensors, auto-download from Hub)
- QK-norm for Qwen3-style models
- Interactive multi-turn chat mode with incremental KV cache
- Jinja2 chat template rendering (any HF model's format)
- Think block support for reasoning models (Qwen3)
- BF16 mixed-precision matmul (bf16 weights, f32 accumulation)
- Zero-copy GQA via floordiv head indexing
- Auto CCD pinning for multi-chiplet AMD CPUs
- CLI progress bars and decode profiling (`scripts/profile.py`)

### Next

- Packed GEMM layout for better cache utilization
- Operator fusion (matmul + bias + relu as one kernel)
- Weight quantization (INT8/INT4)
- More architectures (Llama, Mistral, Phi, Gemma)
- GPU backend (CUDA via `gpu-to-nvvm`)
