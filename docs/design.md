# Homura Design Document

Homura is a Rust ML inference framework that compiles ONNX models through MLIR/LLVM to native shared libraries. Per-kernel compilation (IREE-style) with parallel codegen via rayon. Runs MNIST, ResNet-18, GPT-2.

## The Big Picture

```
ONNX model
  → partition into kernel groups
  → per-kernel MLIR emission (linalg ops)
  → transform schedule (tile + vectorize)
  → bufferize → LLVM IR → .o → link .so
  → ExecutionPlan (Rust-side buffer routing)
```

Each heavy ONNX op becomes its own compiled kernel. Lightweight ops are grouped between heavy ops. All kernels compile in parallel. A Rust-side ExecutionPlan routes buffers between kernels at runtime.

## How It Works: Step by Step

### 1. ONNX Parsing and Partitioning

`parser.rs` loads the ONNX protobuf into `OnnxModel` (nodes, weights, dynamic inputs).
`emitter.rs` partitions nodes into kernel groups:
- Each heavy op (Conv, MatMul, Gemm) becomes its own kernel
- Lightweight ops (Reshape, Transpose, Add, Relu) are grouped between heavy ops
- KV Concat groups are split via `split_kv_concat_groups` into pre-concat, concat, and post-concat sub-groups

### 2. Per-Kernel MLIR Emission

Each kernel group gets its own MLIR context and module. `GraphBuilder` emits
**linalg dialect** ops directly (not TOSA):
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

Kernels without tileable ops (BatchNorm, Relu, elementwise) don't match the
schedule — the transform-interpreter is a no-op, they go straight to scalar
lowering with compact LLVM IR.

### 4. Bufferize and Lower to LLVM

```
linalg ops on tensors
  → transform-interpreter (tile + vectorize)
  → one-shot-bufferize (tensor → memref)
  → convert-linalg-to-loops
  → lower-affine → scf → cf
  → convert to LLVM dialect
  → LLVM IR → .o
```

### 5. Parallel Compilation and Linking

All kernels compile in parallel via rayon. Each produces a `.o` object file.
All objects are linked into a single `.so` via `link_shared_lib`. The `.so`
is loaded once via dlopen; each kernel's function is resolved via dlsym.

### 6. Execution — ExecutionPlan

`ExecutionPlan` holds the compiled kernels and a sequence of `KernelStep`s.
Each step specifies input/output buffer slot indices and either a compiled
kernel index or a `NativeOp` (e.g., Concat for KV cache ops).

The buffer pool routes data between kernels. Dead intermediates are recycled
via a free-list. Model inputs are borrowed (zero-copy), outputs are extracted.

### 7. KV Cache (Autoregressive Decode)

For models with KV Concat ops (axis=-2, 2 inputs), `KvPlanInfo` tracks
which buffer slots are past_kv inputs and present_kv outputs.

Two execution paths share the same plan:
- `run()` — treats KV Concats as normal concat (prefill path)
- `run_kv()` — intercepts KV Concats via `KvPlanInfo`, routes through
  `KvCache.append_key/value` + `view_key/value` (decode path)

`KvCache` pre-allocates `[1, heads, max_seq_len, head_dim]` per layer.
Generation flow: `prefill (run) → init_kv_cache → decode loop (run_kv)`.

### 8. Native ABI

The compiled function ABI uses **N-D memref descriptors** — C structs with
allocated_ptr, aligned_ptr, offset, sizes[N], strides[N]. The
`llvm.emit_c_interface` attribute generates a C-compatible wrapper
`_mlir__mlir_ciface_compute` that accepts a packed pointer array.

## ONNX Support

```
.onnx file ──prost──▶ ModelProto ──parser.rs──▶ OnnxModel
   │                                              │
   │                              emitter.rs: partition + emit per-kernel MLIR
   │                                              │
   ▼                                              ▼
Model { plan, weights } ◀── compile ◀── ExecutionPlan + kernels
   │
   ├── model.run(&[input])    → Vec<Buffer>  (prefill / non-KV)
   └── model.run_kv(&[input]) → Vec<Buffer>  (KV-cached decode)
```

The emitter walks the ONNX graph per kernel group, emitting linalg ops into
each kernel's MLIR module. Shape-prep ops (Shape, Gather, Concat for shapes)
are constant-folded at emit time.

**Supported ONNX ops:** Add, Sub, Mul, Div, Neg, Relu, Exp, Tanh, Pow, Sqrt, Cast, MatMul, Gemm, Softmax, Clip, Reshape, Flatten, Gather, Slice, Concat, Split, Transpose, Where, Conv, MaxPool, BatchNormalization, GlobalAveragePool, ReduceMean, ReduceSum, ReduceMax, Constant, Shape, ConstantOfShape, Range, Squeeze, Unsqueeze, and more.

**Symbolic dimensions:** Models with dynamic dimensions (like GPT-2's `past_sequence_length`) use `SymDim` expression tracking. Compiled once with dynamic dims, resolved at runtime from actual input shapes.

**Multiple outputs:** `Model::run()` returns `Vec<Buffer>`. GPT-2 produces logits + 24 KV cache tensors.

## Compilation Cache

Compiled `.so` files are cached on disk at `~/.cache/homura/` (or `HOMURA_CACHE_DIR`). The cache key is a hash of:
- Model bytes (any model change invalidates)
- Input shapes (different seq_len = different compilation)
- Compiler fingerprint: homura version, LLVM version, host CPU name + features

On cache hit, compilation is skipped entirely — the `.so` is loaded via dlopen in milliseconds. Power-of-2 bucket padding for sequence lengths limits the number of unique compilations to at most 6 (32, 64, 128, 256, 512, 1024).

`homura clean-cache` removes all cached files.

## Text Generation

Two generator modes for transformer models:

```rust
// Single-model (unified): prefill + decode via one model
let gen = UnifiedKvGenerator::load("tests/fixtures/", 1024, 50256).unwrap();
let text = gen.generate("The meaning of life is", 50);

// Two-model: separate prefill and decode models
let gen = KvGenerator::load("tests/fixtures/", 1024, 50256).unwrap();
let text = gen.generate("The meaning of life is", 50);
```

Uses a byte-level BPE tokenizer (GPT-2 compatible, loads `vocab.json` + `merges.txt`).
Generation: prefill (run) → init_kv_cache → decode loop (run_kv) with greedy argmax sampling.
GPT-2 (124M) runs at ~50 tok/s (~20ms/token) on CPU.

## Architecture Decisions

### AOT compilation, not JIT

The original design used MLIR's ExecutionEngine (JIT). This was replaced with ahead-of-time compilation to native `.so` files because:
- The JIT took ~33s for GPT-2 with no way to cache the result
- `dump_to_object_file` failed silently for models using `memrefCopy`
- AOT produces a standard `.so` that can be cached on disk and loaded instantly

The AOT path: MLIR → `mlirTranslateModuleToLLVMIR` → `LLVMRunPasses("default<O3>")` → `LLVMTargetMachineEmitToFile` → `cc -shared` → dlopen. Uses `llvm-sys` for LLVM C API bindings.

### Deferred tracing, not eager execution

Operations record to a trace instead of executing immediately. This lets the compiler see the entire computation graph before generating code, enabling global optimizations. The same pattern as JAX and `torch.compile`.

### Thread-local trace, not explicit context

The trace is stored in a thread-local variable rather than an explicit context object. `&a + &b` just works without passing a graph builder around.

### TOSA as primary backend, linalg.generic as fallback

TOSA provides native ops for most ML operations with well-tested lowering passes. For ops TOSA doesn't support (float div, integer matmul, gather, batched matmul, cast with I64), we fall back to `linalg.generic`.

### ONNX graph replay through Tensor API

Rather than building a separate ONNX-to-MLIR compiler, the ONNX mapper replays the graph through the existing Tensor API. This reuses shape inference, broadcasting, dtype validation, and the entire compilation pipeline.

## Source Layout

```
src/
├── lib.rs            Public API re-exports
├── dtype.rs          DType enum (F32, F64, I32, I64)
├── shape.rs          Shape wrapper over Vec<u64> with broadcast
├── main.rs           CLI: homura info / run / clean-cache
├── graph_builder.rs  MLIR emission (GraphBuilder, GraphContext)
├── compiler.rs       Transform schedule, emit_object_files, link_shared_lib
├── runtime.rs        Buffer, CompiledGraph, ExecutionPlan, KvCache
├── cache.rs          Disk-based per-kernel compilation cache
├── tokenizer.rs      Byte-level BPE tokenizer (GPT-2)
├── generate.rs       Simple autoregressive generation (full recompute)
├── kv_generate.rs    KV-cached generation (KvGenerator, UnifiedKvGenerator)
└── onnx/
    ├── mod.rs        Model struct (load/run/run_kv), lazy compilation
    ├── proto.rs      Prost-generated protobuf types
    ├── parser.rs     ONNX ModelProto → OnnxModel
    ├── emitter.rs    Per-kernel MLIR emission, partition_nodes, split_kv_concat_groups
    └── sym_shapes.rs Symbolic dimension expression tracking (SymDim)

scripts/
└── download_gpt2.sh  Download GPT-2 ONNX models + tokenizer

tests/fixtures/       MNIST, ResNet-18, GPT-2 model files + tokenizer data
```

## Dependencies

- **melior** — Rust bindings for MLIR's C API. IR construction and pass management. TOSA support via `ods-dialects` feature.
- **mlir-sys** — Low-level FFI to `libMLIR-C.so`. Patched for shared LLVM/MLIR libraries.
- **llvm-sys** — LLVM C API bindings for AOT compilation (target machine, pass builder, object emission).
- **prost** / **prost-build** — Protobuf compilation for ONNX model parsing.
- **protobuf-src** — Vendored protoc compiler.
- **clap** — CLI argument parsing.
- **libc** — dlopen/dlsym for loading compiled `.so` files.
- **serde_json** — Parsing `vocab.json` for the BPE tokenizer.
- **fancy-regex** — Pre-tokenization regex (GPT-2 BPE requires lookahead).
- Requires **LLVM 21** with `libMLIR-C.so` and `libmlir_c_runner_utils.so`.

## Current Limitations

- **CPU only** — no GPU backend
- **KV cache views copy data** — zero-copy borrowed views pending
- **No autograd** — forward pass only
- **F32 only for inference** — no quantization (int8/int4)

## Performance

| Model | Kernels | Compile time | Inference |
|---|---|---|---|
| MNIST | 6 | 97ms | — |
| ResNet-18 | 41 | 467ms | — |
| GPT-2 (124M) | 158 compiled + 24 native | ~650ms | ~50 tok/s decode |

## Roadmap

### Completed

- Per-kernel MLIR compilation with parallel codegen via rayon
- Transform dialect schedule (tile + vectorize) for matmul/conv
- Symbolic dimension tracking (compile once, resolve at runtime)
- KV cache with persistent pre-allocated buffers
- Single-model and two-model generation paths
- GPT-2 at ~50 tok/s on CPU

### Next

- Zero-copy KV cache views (eliminate view copies)
- Packed GEMM layout for better cache utilization
- AVX-512 micro-kernel tuning
- Operator fusion (matmul + bias + relu as one kernel)
- GPU backend (CUDA via `gpu-to-nvvm`)
- Support for modern models (SmolLM2, Llama-family)
