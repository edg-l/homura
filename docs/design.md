# Homura Design Document

Homura is a Rust ML inference framework that traces tensor operations into a computation graph, compiles them through MLIR, and JIT-executes the result. Think of it as a mini JAX or tinygrad in Rust.

## The Big Picture

```
User code             Trace              MLIR                  Machine code
─────────────────     ──────────────     ───────────────────   ──────────────
let a = Tensor(...)   Op::Input(a)       bufferization.to_     fadd loop over
let b = Tensor(...)   Op::Input(b)         tensor %arg0        raw memory
let c = &a + &b       Op::Add(a,b)       linalg.generic {      (JIT compiled)
                                           arith.addf
                                         }
```

Nothing runs until you say so. You write math, Homura writes it down, then compiles and runs it all at once.

## How It Works: Step by Step

### 1. Tracing — Recording the Recipe

When you write tensor operations, nothing computes. Homura records each operation into a flat list called a **trace**.

```rust
begin_trace();                          // open the notebook
let a = Tensor::new(&[4], DType::F32);  // write down: "input A, 4 floats"
let b = Tensor::new(&[4], DType::F32);  // write down: "input B, 4 floats"
let c = &a + &b;                        // write down: "add A + B"
let trace = take_trace();               // close the notebook
```

The trace after this looks like:

```
index 0: Input { shape: [4], dtype: F32, arg_index: 0 }
index 1: Input { shape: [4], dtype: F32, arg_index: 1 }
index 2: Add   { lhs: 0, rhs: 1, shape: [4], dtype: F32 }
```

Each entry references earlier entries by index (`NodeId`). A `Tensor` is not actual data — it's just a handle holding a `NodeId`, a `Shape`, and a `DType`.

The trace lives in a **thread-local** variable. This means:
- No context object to pass around — operations implicitly record to the active trace
- Each thread gets its own isolated trace
- Calling `begin_trace()` twice without `take_trace()` panics (one trace at a time)

### 2. Compilation — Turning the Recipe into MLIR

`Compiler::compile(trace, outputs)` walks the trace and emits MLIR intermediate representation (IR). The compiler uses **linalg.generic** operations on tensor types, which is MLIR's way of expressing "do this operation element-wise."

For `c = a + b` with shape `[4]` and dtype `f32`, the generated IR is:

```mlir
func.func @compute(%arg0: memref<4xf32>,   // input a
                    %arg1: memref<4xf32>,   // input b
                    %arg2: memref<4xf32>)   // output c
    attributes { llvm.emit_c_interface } {

  // Convert memref args to tensors (boundary conversion)
  %t0 = bufferization.to_tensor %arg0 restrict : memref<4xf32> to tensor<4xf32>
  %t1 = bufferization.to_tensor %arg1 restrict : memref<4xf32> to tensor<4xf32>

  // Create an empty output tensor
  %init = tensor.empty() : tensor<4xf32>

  // The actual computation: element-wise add
  %result = linalg.generic
      {indexing_maps = [affine_map<(d0) -> (d0)>,    // how to index input 1
                        affine_map<(d0) -> (d0)>,    // how to index input 2
                        affine_map<(d0) -> (d0)>],   // how to index output
       iterator_types = ["parallel"]}                 // iterations are independent
      ins(%t0, %t1 : tensor<4xf32>, tensor<4xf32>)
      outs(%init : tensor<4xf32>) {
    ^bb0(%a_elem: f32, %b_elem: f32, %out_elem: f32):
      %sum = arith.addf %a_elem, %b_elem : f32
      linalg.yield %sum : f32
  } -> tensor<4xf32>

  // Convert result back to memref and copy to output argument
  %out_memref = bufferization.to_buffer %result : tensor<4xf32> to memref<4xf32>
  memref.copy %out_memref, %arg2 : memref<4xf32> to memref<4xf32>
  return
}
```

Key things to notice:

- **The function takes memref arguments** (pointers to memory), not tensors. This is the JIT calling convention.
- **Internally everything is tensors.** `bufferization.to_tensor` at the boundary converts memrefs to tensors; `bufferization.to_buffer` converts back.
- **`linalg.generic` describes what, not how.** It says "apply this function element-wise" without specifying loops. MLIR decides how to execute it.
- **`iterator_types = ["parallel"]`** tells MLIR these iterations are independent — crucial for future GPU support.

### 3. Lowering — From High-Level IR to Machine Code

The MLIR IR goes through a 9-pass pipeline that progressively lowers abstractions:

```
linalg.generic on tensors
        │
        ▼
┌─────────────────────────────────────────────┐
│ one-shot-bufferize                          │
│   Converts tensor ops to memref ops.        │
│   Allocates memory for intermediates.       │
│   The big pass — eliminates tensor          │
│   semantics entirely.                       │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│ convert-linalg-to-loops                     │
│   Turns linalg.generic into scf.for loops   │
│   with explicit load/store operations.      │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│ convert-scf-to-cf                           │
│   Structured loops (for/if) become          │
│   unstructured branches (br/cond_br).       │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│ finalize-memref-to-llvm                     │
│ convert-arith-to-llvm                       │
│ convert-index-to-llvm                       │
│ convert-cf-to-llvm                          │
│ convert-func-to-llvm                        │
│   Five passes that convert everything       │
│   remaining to LLVM dialect.                │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│ reconcile-unrealized-casts                  │
│   Final cleanup: resolves any remaining     │
│   type casts left over from lowering.       │
└─────────────────────────────────────────────┘
        │
        ▼
    LLVM IR → JIT compiled machine code
```

The exact pipeline string passed to MLIR:
```
builtin.module(
    one-shot-bufferize{
        function-boundary-type-conversion=identity-layout-map
        unknown-type-conversion=identity-layout-map},
    convert-linalg-to-loops,
    convert-scf-to-cf,
    finalize-memref-to-llvm,
    convert-arith-to-llvm,
    convert-index-to-llvm,
    convert-cf-to-llvm,
    convert-func-to-llvm,
    reconcile-unrealized-casts)
```

All 9 passes are required. Removing any one causes verification or lowering failures.

### 4. Execution — Running the Machine Code

`CompiledGraph::run(inputs)` marshals Rust data into the format the JIT-compiled function expects, calls it, and extracts the result.

The JIT ABI uses **memref descriptors** — C structs that describe a region of memory:

```rust
#[repr(C)]
struct MemRefDescriptor1D<T> {
    allocated: *mut T,    // base allocation pointer
    aligned:   *mut T,    // aligned data pointer (same as allocated for simple cases)
    offset:    i64,       // 0
    sizes:     [i64; 1],  // [num_elements]
    strides:   [i64; 1],  // [1]
}
```

The `llvm.emit_c_interface` attribute on the function causes MLIR to generate a C-compatible wrapper (`_mlir_ciface_compute`) that accepts pointers to these descriptors. The `invoke_packed` mechanism requires **double indirection**: each argument slot holds a pointer to a pointer to a descriptor.

```
args[0] → &ptr_to_desc_a → MemRefDescriptor1D { data of a }
args[1] → &ptr_to_desc_b → MemRefDescriptor1D { data of b }
args[2] → &ptr_to_desc_c → MemRefDescriptor1D { output buffer }
```

## How Chained Operations Work

For `d = (a + b) + c`, the trace records four ops and the compiler emits two `linalg.generic` operations:

```
Trace:
  0: Input(a)
  1: Input(b)
  2: Add(0, 1)     ← a + b
  3: Input(c)
  4: Add(2, 3)     ← (a + b) + c
```

The compiler walks the trace linearly, maintaining a `NodeId → Value` map. When it hits `Add(2, 3)`, it looks up the tensor value for node 2 (the result of the first add) and node 3 (input c), then emits a second `linalg.generic` that consumes the first one's output.

After bufferization and lowering, this becomes two loops with an intermediate buffer:

```
loop 1: for i in 0..4:  tmp[i] = a[i] + b[i]
loop 2: for i in 0..4:  out[i] = tmp[i] + c[i]
```

MLIR's optimization passes could potentially fuse these into a single loop, but that's not required for correctness.

## Why linalg.generic?

The previous version of Homura emitted explicit `scf.for` loops with `memref.load/store` operations. This worked for CPU but was a dead end:

```
Old approach (manual loops):         New approach (linalg.generic):

  scf.for %i = 0 to N {               linalg.generic {
    %a = memref.load %in[%i]             arith.addf
    %b = memref.load %in[%i]           }
    %s = arith.addf %a, %b
    memref.store %s, %out[%i]          Can lower to:
  }                                      → scf.for loops (CPU)
                                         → gpu.launch (GPU)
  Can only lower to:                     → tiled loops (vectorized)
    → LLVM (CPU only)                   → distributed execution
```

`linalg.generic` describes **what** to compute without specifying **how**. The `iterator_types = ["parallel"]` annotation tells MLIR the iterations are independent, which enables:

- **GPU lowering**: Map parallel iterations to GPU threads
- **Vectorization**: Process multiple elements per instruction
- **Tiling**: Break large operations into cache-friendly chunks
- **Fusion**: Combine adjacent operations into single loops
- **N-D tensors**: Just add more dimensions to the affine maps

## Architecture Decisions

### Deferred tracing, not eager execution

Operations record to a trace instead of executing immediately. This lets the compiler see the entire computation graph before generating code, enabling global optimizations. The trade-off is a two-phase API (`begin_trace/take_trace` then `compile/run`), but it's the same pattern JAX and `torch.compile` use.

### Thread-local trace, not explicit context

The trace is stored in a thread-local variable rather than an explicit context object. This keeps the API clean — `&a + &b` just works without passing a graph builder around. Each thread gets its own isolated trace.

### Flat Vec\<Op\> with NodeId indices, not a recursive tree

Operations are stored in a flat vector. Each `Op` references its inputs by `NodeId` (a `u32` index). This is more memory-efficient than `Box<Op>` trees and enforces a DAG structure (operations can only reference earlier operations). The linear layout also makes the compiler's trace walk simple.

### Memref function boundary, tensor internals

The generated function takes `memref` arguments (matching the JIT's C ABI) but uses `tensor` types internally. `bufferization.to_tensor` and `bufferization.to_buffer` convert at the boundary. This keeps the runtime ABI unchanged while letting the compiler work with high-level tensor abstractions.

### One linalg.generic per operation

Each op becomes its own `linalg.generic`. For `a + b + c`, that's two separate operations. MLIR's fusion passes can merge them later, but the compiler doesn't try to be clever — it emits the simplest correct IR and lets the MLIR pass pipeline optimize.

Binary ops (Add, Sub, Mul, Div) use 3 affine maps, 2 `ins` operands, and a 3-arg body block. Unary ops (Neg, Relu) use 2 affine maps, 1 `ins` operand, and a 2-arg body block. The only difference between ops within each category is the arith operation inside the body.

## Source Layout

```
src/
├── lib.rs          Public API re-exports
├── dtype.rs        DType enum (F32, F64, I32, I64) with MLIR type conversion
├── shape.rs        Shape wrapper over Vec<u64>
├── op.rs           NodeId (u32 index) and Op enum (Input, Add, Sub, Mul, Div, Neg, Relu)
├── trace.rs        Thread-local Trace context, begin_trace/take_trace/record
├── tensor.rs       Tensor handle with operator overloads (Add, Sub, Mul, Div, Neg) and .relu()
├── compiler.rs     Trace → MLIR IR emission, pass pipeline, ExecutionEngine
└── runtime.rs      MemRefDescriptor1D, CompiledGraph::run() with JIT marshalling

examples/
├── add.rs          a + b demo (simple and chained)
└── ops.rs          all ops demo (sub, mul, div, neg, relu)
```

## Dependencies

- **melior** — Rust bindings for MLIR's C API. Used for all IR construction, pass management, and JIT execution.
- **mlir-sys** — Low-level FFI bindings to `libMLIR-C.so`. Patched to support shared LLVM/MLIR libraries.
- Requires **LLVM 21** with MLIR C API support.

## Current Limitations

- **Rank-1 tensors only** — all shapes must be 1D
- **Element-wise ops only** — no matmul, convolution, or reductions
- **CPU JIT only** — no GPU backend
- **Single output** — `compile()` accepts only one output node
- **F32 execution only** — `run()` only handles `f32` data (other dtypes compile but can't execute)
- **Integer division by zero** — `arith.divsi` lowers to x86 `idiv`, which raises SIGFPE. Callers must ensure non-zero divisors.
- **No autograd** — forward pass only, no gradient computation

## Roadmap

### Milestone 1: Run a hand-coded MLP

The minimum to express `relu(x @ W + b)` stacked into layers.

1. **N-D tensors** — add dimensions to affine maps, generalize `MemRefDescriptor` beyond 1D
2. **Matmul** — `linalg.matmul` or `linalg.generic` with `"reduction"` iterator type
3. **Broadcast** — bias add (`[batch, features] + [features]`)
4. **Softmax, tanh** — needed for output layers and activations beyond relu
5. **`Tensor::eval()` sugar** — one-call API wrapping trace/compile/run

### Milestone 2: Run a real pre-trained model

The step from "toy" to "useful." Load a model someone else trained and run inference.

6. **Model loading** — ONNX import or safetensors weight loading. This is what makes Homura usable in practice — nobody hand-codes weight matrices.
7. **More ops** — Conv2d, layer norm, transpose, reshape, concat. A transformer needs ~15-20 ops; a CNN needs Conv2d + pooling at minimum.
8. **Multiple outputs** — extend `compile()` to accept multiple output nodes
9. **GPU backend** — swap `convert-linalg-to-loops` for GPU tiling passes (`linalg` → `gpu.launch`)

### Milestone 3: Production-grade

Competitive with ONNX Runtime / candle for real workloads.

10. **Graph optimizations** — op fusion (merge adjacent `linalg.generic`), constant folding, dead code elimination
11. **Dynamic shapes** — support variable batch sizes without recompilation
12. **Training** — reverse-mode automatic differentiation (walk trace backwards, emit gradient ops)
13. **Memory planning** — buffer reuse, allocation hoisting, minimize peak memory
14. **Multi-device** — distribute across multiple GPUs
