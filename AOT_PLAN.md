# AOT Compilation Pipeline Plan

## Overview

Replace the MLIR ExecutionEngine (JIT) with ahead-of-time compilation: translate MLIR to LLVM IR, run LLVM optimization passes, emit a native object file, link to a shared library, and dlopen at runtime. This eliminates the ~33s LLVM JIT bottleneck for large models and fixes the `dump_to_object_file` silent failure for models referencing `memrefCopy`.

## Requirements

- **R1**: Remove `melior::ExecutionEngine` from all production code paths
- **R2**: After MLIR passes, translate module to LLVM IR via `mlirTranslateModuleToLLVMIR`
- **R3**: Run LLVM optimization passes via `LLVMRunPasses("default<O3>", ...)`
- **R4**: Emit `.o` file via `LLVMTargetMachineEmitToFile`
- **R5**: Link `.o` to `.so` with `cc -shared -lmlir_c_runner_utils`
- **R6**: dlopen `.so` for execution (existing `from_cached_lib` mechanism)
- **R7**: Cache `.so` on disk, skip compilation on cache hit
- **R8**: All 365 tests pass

## Dependencies

- **llvm-sys**: Use the `llvm-sys` crate for all LLVM C API bindings (target machine, pass builder, codegen, module disposal). Provides type-safe enums (`LLVMRelocPIC`, `LLVMCodeGenLevelAggressive`, etc.) and opaque pointer types. Already proven to work in [edg-l/irvm](https://github.com/edg-l/irvm/blob/master/irvm-lower/src/llvm.rs).
- **mlir-sys**: Already a dependency. Used only for the `MlirOperation` type needed by `mlirTranslateModuleToLLVMIR`.
- One `extern "C"` declaration for `mlirTranslateModuleToLLVMIR` (MLIR-specific, not in llvm-sys).

### Compilation flow (after change)

```
Trace -> build_module() -> MLIR module (TOSA+linalg)
  -> MLIR pass pipeline (same as today, lowers to LLVM dialect)
  -> mlirTranslateModuleToLLVMIR()  -> LLVMModuleRef
  -> LLVMRunPasses("default<O3>")   -> optimized LLVM IR
  -> LLVMTargetMachineEmitToFile()  -> /tmp/{key}.o
  -> cc -shared {key}.o -lmlir_c_runner_utils -Wl,-rpath,{dir} -o /tmp/{key}.so
  -> cache.store(key, /tmp/{key}.so, meta)
  -> CompiledGraph::from_cached_lib({key}.so)
```

On cache hit, skip everything after the first arrow — just `from_cached_lib`.

## Implementation Plan

### Phase 1: Add llvm-sys dependency + MLIR FFI (Complexity: Low)

- [ ] **1.1**: Add `llvm-sys = "210"` to Cargo.toml (matching LLVM 21). Verify it compiles alongside mlir-sys without conflicts.

- [ ] **1.2**: Create `src/llvm_ffi.rs` with one `extern "C"` declaration:
  ```rust
  extern "C" {
      pub fn mlirTranslateModuleToLLVMIR(
          module: mlir_sys::MlirOperation,
          context: llvm_sys::prelude::LLVMContextRef,
      ) -> llvm_sys::prelude::LLVMModuleRef;
  }
  ```
  All other LLVM functions come from `llvm_sys::*` directly.

- [ ] **1.3**: Add `mod llvm_ffi;` to `src/lib.rs` (private module)

- [ ] **1.4**: Write a smoke test that initializes X86 target, gets default triple, creates a target machine, and asserts the machine ref is non-null. Validates llvm-sys links correctly.

### Phase 2: AOT emit function (Complexity: Medium)

- [ ] **2.1**: Add `emit_object_file(module: &Module, output_path: &Path) -> Result<(), CompileError>` in `src/compiler.rs` that:
  1. Creates LLVM context via `llvm_sys::core::LLVMContextCreate()`
  2. Translates MLIR to LLVM IR: `mlirTranslateModuleToLLVMIR(module.as_operation().to_raw(), llvm_ctx)`
  3. Checks for null return (translation failure) — dispose context on error via drop guard
  4. Initializes X86 target (idempotent, use `OnceLock`):
     - `LLVM_InitializeAllTargets()`, `LLVM_InitializeAllTargetInfos()`, `LLVM_InitializeAllTargetMCs()`, `LLVM_InitializeAllAsmPrinters()`
  5. Gets host CPU name via `LLVMGetHostCPUName()` and features via `LLVMGetHostCPUFeatures()` — enables AVX2/SSE vectorization
  6. Gets target triple via `LLVMGetDefaultTargetTriple()`
  7. Creates target via `LLVMGetTargetFromTriple()`
  8. Creates target machine: `LLVMCreateTargetMachine(target, triple, cpu, features, Aggressive, PIC, Default)`
  9. Runs LLVM optimization passes: `LLVMRunPasses(llvm_module, "default<O3>", machine, opts)` with `LLVMCreatePassBuilderOptions()`
  10. Optionally verifies module: `LLVMVerifyModule()` (debug builds only)
  11. Emits object file: `LLVMTargetMachineEmitToFile(machine, llvm_module, path, ObjectFile, &error)`
  12. Cleans up in correct order: dispose pass opts, target machine, LLVM module, LLVM context, any message strings
  13. Returns `Err(CompileError::ObjectEmit(msg))` on any failure

  **Reference implementation**: [irvm-lower/src/llvm.rs `compile_object`](https://github.com/edg-l/irvm/blob/master/irvm-lower/src/llvm.rs) — same pattern with llvm-sys.

- [ ] **2.2**: Add `CompileError::ObjectEmit(String)` and `CompileError::Link(String)` variants

- [ ] **2.3**: Write a test that traces `a + b` (4xf32), runs MLIR passes, calls `emit_object_file`, and asserts the `.o` file exists and is non-empty

### Phase 3: Link step (Complexity: Low)

- [ ] **3.1**: Add `link_shared_lib(obj_path: &Path, so_path: &Path) -> Result<(), CompileError>` that:
  1. Determines runner utils directory from `MLIR_RUNNER_UTILS_PATH` env var (dirname) or default `/usr/lib/llvm/21/lib64/`
  2. Links by full path to runner utils (avoids versioned .so name issues):
     ```
     cc -shared -fPIC -o {so_path} {obj_path} {runner_utils_path} -lm -Wl,-rpath,{runner_utils_dir}
     ```
  3. On non-zero exit, returns `Err(CompileError::Link(stderr))`

- [ ] **3.2**: Write a test: trace `a + b`, emit `.o`, link to `.so`, dlopen and run, verify result

### Phase 4: Replace JIT in Compiler::compile (Complexity: Medium)

- [ ] **4.1**: Rewrite `Compiler::compile` core path:
  ```
  // After pass_manager.run(&mut module):
  let tmp_obj = tmp_dir.join(format!("{}.o", random_suffix));
  let tmp_so = tmp_dir.join(format!("{}.so", random_suffix));

  emit_object_file(&module, &tmp_obj)?;
  link_shared_lib(&tmp_obj, &tmp_so)?;
  fs::remove_file(&tmp_obj).ok();  // clean up .o

  let graph = CompiledGraph::from_cached_lib(&tmp_so, num_inputs, output_descs)?;
  fs::remove_file(&tmp_so).ok();  // safe: Linux keeps inode alive while dlopen'd

  if let Some(key) = cache_key {
      // Re-emit to cache location (or copy before deleting)
      cache.store(key, &tmp_so, &meta);
  }
  ```

  **Important**: Always use random temp names (PID + random suffix) even for cached compilations, to avoid concurrent write collisions between processes.

- [ ] **4.2**: Remove `ensure_runner_utils_loaded()` — no longer needed since runner utils is linked into the .so with rpath.

- [ ] **4.3**: Remove `store_native_cache` function — cache store is now inline.

- [ ] **4.4**: For cache hits: `from_cached_lib` loads the cached `.so` directly. No compilation happens.

### Phase 5: Remove JIT from runtime (Complexity: Low)

- [ ] **5.1**: Remove `Backend::Jit(ExecutionEngine)` variant. Flatten `CompiledGraph` to hold `_lib: *mut c_void` + `func: extern "C" fn(*mut *mut ())` directly (no enum needed).

- [ ] **5.2**: Remove `CompiledGraph::new(engine, ...)` constructor. Rename `from_cached_lib` to `load` — it's now the only constructor.

- [ ] **5.3**: Remove `dump_to_object_file()` and `engine()` methods.

- [ ] **5.4**: Remove `use melior::ExecutionEngine;` from runtime.rs.

- [ ] **5.5**: Remove `ensure_runner_utils_loaded()` from compiler.rs.

### Phase 6: Update tests (Complexity: Medium)

- [ ] **6.1**: Update `spike_tosa_add_pipeline` and `spike_tosa_mixed_pipeline` — replace direct `ExecutionEngine::new` + `invoke_packed` with the AOT path via `Compiler::compile`.

- [ ] **6.2**: Update `native_so_cache_roundtrip` — simplify since AOT is now the default path.

- [ ] **6.3**: Remove `investigate_object_symbols` test (debugging spike, no longer relevant).

- [ ] **6.4**: Remove or update `jit_run_f32` test helper — rename to `aot_run_f32` or use `Compiler::compile` directly.

- [ ] **6.5**: Run full test suite and fix any remaining references to `ExecutionEngine`, `Backend::Jit`, `invoke_packed`.

### Phase 7: Cleanup (Complexity: Low)

- [ ] **7.1**: Verify melior is still needed (yes — MLIR context, module building, pass pipeline). Only `ExecutionEngine` usage is removed.

- [ ] **7.2**: Update CLAUDE.md:
  - "JIT ABI" section → "AOT ABI" / "Native library ABI"
  - Document AOT pipeline steps
  - Note `libmlir_c_runner_utils.so` is now a link-time dependency
  - Update pass pipeline docs to include LLVM optimization passes

- [ ] **7.3**: Run `cargo test` — all 365+ tests pass.

## Key Technical Details

### llvm-sys version

`llvm-sys = "210"` matches LLVM 21. The crate auto-detects LLVM installation via `llvm-config`. On the target system (Gentoo), `llvm-config-21` should be in PATH or set via `LLVM_SYS_210_PREFIX`.

### Host CPU optimization

Using `LLVMGetHostCPUName()` + `LLVMGetHostCPUFeatures()` (instead of `"generic"`) enables LLVM to emit AVX2/SSE4.2/etc. instructions. This is critical for matmul performance — the linalg-lowered affine loops benefit significantly from vectorization.

### LLVM optimization pipeline

`LLVMRunPasses(module, "default<O3>", machine, opts)` runs the full O3 pipeline including:
- Loop vectorization (auto-vectorize matmul inner loops)
- SLP vectorization (combine scalar ops into SIMD)
- Loop unrolling
- Function inlining
- Dead code elimination
- etc.

This replaces the JIT's internal `-O3` which did the same thing. `LLVMPassBuilderOptions` can optionally toggle specific optimizations (loop vectorization, SLP, unrolling) but defaults are good.

### Error handling and resource cleanup

All LLVM C resources must be disposed in the correct order:
1. `LLVMDisposePassBuilderOptions(opts)`
2. `LLVMDisposeTargetMachine(machine)`
3. `LLVMDisposeModule(llvm_module)`
4. `LLVMContextDispose(llvm_ctx)`
5. `LLVMDisposeMessage(msg)` for any error/triple strings

On error paths (e.g., `mlirTranslateModuleToLLVMIR` returns null), earlier resources must still be cleaned up. Use Rust drop guards or manual cleanup in each error branch.

### Runner utils linking

Link runner utils by full path (not `-l` name) to handle versioned/symlinked .so names:
```
cc -shared -fPIC -o out.so in.o /usr/lib/llvm/21/lib64/libmlir_c_runner_utils.so -lm -Wl,-rpath,/usr/lib/llvm/21/lib64/
```

The `-Wl,-rpath` bakes the directory into the .so's `DT_RUNPATH`, so the dynamic linker finds runner utils at dlopen time without `LD_LIBRARY_PATH`.

### Symbol name

The `llvm.emit_c_interface` attribute on the `compute` function generates `_mlir__mlir_ciface_compute` in both JIT and AOT paths. This is determined at MLIR IR emission time (in `build_module`), not by the compilation backend. The `from_cached_lib` / dlsym code already looks up this exact symbol.

### Temp file strategy

Always use PID + random suffix for temp files, even for cached compilations:
```rust
let suffix = format!("{}_{:x}", std::process::id(), rand_u64());
let tmp_obj = tmp_dir.join(format!("homura_{suffix}.o"));
```
This prevents concurrent processes (e.g., parallel test runners, multiple inference workers) from corrupting each other's temp files. The cache key is only used for the final stored path.

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| `llvm-sys` version conflict with mlir-sys | Both link against same LLVM 21; verified in irvm |
| `cc` not on PATH | Clear `CompileError::Link` message suggesting gcc/clang |
| Runner utils not found at link/dlopen | Full path linking + rpath + `MLIR_RUNNER_UTILS_PATH` env var |
| Tiny model regression (~100-300ms vs ~5ms JIT) | Acceptable; cache makes second run instant |
| Concurrent temp file collisions | Random suffix per compilation |
| LLVM context leak on error | Drop guards / explicit cleanup on every error path |
| `LLVMRelocPIC` wrong value | Using llvm-sys enum (`LLVMRelocMode::LLVMRelocPIC`) — correct by construction |
| Cache not portable across machines | Cache keyed per model hash, not designed for portability |
