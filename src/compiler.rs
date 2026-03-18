use melior::{
    Context,
    ir::{
        Attribute, Block, Identifier, Location, Module, Region, RegionLike,
        attribute::{ArrayAttribute, StringAttribute, TypeAttribute},
        block::BlockLike,
        operation::OperationBuilder,
        r#type::FunctionType,
    },
};

/// Error type for compilation failures.
#[derive(Debug)]
pub enum CompileError {
    /// MLIR module verification failed.
    Verification,
    /// A lowering pass failed.
    Pass(melior::Error),
    /// No output nodes specified.
    NoOutputs,
    /// Failed to parse an MLIR attribute string.
    AttributeParse(String),
    /// Failed to emit an object file from LLVM IR.
    ObjectEmit(String),
    /// Failed to link the object file into a shared library.
    Link(String),
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Verification => write!(f, "MLIR module verification failed"),
            Self::Pass(e) => write!(f, "lowering pass failed: {e}"),
            Self::NoOutputs => write!(f, "no output nodes specified"),
            Self::AttributeParse(s) => write!(f, "failed to parse MLIR attribute: {s}"),
            Self::ObjectEmit(s) => write!(f, "object emit failed: {s}"),
            Self::Link(s) => write!(f, "link failed: {s}"),
        }
    }
}

impl std::error::Error for CompileError {}

/// Translate a lowered MLIR module to native code and emit an object file.
///
/// Walk every instruction in every function of the LLVM module and set
/// the `AllowContract` fast-math flag on all float operations.
/// This lets LLVM fuse `fmul + fadd → fma` without enabling other
/// fast-math transforms. Safe for inference (FMA has higher precision).
unsafe fn set_contract_fastmath_flags(module: llvm_sys::prelude::LLVMModuleRef) {
    use llvm_sys::core::*;
    const ALLOW_CONTRACT: u32 = 1 << 5; // LLVMFastMathAllowContract

    unsafe {
        let mut func = LLVMGetFirstFunction(module);
        while !func.is_null() {
            let mut bb = LLVMGetFirstBasicBlock(func);
            while !bb.is_null() {
                let mut inst = LLVMGetFirstInstruction(bb);
                while !inst.is_null() {
                    if LLVMCanValueUseFastMathFlags(inst) != 0 {
                        let existing = LLVMGetFastMathFlags(inst);
                        LLVMSetFastMathFlags(inst, existing | ALLOW_CONTRACT);
                    }
                    inst = LLVMGetNextInstruction(inst);
                }
                bb = LLVMGetNextBasicBlock(bb);
            }
            func = LLVMGetNextFunction(func);
        }
    }
}

/// Initialises LLVM targets on first call (via `OnceLock`), translates the
/// module to LLVM IR, runs the O3 optimisation pipeline, and writes a PIC
/// object file to `output_path`.
/// Run O3 optimisation and emit an object file for a single LLVM module.
///
/// Takes ownership: disposes the module and context on completion.
unsafe fn optimise_and_emit(
    llvm_module: llvm_sys::prelude::LLVMModuleRef,
    llvm_ctx: llvm_sys::prelude::LLVMContextRef,
    output_path: &std::path::Path,
    opt_level: &str,
) -> Result<(), CompileError> {
    use llvm_sys::core::*;
    use llvm_sys::error::LLVMGetErrorMessage;
    use llvm_sys::target_machine::*;
    use llvm_sys::transforms::pass_builder::*;
    use std::ffi::{CStr, CString};

    unsafe {
        let cpu = LLVMGetHostCPUName();
        let features = LLVMGetHostCPUFeatures();
        let triple = LLVMGetDefaultTargetTriple();
        let mut target = std::ptr::null_mut();
        let mut error_msg = std::ptr::null_mut();
        if LLVMGetTargetFromTriple(triple, &mut target, &mut error_msg) != 0 {
            let msg = CStr::from_ptr(error_msg).to_string_lossy().into_owned();
            LLVMDisposeMessage(error_msg);
            LLVMDisposeMessage(triple);
            LLVMDisposeMessage(cpu);
            LLVMDisposeMessage(features);
            LLVMDisposeModule(llvm_module);
            LLVMContextDispose(llvm_ctx);
            return Err(CompileError::ObjectEmit(format!(
                "LLVMGetTargetFromTriple: {msg}"
            )));
        }

        let codegen_level = match opt_level {
            "O0" => LLVMCodeGenOptLevel::LLVMCodeGenLevelNone,
            "O1" => LLVMCodeGenOptLevel::LLVMCodeGenLevelLess,
            "O2" => LLVMCodeGenOptLevel::LLVMCodeGenLevelDefault,
            _ => LLVMCodeGenOptLevel::LLVMCodeGenLevelAggressive,
        };
        let machine = LLVMCreateTargetMachine(
            target,
            triple,
            cpu,
            features,
            codegen_level,
            LLVMRelocMode::LLVMRelocPIC,
            LLVMCodeModel::LLVMCodeModelDefault,
        );

        // Set AllowContract fast-math flag on all float instructions.
        // This enables FMA fusion (fmul + fadd → fma) without other fast-math transforms.
        // The flag is safe for inference — FMA is more precise (single rounding).
        set_contract_fastmath_flags(llvm_module);

        let opts = LLVMCreatePassBuilderOptions();
        let passes = CString::new(format!("default<{opt_level}>")).unwrap();
        let pass_error = LLVMRunPasses(llvm_module, passes.as_ptr(), machine, opts);
        if !pass_error.is_null() {
            let msg_ptr = LLVMGetErrorMessage(pass_error);
            let msg = CStr::from_ptr(msg_ptr).to_string_lossy().into_owned();
            LLVMDisposeMessage(msg_ptr);
            LLVMDisposePassBuilderOptions(opts);
            LLVMDisposeTargetMachine(machine);
            LLVMDisposeMessage(triple);
            LLVMDisposeMessage(cpu);
            LLVMDisposeMessage(features);
            LLVMDisposeModule(llvm_module);
            LLVMContextDispose(llvm_ctx);
            return Err(CompileError::ObjectEmit(format!("LLVMRunPasses: {msg}")));
        }
        LLVMDisposePassBuilderOptions(opts);

        let filename = CString::new(output_path.to_str().unwrap()).unwrap();
        let mut emit_error = std::ptr::null_mut();
        let failed = LLVMTargetMachineEmitToFile(
            machine,
            llvm_module,
            filename.as_ptr().cast_mut(),
            LLVMCodeGenFileType::LLVMObjectFile,
            &mut emit_error,
        );

        LLVMDisposeTargetMachine(machine);
        LLVMDisposeMessage(triple);
        LLVMDisposeMessage(cpu);
        LLVMDisposeMessage(features);
        LLVMDisposeModule(llvm_module);
        LLVMContextDispose(llvm_ctx);

        if failed != 0 {
            let msg = if emit_error.is_null() {
                "unknown error".to_string()
            } else {
                let s = CStr::from_ptr(emit_error).to_string_lossy().into_owned();
                LLVMDisposeMessage(emit_error);
                s
            };
            return Err(CompileError::ObjectEmit(format!(
                "LLVMTargetMachineEmitToFile: {msg}"
            )));
        }

        Ok(())
    }
}

/// Translate MLIR → LLVM IR, then emit object file(s).
///
/// When `HOMURA_CODEGEN_THREADS` is set to "1" or the module is small, emits a
/// single object file (monolithic path). Otherwise, splits the LLVM module into
/// partitions via `LLVMSplitModule`, serialises each to bitcode, and compiles
/// them in parallel — each in its own `LLVMContext` on a separate thread.
///
/// Returns the list of `.o` paths produced.
fn emit_object_files(
    module: &melior::ir::Module,
    output_dir: &std::path::Path,
    base_name: &str,
    label: &str,
) -> Result<Vec<std::path::PathBuf>, CompileError> {
    use crate::llvm_ffi::{LLVMSplitModule, mlirTranslateModuleToLLVMIR};
    use llvm_sys::bit_reader::LLVMParseBitcodeInContext2;
    use llvm_sys::bit_writer::LLVMWriteBitcodeToMemoryBuffer;
    use llvm_sys::core::*;
    use llvm_sys::target::*;
    use std::ffi::CString;
    use std::sync::OnceLock;

    unsafe {
        // Initialise all targets once per process.
        static INIT: OnceLock<()> = OnceLock::new();
        INIT.get_or_init(|| {
            LLVM_InitializeAllTargets();
            LLVM_InitializeAllTargetInfos();
            LLVM_InitializeAllTargetMCs();
            LLVM_InitializeAllAsmPrinters();
        });

        // Create LLVM context and translate MLIR → LLVM IR.
        let translate_start = std::time::Instant::now();
        let llvm_ctx = LLVMContextCreate();
        let llvm_module = mlirTranslateModuleToLLVMIR(module.as_operation().to_raw(), llvm_ctx);
        eprintln!(
            "[{:>8.2}s] [{label}] MLIR→LLVM: {}ms",
            crate::log_ts(),
            translate_start.elapsed().as_millis()
        );

        if !llvm_module.is_null() {
            let dl = CString::new(
                "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128",
            )
            .unwrap();
            LLVMSetDataLayout(llvm_module, dl.as_ptr());
        }
        if llvm_module.is_null() {
            LLVMContextDispose(llvm_ctx);
            return Err(CompileError::ObjectEmit(
                "mlirTranslateModuleToLLVMIR returned null".into(),
            ));
        }

        if let Err(e) = emit_packed_wrapper(llvm_module, llvm_ctx) {
            LLVMDisposeModule(llvm_module);
            LLVMContextDispose(llvm_ctx);
            return Err(CompileError::ObjectEmit(format!(
                "emit_packed_wrapper: {e}"
            )));
        }

        // Determine thread count: env override or available parallelism.
        let n_threads: usize = std::env::var("HOMURA_CODEGEN_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
            });

        // Count defined (non-declaration) functions in the module.
        let mut n_funcs = 0usize;
        let mut f = LLVMGetFirstFunction(llvm_module);
        while !f.is_null() {
            if !LLVMGetFirstBasicBlock(f).is_null() {
                n_funcs += 1;
            }
            f = LLVMGetNextFunction(f);
        }

        // Monolithic path: few functions or parallelism disabled.
        // Per-kernel modules typically have 1 real function + 2 dead transform
        // schedule stubs — splitting those is counterproductive.
        if n_threads <= 1 || n_funcs <= 4 {
            let bc_size = {
                let mb = LLVMWriteBitcodeToMemoryBuffer(llvm_module);
                let sz = LLVMGetBufferSize(mb);
                LLVMDisposeMemoryBuffer(mb);
                sz
            };
            eprintln!(
                "[{:>8.2}s] [{label}] monolithic: {n_funcs} func, {}KB",
                crate::log_ts(),
                bc_size / 1024
            );
            let obj_path = output_dir.join(format!("{base_name}.o"));
            optimise_and_emit(llvm_module, llvm_ctx, &obj_path, "O2")?;
            return Ok(vec![obj_path]);
        }

        // Don't create more partitions than functions.
        let n_parts = n_threads.min(n_funcs);
        eprintln!(
            "[{:>8.2}s] [{label}] split: {n_funcs} funcs → {n_parts} parts",
            crate::log_ts()
        );

        tracing::info!(n_funcs, n_parts, "parallel codegen: splitting LLVM module");
        let split_start = std::time::Instant::now();

        // Split module into partitions (all share the original LLVMContext).
        let mut partitions: Vec<llvm_sys::prelude::LLVMModuleRef> = Vec::with_capacity(n_parts);
        unsafe extern "C" fn collect_partition(
            part: llvm_sys::prelude::LLVMModuleRef,
            user_data: *mut std::ffi::c_void,
        ) {
            let vec = unsafe { &mut *(user_data as *mut Vec<llvm_sys::prelude::LLVMModuleRef>) };
            vec.push(part);
        }
        LLVMSplitModule(
            llvm_module,
            n_parts as std::ffi::c_uint,
            collect_partition,
            &mut partitions as *mut Vec<llvm_sys::prelude::LLVMModuleRef> as *mut std::ffi::c_void,
        );

        // Serialise each partition to bitcode (single-threaded, fast).
        let mut bitcode_bufs: Vec<Vec<u8>> = Vec::with_capacity(partitions.len());
        for part in &partitions {
            let membuf = LLVMWriteBitcodeToMemoryBuffer(*part);
            let ptr = LLVMGetBufferStart(membuf) as *const u8;
            let len = LLVMGetBufferSize(membuf);
            let bytes = std::slice::from_raw_parts(ptr, len).to_vec();
            LLVMDisposeMemoryBuffer(membuf);
            bitcode_bufs.push(bytes);
        }

        // Dispose partitions and original module+context (no longer needed).
        for part in partitions {
            LLVMDisposeModule(part);
        }
        LLVMDisposeModule(llvm_module);
        LLVMContextDispose(llvm_ctx);

        let split_ms = split_start.elapsed().as_millis() as u64;
        eprintln!(
            "[{:>8.2}s] [{label}] split+ser: {split_ms}ms, sizes(KB): {:?}",
            crate::log_ts(),
            bitcode_bufs
                .iter()
                .map(|b| b.len() / 1024)
                .collect::<Vec<_>>()
        );

        // Compile each partition in parallel: fresh LLVMContext per thread.
        let obj_paths: Vec<std::path::PathBuf> = (0..bitcode_bufs.len())
            .map(|i| output_dir.join(format!("{base_name}_part{i}.o")))
            .collect();

        let errors: Vec<Option<CompileError>> = std::thread::scope(|s| {
            let handles: Vec<_> = bitcode_bufs
                .iter()
                .zip(obj_paths.iter())
                .enumerate()
                .map(|(i, (bc, obj_path))| {
                    s.spawn(move || {
                        let t0 = std::time::Instant::now();
                        let ctx = LLVMContextCreate();
                        let membuf = LLVMCreateMemoryBufferWithMemoryRangeCopy(
                            bc.as_ptr() as *const i8,
                            bc.len(),
                            c"partition".as_ptr(),
                        );
                        let mut part_module = std::ptr::null_mut();
                        if LLVMParseBitcodeInContext2(ctx, membuf, &mut part_module) != 0 {
                            LLVMContextDispose(ctx);
                            return Some(CompileError::ObjectEmit(format!(
                                "partition {i}: failed to parse bitcode"
                            )));
                        }
                        // O3 for small partitions, O2 for large ones
                        // (O3 is superlinear in IR size)
                        let bc_bytes = bc.len();
                        let opt = if bc_bytes < 50_000 { "O3" } else { "O2" };
                        match optimise_and_emit(part_module, ctx, obj_path, opt) {
                            Ok(()) => {
                                eprintln!(
                                    "[{:>8.2}s] [{label}] part {i} ({opt}, {}KB): {}ms",
                                    crate::log_ts(),
                                    bc_bytes / 1024,
                                    t0.elapsed().as_millis()
                                );
                                None
                            }
                            Err(e) => Some(e),
                        }
                    })
                })
                .collect();

            handles
                .into_iter()
                .map(|h| {
                    h.join().unwrap_or_else(|_| {
                        Some(CompileError::ObjectEmit("codegen thread panicked".into()))
                    })
                })
                .collect()
        });

        // Propagate first error.
        for err in errors {
            if let Some(e) = err {
                // Clean up any .o files that were produced.
                for p in &obj_paths {
                    std::fs::remove_file(p).ok();
                }
                return Err(e);
            }
        }

        eprintln!(
            "[{:>8.2}s] [{label}] all {} parts: {}ms",
            crate::log_ts(),
            obj_paths.len(),
            split_start.elapsed().as_millis()
        );

        Ok(obj_paths)
    }
}

/// Emit the packed-argument wrapper `_mlir__mlir_ciface_compute(void** args)`
/// into an already-translated LLVM module.
///
/// MLIR's `convert-func-to-llvm` pass emits `_mlir_ciface_compute(ptr0, ptr1, ...)`
/// which takes each MemRef descriptor pointer as a separate argument.
/// The packed wrapper takes a single `void**` argument and indexes into it,
/// matching the calling convention that the runtime uses (`invoke_packed`).
///
/// Implementation:
/// ```c
/// void _mlir__mlir_ciface_compute(void** args) {
///     _mlir_ciface_compute(args[0], args[1], ..., args[N-1]);
/// }
/// ```
unsafe fn emit_packed_wrapper(
    llvm_module: llvm_sys::prelude::LLVMModuleRef,
    llvm_ctx: llvm_sys::prelude::LLVMContextRef,
) -> Result<(), String> {
    use llvm_sys::core::*;
    use std::ffi::CString;

    unsafe {
        // Look up the direct C interface function.
        let ciface_name = CString::new("_mlir_ciface_compute").unwrap();
        let ciface_fn = LLVMGetNamedFunction(llvm_module, ciface_name.as_ptr());
        if ciface_fn.is_null() {
            return Err("_mlir_ciface_compute not found in LLVM module".into());
        }

        // Determine how many arguments it takes.
        let n_params = LLVMCountParams(ciface_fn) as usize;
        if n_params == 0 {
            return Err("_mlir_ciface_compute has 0 params".into());
        }

        // Build types: ptr_ty = i8* (opaque pointer), void_ty
        let ptr_ty = LLVMPointerTypeInContext(llvm_ctx, 0);
        let void_ty = LLVMVoidTypeInContext(llvm_ctx);

        // Signature: void wrapper(i8** args)
        let mut wrapper_param_types = [ptr_ty];
        let wrapper_fn_ty = LLVMFunctionType(void_ty, wrapper_param_types.as_mut_ptr(), 1, 0);

        let packed_name = CString::new("_mlir__mlir_ciface_compute").unwrap();
        let wrapper_fn = LLVMAddFunction(llvm_module, packed_name.as_ptr(), wrapper_fn_ty);

        // Set internal linkage so the optimiser can inline it.
        LLVMSetLinkage(wrapper_fn, llvm_sys::LLVMLinkage::LLVMExternalLinkage);

        // Build the function body.
        let entry_bb_name = CString::new("entry").unwrap();
        let builder = LLVMCreateBuilderInContext(llvm_ctx);
        let entry_bb = LLVMAppendBasicBlockInContext(llvm_ctx, wrapper_fn, entry_bb_name.as_ptr());
        LLVMPositionBuilderAtEnd(builder, entry_bb);

        // args_ptr = first parameter (i8**)
        let args_ptr = LLVMGetParam(wrapper_fn, 0);

        // Load each arg[i] and pass to ciface function.
        //
        // The runtime passes args as Vec<*mut ()> where each args[i] is
        // &mut (ptr_to_memref_descriptor) — i.e., double indirection.
        // _mlir_ciface_compute takes single pointers to MemRefDescriptors.
        // So we must load args[i] (pointer-to-pointer) and then dereference again.
        let mut call_args: Vec<llvm_sys::prelude::LLVMValueRef> = Vec::with_capacity(n_params);
        for i in 0..n_params {
            // GEP: &args[i]  (args_ptr is void** = ptr<ptr>)
            let idx = LLVMConstInt(LLVMInt64TypeInContext(llvm_ctx), i as u64, 0);
            let gep_name = CString::new(format!("gep_{i}")).unwrap();
            let elem_ptr = LLVMBuildInBoundsGEP2(
                builder,
                ptr_ty,
                args_ptr,
                [idx].as_mut_ptr(),
                1,
                gep_name.as_ptr(),
            );
            // Load args[i]: gets a void* which is itself a ptr-to-MemRefDescriptor
            let pp_name = CString::new(format!("pp_{i}")).unwrap();
            let pp_val = LLVMBuildLoad2(builder, ptr_ty, elem_ptr, pp_name.as_ptr());
            // Dereference once more: load *args[i] to get the MemRefDescriptor pointer
            let arg_name = CString::new(format!("arg_{i}")).unwrap();
            let arg_val = LLVMBuildLoad2(builder, ptr_ty, pp_val, arg_name.as_ptr());
            call_args.push(arg_val);
        }

        // Get the function type of _mlir_ciface_compute for the call instruction.
        let ciface_fn_ty = LLVMGlobalGetValueType(ciface_fn);
        let call_name = CString::new("").unwrap();
        LLVMBuildCall2(
            builder,
            ciface_fn_ty,
            ciface_fn,
            call_args.as_mut_ptr(),
            n_params as u32,
            call_name.as_ptr(),
        );
        LLVMBuildRetVoid(builder);

        LLVMDisposeBuilder(builder);
        Ok(())
    } // end unsafe
}

/// Return the MLIR/LLVM library prefix directory.
///
/// Reads `MLIR_SYS_210_PREFIX` (set by env-llvm21-dev.sh) and returns its
/// `lib/` subdirectory. Falls back to `/usr/lib/llvm/21/lib64`.
fn mlir_lib_dir() -> std::path::PathBuf {
    if let Ok(prefix) = std::env::var("MLIR_SYS_210_PREFIX") {
        return std::path::PathBuf::from(prefix).join("lib");
    }
    std::path::PathBuf::from("/usr/lib/llvm/21/lib64")
}

/// Return the path to `libmlir_c_runner_utils.so`.
///
/// Reads `MLIR_RUNNER_UTILS_PATH` if set, otherwise derives from the MLIR
/// prefix.
fn runner_utils_lib_path() -> std::path::PathBuf {
    if let Ok(path) = std::env::var("MLIR_RUNNER_UTILS_PATH") {
        return std::path::PathBuf::from(path);
    }
    mlir_lib_dir().join("libmlir_c_runner_utils.so")
}

/// Return the path to `libomp.so`.
///
/// Reads `OMP_LIB_PATH` if set, otherwise derives from the MLIR prefix.
/// The LLVM 21 build installs libomp under a target-triple subdirectory.
fn omp_lib_path() -> std::path::PathBuf {
    if let Ok(path) = std::env::var("OMP_LIB_PATH") {
        return std::path::PathBuf::from(path);
    }
    let base = mlir_lib_dir();
    // LLVM installs libomp under lib/<triple>/ (e.g. lib/x86_64-unknown-linux-gnu/)
    let triple_dir = base.join("x86_64-unknown-linux-gnu").join("libomp.so");
    if triple_dir.exists() {
        return triple_dir;
    }
    // Fall back to lib/libomp.so (system LLVM installs)
    base.join("libomp.so")
}

/// Link an object file into a shared library.
///
/// Links `runner_utils` and `libomp` by full path and bakes their directories
/// into the `.so`'s `DT_RUNPATH` so the dynamic linker finds them at dlopen
/// time without requiring `LD_LIBRARY_PATH`.
fn link_shared_lib(
    obj_paths: &[std::path::PathBuf],
    so_path: &std::path::Path,
) -> Result<(), CompileError> {
    let runner_utils_path = runner_utils_lib_path();
    let runner_utils_dir = runner_utils_path
        .parent()
        .unwrap_or(std::path::Path::new("/usr/lib/llvm/21/lib64"))
        .to_str()
        .unwrap_or("/usr/lib/llvm/21/lib64");

    let omp_path = omp_lib_path();
    let omp_dir = omp_path
        .parent()
        .unwrap_or(std::path::Path::new("/usr/lib/llvm/21/lib64"))
        .to_str()
        .unwrap_or("/usr/lib/llvm/21/lib64");

    let mut args: Vec<&str> = vec!["-shared", "-fPIC", "-o", so_path.to_str().unwrap()];
    for p in obj_paths {
        args.push(p.to_str().unwrap());
    }
    args.push(runner_utils_path.to_str().unwrap());
    args.push(omp_path.to_str().unwrap());
    args.push("-lm");
    let rpath_runner = format!("-Wl,-rpath,{runner_utils_dir}");
    let rpath_omp = format!("-Wl,-rpath,{omp_dir}");
    args.push(&rpath_runner);
    args.push(&rpath_omp);

    let status = std::process::Command::new("cc")
        .args(&args)
        .status()
        .map_err(|e| CompileError::Link(format!("failed to run cc: {e}")))?;

    if !status.success() {
        return Err(CompileError::Link(format!("cc exited with {status}")));
    }
    Ok(())
}

// ── Transform dialect schedule ────────────────────────────────────────────────
//
// Emits transform named_sequences that use foreach_match to find and tile
// contraction-like linalg.generic ops (arith.mulf + arith.addf body).
//
// Structure:
//   @match_contraction_3d  — matches 3-iter-dim contractions (plain matmul)
//   @tile_contraction_3d   — tiles [32, 32, 0] (M, N parallel; K sequential)
//   @match_contraction_4d  — matches 4-iter-dim contractions (batched matmul)
//   @tile_contraction_4d   — tiles [0, 32, 32, 0] (B untiled; M, N parallel; K sequential)
//   @__transform_main      — foreach_match dispatching the two pairs

pub(crate) fn build_transform_schedule<'c>(
    context: &'c Context,
    module: &Module<'c>,
    location: Location<'c>,
) {
    let any_op_type =
        melior::ir::Type::parse(context, "!transform.any_op").expect("parse !transform.any_op");
    let param_i64_type = melior::ir::Type::parse(context, "!transform.param<i64>")
        .expect("parse !transform.param<i64>");

    // Helper: build a @match_contraction_Nd named_sequence that matches
    // linalg.generic contraction ops with exactly `rank` iteration dims.
    //
    // The sequence filters by:
    //   1. transform.match.operation_name — must be "linalg.generic"
    //   2. transform.match.structured body — must have contraction [mulf, addf] body
    //   3. transform.match.structured.rank — iteration dims must equal `rank`
    //   4. transform.match.param.cmpi eq  — enforce exact rank
    let build_match_seq = |name: &str, rank: i64| {
        // ── inner block of match.structured ──────────────────────────────────
        // ^bb0(%arg0: !transform.any_op):
        //   transform.match.structured.body %arg0 {contraction = ["arith.mulf", "arith.addf"]}
        //   %rank = transform.match.structured.rank %arg0 : ... -> !transform.param<i64>
        //   transform.match.structured.yield %arg0, %rank
        let inner_block = Block::new(&[(any_op_type, location)]);
        let inner_arg: melior::ir::Value = inner_block.argument(0).unwrap().into();

        let contraction_attr = ArrayAttribute::new(
            context,
            &[
                StringAttribute::new(context, "arith.mulf").into(),
                StringAttribute::new(context, "arith.addf").into(),
            ],
        );
        let body_check_op = OperationBuilder::new("transform.match.structured.body", location)
            .add_operands(&[inner_arg])
            .add_attributes(&[(
                Identifier::new(context, "contraction"),
                contraction_attr.into(),
            )])
            .build()
            .expect("build transform.match.structured.body");
        inner_block.append_operation(body_check_op);

        // Capture the iteration rank of this structured op.
        let rank_op = OperationBuilder::new("transform.match.structured.rank", location)
            .add_operands(&[inner_arg])
            .add_results(&[param_i64_type])
            .build()
            .expect("build transform.match.structured.rank");
        let rank_ref = inner_block.append_operation(rank_op);
        let rank_param: melior::ir::Value = rank_ref.result(0).unwrap().into();

        // Yield both the op handle and rank param out of the inner block.
        let inner_yield_op = OperationBuilder::new("transform.match.structured.yield", location)
            .add_operands(&[inner_arg, rank_param])
            .build()
            .expect("build transform.match.structured.yield");
        inner_block.append_operation(inner_yield_op);

        let inner_region = Region::new();
        inner_region.append_block(inner_block);

        // ── outer block of @match_contraction_Nd ─────────────────────────────
        let match_outer_block = Block::new(&[(any_op_type, location)]);
        let candidate: melior::ir::Value = match_outer_block.argument(0).unwrap().into();

        // Filter to linalg.generic only.
        let op_name_filter = OperationBuilder::new("transform.match.operation_name", location)
            .add_operands(&[candidate])
            .add_attributes(&[(
                Identifier::new(context, "op_names"),
                ArrayAttribute::new(
                    context,
                    &[StringAttribute::new(context, "linalg.generic").into()],
                )
                .into(),
            )])
            .build()
            .expect("build transform.match.operation_name");
        match_outer_block.append_operation(op_name_filter);

        // match.structured returns (!transform.any_op, !transform.param<i64>)
        let match_structured_op = OperationBuilder::new("transform.match.structured", location)
            .add_operands(&[candidate])
            .add_results(&[any_op_type, param_i64_type])
            .add_regions([inner_region])
            .build()
            .expect("build transform.match.structured");
        let match_structured_ref = match_outer_block.append_operation(match_structured_op);
        let matched_op: melior::ir::Value = match_structured_ref.result(0).unwrap().into();
        let matched_rank: melior::ir::Value = match_structured_ref.result(1).unwrap().into();

        // transform.param.constant <rank> : i64 -> !transform.param<i64>
        let rank_const_attr =
            Attribute::parse(context, &format!("{rank} : i64")).expect("parse rank constant attr");
        let rank_const_op = OperationBuilder::new("transform.param.constant", location)
            .add_attributes(&[(Identifier::new(context, "value"), rank_const_attr)])
            .add_results(&[param_i64_type])
            .build()
            .expect("build transform.param.constant");
        let rank_const_ref = match_outer_block.append_operation(rank_const_op);
        let expected_rank: melior::ir::Value = rank_const_ref.result(0).unwrap().into();

        // transform.match.param.cmpi eq %matched_rank, %expected_rank : !transform.param<i64>
        // predicate attr: I32EnumAttr with value eq=0
        let predicate_attr = Attribute::parse(context, "0 : i32").expect("parse eq predicate");
        let cmpi_op = OperationBuilder::new("transform.match.param.cmpi", location)
            .add_operands(&[matched_rank, expected_rank])
            .add_attributes(&[(Identifier::new(context, "predicate"), predicate_attr)])
            .build()
            .expect("build transform.match.param.cmpi");
        match_outer_block.append_operation(cmpi_op);

        // transform.yield %matched_op
        let match_outer_yield = OperationBuilder::new("transform.yield", location)
            .add_operands(&[matched_op])
            .build()
            .expect("build transform.yield (match_contraction)");
        match_outer_block.append_operation(match_outer_yield);

        let match_outer_region = Region::new();
        match_outer_region.append_block(match_outer_block);

        let match_func_type = FunctionType::new(context, &[any_op_type], &[any_op_type]);
        let match_arg_attrs =
            Attribute::parse(context, "[{transform.readonly}]").expect("parse match arg_attrs");

        OperationBuilder::new("transform.named_sequence", location)
            .add_attributes(&[
                (
                    Identifier::new(context, "sym_name"),
                    StringAttribute::new(context, name).into(),
                ),
                (
                    Identifier::new(context, "function_type"),
                    TypeAttribute::new(match_func_type.into()).into(),
                ),
                (Identifier::new(context, "arg_attrs"), match_arg_attrs),
                (
                    Identifier::new(context, "sym_visibility"),
                    StringAttribute::new(context, "private").into(),
                ),
            ])
            .add_regions([match_outer_region])
            .build()
            .expect("build transform.named_sequence @match_contraction_Nd")
    };

    // Helper: build a @tile_contraction_Nd named_sequence.
    //
    // `forall_sizes`:   static_tile_sizes for tile_using_forall (cache tile, 0 = skip dim)
    // `reg_sizes`:      static_sizes for tile_using_for (register tile)
    // `pad_dims`:       MLIR array attribute text for padding_dimensions
    // `pad_multiples`:  MLIR array<i64:...> text for static_pad_to_multiple_of
    // `n_reg_loops`:    number of non-zero dims in reg_sizes (= number of loop results)
    let build_tile_seq = |name: &str,
                          forall_sizes: &str,
                          scalable_forall: &str,
                          reg_sizes: &str,
                          scalable_reg: &str,
                          pad_dims: &str,
                          pad_multiples: &str,
                          n_reg_loops: usize| {
        let tile_outer_block = Block::new(&[(any_op_type, location)]);
        let op_to_tile: melior::ir::Value = tile_outer_block.argument(0).unwrap().into();

        let static_tile_sizes_attr =
            Attribute::parse(context, forall_sizes).expect("parse static_tile_sizes");
        let static_num_threads_attr =
            Attribute::parse(context, "array<i64>").expect("parse static_num_threads");
        let scalable_sizes_attr =
            Attribute::parse(context, scalable_forall).expect("parse scalable_sizes");
        let operand_segment_sizes = Attribute::parse(context, "array<i32: 1, 0, 0, 0, 0>")
            .expect("parse operandSegmentSizes");

        let tile_op = OperationBuilder::new("transform.structured.tile_using_forall", location)
            .add_operands(&[op_to_tile])
            .add_attributes(&[
                (
                    Identifier::new(context, "static_num_threads"),
                    static_num_threads_attr,
                ),
                (
                    Identifier::new(context, "static_tile_sizes"),
                    static_tile_sizes_attr,
                ),
                (
                    Identifier::new(context, "scalable_sizes"),
                    scalable_sizes_attr,
                ),
                (
                    Identifier::new(context, "operandSegmentSizes"),
                    operand_segment_sizes,
                ),
            ])
            .add_results(&[any_op_type, any_op_type])
            .build()
            .expect("build structured.tile_using_forall (cache tile)");
        let tile_l1_ref = tile_outer_block.append_operation(tile_op);
        let tiled_l1: melior::ir::Value = tile_l1_ref.result(0).unwrap().into();
        let forall_loop: melior::ir::Value = tile_l1_ref.result(1).unwrap().into();

        let reg_static_sizes =
            Attribute::parse(context, reg_sizes).expect("parse reg static_sizes");
        let reg_scalable_sizes =
            Attribute::parse(context, scalable_reg).expect("parse reg scalable_sizes");

        // tile_using_for returns: tiled_op + one loop handle per non-zero tile dim
        let mut reg_results: Vec<melior::ir::Type> = vec![any_op_type];
        for _ in 0..n_reg_loops {
            reg_results.push(any_op_type);
        }
        let reg_tile_op = OperationBuilder::new("transform.structured.tile_using_for", location)
            .add_operands(&[tiled_l1])
            .add_attributes(&[
                (Identifier::new(context, "static_sizes"), reg_static_sizes),
                (
                    Identifier::new(context, "scalable_sizes"),
                    reg_scalable_sizes,
                ),
            ])
            .add_results(&reg_results)
            .build()
            .expect("build structured.tile_using_for (register tile)");
        let reg_tile_ref = tile_outer_block.append_operation(reg_tile_op);
        let tiled_l2: melior::ir::Value = reg_tile_ref.result(0).unwrap().into();

        let pad_dims_attr = Attribute::parse(context, pad_dims).expect("parse padding_dimensions");
        let pad_multiples_attr =
            Attribute::parse(context, pad_multiples).expect("parse pad_to_multiple_of");
        let copy_back = StringAttribute::new(context, "none");
        let pad_op = OperationBuilder::new("transform.structured.pad", location)
            .add_operands(&[tiled_l2])
            .add_attributes(&[
                (
                    Identifier::new(context, "padding_dimensions"),
                    pad_dims_attr,
                ),
                (
                    Identifier::new(context, "static_pad_to_multiple_of"),
                    pad_multiples_attr,
                ),
                (Identifier::new(context, "copy_back_op"), copy_back.into()),
            ])
            .add_results(&[any_op_type, any_op_type, any_op_type])
            .build()
            .expect("build structured.pad");
        tile_outer_block.append_operation(pad_op);

        // Outline the forall loop into a temporary function so we can
        // vectorize it with vectorize_children_and_apply_patterns (which
        // requires an "isolated from above" parent, i.e. func.func).
        // LLVM will inline it back during O2 codegen.
        let outline_name = StringAttribute::new(context, &format!("{name}_kernel"));
        let outline_op = OperationBuilder::new("transform.loop.outline", location)
            .add_operands(&[forall_loop])
            .add_attributes(&[(Identifier::new(context, "func_name"), outline_name.into())])
            .add_results(&[any_op_type, any_op_type])
            .build()
            .expect("build transform.loop.outline");
        let outline_ref = tile_outer_block.append_operation(outline_op);
        let outlined_func: melior::ir::Value = outline_ref.result(0).unwrap().into();

        // Vectorize the outlined function (+ cleanup patterns for bufferization).
        let vectorize_op = OperationBuilder::new(
            "transform.structured.vectorize_children_and_apply_patterns",
            location,
        )
        .add_operands(&[outlined_func])
        .add_results(&[any_op_type])
        .build()
        .expect("build vectorize_children_and_apply_patterns (outlined)");
        tile_outer_block.append_operation(vectorize_op);

        let tile_yield_op = OperationBuilder::new("transform.yield", location)
            .build()
            .expect("build transform.yield (tile_contraction)");
        tile_outer_block.append_operation(tile_yield_op);

        let tile_outer_region = Region::new();
        tile_outer_region.append_block(tile_outer_block);

        let tile_func_type = FunctionType::new(context, &[any_op_type], &[]);
        let tile_arg_attrs =
            Attribute::parse(context, "[{transform.consumed}]").expect("parse tile arg_attrs");

        OperationBuilder::new("transform.named_sequence", location)
            .add_attributes(&[
                (
                    Identifier::new(context, "sym_name"),
                    StringAttribute::new(context, name).into(),
                ),
                (
                    Identifier::new(context, "function_type"),
                    TypeAttribute::new(tile_func_type.into()).into(),
                ),
                (Identifier::new(context, "arg_attrs"), tile_arg_attrs),
                (
                    Identifier::new(context, "sym_visibility"),
                    StringAttribute::new(context, "private").into(),
                ),
            ])
            .add_regions([tile_outer_region])
            .build()
            .expect("build transform.named_sequence @tile_contraction_Nd")
    };

    // ── @match_contraction_3d ──────────────────────────────────────────────────
    // Matches plain (2D) matmul generics: 3 iteration dims (M, N, K).
    module
        .body()
        .append_operation(build_match_seq("match_contraction_3d", 3));

    // ── @tile_contraction_3d ───────────────────────────────────────────────────
    // Tile sizes [32, 32, 0]: tile M and N at L1 cache level; K is reduction (untiled).
    // Register tile [8, 8, 1]: tile M=8, N=8, K=1 for vectorization.
    // Pad dims [0, 1, 2] to multiples of [8, 8, 1].
    module.body().append_operation(build_tile_seq(
        "tile_contraction_3d",
        "array<i64: 32, 32, 0>",
        "array<i1: false, false, false>",
        "array<i64: 8, 8, 1>",
        "array<i1: false, false, false>",
        "[0, 1, 2]",
        "array<i64: 8, 8, 1>",
        3, // M, N, K all non-zero in register tile
    ));

    // ── @match_contraction_4d ──────────────────────────────────────────────────
    // Matches batched matmul generics: 4 iteration dims (B, M, N, K).
    module
        .body()
        .append_operation(build_match_seq("match_contraction_4d", 4));

    // ── @tile_contraction_4d ───────────────────────────────────────────────────
    // Tile sizes [0, 32, 32, 0]: skip B, tile M and N at L1; K is reduction (untiled).
    // Register tile [0, 8, 8, 1]: skip B, tile M=8, N=8, K=1 for vectorization.
    // Pad dims [1, 2, 3] (M, N, K within 4-dim space) to multiples of [8, 8, 1].
    // Note: pad_to_multiple_of must have exactly as many entries as padding_dimensions.
    module.body().append_operation(build_tile_seq(
        "tile_contraction_4d",
        "array<i64: 0, 32, 32, 0>",
        "array<i1: false, false, false, false>",
        "array<i64: 0, 8, 8, 1>",
        "array<i1: false, false, false, false>",
        "[1, 2, 3]",
        "array<i64: 8, 8, 1>", // 3 entries matching padding_dimensions [1, 2, 3]
        3,                     // M, N, K non-zero in register tile (B is 0, so no loop for B)
    ));

    // ── @match_conv_nchw ────────────────────────────────────────────────────
    // Matches linalg.conv_2d_nchw_fchw by operation name.
    {
        let match_block = Block::new(&[(any_op_type, location)]);
        let candidate: melior::ir::Value = match_block.argument(0).unwrap().into();

        // Filter to linalg.conv_2d_nchw_fchw only.
        let op_name_filter = OperationBuilder::new("transform.match.operation_name", location)
            .add_operands(&[candidate])
            .add_attributes(&[(
                Identifier::new(context, "op_names"),
                ArrayAttribute::new(
                    context,
                    &[StringAttribute::new(context, "linalg.conv_2d_nchw_fchw").into()],
                )
                .into(),
            )])
            .build()
            .expect("build transform.match.operation_name (conv)");
        match_block.append_operation(op_name_filter);

        let yield_op = OperationBuilder::new("transform.yield", location)
            .add_operands(&[candidate])
            .build()
            .expect("build transform.yield (match_conv)");
        match_block.append_operation(yield_op);

        let match_region = Region::new();
        match_region.append_block(match_block);

        let match_func_type = FunctionType::new(context, &[any_op_type], &[any_op_type]);
        let match_arg_attrs =
            Attribute::parse(context, "[{transform.readonly}]").expect("parse match arg_attrs");

        let match_seq = OperationBuilder::new("transform.named_sequence", location)
            .add_attributes(&[
                (
                    Identifier::new(context, "sym_name"),
                    StringAttribute::new(context, "match_conv_nchw").into(),
                ),
                (
                    Identifier::new(context, "function_type"),
                    TypeAttribute::new(match_func_type.into()).into(),
                ),
                (Identifier::new(context, "arg_attrs"), match_arg_attrs),
                (
                    Identifier::new(context, "sym_visibility"),
                    StringAttribute::new(context, "private").into(),
                ),
            ])
            .add_regions([match_region])
            .build()
            .expect("build @match_conv_nchw");
        module.body().append_operation(match_seq);
    }

    // ── @tile_conv_nchw ──────────────────────────────────────────────────────
    // Conv iteration dims: [N, CO, OH, OW, CI, KH, KW]
    // Cache tile:    [0, 32, 4, 4, 0, 0, 0] — tile CO+spatial for L1
    // Register tile: [0,  8, 1, 1, 1, 0, 0] — vectorize over CO=8
    // Pad dims [1, 4] (CO, CI) to multiples [8, 1]
    module.body().append_operation(build_tile_seq(
        "tile_conv_nchw",
        "array<i64: 0, 32, 4, 4, 0, 0, 0>",
        "array<i1: false, false, false, false, false, false, false>",
        "array<i64: 0, 8, 1, 1, 1, 0, 0>",
        "array<i1: false, false, false, false, false, false, false>",
        "[1, 4]",
        "array<i64: 8, 1>",
        4, // non-zero register dims: CO=8, OH=1, OW=1, CI=1
    ));

    // ── @__transform_main ─────────────────────────────────────────────────────
    // Tiles + vectorizes matched ops. Vectorization happens per-op inside each
    // tile sequence (on the innermost tiled micro-kernel only), not as a blanket
    // pass over all functions. Untiled ops stay scalar.

    let main_block = Block::new(&[(any_op_type, location)]);
    let module_handle: melior::ir::Value = main_block.argument(0).unwrap().into();

    let matchers_attr = Attribute::parse(
        context,
        "[@match_contraction_3d, @match_contraction_4d, @match_conv_nchw]",
    )
    .expect("parse matchers attr");
    let actions_attr = Attribute::parse(
        context,
        "[@tile_contraction_3d, @tile_contraction_4d, @tile_conv_nchw]",
    )
    .expect("parse actions attr");

    let foreach_op = OperationBuilder::new("transform.foreach_match", location)
        .add_operands(&[module_handle])
        .add_attributes(&[
            (Identifier::new(context, "matchers"), matchers_attr),
            (Identifier::new(context, "actions"), actions_attr),
        ])
        .add_results(&[any_op_type])
        .build()
        .expect("build transform.foreach_match");
    main_block.append_operation(foreach_op);

    // No blanket vectorize — each tile sequence outlines the tiled region
    // into a temporary function and vectorizes it there. Untiled ops stay
    // scalar. LLVM inlines the outlined functions during O2.

    let main_yield_op = OperationBuilder::new("transform.yield", location)
        .build()
        .expect("build transform.yield (__transform_main)");
    main_block.append_operation(main_yield_op);

    let main_region = Region::new();
    main_region.append_block(main_block);

    let main_func_type = FunctionType::new(context, &[any_op_type], &[]);
    let main_arg_attrs =
        Attribute::parse(context, "[{transform.consumed}]").expect("parse main arg_attrs");

    let main_named_seq = OperationBuilder::new("transform.named_sequence", location)
        .add_attributes(&[
            (
                Identifier::new(context, "sym_name"),
                StringAttribute::new(context, "__transform_main").into(),
            ),
            (
                Identifier::new(context, "function_type"),
                TypeAttribute::new(main_func_type.into()).into(),
            ),
            (Identifier::new(context, "arg_attrs"), main_arg_attrs),
            (
                Identifier::new(context, "sym_visibility"),
                StringAttribute::new(context, "private").into(),
            ),
        ])
        .add_regions([main_region])
        .build()
        .expect("build transform.named_sequence @__transform_main");
    module.body().append_operation(main_named_seq);
}

// ── Vectorize-only transform schedule ─────────────────────────────────────────
//
// Like `build_transform_schedule` but without tiling, outlining, or OpenMP.
// Action sequences simply call vectorize_children_and_apply_patterns on the
// parent func.func. Suitable for dynamic-M matmuls (e.g. M=1 decode) where
// tiling + OpenMP fork/join overhead dominates.
//
// Structure:
//   @match_contraction_3d  — identical to full schedule
//   @match_contraction_4d  — identical to full schedule
//   @vectorize_contraction_3d — vectorize_children_and_apply_patterns on parent func
//   @vectorize_contraction_4d — vectorize_children_and_apply_patterns on parent func
//   @__transform_main      — foreach_match dispatching the two pairs

pub(crate) fn build_vectorize_only_schedule<'c>(
    context: &'c Context,
    module: &Module<'c>,
    location: Location<'c>,
) {
    let any_op_type =
        melior::ir::Type::parse(context, "!transform.any_op").expect("parse !transform.any_op");
    let param_i64_type = melior::ir::Type::parse(context, "!transform.param<i64>")
        .expect("parse !transform.param<i64>");

    // Reuse same match sequence builder as build_transform_schedule.
    let build_match_seq = |name: &str, rank: i64| {
        let inner_block = Block::new(&[(any_op_type, location)]);
        let inner_arg: melior::ir::Value = inner_block.argument(0).unwrap().into();

        let contraction_attr = ArrayAttribute::new(
            context,
            &[
                StringAttribute::new(context, "arith.mulf").into(),
                StringAttribute::new(context, "arith.addf").into(),
            ],
        );
        let body_check_op = OperationBuilder::new("transform.match.structured.body", location)
            .add_operands(&[inner_arg])
            .add_attributes(&[(
                Identifier::new(context, "contraction"),
                contraction_attr.into(),
            )])
            .build()
            .expect("build transform.match.structured.body");
        inner_block.append_operation(body_check_op);

        let rank_op = OperationBuilder::new("transform.match.structured.rank", location)
            .add_operands(&[inner_arg])
            .add_results(&[param_i64_type])
            .build()
            .expect("build transform.match.structured.rank");
        let rank_ref = inner_block.append_operation(rank_op);
        let rank_param: melior::ir::Value = rank_ref.result(0).unwrap().into();

        let inner_yield_op = OperationBuilder::new("transform.match.structured.yield", location)
            .add_operands(&[inner_arg, rank_param])
            .build()
            .expect("build transform.match.structured.yield");
        inner_block.append_operation(inner_yield_op);

        let inner_region = Region::new();
        inner_region.append_block(inner_block);

        let match_outer_block = Block::new(&[(any_op_type, location)]);
        let candidate: melior::ir::Value = match_outer_block.argument(0).unwrap().into();

        let op_name_filter = OperationBuilder::new("transform.match.operation_name", location)
            .add_operands(&[candidate])
            .add_attributes(&[(
                Identifier::new(context, "op_names"),
                ArrayAttribute::new(
                    context,
                    &[StringAttribute::new(context, "linalg.generic").into()],
                )
                .into(),
            )])
            .build()
            .expect("build transform.match.operation_name");
        match_outer_block.append_operation(op_name_filter);

        let match_structured_op = OperationBuilder::new("transform.match.structured", location)
            .add_operands(&[candidate])
            .add_results(&[any_op_type, param_i64_type])
            .add_regions([inner_region])
            .build()
            .expect("build transform.match.structured");
        let match_structured_ref = match_outer_block.append_operation(match_structured_op);
        let matched_op: melior::ir::Value = match_structured_ref.result(0).unwrap().into();
        let matched_rank: melior::ir::Value = match_structured_ref.result(1).unwrap().into();

        let rank_const_attr =
            Attribute::parse(context, &format!("{rank} : i64")).expect("parse rank constant attr");
        let rank_const_op = OperationBuilder::new("transform.param.constant", location)
            .add_attributes(&[(Identifier::new(context, "value"), rank_const_attr)])
            .add_results(&[param_i64_type])
            .build()
            .expect("build transform.param.constant");
        let rank_const_ref = match_outer_block.append_operation(rank_const_op);
        let expected_rank: melior::ir::Value = rank_const_ref.result(0).unwrap().into();

        let predicate_attr = Attribute::parse(context, "0 : i32").expect("parse eq predicate");
        let cmpi_op = OperationBuilder::new("transform.match.param.cmpi", location)
            .add_operands(&[matched_rank, expected_rank])
            .add_attributes(&[(Identifier::new(context, "predicate"), predicate_attr)])
            .build()
            .expect("build transform.match.param.cmpi");
        match_outer_block.append_operation(cmpi_op);

        let match_outer_yield = OperationBuilder::new("transform.yield", location)
            .add_operands(&[matched_op])
            .build()
            .expect("build transform.yield (match_contraction)");
        match_outer_block.append_operation(match_outer_yield);

        let match_outer_region = Region::new();
        match_outer_region.append_block(match_outer_block);

        let match_func_type = FunctionType::new(context, &[any_op_type], &[any_op_type]);
        let match_arg_attrs =
            Attribute::parse(context, "[{transform.readonly}]").expect("parse match arg_attrs");

        OperationBuilder::new("transform.named_sequence", location)
            .add_attributes(&[
                (
                    Identifier::new(context, "sym_name"),
                    StringAttribute::new(context, name).into(),
                ),
                (
                    Identifier::new(context, "function_type"),
                    TypeAttribute::new(match_func_type.into()).into(),
                ),
                (Identifier::new(context, "arg_attrs"), match_arg_attrs),
                (
                    Identifier::new(context, "sym_visibility"),
                    StringAttribute::new(context, "private").into(),
                ),
            ])
            .add_regions([match_outer_region])
            .build()
            .expect("build transform.named_sequence @match_contraction_Nd")
    };

    // Helper: build a @vectorize_contraction_Nd named_sequence.
    //
    // Tiles N dimension with tile_using_for (no forall → no OpenMP) to get
    // manageable vector sizes (16 floats = AVX-512), pads to multiple,
    // then vectorizes the parent function.
    //
    // tile_sizes:     [0, 16, 0] for 3D, [0, 0, 16, 0] for 4D — tile N only
    // scalable:       all false
    // n_tile_loops:   1 (only N is non-zero)
    // pad_dims/mults: pad N to multiple of 16
    let build_vectorize_seq = |name: &str,
                                tile_sizes: &str,
                                scalable: &str,
                                n_tile_loops: usize| {
        let vec_block = Block::new(&[(any_op_type, location)]);
        let op_handle: melior::ir::Value = vec_block.argument(0).unwrap().into();

        // Get parent func BEFORE tiling (op_handle is consumed by tile_using_for).
        let get_parent_op = OperationBuilder::new("transform.get_parent_op", location)
            .add_operands(&[op_handle])
            .add_results(&[any_op_type])
            .add_attributes(&[
                (
                    Identifier::new(context, "op_name"),
                    StringAttribute::new(context, "func.func").into(),
                ),
                (
                    Identifier::new(context, "deduplicate"),
                    Attribute::unit(context),
                ),
            ])
            .build()
            .expect("build transform.get_parent_op");
        let get_parent_ref = vec_block.append_operation(get_parent_op);
        let func_handle: melior::ir::Value = get_parent_ref.result(0).unwrap().into();

        // tile_using_for on N dimension only — produces scf.for, not scf.forall.
        let static_sizes =
            Attribute::parse(context, tile_sizes).expect("parse tile static_sizes");
        let scalable_sizes =
            Attribute::parse(context, scalable).expect("parse tile scalable_sizes");
        let mut tile_results: Vec<melior::ir::Type> = vec![any_op_type];
        for _ in 0..n_tile_loops {
            tile_results.push(any_op_type);
        }
        let tile_op = OperationBuilder::new("transform.structured.tile_using_for", location)
            .add_operands(&[op_handle])
            .add_attributes(&[
                (Identifier::new(context, "static_sizes"), static_sizes),
                (Identifier::new(context, "scalable_sizes"), scalable_sizes),
            ])
            .add_results(&tile_results)
            .build()
            .expect("build structured.tile_using_for (vectorize-only)");
        vec_block.append_operation(tile_op);

        // No padding — dynamic M can't be padded. N is already tiled to 16
        // which is a clean vector width. K (768, 2304, 3072) divides evenly.

        // Vectorize the parent function (which now contains tiled ops).
        let vectorize_op = OperationBuilder::new(
            "transform.structured.vectorize_children_and_apply_patterns",
            location,
        )
        .add_operands(&[func_handle])
        .add_results(&[any_op_type])
        .build()
        .expect("build vectorize_children_and_apply_patterns");
        vec_block.append_operation(vectorize_op);

        let vec_yield_op = OperationBuilder::new("transform.yield", location)
            .build()
            .expect("build transform.yield (vectorize)");
        vec_block.append_operation(vec_yield_op);

        let vec_region = Region::new();
        vec_region.append_block(vec_block);

        let vec_func_type = FunctionType::new(context, &[any_op_type], &[]);
        let vec_arg_attrs =
            Attribute::parse(context, "[{transform.consumed}]").expect("parse vec arg_attrs");

        OperationBuilder::new("transform.named_sequence", location)
            .add_attributes(&[
                (
                    Identifier::new(context, "sym_name"),
                    StringAttribute::new(context, name).into(),
                ),
                (
                    Identifier::new(context, "function_type"),
                    TypeAttribute::new(vec_func_type.into()).into(),
                ),
                (Identifier::new(context, "arg_attrs"), vec_arg_attrs),
                (
                    Identifier::new(context, "sym_visibility"),
                    StringAttribute::new(context, "private").into(),
                ),
            ])
            .add_regions([vec_region])
            .build()
            .expect("build transform.named_sequence @vectorize_contraction_Nd")
    };

    // ── @match_contraction_3d / @match_contraction_4d ─────────────────────────
    module
        .body()
        .append_operation(build_match_seq("match_contraction_3d", 3));
    module
        .body()
        .append_operation(build_match_seq("match_contraction_4d", 4));

    // ── @vectorize_contraction_3d ──────────────────────────────────────────────
    // 3D matmul (M, N, K): tile N=16 and K=16. No padding (M is dynamic).
    // Produces vector<16xf32> on N, with K=16 inner reduction.
    module.body().append_operation(build_vectorize_seq(
        "vectorize_contraction_3d",
        "array<i64: 0, 16, 16>",         // tile N=16, K=16, skip M
        "array<i1: false, false, false>",
        2,                                // 2 non-zero tile dims (N, K)
    ));
    // ── @vectorize_contraction_4d ──────────────────────────────────────────────
    // 4D batched matmul (B, M, N, K): tile N=16 and K=16.
    module.body().append_operation(build_vectorize_seq(
        "vectorize_contraction_4d",
        "array<i64: 0, 0, 16, 16>",             // tile N=16, K=16, skip B and M
        "array<i1: false, false, false, false>",
        2,                                        // 2 non-zero tile dims (N, K)
    ));

    // ── @__transform_main ─────────────────────────────────────────────────────
    let main_block = Block::new(&[(any_op_type, location)]);
    let module_handle: melior::ir::Value = main_block.argument(0).unwrap().into();

    let matchers_attr = Attribute::parse(
        context,
        "[@match_contraction_3d, @match_contraction_4d]",
    )
    .expect("parse matchers attr");
    let actions_attr = Attribute::parse(
        context,
        "[@vectorize_contraction_3d, @vectorize_contraction_4d]",
    )
    .expect("parse actions attr");

    let foreach_op = OperationBuilder::new("transform.foreach_match", location)
        .add_operands(&[module_handle])
        .add_attributes(&[
            (Identifier::new(context, "matchers"), matchers_attr),
            (Identifier::new(context, "actions"), actions_attr),
        ])
        .add_results(&[any_op_type])
        .build()
        .expect("build transform.foreach_match");
    main_block.append_operation(foreach_op);

    let main_yield_op = OperationBuilder::new("transform.yield", location)
        .build()
        .expect("build transform.yield (__transform_main)");
    main_block.append_operation(main_yield_op);

    let main_region = Region::new();
    main_region.append_block(main_block);

    let main_func_type = FunctionType::new(context, &[any_op_type], &[]);
    let main_arg_attrs =
        Attribute::parse(context, "[{transform.consumed}]").expect("parse main arg_attrs");

    let main_named_seq = OperationBuilder::new("transform.named_sequence", location)
        .add_attributes(&[
            (
                Identifier::new(context, "sym_name"),
                StringAttribute::new(context, "__transform_main").into(),
            ),
            (
                Identifier::new(context, "function_type"),
                TypeAttribute::new(main_func_type.into()).into(),
            ),
            (Identifier::new(context, "arg_attrs"), main_arg_attrs),
            (
                Identifier::new(context, "sym_visibility"),
                StringAttribute::new(context, "private").into(),
            ),
        ])
        .add_regions([main_region])
        .build()
        .expect("build transform.named_sequence @__transform_main (vectorize-only)");
    module.body().append_operation(main_named_seq);
}

// ── Public wrappers for graph_builder ─────────────────────────────────────────

/// Public wrapper so `graph_builder` can call the AOT object emitter.
pub(crate) fn emit_object_files_pub(
    module: &melior::ir::Module,
    output_dir: &std::path::Path,
    base_name: &str,
    label: &str,
) -> Result<Vec<std::path::PathBuf>, CompileError> {
    emit_object_files(module, output_dir, base_name, label)
}

/// Public wrapper so `graph_builder` can call the shared lib linker.
pub(crate) fn link_shared_lib_pub(
    obj_paths: &[std::path::PathBuf],
    so_path: &std::path::Path,
) -> Result<(), CompileError> {
    link_shared_lib(obj_paths, so_path)
}
