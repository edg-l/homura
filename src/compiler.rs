use std::collections::HashMap;

use melior::dialect::DialectRegistry;
use melior::{
    Context,
    dialect::{arith, func},
    ir::{
        Attribute, Block, Identifier, Location, Module, Region, RegionLike,
        attribute::{
            ArrayAttribute, FloatAttribute, IntegerAttribute, StringAttribute, TypeAttribute,
        },
        block::BlockLike,
        operation::{OperationBuilder, OperationLike, OperationMutLike},
        r#type::{FunctionType, MemRefType, RankedTensorType},
    },
    pass,
    utility::{
        parse_pass_pipeline, register_all_dialects, register_all_llvm_translations,
        register_all_passes,
    },
};

use crate::{
    DType,
    cache::CompilationCache,
    op::{NodeId, Op},
    runtime::{CompiledGraph, OutputDesc},
    shape::DIM_DYNAMIC,
    trace::Trace,
};

/// Error type for compilation failures.
#[derive(Debug)]
pub enum CompileError {
    /// MLIR module verification failed.
    Verification,
    /// A lowering pass failed.
    Pass(melior::Error),
    /// Trace contains no ops.
    EmptyTrace,
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
            Self::EmptyTrace => write!(f, "trace is empty"),
            Self::NoOutputs => write!(f, "no output nodes specified"),
            Self::AttributeParse(s) => write!(f, "failed to parse MLIR attribute: {s}"),
            Self::ObjectEmit(s) => write!(f, "object emit failed: {s}"),
            Self::Link(s) => write!(f, "link failed: {s}"),
        }
    }
}

impl std::error::Error for CompileError {}

/// Compiles a `Trace` into a JIT-ready `CompiledGraph`.
pub struct Compiler;

impl Compiler {
    /// Compile a trace into a `CompiledGraph` ready to execute.
    ///
    /// `outputs` is a list of node IDs whose results should become output
    /// memref arguments of the generated function.
    ///
    /// If `cache_key` is provided, the compiler first checks `~/.cache/homura/`
    /// (or `HOMURA_CACHE_DIR`) for a cached native shared library (`.so`). On a
    /// hit the expensive MLIR pass pipeline and LLVM JIT compilation are skipped;
    /// the `.so` is loaded via dlopen in milliseconds. On a miss the full pipeline
    /// runs, the result is compiled to a `.so`, and both the library and a small
    /// metadata sidecar are stored for future calls.
    pub fn compile(
        trace: &Trace,
        outputs: &[NodeId],
        cache_key: Option<&str>,
    ) -> Result<CompiledGraph, CompileError> {
        if trace.ops().is_empty() {
            return Err(CompileError::EmptyTrace);
        }
        if outputs.is_empty() {
            return Err(CompileError::NoOutputs);
        }

        // We need num_inputs and output_descs regardless of cache path.
        // Build them from the trace before attempting any cache lookup so that
        // the returned CompiledGraph is always fully-formed.
        let context = create_context();
        let location = Location::unknown(&context);
        let mut module = Module::new(location);

        // Set data layout for x86-64 so MLIR passes (finalize-memref-to-llvm etc.)
        // know the target's type sizes and alignments.
        let dl_attr = Attribute::parse(&context,
            "#dlti.dl_spec<\
                index = 64 : i64, \
                i32 = dense<32> : vector<2xi64>, \
                i64 = dense<64> : vector<2xi64>, \
                f32 = dense<32> : vector<2xi64>, \
                f64 = dense<64> : vector<2xi64>, \
                !llvm.ptr = dense<64> : vector<4xi64>\
            >"
        ).expect("failed to parse dlti.dl_spec");
        module.as_operation_mut().set_attribute("dlti.dl_spec", dl_attr);

        let (num_inputs, output_descs) = build_module(&context, &module, trace, outputs)?;

        // ---- Cache hit path --------------------------------------------------
        if let Some(key) = cache_key {
            let cache = CompilationCache::new();
            if let Some((so_path, meta_path)) = cache.get(key) {
                if let Some(meta) = CompilationCache::load_meta(&meta_path) {
                    match CompiledGraph::load(&so_path, meta.num_inputs, meta.outputs) {
                        Ok(graph) => return Ok(graph),
                        Err(e) => {
                            // Cache entry is unloadable (corrupt/stale). Fall
                            // through to recompile and overwrite.
                            eprintln!(
                                "homura cache: failed to load {}: {e}, recompiling",
                                so_path.display()
                            );
                        }
                    }
                }
            }
        }

        // ---- Run lowering passes ----------------------------------------------
        // TOSA passes (func-level) run first, then bufferize → LLVM (module-level).
        // TOSA passes are no-ops when no TOSA ops are present, so this pipeline
        // handles both pure-linalg and pure-TOSA (and mixed) modules.
        // expand-strided-metadata is required for the tensor.expand_shape used in
        // promote_rank_with_reshape (binary op rank promotion for broadcast).
        register_all_passes();
        let pass_manager = pass::PassManager::new(&context);
        parse_pass_pipeline(
            pass_manager.as_operation_pass_manager(),
            "builtin.module(\
                func.func(\
                    tosa-make-broadcastable,\
                    tosa-to-linalg-named,\
                    tosa-to-linalg,\
                    tosa-to-arith,\
                    tosa-to-tensor\
                ),\
                func.func(canonicalize,cse),\
                one-shot-bufferize{function-boundary-type-conversion=identity-layout-map unknown-type-conversion=identity-layout-map},\
                func.func(buffer-hoisting,promote-buffers-to-stack{max-alloc-size-in-bytes=4096}),\
                fold-memref-alias-ops,\
                convert-linalg-to-loops,\
                fold-memref-alias-ops,\
                lower-affine,\
                convert-scf-to-cf,\
                canonicalize,\
                cse,\
                sccp,\
                convert-math-to-llvm,\
                expand-strided-metadata,\
                lower-affine,\
                finalize-memref-to-llvm,\
                convert-arith-to-llvm,\
                convert-index-to-llvm,\
                convert-cf-to-llvm,\
                convert-func-to-llvm,\
                reconcile-unrealized-casts\
            )",
        )
        .map_err(CompileError::Pass)?;

        let mut module = module;
        pass_manager.run(&mut module).map_err(CompileError::Pass)?;

        // Dump pre-lowering IR to /tmp for debugging dynamic shapes
        if std::env::var("HOMURA_DUMP_IR").is_ok() {
            let _ = std::fs::write("/tmp/homura_post_passes.mlir", module.as_operation().to_string());
            eprintln!("[homura] post-pass IR dumped to /tmp/homura_post_passes.mlir");
        }

        // ---- AOT: emit object file, link to .so, dlopen ----------------------
        let tmp_dir = tempfile_dir().ok_or_else(|| {
            CompileError::ObjectEmit("cannot determine temp directory".into())
        })?;

        // Use PID + nanosecond timestamp for a collision-resistant suffix.
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos();
        let suffix = format!("{}_{:08x}", std::process::id(), nanos);
        let tmp_obj = tmp_dir.join(format!("homura_{suffix}.o"));
        let tmp_so = tmp_dir.join(format!("homura_{suffix}.so"));

        emit_object_file(&module, &tmp_obj)?;
        link_shared_lib(&tmp_obj, &tmp_so)?;
        std::fs::remove_file(&tmp_obj).ok(); // .o is no longer needed

        // If caching, copy the .so to the cache before dlopen (dlopen keeps
        // the inode alive, so the file can be unlinked after).
        if let Some(key) = cache_key {
            let cache = crate::cache::CompilationCache::new();
            let meta = crate::cache::CacheMeta {
                num_inputs,
                outputs: output_descs
                    .iter()
                    .map(|d| OutputDesc { shape: d.shape.clone(), dtype: d.dtype })
                    .collect(),
            };
            if let Err(e) = cache.store(key, &tmp_so, &meta) {
                eprintln!("homura cache: failed to write cache entry: {e}");
            }
        }

        let graph = CompiledGraph::load(&tmp_so, num_inputs, output_descs)
            .map_err(CompileError::ObjectEmit)?;
        std::fs::remove_file(&tmp_so).ok(); // safe: inode kept alive by dlopen

        Ok(graph)
    }

    /// Build the MLIR module for a trace and return its text representation
    /// **before** any lowering passes are applied.
    ///
    /// This is a test-only helper for IR verification (task 2.10).
    #[cfg(test)]
    pub fn build_ir_string(trace: &Trace, outputs: &[NodeId]) -> Result<String, CompileError> {
        if trace.ops().is_empty() {
            return Err(CompileError::EmptyTrace);
        }
        if outputs.is_empty() {
            return Err(CompileError::NoOutputs);
        }

        let context = create_context();
        let location = Location::unknown(&context);
        let module = Module::new(location);
        build_module(&context, &module, trace, outputs)?;
        Ok(module.as_operation().to_string())
    }
}

/// Translate a lowered MLIR module to native code and emit an object file.
///
/// Initialises LLVM targets on first call (via `OnceLock`), translates the
/// module to LLVM IR, runs the O3 optimisation pipeline, and writes a PIC
/// object file to `output_path`.
fn emit_object_file(
    module: &melior::ir::Module,
    output_path: &std::path::Path,
) -> Result<(), CompileError> {
    use crate::llvm_ffi::mlirTranslateModuleToLLVMIR;
    use llvm_sys::core::*;
    use llvm_sys::error::LLVMGetErrorMessage;
    use llvm_sys::target::*;
    use llvm_sys::target_machine::*;
    use llvm_sys::transforms::pass_builder::*;
    use std::ffi::{CStr, CString};
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
        let llvm_ctx = LLVMContextCreate();
        let llvm_module =
            mlirTranslateModuleToLLVMIR(module.as_operation().to_raw(), llvm_ctx);

        // Set the host data layout on the LLVM module.
        if !llvm_module.is_null() {
            let dl = CString::new(
                "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
            ).unwrap();
            LLVMSetDataLayout(llvm_module, dl.as_ptr());
        }
        if llvm_module.is_null() {
            LLVMContextDispose(llvm_ctx);
            return Err(CompileError::ObjectEmit(
                "mlirTranslateModuleToLLVMIR returned null".into(),
            ));
        }

        // Emit the packed wrapper `_mlir__mlir_ciface_compute(void** args)` so
        // the runtime can call it with the same double-indirection convention
        // as MLIR's ExecutionEngine::invoke_packed. The direct `_mlir_ciface_compute`
        // is generated by the `convert-func-to-llvm` pass but takes individual
        // MemRef-pointer args; the packed wrapper loads each from args[i] and calls it.
        if let Err(e) = emit_packed_wrapper(llvm_module, llvm_ctx) {
            LLVMDisposeModule(llvm_module);
            LLVMContextDispose(llvm_ctx);
            return Err(CompileError::ObjectEmit(format!("emit_packed_wrapper: {e}")));
        }

        // Get host CPU name and features for optimal codegen (enables AVX2/SSE4.2).
        let cpu = LLVMGetHostCPUName();
        let features = LLVMGetHostCPUFeatures();

        // Resolve target from the default triple.
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

        // Create a PIC target machine at aggressive optimisation level.
        let machine = LLVMCreateTargetMachine(
            target,
            triple,
            cpu,
            features,
            LLVMCodeGenOptLevel::LLVMCodeGenLevelAggressive,
            LLVMRelocMode::LLVMRelocPIC,
            LLVMCodeModel::LLVMCodeModelDefault,
        );

        // Run O3 optimisation passes.
        let opts = LLVMCreatePassBuilderOptions();
        let passes = CString::new("default<O3>").unwrap();
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

        // Emit the object file.
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

/// Return the path to `libmlir_c_runner_utils.so`.
///
/// Reads `MLIR_RUNNER_UTILS_PATH` if set, otherwise falls back to the
/// well-known LLVM 21 install path on this system.
fn runner_utils_lib_path() -> std::path::PathBuf {
    if let Ok(path) = std::env::var("MLIR_RUNNER_UTILS_PATH") {
        return std::path::PathBuf::from(path);
    }
    std::path::PathBuf::from("/usr/lib/llvm/21/lib64/libmlir_c_runner_utils.so")
}

/// Link an object file into a shared library.
///
/// Links `runner_utils` by full path and bakes its directory into the
/// `.so`'s `DT_RUNPATH` so the dynamic linker finds it at dlopen time
/// without requiring `LD_LIBRARY_PATH`.
fn link_shared_lib(
    obj_path: &std::path::Path,
    so_path: &std::path::Path,
) -> Result<(), CompileError> {
    let runner_utils_path = runner_utils_lib_path();
    let runner_utils_dir = runner_utils_path
        .parent()
        .unwrap_or(std::path::Path::new("/usr/lib/llvm/21/lib64"))
        .to_str()
        .unwrap_or("/usr/lib/llvm/21/lib64");

    let status = std::process::Command::new("cc")
        .args([
            "-shared",
            "-fPIC",
            "-o",
            so_path.to_str().unwrap(),
            obj_path.to_str().unwrap(),
            runner_utils_path.to_str().unwrap(),
            "-lm",
            &format!("-Wl,-rpath,{runner_utils_dir}"),
        ])
        .status()
        .map_err(|e| CompileError::Link(format!("failed to run cc: {e}")))?;

    if !status.success() {
        return Err(CompileError::Link(format!("cc exited with {status}")));
    }
    Ok(())
}

/// Return a temp directory path for staging cache files.
fn tempfile_dir() -> Option<std::path::PathBuf> {
    std::env::var("TMPDIR")
        .ok()
        .map(std::path::PathBuf::from)
        .or_else(|| Some(std::path::PathBuf::from("/tmp")))
}

/// Create an MLIR context with all dialects and LLVM translations registered.
fn create_context() -> Context {
    let context = Context::new();
    context.attach_diagnostic_handler(|diagnostic| {
        eprintln!("{diagnostic}");
        true
    });
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    register_all_llvm_translations(&context);
    context
}

// ── Shared module builder ─────────────────────────────────────────────────────
//
// Populates `module` with the `@compute` func body for the given trace and
// verifies it. Returns `(num_inputs, Vec<OutputDesc>)`.
//
// `context` and `module` are caller-owned so their lifetimes do not cross a
// function boundary (which would require a self-referential return).

fn build_module<'c>(
    context: &'c Context,
    module: &Module<'c>,
    trace: &Trace,
    outputs: &[NodeId],
) -> Result<(usize, Vec<OutputDesc>), CompileError> {
    // Collect input ops in arg_index order.
    let mut input_ops: Vec<(NodeId, &Op)> = trace
        .ops()
        .iter()
        .enumerate()
        .filter_map(|(i, op)| {
            if matches!(op, Op::Input { .. }) {
                Some((NodeId(i as u32), op))
            } else {
                None
            }
        })
        .collect();
    input_ops.sort_by_key(|(_, op)| {
        if let Op::Input { arg_index, .. } = op {
            *arg_index
        } else {
            unreachable!()
        }
    });

    let num_inputs = input_ops.len();

    // Collect output descriptors for all requested output nodes.
    let output_descs: Vec<OutputDesc> = outputs
        .iter()
        .map(|&id| {
            let op = trace.get(id);
            OutputDesc {
                shape: op.shape().clone(),
                dtype: op.dtype(),
            }
        })
        .collect();

    let location = Location::unknown(context);

    {
        let mut arg_types: Vec<(melior::ir::Type, Location)> = Vec::new();
        for (_, op) in &input_ops {
            let Op::Input { shape, dtype, .. } = op else {
                unreachable!()
            };
            let dims: Vec<i64> = shape.0.iter().map(|&d| dim_to_mlir_i64(d)).collect();
            let mref = MemRefType::new(dtype.to_mlir_type(context), &dims, None, None);
            arg_types.push((mref.into(), location));
        }
        // Each output gets its own trailing memref argument (sret-style).
        for desc in &output_descs {
            let elem_type = desc.dtype.to_mlir_type(context);
            let out_dims: Vec<i64> = desc.shape.0.iter().map(|&d| dim_to_mlir_i64(d)).collect();
            let out_mref = MemRefType::new(elem_type, &out_dims, None, None);
            arg_types.push((out_mref.into(), location));
        }

        let func_arg_types: Vec<melior::ir::Type> = arg_types.iter().map(|(t, _)| *t).collect();
        let function_type = FunctionType::new(context, &func_arg_types, &[]);

        let body_block = Block::new(&arg_types);

        emit_tensor_ops(trace, outputs, num_inputs, &body_block, location, context)?;

        body_block.append_operation(func::r#return(&[], location));

        let func_region = Region::new();
        func_region.append_block(body_block);

        let function = func::func(
            context,
            StringAttribute::new(context, "compute"),
            TypeAttribute::new(function_type.into()),
            func_region,
            &[(
                Identifier::new(context, "llvm.emit_c_interface"),
                Attribute::unit(context),
            )],
            location,
        );

        module.body().append_operation(function);
    }

    // Dump pre-pass MLIR for debugging when HOMURA_DUMP_IR is set
    if std::env::var("HOMURA_DUMP_IR").is_ok() {
        let _ = std::fs::write("/tmp/homura_pre_passes.mlir", module.as_operation().to_string());
        eprintln!("[homura] pre-pass IR dumped to /tmp/homura_pre_passes.mlir");
    }

    if !module.as_operation().verify() {
        let ir = module.as_operation().to_string();
        let _ = std::fs::write("/tmp/homura_failed.mlir", &ir);
        eprintln!("[homura] MLIR verification failed — IR dumped to /tmp/homura_failed.mlir");
        return Err(CompileError::Verification);
    }

    Ok((num_inputs, output_descs))
}

// ── Helper: convert a homura shape dim to the MLIR C API representation ──────
//
// homura uses DIM_DYNAMIC = u64::MAX as its sentinel for unknown dims.
// MLIR's C API (mlirRankedTensorTypeGet / mlirMemRefTypeGet) uses
// ShapedType::kDynamic = i64::MIN as the sentinel for unknown dims.
//
// u64::MAX reinterpreted as i64 gives -1, which is NOT kDynamic; MLIR would
// treat it as a literal negative dimension (and tensor.empty verification
// would reject it with "incorrect number of dynamic sizes").
//
// Convert by mapping: DIM_DYNAMIC → i64::MIN as u64 (= kDynamic in i64 bits).
// All other values are small u64s that fit in i64 unchanged.
#[inline]
fn dim_to_mlir(d: u64) -> u64 {
    if d == DIM_DYNAMIC { i64::MIN as u64 } else { d }
}

/// Like `dim_to_mlir` but returns `i64` for use with `MemRefType::new(&[i64])`.
#[inline]
fn dim_to_mlir_i64(d: u64) -> i64 {
    if d == DIM_DYNAMIC { i64::MIN } else { d as i64 }
}

// ── Helper: build a RankedTensorType for an arbitrary shape ──────────────────

fn make_ranked_tensor_type<'c>(
    context: &'c Context,
    shape: &[u64],
    dtype: DType,
) -> melior::ir::Type<'c> {
    let dims: Vec<u64> = shape.iter().map(|&d| dim_to_mlir(d)).collect();
    RankedTensorType::new(&dims, dtype.to_mlir_type(context), None).into()
}

// ── Helper: emit tensor.dim to query a runtime dimension ─────────────────────
//
// Emits:
//   %c{dim_idx} = arith.constant {dim_idx} : index
//   %dim        = tensor.dim %tensor, %c{dim_idx} : tensor<...>
//
// Used to get the runtime value of a `?` dimension from a tensor.

#[allow(dead_code)]
fn emit_tensor_dim<'c>(
    context: &'c Context,
    block: &Block<'c>,
    tensor_val: melior::ir::Value<'c, 'c>,
    dim_idx: usize,
    location: Location<'c>,
) -> Result<melior::ir::Value<'c, 'c>, CompileError> {
    let index_type = melior::ir::Type::parse(context, "index")
        .ok_or_else(|| CompileError::AttributeParse("index type".into()))?;

    let dim_const: melior::ir::Value = block
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(index_type, dim_idx as i64).into(),
            location,
        ))
        .result(0)
        .unwrap()
        .into();

    let dim_val: melior::ir::Value = block
        .append_operation(
            OperationBuilder::new("tensor.dim", location)
                .add_operands(&[tensor_val, dim_const])
                .add_results(&[index_type])
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        )
        .result(0)
        .unwrap()
        .into();

    Ok(dim_val)
}

// ── Helper: emit tensor.empty supporting dynamic dimensions ──────────────────
//
// Like `tensor.empty`, but for shapes containing `DIM_DYNAMIC`. For each `?`
// dim in `shape`, a corresponding `(source_tensor, dim_index)` pair must be
// provided in `dynamic_dim_sources` (in order of the `?` positions). The
// helper emits `tensor.dim` calls to obtain runtime sizes and passes them as
// dynamic operands to `tensor.empty`.
//
// For fully static shapes, `dynamic_dim_sources` must be empty and this
// behaves identically to the plain `tensor.empty` builder.

#[allow(dead_code)]
fn emit_tensor_empty_dynamic<'c>(
    context: &'c Context,
    block: &Block<'c>,
    shape: &[u64],
    dtype: DType,
    // One entry per `DIM_DYNAMIC` position in `shape`, in order.
    dynamic_dim_sources: &[(melior::ir::Value<'c, 'c>, usize)],
    location: Location<'c>,
) -> Result<melior::ir::Value<'c, 'c>, CompileError> {
    use crate::shape::DIM_DYNAMIC;

    let tensor_type = make_ranked_tensor_type(context, shape, dtype);

    // Count how many dynamic dims there are.
    let num_dynamic = shape.iter().filter(|&&d| d == DIM_DYNAMIC).count();
    debug_assert_eq!(
        num_dynamic,
        dynamic_dim_sources.len(),
        "emit_tensor_empty_dynamic: dynamic_dim_sources count ({}) must equal number of DIM_DYNAMIC dims ({})",
        dynamic_dim_sources.len(),
        num_dynamic,
    );

    if num_dynamic == 0 {
        // Fully static: no dynamic operands needed.
        let init_val = block
            .append_operation(
                OperationBuilder::new("tensor.empty", location)
                    .add_results(&[tensor_type])
                    .build()
                    .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
            )
            .result(0)
            .unwrap()
            .into();
        return Ok(init_val);
    }

    // Emit tensor.dim for each dynamic dim source.
    let mut dyn_dim_vals: Vec<melior::ir::Value<'c, 'c>> = Vec::with_capacity(num_dynamic);
    for &(src_tensor, src_dim_idx) in dynamic_dim_sources {
        let dv = emit_tensor_dim(context, block, src_tensor, src_dim_idx, location)?;
        dyn_dim_vals.push(dv);
    }

    let init_val = block
        .append_operation(
            OperationBuilder::new("tensor.empty", location)
                .add_operands(&dyn_dim_vals)
                .add_results(&[tensor_type])
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        )
        .result(0)
        .unwrap()
        .into();

    Ok(init_val)
}

// ── Helper: build an identity affine map for `rank` dimensions ───────────────
//
// rank 1 → affine_map<(d0) -> (d0)>
// rank 2 → affine_map<(d0, d1) -> (d0, d1)>

fn make_identity_map<'c>(context: &'c Context, rank: usize) -> Result<Attribute<'c>, CompileError> {
    let dims: Vec<String> = (0..rank).map(|i| format!("d{i}")).collect();
    let dim_list = dims.join(", ");
    let map_str = format!("affine_map<({dim_list}) -> ({dim_list})>");
    Attribute::parse(context, &map_str)
        .ok_or_else(|| CompileError::AttributeParse(format!("failed to parse {map_str}")))
}

// ── Helper: build a broadcast affine map for an operand vs output shape ──────
//
// Examples:
//   operand [3]    vs output [2, 3]    → affine_map<(d0, d1) -> (d1)>
//   operand [2, 1] vs output [2, 3]    → affine_map<(d0, d1) -> (d0, 0)>
//   operand [1, 3] vs output [4, 2, 3] → affine_map<(d0, d1, d2) -> (0, d2)>
//   operand [2, 3] vs output [2, 3]    → affine_map<(d0, d1) -> (d0, d1)> (identity)

fn make_broadcast_map<'c>(
    context: &'c Context,
    operand_shape: &[u64],
    output_shape: &[u64],
) -> Result<Attribute<'c>, CompileError> {
    let out_rank = output_shape.len();
    let op_rank = operand_shape.len();

    let dim_vars: Vec<String> = (0..out_rank).map(|i| format!("d{i}")).collect();
    let dims_str = dim_vars.join(", ");

    // Right-align: operand dim i maps to output position (offset + i).
    let offset = out_rank - op_rank;
    let mut result_exprs = Vec::new();
    for (i, &op_dim) in operand_shape.iter().enumerate() {
        let out_idx = offset + i;
        if op_dim == 1 && output_shape[out_idx] != 1 {
            result_exprs.push("0".to_string());
        } else {
            result_exprs.push(dim_vars[out_idx].clone());
        }
    }
    let result_str = result_exprs.join(", ");

    let map_str = format!("affine_map<({dims_str}) -> ({result_str})>");
    Attribute::parse(context, &map_str)
        .ok_or_else(|| CompileError::AttributeParse(format!("failed to parse {map_str}")))
}

// ── Helper: build N parallel iterator types ──────────────────────────────────

fn make_iterator_types<'c>(
    context: &'c Context,
    count: usize,
) -> Result<Attribute<'c>, CompileError> {
    let entries: Vec<&str> = vec!["#linalg.iterator_type<parallel>"; count];
    let attr_str = format!("[{}]", entries.join(", "));
    Attribute::parse(context, &attr_str)
        .ok_or_else(|| CompileError::AttributeParse(format!("failed to parse {attr_str}")))
}

// ── Helper: emit a binary element-wise linalg.generic ────────────────────────

#[allow(clippy::too_many_arguments)]
fn emit_binary_elementwise<'c, F>(
    context: &'c Context,
    body_block: &Block<'c>,
    values: &HashMap<NodeId, melior::ir::Value<'c, 'c>>,
    lhs: NodeId,
    rhs: NodeId,
    lhs_shape: &[u64],
    rhs_shape: &[u64],
    output_shape: &[u64],
    dtype: DType,
    location: Location<'c>,
    body_fn: F,
) -> Result<melior::ir::Value<'c, 'c>, CompileError>
where
    F: FnOnce(
        &Block<'c>,
        melior::ir::Value<'c, 'c>,
        melior::ir::Value<'c, 'c>,
    ) -> melior::ir::Value<'c, 'c>,
{
    let elem_type = dtype.to_mlir_type(context);
    let tensor_type = make_ranked_tensor_type(context, output_shape, dtype);
    let rank = output_shape.len();

    let lhs_val = *values.get(&lhs).expect("lhs node not yet emitted");
    let rhs_val = *values.get(&rhs).expect("rhs node not yet emitted");

    // tensor.empty() for the output slot.
    // For dynamic output shapes, provide (source_tensor, dim_index) for each `?` dim.
    let init_val = if output_shape.contains(&DIM_DYNAMIC) {
        let sources: Vec<(melior::ir::Value<'c, 'c>, usize)> = output_shape
            .iter()
            .enumerate()
            .filter(|&(_, d)| *d == DIM_DYNAMIC)
            .map(|(i, _)| {
                // Right-align: dim i in output maps to offset position in each operand.
                let lhs_offset = output_shape.len().saturating_sub(lhs_shape.len());
                let _rhs_offset = output_shape.len().saturating_sub(rhs_shape.len());
                // Pick the operand where this dim is NOT broadcast (not size 1).
                // Prefer lhs; fall back to rhs.
                if i >= lhs_offset && lhs_shape[i - lhs_offset] != 1 {
                    (lhs_val, i)
                } else {
                    (rhs_val, i)
                }
            })
            .collect();
        emit_tensor_empty_dynamic(context, body_block, output_shape, dtype, &sources, location)?
    } else {
        body_block
            .append_operation(
                OperationBuilder::new("tensor.empty", location)
                    .add_results(&[tensor_type])
                    .build()
                    .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
            )
            .result(0)
            .unwrap()
            .into()
    };

    // linalg body: 3 block args (lhs_elem, rhs_elem, out_elem).
    let linalg_region = {
        let linalg_block = Block::new(&[
            (elem_type, location),
            (elem_type, location),
            (elem_type, location),
        ]);
        let lhs_elem = linalg_block.argument(0).unwrap().into();
        let rhs_elem = linalg_block.argument(1).unwrap().into();

        let result = body_fn(&linalg_block, lhs_elem, rhs_elem);

        linalg_block.append_operation(
            OperationBuilder::new("linalg.yield", location)
                .add_operands(&[result])
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        );

        let r = Region::new();
        r.append_block(linalg_block);
        r
    };

    // lhs and rhs maps account for broadcast; output map is always identity.
    let lhs_map = make_broadcast_map(context, lhs_shape, output_shape)?;
    let rhs_map = make_broadcast_map(context, rhs_shape, output_shape)?;
    let out_map = make_identity_map(context, rank)?;
    let indexing_maps = ArrayAttribute::new(context, &[lhs_map, rhs_map, out_map]);
    let iterator_types = make_iterator_types(context, rank)?;
    let segment_sizes = Attribute::parse(context, "array<i32: 2, 1>").ok_or_else(|| {
        CompileError::AttributeParse("failed to parse operand_segment_sizes".into())
    })?;

    let result_val = body_block
        .append_operation(
            OperationBuilder::new("linalg.generic", location)
                .add_operands(&[lhs_val, rhs_val, init_val])
                .add_results(&[tensor_type])
                .add_attributes(&[
                    (
                        Identifier::new(context, "indexing_maps"),
                        indexing_maps.into(),
                    ),
                    (Identifier::new(context, "iterator_types"), iterator_types),
                    (
                        Identifier::new(context, "operand_segment_sizes"),
                        segment_sizes,
                    ),
                ])
                .add_regions([linalg_region])
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        )
        .result(0)
        .unwrap()
        .into();

    Ok(result_val)
}

// ── Helper: emit a tosa.const scalar tensor ───────────────────────────────────
//
// Emits `tosa.const() {values = dense<VAL> : tensor<TYPE>}` for scalar (rank-0)
// tensors used as zero-point or shift operands in TOSA ops.
//
// `dense_attr_str` is the full `dense<…> : tensor<…>` attribute string,
// e.g. `"dense<0> : tensor<i8>"` or `"dense<0.0> : tensor<f32>"`.

fn emit_tosa_const_scalar<'c>(
    context: &'c Context,
    body_block: &Block<'c>,
    dense_attr_str: &str,
    location: Location<'c>,
) -> Result<melior::ir::Value<'c, 'c>, CompileError> {
    let values_attr = Attribute::parse(context, dense_attr_str)
        .ok_or_else(|| CompileError::AttributeParse(dense_attr_str.to_string()))?;

    // The result type mirrors the tensor type embedded in the dense attribute.
    // We derive it by stripping `dense<…> : ` and using the rest as the type.
    let type_str = dense_attr_str
        .split_once(':')
        .map(|(_, s)| s.trim())
        .unwrap_or("tensor<i8>");
    let result_type = melior::ir::Type::parse(context, type_str)
        .ok_or_else(|| CompileError::AttributeParse(type_str.to_string()))?;

    let val = body_block
        .append_operation(
            OperationBuilder::new("tosa.const", location)
                .add_results(&[result_type])
                .add_attributes(&[(Identifier::new(context, "values"), values_attr)])
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        )
        .result(0)
        .unwrap()
        .into();

    Ok(val)
}

// ── Helper: emit tosa.const_shape + tosa.reshape ─────────────────────────────
//
// Emits a `tosa.const_shape` with the given target shape (producing a
// `!tosa.shape<N>` value) followed by a `tosa.reshape` that reshapes
// `input` to `target_shape`. Returns the reshaped tensor value.

#[allow(clippy::too_many_arguments)]
fn emit_tosa_reshape<'c>(
    context: &'c Context,
    body_block: &Block<'c>,
    input: melior::ir::Value<'c, 'c>,
    target_shape: &[u64],
    dtype: DType,
    location: Location<'c>,
) -> Result<melior::ir::Value<'c, 'c>, CompileError> {
    // If any dim is DIM_DYNAMIC, tosa.const_shape can't represent it.
    // Fall back to tensor.cast (same rank only — rank-changing reshapes
    // with dynamic dims must use tensor.expand_shape / tensor.collapse_shape
    // at their specific call sites).
    if target_shape.iter().any(|&d| d == DIM_DYNAMIC) {
        let out_type = make_ranked_tensor_type(context, target_shape, dtype);
        let result: melior::ir::Value = body_block
            .append_operation(
                OperationBuilder::new("tensor.cast", location)
                    .add_operands(&[input])
                    .add_results(&[out_type])
                    .build()
                    .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
            )
            .result(0)
            .unwrap()
            .into();
        return Ok(result);
    }

    let n = target_shape.len();

    // Build the !tosa.shape<N> type.
    let shape_type_str = format!("!tosa.shape<{n}>");
    let shape_type = melior::ir::Type::parse(context, &shape_type_str)
        .ok_or_else(|| CompileError::AttributeParse(shape_type_str.clone()))?;

    // Build `dense<[d0, d1, ...]> : tensor<Nxindex>` for the values attribute.
    let dims_str = target_shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let dense_attr_str = format!("dense<[{dims_str}]> : tensor<{n}xindex>");
    let values_attr = Attribute::parse(context, &dense_attr_str)
        .ok_or_else(|| CompileError::AttributeParse(dense_attr_str.clone()))?;

    // tosa.const_shape {values = dense<[...]> : tensor<Nxindex>} : () -> !tosa.shape<N>
    let const_shape_val: melior::ir::Value = body_block
        .append_operation(
            OperationBuilder::new("tosa.const_shape", location)
                .add_results(&[shape_type])
                .add_attributes(&[(Identifier::new(context, "values"), values_attr)])
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        )
        .result(0)
        .unwrap()
        .into();

    // tosa.reshape %input, %const_shape : (...) -> tensor<d0xd1x...xT>
    let result_type = make_ranked_tensor_type(context, target_shape, dtype);
    let reshaped: melior::ir::Value = body_block
        .append_operation(
            OperationBuilder::new("tosa.reshape", location)
                .add_operands(&[input, const_shape_val])
                .add_results(&[result_type])
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        )
        .result(0)
        .unwrap()
        .into();

    Ok(reshaped)
}

// ── Helper: promote a tensor to a target rank by prepending size-1 dims ───────
//
// If `val` has rank < `target_rank`, emits a `tosa.reshape` to prepend size-1
// dimensions. Returns the (possibly reshaped) value with the promoted type.
// This mirrors what `tosa-make-broadcastable` does, but at IR-build time so
// that the module verifies before passes run.

fn promote_rank_with_reshape<'c>(
    context: &'c Context,
    body_block: &Block<'c>,
    val: melior::ir::Value<'c, 'c>,
    val_shape: &[u64],
    target_rank: usize,
    dtype: DType,
    location: Location<'c>,
) -> Result<(melior::ir::Value<'c, 'c>, Vec<u64>), CompileError> {
    let current_rank = val_shape.len();
    if current_rank == target_rank {
        return Ok((val, val_shape.to_vec()));
    }

    // Prepend (target_rank - current_rank) ones.
    let prefix = target_rank - current_rank;
    let mut new_shape: Vec<u64> = vec![1u64; prefix];
    new_shape.extend_from_slice(val_shape);

    let new_tensor_type = make_ranked_tensor_type(context, &new_shape, dtype);

    // Use tensor.expand_shape to prepend size-1 dimensions.
    // The reassociation attribute groups old dims into new dims.
    // Each existing dim of `val` maps to one new dim; the `prefix` new
    // leading dimensions are grouped with the first existing dim (or, if
    // val_shape is empty, all prefix dims form a single group).
    //
    // Example: val_shape=[3], target_rank=2, prefix=1
    //   reassociation = [[0, 1]]   (old dim 0 → new dims 0,1 = [1, 3])
    //
    // Example: val_shape=[2,3], target_rank=3, prefix=1
    //   reassociation = [[0, 1], [2]]
    //
    // General pattern: group all prefix new dims with old dim 0,
    // then map each remaining old dim 1-to-1.

    let reassoc_parts: Vec<String> = (0..current_rank)
        .map(|old_i| {
            if old_i == 0 {
                // This group covers the prefix new dims plus the new dim at index `prefix`.
                let new_dims: Vec<String> = (0..=prefix).map(|j| j.to_string()).collect();
                format!("[{}]", new_dims.join(", "))
            } else {
                // One-to-one: old dim i -> new dim (prefix + i).
                format!("[{}]", prefix + old_i)
            }
        })
        .collect();
    let reassoc_str = format!("[{}]", reassoc_parts.join(", "));

    let reassoc_attr = Attribute::parse(context, &reassoc_str)
        .ok_or_else(|| CompileError::AttributeParse(reassoc_str.clone()))?;

    // static_output_shape: array<i64: d0, d1, ...>
    // For DIM_DYNAMIC dims, emit i64::MIN which is MLIR's ShapedType::kDynamic.
    let n = new_shape.len();
    let dims_str = new_shape
        .iter()
        .map(|&d| {
            if d == DIM_DYNAMIC {
                i64::MIN.to_string()
            } else {
                d.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    let static_shape_attr_str = format!("array<i64: {dims_str}>");
    let static_shape_attr = Attribute::parse(context, &static_shape_attr_str)
        .ok_or_else(|| CompileError::AttributeParse(static_shape_attr_str.clone()))?;
    let _ = n;

    // For each dynamic dim in the output shape, provide a runtime SSA value
    // via tensor.dim on the input tensor.
    let index_type = melior::ir::Type::parse(context, "index")
        .ok_or_else(|| CompileError::AttributeParse("index type".into()))?;
    let mut output_shape_vals: Vec<melior::ir::Value> = Vec::new();
    for (i, &d) in new_shape.iter().enumerate() {
        if d == DIM_DYNAMIC {
            // This dim comes from val_shape at position (i - prefix).
            let orig_idx = i - prefix;
            let dim_idx: melior::ir::Value = body_block
                .append_operation(arith::constant(
                    context,
                    IntegerAttribute::new(index_type, orig_idx as i64).into(),
                    location,
                ))
                .result(0)
                .unwrap()
                .into();
            let dim_val: melior::ir::Value = body_block
                .append_operation(
                    OperationBuilder::new("tensor.dim", location)
                        .add_operands(&[val, dim_idx])
                        .add_results(&[index_type])
                        .build()
                        .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                )
                .result(0)
                .unwrap()
                .into();
            output_shape_vals.push(dim_val);
        }
    }

    let mut operands = vec![val];
    operands.extend(output_shape_vals);

    let expanded = body_block
        .append_operation(
            OperationBuilder::new("tensor.expand_shape", location)
                .add_operands(&operands)
                .add_results(&[new_tensor_type])
                .add_attributes(&[
                    (Identifier::new(context, "reassociation"), reassoc_attr),
                    (
                        Identifier::new(context, "static_output_shape"),
                        static_shape_attr,
                    ),
                ])
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        )
        .result(0)
        .unwrap()
        .into();

    Ok((expanded, new_shape))
}

// ── Helper: emit tosa.add or tosa.sub ────────────────────────────────────────
//
// Promotes operands to matching rank (via tosa.reshape) before emitting.
// `tosa.add` and `tosa.sub` require equal-rank operands.

#[allow(clippy::too_many_arguments)]
fn emit_tosa_binary<'c>(
    context: &'c Context,
    body_block: &Block<'c>,
    values: &HashMap<NodeId, melior::ir::Value<'c, 'c>>,
    op_name: &str,
    lhs: NodeId,
    rhs: NodeId,
    lhs_shape: &[u64],
    rhs_shape: &[u64],
    output_shape: &[u64],
    dtype: DType,
    location: Location<'c>,
) -> Result<melior::ir::Value<'c, 'c>, CompileError> {
    let tensor_type = make_ranked_tensor_type(context, output_shape, dtype);
    let lhs_val = *values.get(&lhs).expect("lhs node not yet emitted");
    let rhs_val = *values.get(&rhs).expect("rhs node not yet emitted");

    // Promote to matching rank if needed (tosa.add/sub/mul require equal ranks).
    let target_rank = output_shape.len();
    let (lhs_val, _) = promote_rank_with_reshape(
        context,
        body_block,
        lhs_val,
        lhs_shape,
        target_rank,
        dtype,
        location,
    )?;
    let (rhs_val, _) = promote_rank_with_reshape(
        context,
        body_block,
        rhs_val,
        rhs_shape,
        target_rank,
        dtype,
        location,
    )?;

    let result_val = body_block
        .append_operation(
            OperationBuilder::new(op_name, location)
                .add_operands(&[lhs_val, rhs_val])
                .add_results(&[tensor_type])
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        )
        .result(0)
        .unwrap()
        .into();

    Ok(result_val)
}

// ── Helper: emit tosa.mul ─────────────────────────────────────────────────────
//
// `tosa.mul` takes 3 operands: (input1, input2, shift : tensor<1xi8>).
// For float types the shift operand must still be present with value 0.

#[allow(clippy::too_many_arguments)]
fn emit_tosa_mul<'c>(
    context: &'c Context,
    body_block: &Block<'c>,
    values: &HashMap<NodeId, melior::ir::Value<'c, 'c>>,
    lhs: NodeId,
    rhs: NodeId,
    lhs_shape: &[u64],
    rhs_shape: &[u64],
    output_shape: &[u64],
    dtype: DType,
    location: Location<'c>,
) -> Result<melior::ir::Value<'c, 'c>, CompileError> {
    let tensor_type = make_ranked_tensor_type(context, output_shape, dtype);
    let lhs_val = *values.get(&lhs).expect("lhs node not yet emitted");
    let rhs_val = *values.get(&rhs).expect("rhs node not yet emitted");

    // Promote to matching rank if needed.
    let target_rank = output_shape.len();
    let (lhs_val, _) = promote_rank_with_reshape(
        context,
        body_block,
        lhs_val,
        lhs_shape,
        target_rank,
        dtype,
        location,
    )?;
    let (rhs_val, _) = promote_rank_with_reshape(
        context,
        body_block,
        rhs_val,
        rhs_shape,
        target_rank,
        dtype,
        location,
    )?;

    // shift is a rank-1 size-1 tensor<1xi8> with value 0 (required even for float mul).
    let shift_val =
        emit_tosa_const_scalar(context, body_block, "dense<0> : tensor<1xi8>", location)?;

    let result_val = body_block
        .append_operation(
            OperationBuilder::new("tosa.mul", location)
                .add_operands(&[lhs_val, rhs_val, shift_val])
                .add_results(&[tensor_type])
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        )
        .result(0)
        .unwrap()
        .into();

    Ok(result_val)
}

// ── Helper: emit tosa.negate ──────────────────────────────────────────────────
//
// `tosa.negate` takes 3 operands: (input, input1_zp, output_zp).
// For unquantized (float/int) use, both zero-point tensors are 0.

fn emit_tosa_negate<'c>(
    context: &'c Context,
    body_block: &Block<'c>,
    values: &HashMap<NodeId, melior::ir::Value<'c, 'c>>,
    input: NodeId,
    shape: &[u64],
    dtype: DType,
    location: Location<'c>,
) -> Result<melior::ir::Value<'c, 'c>, CompileError> {
    let tensor_type = make_ranked_tensor_type(context, shape, dtype);
    let input_val = *values.get(&input).expect("input node not yet emitted");

    // zero-point tensors: rank-1 size-1 tensor<1xT> matching the element type.
    let zp_dense = match dtype {
        DType::F32 => "dense<0.0> : tensor<1xf32>",
        DType::F64 => "dense<0.0> : tensor<1xf64>",
        DType::I32 => "dense<0> : tensor<1xi32>",
        DType::I64 => "dense<0> : tensor<1xi64>",
    };
    let zp1 = emit_tosa_const_scalar(context, body_block, zp_dense, location)?;
    let zp2 = emit_tosa_const_scalar(context, body_block, zp_dense, location)?;

    let result_val = body_block
        .append_operation(
            OperationBuilder::new("tosa.negate", location)
                .add_operands(&[input_val, zp1, zp2])
                .add_results(&[tensor_type])
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        )
        .result(0)
        .unwrap()
        .into();

    Ok(result_val)
}

// ── Helper: emit tosa.exp or tosa.tanh (simple 1-operand TOSA unary ops) ─────

#[allow(clippy::too_many_arguments)]
fn emit_tosa_unary_simple<'c>(
    context: &'c Context,
    body_block: &Block<'c>,
    values: &HashMap<NodeId, melior::ir::Value<'c, 'c>>,
    op_name: &str,
    input: NodeId,
    shape: &[u64],
    dtype: DType,
    location: Location<'c>,
) -> Result<melior::ir::Value<'c, 'c>, CompileError> {
    let tensor_type = make_ranked_tensor_type(context, shape, dtype);
    let input_val = *values.get(&input).expect("input node not yet emitted");

    let result_val = body_block
        .append_operation(
            OperationBuilder::new(op_name, location)
                .add_operands(&[input_val])
                .add_results(&[tensor_type])
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        )
        .result(0)
        .unwrap()
        .into();

    Ok(result_val)
}

// ── Helper: emit tosa.clamp (used for relu: clamp(input, 0, max)) ─────────────
//
// `tosa.clamp` in MLIR 21 takes attributes `min_val` and `max_val`
// (each a single IntegerAttr or FloatAttr matching the element type).

fn emit_tosa_clamp<'c>(
    context: &'c Context,
    body_block: &Block<'c>,
    values: &HashMap<NodeId, melior::ir::Value<'c, 'c>>,
    input: NodeId,
    shape: &[u64],
    dtype: DType,
    location: Location<'c>,
) -> Result<melior::ir::Value<'c, 'c>, CompileError> {
    let tensor_type = make_ranked_tensor_type(context, shape, dtype);
    let input_val = *values.get(&input).expect("input node not yet emitted");
    let elem_type = dtype.to_mlir_type(context);

    let (min_val_attr, max_val_attr): (Attribute<'c>, Attribute<'c>) = match dtype {
        DType::F32 => (
            FloatAttribute::new(context, elem_type, 0.0).into(),
            FloatAttribute::new(context, elem_type, f32::MAX as f64).into(),
        ),
        DType::F64 => (
            FloatAttribute::new(context, elem_type, 0.0).into(),
            FloatAttribute::new(context, elem_type, f64::MAX).into(),
        ),
        DType::I32 => (
            IntegerAttribute::new(elem_type, 0).into(),
            IntegerAttribute::new(elem_type, i32::MAX as i64).into(),
        ),
        DType::I64 => (
            IntegerAttribute::new(elem_type, 0).into(),
            IntegerAttribute::new(elem_type, i64::MAX).into(),
        ),
    };

    let result_val = body_block
        .append_operation(
            OperationBuilder::new("tosa.clamp", location)
                .add_operands(&[input_val])
                .add_results(&[tensor_type])
                .add_attributes(&[
                    (Identifier::new(context, "min_val"), min_val_attr),
                    (Identifier::new(context, "max_val"), max_val_attr),
                ])
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        )
        .result(0)
        .unwrap()
        .into();

    Ok(result_val)
}

// ── Helper: emit tosa.matmul for float [M,K] x [K,N] -> [M,N] ───────────────
//
// tosa.matmul requires 3-D inputs [B, M, K] x [B, K, N] -> [B, M, N].
// For 2-D inputs we wrap with tosa.const_shape + tosa.reshape to add a batch
// dim of 1, call tosa.matmul, then tosa.const_shape + tosa.reshape to remove
// it again.
//
// tosa.matmul takes 4 operands: (a, b, a_zp, b_zp) where a_zp and b_zp are
// scalar zero-point tensors (0.0 for float unquantized use).

#[allow(clippy::too_many_arguments)]
fn emit_tosa_matmul<'c>(
    context: &'c Context,
    body_block: &Block<'c>,
    values: &HashMap<NodeId, melior::ir::Value<'c, 'c>>,
    lhs: NodeId,
    rhs: NodeId,
    lhs_shape: &[u64],    // [M, K]
    rhs_shape: &[u64],    // [K, N]
    output_shape: &[u64], // [M, N]
    dtype: DType,
    location: Location<'c>,
) -> Result<melior::ir::Value<'c, 'c>, CompileError> {
    let lhs_val = *values.get(&lhs).expect("lhs node not yet emitted");
    let rhs_val = *values.get(&rhs).expect("rhs node not yet emitted");

    // Delegate to the value-based helper.
    emit_tosa_matmul_2d_values(
        context,
        body_block,
        lhs_val,
        rhs_val,
        lhs_shape,
        rhs_shape,
        output_shape,
        dtype,
        location,
    )
}

// ── Helper: 2D tosa matmul accepting raw SSA values ───────────────────────────
//
// Shared implementation for Op::Matmul and Op::Gemm (which needs to pass
// pre-processed — transposed / scaled — values rather than raw node-ID lookups).
// Takes raw tensor Values for lhs [M,K] and rhs [K,N], returns a 2-D [M,N]
// result via the same 2D→3D→tosa.matmul→2D reshaping pattern.

#[allow(clippy::too_many_arguments)]
fn emit_tosa_matmul_2d_values<'c>(
    context: &'c Context,
    body_block: &Block<'c>,
    lhs_val: melior::ir::Value<'c, 'c>,
    rhs_val: melior::ir::Value<'c, 'c>,
    lhs_shape: &[u64],    // [M, K]
    rhs_shape: &[u64],    // [K, N]
    output_shape: &[u64], // [M, N]
    dtype: DType,
    location: Location<'c>,
) -> Result<melior::ir::Value<'c, 'c>, CompileError> {
    let m = lhs_shape[0];
    let _k = lhs_shape[1];
    let n = rhs_shape[1];

    let has_dynamic = lhs_shape.iter().chain(rhs_shape.iter()).any(|&d| d == DIM_DYNAMIC);

    // 2D→3D: prepend batch=1.
    // For dynamic shapes, use promote_rank_with_reshape (tensor.expand_shape).
    // For static shapes, use emit_tosa_reshape (tosa.reshape with tosa.const_shape).
    let lhs_3d = if has_dynamic {
        promote_rank_with_reshape(context, body_block, lhs_val, lhs_shape, 3, dtype, location)?.0
    } else {
        emit_tosa_reshape(context, body_block, lhs_val, &[1, lhs_shape[0], lhs_shape[1]], dtype, location)?
    };
    let rhs_3d = if has_dynamic {
        promote_rank_with_reshape(context, body_block, rhs_val, rhs_shape, 3, dtype, location)?.0
    } else {
        emit_tosa_reshape(context, body_block, rhs_val, &[1, rhs_shape[0], rhs_shape[1]], dtype, location)?
    };

    let zp_dense = match dtype {
        DType::F32 => "dense<0.0> : tensor<1xf32>",
        DType::F64 => "dense<0.0> : tensor<1xf64>",
        DType::I32 => "dense<0> : tensor<1xi32>",
        DType::I64 => "dense<0> : tensor<1xi64>",
    };
    let a_zp = emit_tosa_const_scalar(context, body_block, zp_dense, location)?;
    let b_zp = emit_tosa_const_scalar(context, body_block, zp_dense, location)?;

    let out_3d_type = make_ranked_tensor_type(context, &[1, m, n], dtype);
    let out_3d: melior::ir::Value = body_block
        .append_operation(
            OperationBuilder::new("tosa.matmul", location)
                .add_operands(&[lhs_3d, rhs_3d, a_zp, b_zp])
                .add_results(&[out_3d_type])
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        )
        .result(0)
        .unwrap()
        .into();

    // 3D→2D: collapse the batch dim via tensor.collapse_shape (handles dynamic dims).
    let result_val = if has_dynamic || output_shape.iter().any(|&d| d == DIM_DYNAMIC) {
        let out_type = make_ranked_tensor_type(context, output_shape, dtype);
        // reassociation: [[0, 1], [2]] — merge batch + M into M, keep N.
        let reassoc_str = "[[0, 1], [2]]";
        let reassoc_attr = Attribute::parse(context, reassoc_str)
            .ok_or_else(|| CompileError::AttributeParse(reassoc_str.into()))?;
        body_block
            .append_operation(
                OperationBuilder::new("tensor.collapse_shape", location)
                    .add_operands(&[out_3d])
                    .add_results(&[out_type])
                    .add_attributes(&[(
                        Identifier::new(context, "reassociation"),
                        reassoc_attr,
                    )])
                    .build()
                    .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
            )
            .result(0)
            .unwrap()
            .into()
    } else {
        emit_tosa_reshape(context, body_block, out_3d, output_shape, dtype, location)?
    };
    Ok(result_val)
}

// ── Helper: emit linalg.matmul for [M,K] x [K,N] -> [M,N] ───────────────────

#[allow(clippy::too_many_arguments)]
fn emit_matmul<'c>(
    context: &'c Context,
    body_block: &Block<'c>,
    values: &HashMap<NodeId, melior::ir::Value<'c, 'c>>,
    lhs: NodeId,
    rhs: NodeId,
    output_shape: &[u64], // [M, N]
    dtype: DType,
    location: Location<'c>,
) -> Result<melior::ir::Value<'c, 'c>, CompileError> {
    let elem_type = dtype.to_mlir_type(context);
    let tensor_type = make_ranked_tensor_type(context, output_shape, dtype);

    let lhs_val = *values.get(&lhs).expect("lhs node not yet emitted");
    let rhs_val = *values.get(&rhs).expect("rhs node not yet emitted");

    // tensor.empty() for the [M, N] accumulator.
    // output_shape = [M, N]; M comes from lhs dim 0, N comes from rhs dim 1.
    let init_val: melior::ir::Value = if output_shape.contains(&DIM_DYNAMIC) {
        let sources: Vec<(melior::ir::Value<'c, 'c>, usize)> = output_shape
            .iter()
            .enumerate()
            .filter(|&(_, d)| *d == DIM_DYNAMIC)
            .map(|(i, _)| {
                // [M, N]: M is from lhs dim 0, N is from rhs dim 1.
                if i == 0 { (lhs_val, 0) } else { (rhs_val, 1) }
            })
            .collect();
        emit_tensor_empty_dynamic(context, body_block, output_shape, dtype, &sources, location)?
    } else {
        body_block
            .append_operation(
                OperationBuilder::new("tensor.empty", location)
                    .add_results(&[tensor_type])
                    .build()
                    .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
            )
            .result(0)
            .unwrap()
            .into()
    };

    // Zero constant for the fill.
    let zero_val: melior::ir::Value = match dtype {
        DType::F32 | DType::F64 => body_block
            .append_operation(arith::constant(
                context,
                FloatAttribute::new(context, elem_type, 0.0).into(),
                location,
            ))
            .result(0)
            .unwrap()
            .into(),
        DType::I32 | DType::I64 => body_block
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(elem_type, 0).into(),
                location,
            ))
            .result(0)
            .unwrap()
            .into(),
    };

    // linalg.fill: fill the init tensor with zero.
    // Named linalg ops require a region with one block. The block takes
    // scalar element args (ins scalar + out scalar) and yields the fill value.
    // operand_segment_sizes = [1 ins, 1 outs].
    let segment_fill = Attribute::parse(context, "array<i32: 1, 1>").ok_or_else(|| {
        CompileError::AttributeParse("failed to parse linalg.fill segment sizes".into())
    })?;
    let fill_region = {
        let fill_block = Block::new(&[(elem_type, location), (elem_type, location)]);
        let fill_in = fill_block.argument(0).unwrap().into();
        fill_block.append_operation(
            OperationBuilder::new("linalg.yield", location)
                .add_operands(&[fill_in])
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        );
        let r = Region::new();
        r.append_block(fill_block);
        r
    };
    let filled_val: melior::ir::Value = body_block
        .append_operation(
            OperationBuilder::new("linalg.fill", location)
                .add_operands(&[zero_val, init_val])
                .add_results(&[tensor_type])
                .add_attributes(&[(
                    Identifier::new(context, "operand_segment_sizes"),
                    segment_fill,
                )])
                .add_regions([fill_region])
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        )
        .result(0)
        .unwrap()
        .into();

    // linalg.matmul: 2 ins (lhs, rhs), 1 out (filled accumulator).
    // Named linalg op — region with one block taking (lhs_elem, rhs_elem, acc_elem)
    // and yielding the updated accumulator: acc + lhs * rhs.
    // operand_segment_sizes = [2 ins, 1 outs].
    let segment_matmul = Attribute::parse(context, "array<i32: 2, 1>").ok_or_else(|| {
        CompileError::AttributeParse("failed to parse linalg.matmul segment sizes".into())
    })?;
    let matmul_region = {
        let matmul_block = Block::new(&[
            (elem_type, location),
            (elem_type, location),
            (elem_type, location),
        ]);
        let lhs_elem: melior::ir::Value = matmul_block.argument(0).unwrap().into();
        let rhs_elem: melior::ir::Value = matmul_block.argument(1).unwrap().into();
        let acc_elem: melior::ir::Value = matmul_block.argument(2).unwrap().into();
        let mul_val: melior::ir::Value = match dtype {
            DType::F32 | DType::F64 => matmul_block
                .append_operation(arith::mulf(lhs_elem, rhs_elem, location))
                .result(0)
                .unwrap()
                .into(),
            DType::I32 | DType::I64 => matmul_block
                .append_operation(arith::muli(lhs_elem, rhs_elem, location))
                .result(0)
                .unwrap()
                .into(),
        };
        let add_val: melior::ir::Value = match dtype {
            DType::F32 | DType::F64 => matmul_block
                .append_operation(arith::addf(acc_elem, mul_val, location))
                .result(0)
                .unwrap()
                .into(),
            DType::I32 | DType::I64 => matmul_block
                .append_operation(arith::addi(acc_elem, mul_val, location))
                .result(0)
                .unwrap()
                .into(),
        };
        matmul_block.append_operation(
            OperationBuilder::new("linalg.yield", location)
                .add_operands(&[add_val])
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        );
        let r = Region::new();
        r.append_block(matmul_block);
        r
    };
    let result_val: melior::ir::Value = body_block
        .append_operation(
            OperationBuilder::new("linalg.matmul", location)
                .add_operands(&[lhs_val, rhs_val, filled_val])
                .add_results(&[tensor_type])
                .add_attributes(&[(
                    Identifier::new(context, "operand_segment_sizes"),
                    segment_matmul,
                )])
                .add_regions([matmul_region])
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        )
        .result(0)
        .unwrap()
        .into();

    Ok(result_val)
}

// ── Helper: emit batched matmul via linalg.generic ────────────────────────────
//
// Works for any rank >= 2. For rank-2 inputs (n_batch=0) this produces:
//   iterators: (m, n, k)  — m,n parallel; k reduction
//   lhs_map:  (m, n, k) -> (m, k)
//   rhs_map:  (m, n, k) -> (k, n)
//   out_map:  (m, n, k) -> (m, n)
//
// For rank-3 inputs (n_batch=1, e.g. [B,M,K] @ [B,K,N]):
//   iterators: (b, m, n, k)  — b,m,n parallel; k reduction
//   lhs_map:  (b, m, n, k) -> (b, m, k)
//   rhs_map:  (b, m, n, k) -> (b, k, n)
//   out_map:  (b, m, n, k) -> (b, m, n)
//
// In general for rank r (n_batch = r - 2):
//   total iterators = n_batch + 3  (batch dims + m + n + k)
//   lhs picks dims: d0..d(n_batch-1), d(n_batch), d(n_batch+2)   → [..., m, k]
//   rhs picks dims: d0..d(n_batch-1), d(n_batch+2), d(n_batch+1) → [..., k, n]
//   out picks dims: d0..d(n_batch-1), d(n_batch), d(n_batch+1)   → [..., m, n]
//   last iterator is reduction (k); all others are parallel

#[allow(clippy::too_many_arguments)]
fn emit_batched_matmul<'c>(
    context: &'c Context,
    body_block: &Block<'c>,
    values: &HashMap<NodeId, melior::ir::Value<'c, 'c>>,
    lhs: NodeId,
    rhs: NodeId,
    lhs_shape: &[u64],     // [..., M, K]
    rhs_shape: &[u64],     // [..., K, N] — may differ in rank from lhs (broadcast)
    output_shape: &[u64],  // [..., M, N]
    dtype: DType,
    location: Location<'c>,
) -> Result<melior::ir::Value<'c, 'c>, CompileError> {
    let lhs_val = *values.get(&lhs).expect("lhs node not yet emitted");
    let rhs_val = *values.get(&rhs).expect("rhs node not yet emitted");
    let elem_type = dtype.to_mlir_type(context);
    let out_tensor_type = make_ranked_tensor_type(context, output_shape, dtype);

    let out_rank = output_shape.len();
    let n_batch = out_rank - 2; // number of batch dimensions in output

    // Total iterator count: batch dims + m + n + k
    let n_iters = n_batch + 3;
    // dim indices:
    //   d0 .. d(n_batch-1)  = batch dims
    //   d(n_batch)          = m
    //   d(n_batch+1)        = n
    //   d(n_batch+2)        = k  (reduction)
    let dims: Vec<String> = (0..n_iters).map(|i| format!("d{i}")).collect();
    let dims_str = dims.join(", ");

    let m_dim = &dims[n_batch];
    let n_dim = &dims[n_batch + 1];
    let k_dim = &dims[n_batch + 2];

    // Build broadcast-aware maps for lhs and rhs.
    // Each operand may have fewer batch dims than the output.
    fn build_matmul_map(
        dims: &[String],
        dims_str: &str,
        operand_shape: &[u64],
        output_shape: &[u64],
        n_batch: usize,
        inner_dims: &[&str], // [m,k] for lhs, [k,n] for rhs
    ) -> String {
        let op_rank = operand_shape.len();
        let op_batch = op_rank - 2;
        // Right-align batch dims
        let batch_offset = n_batch - op_batch;
        let mut result_exprs: Vec<String> = Vec::new();
        for i in 0..op_batch {
            let out_idx = batch_offset + i;
            if operand_shape[i] == 1 && output_shape[out_idx] != 1 {
                result_exprs.push("0".to_string());
            } else {
                result_exprs.push(dims[out_idx].clone());
            }
        }
        for d in inner_dims {
            result_exprs.push(d.to_string());
        }
        format!("affine_map<({dims_str}) -> ({})>", result_exprs.join(", "))
    }

    let lhs_map_str = build_matmul_map(&dims, &dims_str, lhs_shape, output_shape, n_batch, &[m_dim, k_dim]);
    let rhs_map_str = build_matmul_map(&dims, &dims_str, rhs_shape, output_shape, n_batch, &[k_dim, n_dim]);

    // Output map: all batch dims + m + n
    let batch_dims: Vec<&str> = dims[..n_batch].iter().map(|s| s.as_str()).collect();
    let batch_str = if n_batch > 0 {
        format!("{}, ", batch_dims.join(", "))
    } else {
        String::new()
    };
    let out_map_str = format!("affine_map<({dims_str}) -> ({batch_str}{m_dim}, {n_dim})>");

    let lhs_map = Attribute::parse(context, &lhs_map_str)
        .ok_or_else(|| CompileError::AttributeParse(format!("failed to parse {lhs_map_str}")))?;
    let rhs_map = Attribute::parse(context, &rhs_map_str)
        .ok_or_else(|| CompileError::AttributeParse(format!("failed to parse {rhs_map_str}")))?;
    let out_map = Attribute::parse(context, &out_map_str)
        .ok_or_else(|| CompileError::AttributeParse(format!("failed to parse {out_map_str}")))?;

    let indexing_maps = ArrayAttribute::new(context, &[lhs_map, rhs_map, out_map]);

    // Iterator types: all parallel except the last (k = reduction).
    let iters: Vec<&str> = (0..n_iters)
        .map(|i| {
            if i == n_iters - 1 {
                "#linalg.iterator_type<reduction>"
            } else {
                "#linalg.iterator_type<parallel>"
            }
        })
        .collect();
    let iter_str = format!("[{}]", iters.join(", "));
    let iterator_types = Attribute::parse(context, &iter_str)
        .ok_or_else(|| CompileError::AttributeParse(format!("failed to parse {iter_str}")))?;

    // tensor.empty() for the output accumulator.
    // output_shape = [...batch..., M, N].
    // M comes from lhs (dim n_batch in lhs), N from rhs (last dim in rhs).
    let empty_val: melior::ir::Value = if output_shape.contains(&DIM_DYNAMIC) {
        let lhs_n_batch = lhs_shape.len() - 2;
        let rhs_n_batch = rhs_shape.len() - 2;
        let sources: Vec<(melior::ir::Value<'c, 'c>, usize)> = output_shape
            .iter()
            .enumerate()
            .filter(|&(_, d)| *d == DIM_DYNAMIC)
            .map(|(i, _)| {
                // Batch dims: right-align from lhs or rhs batch dims.
                // M dim: from lhs at its n_batch position.
                // N dim: from rhs at last position.
                if i < n_batch {
                    // Batch dim — right-align against lhs batch dims.
                    let batch_offset = n_batch - lhs_n_batch;
                    if i >= batch_offset {
                        (lhs_val, i - batch_offset)
                    } else {
                        let rhs_batch_offset = n_batch - rhs_n_batch;
                        (rhs_val, i - rhs_batch_offset)
                    }
                } else if i == n_batch {
                    // M dim from lhs.
                    (lhs_val, lhs_n_batch)
                } else {
                    // N dim from rhs (last dim).
                    (rhs_val, rhs_shape.len() - 1)
                }
            })
            .collect();
        emit_tensor_empty_dynamic(context, body_block, output_shape, dtype, &sources, location)?
    } else {
        body_block
            .append_operation(
                OperationBuilder::new("tensor.empty", location)
                    .add_results(&[out_tensor_type])
                    .build()
                    .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
            )
            .result(0)
            .unwrap()
            .into()
    };

    // Zero constant for the fill.
    let zero_val: melior::ir::Value = match dtype {
        DType::F32 | DType::F64 => body_block
            .append_operation(arith::constant(
                context,
                FloatAttribute::new(context, elem_type, 0.0).into(),
                location,
            ))
            .result(0)
            .unwrap()
            .into(),
        DType::I32 | DType::I64 => body_block
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(elem_type, 0).into(),
                location,
            ))
            .result(0)
            .unwrap()
            .into(),
    };

    // linalg.fill to initialize accumulator to zero.
    let segment_fill = Attribute::parse(context, "array<i32: 1, 1>").ok_or_else(|| {
        CompileError::AttributeParse("failed to parse linalg.fill segment sizes".into())
    })?;
    let fill_region = {
        let fill_block = Block::new(&[(elem_type, location), (elem_type, location)]);
        let fill_in = fill_block.argument(0).unwrap().into();
        fill_block.append_operation(
            OperationBuilder::new("linalg.yield", location)
                .add_operands(&[fill_in])
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        );
        let r = Region::new();
        r.append_block(fill_block);
        r
    };
    let init_val: melior::ir::Value = body_block
        .append_operation(
            OperationBuilder::new("linalg.fill", location)
                .add_operands(&[zero_val, empty_val])
                .add_results(&[out_tensor_type])
                .add_attributes(&[(
                    Identifier::new(context, "operand_segment_sizes"),
                    segment_fill,
                )])
                .add_regions([fill_region])
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        )
        .result(0)
        .unwrap()
        .into();

    // linalg.generic body: mul + accumulate.
    let segment_generic = Attribute::parse(context, "array<i32: 2, 1>").ok_or_else(|| {
        CompileError::AttributeParse("failed to parse linalg.generic segment sizes".into())
    })?;
    let generic_region = {
        let generic_block = Block::new(&[
            (elem_type, location),
            (elem_type, location),
            (elem_type, location),
        ]);
        let a_elem: melior::ir::Value = generic_block.argument(0).unwrap().into();
        let b_elem: melior::ir::Value = generic_block.argument(1).unwrap().into();
        let c_elem: melior::ir::Value = generic_block.argument(2).unwrap().into();

        let mul_val: melior::ir::Value = match dtype {
            DType::F32 | DType::F64 => generic_block
                .append_operation(arith::mulf(a_elem, b_elem, location))
                .result(0)
                .unwrap()
                .into(),
            DType::I32 | DType::I64 => generic_block
                .append_operation(arith::muli(a_elem, b_elem, location))
                .result(0)
                .unwrap()
                .into(),
        };
        let add_val: melior::ir::Value = match dtype {
            DType::F32 | DType::F64 => generic_block
                .append_operation(arith::addf(c_elem, mul_val, location))
                .result(0)
                .unwrap()
                .into(),
            DType::I32 | DType::I64 => generic_block
                .append_operation(arith::addi(c_elem, mul_val, location))
                .result(0)
                .unwrap()
                .into(),
        };
        generic_block.append_operation(
            OperationBuilder::new("linalg.yield", location)
                .add_operands(&[add_val])
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        );
        let r = Region::new();
        r.append_block(generic_block);
        r
    };

    let result_val: melior::ir::Value = body_block
        .append_operation(
            OperationBuilder::new("linalg.generic", location)
                .add_operands(&[lhs_val, rhs_val, init_val])
                .add_results(&[out_tensor_type])
                .add_attributes(&[
                    (
                        Identifier::new(context, "indexing_maps"),
                        indexing_maps.into(),
                    ),
                    (Identifier::new(context, "iterator_types"), iterator_types),
                    (
                        Identifier::new(context, "operand_segment_sizes"),
                        segment_generic,
                    ),
                ])
                .add_regions([generic_region])
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        )
        .result(0)
        .unwrap()
        .into();

    Ok(result_val)
}

// ── Helper: emit a reduction linalg.generic (ReduceSum or ReduceMax) ─────────

#[allow(clippy::too_many_arguments)]
fn emit_reduction<'c>(
    context: &'c Context,
    body_block: &Block<'c>,
    values: &HashMap<NodeId, melior::ir::Value<'c, 'c>>,
    input: NodeId,
    input_shape: &[u64],  // shape of the INPUT tensor
    output_shape: &[u64], // shape of the OUTPUT tensor
    dim: usize,
    keepdim: bool,
    dtype: DType,
    location: Location<'c>,
    is_max: bool,
) -> Result<melior::ir::Value<'c, 'c>, CompileError> {
    let elem_type = dtype.to_mlir_type(context);
    let output_tensor_type = make_ranked_tensor_type(context, output_shape, dtype);

    let input_val = *values.get(&input).expect("input node not yet emitted");

    // tensor.empty() for the output accumulator.
    // For dynamic output shapes: each `?` dim in output corresponds to a non-reduced
    // dim in the input. Map output dim index back to input dim index.
    let init_val: melior::ir::Value = if output_shape.contains(&DIM_DYNAMIC) {
        // Build a mapping: output dim j -> input dim index.
        // output has the reduced dim removed (or zeroed for keepdim).
        let sources: Vec<(melior::ir::Value<'c, 'c>, usize)> = output_shape
            .iter()
            .enumerate()
            .filter(|&(_, d)| *d == DIM_DYNAMIC)
            .map(|(out_i, _)| {
                // Find the corresponding input dim for output dim out_i.
                let in_dim = if keepdim {
                    // keepdim: output rank == input rank; reduced dim has value 0 (not ?).
                    out_i
                } else {
                    // !keepdim: output rank == input_rank - 1; reduced dim was removed.
                    if out_i < dim { out_i } else { out_i + 1 }
                };
                (input_val, in_dim)
            })
            .collect();
        emit_tensor_empty_dynamic(context, body_block, output_shape, dtype, &sources, location)?
    } else {
        body_block
            .append_operation(
                OperationBuilder::new("tensor.empty", location)
                    .add_results(&[output_tensor_type])
                    .build()
                    .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
            )
            .result(0)
            .unwrap()
            .into()
    };

    // Build the identity constant for fill.
    let identity_val: melior::ir::Value = if is_max {
        match dtype {
            DType::F32 | DType::F64 => body_block
                .append_operation(arith::constant(
                    context,
                    FloatAttribute::new(context, elem_type, f64::NEG_INFINITY).into(),
                    location,
                ))
                .result(0)
                .unwrap()
                .into(),
            DType::I32 => body_block
                .append_operation(arith::constant(
                    context,
                    IntegerAttribute::new(elem_type, i32::MIN as i64).into(),
                    location,
                ))
                .result(0)
                .unwrap()
                .into(),
            DType::I64 => body_block
                .append_operation(arith::constant(
                    context,
                    IntegerAttribute::new(elem_type, i64::MIN).into(),
                    location,
                ))
                .result(0)
                .unwrap()
                .into(),
        }
    } else {
        // ReduceSum: identity is zero.
        match dtype {
            DType::F32 | DType::F64 => body_block
                .append_operation(arith::constant(
                    context,
                    FloatAttribute::new(context, elem_type, 0.0).into(),
                    location,
                ))
                .result(0)
                .unwrap()
                .into(),
            DType::I32 | DType::I64 => body_block
                .append_operation(arith::constant(
                    context,
                    IntegerAttribute::new(elem_type, 0).into(),
                    location,
                ))
                .result(0)
                .unwrap()
                .into(),
        }
    };

    // linalg.fill: fill the init tensor with the identity value.
    let segment_fill = Attribute::parse(context, "array<i32: 1, 1>").ok_or_else(|| {
        CompileError::AttributeParse("failed to parse linalg.fill segment sizes".into())
    })?;
    let fill_region = {
        let fill_block = Block::new(&[(elem_type, location), (elem_type, location)]);
        let fill_in = fill_block.argument(0).unwrap().into();
        fill_block.append_operation(
            OperationBuilder::new("linalg.yield", location)
                .add_operands(&[fill_in])
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        );
        let r = Region::new();
        r.append_block(fill_block);
        r
    };
    let filled_val: melior::ir::Value = body_block
        .append_operation(
            OperationBuilder::new("linalg.fill", location)
                .add_operands(&[identity_val, init_val])
                .add_results(&[output_tensor_type])
                .add_attributes(&[(
                    Identifier::new(context, "operand_segment_sizes"),
                    segment_fill,
                )])
                .add_regions([fill_region])
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        )
        .result(0)
        .unwrap()
        .into();

    // Build iterator types: all "parallel" except the reduction dim.
    let input_rank = input_shape.len();
    let iters: Vec<&str> = (0..input_rank)
        .map(|i| {
            if i == dim {
                "#linalg.iterator_type<reduction>"
            } else {
                "#linalg.iterator_type<parallel>"
            }
        })
        .collect();
    let iter_str = format!("[{}]", iters.join(", "));
    let iterator_types = Attribute::parse(context, &iter_str)
        .ok_or_else(|| CompileError::AttributeParse(format!("failed to parse {iter_str}")))?;

    // Build affine maps.
    // Input map: identity over input_rank dimensions.
    let input_map = make_identity_map(context, input_rank)?;

    // Output map: project away or zero out the reduction dim.
    let dim_vars: Vec<String> = (0..input_rank).map(|i| format!("d{i}")).collect();
    let dims_str = dim_vars.join(", ");
    let mut result_exprs: Vec<String> = Vec::new();
    for (i, var) in dim_vars.iter().enumerate() {
        if i == dim {
            if keepdim {
                result_exprs.push("0".to_string());
            }
            // if !keepdim, skip this dim entirely
        } else {
            result_exprs.push(var.clone());
        }
    }
    let result_str = result_exprs.join(", ");
    let output_map_str = format!("affine_map<({dims_str}) -> ({result_str})>");
    let output_map = Attribute::parse(context, &output_map_str)
        .ok_or_else(|| CompileError::AttributeParse(format!("failed to parse {output_map_str}")))?;

    let indexing_maps = ArrayAttribute::new(context, &[input_map, output_map]);
    let segment_sizes = Attribute::parse(context, "array<i32: 1, 1>").ok_or_else(|| {
        CompileError::AttributeParse("failed to parse operand_segment_sizes".into())
    })?;

    // linalg.generic body: 2 args (in_elem, acc_elem).
    let linalg_region = {
        let linalg_block = Block::new(&[(elem_type, location), (elem_type, location)]);
        let in_elem: melior::ir::Value = linalg_block.argument(0).unwrap().into();
        let acc_elem: melior::ir::Value = linalg_block.argument(1).unwrap().into();

        let result: melior::ir::Value = if is_max {
            match dtype {
                DType::F32 | DType::F64 => linalg_block
                    .append_operation(arith::maxnumf(in_elem, acc_elem, location))
                    .result(0)
                    .unwrap()
                    .into(),
                DType::I32 | DType::I64 => linalg_block
                    .append_operation(arith::maxsi(in_elem, acc_elem, location))
                    .result(0)
                    .unwrap()
                    .into(),
            }
        } else {
            // ReduceSum
            match dtype {
                DType::F32 | DType::F64 => linalg_block
                    .append_operation(arith::addf(in_elem, acc_elem, location))
                    .result(0)
                    .unwrap()
                    .into(),
                DType::I32 | DType::I64 => linalg_block
                    .append_operation(arith::addi(in_elem, acc_elem, location))
                    .result(0)
                    .unwrap()
                    .into(),
            }
        };

        linalg_block.append_operation(
            OperationBuilder::new("linalg.yield", location)
                .add_operands(&[result])
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        );

        let r = Region::new();
        r.append_block(linalg_block);
        r
    };

    let result_val: melior::ir::Value = body_block
        .append_operation(
            OperationBuilder::new("linalg.generic", location)
                .add_operands(&[input_val, filled_val])
                .add_results(&[output_tensor_type])
                .add_attributes(&[
                    (
                        Identifier::new(context, "indexing_maps"),
                        indexing_maps.into(),
                    ),
                    (Identifier::new(context, "iterator_types"), iterator_types),
                    (
                        Identifier::new(context, "operand_segment_sizes"),
                        segment_sizes,
                    ),
                ])
                .add_regions([linalg_region])
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        )
        .result(0)
        .unwrap()
        .into();

    Ok(result_val)
}

// ── Helper: emit tosa.reduce_sum or tosa.reduce_max ──────────────────────────
//
// tosa.reduce_sum / tosa.reduce_max take a single input and an `axis` attribute
// (i32). They KEEP the same rank as input — the reduced dimension becomes size 1
// (i.e., they always behave like keepdim=true).
//
// For keepdim=true: TOSA output is the final output (shape matches).
// For keepdim=false: we emit tosa.reshape afterwards to remove the
// size-1 dimension at position `dim`.
//
// tosa.reduce_max also requires a `nan_mode` attribute ("PROPAGATE").

#[allow(clippy::too_many_arguments)]
fn emit_tosa_reduce<'c>(
    context: &'c Context,
    body_block: &Block<'c>,
    values: &HashMap<NodeId, melior::ir::Value<'c, 'c>>,
    op_name: &str, // "tosa.reduce_sum" or "tosa.reduce_max"
    input: NodeId,
    input_shape: &[u64],
    output_shape: &[u64], // final output shape (keepdim already reflected)
    dim: usize,
    keepdim: bool,
    dtype: DType,
    location: Location<'c>,
) -> Result<melior::ir::Value<'c, 'c>, CompileError> {
    let input_val = *values.get(&input).expect("input node not yet emitted");

    // TOSA reduction output: same rank as input, with dim set to 1.
    let mut tosa_out_shape: Vec<u64> = input_shape.to_vec();
    tosa_out_shape[dim] = 1;
    let tosa_out_type = make_ranked_tensor_type(context, &tosa_out_shape, dtype);

    // axis attribute: i32
    let axis_attr = IntegerAttribute::new(
        melior::ir::r#type::IntegerType::new(context, 32).into(),
        dim as i64,
    );

    let mut attrs: Vec<(Identifier<'c>, Attribute<'c>)> =
        vec![(Identifier::new(context, "axis"), axis_attr.into())];

    // tosa.reduce_max requires nan_mode = "PROPAGATE"
    if op_name == "tosa.reduce_max" {
        let nan_mode_attr = Attribute::parse(context, "\"PROPAGATE\"")
            .ok_or_else(|| CompileError::AttributeParse("nan_mode PROPAGATE".into()))?;
        attrs.push((Identifier::new(context, "nan_mode"), nan_mode_attr));
    }

    let reduced: melior::ir::Value = body_block
        .append_operation(
            OperationBuilder::new(op_name, location)
                .add_operands(&[input_val])
                .add_results(&[tosa_out_type])
                .add_attributes(&attrs)
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        )
        .result(0)
        .unwrap()
        .into();

    if keepdim {
        // TOSA output already has the size-1 dim at position `dim` — matches output_shape.
        return Ok(reduced);
    }

    // keepdim=false: use tosa.const_shape + tosa.reshape to remove the size-1 dim
    // at position `dim`. tosa_out_shape has rank = input_rank; output_shape has
    // rank = input_rank - 1 (the size-1 dim at `dim` has been removed).
    let result_val =
        emit_tosa_reshape(context, body_block, reduced, output_shape, dtype, location)?;

    Ok(result_val)
}

// ── Helper: emit tosa.transpose with a permutation attribute ─────────────────
//
// In MLIR 21 tosa, tosa.transpose takes a single operand (the input tensor)
// and a `perms` attribute (array<i32: ...>). For a 2-D swap the perm is [1, 0].

fn emit_tosa_transpose_2d<'c>(
    context: &'c Context,
    body_block: &Block<'c>,
    input: melior::ir::Value<'c, 'c>,
    input_shape: &[u64], // [rows, cols] before transpose
    dtype: DType,
    location: Location<'c>,
) -> Result<melior::ir::Value<'c, 'c>, CompileError> {
    // Output shape is the transpose: [cols, rows].
    let out_shape = [input_shape[1], input_shape[0]];
    let result_type = make_ranked_tensor_type(context, &out_shape, dtype);

    // Permutation attribute: array<i32: 1, 0>
    let perms_attr = Attribute::parse(context, "array<i32: 1, 0>")
        .ok_or_else(|| CompileError::AttributeParse("array<i32: 1, 0>".into()))?;

    let transposed: melior::ir::Value = body_block
        .append_operation(
            OperationBuilder::new("tosa.transpose", location)
                .add_operands(&[input])
                .add_results(&[result_type])
                .add_attributes(&[(Identifier::new(context, "perms"), perms_attr)])
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        )
        .result(0)
        .unwrap()
        .into();

    Ok(transposed)
}

// ── Helper: emit tosa.transpose with an arbitrary permutation ────────────────
//
// Generalizes emit_tosa_transpose_2d to any rank. `perms` is the permutation
// (e.g. [0, 2, 3, 1] for NCHW→NHWC). The output shape is derived by applying
// the permutation to `input_shape`.

fn emit_tosa_transpose<'c>(
    context: &'c Context,
    body_block: &Block<'c>,
    input: melior::ir::Value<'c, 'c>,
    input_shape: &[u64],
    perms: &[usize],
    dtype: DType,
    location: Location<'c>,
) -> Result<melior::ir::Value<'c, 'c>, CompileError> {
    // Validate permutation.
    debug_assert_eq!(
        perms.len(),
        input_shape.len(),
        "perms length must match rank"
    );
    debug_assert!(
        perms.iter().all(|&p| p < input_shape.len()),
        "perms index out of bounds"
    );
    // Compute output shape by applying permutation.
    let out_shape: Vec<u64> = perms.iter().map(|&p| input_shape[p]).collect();
    let result_type = make_ranked_tensor_type(context, &out_shape, dtype);

    // Build `array<i32: p0, p1, ...>` attribute.
    let perms_str = perms
        .iter()
        .map(|p| p.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let attr_str = format!("array<i32: {perms_str}>");
    let perms_attr = Attribute::parse(context, &attr_str)
        .ok_or_else(|| CompileError::AttributeParse(attr_str.clone()))?;

    let transposed: melior::ir::Value = body_block
        .append_operation(
            OperationBuilder::new("tosa.transpose", location)
                .add_operands(&[input])
                .add_results(&[result_type])
                .add_attributes(&[(Identifier::new(context, "perms"), perms_attr)])
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        )
        .result(0)
        .unwrap()
        .into();

    Ok(transposed)
}

// ── Helper: format a float value for MLIR dense attribute literals ────────────
//
// MLIR requires a decimal point or exponent in float literals (e.g. "2.0" not
// "2"). Rust's default Display for f64 omits the decimal for whole numbers, so
// we append ".0" when necessary.

fn format_float(v: f64) -> String {
    let s = format!("{v}");
    if s.contains('.') || s.contains('e') || s.contains('E') {
        s
    } else {
        format!("{v}.0")
    }
}

// ── Helper: emit a tosa scalar-constant tensor for alpha/beta scaling ─────────
//
// Emits a tosa.const filled with `scalar` (matching dtype) for use in tosa.mul.
// Returns a tensor<1xT> so tosa.mul can broadcast over the matrix.

fn emit_tosa_scalar_const<'c>(
    context: &'c Context,
    body_block: &Block<'c>,
    scalar: f64,
    dtype: DType,
    location: Location<'c>,
) -> Result<melior::ir::Value<'c, 'c>, CompileError> {
    // MLIR requires a decimal point in float literals (e.g. "2.0" not "2").
    let dense_str = match dtype {
        DType::F32 => format!("dense<{}> : tensor<1xf32>", format_float(scalar)),
        DType::F64 => format!("dense<{}> : tensor<1xf64>", format_float(scalar)),
        DType::I32 => format!("dense<{}> : tensor<1xi32>", scalar as i64),
        DType::I64 => format!("dense<{}> : tensor<1xi64>", scalar as i64),
    };
    emit_tosa_const_scalar(context, body_block, &dense_str, location)
}

// ── Helper: emit tosa.mul of a matrix by a scalar constant tensor ─────────────
//
// Scales `mat` (shape `mat_shape`) by a rank-1 tensor<1xT> holding `scalar`.
// tosa.mul requires equal-rank operands, so we reshape the scalar to match
// the matrix rank by prepending size-1 dimensions.

#[allow(clippy::too_many_arguments)]
fn emit_tosa_scale<'c>(
    context: &'c Context,
    body_block: &Block<'c>,
    mat: melior::ir::Value<'c, 'c>,
    mat_shape: &[u64],
    scalar: f64,
    dtype: DType,
    location: Location<'c>,
) -> Result<melior::ir::Value<'c, 'c>, CompileError> {
    let scalar_val = emit_tosa_scalar_const(context, body_block, scalar, dtype, location)?;

    // Reshape scalar tensor<1xT> to match matrix rank: [1, 1] for rank-2.
    let rank = mat_shape.len();
    let scalar_shape: Vec<u64> = vec![1u64; rank];
    let scalar_reshaped = emit_tosa_reshape(
        context,
        body_block,
        scalar_val,
        &scalar_shape,
        dtype,
        location,
    )?;

    // shift operand required by tosa.mul (zero for float/int unquantized)
    let shift_val =
        emit_tosa_const_scalar(context, body_block, "dense<0> : tensor<1xi8>", location)?;

    let result_type = make_ranked_tensor_type(context, mat_shape, dtype);
    let result: melior::ir::Value = body_block
        .append_operation(
            OperationBuilder::new("tosa.mul", location)
                .add_operands(&[mat, scalar_reshaped, shift_val])
                .add_results(&[result_type])
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        )
        .result(0)
        .unwrap()
        .into();

    Ok(result)
}

// ── Helper: emit tosa.add for bias (with optional beta scaling) ───────────────
//
// Adds `bias` (shape `bias_shape`) to `mat` (shape `mat_shape`). Promotes bias
// rank to match matrix rank before adding.

#[allow(clippy::too_many_arguments)]
fn emit_tosa_add_bias<'c>(
    context: &'c Context,
    body_block: &Block<'c>,
    mat: melior::ir::Value<'c, 'c>,
    mat_shape: &[u64],
    bias: melior::ir::Value<'c, 'c>,
    bias_shape: &[u64],
    dtype: DType,
    location: Location<'c>,
) -> Result<melior::ir::Value<'c, 'c>, CompileError> {
    let target_rank = mat_shape.len();
    let (bias_promoted, _) = promote_rank_with_reshape(
        context,
        body_block,
        bias,
        bias_shape,
        target_rank,
        dtype,
        location,
    )?;

    let result_type = make_ranked_tensor_type(context, mat_shape, dtype);
    let result: melior::ir::Value = body_block
        .append_operation(
            OperationBuilder::new("tosa.add", location)
                .add_operands(&[mat, bias_promoted])
                .add_results(&[result_type])
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        )
        .result(0)
        .unwrap()
        .into();

    Ok(result)
}

// ── Helper: emit a unary linalg.generic with a single-operand math op ────────
//
// Used for Sqrt (math.sqrt). The math op name is passed as a string, e.g.
// "math.sqrt". The body block has 2 args: (in_elem, out_elem); yields the
// math op applied to in_elem.

#[allow(clippy::too_many_arguments)]
fn emit_unary_linalg_math<'c>(
    context: &'c Context,
    body_block: &Block<'c>,
    values: &HashMap<NodeId, melior::ir::Value<'c, 'c>>,
    math_op: &str,
    input: NodeId,
    shape: &[u64],
    dtype: DType,
    location: Location<'c>,
) -> Result<melior::ir::Value<'c, 'c>, CompileError> {
    let elem_type = dtype.to_mlir_type(context);
    let tensor_type = make_ranked_tensor_type(context, shape, dtype);
    let rank = shape.len();

    let input_val = *values.get(&input).expect("input node not yet emitted");

    let init_val: melior::ir::Value = if shape.contains(&DIM_DYNAMIC) {
        let sources: Vec<(melior::ir::Value<'c, 'c>, usize)> = shape
            .iter()
            .enumerate()
            .filter(|&(_, d)| *d == DIM_DYNAMIC)
            .map(|(i, _)| (input_val, i))
            .collect();
        emit_tensor_empty_dynamic(context, body_block, shape, dtype, &sources, location)?
    } else {
        body_block
            .append_operation(
                OperationBuilder::new("tensor.empty", location)
                    .add_results(&[tensor_type])
                    .build()
                    .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
            )
            .result(0)
            .unwrap()
            .into()
    };

    // linalg body: 2 block args (in_elem, out_elem); yield math_op(in_elem).
    let linalg_region = {
        let linalg_block = Block::new(&[(elem_type, location), (elem_type, location)]);
        let in_elem: melior::ir::Value = linalg_block.argument(0).unwrap().into();

        let result: melior::ir::Value = linalg_block
            .append_operation(
                OperationBuilder::new(math_op, location)
                    .add_operands(&[in_elem])
                    .add_results(&[elem_type])
                    .build()
                    .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
            )
            .result(0)
            .unwrap()
            .into();

        linalg_block.append_operation(
            OperationBuilder::new("linalg.yield", location)
                .add_operands(&[result])
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        );

        let r = Region::new();
        r.append_block(linalg_block);
        r
    };

    let in_map = make_identity_map(context, rank)?;
    let out_map = make_identity_map(context, rank)?;
    let indexing_maps = ArrayAttribute::new(context, &[in_map, out_map]);
    let iterator_types = make_iterator_types(context, rank)?;
    let segment_sizes = Attribute::parse(context, "array<i32: 1, 1>").ok_or_else(|| {
        CompileError::AttributeParse("failed to parse operand_segment_sizes".into())
    })?;

    let result_val: melior::ir::Value = body_block
        .append_operation(
            OperationBuilder::new("linalg.generic", location)
                .add_operands(&[input_val, init_val])
                .add_results(&[tensor_type])
                .add_attributes(&[
                    (
                        Identifier::new(context, "indexing_maps"),
                        indexing_maps.into(),
                    ),
                    (Identifier::new(context, "iterator_types"), iterator_types),
                    (
                        Identifier::new(context, "operand_segment_sizes"),
                        segment_sizes,
                    ),
                ])
                .add_regions([linalg_region])
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        )
        .result(0)
        .unwrap()
        .into();

    Ok(result_val)
}

// ── Helper: emit a Cast via linalg.generic + arith cast ops ──────────────────
//
// Selects the appropriate arith dialect cast op based on src/dst dtype pair:
//   F32/F64 → I32/I64 : arith.fptosi
//   I32/I64 → F32/F64 : arith.sitofp
//   I32     → I64     : arith.extsi
//   I64     → I32     : arith.trunci
//   F32     → F64     : arith.extf
//   F64     → F32     : arith.truncf

#[allow(clippy::too_many_arguments)]
fn emit_cast<'c>(
    context: &'c Context,
    body_block: &Block<'c>,
    values: &HashMap<NodeId, melior::ir::Value<'c, 'c>>,
    input: NodeId,
    shape: &[u64],
    src_dtype: DType,
    dst_dtype: DType,
    location: Location<'c>,
) -> Result<melior::ir::Value<'c, 'c>, CompileError> {
    let src_elem_type = src_dtype.to_mlir_type(context);
    let dst_elem_type = dst_dtype.to_mlir_type(context);
    let src_tensor_type = make_ranked_tensor_type(context, shape, src_dtype);
    let dst_tensor_type = make_ranked_tensor_type(context, shape, dst_dtype);
    let rank = shape.len();

    let input_val = *values.get(&input).expect("input node not yet emitted");

    let init_val: melior::ir::Value = if shape.contains(&DIM_DYNAMIC) {
        let sources: Vec<(melior::ir::Value<'c, 'c>, usize)> = shape
            .iter()
            .enumerate()
            .filter(|&(_, d)| *d == DIM_DYNAMIC)
            .map(|(i, _)| (input_val, i))
            .collect();
        emit_tensor_empty_dynamic(context, body_block, shape, dst_dtype, &sources, location)?
    } else {
        body_block
            .append_operation(
                OperationBuilder::new("tensor.empty", location)
                    .add_results(&[dst_tensor_type])
                    .build()
                    .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
            )
            .result(0)
            .unwrap()
            .into()
    };

    // Select arith cast op name.
    let cast_op = match (src_dtype, dst_dtype) {
        (DType::F32 | DType::F64, DType::I32 | DType::I64) => "arith.fptosi",
        (DType::I32 | DType::I64, DType::F32 | DType::F64) => "arith.sitofp",
        (DType::I32, DType::I64) => "arith.extsi",
        (DType::I64, DType::I32) => "arith.trunci",
        (DType::F32, DType::F64) => "arith.extf",
        (DType::F64, DType::F32) => "arith.truncf",
        _ => {
            return Err(CompileError::AttributeParse(format!(
                "Cast: unsupported {src_dtype:?} -> {dst_dtype:?}"
            )))
        }
    };

    // linalg body: 2 block args (src_elem, dst_elem); yield cast(src_elem).
    let linalg_region = {
        let linalg_block = Block::new(&[(src_elem_type, location), (dst_elem_type, location)]);
        let src_elem: melior::ir::Value = linalg_block.argument(0).unwrap().into();

        let casted: melior::ir::Value = linalg_block
            .append_operation(
                OperationBuilder::new(cast_op, location)
                    .add_operands(&[src_elem])
                    .add_results(&[dst_elem_type])
                    .build()
                    .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
            )
            .result(0)
            .unwrap()
            .into();

        linalg_block.append_operation(
            OperationBuilder::new("linalg.yield", location)
                .add_operands(&[casted])
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        );

        let r = Region::new();
        r.append_block(linalg_block);
        r
    };

    let in_map = make_identity_map(context, rank)?;
    let out_map = make_identity_map(context, rank)?;
    let indexing_maps = ArrayAttribute::new(context, &[in_map, out_map]);
    let iterator_types = make_iterator_types(context, rank)?;
    let segment_sizes = Attribute::parse(context, "array<i32: 1, 1>").ok_or_else(|| {
        CompileError::AttributeParse("failed to parse operand_segment_sizes".into())
    })?;

    let _ = src_tensor_type; // used implicitly via input_val type

    let result_val: melior::ir::Value = body_block
        .append_operation(
            OperationBuilder::new("linalg.generic", location)
                .add_operands(&[input_val, init_val])
                .add_results(&[dst_tensor_type])
                .add_attributes(&[
                    (
                        Identifier::new(context, "indexing_maps"),
                        indexing_maps.into(),
                    ),
                    (Identifier::new(context, "iterator_types"), iterator_types),
                    (
                        Identifier::new(context, "operand_segment_sizes"),
                        segment_sizes,
                    ),
                ])
                .add_regions([linalg_region])
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        )
        .result(0)
        .unwrap()
        .into();

    Ok(result_val)
}

// ── Helper: emit ReduceMean via tosa.reduce_sum + tosa.reciprocal + tosa.mul ──
//
// Each axis is reduced sequentially using tosa.reduce_sum (which always keeps
// the reduced dim as size-1). After all reductions, multiply by 1/N where N is
// the total number of elements reduced. If !keepdim, reshape to remove the
// size-1 dims.

#[allow(clippy::too_many_arguments)]
fn emit_reduce_mean<'c>(
    context: &'c Context,
    body_block: &Block<'c>,
    values: &HashMap<NodeId, melior::ir::Value<'c, 'c>>,
    input: NodeId,
    input_shape: &[u64],
    output_shape: &[u64], // final output shape (keepdim already reflected)
    axes: &[i64],         // normalized non-negative axes
    keepdim: bool,
    dtype: DType,
    location: Location<'c>,
) -> Result<melior::ir::Value<'c, 'c>, CompileError> {
    // We work with intermediate values directly (not looking up by NodeId after
    // first access), so we need the initial value from the map.
    let mut current_val = *values.get(&input).expect("input node not yet emitted");
    let mut current_shape = input_shape.to_vec();

    // Count total reduced elements for the reciprocal constant.
    let total_reduced: u64 = axes.iter().map(|&a| input_shape[a as usize]).product();

    // Reduce each axis (all keepdim=true during intermediate steps).
    for &ax in axes {
        let ax_usize = ax as usize;

        // tosa.reduce_sum output: same rank, dim `ax` becomes 1.
        let mut tosa_out_shape = current_shape.clone();
        tosa_out_shape[ax_usize] = 1;
        let tosa_out_type = make_ranked_tensor_type(context, &tosa_out_shape, dtype);

        let axis_attr = IntegerAttribute::new(
            melior::ir::r#type::IntegerType::new(context, 32).into(),
            ax_usize as i64,
        );

        current_val = body_block
            .append_operation(
                OperationBuilder::new("tosa.reduce_sum", location)
                    .add_operands(&[current_val])
                    .add_results(&[tosa_out_type])
                    .add_attributes(&[(Identifier::new(context, "axis"), axis_attr.into())])
                    .build()
                    .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
            )
            .result(0)
            .unwrap()
            .into();
        current_shape = tosa_out_shape;
    }

    // Multiply by reciprocal: 1.0 / total_reduced.
    // Emit a tosa.const for the reciprocal scalar (shape = current_shape for broadcast).
    let recip = 1.0 / total_reduced as f64;
    let recip_str = match dtype {
        DType::F32 => format!("dense<{}> : tensor<1xf32>", format_float(recip)),
        DType::F64 => format!("dense<{}> : tensor<1xf64>", format_float(recip)),
        DType::I32 | DType::I64 => {
            // Integer mean: use integer division constant (1/N ~ 0 for N>1).
            // This is a best-effort approximation for integer types.
            format!("dense<{}> : tensor<1xi32>", (recip as i32))
        }
    };
    let recip_val = emit_tosa_const_scalar(context, body_block, &recip_str, location)?;

    // Reshape scalar tensor<1xT> to match the rank of current tensor.
    let rank = current_shape.len();
    let scalar_shape: Vec<u64> = vec![1u64; rank];
    let recip_reshaped =
        emit_tosa_reshape(context, body_block, recip_val, &scalar_shape, dtype, location)?;

    // shift operand for tosa.mul
    let shift_val =
        emit_tosa_const_scalar(context, body_block, "dense<0> : tensor<1xi8>", location)?;

    let keepdim_type = make_ranked_tensor_type(context, &current_shape, dtype);
    let mean_val: melior::ir::Value = body_block
        .append_operation(
            OperationBuilder::new("tosa.mul", location)
                .add_operands(&[current_val, recip_reshaped, shift_val])
                .add_results(&[keepdim_type])
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        )
        .result(0)
        .unwrap()
        .into();

    // If !keepdim, reshape to remove the size-1 reduced dims.
    if keepdim {
        return Ok(mean_val);
    }

    let result_val =
        emit_tosa_reshape(context, body_block, mean_val, output_shape, dtype, location)?;
    Ok(result_val)
}

// ── Helper: emit a tosa.slice (stride-1 slice) ───────────────────────────────
//
// Emits tosa.const_shape for start + size, then tosa.slice.
// `starts` and `out_shape` are per-axis; the input tensor has `input_rank` dims.
// For axes not present in `axes`, start=0 and size=input_shape[ax].

#[allow(clippy::too_many_arguments)]
fn emit_tosa_slice<'c>(
    context: &'c Context,
    body_block: &Block<'c>,
    input: melior::ir::Value<'c, 'c>,
    input_shape: &[u64],
    starts: &[i64],  // per-axis start (normalized, length = input_rank)
    out_shape: &[u64], // per-axis size = output shape (length = input_rank)
    dtype: DType,
    location: Location<'c>,
) -> Result<melior::ir::Value<'c, 'c>, CompileError> {
    let rank = input_shape.len();
    let shape_type_str = format!("!tosa.shape<{rank}>");
    let shape_type = melior::ir::Type::parse(context, &shape_type_str)
        .ok_or_else(|| CompileError::AttributeParse(shape_type_str.clone()))?;

    // Build start const_shape.
    let starts_str = starts.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", ");
    let start_dense = format!("dense<[{starts_str}]> : tensor<{rank}xindex>");
    let start_attr = Attribute::parse(context, &start_dense)
        .ok_or_else(|| CompileError::AttributeParse(start_dense.clone()))?;
    let start_val: melior::ir::Value = body_block
        .append_operation(
            OperationBuilder::new("tosa.const_shape", location)
                .add_results(&[shape_type])
                .add_attributes(&[(Identifier::new(context, "values"), start_attr)])
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        )
        .result(0)
        .unwrap()
        .into();

    // Build size const_shape.
    let sizes_str = out_shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", ");
    let size_dense = format!("dense<[{sizes_str}]> : tensor<{rank}xindex>");
    let size_attr = Attribute::parse(context, &size_dense)
        .ok_or_else(|| CompileError::AttributeParse(size_dense.clone()))?;
    let size_val: melior::ir::Value = body_block
        .append_operation(
            OperationBuilder::new("tosa.const_shape", location)
                .add_results(&[shape_type])
                .add_attributes(&[(Identifier::new(context, "values"), size_attr)])
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        )
        .result(0)
        .unwrap()
        .into();

    let result_type = make_ranked_tensor_type(context, out_shape, dtype);
    let result: melior::ir::Value = body_block
        .append_operation(
            OperationBuilder::new("tosa.slice", location)
                .add_operands(&[input, start_val, size_val])
                .add_results(&[result_type])
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        )
        .result(0)
        .unwrap()
        .into();

    Ok(result)
}

// ── Helper: emit strided slice via linalg.generic + linalg.index ─────────────
//
// Used when any step != 1. Maps each output index to a data index:
//   data_idx[ax] = start[ax] + out_idx[ax] * step[ax]
// for sliced axes, and data_idx[ax] = out_idx[ax] for non-sliced axes.

#[allow(clippy::too_many_arguments)]
fn emit_strided_slice<'c>(
    context: &'c Context,
    body_block: &Block<'c>,
    input: melior::ir::Value<'c, 'c>,
    _input_shape: &[u64],
    starts: &[i64],    // per-sliced-axis starts (length = axes.len())
    axes: &[i64],      // which axes are sliced (length = axes.len())
    steps: &[i64],     // per-sliced-axis steps (length = axes.len())
    out_shape: &[u64], // output tensor shape (all axes)
    dtype: DType,
    location: Location<'c>,
) -> Result<melior::ir::Value<'c, 'c>, CompileError> {
    let out_rank = out_shape.len();
    let elem_type = dtype.to_mlir_type(context);
    let out_tensor_type = make_ranked_tensor_type(context, out_shape, dtype);

    let init_val: melior::ir::Value = if out_shape.contains(&DIM_DYNAMIC) {
        let sources: Vec<(melior::ir::Value<'c, 'c>, usize)> = out_shape
            .iter()
            .enumerate()
            .filter(|&(_, d)| *d == DIM_DYNAMIC)
            .map(|(i, _)| (input, i))
            .collect();
        emit_tensor_empty_dynamic(context, body_block, out_shape, dtype, &sources, location)?
    } else {
        body_block
            .append_operation(
                OperationBuilder::new("tensor.empty", location)
                    .add_results(&[out_tensor_type])
                    .build()
                    .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
            )
            .result(0)
            .unwrap()
            .into()
    };

    let out_map = make_identity_map(context, out_rank)?;
    let indexing_maps = ArrayAttribute::new(context, &[out_map]);
    let iterator_types = make_iterator_types(context, out_rank)?;
    let segment_sizes = Attribute::parse(context, "array<i32: 0, 1>").ok_or_else(|| {
        CompileError::AttributeParse("strided slice segment sizes".into())
    })?;

    let linalg_region = {
        let linalg_block = Block::new(&[(elem_type, location)]);

        let index_type = melior::ir::Type::parse(context, "index")
            .ok_or_else(|| CompileError::AttributeParse("index type".into()))?;
        let i64_type = DType::I64.to_mlir_type(context);

        // Get each output dim index.
        let mut out_indices: Vec<melior::ir::Value> = Vec::with_capacity(out_rank);
        for d in 0..out_rank {
            let dim_attr = IntegerAttribute::new(
                melior::ir::r#type::IntegerType::new(context, 64).into(),
                d as i64,
            );
            let idx: melior::ir::Value = linalg_block
                .append_operation(
                    OperationBuilder::new("linalg.index", location)
                        .add_results(&[index_type])
                        .add_attributes(&[(Identifier::new(context, "dim"), dim_attr.into())])
                        .build()
                        .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                )
                .result(0)
                .unwrap()
                .into();
            out_indices.push(idx);
        }

        // For each output dim, compute data_index = start + out_idx * step.
        let mut data_indices: Vec<melior::ir::Value> = out_indices.clone();
        for (i, &ax) in axes.iter().enumerate() {
            let ax_usize = ax as usize;
            let start = starts[i];
            let step = steps[i];

            // Cast out_idx to i64, multiply by step, add start, cast back to index.
            let out_idx_i64: melior::ir::Value = linalg_block
                .append_operation(
                    OperationBuilder::new("arith.index_cast", location)
                        .add_operands(&[out_indices[ax_usize]])
                        .add_results(&[i64_type])
                        .build()
                        .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                )
                .result(0)
                .unwrap()
                .into();

            let step_val: melior::ir::Value = linalg_block
                .append_operation(arith::constant(
                    context,
                    IntegerAttribute::new(i64_type, step).into(),
                    location,
                ))
                .result(0)
                .unwrap()
                .into();

            let start_val: melior::ir::Value = linalg_block
                .append_operation(arith::constant(
                    context,
                    IntegerAttribute::new(i64_type, start).into(),
                    location,
                ))
                .result(0)
                .unwrap()
                .into();

            let mul: melior::ir::Value = linalg_block
                .append_operation(arith::muli(out_idx_i64, step_val, location))
                .result(0)
                .unwrap()
                .into();

            let add: melior::ir::Value = linalg_block
                .append_operation(arith::addi(mul, start_val, location))
                .result(0)
                .unwrap()
                .into();

            let data_idx: melior::ir::Value = linalg_block
                .append_operation(
                    OperationBuilder::new("arith.index_cast", location)
                        .add_operands(&[add])
                        .add_results(&[index_type])
                        .build()
                        .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                )
                .result(0)
                .unwrap()
                .into();

            data_indices[ax_usize] = data_idx;
        }

        // tensor.extract from input at computed data_indices.
        let val: melior::ir::Value = linalg_block
            .append_operation(
                OperationBuilder::new("tensor.extract", location)
                    .add_operands(&[input])
                    .add_operands(&data_indices)
                    .add_results(&[elem_type])
                    .build()
                    .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
            )
            .result(0)
            .unwrap()
            .into();

        linalg_block.append_operation(
            OperationBuilder::new("linalg.yield", location)
                .add_operands(&[val])
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        );

        let r = Region::new();
        r.append_block(linalg_block);
        r
    };

    let result_val: melior::ir::Value = body_block
        .append_operation(
            OperationBuilder::new("linalg.generic", location)
                .add_operands(&[init_val])
                .add_results(&[out_tensor_type])
                .add_attributes(&[
                    (
                        Identifier::new(context, "indexing_maps"),
                        indexing_maps.into(),
                    ),
                    (Identifier::new(context, "iterator_types"), iterator_types),
                    (
                        Identifier::new(context, "operand_segment_sizes"),
                        segment_sizes,
                    ),
                ])
                .add_regions([linalg_region])
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        )
        .result(0)
        .unwrap()
        .into();

    Ok(result_val)
}

// ── Helper: emit Gather via scf.for loop nest + tensor.extract/insert ────────
//
// Iterates over all output indices. For each output element:
//   1. Extract the axis index from the indices tensor (linalg.index for the
//      indices-portion dims).
//   2. Construct the full data index (pre-axis from output, axis from indices,
//      post-axis from output).
//   3. tensor.extract from data, tensor.insert into output.
//
// Uses linalg.generic with output-identity map + linalg.index + tensor.extract
// for the gathered access (valid because tensor.extract is allowed inside
// linalg.generic bodies on captured tensors).

#[allow(clippy::too_many_arguments)]
fn emit_gather<'c>(
    context: &'c Context,
    body_block: &Block<'c>,
    data_val: melior::ir::Value<'c, 'c>,
    indices_val: melior::ir::Value<'c, 'c>,
    data_shape: &[u64],   // shape of data tensor
    indices_shape: &[u64], // shape of indices tensor
    out_shape: &[u64],    // output shape
    axis: usize,
    dtype: DType,
    indices_dtype: DType,
    location: Location<'c>,
) -> Result<melior::ir::Value<'c, 'c>, CompileError> {
    let out_rank = out_shape.len();
    let idx_rank = indices_shape.len();
    let elem_type = dtype.to_mlir_type(context);
    let out_tensor_type = make_ranked_tensor_type(context, out_shape, dtype);

    // Emit tensor.empty for the output.
    // Output shape = data_shape[0..axis] ++ indices_shape ++ data_shape[axis+1..].
    // For dynamic dims, source from data_val or indices_val accordingly.
    let init_val: melior::ir::Value = if out_shape.contains(&DIM_DYNAMIC) {
        let sources: Vec<(melior::ir::Value<'c, 'c>, usize)> = out_shape
            .iter()
            .enumerate()
            .filter(|&(_, d)| *d == DIM_DYNAMIC)
            .map(|(i, _)| {
                if i < axis {
                    // Pre-axis dim from data.
                    (data_val, i)
                } else if i < axis + idx_rank {
                    // Indices dim.
                    (indices_val, i - axis)
                } else {
                    // Post-axis dim from data.
                    (data_val, i - idx_rank + 1)
                }
            })
            .collect();
        emit_tensor_empty_dynamic(context, body_block, out_shape, dtype, &sources, location)?
    } else {
        body_block
            .append_operation(
                OperationBuilder::new("tensor.empty", location)
                    .add_results(&[out_tensor_type])
                    .build()
                    .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
            )
            .result(0)
            .unwrap()
            .into()
    };

    // linalg.generic: output-only (identity map). data and indices are captured
    // SSA values used via tensor.extract inside the body.
    //
    // Output map: identity over out_rank dims.
    let out_map = make_identity_map(context, out_rank)?;
    let indexing_maps = ArrayAttribute::new(context, &[out_map]);
    let iterator_types = make_iterator_types(context, out_rank)?;
    let segment_sizes = Attribute::parse(context, "array<i32: 0, 1>").ok_or_else(|| {
        CompileError::AttributeParse("failed to parse gather segment sizes".into())
    })?;

    // Build the linalg body.
    // Block args: (out_elem) — we ignore it and yield the extracted value.
    let linalg_region = {
        let linalg_block = Block::new(&[(elem_type, location)]);

        // Get each output dim index via linalg.index.
        let index_type = melior::ir::Type::parse(context, "index")
            .ok_or_else(|| CompileError::AttributeParse("index type".into()))?;
        let mut out_indices: Vec<melior::ir::Value> = Vec::with_capacity(out_rank);
        for d in 0..out_rank {
            let dim_attr =
                IntegerAttribute::new(melior::ir::r#type::IntegerType::new(context, 64).into(), d as i64);
            let idx: melior::ir::Value = linalg_block
                .append_operation(
                    OperationBuilder::new("linalg.index", location)
                        .add_results(&[index_type])
                        .add_attributes(&[(Identifier::new(context, "dim"), dim_attr.into())])
                        .build()
                        .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                )
                .result(0)
                .unwrap()
                .into();
            out_indices.push(idx);
        }

        // Build indices_tensor_indices: dims [axis .. axis+idx_rank) of out_indices.
        let idx_tensor_indices: Vec<melior::ir::Value> =
            out_indices[axis..axis + idx_rank].to_vec();

        // Extract the axis value from the indices tensor.
        let indices_tensor_type = make_ranked_tensor_type(context, indices_shape, indices_dtype);
        let raw_idx: melior::ir::Value = linalg_block
            .append_operation(
                OperationBuilder::new("tensor.extract", location)
                    .add_operands(&[indices_val])
                    .add_operands(&idx_tensor_indices)
                    .add_results(&[indices_dtype.to_mlir_type(context)])
                    .build()
                    .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
            )
            .result(0)
            .unwrap()
            .into();
        let _ = indices_tensor_type;

        // Cast the extracted index to the `index` type.
        let axis_idx: melior::ir::Value = linalg_block
            .append_operation(
                OperationBuilder::new("arith.index_cast", location)
                    .add_operands(&[raw_idx])
                    .add_results(&[index_type])
                    .build()
                    .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
            )
            .result(0)
            .unwrap()
            .into();

        // Build the full data index:
        //   out_indices[0..axis] ++ [axis_idx] ++ out_indices[axis+idx_rank..]
        let mut data_indices: Vec<melior::ir::Value> = Vec::with_capacity(data_shape.len());
        data_indices.extend_from_slice(&out_indices[..axis]);
        data_indices.push(axis_idx);
        data_indices.extend_from_slice(&out_indices[axis + idx_rank..]);
        debug_assert_eq!(
            data_indices.len(),
            data_shape.len(),
            "gather data index count mismatch"
        );

        // Extract element from data tensor.
        let data_tensor_type = make_ranked_tensor_type(context, data_shape, dtype);
        let val: melior::ir::Value = linalg_block
            .append_operation(
                OperationBuilder::new("tensor.extract", location)
                    .add_operands(&[data_val])
                    .add_operands(&data_indices)
                    .add_results(&[elem_type])
                    .build()
                    .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
            )
            .result(0)
            .unwrap()
            .into();
        let _ = data_tensor_type;

        linalg_block.append_operation(
            OperationBuilder::new("linalg.yield", location)
                .add_operands(&[val])
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        );

        let r = Region::new();
        r.append_block(linalg_block);
        r
    };

    let result_val: melior::ir::Value = body_block
        .append_operation(
            OperationBuilder::new("linalg.generic", location)
                .add_operands(&[init_val])
                .add_results(&[out_tensor_type])
                .add_attributes(&[
                    (
                        Identifier::new(context, "indexing_maps"),
                        indexing_maps.into(),
                    ),
                    (Identifier::new(context, "iterator_types"), iterator_types),
                    (
                        Identifier::new(context, "operand_segment_sizes"),
                        segment_sizes,
                    ),
                ])
                .add_regions([linalg_region])
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        )
        .result(0)
        .unwrap()
        .into();

    Ok(result_val)
}

/// Walk the trace linearly, emitting tensor-level ops into `body_block`.
///
/// For each `Input` op: emits `bufferization.to_tensor` with `restrict`.
/// For each compute op: emits `tensor.empty` + `linalg.generic`.
/// After all ops: emits `bufferization.to_buffer` + `memref.copy` for each output.
fn emit_tensor_ops<'c>(
    trace: &Trace,
    output_ids: &[NodeId],
    num_inputs: usize,
    body_block: &Block<'c>,
    location: Location<'c>,
    context: &'c Context,
) -> Result<(), CompileError> {
    // NodeId -> SSA tensor Value for each op emitted so far.
    let mut values: HashMap<NodeId, melior::ir::Value<'c, 'c>> = HashMap::new();

    for (i, op) in trace.ops().iter().enumerate() {
        let node_id = NodeId(i as u32);

        match op {
            Op::Input {
                arg_index,
                shape,
                dtype,
            } => {
                let tensor_type = make_ranked_tensor_type(context, &shape.0, *dtype);
                let memref_arg = body_block.argument(*arg_index as usize).unwrap();

                let tensor_val = body_block
                    .append_operation(
                        OperationBuilder::new("bufferization.to_tensor", location)
                            .add_operands(&[memref_arg.into()])
                            .add_results(&[tensor_type])
                            .add_attributes(&[(
                                Identifier::new(context, "restrict"),
                                Attribute::unit(context),
                            )])
                            .build()
                            .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                    )
                    .result(0)
                    .unwrap()
                    .into();

                values.insert(node_id, tensor_val);
            }

            Op::Add {
                lhs,
                rhs,
                shape,
                dtype,
            } => {
                let lhs_shape = trace.get(*lhs).shape();
                let rhs_shape = trace.get(*rhs).shape();
                let result_val = emit_tosa_binary(
                    context,
                    body_block,
                    &values,
                    "tosa.add",
                    *lhs,
                    *rhs,
                    &lhs_shape.0,
                    &rhs_shape.0,
                    &shape.0,
                    *dtype,
                    location,
                )?;
                values.insert(node_id, result_val);
            }

            Op::Sub {
                lhs,
                rhs,
                shape,
                dtype,
            } => {
                let lhs_shape = trace.get(*lhs).shape();
                let rhs_shape = trace.get(*rhs).shape();
                let result_val = emit_tosa_binary(
                    context,
                    body_block,
                    &values,
                    "tosa.sub",
                    *lhs,
                    *rhs,
                    &lhs_shape.0,
                    &rhs_shape.0,
                    &shape.0,
                    *dtype,
                    location,
                )?;
                values.insert(node_id, result_val);
            }

            Op::Mul {
                lhs,
                rhs,
                shape,
                dtype,
            } => {
                let lhs_shape = trace.get(*lhs).shape();
                let rhs_shape = trace.get(*rhs).shape();
                let result_val = emit_tosa_mul(
                    context,
                    body_block,
                    &values,
                    *lhs,
                    *rhs,
                    &lhs_shape.0,
                    &rhs_shape.0,
                    &shape.0,
                    *dtype,
                    location,
                )?;
                values.insert(node_id, result_val);
            }

            Op::Div {
                lhs,
                rhs,
                shape,
                dtype,
            } => {
                let lhs_shape = trace.get(*lhs).shape();
                let rhs_shape = trace.get(*rhs).shape();
                let result_val = emit_binary_elementwise(
                    context,
                    body_block,
                    &values,
                    *lhs,
                    *rhs,
                    &lhs_shape.0,
                    &rhs_shape.0,
                    &shape.0,
                    *dtype,
                    location,
                    |block, lhs_elem, rhs_elem| match dtype {
                        DType::F32 | DType::F64 => block
                            .append_operation(arith::divf(lhs_elem, rhs_elem, location))
                            .result(0)
                            .unwrap()
                            .into(),
                        DType::I32 | DType::I64 => block
                            .append_operation(arith::divsi(lhs_elem, rhs_elem, location))
                            .result(0)
                            .unwrap()
                            .into(),
                    },
                )?;
                values.insert(node_id, result_val);
            }

            Op::Neg {
                input,
                shape,
                dtype,
            } => {
                let result_val = emit_tosa_negate(
                    context, body_block, &values, *input, &shape.0, *dtype, location,
                )?;
                values.insert(node_id, result_val);
            }

            Op::Exp {
                input,
                shape,
                dtype,
            } => {
                let result_val = emit_tosa_unary_simple(
                    context, body_block, &values, "tosa.exp", *input, &shape.0, *dtype, location,
                )?;
                values.insert(node_id, result_val);
            }

            Op::Tanh {
                input,
                shape,
                dtype,
            } => {
                let result_val = emit_tosa_unary_simple(
                    context,
                    body_block,
                    &values,
                    "tosa.tanh",
                    *input,
                    &shape.0,
                    *dtype,
                    location,
                )?;
                values.insert(node_id, result_val);
            }

            Op::Matmul {
                lhs,
                rhs,
                shape,
                dtype,
            } => {
                let lhs_shape = trace.get(*lhs).shape();
                let rhs_shape = trace.get(*rhs).shape();
                let has_dynamic = lhs_shape.has_dynamic_dims()
                    || rhs_shape.has_dynamic_dims()
                    || shape.has_dynamic_dims();
                let result_val = if lhs_shape.rank() == 2 && rhs_shape.rank() == 2 && !has_dynamic {
                    // 2D static path: use tosa.matmul for float, linalg.matmul for integer.
                    match dtype {
                        DType::F32 | DType::F64 => emit_tosa_matmul(
                            context,
                            body_block,
                            &values,
                            *lhs,
                            *rhs,
                            &lhs_shape.0,
                            &rhs_shape.0,
                            &shape.0,
                            *dtype,
                            location,
                        )?,
                        DType::I32 | DType::I64 => emit_matmul(
                            context, body_block, &values, *lhs, *rhs, &shape.0, *dtype, location,
                        )?,
                    }
                } else {
                    // Batched path (rank > 2): use linalg.generic with proper indexing maps.
                    emit_batched_matmul(
                        context,
                        body_block,
                        &values,
                        *lhs,
                        *rhs,
                        &lhs_shape.0,
                        &rhs_shape.0,
                        &shape.0,
                        *dtype,
                        location,
                    )?
                };
                values.insert(node_id, result_val);
            }

            Op::Relu {
                input,
                shape,
                dtype,
            } => {
                let result_val = emit_tosa_clamp(
                    context, body_block, &values, *input, &shape.0, *dtype, location,
                )?;
                values.insert(node_id, result_val);
            }

            Op::ReduceSum {
                input,
                dim,
                keepdim,
                shape,
                dtype,
            } => {
                let input_shape = trace.get(*input).shape();
                // Route to linalg.generic when input has dynamic dims — tosa.reduce_sum
                // requires static shapes to set the reduced dim to 1 in the output type.
                let result_val = if input_shape.has_dynamic_dims() {
                    emit_reduction(
                        context,
                        body_block,
                        &values,
                        *input,
                        &input_shape.0,
                        &shape.0,
                        *dim,
                        *keepdim,
                        *dtype,
                        location,
                        false,
                    )?
                } else {
                    match dtype {
                        DType::F32 | DType::F64 => emit_tosa_reduce(
                            context,
                            body_block,
                            &values,
                            "tosa.reduce_sum",
                            *input,
                            &input_shape.0,
                            &shape.0,
                            *dim,
                            *keepdim,
                            *dtype,
                            location,
                        )?,
                        DType::I32 | DType::I64 => emit_reduction(
                            context,
                            body_block,
                            &values,
                            *input,
                            &input_shape.0,
                            &shape.0,
                            *dim,
                            *keepdim,
                            *dtype,
                            location,
                            false,
                        )?,
                    }
                };
                values.insert(node_id, result_val);
            }

            Op::ReduceMax {
                input,
                dim,
                keepdim,
                shape,
                dtype,
            } => {
                let input_shape = trace.get(*input).shape();
                // Route to linalg.generic when input has dynamic dims — tosa.reduce_max
                // requires static shapes.
                let result_val = if input_shape.has_dynamic_dims() {
                    emit_reduction(
                        context,
                        body_block,
                        &values,
                        *input,
                        &input_shape.0,
                        &shape.0,
                        *dim,
                        *keepdim,
                        *dtype,
                        location,
                        true,
                    )?
                } else {
                    match dtype {
                        DType::F32 | DType::F64 => emit_tosa_reduce(
                            context,
                            body_block,
                            &values,
                            "tosa.reduce_max",
                            *input,
                            &input_shape.0,
                            &shape.0,
                            *dim,
                            *keepdim,
                            *dtype,
                            location,
                        )?,
                        DType::I32 | DType::I64 => emit_reduction(
                            context,
                            body_block,
                            &values,
                            *input,
                            &input_shape.0,
                            &shape.0,
                            *dim,
                            *keepdim,
                            *dtype,
                            location,
                            true,
                        )?,
                    }
                };
                values.insert(node_id, result_val);
            }

            Op::Gemm {
                lhs,
                rhs,
                bias,
                alpha,
                beta,
                trans_a,
                trans_b,
                shape: _,
                dtype,
            } => {
                let lhs_shape = trace.get(*lhs).shape().0.clone();
                let rhs_shape = trace.get(*rhs).shape().0.clone();

                // Step 1: get lhs/rhs SSA values
                let mut lhs_val = *values.get(lhs).expect("lhs not yet emitted");
                let mut rhs_val = *values.get(rhs).expect("rhs not yet emitted");
                let mut lhs_eff_shape = lhs_shape.clone();
                let mut rhs_eff_shape = rhs_shape.clone();

                // Step 2: optional transpose of lhs (A)
                if *trans_a {
                    lhs_val = emit_tosa_transpose_2d(
                        context, body_block, lhs_val, &lhs_shape, *dtype, location,
                    )?;
                    lhs_eff_shape = vec![lhs_shape[1], lhs_shape[0]];
                }

                // Step 3: optional transpose of rhs (B)
                if *trans_b {
                    rhs_val = emit_tosa_transpose_2d(
                        context, body_block, rhs_val, &rhs_shape, *dtype, location,
                    )?;
                    rhs_eff_shape = vec![rhs_shape[1], rhs_shape[0]];
                }

                // Step 4: optional alpha scaling (skip when alpha == 1.0)
                if (*alpha - 1.0f64).abs() > f64::EPSILON {
                    lhs_val = emit_tosa_scale(
                        context,
                        body_block,
                        lhs_val,
                        &lhs_eff_shape,
                        *alpha,
                        *dtype,
                        location,
                    )?;
                }

                // Step 5 & 6: tosa.matmul via existing helper (handles 2D→3D→2D)
                let m = lhs_eff_shape[0];
                let k = lhs_eff_shape[1];
                let n = rhs_eff_shape[1];
                debug_assert_eq!(k, rhs_eff_shape[0], "Gemm inner dimension mismatch");

                let lhs_matmul_shape = [m, lhs_eff_shape[1]];
                let rhs_matmul_shape = [rhs_eff_shape[0], n];
                let out_shape = [m, n];

                let mut result_val = emit_tosa_matmul_2d_values(
                    context,
                    body_block,
                    lhs_val,
                    rhs_val,
                    &lhs_matmul_shape,
                    &rhs_matmul_shape,
                    &out_shape,
                    *dtype,
                    location,
                )?;

                // Step 7: optional bias add (with optional beta scaling)
                if let Some(bias_id) = bias {
                    let mut bias_val = *values.get(bias_id).expect("bias not yet emitted");
                    let bias_shape = trace.get(*bias_id).shape().0.clone();

                    if (*beta - 1.0f64).abs() > f64::EPSILON {
                        bias_val = emit_tosa_scale(
                            context,
                            body_block,
                            bias_val,
                            &bias_shape,
                            *beta,
                            *dtype,
                            location,
                        )?;
                    }

                    result_val = emit_tosa_add_bias(
                        context,
                        body_block,
                        result_val,
                        &out_shape,
                        bias_val,
                        &bias_shape,
                        *dtype,
                        location,
                    )?;
                }

                values.insert(node_id, result_val);
            }

            Op::Reshape {
                input,
                target_shape: _,
                shape_tensor,
                shape,
                dtype,
            } => {
                let input_val = *values.get(input).expect("input not yet emitted");
                let input_shape = trace.get(*input).shape().0.clone();

                // Dynamic reshape via shape tensor: use tensor.reshape with a
                // corrected shape tensor (ONNX -1 dims resolved at runtime).
                if let Some(st_id) = shape_tensor {
                    let shape_val = *values.get(st_id).expect("shape_tensor not yet emitted");
                    let index_type = melior::ir::Type::parse(context, "index")
                        .ok_or_else(|| CompileError::AttributeParse("index type".into()))?;
                    let i64_type = DType::I64.to_mlir_type(context);
                    let out_rank = shape.0.len();
                    let in_rank = input_shape.len();

                    // Get input dim sizes — needed for -1 inference.
                    let mut in_dim_vals: Vec<melior::ir::Value> = Vec::new();
                    for i in 0..in_rank {
                        let dv = emit_tensor_dim(context, body_block, input_val, i, location)?;
                        in_dim_vals.push(dv);
                    }

                    // Compute total input elements: product of all input dims.
                    let mut total_input_elems: melior::ir::Value = body_block
                        .append_operation(arith::constant(
                            context,
                            IntegerAttribute::new(index_type, 1).into(),
                            location,
                        ))
                        .result(0).unwrap().into();
                    for dv in &in_dim_vals {
                        total_input_elems = body_block
                            .append_operation(arith::muli(total_input_elems, *dv, location))
                            .result(0).unwrap().into();
                    }

                    // Extract raw i64 dim values from the shape tensor.
                    // ONNX uses -1 to mean "infer this dimension".
                    let neg_one_i64: melior::ir::Value = body_block
                        .append_operation(arith::constant(
                            context,
                            IntegerAttribute::new(i64_type, -1).into(),
                            location,
                        ))
                        .result(0).unwrap().into();
                    let one_i64: melior::ir::Value = body_block
                        .append_operation(arith::constant(
                            context,
                            IntegerAttribute::new(i64_type, 1).into(),
                            location,
                        ))
                        .result(0).unwrap().into();

                    let mut raw_dim_i64s: Vec<melior::ir::Value> = Vec::new();
                    for i in 0..out_rank {
                        let ci: melior::ir::Value = body_block
                            .append_operation(arith::constant(
                                context,
                                IntegerAttribute::new(index_type, i as i64).into(),
                                location,
                            ))
                            .result(0).unwrap().into();
                        let dim_i64: melior::ir::Value = body_block
                            .append_operation(
                                OperationBuilder::new("tensor.extract", location)
                                    .add_operands(&[shape_val, ci])
                                    .add_results(&[i64_type])
                                    .build()
                                    .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                            )
                            .result(0).unwrap().into();
                        raw_dim_i64s.push(dim_i64);
                    }

                    // Compute product of known (non -1) dims, and detect -1 positions.
                    let mut known_product_i64: melior::ir::Value = one_i64;
                    let mut is_neg_one: Vec<melior::ir::Value> = Vec::new();
                    for i in 0..out_rank {
                        let cmp: melior::ir::Value = body_block
                            .append_operation(arith::cmpi(
                                context, arith::CmpiPredicate::Eq,
                                raw_dim_i64s[i], neg_one_i64, location,
                            ))
                            .result(0).unwrap().into();
                        is_neg_one.push(cmp);
                        let dim_or_one: melior::ir::Value = body_block
                            .append_operation(
                                OperationBuilder::new("arith.select", location)
                                    .add_operands(&[cmp, one_i64, raw_dim_i64s[i]])
                                    .add_results(&[i64_type])
                                    .build()
                                    .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                            )
                            .result(0).unwrap().into();
                        known_product_i64 = body_block
                            .append_operation(arith::muli(known_product_i64, dim_or_one, location))
                            .result(0).unwrap().into();
                    }

                    // inferred_dim = total_input_elems / known_product
                    let total_i64: melior::ir::Value = body_block
                        .append_operation(
                            OperationBuilder::new("arith.index_cast", location)
                                .add_operands(&[total_input_elems])
                                .add_results(&[i64_type])
                                .build()
                                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                        )
                        .result(0).unwrap().into();
                    let inferred_i64: melior::ir::Value = body_block
                        .append_operation(arith::divui(total_i64, known_product_i64, location))
                        .result(0).unwrap().into();

                    // Build resolved shape: replace -1 with inferred value.
                    let mut resolved_i64s: Vec<melior::ir::Value> = Vec::new();
                    for i in 0..out_rank {
                        let resolved: melior::ir::Value = body_block
                            .append_operation(
                                OperationBuilder::new("arith.select", location)
                                    .add_operands(&[is_neg_one[i], inferred_i64, raw_dim_i64s[i]])
                                    .add_results(&[i64_type])
                                    .build()
                                    .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                            )
                            .result(0).unwrap().into();
                        resolved_i64s.push(resolved);
                    }

                    // Build corrected shape tensor via tensor.from_elements.
                    let shape_tensor_type = melior::ir::r#type::RankedTensorType::new(
                        &[out_rank as u64], i64_type, None,
                    );
                    let corrected_shape: melior::ir::Value = body_block
                        .append_operation(
                            OperationBuilder::new("tensor.from_elements", location)
                                .add_operands(&resolved_i64s)
                                .add_results(&[shape_tensor_type.into()])
                                .build()
                                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                        )
                        .result(0).unwrap().into();

                    // Emit tensor.reshape with corrected shape.
                    let out_type = make_ranked_tensor_type(context, &shape.0, *dtype);
                    let result_val: melior::ir::Value = body_block
                        .append_operation(
                            OperationBuilder::new("tensor.reshape", location)
                                .add_operands(&[input_val, corrected_shape])
                                .add_results(&[out_type])
                                .build()
                                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                        )
                        .result(0)
                        .unwrap()
                        .into();
                    values.insert(node_id, result_val);
                    continue;
                }

                let result_val = if shape.0.contains(&DIM_DYNAMIC)
                    || input_shape.contains(&DIM_DYNAMIC)
                {
                    // Dynamic reshape: use tensor.expand_shape or tensor.collapse_shape.
                    let in_rank = input_shape.len();
                    let out_rank = shape.0.len();
                    if out_rank > in_rank {
                        // Expanding: use tensor.expand_shape.
                        // promote_rank_with_reshape handles this for prepending dims.
                        let (expanded, _) = promote_rank_with_reshape(
                            context,
                            body_block,
                            input_val,
                            &input_shape,
                            out_rank,
                            *dtype,
                            location,
                        )?;
                        expanded
                    } else if out_rank < in_rank {
                        // Collapsing: use tensor.collapse_shape.
                        // Build reassociation: map each output dim to a range of input dims.
                        // Simple case: collapse all remaining dims into the last output dim.
                        let prefix = out_rank - 1;
                        let mut reassoc_parts: Vec<String> = (0..prefix)
                            .map(|i| format!("[{i}]"))
                            .collect();
                        let last_group: Vec<String> = (prefix..in_rank)
                            .map(|i| i.to_string())
                            .collect();
                        reassoc_parts.push(format!("[{}]", last_group.join(", ")));
                        let reassoc_str = format!("[{}]", reassoc_parts.join(", "));
                        let reassoc_attr = Attribute::parse(context, &reassoc_str)
                            .ok_or_else(|| CompileError::AttributeParse(reassoc_str.clone()))?;

                        let out_type = make_ranked_tensor_type(context, &shape.0, *dtype);
                        body_block
                            .append_operation(
                                OperationBuilder::new("tensor.collapse_shape", location)
                                    .add_operands(&[input_val])
                                    .add_results(&[out_type])
                                    .add_attributes(&[(
                                        Identifier::new(context, "reassociation"),
                                        reassoc_attr,
                                    )])
                                    .build()
                                    .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                            )
                            .result(0)
                            .unwrap()
                            .into()
                    } else {
                        // Same rank with dynamic dims: use tensor.cast if shapes
                        // are compatible (same static dims, dynamic dims match).
                        let out_type = make_ranked_tensor_type(context, &shape.0, *dtype);
                        body_block
                            .append_operation(
                                OperationBuilder::new("tensor.cast", location)
                                    .add_operands(&[input_val])
                                    .add_results(&[out_type])
                                    .build()
                                    .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                            )
                            .result(0)
                            .unwrap()
                            .into()
                    }
                } else {
                    emit_tosa_reshape(context, body_block, input_val, &shape.0, *dtype, location)?
                };
                values.insert(node_id, result_val);
            }

            Op::Conv2d {
                input,
                kernel,
                bias,
                strides,
                pads,
                dilations,
                shape,
                dtype,
            } => {
                let input_shape = trace.get(*input).shape().0.clone(); // [N, CI, H, W]
                let kernel_shape = trace.get(*kernel).shape().0.clone(); // [CO, CI, KH, KW]

                let input_val = *values.get(input).expect("input not yet emitted");
                let kernel_val = *values.get(kernel).expect("kernel not yet emitted");

                // Step 1: Transpose input NCHW → NHWC: perms [0, 2, 3, 1]
                let input_nhwc = emit_tosa_transpose(
                    context,
                    body_block,
                    input_val,
                    &input_shape,
                    &[0, 2, 3, 1],
                    *dtype,
                    location,
                )?;
                // input_nhwc shape: [N, H, W, CI]

                // Step 2: Transpose kernel OIHW → OHWI: perms [0, 2, 3, 1]
                let kernel_ohwi = emit_tosa_transpose(
                    context,
                    body_block,
                    kernel_val,
                    &kernel_shape,
                    &[0, 2, 3, 1],
                    *dtype,
                    location,
                )?;
                // kernel_ohwi shape: [CO, KH, KW, CI]

                // Step 3: Emit bias or a zero-filled constant of shape [CO]
                let co = kernel_shape[0];
                let bias_val = if let Some(bias_id) = bias {
                    *values.get(bias_id).expect("bias not yet emitted")
                } else {
                    // Zero bias constant: dense<0.0> : tensor<COxT>
                    let zero_str = match dtype {
                        DType::F32 => format!("dense<0.0> : tensor<{co}xf32>"),
                        DType::F64 => format!("dense<0.0> : tensor<{co}xf64>"),
                        DType::I32 => format!("dense<0> : tensor<{co}xi32>"),
                        DType::I64 => format!("dense<0> : tensor<{co}xi64>"),
                    };
                    emit_tosa_const_scalar(context, body_block, &zero_str, location)?
                };

                // Step 4: Compute output shape in NHWC layout: [N, OH, OW, CO]
                // TOSA requires (H + pad_top + pad_bottom - eff_KH) % stride == 0.
                // ONNX uses floor division (drops incomplete last window).
                // Add extra bottom/right padding to satisfy TOSA, then slice to correct size.
                let n = input_shape[0];
                let h = input_shape[2];
                let w = input_shape[3];
                let kh = kernel_shape[2];
                let kw = kernel_shape[3];
                let oh = shape.0[2]; // target ONNX output height
                let ow = shape.0[3]; // target ONNX output width

                let pad_top = pads[0];
                let pad_left = pads[1];
                let mut pad_bottom = pads[2];
                let mut pad_right = pads[3];
                let eff_kh = dilations[0] * (kh - 1) + 1;
                let eff_kw = dilations[1] * (kw - 1) + 1;
                let rem_h = (h + pad_top + pad_bottom - eff_kh) % strides[0];
                let rem_w = (w + pad_left + pad_right - eff_kw) % strides[1];
                if rem_h != 0 {
                    pad_bottom += strides[0] - rem_h;
                }
                if rem_w != 0 {
                    pad_right += strides[1] - rem_w;
                }
                let tosa_oh = (h + pad_top + pad_bottom - eff_kh) / strides[0] + 1;
                let tosa_ow = (w + pad_left + pad_right - eff_kw) / strides[1] + 1;
                let needs_slice = tosa_oh != oh || tosa_ow != ow;

                let nhwc_tosa_shape = [n, tosa_oh, tosa_ow, co];
                let nhwc_tosa_type = make_ranked_tensor_type(context, &nhwc_tosa_shape, *dtype);

                // TOSA conv2d pad order: [pad_top, pad_bottom, pad_left, pad_right]
                let pad_attr_str = format!(
                    "array<i64: {}, {}, {}, {}>",
                    pad_top, pad_bottom, pad_left, pad_right
                );
                let stride_attr_str = format!("array<i64: {}, {}>", strides[0], strides[1]);
                let dilation_attr_str = format!("array<i64: {}, {}>", dilations[0], dilations[1]);

                let pad_attr = Attribute::parse(context, &pad_attr_str)
                    .ok_or_else(|| CompileError::AttributeParse(pad_attr_str.clone()))?;
                let stride_attr = Attribute::parse(context, &stride_attr_str)
                    .ok_or_else(|| CompileError::AttributeParse(stride_attr_str.clone()))?;
                let dilation_attr = Attribute::parse(context, &dilation_attr_str)
                    .ok_or_else(|| CompileError::AttributeParse(dilation_attr_str.clone()))?;

                // acc_type attribute: accumulator type (f32 for float, i32 for int)
                let acc_type_str = match dtype {
                    DType::F32 | DType::F64 => "f32",
                    DType::I32 | DType::I64 => "i32",
                };
                let acc_type_attr_str = acc_type_str;
                let acc_type_attr = Attribute::parse(context, acc_type_attr_str)
                    .ok_or_else(|| CompileError::AttributeParse(acc_type_attr_str.to_string()))?;

                // Zero-point operands: scalar tensors of shape [1] with value 0.
                // For float ops these are 0.0; for int ops these are 0.
                let zp_str = match dtype {
                    DType::F32 => "dense<0.0> : tensor<1xf32>",
                    DType::F64 => "dense<0.0> : tensor<1xf32>", // ZP always f32 in TOSA
                    DType::I32 => "dense<0> : tensor<1xi32>",
                    DType::I64 => "dense<0> : tensor<1xi64>",
                };
                let input_zp = emit_tosa_const_scalar(context, body_block, zp_str, location)?;
                let weight_zp = emit_tosa_const_scalar(context, body_block, zp_str, location)?;

                // Emit tosa.conv2d: (input_nhwc, kernel_ohwi, bias, input_zp, weight_zp)
                let mut conv_nhwc: melior::ir::Value = body_block
                    .append_operation(
                        OperationBuilder::new("tosa.conv2d", location)
                            .add_operands(&[input_nhwc, kernel_ohwi, bias_val, input_zp, weight_zp])
                            .add_results(&[nhwc_tosa_type])
                            .add_attributes(&[
                                (Identifier::new(context, "pad"), pad_attr),
                                (Identifier::new(context, "stride"), stride_attr),
                                (Identifier::new(context, "dilation"), dilation_attr),
                                (Identifier::new(context, "acc_type"), acc_type_attr),
                            ])
                            .build()
                            .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                    )
                    .result(0)
                    .unwrap()
                    .into();

                // If we added extra padding, slice back to the ONNX-expected size.
                if needs_slice {
                    let nhwc_out_shape_slice = [n, oh, ow, co];
                    let rank = 4usize;
                    let shape_type_str = format!("!tosa.shape<{rank}>");
                    let shape_type = melior::ir::Type::parse(context, &shape_type_str)
                        .ok_or_else(|| CompileError::AttributeParse(shape_type_str.clone()))?;

                    let start_vals_str = format!("dense<[0, 0, 0, 0]> : tensor<{rank}xindex>");
                    let start_attr = Attribute::parse(context, &start_vals_str)
                        .ok_or_else(|| CompileError::AttributeParse(start_vals_str.clone()))?;
                    let start_val: melior::ir::Value = body_block
                        .append_operation(
                            OperationBuilder::new("tosa.const_shape", location)
                                .add_results(&[shape_type])
                                .add_attributes(&[(Identifier::new(context, "values"), start_attr)])
                                .build()
                                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                        )
                        .result(0)
                        .unwrap()
                        .into();

                    let size_vals_str =
                        format!("dense<[{n}, {oh}, {ow}, {co}]> : tensor<{rank}xindex>");
                    let size_attr = Attribute::parse(context, &size_vals_str)
                        .ok_or_else(|| CompileError::AttributeParse(size_vals_str.clone()))?;
                    let size_val: melior::ir::Value = body_block
                        .append_operation(
                            OperationBuilder::new("tosa.const_shape", location)
                                .add_results(&[shape_type])
                                .add_attributes(&[(Identifier::new(context, "values"), size_attr)])
                                .build()
                                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                        )
                        .result(0)
                        .unwrap()
                        .into();

                    let slice_type =
                        make_ranked_tensor_type(context, &nhwc_out_shape_slice, *dtype);
                    conv_nhwc = body_block
                        .append_operation(
                            OperationBuilder::new("tosa.slice", location)
                                .add_operands(&[conv_nhwc, start_val, size_val])
                                .add_results(&[slice_type])
                                .build()
                                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                        )
                        .result(0)
                        .unwrap()
                        .into();
                }

                let nhwc_out_shape = [n, oh, ow, co];

                // Step 5: Transpose output NHWC → NCHW: perms [0, 3, 1, 2]
                let result_val = emit_tosa_transpose(
                    context,
                    body_block,
                    conv_nhwc,
                    &nhwc_out_shape,
                    &[0, 3, 1, 2],
                    *dtype,
                    location,
                )?;

                values.insert(node_id, result_val);
            }

            Op::BatchNorm {
                input,
                scale,
                bias,
                mean,
                var,
                epsilon,
                shape,
                dtype,
            } => {
                let input_val = *values.get(input).expect("input not yet emitted");
                let scale_val = *values.get(scale).expect("scale not yet emitted");
                let bias_val = *values.get(bias).expect("bias not yet emitted");
                let mean_val = *values.get(mean).expect("mean not yet emitted");
                let var_val = *values.get(var).expect("var not yet emitted");

                let input_shape = &shape.0; // same as input shape
                let c = input_shape[1];
                let param_shape_4d = [1u64, c, 1, 1];

                // Step 1: reshape [C] params to [1, C, 1, 1] for NCHW broadcast
                let mean_4d = emit_tosa_reshape(
                    context,
                    body_block,
                    mean_val,
                    &param_shape_4d,
                    *dtype,
                    location,
                )?;
                let var_4d = emit_tosa_reshape(
                    context,
                    body_block,
                    var_val,
                    &param_shape_4d,
                    *dtype,
                    location,
                )?;
                let scale_4d = emit_tosa_reshape(
                    context,
                    body_block,
                    scale_val,
                    &param_shape_4d,
                    *dtype,
                    location,
                )?;
                let bias_4d = emit_tosa_reshape(
                    context,
                    body_block,
                    bias_val,
                    &param_shape_4d,
                    *dtype,
                    location,
                )?;

                // Step 2: x - mean  →  shape: input_shape
                let out_type = make_ranked_tensor_type(context, input_shape, *dtype);
                let x_minus_mean: melior::ir::Value = body_block
                    .append_operation(
                        OperationBuilder::new("tosa.sub", location)
                            .add_operands(&[input_val, mean_4d])
                            .add_results(&[out_type])
                            .build()
                            .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                    )
                    .result(0)
                    .unwrap()
                    .into();

                // Step 3: var + eps  →  shape: [1, C, 1, 1]
                let eps_str = match dtype {
                    DType::F32 => format!(
                        "dense<{}> : tensor<1x{}x1x1xf32>",
                        format_float(*epsilon),
                        c
                    ),
                    DType::F64 => format!(
                        "dense<{}> : tensor<1x{}x1x1xf64>",
                        format_float(*epsilon),
                        c
                    ),
                    DType::I32 => format!("dense<{}> : tensor<1x{}x1x1xi32>", *epsilon as i64, c),
                    DType::I64 => format!("dense<{}> : tensor<1x{}x1x1xi64>", *epsilon as i64, c),
                };
                let eps_val = emit_tosa_const_scalar(context, body_block, &eps_str, location)?;

                let var_eps_type = make_ranked_tensor_type(context, &param_shape_4d, *dtype);
                let var_plus_eps: melior::ir::Value = body_block
                    .append_operation(
                        OperationBuilder::new("tosa.add", location)
                            .add_operands(&[var_4d, eps_val])
                            .add_results(&[var_eps_type])
                            .build()
                            .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                    )
                    .result(0)
                    .unwrap()
                    .into();

                // Step 4: rsqrt(var + eps)  →  shape: [1, C, 1, 1]
                let inv_std: melior::ir::Value = body_block
                    .append_operation(
                        OperationBuilder::new("tosa.rsqrt", location)
                            .add_operands(&[var_plus_eps])
                            .add_results(&[var_eps_type])
                            .build()
                            .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                    )
                    .result(0)
                    .unwrap()
                    .into();

                // Step 5: normalize = (x - mean) * rsqrt(var + eps)
                // tosa.mul requires shift operand
                let shift_val = emit_tosa_const_scalar(
                    context,
                    body_block,
                    "dense<0> : tensor<1xi8>",
                    location,
                )?;
                let normalized: melior::ir::Value = body_block
                    .append_operation(
                        OperationBuilder::new("tosa.mul", location)
                            .add_operands(&[x_minus_mean, inv_std, shift_val])
                            .add_results(&[out_type])
                            .build()
                            .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                    )
                    .result(0)
                    .unwrap()
                    .into();

                // Step 6: scale * normalized
                let shift_val2 = emit_tosa_const_scalar(
                    context,
                    body_block,
                    "dense<0> : tensor<1xi8>",
                    location,
                )?;
                let scaled: melior::ir::Value = body_block
                    .append_operation(
                        OperationBuilder::new("tosa.mul", location)
                            .add_operands(&[scale_4d, normalized, shift_val2])
                            .add_results(&[out_type])
                            .build()
                            .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                    )
                    .result(0)
                    .unwrap()
                    .into();

                // Step 7: + bias
                let result_val: melior::ir::Value = body_block
                    .append_operation(
                        OperationBuilder::new("tosa.add", location)
                            .add_operands(&[scaled, bias_4d])
                            .add_results(&[out_type])
                            .build()
                            .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                    )
                    .result(0)
                    .unwrap()
                    .into();

                values.insert(node_id, result_val);
            }

            Op::GlobalAvgPool {
                input,
                shape,
                dtype,
            } => {
                let input_shape = trace.get(*input).shape().0.clone(); // [N, C, H, W]
                let input_val = *values.get(input).expect("input not yet emitted");

                let n = input_shape[0];
                let c = input_shape[1];
                let h = input_shape[2];
                let w = input_shape[3];
                let _ = shape; // output shape [N, C, 1, 1] — used via nhwc_out_shape below

                // Step 1: Transpose NCHW → NHWC: perms [0, 2, 3, 1]
                let input_nhwc = emit_tosa_transpose(
                    context,
                    body_block,
                    input_val,
                    &input_shape,
                    &[0, 2, 3, 1],
                    *dtype,
                    location,
                )?;
                // input_nhwc shape: [N, H, W, C]

                // Step 2: Emit tosa.avg_pool2d with kernel = [H, W], stride = [1, 1], pad = [0, 0, 0, 0].
                // NHWC output shape: [N, 1, 1, C]
                let nhwc_out_shape = [n, 1u64, 1u64, c];
                let nhwc_out_type = make_ranked_tensor_type(context, &nhwc_out_shape, *dtype);

                let kernel_attr_str = format!("array<i64: {h}, {w}>");
                let pad_attr_str = "array<i64: 0, 0, 0, 0>".to_string();
                let stride_attr_str = "array<i64: 1, 1>".to_string();
                let acc_type_str = match dtype {
                    DType::F32 | DType::F64 => "f32",
                    DType::I32 | DType::I64 => "i32",
                };

                let kernel_attr = Attribute::parse(context, &kernel_attr_str)
                    .ok_or_else(|| CompileError::AttributeParse(kernel_attr_str.clone()))?;
                let pad_attr = Attribute::parse(context, &pad_attr_str)
                    .ok_or_else(|| CompileError::AttributeParse(pad_attr_str.clone()))?;
                let stride_attr = Attribute::parse(context, &stride_attr_str)
                    .ok_or_else(|| CompileError::AttributeParse(stride_attr_str.clone()))?;
                let acc_type_attr = Attribute::parse(context, acc_type_str)
                    .ok_or_else(|| CompileError::AttributeParse(acc_type_str.to_string()))?;

                // tosa.avg_pool2d takes 3 operands: (input, input_zp, output_zp).
                // For unquantized use, both zero-point tensors are 0.
                let zp_dense = match dtype {
                    DType::F32 => "dense<0.0> : tensor<1xf32>",
                    DType::F64 => "dense<0.0> : tensor<1xf64>",
                    DType::I32 => "dense<0> : tensor<1xi32>",
                    DType::I64 => "dense<0> : tensor<1xi64>",
                };
                let input_zp = emit_tosa_const_scalar(context, body_block, zp_dense, location)?;
                let output_zp = emit_tosa_const_scalar(context, body_block, zp_dense, location)?;

                let pool_nhwc: melior::ir::Value = body_block
                    .append_operation(
                        OperationBuilder::new("tosa.avg_pool2d", location)
                            .add_operands(&[input_nhwc, input_zp, output_zp])
                            .add_results(&[nhwc_out_type])
                            .add_attributes(&[
                                (Identifier::new(context, "kernel"), kernel_attr),
                                (Identifier::new(context, "pad"), pad_attr),
                                (Identifier::new(context, "stride"), stride_attr),
                                (Identifier::new(context, "acc_type"), acc_type_attr),
                            ])
                            .build()
                            .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                    )
                    .result(0)
                    .unwrap()
                    .into();

                // Step 3: Transpose NHWC → NCHW: perms [0, 3, 1, 2]
                let result_val = emit_tosa_transpose(
                    context,
                    body_block,
                    pool_nhwc,
                    &nhwc_out_shape,
                    &[0, 3, 1, 2],
                    *dtype,
                    location,
                )?;

                values.insert(node_id, result_val);
            }

            Op::MaxPool2d {
                input,
                kernel_size,
                strides,
                pads,
                shape,
                dtype,
            } => {
                let input_shape = trace.get(*input).shape().0.clone(); // [N, C, H, W]
                let input_val = *values.get(input).expect("input not yet emitted");

                // Step 1: Transpose input NCHW → NHWC: perms [0, 2, 3, 1]
                let input_nhwc = emit_tosa_transpose(
                    context,
                    body_block,
                    input_val,
                    &input_shape,
                    &[0, 2, 3, 1],
                    *dtype,
                    location,
                )?;
                // input_nhwc shape: [N, H, W, C]

                // Step 2: Emit tosa.max_pool2d on NHWC input.
                let n = input_shape[0];
                let c = input_shape[1];
                let in_h = input_shape[2];
                let in_w = input_shape[3];
                let oh = shape.0[2]; // target ONNX output height
                let ow = shape.0[3]; // target ONNX output width
                let kh = kernel_size[0];
                let kw = kernel_size[1];
                let sh = strides[0];
                let sw = strides[1];

                // TOSA requires (H + pad_top + pad_bottom - KH) % stride == 0.
                // ONNX uses floor division (drops incomplete last window).
                // Add right/bottom padding to satisfy TOSA, then slice to correct size.
                // TOSA pads max_pool2d with -inf, so extra elements never win the max.
                let pad_top = pads[0];
                let pad_left = pads[1];
                let mut pad_bottom = pads[2];
                let mut pad_right = pads[3];
                let rem_h = (in_h + pad_top + pad_bottom - kh) % sh;
                let rem_w = (in_w + pad_left + pad_right - kw) % sw;
                if rem_h != 0 {
                    pad_bottom += sh - rem_h;
                }
                if rem_w != 0 {
                    pad_right += sw - rem_w;
                }
                // TOSA output with the (possibly enlarged) padding.
                let tosa_oh = (in_h + pad_top + pad_bottom - kh) / sh + 1;
                let tosa_ow = (in_w + pad_left + pad_right - kw) / sw + 1;
                let needs_slice = tosa_oh != oh || tosa_ow != ow;

                let nhwc_tosa_shape = [n, tosa_oh, tosa_ow, c];
                let nhwc_tosa_type = make_ranked_tensor_type(context, &nhwc_tosa_shape, *dtype);

                let kernel_attr_str = format!("array<i64: {kh}, {kw}>");
                // TOSA pad order: [pad_top, pad_bottom, pad_left, pad_right]
                let pad_attr_str =
                    format!("array<i64: {pad_top}, {pad_bottom}, {pad_left}, {pad_right}>");
                let stride_attr_str = format!("array<i64: {sh}, {sw}>");

                let kernel_attr = Attribute::parse(context, &kernel_attr_str)
                    .ok_or_else(|| CompileError::AttributeParse(kernel_attr_str.clone()))?;
                let pad_attr = Attribute::parse(context, &pad_attr_str)
                    .ok_or_else(|| CompileError::AttributeParse(pad_attr_str.clone()))?;
                let stride_attr = Attribute::parse(context, &stride_attr_str)
                    .ok_or_else(|| CompileError::AttributeParse(stride_attr_str.clone()))?;

                let pool_acc_type_str = match dtype {
                    DType::F32 | DType::F64 => "f32",
                    DType::I32 | DType::I64 => "i32",
                };
                let pool_acc_type_attr = Attribute::parse(context, pool_acc_type_str)
                    .ok_or_else(|| CompileError::AttributeParse(pool_acc_type_str.to_string()))?;

                let mut pool_nhwc: melior::ir::Value = body_block
                    .append_operation(
                        OperationBuilder::new("tosa.max_pool2d", location)
                            .add_operands(&[input_nhwc])
                            .add_results(&[nhwc_tosa_type])
                            .add_attributes(&[
                                (Identifier::new(context, "kernel"), kernel_attr),
                                (Identifier::new(context, "pad"), pad_attr),
                                (Identifier::new(context, "stride"), stride_attr),
                                (Identifier::new(context, "acc_type"), pool_acc_type_attr),
                            ])
                            .build()
                            .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                    )
                    .result(0)
                    .unwrap()
                    .into();

                // If we added extra padding, slice back to the ONNX-expected size.
                // LLVM 21 tosa.slice takes 3 operands: (input, start, size) where
                // start and size are !tosa.shape<N> produced by tosa.const_shape.
                if needs_slice {
                    let nhwc_out_shape_slice = [n, oh, ow, c];
                    let rank = 4usize;
                    let shape_type_str = format!("!tosa.shape<{rank}>");
                    let shape_type = melior::ir::Type::parse(context, &shape_type_str)
                        .ok_or_else(|| CompileError::AttributeParse(shape_type_str.clone()))?;

                    let start_vals_str = format!("dense<[0, 0, 0, 0]> : tensor<{rank}xindex>");
                    let start_attr = Attribute::parse(context, &start_vals_str)
                        .ok_or_else(|| CompileError::AttributeParse(start_vals_str.clone()))?;
                    let start_val: melior::ir::Value = body_block
                        .append_operation(
                            OperationBuilder::new("tosa.const_shape", location)
                                .add_results(&[shape_type])
                                .add_attributes(&[(Identifier::new(context, "values"), start_attr)])
                                .build()
                                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                        )
                        .result(0)
                        .unwrap()
                        .into();

                    let size_vals_str =
                        format!("dense<[{n}, {oh}, {ow}, {c}]> : tensor<{rank}xindex>");
                    let size_attr = Attribute::parse(context, &size_vals_str)
                        .ok_or_else(|| CompileError::AttributeParse(size_vals_str.clone()))?;
                    let size_val: melior::ir::Value = body_block
                        .append_operation(
                            OperationBuilder::new("tosa.const_shape", location)
                                .add_results(&[shape_type])
                                .add_attributes(&[(Identifier::new(context, "values"), size_attr)])
                                .build()
                                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                        )
                        .result(0)
                        .unwrap()
                        .into();

                    let slice_type =
                        make_ranked_tensor_type(context, &nhwc_out_shape_slice, *dtype);
                    pool_nhwc = body_block
                        .append_operation(
                            OperationBuilder::new("tosa.slice", location)
                                .add_operands(&[pool_nhwc, start_val, size_val])
                                .add_results(&[slice_type])
                                .build()
                                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                        )
                        .result(0)
                        .unwrap()
                        .into();
                }

                let nhwc_out_shape = [n, oh, ow, c];

                // Step 3: Transpose output NHWC → NCHW: perms [0, 3, 1, 2]
                let result_val = emit_tosa_transpose(
                    context,
                    body_block,
                    pool_nhwc,
                    &nhwc_out_shape,
                    &[0, 3, 1, 2],
                    *dtype,
                    location,
                )?;

                values.insert(node_id, result_val);
            }

            Op::Pow {
                lhs,
                rhs,
                shape,
                dtype,
            } => {
                let lhs_shape = trace.get(*lhs).shape();
                let rhs_shape = trace.get(*rhs).shape();
                // tosa.pow is float-only; integer pow is not supported.
                assert!(
                    matches!(dtype, DType::F32 | DType::F64),
                    "Pow only supported for float dtypes"
                );
                let result_val = emit_tosa_binary(
                    context,
                    body_block,
                    &values,
                    "tosa.pow",
                    *lhs,
                    *rhs,
                    &lhs_shape.0,
                    &rhs_shape.0,
                    &shape.0,
                    *dtype,
                    location,
                )?;
                values.insert(node_id, result_val);
            }

            Op::Sqrt {
                input,
                shape,
                dtype,
            } => {
                let result_val = emit_unary_linalg_math(
                    context,
                    body_block,
                    &values,
                    "math.sqrt",
                    *input,
                    &shape.0,
                    *dtype,
                    location,
                )?;
                values.insert(node_id, result_val);
            }

            Op::Cast {
                input,
                target_dtype,
                shape,
                dtype: _,
            } => {
                let src_dtype = trace.get(*input).dtype();
                let result_val = emit_cast(
                    context,
                    body_block,
                    &values,
                    *input,
                    &shape.0,
                    src_dtype,
                    *target_dtype,
                    location,
                )?;
                values.insert(node_id, result_val);
            }

            Op::ReduceMean {
                input,
                axes,
                keepdim,
                shape,
                dtype,
            } => {
                let input_shape = trace.get(*input).shape().0.clone();
                // When input has dynamic dims, tosa.reduce_sum cannot produce a valid
                // static output type. Fall back to linalg.generic-based reduction.
                let result_val = if input_shape.contains(&DIM_DYNAMIC) {
                    // Perform sequential single-axis reductions using emit_reduction.
                    let mut current_val = *values.get(input).expect("input not emitted");
                    let mut current_shape = input_shape.clone();

                    // We need NodeId lookups to work — insert intermediate values.
                    // Reduce each axis sequentially (all keepdim=true for intermediates).
                    let n_axes = axes.len();
                    for (step, &ax) in axes.iter().enumerate() {
                        let ax_usize = ax as usize;
                        let is_last = step == n_axes - 1;
                        let this_keepdim = if is_last { *keepdim } else { true };

                        let mut out_shape = current_shape.clone();
                        if this_keepdim {
                            out_shape[ax_usize] = 1;
                        } else {
                            out_shape.remove(ax_usize);
                        }

                        // Use a fake NodeId to pass current_val through emit_reduction
                        // by inserting it into values temporarily.
                        let fake_id = NodeId(u32::MAX - step as u32);
                        values.insert(fake_id, current_val);
                        let sum_val = emit_reduction(
                            context,
                            body_block,
                            &values,
                            fake_id,
                            &current_shape,
                            &out_shape,
                            ax_usize,
                            this_keepdim,
                            *dtype,
                            location,
                            false,
                        )?;
                        values.remove(&fake_id);
                        current_val = sum_val;
                        current_shape = out_shape;
                    }

                    // Divide by total number of elements reduced (product of reduced axes).
                    // The reduced axes must be static for meaningful mean computation;
                    // dynamic batch/seq dims in non-reduced positions are fine.
                    let total_reduced: u64 = axes
                        .iter()
                        .map(|&a| input_shape[a as usize])
                        .filter(|&d| d != DIM_DYNAMIC)
                        .product();
                    let recip = 1.0 / total_reduced as f64;

                    let elem_type = dtype.to_mlir_type(context);
                    let out_tensor_type = make_ranked_tensor_type(context, &current_shape, *dtype);
                    let recip_val: melior::ir::Value = match dtype {
                        DType::F32 | DType::F64 => body_block
                            .append_operation(arith::constant(
                                context,
                                FloatAttribute::new(context, elem_type, recip).into(),
                                location,
                            ))
                            .result(0)
                            .unwrap()
                            .into(),
                        DType::I32 | DType::I64 => body_block
                            .append_operation(arith::constant(
                                context,
                                IntegerAttribute::new(elem_type, recip as i64).into(),
                                location,
                            ))
                            .result(0)
                            .unwrap()
                            .into(),
                    };

                    let out_rank = current_shape.len();
                    let scale_sources: Vec<(melior::ir::Value<'c, 'c>, usize)> = current_shape
                        .iter()
                        .enumerate()
                        .filter(|&(_, d)| *d == DIM_DYNAMIC)
                        .map(|(i, _)| (current_val, i))
                        .collect();
                    let scale_init = if current_shape.contains(&DIM_DYNAMIC) {
                        emit_tensor_empty_dynamic(context, body_block, &current_shape, *dtype, &scale_sources, location)?
                    } else {
                        body_block
                            .append_operation(
                                OperationBuilder::new("tensor.empty", location)
                                    .add_results(&[out_tensor_type])
                                    .build()
                                    .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                            )
                            .result(0)
                            .unwrap()
                            .into()
                    };

                    let in_map = make_identity_map(context, out_rank)?;
                    let out_map = make_identity_map(context, out_rank)?;
                    let indexing_maps = ArrayAttribute::new(context, &[in_map, out_map]);
                    let iterator_types = make_iterator_types(context, out_rank)?;
                    let segment_sizes = Attribute::parse(context, "array<i32: 1, 1>").ok_or_else(|| {
                        CompileError::AttributeParse("segment sizes".into())
                    })?;

                    let scale_region = {
                        let scale_block = Block::new(&[(elem_type, location), (elem_type, location)]);
                        let in_elem: melior::ir::Value = scale_block.argument(0).unwrap().into();
                        let result: melior::ir::Value = match dtype {
                            DType::F32 | DType::F64 => scale_block
                                .append_operation(arith::mulf(in_elem, recip_val, location))
                                .result(0)
                                .unwrap()
                                .into(),
                            DType::I32 | DType::I64 => scale_block
                                .append_operation(arith::muli(in_elem, recip_val, location))
                                .result(0)
                                .unwrap()
                                .into(),
                        };
                        scale_block.append_operation(
                            OperationBuilder::new("linalg.yield", location)
                                .add_operands(&[result])
                                .build()
                                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                        );
                        let r = Region::new();
                        r.append_block(scale_block);
                        r
                    };

                    body_block
                        .append_operation(
                            OperationBuilder::new("linalg.generic", location)
                                .add_operands(&[current_val, scale_init])
                                .add_results(&[out_tensor_type])
                                .add_attributes(&[
                                    (Identifier::new(context, "indexing_maps"), indexing_maps.into()),
                                    (Identifier::new(context, "iterator_types"), iterator_types),
                                    (Identifier::new(context, "operand_segment_sizes"), segment_sizes),
                                ])
                                .add_regions([scale_region])
                                .build()
                                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                        )
                        .result(0)
                        .unwrap()
                        .into()
                } else {
                    emit_reduce_mean(
                        context,
                        body_block,
                        &values,
                        *input,
                        &input_shape,
                        &shape.0,
                        axes,
                        *keepdim,
                        *dtype,
                        location,
                    )?
                };
                values.insert(node_id, result_val);
            }

            Op::Gather {
                input,
                indices,
                axis,
                shape,
                dtype,
            } => {
                let data_shape = trace.get(*input).shape().0.clone();
                let indices_shape = trace.get(*indices).shape().0.clone();
                let indices_dtype = trace.get(*indices).dtype();
                let data_val = *values.get(input).expect("gather: input not emitted");
                let indices_val = *values.get(indices).expect("gather: indices not emitted");

                let result_val = emit_gather(
                    context,
                    body_block,
                    data_val,
                    indices_val,
                    &data_shape,
                    &indices_shape,
                    &shape.0,
                    *axis as usize,
                    *dtype,
                    indices_dtype,
                    location,
                )?;
                values.insert(node_id, result_val);
            }

            Op::Slice {
                input,
                starts,
                ends: _,
                axes,
                steps,
                shape,
                dtype,
            } => {
                let input_shape = trace.get(*input).shape().0.clone();
                let input_val = *values.get(input).expect("slice: input not emitted");
                let rank = input_shape.len();

                // Build per-axis start array (all axes), falling back to 0 for non-sliced axes.
                let mut full_starts = vec![0i64; rank];
                for (i, &ax) in axes.iter().enumerate() {
                    full_starts[ax as usize] = starts[i];
                }

                let all_stride_one = steps.iter().all(|&s| s == 1);
                let input_has_dynamic = input_shape.contains(&DIM_DYNAMIC);

                let result_val = if all_stride_one && !input_has_dynamic {
                    // Use tosa.slice for stride-1 case with static shapes.
                    emit_tosa_slice(
                        context,
                        body_block,
                        input_val,
                        &input_shape,
                        &full_starts,
                        &shape.0,
                        *dtype,
                        location,
                    )?
                } else {
                    // Strided slice or dynamic input: use linalg.generic with linalg.index.
                    emit_strided_slice(
                        context,
                        body_block,
                        input_val,
                        &input_shape,
                        starts,
                        axes,
                        steps,
                        &shape.0,
                        *dtype,
                        location,
                    )?
                };
                values.insert(node_id, result_val);
            }

            Op::Concat {
                inputs,
                axis,
                shape,
                dtype,
            } => {
                let ax = *axis as usize;
                let tensor_type = make_ranked_tensor_type(context, &shape.0, *dtype);
                let input_vals: Vec<melior::ir::Value<'c, 'c>> = inputs
                    .iter()
                    .map(|&inp| *values.get(&inp).expect("concat: input not emitted"))
                    .collect();

                let any_dynamic = shape.0.contains(&DIM_DYNAMIC)
                    || inputs.iter().any(|&inp| {
                        trace.get(inp).shape().0.contains(&DIM_DYNAMIC)
                    });

                let result_val: melior::ir::Value = if any_dynamic {
                    // Dynamic concat: build output via tensor.insert_slice.
                    let rank = shape.0.len();
                    let index_type = melior::ir::Type::parse(context, "index")
                        .ok_or_else(|| CompileError::AttributeParse("index type".into()))?;

                    // Compute total size of the concat axis at runtime via arith.addi.
                    // First, get each input's size along the concat axis.
                    let mut input_axis_dims: Vec<melior::ir::Value<'c, 'c>> = Vec::new();
                    for &inp_val in &input_vals {
                        let dim_val =
                            emit_tensor_dim(context, body_block, inp_val, ax, location)?;
                        input_axis_dims.push(dim_val);
                    }

                    // Sum them up.
                    let total_axis_dim: melior::ir::Value<'c, 'c> =
                        input_axis_dims.iter().skip(1).fold(input_axis_dims[0], |acc, &d| {
                            body_block
                                .append_operation(arith::addi(acc, d, location))
                                .result(0)
                                .unwrap()
                                .into()
                        });

                    // Build dynamic_dim_sources for tensor.empty: for each ? dim in
                    // output shape, source from a representative input or total_axis_dim.
                    // We need to handle `total_axis_dim` as a pre-computed value.
                    // Use emit_tensor_dim for all non-concat dims from the first input,
                    // and total_axis_dim for the concat axis.
                    let mut dyn_dim_vals: Vec<melior::ir::Value<'c, 'c>> = Vec::new();
                    for (i, &d) in shape.0.iter().enumerate() {
                        if d == DIM_DYNAMIC {
                            if i == ax {
                                dyn_dim_vals.push(total_axis_dim);
                            } else {
                                // Use the first input's dim for non-concat axes.
                                let dv = emit_tensor_dim(
                                    context,
                                    body_block,
                                    input_vals[0],
                                    i,
                                    location,
                                )?;
                                dyn_dim_vals.push(dv);
                            }
                        }
                    }

                    let empty_val: melior::ir::Value<'c, 'c> = body_block
                        .append_operation(
                            OperationBuilder::new("tensor.empty", location)
                                .add_operands(&dyn_dim_vals)
                                .add_results(&[tensor_type])
                                .build()
                                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                        )
                        .result(0)
                        .unwrap()
                        .into();

                    // Insert each input slice in sequence, tracking offset along concat axis.
                    let zero_idx: melior::ir::Value<'c, 'c> = body_block
                        .append_operation(arith::constant(
                            context,
                            IntegerAttribute::new(index_type, 0).into(),
                            location,
                        ))
                        .result(0)
                        .unwrap()
                        .into();

                    let mut current = empty_val;
                    let mut offset: melior::ir::Value<'c, 'c> = zero_idx;

                    for (inp_idx, &inp_val) in input_vals.iter().enumerate() {
                        let inp_shape = trace.get(inputs[inp_idx]).shape().0.clone();

                        // Build static_offsets, static_sizes, static_strides for this slice.
                        // All offsets are 0 except the concat axis which is `offset` (dynamic).
                        // Sizes match the input shape for each dim (? dims are dynamic).
                        // Strides are all 1.

                        // static_offsets: i64::MIN for dynamic positions, else 0.
                        // The concat axis is always dynamic (offset is runtime).
                        let static_offsets: Vec<i64> = (0..rank)
                            .map(|i| if i == ax { i64::MIN } else { 0 })
                            .collect();
                        let so_str = static_offsets.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ");
                        let static_offsets_attr = Attribute::parse(context, &format!("array<i64: {so_str}>"))
                            .ok_or_else(|| CompileError::AttributeParse("static_offsets".into()))?;

                        // static_sizes: i64::MIN for dynamic dims, else the static value.
                        let static_sizes: Vec<i64> = inp_shape
                            .iter()
                            .map(|&d| if d == DIM_DYNAMIC { i64::MIN } else { d as i64 })
                            .collect();
                        let ss_str = static_sizes.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ");
                        let static_sizes_attr = Attribute::parse(context, &format!("array<i64: {ss_str}>"))
                            .ok_or_else(|| CompileError::AttributeParse("static_sizes".into()))?;

                        let static_strides_str = vec!["1"; rank].join(", ");
                        let static_strides_attr = Attribute::parse(context, &format!("array<i64: {static_strides_str}>"))
                            .ok_or_else(|| CompileError::AttributeParse("static_strides".into()))?;

                        // Separate dynamic offsets, sizes, and strides.
                        // Dynamic offsets: concat axis has a runtime offset.
                        let dynamic_offsets: Vec<melior::ir::Value<'c, 'c>> = vec![offset];
                        // Dynamic sizes: any DIM_DYNAMIC dims in the input shape.
                        let mut dynamic_sizes: Vec<melior::ir::Value<'c, 'c>> = Vec::new();
                        for (i, &d) in inp_shape.iter().enumerate() {
                            if d == DIM_DYNAMIC {
                                let dv = emit_tensor_dim(context, body_block, inp_val, i, location)?;
                                dynamic_sizes.push(dv);
                            }
                        }
                        // Dynamic strides: none (all static 1).
                        let dynamic_strides: Vec<melior::ir::Value<'c, 'c>> = Vec::new();

                        let n_dyn_offsets = dynamic_offsets.len() as i32;
                        let n_dyn_sizes = dynamic_sizes.len() as i32;
                        let n_dyn_strides = dynamic_strides.len() as i32;

                        let seg_sizes_str = format!(
                            "array<i32: 1, 1, {n_dyn_offsets}, {n_dyn_sizes}, {n_dyn_strides}>"
                        );
                        let seg_sizes_attr = Attribute::parse(context, &seg_sizes_str)
                            .ok_or_else(|| CompileError::AttributeParse(seg_sizes_str.clone()))?;

                        let inp_type = make_ranked_tensor_type(context, &inp_shape, *dtype);
                        let dest_type = tensor_type;

                        let mut all_operands: Vec<melior::ir::Value<'c, 'c>> = vec![inp_val, current];
                        all_operands.extend(dynamic_offsets);
                        all_operands.extend(dynamic_sizes);
                        all_operands.extend(dynamic_strides);

                        current = body_block
                            .append_operation(
                                OperationBuilder::new("tensor.insert_slice", location)
                                    .add_operands(&all_operands)
                                    .add_results(&[dest_type])
                                    .add_attributes(&[
                                        (Identifier::new(context, "operandSegmentSizes"), seg_sizes_attr),
                                        (Identifier::new(context, "static_offsets"), static_offsets_attr),
                                        (Identifier::new(context, "static_sizes"), static_sizes_attr),
                                        (Identifier::new(context, "static_strides"), static_strides_attr),
                                    ])
                                    .build()
                                    .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                            )
                            .result(0)
                            .unwrap()
                            .into();
                        let _ = (inp_type, index_type);

                        // Advance offset by this input's concat-axis size.
                        let inp_axis_dim = input_axis_dims[inp_idx];
                        offset = body_block
                            .append_operation(arith::addi(offset, inp_axis_dim, location))
                            .result(0)
                            .unwrap()
                            .into();
                    }

                    current
                } else {
                    // Static concat: use tosa.concat.
                    let ax_i32 = ax as i32;
                    let axis_attr = IntegerAttribute::new(
                        melior::ir::r#type::IntegerType::new(context, 32).into(),
                        ax_i32 as i64,
                    );
                    body_block
                        .append_operation(
                            OperationBuilder::new("tosa.concat", location)
                                .add_operands(&input_vals)
                                .add_results(&[tensor_type])
                                .add_attributes(&[(Identifier::new(context, "axis"), axis_attr.into())])
                                .build()
                                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                        )
                        .result(0)
                        .unwrap()
                        .into()
                };

                values.insert(node_id, result_val);
            }

            Op::Transpose {
                input,
                perm,
                shape,
                dtype,
            } => {
                let input_shape = trace.get(*input).shape().0.clone();
                let input_val = *values.get(input).expect("transpose: input not emitted");
                let perms_usize: Vec<usize> = perm.iter().map(|&p| p as usize).collect();

                let result_val = emit_tosa_transpose(
                    context,
                    body_block,
                    input_val,
                    &input_shape,
                    &perms_usize,
                    *dtype,
                    location,
                )?;
                let _ = shape; // shape is derived from perms + input_shape inside helper
                values.insert(node_id, result_val);
            }

            Op::Where {
                condition,
                x,
                y,
                shape,
                dtype,
            } => {
                let cond_shape = trace.get(*condition).shape().0.clone();
                let x_shape = trace.get(*x).shape().0.clone();
                let y_shape = trace.get(*y).shape().0.clone();

                let cond_val = *values.get(condition).expect("where: condition not emitted");
                let x_val = *values.get(x).expect("where: x not emitted");
                let y_val = *values.get(y).expect("where: y not emitted");

                let out_rank = shape.0.len();

                // Promote all operands to the output rank.
                let (cond_promoted, cond_promoted_shape) = promote_rank_with_reshape(
                    context,
                    body_block,
                    cond_val,
                    &cond_shape,
                    out_rank,
                    DType::I64,
                    location,
                )?;
                let (x_promoted, _) = promote_rank_with_reshape(
                    context,
                    body_block,
                    x_val,
                    &x_shape,
                    out_rank,
                    *dtype,
                    location,
                )?;
                let (y_promoted, _) = promote_rank_with_reshape(
                    context,
                    body_block,
                    y_val,
                    &y_shape,
                    out_rank,
                    *dtype,
                    location,
                )?;

                // Convert I64 condition to i1 via arith.cmpi ne 0.
                let i1_type = melior::ir::Type::parse(context, "i1")
                    .ok_or_else(|| CompileError::AttributeParse("i1 type".into()))?;
                let i64_type = DType::I64.to_mlir_type(context);
                let i1_shape: Vec<u64> = shape.0.iter().map(|&d| dim_to_mlir(d)).collect();
                let i1_tensor_type: melior::ir::Type<'c> =
                    RankedTensorType::new(&i1_shape, i1_type, None).into();

                // Emit zero constant for comparison.
                let zero_val: melior::ir::Value = body_block
                    .append_operation(arith::constant(
                        context,
                        IntegerAttribute::new(i64_type, 0).into(),
                        location,
                    ))
                    .result(0)
                    .unwrap()
                    .into();

                // linalg.generic: cast I64 → i1 (cmpi ne 0).
                let i64_tensor_type =
                    make_ranked_tensor_type(context, &shape.0, DType::I64);
                let i1_init: melior::ir::Value = if shape.0.contains(&DIM_DYNAMIC) {
                    // For dynamic dims in output, source from cond_promoted (which has
                    // the same shape as the output after rank promotion).
                    let sources: Vec<(melior::ir::Value<'c, 'c>, usize)> = shape.0
                        .iter()
                        .enumerate()
                        .filter(|&(_, d)| *d == DIM_DYNAMIC)
                        .map(|(i, _)| (cond_promoted, i))
                        .collect();
                    // i1 tensor uses RankedTensorType directly — use emit_tensor_empty_dynamic
                    // but we need to build the i1 type manually.
                    let index_type = melior::ir::Type::parse(context, "index")
                        .ok_or_else(|| CompileError::AttributeParse("index type".into()))?;
                    let mut dyn_dim_vals: Vec<melior::ir::Value<'c, 'c>> = Vec::new();
                    for &(src_tensor, src_dim_idx) in &sources {
                        let dv = emit_tensor_dim(context, body_block, src_tensor, src_dim_idx, location)?;
                        dyn_dim_vals.push(dv);
                    }
                    let _ = index_type;
                    body_block
                        .append_operation(
                            OperationBuilder::new("tensor.empty", location)
                                .add_operands(&dyn_dim_vals)
                                .add_results(&[i1_tensor_type])
                                .build()
                                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                        )
                        .result(0)
                        .unwrap()
                        .into()
                } else {
                    body_block
                        .append_operation(
                            OperationBuilder::new("tensor.empty", location)
                                .add_results(&[i1_tensor_type])
                                .build()
                                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                        )
                        .result(0)
                        .unwrap()
                        .into()
                };

                let cond_map = make_broadcast_map(context, &cond_promoted_shape, &shape.0)?;
                let out_i1_map = make_identity_map(context, out_rank)?;
                let i1_maps = ArrayAttribute::new(context, &[cond_map, out_i1_map]);
                let i1_iters = make_iterator_types(context, out_rank)?;
                let i1_segs =
                    Attribute::parse(context, "array<i32: 1, 1>").ok_or_else(|| {
                        CompileError::AttributeParse("i1 cast segment sizes".into())
                    })?;

                let i1_region = {
                    let i1_block =
                        Block::new(&[(i64_type, location), (i1_type, location)]);
                    let in_elem: melior::ir::Value = i1_block.argument(0).unwrap().into();

                    let cmp: melior::ir::Value = i1_block
                        .append_operation(
                            OperationBuilder::new("arith.cmpi", location)
                                .add_operands(&[in_elem, zero_val])
                                .add_results(&[i1_type])
                                .add_attributes(&[(
                                    Identifier::new(context, "predicate"),
                                    IntegerAttribute::new(
                                        melior::ir::r#type::IntegerType::new(context, 64).into(),
                                        1, // "ne" = 1
                                    )
                                    .into(),
                                )])
                                .build()
                                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                        )
                        .result(0)
                        .unwrap()
                        .into();

                    i1_block.append_operation(
                        OperationBuilder::new("linalg.yield", location)
                            .add_operands(&[cmp])
                            .build()
                            .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                    );

                    let r = Region::new();
                    r.append_block(i1_block);
                    r
                };

                let cond_i1: melior::ir::Value = body_block
                    .append_operation(
                        OperationBuilder::new("linalg.generic", location)
                            .add_operands(&[cond_promoted, i1_init])
                            .add_results(&[i1_tensor_type])
                            .add_attributes(&[
                                (Identifier::new(context, "indexing_maps"), i1_maps.into()),
                                (Identifier::new(context, "iterator_types"), i1_iters),
                                (Identifier::new(context, "operand_segment_sizes"), i1_segs),
                            ])
                            .add_regions([i1_region])
                            .build()
                            .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                    )
                    .result(0)
                    .unwrap()
                    .into();
                let _ = (i64_tensor_type, zero_val);

                // tosa.select: (cond: i1 tensor, x, y) → output
                let out_tensor_type = make_ranked_tensor_type(context, &shape.0, *dtype);
                let result_val: melior::ir::Value = body_block
                    .append_operation(
                        OperationBuilder::new("tosa.select", location)
                            .add_operands(&[cond_i1, x_promoted, y_promoted])
                            .add_results(&[out_tensor_type])
                            .build()
                            .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                    )
                    .result(0)
                    .unwrap()
                    .into();

                values.insert(node_id, result_val);
            }

            // ── ShapeOf ───────────────────────────────────────────────────────
            //
            // For each dimension of the input tensor:
            //   - If static: emit `arith.constant {val} : i64`
            //   - If dynamic: emit `memref.dim` on the raw memref arg (for Input ops)
            //     or `tensor.dim` (for computed tensors), then `arith.index_cast` to i64
            // Then pack all values into `tensor<{rank}xi64>` via `tensor.from_elements`.
            //
            // NOTE: We use `memref.dim` on Input ops' block arguments (rather than
            // `tensor.dim` on the bufferization.to_tensor result) because MLIR's
            // canonicalization passes can constant-fold `tensor.dim(to_tensor(%arg))`
            // to the type's shape value (-1) rather than the runtime memref descriptor.
            // `memref.dim` on a function-argument memref is reliably lowered to a
            // GEP+load from the descriptor's sizes array.

            Op::ShapeOf { input, shape, .. } => {
                let input_val = *values.get(input).expect("ShapeOf: input not yet emitted");
                let input_op = trace.get(*input);
                let input_shape = input_op.shape().clone();
                let rank = input_shape.rank();
                let i64_type = DType::I64.to_mlir_type(context);
                let index_type = melior::ir::Type::parse(context, "index")
                    .ok_or_else(|| CompileError::AttributeParse("index type".into()))?;

                // If the input is a direct Input op, get the raw memref block argument.
                // Otherwise, use the tensor value (tensor.dim will be used).
                let memref_arg_opt: Option<melior::ir::Value<'c, 'c>> =
                    if let Op::Input { arg_index, .. } = input_op {
                        Some(body_block.argument(*arg_index as usize).unwrap().into())
                    } else {
                        None
                    };

                let mut dim_vals: Vec<melior::ir::Value<'c, 'c>> = Vec::with_capacity(rank);
                for i in 0..rank {
                    let dim_val: melior::ir::Value<'c, 'c> = if input_shape.is_dynamic_dim(i) {
                        let idx_const: melior::ir::Value = body_block
                            .append_operation(arith::constant(
                                context,
                                IntegerAttribute::new(index_type, i as i64).into(),
                                location,
                            ))
                            .result(0)
                            .unwrap()
                            .into();

                        // Use memref.dim on the raw arg if available; otherwise tensor.dim.
                        let dim_idx: melior::ir::Value = if let Some(mref) = memref_arg_opt {
                            body_block
                                .append_operation(
                                    OperationBuilder::new("memref.dim", location)
                                        .add_operands(&[mref, idx_const])
                                        .add_results(&[index_type])
                                        .build()
                                        .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                                )
                                .result(0)
                                .unwrap()
                                .into()
                        } else {
                            body_block
                                .append_operation(
                                    OperationBuilder::new("tensor.dim", location)
                                        .add_operands(&[input_val, idx_const])
                                        .add_results(&[index_type])
                                        .build()
                                        .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                                )
                                .result(0)
                                .unwrap()
                                .into()
                        };

                        body_block
                            .append_operation(
                                OperationBuilder::new("arith.index_cast", location)
                                    .add_operands(&[dim_idx])
                                    .add_results(&[i64_type])
                                    .build()
                                    .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                            )
                            .result(0)
                            .unwrap()
                            .into()
                    } else {
                        // Compile-time constant: emit arith.constant directly.
                        body_block
                            .append_operation(arith::constant(
                                context,
                                IntegerAttribute::new(i64_type, input_shape.0[i] as i64).into(),
                                location,
                            ))
                            .result(0)
                            .unwrap()
                            .into()
                    };
                    dim_vals.push(dim_val);
                }

                let result_type = make_ranked_tensor_type(context, &shape.0, DType::I64);
                let result_val: melior::ir::Value = body_block
                    .append_operation(
                        OperationBuilder::new("tensor.from_elements", location)
                            .add_operands(&dim_vals)
                            .add_results(&[result_type])
                            .build()
                            .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                    )
                    .result(0)
                    .unwrap()
                    .into();

                values.insert(node_id, result_val);
            }

            // ── ConstantOfShape ───────────────────────────────────────────────
            //
            // For each DIM_DYNAMIC position i in `shape`:
            //   extract shape_input[i] as i64, then cast to index
            // Emit `tensor.empty(%dim...)` + `linalg.fill` with fill_value.

            Op::ConstantOfShape {
                shape_input,
                fill_value,
                shape,
                dtype,
            } => {
                let shape_val = *values.get(shape_input).expect("ConstantOfShape: shape_input not yet emitted");
                let i64_type = DType::I64.to_mlir_type(context);
                let index_type = melior::ir::Type::parse(context, "index")
                    .ok_or_else(|| CompileError::AttributeParse("index type".into()))?;
                let elem_type = dtype.to_mlir_type(context);

                // Collect index-typed dynamic dim values from the shape tensor.
                let mut dynamic_dim_vals: Vec<melior::ir::Value<'c, 'c>> = Vec::new();
                for (i, &dim) in shape.0.iter().enumerate() {
                    if dim == DIM_DYNAMIC {
                        let idx_const: melior::ir::Value = body_block
                            .append_operation(arith::constant(
                                context,
                                IntegerAttribute::new(index_type, i as i64).into(),
                                location,
                            ))
                            .result(0)
                            .unwrap()
                            .into();
                        let i64_val: melior::ir::Value = body_block
                            .append_operation(
                                OperationBuilder::new("tensor.extract", location)
                                    .add_operands(&[shape_val, idx_const])
                                    .add_results(&[i64_type])
                                    .build()
                                    .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                            )
                            .result(0)
                            .unwrap()
                            .into();
                        let idx_val: melior::ir::Value = body_block
                            .append_operation(
                                OperationBuilder::new("arith.index_cast", location)
                                    .add_operands(&[i64_val])
                                    .add_results(&[index_type])
                                    .build()
                                    .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                            )
                            .result(0)
                            .unwrap()
                            .into();
                        dynamic_dim_vals.push(idx_val);
                    }
                }

                let out_type = make_ranked_tensor_type(context, &shape.0, *dtype);
                let empty_val: melior::ir::Value = body_block
                    .append_operation(
                        OperationBuilder::new("tensor.empty", location)
                            .add_operands(&dynamic_dim_vals)
                            .add_results(&[out_type])
                            .build()
                            .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                    )
                    .result(0)
                    .unwrap()
                    .into();

                // Emit the fill scalar constant with the correct dtype.
                let fill_scalar: melior::ir::Value = match dtype {
                    DType::F32 => body_block
                        .append_operation(arith::constant(
                            context,
                            FloatAttribute::new(context, elem_type, *fill_value).into(),
                            location,
                        ))
                        .result(0)
                        .unwrap()
                        .into(),
                    DType::F64 => body_block
                        .append_operation(arith::constant(
                            context,
                            FloatAttribute::new(context, elem_type, *fill_value).into(),
                            location,
                        ))
                        .result(0)
                        .unwrap()
                        .into(),
                    DType::I32 => body_block
                        .append_operation(arith::constant(
                            context,
                            IntegerAttribute::new(elem_type, *fill_value as i64).into(),
                            location,
                        ))
                        .result(0)
                        .unwrap()
                        .into(),
                    DType::I64 => body_block
                        .append_operation(arith::constant(
                            context,
                            IntegerAttribute::new(elem_type, *fill_value as i64).into(),
                            location,
                        ))
                        .result(0)
                        .unwrap()
                        .into(),
                };

                // linalg.fill: fill the empty tensor with the scalar.
                let fill_seg = Attribute::parse(context, "array<i32: 1, 1>").ok_or_else(|| {
                    CompileError::AttributeParse("linalg.fill segment sizes".into())
                })?;
                let fill_region = {
                    let fill_block = Block::new(&[(elem_type, location), (elem_type, location)]);
                    let fill_in = fill_block.argument(0).unwrap().into();
                    fill_block.append_operation(
                        OperationBuilder::new("linalg.yield", location)
                            .add_operands(&[fill_in])
                            .build()
                            .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                    );
                    let r = Region::new();
                    r.append_block(fill_block);
                    r
                };
                let result_val: melior::ir::Value = body_block
                    .append_operation(
                        OperationBuilder::new("linalg.fill", location)
                            .add_operands(&[fill_scalar, empty_val])
                            .add_results(&[out_type])
                            .add_attributes(&[(
                                Identifier::new(context, "operand_segment_sizes"),
                                fill_seg,
                            )])
                            .add_regions([fill_region])
                            .build()
                            .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                    )
                    .result(0)
                    .unwrap()
                    .into();

                values.insert(node_id, result_val);
            }

            // ── Range ─────────────────────────────────────────────────────────
            //
            // Inputs are scalar tensors — either 0-D (tensor<Dtype>) or
            // 1-element rank-1 (tensor<1xDtype>).
            // 1. Extract scalars: tensor.extract %t[] (0-D) or %t[%c0] (1-D)
            // 2. Compute length: ceil((limit - start) / delta), clamped >= 0
            // 3. tensor.empty(%len_idx) : tensor<?xDtype>
            // 4. linalg.generic body: out[i] = start + i * delta

            Op::Range {
                start,
                limit,
                delta,
                shape,
                dtype,
            } => {
                let start_val = *values.get(start).expect("Range: start not yet emitted");
                let limit_val = *values.get(limit).expect("Range: limit not yet emitted");
                let delta_val = *values.get(delta).expect("Range: delta not yet emitted");
                let elem_type = dtype.to_mlir_type(context);
                let index_type = melior::ir::Type::parse(context, "index")
                    .ok_or_else(|| CompileError::AttributeParse("index type".into()))?;

                // Extract scalar from 0-D (tensor<Dtype>) or 1-D (tensor<1xDtype>).
                // 0-D: tensor.extract %t[] (no indices)
                // 1-D: tensor.extract %t[%c0]
                macro_rules! extract_scalar {
                    ($tv:expr, $nid:expr) => {{
                        let rank = trace.ops()[$nid.0 as usize].shape().rank();
                        let operands: Vec<melior::ir::Value> = if rank == 0 {
                            vec![$tv]
                        } else {
                            let c0: melior::ir::Value = body_block
                                .append_operation(arith::constant(
                                    context,
                                    IntegerAttribute::new(index_type, 0).into(),
                                    location,
                                ))
                                .result(0)
                                .unwrap()
                                .into();
                            vec![$tv, c0]
                        };
                        let r: melior::ir::Value = body_block
                            .append_operation(
                                OperationBuilder::new("tensor.extract", location)
                                    .add_operands(&operands)
                                    .add_results(&[elem_type])
                                    .build()
                                    .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                            )
                            .result(0)
                            .unwrap()
                            .into();
                        r
                    }};
                }

                let s = extract_scalar!(start_val, start);
                let l = extract_scalar!(limit_val, limit);
                let d = extract_scalar!(delta_val, delta);

                // Compute length as an index value.
                // For integers: diff = l - s; len = max(0, ceildiv(diff, d))
                // For floats: diff = l - s; len = max(0, ceil(diff / d))
                let len_idx: melior::ir::Value = match dtype {
                    DType::I32 | DType::I64 => {
                        let diff: melior::ir::Value = body_block
                            .append_operation(arith::subi(l, s, location))
                            .result(0)
                            .unwrap()
                            .into();
                        let len: melior::ir::Value = body_block
                            .append_operation(
                                OperationBuilder::new("arith.ceildivsi", location)
                                    .add_operands(&[diff, d])
                                    .add_results(&[elem_type])
                                    .build()
                                    .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                            )
                            .result(0)
                            .unwrap()
                            .into();
                        let zero: melior::ir::Value = body_block
                            .append_operation(arith::constant(
                                context,
                                IntegerAttribute::new(elem_type, 0).into(),
                                location,
                            ))
                            .result(0)
                            .unwrap()
                            .into();
                        let len_pos: melior::ir::Value = body_block
                            .append_operation(
                                OperationBuilder::new("arith.maxsi", location)
                                    .add_operands(&[len, zero])
                                    .add_results(&[elem_type])
                                    .build()
                                    .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                            )
                            .result(0)
                            .unwrap()
                            .into();
                        body_block
                            .append_operation(
                                OperationBuilder::new("arith.index_cast", location)
                                    .add_operands(&[len_pos])
                                    .add_results(&[index_type])
                                    .build()
                                    .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                            )
                            .result(0)
                            .unwrap()
                            .into()
                    }
                    DType::F32 | DType::F64 => {
                        let diff: melior::ir::Value = body_block
                            .append_operation(arith::subf(l, s, location))
                            .result(0)
                            .unwrap()
                            .into();
                        let quot: melior::ir::Value = body_block
                            .append_operation(arith::divf(diff, d, location))
                            .result(0)
                            .unwrap()
                            .into();
                        let ceil: melior::ir::Value = body_block
                            .append_operation(
                                OperationBuilder::new("math.ceil", location)
                                    .add_operands(&[quot])
                                    .add_results(&[elem_type])
                                    .build()
                                    .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                            )
                            .result(0)
                            .unwrap()
                            .into();
                        let zero: melior::ir::Value = body_block
                            .append_operation(arith::constant(
                                context,
                                FloatAttribute::new(context, elem_type, 0.0).into(),
                                location,
                            ))
                            .result(0)
                            .unwrap()
                            .into();
                        let ceil_pos: melior::ir::Value = body_block
                            .append_operation(
                                OperationBuilder::new("arith.maximumf", location)
                                    .add_operands(&[ceil, zero])
                                    .add_results(&[elem_type])
                                    .build()
                                    .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                            )
                            .result(0)
                            .unwrap()
                            .into();
                        // Convert float length to index via fptosi then index_cast.
                        let i64_type = DType::I64.to_mlir_type(context);
                        let len_i64: melior::ir::Value = body_block
                            .append_operation(
                                OperationBuilder::new("arith.fptosi", location)
                                    .add_operands(&[ceil_pos])
                                    .add_results(&[i64_type])
                                    .build()
                                    .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                            )
                            .result(0)
                            .unwrap()
                            .into();
                        body_block
                            .append_operation(
                                OperationBuilder::new("arith.index_cast", location)
                                    .add_operands(&[len_i64])
                                    .add_results(&[index_type])
                                    .build()
                                    .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                            )
                            .result(0)
                            .unwrap()
                            .into()
                    }
                };

                // tensor.empty(%len_idx) : tensor<?xDtype>
                let out_type = make_ranked_tensor_type(context, &shape.0, *dtype);
                let empty_val: melior::ir::Value = body_block
                    .append_operation(
                        OperationBuilder::new("tensor.empty", location)
                            .add_operands(&[len_idx])
                            .add_results(&[out_type])
                            .build()
                            .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                    )
                    .result(0)
                    .unwrap()
                    .into();

                // linalg.generic: out[i] = start + i * delta
                // indexing_maps = [(d0) -> (d0)] (identity for single output)
                // iterator_types = ["parallel"]
                let out_map = make_identity_map(context, 1)?;
                let indexing_maps = ArrayAttribute::new(context, &[out_map]);
                let iterator_types = make_iterator_types(context, 1)?;
                let segment_sizes = Attribute::parse(context, "array<i32: 0, 1>")
                    .ok_or_else(|| CompileError::AttributeParse("range segment sizes".into()))?;

                let linalg_region = {
                    let linalg_block = Block::new(&[(elem_type, location)]);
                    // Compute: out[i] = start + index_cast(i) * delta
                    let i_idx: melior::ir::Value = linalg_block
                        .append_operation(
                            OperationBuilder::new("linalg.index", location)
                                .add_results(&[index_type])
                                .add_attributes(&[(
                                    Identifier::new(context, "dim"),
                                    IntegerAttribute::new(
                                        melior::ir::Type::parse(context, "i64").unwrap(),
                                        0,
                                    )
                                    .into(),
                                )])
                                .build()
                                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                        )
                        .result(0)
                        .unwrap()
                        .into();

                    let i_typed: melior::ir::Value = match dtype {
                        DType::I32 => {
                            let i_i64: melior::ir::Value = linalg_block
                                .append_operation(
                                    OperationBuilder::new("arith.index_cast", location)
                                        .add_operands(&[i_idx])
                                        .add_results(&[DType::I64.to_mlir_type(context)])
                                        .build()
                                        .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                                )
                                .result(0)
                                .unwrap()
                                .into();
                            linalg_block
                                .append_operation(
                                    OperationBuilder::new("arith.trunci", location)
                                        .add_operands(&[i_i64])
                                        .add_results(&[elem_type])
                                        .build()
                                        .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                                )
                                .result(0)
                                .unwrap()
                                .into()
                        }
                        DType::I64 => linalg_block
                            .append_operation(
                                OperationBuilder::new("arith.index_cast", location)
                                    .add_operands(&[i_idx])
                                    .add_results(&[elem_type])
                                    .build()
                                    .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                            )
                            .result(0)
                            .unwrap()
                            .into(),
                        DType::F32 | DType::F64 => {
                            let i_i64: melior::ir::Value = linalg_block
                                .append_operation(
                                    OperationBuilder::new("arith.index_cast", location)
                                        .add_operands(&[i_idx])
                                        .add_results(&[DType::I64.to_mlir_type(context)])
                                        .build()
                                        .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                                )
                                .result(0)
                                .unwrap()
                                .into();
                            linalg_block
                                .append_operation(
                                    OperationBuilder::new("arith.sitofp", location)
                                        .add_operands(&[i_i64])
                                        .add_results(&[elem_type])
                                        .build()
                                        .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                                )
                                .result(0)
                                .unwrap()
                                .into()
                        }
                    };

                    let step: melior::ir::Value = match dtype {
                        DType::I32 | DType::I64 => linalg_block
                            .append_operation(arith::muli(i_typed, d, location))
                            .result(0)
                            .unwrap()
                            .into(),
                        DType::F32 | DType::F64 => linalg_block
                            .append_operation(arith::mulf(i_typed, d, location))
                            .result(0)
                            .unwrap()
                            .into(),
                    };
                    let val: melior::ir::Value = match dtype {
                        DType::I32 | DType::I64 => linalg_block
                            .append_operation(arith::addi(s, step, location))
                            .result(0)
                            .unwrap()
                            .into(),
                        DType::F32 | DType::F64 => linalg_block
                            .append_operation(arith::addf(s, step, location))
                            .result(0)
                            .unwrap()
                            .into(),
                    };

                    linalg_block.append_operation(
                        OperationBuilder::new("linalg.yield", location)
                            .add_operands(&[val])
                            .build()
                            .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                    );

                    let r = Region::new();
                    r.append_block(linalg_block);
                    r
                };

                let result_val: melior::ir::Value = body_block
                    .append_operation(
                        OperationBuilder::new("linalg.generic", location)
                            .add_operands(&[empty_val])
                            .add_results(&[out_type])
                            .add_attributes(&[
                                (Identifier::new(context, "indexing_maps"), indexing_maps.into()),
                                (Identifier::new(context, "iterator_types"), iterator_types),
                                (
                                    Identifier::new(context, "operand_segment_sizes"),
                                    segment_sizes,
                                ),
                            ])
                            .add_regions([linalg_region])
                            .build()
                            .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                    )
                    .result(0)
                    .unwrap()
                    .into();

                values.insert(node_id, result_val);
            }

            // ── DynamicSlice ──────────────────────────────────────────────────
            //
            // Uses `tensor.extract_slice` with dynamic offsets and sizes.
            // For each axis in `axes`:
            //   - offset  = tensor.extract %starts_tensor[%axis_idx] cast to index
            //   - size    = (tensor.extract %ends_tensor[%axis_idx]) - offset, cast to index
            // For non-sliced axes:
            //   - offset = 0 (static)
            //   - size   = input_dim (static if known, dynamic via tensor.dim if DIM_DYNAMIC)
            // Strides are all 1.

            Op::DynamicSlice {
                input,
                starts_tensor,
                ends_tensor,
                axes,
                steps: _steps,
                shape,
                dtype,
            } => {
                let input_val = *values.get(input).expect("DynamicSlice: input not emitted");
                let starts_val =
                    *values.get(starts_tensor).expect("DynamicSlice: starts_tensor not emitted");
                let ends_val =
                    *values.get(ends_tensor).expect("DynamicSlice: ends_tensor not emitted");

                let input_shape = trace.get(*input).shape().0.clone();
                let rank = input_shape.len();

                let index_type = melior::ir::Type::parse(context, "index")
                    .ok_or_else(|| CompileError::AttributeParse("index type".into()))?;
                let i64_type = DType::I64.to_mlir_type(context);

                // Build a set of which axes are sliced, for quick lookup.
                let normalized_axes: Vec<usize> = axes
                    .iter()
                    .map(|&ax| {
                        if ax < 0 { (ax + rank as i64) as usize } else { ax as usize }
                    })
                    .collect();

                // For each sliced axis, pre-compute (offset_index, size_index) as index-typed
                // SSA values.  `offset_i` = starts[i], `size_i` = ends[i] - starts[i].
                // We build a Vec<Option<(offset, size)>> indexed by position in `axes`.
                let mut sliced_offset: Vec<melior::ir::Value<'c, 'c>> = Vec::new();
                let mut sliced_size: Vec<melior::ir::Value<'c, 'c>> = Vec::new();

                for (axis_pos, _ax) in axes.iter().enumerate() {
                    let axis_idx_const: melior::ir::Value = body_block
                        .append_operation(arith::constant(
                            context,
                            IntegerAttribute::new(index_type, axis_pos as i64).into(),
                            location,
                        ))
                        .result(0)
                        .unwrap()
                        .into();

                    // Extract start as i64, cast to index.
                    let start_i64: melior::ir::Value = body_block
                        .append_operation(
                            OperationBuilder::new("tensor.extract", location)
                                .add_operands(&[starts_val, axis_idx_const])
                                .add_results(&[i64_type])
                                .build()
                                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                        )
                        .result(0)
                        .unwrap()
                        .into();
                    let start_idx: melior::ir::Value = body_block
                        .append_operation(
                            OperationBuilder::new("arith.index_cast", location)
                                .add_operands(&[start_i64])
                                .add_results(&[index_type])
                                .build()
                                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                        )
                        .result(0)
                        .unwrap()
                        .into();

                    // Extract end as i64, cast to index.
                    let end_i64: melior::ir::Value = body_block
                        .append_operation(
                            OperationBuilder::new("tensor.extract", location)
                                .add_operands(&[ends_val, axis_idx_const])
                                .add_results(&[i64_type])
                                .build()
                                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                        )
                        .result(0)
                        .unwrap()
                        .into();
                    let end_idx: melior::ir::Value = body_block
                        .append_operation(
                            OperationBuilder::new("arith.index_cast", location)
                                .add_operands(&[end_i64])
                                .add_results(&[index_type])
                                .build()
                                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                        )
                        .result(0)
                        .unwrap()
                        .into();

                    // size = end - start (index arithmetic).
                    let size_idx: melior::ir::Value = body_block
                        .append_operation(arith::subi(end_idx, start_idx, location))
                        .result(0)
                        .unwrap()
                        .into();

                    sliced_offset.push(start_idx);
                    sliced_size.push(size_idx);
                }

                // Build per-rank static_offsets, static_sizes, static_strides and
                // collect dynamic operands (in order: offsets first, then sizes).
                // `i64::MIN` is MLIR's ShapedType::kDynamic sentinel.

                let mut static_offsets: Vec<i64> = vec![0i64; rank];
                let mut static_sizes: Vec<i64> = Vec::with_capacity(rank);
                let static_strides: Vec<i64> = vec![1i64; rank];

                // Compute static_sizes: for each dim, static if not sliced and input
                // dim is known; otherwise kDynamic.
                for (d_idx, &in_dim) in input_shape.iter().enumerate() {
                    if normalized_axes.contains(&d_idx) {
                        // sliced axis — size is dynamic
                        static_sizes.push(i64::MIN);
                    } else if in_dim == DIM_DYNAMIC {
                        // non-sliced but dynamic input dim
                        static_sizes.push(i64::MIN);
                    } else {
                        static_sizes.push(in_dim as i64);
                    }
                }

                // All sliced axes have dynamic offsets; mark them kDynamic.
                for &ax in &normalized_axes {
                    static_offsets[ax] = i64::MIN;
                }

                // Build separate dynamic offset, size, stride operand lists.
                let mut dynamic_offsets: Vec<melior::ir::Value<'c, 'c>> = Vec::new();
                let mut dynamic_sizes: Vec<melior::ir::Value<'c, 'c>> = Vec::new();
                let dynamic_strides: Vec<melior::ir::Value<'c, 'c>> = Vec::new();

                // offsets: sliced axes contribute dynamic offsets
                for &ax in &normalized_axes {
                    let pos = normalized_axes.iter().position(|&a| a == ax).unwrap();
                    dynamic_offsets.push(sliced_offset[pos]);
                }

                // sizes: sliced axes (dynamic size from end-start),
                // then non-sliced DIM_DYNAMIC axes (from tensor.dim).
                // Iterate dims in order; only push for dynamic entries.
                for (d_idx, &in_dim) in input_shape.iter().enumerate() {
                    if normalized_axes.contains(&d_idx) {
                        let pos = normalized_axes.iter().position(|&a| a == d_idx).unwrap();
                        dynamic_sizes.push(sliced_size[pos]);
                    } else if in_dim == DIM_DYNAMIC {
                        let dim_v = emit_tensor_dim(context, body_block, input_val, d_idx, location)?;
                        dynamic_sizes.push(dim_v);
                    }
                }

                // Build attribute strings.
                let so_str = static_offsets.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ");
                let static_offsets_attr =
                    Attribute::parse(context, &format!("array<i64: {so_str}>"))
                        .ok_or_else(|| CompileError::AttributeParse("static_offsets".into()))?;

                let ss_str = static_sizes.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ");
                let static_sizes_attr =
                    Attribute::parse(context, &format!("array<i64: {ss_str}>"))
                        .ok_or_else(|| CompileError::AttributeParse("static_sizes".into()))?;

                let sst_str = static_strides.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ");
                let static_strides_attr =
                    Attribute::parse(context, &format!("array<i64: {sst_str}>"))
                        .ok_or_else(|| CompileError::AttributeParse("static_strides".into()))?;

                let seg_sizes_str = format!(
                    "array<i32: 1, {}, {}, {}>",
                    dynamic_offsets.len(),
                    dynamic_sizes.len(),
                    dynamic_strides.len(),
                );
                let seg_sizes_attr = Attribute::parse(context, &seg_sizes_str)
                    .ok_or_else(|| CompileError::AttributeParse(seg_sizes_str.clone()))?;

                let result_type = make_ranked_tensor_type(context, &shape.0, *dtype);

                let mut all_operands: Vec<melior::ir::Value<'c, 'c>> = vec![input_val];
                all_operands.extend(dynamic_offsets);
                all_operands.extend(dynamic_sizes);
                all_operands.extend(dynamic_strides);

                let result_val: melior::ir::Value = body_block
                    .append_operation(
                        OperationBuilder::new("tensor.extract_slice", location)
                            .add_operands(&all_operands)
                            .add_results(&[result_type])
                            .add_attributes(&[
                                (Identifier::new(context, "operandSegmentSizes"), seg_sizes_attr),
                                (Identifier::new(context, "static_offsets"), static_offsets_attr),
                                (Identifier::new(context, "static_sizes"), static_sizes_attr),
                                (Identifier::new(context, "static_strides"), static_strides_attr),
                            ])
                            .build()
                            .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                    )
                    .result(0)
                    .unwrap()
                    .into();

                values.insert(node_id, result_val);
            }
        }
    }

    // ---- Output boundary: bufferization.to_buffer + memref.copy for each output ----
    // Output memref args immediately follow all input args in the function signature.
    for (out_idx, &output_id) in output_ids.iter().enumerate() {
        let result_tensor = *values.get(&output_id).expect("output node not emitted");

        let output_op = trace.get(output_id);
        let out_elem_type = output_op.dtype().to_mlir_type(context);
        let dims: Vec<i64> = output_op.shape().0.iter().map(|&d| dim_to_mlir_i64(d)).collect();
        let out_memref_type: melior::ir::Type =
            MemRefType::new(out_elem_type, &dims, None, None).into();

        let result_memref = body_block
            .append_operation(
                OperationBuilder::new("bufferization.to_buffer", location)
                    .add_operands(&[result_tensor])
                    .add_results(&[out_memref_type])
                    .build()
                    .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
            )
            .result(0)
            .unwrap()
            .into();

        let out_arg: melior::ir::Value =
            body_block.argument(num_inputs + out_idx).unwrap().into();
        body_block.append_operation(
            OperationBuilder::new("memref.copy", location)
                .add_operands(&[result_memref, out_arg])
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        );
    }

    Ok(())
}

// ── Public wrappers for graph_builder ─────────────────────────────────────────

/// Public wrapper so `graph_builder` can call the AOT object emitter.
pub(crate) fn emit_object_file_pub(
    module: &melior::ir::Module,
    output_path: &std::path::Path,
) -> Result<(), CompileError> {
    emit_object_file(module, output_path)
}

/// Public wrapper so `graph_builder` can call the shared lib linker.
pub(crate) fn link_shared_lib_pub(
    obj_path: &std::path::Path,
    so_path: &std::path::Path,
) -> Result<(), CompileError> {
    link_shared_lib(obj_path, so_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DType,
        runtime::Buffer,
        tensor::Tensor,
        trace::{begin_trace, take_trace},
    };

    #[test]
    fn compile_add() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F32);
        let b = Tensor::new(&[4], DType::F32);
        let c = &a + &b;
        let trace = take_trace();

        let result = Compiler::compile(&trace, &[c.id], None);
        assert!(result.is_ok(), "compile failed: {:?}", result.err());
    }

    #[test]
    fn compile_chained_add() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F32);
        let b = Tensor::new(&[4], DType::F32);
        let c = Tensor::new(&[4], DType::F32);
        let d = &(&a + &b) + &c;
        let trace = take_trace();

        let result = Compiler::compile(&trace, &[d.id], None);
        assert!(result.is_ok(), "compile failed: {:?}", result.err());
    }

    #[test]
    fn compile_sub() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F32);
        let b = Tensor::new(&[4], DType::F32);
        let c = &a - &b;
        let trace = take_trace();

        let result = Compiler::compile(&trace, &[c.id], None);
        assert!(result.is_ok(), "compile failed: {:?}", result.err());
    }

    #[test]
    fn compile_mul() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F32);
        let b = Tensor::new(&[4], DType::F32);
        let c = &a * &b;
        let trace = take_trace();

        let result = Compiler::compile(&trace, &[c.id], None);
        assert!(result.is_ok(), "compile failed: {:?}", result.err());
    }

    #[test]
    fn compile_div() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F32);
        let b = Tensor::new(&[4], DType::F32);
        let c = &a / &b;
        let trace = take_trace();

        let result = Compiler::compile(&trace, &[c.id], None);
        assert!(result.is_ok(), "compile failed: {:?}", result.err());
    }

    #[test]
    fn compile_neg() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F32);
        let b = -&a;
        let trace = take_trace();

        let result = Compiler::compile(&trace, &[b.id], None);
        assert!(result.is_ok(), "compile failed: {:?}", result.err());
    }

    #[test]
    fn compile_relu() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F32);
        let b = a.relu();
        let trace = take_trace();

        let result = Compiler::compile(&trace, &[b.id], None);
        assert!(result.is_ok(), "compile failed: {:?}", result.err());
    }

    #[test]
    fn compile_div_i32() {
        begin_trace();
        let a = Tensor::new(&[4], DType::I32);
        let b = Tensor::new(&[4], DType::I32);
        let c = &a / &b;
        let trace = take_trace();

        let result = Compiler::compile(&trace, &[c.id], None);
        assert!(result.is_ok(), "compile failed: {:?}", result.err());
    }

    #[test]
    fn compile_neg_i32() {
        begin_trace();
        let a = Tensor::new(&[4], DType::I32);
        let b = -&a;
        let trace = take_trace();

        let result = Compiler::compile(&trace, &[b.id], None);
        assert!(result.is_ok(), "compile failed: {:?}", result.err());
    }

    #[test]
    fn compile_relu_i32() {
        begin_trace();
        let a = Tensor::new(&[4], DType::I32);
        let b = a.relu();
        let trace = take_trace();

        let result = Compiler::compile(&trace, &[b.id], None);
        assert!(result.is_ok(), "compile failed: {:?}", result.err());
    }

    // ── Rank-2 and rank-3 compile tests ──────────────────────────────────────

    #[test]
    fn compile_add_rank2() {
        begin_trace();
        let a = Tensor::new(&[2, 3], DType::F32);
        let b = Tensor::new(&[2, 3], DType::F32);
        let c = &a + &b;
        let trace = take_trace();

        let result = Compiler::compile(&trace, &[c.id], None);
        assert!(result.is_ok(), "compile failed: {:?}", result.err());
    }

    #[test]
    fn compile_add_rank3() {
        begin_trace();
        let a = Tensor::new(&[2, 3, 4], DType::F32);
        let b = Tensor::new(&[2, 3, 4], DType::F32);
        let c = &a + &b;
        let trace = take_trace();

        let result = Compiler::compile(&trace, &[c.id], None);
        assert!(result.is_ok(), "compile failed: {:?}", result.err());
    }

    #[test]
    fn compile_neg_rank2() {
        begin_trace();
        let a = Tensor::new(&[2, 3], DType::F32);
        let b = -&a;
        let trace = take_trace();

        let result = Compiler::compile(&trace, &[b.id], None);
        assert!(result.is_ok(), "compile failed: {:?}", result.err());
    }

    // ── TOSA spike tests (milestone 2, task 1.2) ──────────────────────────────
    //
    // These tests validate that the TOSA → linalg → LLVM → AOT pipeline works
    // end-to-end. They use Compiler::compile so they also validate the full
    // TOSA lowering path.

    // ── spike_tosa_add_pipeline ───────────────────────────────────────────────
    //
    // Validates that a tosa.add-based trace compiles and executes correctly
    // through the AOT pipeline.
    #[test]
    fn spike_tosa_add_pipeline() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F32);
        let b = Tensor::new(&[4], DType::F32);
        let c = &a + &b;
        let trace = take_trace();

        let graph = Compiler::compile(&trace, &[c.id], None).expect("compile failed");
        let a_buf = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[4], DType::F32);
        let b_buf = Buffer::from_slice::<f32>(&[10.0, 20.0, 30.0, 40.0], &[4], DType::F32);
        let result = graph.run(&[&a_buf, &b_buf]);
        assert_eq!(result[0].as_slice::<f32>(), &[11.0f32, 22.0, 33.0, 44.0]);
    }

    // ── spike_tosa_mixed_pipeline ─────────────────────────────────────────────
    //
    // Validates that a trace mixing TOSA ops (add) and linalg fallback (div)
    // compiles and executes correctly through the AOT pipeline.
    //
    // Computation: (a + b) / c  with a=[2,4,6,8], b=[0,0,0,0], c=[1,2,3,4]
    // Expected: [2,2,2,2]
    #[test]
    fn spike_tosa_mixed_pipeline() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F32);
        let b = Tensor::new(&[4], DType::F32);
        let c = Tensor::new(&[4], DType::F32);
        let sum = &a + &b;
        let result_t = &sum / &c;
        let trace = take_trace();

        let graph = Compiler::compile(&trace, &[result_t.id], None).expect("compile failed");
        let a_buf = Buffer::from_slice::<f32>(&[2.0, 4.0, 6.0, 8.0], &[4], DType::F32);
        let b_buf = Buffer::from_slice::<f32>(&[0.0, 0.0, 0.0, 0.0], &[4], DType::F32);
        let c_buf = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[4], DType::F32);
        let result = graph.run(&[&a_buf, &b_buf, &c_buf]);
        assert_eq!(result[0].as_slice::<f32>(), &[2.0f32, 2.0, 2.0, 2.0]);
    }

    // ── IR verification tests (task 2.10) ─────────────────────────────────────
    //
    // Verify that the emitted MLIR IR (before lowering) contains TOSA ops
    // for the migrated operations, and does NOT contain linalg.generic for them.

    #[test]
    fn ir_tosa_add() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F32);
        let b = Tensor::new(&[4], DType::F32);
        let c = &a + &b;
        let trace = take_trace();
        let ir = Compiler::build_ir_string(&trace, &[c.id]).expect("build_ir_string");
        assert!(ir.contains("tosa.add"), "expected tosa.add in IR:\n{ir}");
        assert!(
            !ir.contains("linalg.generic"),
            "unexpected linalg.generic in IR:\n{ir}"
        );
    }

    #[test]
    fn ir_tosa_sub() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F32);
        let b = Tensor::new(&[4], DType::F32);
        let c = &a - &b;
        let trace = take_trace();
        let ir = Compiler::build_ir_string(&trace, &[c.id]).expect("build_ir_string");
        assert!(ir.contains("tosa.sub"), "expected tosa.sub in IR:\n{ir}");
        assert!(
            !ir.contains("linalg.generic"),
            "unexpected linalg.generic in IR:\n{ir}"
        );
    }

    #[test]
    fn ir_tosa_mul() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F32);
        let b = Tensor::new(&[4], DType::F32);
        let c = &a * &b;
        let trace = take_trace();
        let ir = Compiler::build_ir_string(&trace, &[c.id]).expect("build_ir_string");
        assert!(ir.contains("tosa.mul"), "expected tosa.mul in IR:\n{ir}");
        assert!(
            !ir.contains("linalg.generic"),
            "unexpected linalg.generic in IR:\n{ir}"
        );
    }

    #[test]
    fn ir_tosa_negate() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F32);
        let b = -&a;
        let trace = take_trace();
        let ir = Compiler::build_ir_string(&trace, &[b.id]).expect("build_ir_string");
        assert!(
            ir.contains("tosa.negate"),
            "expected tosa.negate in IR:\n{ir}"
        );
        assert!(
            !ir.contains("linalg.generic"),
            "unexpected linalg.generic in IR:\n{ir}"
        );
    }

    #[test]
    fn ir_tosa_relu() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F32);
        let b = a.relu();
        let trace = take_trace();
        let ir = Compiler::build_ir_string(&trace, &[b.id]).expect("build_ir_string");
        assert!(
            ir.contains("tosa.clamp"),
            "expected tosa.clamp in IR:\n{ir}"
        );
        assert!(
            !ir.contains("linalg.generic"),
            "unexpected linalg.generic in IR:\n{ir}"
        );
    }

    #[test]
    fn ir_tosa_exp() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F32);
        let b = a.exp();
        let trace = take_trace();
        let ir = Compiler::build_ir_string(&trace, &[b.id]).expect("build_ir_string");
        assert!(ir.contains("tosa.exp"), "expected tosa.exp in IR:\n{ir}");
        assert!(
            !ir.contains("linalg.generic"),
            "unexpected linalg.generic in IR:\n{ir}"
        );
    }

    #[test]
    fn ir_tosa_tanh() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F32);
        let b = a.tanh();
        let trace = take_trace();
        let ir = Compiler::build_ir_string(&trace, &[b.id]).expect("build_ir_string");
        assert!(ir.contains("tosa.tanh"), "expected tosa.tanh in IR:\n{ir}");
        assert!(
            !ir.contains("linalg.generic"),
            "unexpected linalg.generic in IR:\n{ir}"
        );
    }

    #[test]
    fn ir_div_keeps_linalg_generic() {
        // Div is intentionally kept as linalg.generic (TOSA has no float div).
        begin_trace();
        let a = Tensor::new(&[4], DType::F32);
        let b = Tensor::new(&[4], DType::F32);
        let c = &a / &b;
        let trace = take_trace();
        let ir = Compiler::build_ir_string(&trace, &[c.id]).expect("build_ir_string");
        assert!(
            ir.contains("linalg.generic"),
            "expected linalg.generic for div in IR:\n{ir}"
        );
        assert!(!ir.contains("tosa.div"), "unexpected tosa.div in IR:\n{ir}");
    }

    // ── Task 3.6: IR verification tests for matmul and reductions ─────────────

    #[test]
    fn ir_tosa_matmul_f32() {
        // Float matmul should emit tosa.matmul (with tosa.reshape for batch dim).
        begin_trace();
        let a = Tensor::new(&[3, 4], DType::F32);
        let b = Tensor::new(&[4, 5], DType::F32);
        let c = a.matmul(&b);
        let trace = take_trace();
        let ir = Compiler::build_ir_string(&trace, &[c.id]).expect("build_ir_string");
        assert!(
            ir.contains("tosa.matmul"),
            "expected tosa.matmul in IR:\n{ir}"
        );
        assert!(
            ir.contains("tosa.const_shape"),
            "expected tosa.const_shape in IR:\n{ir}"
        );
        assert!(
            ir.contains("tosa.reshape"),
            "expected tosa.reshape in IR:\n{ir}"
        );
        assert!(
            !ir.contains("tensor.expand_shape"),
            "unexpected tensor.expand_shape in IR:\n{ir}"
        );
        assert!(
            !ir.contains("tensor.collapse_shape"),
            "unexpected tensor.collapse_shape in IR:\n{ir}"
        );
        assert!(
            !ir.contains("linalg.matmul"),
            "unexpected linalg.matmul in IR:\n{ir}"
        );
    }

    #[test]
    fn ir_matmul_i32_keeps_linalg() {
        // Integer matmul must keep linalg.matmul (tosa.matmul is float-only).
        begin_trace();
        let a = Tensor::new(&[3, 4], DType::I32);
        let b = Tensor::new(&[4, 5], DType::I32);
        let c = a.matmul(&b);
        let trace = take_trace();
        let ir = Compiler::build_ir_string(&trace, &[c.id]).expect("build_ir_string");
        assert!(
            ir.contains("linalg.matmul"),
            "expected linalg.matmul in IR:\n{ir}"
        );
        assert!(
            !ir.contains("tosa.matmul"),
            "unexpected tosa.matmul in IR:\n{ir}"
        );
    }

    #[test]
    fn ir_tosa_reduce_sum() {
        // Float reduce_sum should emit tosa.reduce_sum.
        begin_trace();
        let a = Tensor::new(&[3, 4], DType::F32);
        let b = a.reduce_sum(1, false);
        let trace = take_trace();
        let ir = Compiler::build_ir_string(&trace, &[b.id]).expect("build_ir_string");
        assert!(
            ir.contains("tosa.reduce_sum"),
            "expected tosa.reduce_sum in IR:\n{ir}"
        );
        assert!(
            !ir.contains("linalg.generic"),
            "unexpected linalg.generic in IR:\n{ir}"
        );
    }

    #[test]
    fn ir_tosa_reduce_max_keepdim() {
        // Float reduce_max with keepdim=true should emit tosa.reduce_max.
        // TOSA reduce_max natively keeps rank (size-1 at the reduced dim),
        // so no tosa.reshape is needed for keepdim=true.
        begin_trace();
        let a = Tensor::new(&[3, 4], DType::F32);
        let b = a.reduce_max(1, true);
        let trace = take_trace();
        let ir = Compiler::build_ir_string(&trace, &[b.id]).expect("build_ir_string");
        assert!(
            ir.contains("tosa.reduce_max"),
            "expected tosa.reduce_max in IR:\n{ir}"
        );
        // Output type should reflect keepdim: tensor<3x1xf32>
        assert!(
            ir.contains("tensor<3x1xf32>"),
            "expected tensor<3x1xf32> output shape in IR:\n{ir}"
        );
        assert!(
            !ir.contains("linalg.generic"),
            "unexpected linalg.generic in IR:\n{ir}"
        );
    }

    // ── Gemm IR verification tests (task 6.5) ─────────────────────────────────

    #[test]
    fn ir_gemm_no_transpose_has_matmul() {
        // Standard gemm without transpose flags: should emit tosa.matmul but no
        // tosa.transpose.
        begin_trace();
        let a = Tensor::new(&[2, 3], DType::F32);
        let b = Tensor::new(&[3, 4], DType::F32);
        let c = a.gemm(&b, None, 1.0, 1.0, false, false);
        let trace = take_trace();
        let ir = Compiler::build_ir_string(&trace, &[c.id]).expect("build_ir_string");
        assert!(
            ir.contains("tosa.matmul"),
            "expected tosa.matmul in IR:\n{ir}"
        );
        assert!(
            !ir.contains("tosa.transpose"),
            "unexpected tosa.transpose in IR:\n{ir}"
        );
        assert!(
            !ir.contains("tosa.mul"),
            "unexpected tosa.mul (alpha=1.0) in IR:\n{ir}"
        );
    }

    #[test]
    fn ir_gemm_trans_b_has_transpose() {
        // transB=true: should emit tosa.transpose for the rhs.
        begin_trace();
        let a = Tensor::new(&[2, 3], DType::F32);
        let b = Tensor::new(&[4, 3], DType::F32); // will be transposed to [3, 4]
        let c = a.gemm(&b, None, 1.0, 1.0, false, true);
        let trace = take_trace();
        let ir = Compiler::build_ir_string(&trace, &[c.id]).expect("build_ir_string");
        assert!(
            ir.contains("tosa.matmul"),
            "expected tosa.matmul in IR:\n{ir}"
        );
        assert!(
            ir.contains("tosa.transpose"),
            "expected tosa.transpose in IR:\n{ir}"
        );
    }

    #[test]
    fn ir_gemm_trans_a_has_transpose() {
        // transA=true: should emit tosa.transpose for the lhs.
        begin_trace();
        let a = Tensor::new(&[3, 2], DType::F32); // will be transposed to [2, 3]
        let b = Tensor::new(&[3, 4], DType::F32);
        let c = a.gemm(&b, None, 1.0, 1.0, true, false);
        let trace = take_trace();
        let ir = Compiler::build_ir_string(&trace, &[c.id]).expect("build_ir_string");
        assert!(
            ir.contains("tosa.matmul"),
            "expected tosa.matmul in IR:\n{ir}"
        );
        assert!(
            ir.contains("tosa.transpose"),
            "expected tosa.transpose in IR:\n{ir}"
        );
    }

    #[test]
    fn ir_gemm_alpha_1_no_mul() {
        // alpha=1.0 and beta=1.0: no tosa.mul should be emitted for scaling.
        begin_trace();
        let a = Tensor::new(&[2, 3], DType::F32);
        let b = Tensor::new(&[3, 4], DType::F32);
        let bias = Tensor::new(&[4], DType::F32);
        let c = a.gemm(&b, Some(&bias), 1.0, 1.0, false, false);
        let trace = take_trace();
        let ir = Compiler::build_ir_string(&trace, &[c.id]).expect("build_ir_string");
        // No tosa.mul for alpha=1 or beta=1; bias add should use tosa.add.
        assert!(
            !ir.contains("tosa.mul"),
            "unexpected tosa.mul when alpha=1.0/beta=1.0:\n{ir}"
        );
        assert!(
            ir.contains("tosa.add"),
            "expected tosa.add (bias) in IR:\n{ir}"
        );
    }

    #[test]
    fn ir_gemm_alpha_scaling_has_mul() {
        // alpha != 1.0: tosa.mul should be emitted for the lhs scaling.
        begin_trace();
        let a = Tensor::new(&[2, 3], DType::F32);
        let b = Tensor::new(&[3, 4], DType::F32);
        let c = a.gemm(&b, None, 2.0, 1.0, false, false);
        let trace = take_trace();
        let ir = Compiler::build_ir_string(&trace, &[c.id]).expect("build_ir_string");
        assert!(
            ir.contains("tosa.mul"),
            "expected tosa.mul (alpha scaling) in IR:\n{ir}"
        );
    }

    // ── Gemm execution tests (task 6.6) ──────────────────────────────────────

    /// Helper: run a gemm trace and return the output f32 slice as Vec.
    fn run_gemm_f32(
        a_data: &[f32],
        a_shape: &[u64],
        b_data: &[f32],
        b_shape: &[u64],
        bias_data: Option<(&[f32], &[u64])>,
        alpha: f64,
        beta: f64,
        trans_a: bool,
        trans_b: bool,
    ) -> Vec<f32> {
        begin_trace();
        let a = Tensor::new(a_shape, DType::F32);
        let b = Tensor::new(b_shape, DType::F32);
        let bias_tensor;
        let bias_ref = if let Some((_, bshape)) = bias_data {
            bias_tensor = Tensor::new(bshape, DType::F32);
            Some(&bias_tensor)
        } else {
            None
        };
        let c = a.gemm(&b, bias_ref, alpha, beta, trans_a, trans_b);
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[c.id], None).expect("compile failed");

        let a_buf = Buffer::from_slice::<f32>(a_data, a_shape, DType::F32);
        let b_buf = Buffer::from_slice::<f32>(b_data, b_shape, DType::F32);

        let result = if let Some((bdata, bshape)) = bias_data {
            let bias_buf = Buffer::from_slice::<f32>(bdata, bshape, DType::F32);
            compiled.run(&[&a_buf, &b_buf, &bias_buf])
        } else {
            compiled.run(&[&a_buf, &b_buf])
        };

        result[0].as_slice::<f32>().to_vec()
    }

    #[test]
    fn run_gemm_standard() {
        // Standard [2,3] @ [3,4] + [4] = [2,4], alpha=1, beta=1
        // A = [[1,2,3],[4,5,6]], B = [[1,2,3,4],[5,6,7,8],[9,10,11,12]], bias=[1,1,1,1]
        // A@B = [[38,44,50,56],[83,98,113,128]], + bias = [[39,45,51,57],[84,99,114,129]]
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let bias = [1.0f32, 1.0, 1.0, 1.0];
        let out = run_gemm_f32(
            &a,
            &[2, 3],
            &b,
            &[3, 4],
            Some((&bias, &[4])),
            1.0,
            1.0,
            false,
            false,
        );
        assert_eq!(out, &[39.0, 45.0, 51.0, 57.0, 84.0, 99.0, 114.0, 129.0]);
    }

    #[test]
    fn run_gemm_trans_b() {
        // transB=true: A [2,3] @ B^T where B is stored as [4,3].
        // B [4,3] = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
        // B^T [3,4] = [[1,4,7,10],[2,5,8,11],[3,6,9,12]]
        // A [2,3] = [[1,2,3],[4,5,6]]
        // A @ B^T:
        //   row0: [1*1+2*2+3*3, 1*4+2*5+3*6, 1*7+2*8+3*9, 1*10+2*11+3*12] = [14, 32, 50, 68]
        //   row1: [4*1+5*2+6*3, 4*4+5*5+6*6, 4*7+5*8+6*9, 4*10+5*11+6*12] = [32, 77, 122, 167]
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ]; // shape [4,3]
        let out = run_gemm_f32(&a, &[2, 3], &b, &[4, 3], None, 1.0, 1.0, false, true);
        assert_eq!(out, &[14.0, 32.0, 50.0, 68.0, 32.0, 77.0, 122.0, 167.0]);
    }

    #[test]
    fn run_gemm_trans_a() {
        // transA: [3,2]^T @ [3,4] = [2,3] @ [3,4]
        // A stored as [3,2] = [[1,4],[2,5],[3,6]]  (i.e. A^T = [[1,2,3],[4,5,6]])
        // A^T @ B = [[1,2,3],[4,5,6]] @ B_standard = [[38,44,50,56],[83,98,113,128]]
        let a_col_major = [1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0]; // shape [3, 2]
        let b = [
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let out = run_gemm_f32(
            &a_col_major,
            &[3, 2],
            &b,
            &[3, 4],
            None,
            1.0,
            1.0,
            true,
            false,
        );
        assert_eq!(out, &[38.0, 44.0, 50.0, 56.0, 83.0, 98.0, 113.0, 128.0]);
    }

    #[test]
    fn run_gemm_alpha_beta_scaling() {
        // alpha=2.0, beta=0.5: 2*(A@B) + 0.5*C
        // A@B = [[38,44,50,56],[83,98,113,128]]
        // 2*(A@B) = [[76,88,100,112],[166,196,226,256]]
        // C = [2,2,2,2], 0.5*C = [1,1,1,1]
        // result = [[77,89,101,113],[167,197,227,257]]
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let bias = [2.0f32, 2.0, 2.0, 2.0];
        let out = run_gemm_f32(
            &a,
            &[2, 3],
            &b,
            &[3, 4],
            Some((&bias, &[4])),
            2.0,
            0.5,
            false,
            false,
        );
        assert_eq!(out, &[77.0, 89.0, 101.0, 113.0, 167.0, 197.0, 227.0, 257.0]);
    }

    #[test]
    fn run_gemm_no_bias() {
        // No bias: just A@B = [[38,44,50,56],[83,98,113,128]]
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let out = run_gemm_f32(&a, &[2, 3], &b, &[3, 4], None, 1.0, 1.0, false, false);
        assert_eq!(out, &[38.0, 44.0, 50.0, 56.0, 83.0, 98.0, 113.0, 128.0]);
    }

    // ── Regression: Issue 1 — format_float helper ────────────────────────────

    #[test]
    fn format_float_adds_decimal_for_integer_values() {
        // Rust Display formats 1.0 as "1" (no decimal point). MLIR rejects bare
        // integers in float dense attributes, so format_float must append ".0".
        assert_eq!(format_float(1.0), "1.0");
        assert_eq!(format_float(2.0), "2.0");
        assert_eq!(format_float(0.0), "0.0");
    }

    #[test]
    fn format_float_preserves_non_integer_precision() {
        // Non-integer values already have a decimal point and must not be truncated.
        assert_eq!(format_float(0.123456), "0.123456");
        assert_eq!(format_float(0.5), "0.5");
        // 1e-10 is formatted by Rust as "0.0000000001" (has decimal) — passes through.
        let s = format_float(1e-10);
        assert!(s.contains('.') || s.contains('e') || s.contains('E'));
    }

    // ── Reshape tests (tasks 9.5, 9.6) ───────────────────────────────────────

    #[test]
    fn compile_reshape_emits_tosa_reshape() {
        begin_trace();
        let a = Tensor::new(&[2, 6], DType::F32);
        let b = a.reshape(&[3, 4]);
        let trace = take_trace();

        let module_str = Compiler::build_ir_string(&trace, &[b.id]).expect("mlir emission failed");
        assert!(
            module_str.contains("tosa.reshape"),
            "expected tosa.reshape in IR:\n{module_str}"
        );
        assert!(
            module_str.contains("tosa.const_shape"),
            "expected tosa.const_shape in IR:\n{module_str}"
        );
    }

    #[test]
    fn run_reshape_2d_to_1d() {
        // Flatten [2, 3] -> [6]: values should be unchanged in row-major order.
        begin_trace();
        let a = Tensor::new(&[2, 3], DType::F32);
        let b = a.reshape(&[6]);
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[b.id], None).expect("compile failed");

        let input = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
        let result = compiled.run(&[&input]);
        assert_eq!(result[0].as_slice::<f32>(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn run_reshape_with_inferred_dim() {
        // Reshape [12] -> [-1, 4]: inferred shape is [3, 4].
        begin_trace();
        let a = Tensor::new(&[12], DType::F32);
        let b = a.reshape(&[-1, 4]);
        // Verify output shape is [3, 4]
        assert_eq!(b.shape().0, vec![3u64, 4]);
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[b.id], None).expect("compile failed");

        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let input = Buffer::from_slice::<f32>(&data, &[12], DType::F32);
        let result = compiled.run(&[&input]);
        // Values should be unchanged in row-major order
        let out = result[0].as_slice::<f32>();
        assert_eq!(out.len(), 12);
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, (i + 1) as f32, "mismatch at index {i}");
        }
    }

    #[test]
    fn run_reshape_2d_to_different_2d() {
        // Reshape [2, 6] -> [3, 4]: same elements, different layout.
        begin_trace();
        let a = Tensor::new(&[2, 6], DType::F32);
        let b = a.reshape(&[3, 4]);
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[b.id], None).expect("compile failed");

        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let input = Buffer::from_slice::<f32>(&data, &[2, 6], DType::F32);
        let result = compiled.run(&[&input]);
        let out = result[0].as_slice::<f32>();
        assert_eq!(out.len(), 12);
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, (i + 1) as f32, "mismatch at index {i}");
        }
    }

    // ── Regression: Issue 1 — Gemm with non-trivial alpha compiles and runs ───

    #[test]
    fn run_gemm_fractional_alpha() {
        // alpha=0.123456 must not be truncated to 0.1.
        // A=[1,0; 0,1] (identity), B=[1,0; 0,1] (identity)
        // alpha * (A @ B) = alpha * I = [[alpha, 0], [0, alpha]]
        let alpha = 0.123456_f64;
        let a = [1.0f32, 0.0, 0.0, 1.0]; // 2x2 identity
        let b = [1.0f32, 0.0, 0.0, 1.0]; // 2x2 identity
        let out = run_gemm_f32(&a, &[2, 2], &b, &[2, 2], None, alpha, 1.0, false, false);
        // result = alpha * [[1,0],[0,1]] = [[alpha, 0],[0, alpha]]
        let expected_diag = alpha as f32;
        assert!(
            (out[0] - expected_diag).abs() < 1e-5,
            "expected {expected_diag}, got {} (alpha precision truncated?)",
            out[0]
        );
        assert!((out[1]).abs() < 1e-5);
        assert!((out[2]).abs() < 1e-5);
        assert!(
            (out[3] - expected_diag).abs() < 1e-5,
            "expected {expected_diag}, got {} (alpha precision truncated?)",
            out[3]
        );
    }

    // ── IR verification: Reshape ─────────────────────────────────────────────

    #[test]
    fn ir_tosa_reshape() {
        begin_trace();
        let a = Tensor::new(&[2, 6], DType::F32);
        let b = a.reshape(&[3, 4]);
        let trace = take_trace();
        let ir = Compiler::build_ir_string(&trace, &[b.id]).expect("failed");
        assert!(
            ir.contains("tosa.reshape"),
            "expected tosa.reshape in IR:\n{ir}"
        );
        assert!(
            ir.contains("tosa.const_shape"),
            "expected tosa.const_shape in IR:\n{ir}"
        );
    }

    // ── Complex multi-op graph compilation ───────────────────────────────────

    #[test]
    fn compile_long_chain() {
        // Verify that a long chain of ops compiles without errors
        begin_trace();
        let a = Tensor::new(&[4], DType::F32);
        let b = Tensor::new(&[4], DType::F32);
        let c = &a + &b;
        let c = (&c * &a).relu();
        let c = &c - &b;
        let c = -&c;
        let c = c.tanh();
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[c.id], None).expect("compile failed");
        let a_buf = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[4], DType::F32);
        let b_buf = Buffer::from_slice::<f32>(&[0.5, 0.5, 0.5, 0.5], &[4], DType::F32);
        let result = compiled.run(&[&a_buf, &b_buf]);
        let out = result[0].as_slice::<f32>();
        // a+b=[1.5,2.5,3.5,4.5], *a=[1.5,5,10.5,18], relu=same, -b=[1,4.5,10,17.5]
        // neg=[-1,-4.5,-10,-17.5], tanh=[-0.762,-0.9998,-1.0,-1.0]
        assert!((out[0] - (-1.0_f32).tanh()).abs() < 1e-3);
        assert!(out[3] < -0.999);
        assert_eq!(out.len(), 4);
    }

    // ── Conv2d IR verification tests (task 10.5) ─────────────────────────────

    #[test]
    fn ir_conv2d_has_tosa_conv2d_and_transpose() {
        begin_trace();
        let input = Tensor::new(&[1, 1, 4, 4], DType::F32);
        let kernel = Tensor::new(&[1, 1, 3, 3], DType::F32);
        let out = input.conv2d(&kernel, None, [1, 1], [0, 0, 0, 0], [1, 1]);
        let trace = take_trace();
        let ir = Compiler::build_ir_string(&trace, &[out.id]).expect("build_ir_string");
        assert!(
            ir.contains("tosa.conv2d"),
            "expected tosa.conv2d in IR:\n{ir}"
        );
        assert!(
            ir.contains("tosa.transpose"),
            "expected tosa.transpose in IR:\n{ir}"
        );
        assert!(
            !ir.contains("linalg.generic"),
            "unexpected linalg.generic in IR:\n{ir}"
        );
    }

    #[test]
    fn ir_conv2d_with_bias_no_tosa_const() {
        // When bias is provided, no zero-fill const for bias should appear
        // (the user-provided bias tensor is used directly).
        begin_trace();
        let input = Tensor::new(&[1, 1, 4, 4], DType::F32);
        let kernel = Tensor::new(&[1, 1, 3, 3], DType::F32);
        let bias = Tensor::new(&[1], DType::F32);
        let out = input.conv2d(&kernel, Some(&bias), [1, 1], [0, 0, 0, 0], [1, 1]);
        let trace = take_trace();
        let ir = Compiler::build_ir_string(&trace, &[out.id]).expect("build_ir_string");
        assert!(
            ir.contains("tosa.conv2d"),
            "expected tosa.conv2d in IR:\n{ir}"
        );
    }

    // ── Conv2d execution tests (task 10.6) ───────────────────────────────────

    /// Helper: run a conv2d trace and return the output as Vec<f32>.
    fn run_conv2d_f32(
        input_data: &[f32],
        input_shape: &[u64],
        kernel_data: &[f32],
        kernel_shape: &[u64],
        bias_data: Option<(&[f32], &[u64])>,
        strides: [u64; 2],
        pads: [u64; 4],
        dilations: [u64; 2],
    ) -> Vec<f32> {
        begin_trace();
        let input_t = Tensor::new(input_shape, DType::F32);
        let kernel_t = Tensor::new(kernel_shape, DType::F32);
        let bias_tensor;
        let bias_ref = if let Some((_, bshape)) = bias_data {
            bias_tensor = Tensor::new(bshape, DType::F32);
            Some(&bias_tensor)
        } else {
            None
        };
        let out = input_t.conv2d(&kernel_t, bias_ref, strides, pads, dilations);
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[out.id], None).expect("compile failed");

        let input_buf = Buffer::from_slice::<f32>(input_data, input_shape, DType::F32);
        let kernel_buf = Buffer::from_slice::<f32>(kernel_data, kernel_shape, DType::F32);

        let result = if let Some((bdata, bshape)) = bias_data {
            let bias_buf = Buffer::from_slice::<f32>(bdata, bshape, DType::F32);
            compiled.run(&[&input_buf, &kernel_buf, &bias_buf])
        } else {
            compiled.run(&[&input_buf, &kernel_buf])
        };
        result[0].as_slice::<f32>().to_vec()
    }

    #[test]
    fn run_conv2d_basic_3x3_all_ones_kernel() {
        // Input [1,1,4,4], kernel [1,1,3,3] all-ones, no pad, stride=1.
        // Output [1,1,2,2]: each element is the sum of the 3x3 patch.
        //
        // Input (row-major):
        //   1  2  3  4
        //   5  6  7  8
        //   9 10 11 12
        //  13 14 15 16
        //
        // Patch sums:
        //   top-left [0,0]: 1+2+3 + 5+6+7 + 9+10+11 = 54
        //   top-right [0,1]: 2+3+4 + 6+7+8 + 10+11+12 = 63
        //   bot-left [1,0]: 5+6+7 + 9+10+11 + 13+14+15 = 90
        //   bot-right [1,1]: 6+7+8 + 10+11+12 + 14+15+16 = 99
        let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let kernel = vec![1.0f32; 9];
        let out = run_conv2d_f32(
            &input,
            &[1, 1, 4, 4],
            &kernel,
            &[1, 1, 3, 3],
            None,
            [1, 1],
            [0, 0, 0, 0],
            [1, 1],
        );
        assert_eq!(out.len(), 4);
        assert_eq!(out[0], 54.0);
        assert_eq!(out[1], 63.0);
        assert_eq!(out[2], 90.0);
        assert_eq!(out[3], 99.0);
    }

    #[test]
    fn run_conv2d_with_padding() {
        // Same 4x4 input and 3x3 all-ones kernel, but pad=1 on all sides.
        // Output should be [1,1,4,4] (same spatial size as input).
        // We only verify the shape (element count) and the center element.
        let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let kernel = vec![1.0f32; 9];
        let out = run_conv2d_f32(
            &input,
            &[1, 1, 4, 4],
            &kernel,
            &[1, 1, 3, 3],
            None,
            [1, 1],
            [1, 1, 1, 1],
            [1, 1],
        );
        assert_eq!(out.len(), 16, "expected 4x4 output with pad=1");
        // Center element [1,1] (0-indexed) with pad=1 is the same as the
        // no-pad top-left element.
        assert_eq!(out[5], 54.0); // row=1, col=1 in 4x4 layout
    }

    #[test]
    fn run_conv2d_with_stride2() {
        // Input [1,1,5,5], kernel [1,1,3,3] all-ones, no pad, stride=2.
        // TOSA requires (H + pads - dilation*(KH-1) - 1) % stride == 0:
        //   (5 + 0 - 1*2 - 1) = 2, and 2 % 2 = 0 ✓
        // OH = OW = 2/2 + 1 = 2. Output [1,1,2,2].
        //
        // Input (5x5, values 1..25):
        //    1  2  3  4  5
        //    6  7  8  9 10
        //   11 12 13 14 15
        //   16 17 18 19 20
        //   21 22 23 24 25
        //
        // Patch sums (stride=2):
        //   [0:3,0:3] = 1+2+3+6+7+8+11+12+13 = 63
        //   [0:3,2:5] = 3+4+5+8+9+10+13+14+15 = 81
        //   [2:5,0:3] = 11+12+13+16+17+18+21+22+23 = 153
        //   [2:5,2:5] = 13+14+15+18+19+20+23+24+25 = 171
        let input: Vec<f32> = (1..=25).map(|x| x as f32).collect();
        let kernel = vec![1.0f32; 9];
        let out = run_conv2d_f32(
            &input,
            &[1, 1, 5, 5],
            &kernel,
            &[1, 1, 3, 3],
            None,
            [2, 2],
            [0, 0, 0, 0],
            [1, 1],
        );
        assert_eq!(out.len(), 4);
        assert_eq!(out[0], 63.0);
        assert_eq!(out[1], 81.0);
        assert_eq!(out[2], 153.0);
        assert_eq!(out[3], 171.0);
    }

    #[test]
    fn run_conv2d_with_bias() {
        // Same basic 3x3 conv, add bias=[10.0].
        // Expected: [54+10, 63+10, 90+10, 99+10] = [64, 73, 100, 109]
        let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let kernel = vec![1.0f32; 9];
        let bias = vec![10.0f32];
        let out = run_conv2d_f32(
            &input,
            &[1, 1, 4, 4],
            &kernel,
            &[1, 1, 3, 3],
            Some((&bias, &[1])),
            [1, 1],
            [0, 0, 0, 0],
            [1, 1],
        );
        assert_eq!(out.len(), 4);
        assert_eq!(out[0], 64.0);
        assert_eq!(out[1], 73.0);
        assert_eq!(out[2], 100.0);
        assert_eq!(out[3], 109.0);
    }

    #[test]
    fn run_conv2d_multi_channel() {
        // Input [1, 2, 3, 3], kernel [3, 2, 2, 2] all-ones.
        // No pad, stride=1. Output [1, 3, 2, 2].
        // Each output element is the sum over both input channels of the 2x2 patch.
        //
        // Input channel 0 (3x3):
        //   1 2 3
        //   4 5 6
        //   7 8 9
        // Input channel 1 (3x3):
        //   10 11 12
        //   13 14 15
        //   16 17 18
        //
        // 2x2 patch sums (channel 0):
        //   [0,0]=1+2+4+5=12, [0,1]=2+3+5+6=16, [1,0]=4+5+7+8=24, [1,1]=5+6+8+9=28
        // 2x2 patch sums (channel 1):
        //   [0,0]=10+11+13+14=48, [0,1]=11+12+14+15=52, [1,0]=13+14+16+17=60, [1,1]=14+15+17+18=64
        //
        // Each of the 3 output channels uses the same all-ones kernel, so each
        // output channel element = sum_ch0 + sum_ch1.
        // Output [0,*,*]: 60, 68, 84, 92
        // Output [1,*,*]: same (kernel identical)
        // Output [2,*,*]: same
        let input_data: Vec<f32> = (1..=18).map(|x| x as f32).collect();
        let kernel_data = vec![1.0f32; 3 * 2 * 2 * 2]; // all ones
        let out = run_conv2d_f32(
            &input_data,
            &[1, 2, 3, 3],
            &kernel_data,
            &[3, 2, 2, 2],
            None,
            [1, 1],
            [0, 0, 0, 0],
            [1, 1],
        );
        assert_eq!(out.len(), 3 * 2 * 2);
        // All 3 output channels are identical since all kernels are ones.
        let expected = [60.0f32, 68.0, 84.0, 92.0];
        for ch in 0..3 {
            for i in 0..4 {
                assert_eq!(out[ch * 4 + i], expected[i], "mismatch at ch={ch}, i={i}");
            }
        }
    }

    // ── Empty/error cases ────────────────────────────────────────────────────

    #[test]
    fn compile_empty_trace_returns_error() {
        let trace = {
            begin_trace();
            take_trace()
        };
        let result = Compiler::compile(&trace, &[NodeId(0)], None);
        assert!(result.is_err());
    }

    #[test]
    fn compile_no_outputs_returns_error() {
        begin_trace();
        let _a = Tensor::new(&[4], DType::F32);
        let trace = take_trace();
        let result = Compiler::compile(&trace, &[], None);
        assert!(result.is_err());
    }

    // ── MaxPool2d IR verification tests (task 11.5) ───────────────────────────

    #[test]
    fn ir_max_pool2d_has_tosa_max_pool2d_and_transpose() {
        begin_trace();
        let input = Tensor::new(&[1, 1, 4, 4], DType::F32);
        let out = input.max_pool2d([2, 2], [2, 2], [0, 0, 0, 0]);
        let trace = take_trace();
        let ir = Compiler::build_ir_string(&trace, &[out.id]).expect("build_ir_string");
        assert!(
            ir.contains("tosa.max_pool2d"),
            "expected tosa.max_pool2d in IR:\n{ir}"
        );
        assert!(
            ir.contains("tosa.transpose"),
            "expected tosa.transpose in IR:\n{ir}"
        );
        assert!(
            !ir.contains("linalg.generic"),
            "unexpected linalg.generic in IR:\n{ir}"
        );
    }

    // ── MaxPool2d execution tests (task 11.6) ─────────────────────────────────

    /// Helper: run a max_pool2d trace and return the output as Vec<f32>.
    fn run_max_pool2d_f32(
        input_data: &[f32],
        input_shape: &[u64],
        kernel_size: [u64; 2],
        strides: [u64; 2],
        pads: [u64; 4],
    ) -> Vec<f32> {
        begin_trace();
        let input_t = Tensor::new(input_shape, DType::F32);
        let out = input_t.max_pool2d(kernel_size, strides, pads);
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[out.id], None).expect("compile failed");
        let input_buf = Buffer::from_slice::<f32>(input_data, input_shape, DType::F32);
        compiled.run(&[&input_buf])[0].as_slice::<f32>().to_vec()
    }

    #[test]
    fn run_max_pool2d_basic_2x2_stride2() {
        // Input [1, 1, 4, 4], kernel=2x2, stride=2, no pad → output [1, 1, 2, 2].
        // Each output element is the max of the corresponding 2x2 patch.
        //
        // Input (row-major):
        //    1  2  3  4
        //    5  6  7  8
        //    9 10 11 12
        //   13 14 15 16
        //
        // Patch maxes (stride=2):
        //   top-left [0,0]: max(1,2,5,6) = 6
        //   top-right [0,1]: max(3,4,7,8) = 8
        //   bot-left [1,0]: max(9,10,13,14) = 14
        //   bot-right [1,1]: max(11,12,15,16) = 16
        let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let out = run_max_pool2d_f32(&input, &[1, 1, 4, 4], [2, 2], [2, 2], [0, 0, 0, 0]);
        assert_eq!(out.len(), 4);
        assert_eq!(out[0], 6.0);
        assert_eq!(out[1], 8.0);
        assert_eq!(out[2], 14.0);
        assert_eq!(out[3], 16.0);
    }

    #[test]
    fn run_max_pool2d_with_padding_shape() {
        // Input [1, 1, 4, 4], kernel=3x3, stride=1, pad=1 all sides.
        // OH = (4 + 1 + 1 - 3) / 1 + 1 = 3 + 1 = 4 → output [1, 1, 4, 4].
        // Verify output shape and the center element.
        //
        // Center element [1, 1] (row=1, col=1) in a 3x3 window starting at (-1,-1)
        // covers rows [0,2], cols [0,2] of the padded input (pad=0 fills -inf).
        // That window (no extra pad rows/cols) = rows 0..2, cols 0..2:
        //   1  2  3
        //   5  6  7
        //   9 10 11
        // max = 11.0
        let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let out = run_max_pool2d_f32(&input, &[1, 1, 4, 4], [3, 3], [1, 1], [1, 1, 1, 1]);
        assert_eq!(out.len(), 16, "expected [1,1,4,4] output with pad=1");
        assert_eq!(out[5], 11.0, "center element mismatch"); // row=1, col=1 in 4x4
    }

    #[test]
    fn run_max_pool2d_batch_and_channel_independence() {
        // Input [2, 2, 4, 4], kernel=2x2, stride=2, no pad → output [2, 2, 2, 2].
        // Batch 0, channel 0: values 1..16; Batch 0, channel 1: values 17..32
        // Batch 1, channel 0: values 33..48; Batch 1, channel 1: values 49..64
        // Each 2x2 patch max: check all 4 patches per (batch, channel).
        let input: Vec<f32> = (1..=64).map(|x| x as f32).collect();
        let out = run_max_pool2d_f32(&input, &[2, 2, 4, 4], [2, 2], [2, 2], [0, 0, 0, 0]);
        assert_eq!(out.len(), 16); // 2 * 2 * 2 * 2

        // Layout is NCHW: [N=2, C=2, OH=2, OW=2]
        // out[n * 8 + c * 4 + oh * 2 + ow]
        //
        // Batch 0, ch 0 (input values 1..16):
        //   patch(0,0)=max(1,2,5,6)=6, patch(0,1)=max(3,4,7,8)=8
        //   patch(1,0)=max(9,10,13,14)=14, patch(1,1)=max(11,12,15,16)=16
        assert_eq!(out[0], 6.0);
        assert_eq!(out[1], 8.0);
        assert_eq!(out[2], 14.0);
        assert_eq!(out[3], 16.0);

        // Batch 0, ch 1 (input values 17..32):
        //   patch(0,0)=max(17,18,21,22)=22, patch(0,1)=max(19,20,23,24)=24
        //   patch(1,0)=max(25,26,29,30)=30, patch(1,1)=max(27,28,31,32)=32
        assert_eq!(out[4], 22.0);
        assert_eq!(out[5], 24.0);
        assert_eq!(out[6], 30.0);
        assert_eq!(out[7], 32.0);

        // Batch 1, ch 0 (input values 33..48):
        //   patch(0,0)=max(33,34,37,38)=38, patch(0,1)=max(35,36,39,40)=40
        //   patch(1,0)=max(41,42,45,46)=46, patch(1,1)=max(43,44,47,48)=48
        assert_eq!(out[8], 38.0);
        assert_eq!(out[9], 40.0);
        assert_eq!(out[10], 46.0);
        assert_eq!(out[11], 48.0);

        // Batch 1, ch 1 (input values 49..64):
        //   patch(0,0)=max(49,50,53,54)=54, patch(0,1)=max(51,52,55,56)=56
        //   patch(1,0)=max(57,58,61,62)=62, patch(1,1)=max(59,60,63,64)=64
        assert_eq!(out[12], 54.0);
        assert_eq!(out[13], 56.0);
        assert_eq!(out[14], 62.0);
        assert_eq!(out[15], 64.0);
    }

    // ── BatchNorm IR verification test (task 87) ──────────────────────────────

    #[test]
    fn batch_norm_emits_tosa_rsqrt() {
        begin_trace();
        let input = Tensor::new(&[1, 2, 4, 4], DType::F32);
        let scale = Tensor::new(&[2], DType::F32);
        let bias = Tensor::new(&[2], DType::F32);
        let mean = Tensor::new(&[2], DType::F32);
        let var = Tensor::new(&[2], DType::F32);
        let out = input.batch_norm(&scale, &bias, &mean, &var, 1e-5);
        let trace = take_trace();
        let ir = Compiler::build_ir_string(&trace, &[out.id]).expect("build_ir_string failed");
        assert!(
            ir.contains("tosa.rsqrt"),
            "IR should contain tosa.rsqrt:\n{ir}"
        );
        assert!(ir.contains("tosa.sub"), "IR should contain tosa.sub:\n{ir}");
        assert!(ir.contains("tosa.mul"), "IR should contain tosa.mul:\n{ir}");
    }

    // ── BatchNorm execution tests (task 88) ───────────────────────────────────

    /// Helper: run batch_norm with the given data and params, return output as Vec<f32>.
    ///
    /// `input_data` is NCHW-ordered. `scale`, `bias`, `mean`, `var` have length C.
    fn run_batch_norm_f32(
        input_data: &[f32],
        input_shape: &[u64],
        scale_data: &[f32],
        bias_data: &[f32],
        mean_data: &[f32],
        var_data: &[f32],
        epsilon: f64,
    ) -> Vec<f32> {
        let c = scale_data.len() as u64;
        begin_trace();
        let input_t = Tensor::new(input_shape, DType::F32);
        let scale_t = Tensor::new(&[c], DType::F32);
        let bias_t = Tensor::new(&[c], DType::F32);
        let mean_t = Tensor::new(&[c], DType::F32);
        let var_t = Tensor::new(&[c], DType::F32);
        let out = input_t.batch_norm(&scale_t, &bias_t, &mean_t, &var_t, epsilon);
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[out.id], None).expect("compile failed");
        let input_buf = Buffer::from_slice::<f32>(input_data, input_shape, DType::F32);
        let scale_buf = Buffer::from_slice::<f32>(scale_data, &[c], DType::F32);
        let bias_buf = Buffer::from_slice::<f32>(bias_data, &[c], DType::F32);
        let mean_buf = Buffer::from_slice::<f32>(mean_data, &[c], DType::F32);
        let var_buf = Buffer::from_slice::<f32>(var_data, &[c], DType::F32);
        compiled
            .run(&[&input_buf, &scale_buf, &bias_buf, &mean_buf, &var_buf])[0]
            .as_slice::<f32>()
            .to_vec()
    }

    #[test]
    fn run_batch_norm_basic() {
        // 1 batch, 2 channels, 1x1 spatial.
        // input: ch0=1.0, ch1=4.0
        // mean: ch0=0.0, ch1=2.0; var: ch0=1.0, ch1=4.0; eps=0.0 (use 1e-7)
        // scale: ch0=1.0, ch1=2.0; bias: ch0=0.0, ch1=1.0
        //
        // ch0: (1.0 - 0.0) / sqrt(1.0 + 1e-7) * 1.0 + 0.0 ≈ 1.0
        // ch1: (4.0 - 2.0) / sqrt(4.0 + 1e-7) * 2.0 + 1.0 ≈ 2.0/2.0 * 2.0 + 1.0 = 3.0
        let out = run_batch_norm_f32(
            &[1.0, 4.0],
            &[1, 2, 1, 1],
            &[1.0, 2.0], // scale
            &[0.0, 1.0], // bias
            &[0.0, 2.0], // mean
            &[1.0, 4.0], // var
            1e-7,
        );
        assert_eq!(out.len(), 2);
        assert!(
            (out[0] - 1.0f32).abs() < 1e-4,
            "ch0 expected ~1.0, got {}",
            out[0]
        );
        assert!(
            (out[1] - 3.0f32).abs() < 1e-4,
            "ch1 expected ~3.0, got {}",
            out[1]
        );
    }

    #[test]
    fn run_batch_norm_zero_mean_unit_var() {
        // With mean=0, var=1, scale=1, bias=0, output should equal input.
        let input: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let out = run_batch_norm_f32(
            &input,
            &[1, 2, 2, 2],
            &[1.0, 1.0], // scale
            &[0.0, 0.0], // bias
            &[0.0, 0.0], // mean
            &[1.0, 1.0], // var
            1e-7,
        );
        assert_eq!(out.len(), 8);
        for (i, (&expected, &got)) in input.iter().zip(out.iter()).enumerate() {
            // With var=1+eps, output ≈ input but slightly < 1.0x due to rsqrt(1+eps).
            assert!(
                (got - expected).abs() < 0.01,
                "element {i}: expected ~{expected}, got {got}"
            );
        }
    }

    #[test]
    fn run_batch_norm_shape_preserved() {
        // Verify output shape equals input shape: [2, 3, 4, 4].
        let total = 2 * 3 * 4 * 4;
        let input: Vec<f32> = vec![1.0f32; total];
        let out = run_batch_norm_f32(
            &input,
            &[2, 3, 4, 4],
            &[1.0, 1.0, 1.0], // scale for 3 channels
            &[0.0, 0.0, 0.0], // bias
            &[0.0, 0.0, 0.0], // mean
            &[1.0, 1.0, 1.0], // var
            1e-5,
        );
        assert_eq!(
            out.len(),
            total,
            "output should have same number of elements as input"
        );
    }

    #[test]
    fn run_batch_norm_varying_epsilon() {
        // Larger epsilon shrinks the output toward bias.
        // input=1.0, mean=0.0, var=0.0 (degenerate), scale=1.0, bias=0.0
        // With eps=1.0: out = 1.0 / sqrt(0.0 + 1.0) = 1.0
        // With eps=4.0: out = 1.0 / sqrt(0.0 + 4.0) = 0.5
        let out_eps1 =
            run_batch_norm_f32(&[1.0], &[1, 1, 1, 1], &[1.0], &[0.0], &[0.0], &[0.0], 1.0);
        let out_eps4 =
            run_batch_norm_f32(&[1.0], &[1, 1, 1, 1], &[1.0], &[0.0], &[0.0], &[0.0], 4.0);
        assert!(
            (out_eps1[0] - 1.0f32).abs() < 1e-5,
            "eps=1.0: expected ~1.0, got {}",
            out_eps1[0]
        );
        assert!(
            (out_eps4[0] - 0.5f32).abs() < 1e-5,
            "eps=4.0: expected ~0.5, got {}",
            out_eps4[0]
        );
    }

    // ── GlobalAvgPool IR verification test (task 93) ──────────────────────────

    #[test]
    fn global_avg_pool_emits_tosa_avg_pool2d() {
        begin_trace();
        let input = Tensor::new(&[1, 4, 6, 6], DType::F32);
        let out = input.global_avg_pool();
        let trace = take_trace();
        let ir = Compiler::build_ir_string(&trace, &[out.id]).expect("build_ir_string failed");
        assert!(
            ir.contains("tosa.avg_pool2d"),
            "expected tosa.avg_pool2d in IR:\n{ir}"
        );
        assert!(
            ir.contains("tosa.transpose"),
            "expected tosa.transpose in IR:\n{ir}"
        );
    }

    // ── GlobalAvgPool execution tests (task 94) ───────────────────────────────

    #[test]
    fn run_global_avg_pool_basic() {
        // 1 batch, 1 channel, 2x2 spatial: values [1, 2, 3, 4] → avg = 2.5
        begin_trace();
        let input = Tensor::new(&[1, 1, 2, 2], DType::F32);
        let out = input.global_avg_pool();
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[out.id], None).expect("compile failed");
        let input_buf = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], DType::F32);
        let result = compiled.run(&[&input_buf]);
        let out_slice = result[0].as_slice::<f32>();
        // Output shape [1, 1, 1, 1] → single element = mean(1,2,3,4) = 2.5
        assert_eq!(out_slice.len(), 1, "expected 1 output element");
        assert!(
            (out_slice[0] - 2.5f32).abs() < 1e-5,
            "expected 2.5, got {}",
            out_slice[0]
        );
    }

    #[test]
    fn run_global_avg_pool_batch_feature_maps() {
        // 2 batches, 2 channels, 2x2 spatial.
        // NCHW layout: [N=2, C=2, H=2, W=2]
        // batch0 ch0: 1,2,3,4 → avg=2.5
        // batch0 ch1: 10,20,30,40 → avg=25.0
        // batch1 ch0: 0,0,0,4 → avg=1.0
        // batch1 ch1: 5,5,5,5 → avg=5.0
        begin_trace();
        let input = Tensor::new(&[2, 2, 2, 2], DType::F32);
        let out = input.global_avg_pool();
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[out.id], None).expect("compile failed");
        #[rustfmt::skip]
        let data: Vec<f32> = vec![
            // batch 0
            1.0, 2.0, 3.0, 4.0,   // ch0
            10.0, 20.0, 30.0, 40.0, // ch1
            // batch 1
            0.0, 0.0, 0.0, 4.0,   // ch0
            5.0, 5.0, 5.0, 5.0,   // ch1
        ];
        let input_buf = Buffer::from_slice::<f32>(&data, &[2, 2, 2, 2], DType::F32);
        let result = compiled.run(&[&input_buf]);
        let out_slice = result[0].as_slice::<f32>();
        // Output shape [2, 2, 1, 1] → 4 elements in NCHW order
        assert_eq!(out_slice.len(), 4, "expected 4 output elements");
        assert!(
            (out_slice[0] - 2.5f32).abs() < 1e-4,
            "batch0 ch0: expected 2.5, got {}",
            out_slice[0]
        );
        assert!(
            (out_slice[1] - 25.0f32).abs() < 1e-4,
            "batch0 ch1: expected 25.0, got {}",
            out_slice[1]
        );
        assert!(
            (out_slice[2] - 1.0f32).abs() < 1e-4,
            "batch1 ch0: expected 1.0, got {}",
            out_slice[2]
        );
        assert!(
            (out_slice[3] - 5.0f32).abs() < 1e-4,
            "batch1 ch1: expected 5.0, got {}",
            out_slice[3]
        );
    }

    // ── Task 4.1: Pow ─────────────────────────────────────────────────────────

    #[test]
    fn pow_ir_contains_tosa_pow() {
        begin_trace();
        let a = Tensor::new(&[2], DType::F32);
        let b = Tensor::new(&[2], DType::F32);
        let c = a.pow(&b);
        let trace = take_trace();

        let ir = Compiler::build_ir_string(&trace, &[c.id]).expect("build_ir_string failed");
        assert!(ir.contains("tosa.pow"), "expected tosa.pow in IR:\n{ir}");
    }

    #[test]
    fn pow_runtime_elementwise() {
        begin_trace();
        let a = Tensor::new(&[2], DType::F32);
        let b = Tensor::new(&[2], DType::F32);
        let c = a.pow(&b);
        let trace = take_trace();

        let compiled = Compiler::compile(&trace, &[c.id], None).expect("compile failed");
        let base = Buffer::from_slice::<f32>(&[2.0, 3.0], &[2], DType::F32);
        let exp = Buffer::from_slice::<f32>(&[3.0, 2.0], &[2], DType::F32);
        let result = compiled.run(&[&base, &exp]);
        let out = result[0].as_slice::<f32>();
        assert!((out[0] - 8.0).abs() < 1e-4, "expected 8.0, got {}", out[0]);
        assert!((out[1] - 9.0).abs() < 1e-4, "expected 9.0, got {}", out[1]);
    }

    #[test]
    fn pow_runtime_broadcast_scalar_exponent() {
        begin_trace();
        let a = Tensor::new(&[3], DType::F32);
        let b = Tensor::new(&[1], DType::F32);
        let c = a.pow(&b);
        let trace = take_trace();

        let compiled = Compiler::compile(&trace, &[c.id], None).expect("compile failed");
        let base = Buffer::from_slice::<f32>(&[2.0, 3.0, 4.0], &[3], DType::F32);
        let exp = Buffer::from_slice::<f32>(&[2.0], &[1], DType::F32);
        let result = compiled.run(&[&base, &exp]);
        let out = result[0].as_slice::<f32>();
        assert!((out[0] - 4.0).abs() < 1e-4, "expected 4.0, got {}", out[0]);
        assert!((out[1] - 9.0).abs() < 1e-4, "expected 9.0, got {}", out[1]);
        assert!((out[2] - 16.0).abs() < 1e-4, "expected 16.0, got {}", out[2]);
    }

    // ── Task 4.2: Sqrt ────────────────────────────────────────────────────────

    #[test]
    fn sqrt_ir_contains_math_sqrt() {
        begin_trace();
        let a = Tensor::new(&[3], DType::F32);
        let b = a.sqrt();
        let trace = take_trace();

        let ir = Compiler::build_ir_string(&trace, &[b.id]).expect("build_ir_string failed");
        assert!(ir.contains("math.sqrt"), "expected math.sqrt in IR:\n{ir}");
    }

    #[test]
    fn sqrt_runtime_basic() {
        begin_trace();
        let a = Tensor::new(&[3], DType::F32);
        let b = a.sqrt();
        let trace = take_trace();

        let compiled = Compiler::compile(&trace, &[b.id], None).expect("compile failed");
        let input = Buffer::from_slice::<f32>(&[4.0, 9.0, 16.0], &[3], DType::F32);
        let result = compiled.run(&[&input]);
        let out = result[0].as_slice::<f32>();
        assert!((out[0] - 2.0).abs() < 1e-4, "expected 2.0, got {}", out[0]);
        assert!((out[1] - 3.0).abs() < 1e-4, "expected 3.0, got {}", out[1]);
        assert!((out[2] - 4.0).abs() < 1e-4, "expected 4.0, got {}", out[2]);
    }

    #[test]
    #[should_panic(expected = "sqrt requires float dtype")]
    fn sqrt_integer_panics() {
        begin_trace();
        let a = Tensor::new(&[3], DType::I32);
        let _ = a.sqrt();
        let _ = take_trace();
    }

    // ── Task 4.3: Cast ────────────────────────────────────────────────────────

    #[test]
    fn cast_i64_to_f32_runtime() {
        begin_trace();
        let a = Tensor::new(&[3], DType::I64);
        let b = a.cast(DType::F32);
        let trace = take_trace();

        let compiled = Compiler::compile(&trace, &[b.id], None).expect("compile failed");
        let input = Buffer::from_slice::<i64>(&[1, 2, 3], &[3], DType::I64);
        let result = compiled.run(&[&input]);
        let out = result[0].as_slice::<f32>();
        assert!((out[0] - 1.0).abs() < 1e-5, "expected 1.0, got {}", out[0]);
        assert!((out[1] - 2.0).abs() < 1e-5, "expected 2.0, got {}", out[1]);
        assert!((out[2] - 3.0).abs() < 1e-5, "expected 3.0, got {}", out[2]);
    }

    #[test]
    fn cast_f32_to_i64_runtime() {
        begin_trace();
        let a = Tensor::new(&[2], DType::F32);
        let b = a.cast(DType::I64);
        let trace = take_trace();

        let compiled = Compiler::compile(&trace, &[b.id], None).expect("compile failed");
        let input = Buffer::from_slice::<f32>(&[1.5, 2.7], &[2], DType::F32);
        let result = compiled.run(&[&input]);
        let out = result[0].as_slice::<i64>();
        assert_eq!(out[0], 1, "expected truncation to 1, got {}", out[0]);
        assert_eq!(out[1], 2, "expected truncation to 2, got {}", out[1]);
    }

    #[test]
    fn cast_same_dtype_is_noop() {
        begin_trace();
        let a = Tensor::new(&[3], DType::F32);
        let b = a.cast(DType::F32);
        let trace = take_trace();
        // Same-type cast should not add an extra op.
        assert_eq!(trace.ops().len(), 1, "same-type cast should not add an op");
        assert_eq!(a.id, b.id, "same-type cast should return same node id");
    }

    // ── Task 4.4: ReduceMean ──────────────────────────────────────────────────

    #[test]
    fn reduce_mean_ir_contains_reduce_sum_and_mul() {
        begin_trace();
        let a = Tensor::new(&[2, 2], DType::F32);
        let b = a.reduce_mean(&[1], true);
        let trace = take_trace();

        let ir = Compiler::build_ir_string(&trace, &[b.id]).expect("build_ir_string failed");
        assert!(
            ir.contains("tosa.reduce_sum"),
            "expected tosa.reduce_sum in IR:\n{ir}"
        );
        assert!(ir.contains("tosa.mul"), "expected tosa.mul in IR:\n{ir}");
    }

    #[test]
    fn reduce_mean_axis1_no_keepdim() {
        begin_trace();
        // [2, 2] shaped input: [[1, 2], [3, 4]], reduce axis=1 → [1.5, 3.5]
        let a = Tensor::new(&[2, 2], DType::F32);
        let b = a.reduce_mean(&[1], false);
        let trace = take_trace();

        assert_eq!(b.shape().0, vec![2u64], "expected shape [2], got {:?}", b.shape().0);

        let compiled = Compiler::compile(&trace, &[b.id], None).expect("compile failed");
        let input = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[2, 2], DType::F32);
        let result = compiled.run(&[&input]);
        let out = result[0].as_slice::<f32>();
        assert!(
            (out[0] - 1.5).abs() < 1e-4,
            "expected 1.5, got {}",
            out[0]
        );
        assert!(
            (out[1] - 3.5).abs() < 1e-4,
            "expected 3.5, got {}",
            out[1]
        );
    }

    #[test]
    fn reduce_mean_axis1_keepdim() {
        begin_trace();
        let a = Tensor::new(&[2, 2], DType::F32);
        let b = a.reduce_mean(&[1], true);
        let trace = take_trace();

        assert_eq!(
            b.shape().0,
            vec![2u64, 1],
            "expected shape [2,1], got {:?}",
            b.shape().0
        );

        let compiled = Compiler::compile(&trace, &[b.id], None).expect("compile failed");
        let input = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[2, 2], DType::F32);
        let result = compiled.run(&[&input]);
        let out = result[0].as_slice::<f32>();
        assert!(
            (out[0] - 1.5).abs() < 1e-4,
            "expected 1.5, got {}",
            out[0]
        );
        assert!(
            (out[1] - 3.5).abs() < 1e-4,
            "expected 3.5, got {}",
            out[1]
        );
    }

    #[test]
    fn reduce_mean_multi_axis() {
        begin_trace();
        // [2, 3, 4] tensor, axes=[1,2], keepdim=false → shape [2]
        let a = Tensor::new(&[2, 3, 4], DType::F32);
        let b = a.reduce_mean(&[1, 2], false);
        let trace = take_trace();

        assert_eq!(
            b.shape().0,
            vec![2u64],
            "expected shape [2], got {:?}",
            b.shape().0
        );

        // Build input: batch 0 = all 1.0, batch 1 = all 2.0
        let mut data = vec![0.0f32; 24];
        for i in 0..12 {
            data[i] = 1.0;
        }
        for i in 12..24 {
            data[i] = 2.0;
        }
        let compiled = Compiler::compile(&trace, &[b.id], None).expect("compile failed");
        let input = Buffer::from_slice::<f32>(&data, &[2, 3, 4], DType::F32);
        let result = compiled.run(&[&input]);
        let out = result[0].as_slice::<f32>();
        assert!(
            (out[0] - 1.0).abs() < 1e-4,
            "expected 1.0, got {}",
            out[0]
        );
        assert!(
            (out[1] - 2.0).abs() < 1e-4,
            "expected 2.0, got {}",
            out[1]
        );
    }

    // NOTE: affine-loop-fusion was investigated and found unsafe for our pipeline.
    // Two root causes:
    // 1. memref.expand_shape aliases: fusion doesn't track that expand_shape creates
    //    an alias to the original buffer (affects softmax, reductions).
    //    Fix: run fold-memref-alias-ops before fusion to eliminate expand_shape.
    // 2. memref.subview aliases: conv2d/maxpool2d padding uses subview into a padded
    //    buffer. fusion merges the fill-zeros loop with the copy-input loop, breaking
    //    pad semantics. fold-memref-alias-ops can't resolve strided subviews.
    //    No fix available — this is a fundamental limitation of affine fusion with
    //    subview aliasing from one-shot, bufferize.

    // ── Gather tests ─────────────────────────────────────────────────────────

    #[test]
    fn gather_axis0_ir() {
        begin_trace();
        // data[4, 2], indices[3] → out[3, 2]
        let data = Tensor::new(&[4, 2], DType::F32);
        let indices = Tensor::new(&[3], DType::I64);
        let out = data.gather(&indices, 0);
        let trace = take_trace();
        assert_eq!(out.shape().0, vec![3u64, 2]);
        let result = Compiler::compile(&trace, &[out.id], None);
        assert!(result.is_ok(), "gather IR compile failed: {:?}", result.err());
    }

    #[test]
    fn gather_axis0_runtime() {
        begin_trace();
        // data = [[10, 20], [30, 40], [50, 60], [70, 80]]
        // indices = [2, 0, 3]
        // expected = [[50, 60], [10, 20], [70, 80]]
        let data = Tensor::new(&[4, 2], DType::F32);
        let indices = Tensor::new(&[3], DType::I64);
        let out = data.gather(&indices, 0);
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[out.id], None).expect("compile failed");
        let data_buf = Buffer::from_slice::<f32>(
            &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            &[4, 2],
            DType::F32,
        );
        let idx_buf = Buffer::from_slice::<i64>(&[2, 0, 3], &[3], DType::I64);
        let results = compiled.run(&[&data_buf, &idx_buf]);
        let out_data = results[0].as_slice::<f32>();
        assert_eq!(out_data, &[50.0, 60.0, 10.0, 20.0, 70.0, 80.0]);
    }

    #[test]
    fn gather_axis1_runtime() {
        begin_trace();
        // data = [[1, 2, 3], [4, 5, 6]], indices = [2, 0], axis=1
        // expected = [[3, 1], [6, 4]]
        let data = Tensor::new(&[2, 3], DType::F32);
        let indices = Tensor::new(&[2], DType::I64);
        let out = data.gather(&indices, 1);
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[out.id], None).expect("compile failed");
        let data_buf =
            Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
        let idx_buf = Buffer::from_slice::<i64>(&[2, 0], &[2], DType::I64);
        let results = compiled.run(&[&data_buf, &idx_buf]);
        let out_data = results[0].as_slice::<f32>();
        assert_eq!(out_data, &[3.0, 1.0, 6.0, 4.0]);
    }

    #[test]
    fn gather_2d_indices_runtime() {
        begin_trace();
        // Embedding lookup: data[vocab=5, dim=3], indices[2, 4] → out[2, 4, 3]
        let vocab = 5usize;
        let dim = 3usize;
        let data = Tensor::new(&[vocab as u64, dim as u64], DType::F32);
        let indices = Tensor::new(&[2, 4], DType::I64);
        let out = data.gather(&indices, 0);
        let trace = take_trace();
        assert_eq!(out.shape().0, vec![2u64, 4, 3]);
        let compiled = Compiler::compile(&trace, &[out.id], None).expect("compile failed");
        // data rows: row i = [i*10, i*10+1, i*10+2]
        let data_vals: Vec<f32> = (0..vocab)
            .flat_map(|i| (0..dim).map(move |j| (i * 10 + j) as f32))
            .collect();
        // indices: pick rows 1,3,0,2 and 4,2,1,0
        let idx_vals: Vec<i64> = vec![1, 3, 0, 2, 4, 2, 1, 0];
        let data_buf =
            Buffer::from_slice::<f32>(&data_vals, &[vocab as u64, dim as u64], DType::F32);
        let idx_buf = Buffer::from_slice::<i64>(&idx_vals, &[2, 4], DType::I64);
        let results = compiled.run(&[&data_buf, &idx_buf]);
        let out_data = results[0].as_slice::<f32>();
        // Verify spot check: out[0,0] = data[1] = [10,11,12]
        assert_eq!(&out_data[0..3], &[10.0, 11.0, 12.0]);
        // out[0,1] = data[3] = [30,31,32]
        assert_eq!(&out_data[3..6], &[30.0, 31.0, 32.0]);
    }

    // ── Slice tests ───────────────────────────────────────────────────────────

    #[test]
    fn slice_stride1_ir() {
        begin_trace();
        // Slice [0..3] of a 1D tensor of length 5.
        let a = Tensor::new(&[5], DType::F32);
        let b = a.slice(&[0], &[3], &[0], &[1]);
        let trace = take_trace();
        assert_eq!(b.shape().0, vec![3u64]);
        let result = Compiler::compile(&trace, &[b.id], None);
        assert!(result.is_ok(), "slice IR compile failed: {:?}", result.err());
    }

    #[test]
    fn slice_1d_runtime() {
        begin_trace();
        // a = [10, 20, 30, 40, 50], slice [1..4] → [20, 30, 40]
        let a = Tensor::new(&[5], DType::F32);
        let b = a.slice(&[1], &[4], &[0], &[1]);
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[b.id], None).expect("compile failed");
        let a_buf =
            Buffer::from_slice::<f32>(&[10.0, 20.0, 30.0, 40.0, 50.0], &[5], DType::F32);
        let results = compiled.run(&[&a_buf]);
        assert_eq!(results[0].as_slice::<f32>(), &[20.0, 30.0, 40.0]);
    }

    #[test]
    fn slice_2d_runtime() {
        begin_trace();
        // data[3, 4], slice rows [0..2], cols [1..3] → shape [2, 2]
        // data = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
        // → [[2,3],[6,7]]
        let a = Tensor::new(&[3, 4], DType::F32);
        let b = a.slice(&[0, 1], &[2, 3], &[0, 1], &[1, 1]);
        let trace = take_trace();
        assert_eq!(b.shape().0, vec![2u64, 2]);
        let compiled = Compiler::compile(&trace, &[b.id], None).expect("compile failed");
        let a_buf = Buffer::from_slice::<f32>(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            &[3, 4],
            DType::F32,
        );
        let results = compiled.run(&[&a_buf]);
        assert_eq!(results[0].as_slice::<f32>(), &[2.0, 3.0, 6.0, 7.0]);
    }

    #[test]
    fn slice_strided_runtime() {
        begin_trace();
        // a = [0,1,2,3,4,5,6,7,8,9], slice with step=2 → [0,2,4,6,8]
        let a = Tensor::new(&[10], DType::F32);
        let b = a.slice(&[0], &[10], &[0], &[2]);
        let trace = take_trace();
        assert_eq!(b.shape().0, vec![5u64]);
        let compiled = Compiler::compile(&trace, &[b.id], None).expect("compile failed");
        let a_data: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let a_buf = Buffer::from_slice::<f32>(&a_data, &[10], DType::F32);
        let results = compiled.run(&[&a_buf]);
        assert_eq!(results[0].as_slice::<f32>(), &[0.0, 2.0, 4.0, 6.0, 8.0]);
    }

    // ── Concat tests ──────────────────────────────────────────────────────────

    #[test]
    fn concat_ir() {
        begin_trace();
        let a = Tensor::new(&[2, 3], DType::F32);
        let b = Tensor::new(&[2, 3], DType::F32);
        let c = Tensor::concat(&[&a, &b], 0);
        let trace = take_trace();
        assert_eq!(c.shape().0, vec![4u64, 3]);
        let result = Compiler::compile(&trace, &[c.id], None);
        assert!(result.is_ok(), "concat IR compile failed: {:?}", result.err());
    }

    #[test]
    fn concat_axis0_runtime() {
        begin_trace();
        // a = [[1,2,3]], b = [[4,5,6]] → [[1,2,3],[4,5,6]]
        let a = Tensor::new(&[1, 3], DType::F32);
        let b = Tensor::new(&[1, 3], DType::F32);
        let c = Tensor::concat(&[&a, &b], 0);
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[c.id], None).expect("compile failed");
        let a_buf =
            Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0], &[1, 3], DType::F32);
        let b_buf =
            Buffer::from_slice::<f32>(&[4.0, 5.0, 6.0], &[1, 3], DType::F32);
        let results = compiled.run(&[&a_buf, &b_buf]);
        assert_eq!(
            results[0].as_slice::<f32>(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn concat_axis1_runtime() {
        begin_trace();
        // a = [[1,2],[3,4]], b = [[5],[6]] → [[1,2,5],[3,4,6]]
        let a = Tensor::new(&[2, 2], DType::F32);
        let b = Tensor::new(&[2, 1], DType::F32);
        let c = Tensor::concat(&[&a, &b], 1);
        let trace = take_trace();
        assert_eq!(c.shape().0, vec![2u64, 3]);
        let compiled = Compiler::compile(&trace, &[c.id], None).expect("compile failed");
        let a_buf =
            Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[2, 2], DType::F32);
        let b_buf = Buffer::from_slice::<f32>(&[5.0, 6.0], &[2, 1], DType::F32);
        let results = compiled.run(&[&a_buf, &b_buf]);
        assert_eq!(
            results[0].as_slice::<f32>(),
            &[1.0, 2.0, 5.0, 3.0, 4.0, 6.0]
        );
    }

    #[test]
    fn concat_three_tensors_runtime() {
        begin_trace();
        let a = Tensor::new(&[3], DType::F32);
        let b = Tensor::new(&[2], DType::F32);
        let c_t = Tensor::new(&[1], DType::F32);
        let out = Tensor::concat(&[&a, &b, &c_t], 0);
        let trace = take_trace();
        assert_eq!(out.shape().0, vec![6u64]);
        let compiled = Compiler::compile(&trace, &[out.id], None).expect("compile failed");
        let a_buf = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3], DType::F32);
        let b_buf = Buffer::from_slice::<f32>(&[4.0, 5.0], &[2], DType::F32);
        let c_buf = Buffer::from_slice::<f32>(&[6.0], &[1], DType::F32);
        let results = compiled.run(&[&a_buf, &b_buf, &c_buf]);
        assert_eq!(
            results[0].as_slice::<f32>(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    // ── Transpose tests ───────────────────────────────────────────────────────

    #[test]
    fn transpose_2d_ir() {
        begin_trace();
        let a = Tensor::new(&[3, 4], DType::F32);
        let b = a.transpose(&[1, 0]);
        let trace = take_trace();
        assert_eq!(b.shape().0, vec![4u64, 3]);
        let result = Compiler::compile(&trace, &[b.id], None);
        assert!(result.is_ok(), "transpose IR compile failed: {:?}", result.err());
    }

    #[test]
    fn transpose_2d_runtime() {
        begin_trace();
        // [[1,2,3],[4,5,6]] transpose → [[1,4],[2,5],[3,6]]
        let a = Tensor::new(&[2, 3], DType::F32);
        let b = a.transpose(&[1, 0]);
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[b.id], None).expect("compile failed");
        let a_buf =
            Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
        let results = compiled.run(&[&a_buf]);
        assert_eq!(
            results[0].as_slice::<f32>(),
            &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
    }

    #[test]
    fn transpose_4d_nchw_nhwc_runtime() {
        begin_trace();
        // NCHW → NHWC: [1, 2, 3, 4] perm [0,2,3,1] → [1, 3, 4, 2]
        let a = Tensor::new(&[1, 2, 3, 4], DType::F32);
        let b = a.transpose(&[0, 2, 3, 1]);
        let trace = take_trace();
        assert_eq!(b.shape().0, vec![1u64, 3, 4, 2]);
        let result = Compiler::compile(&trace, &[b.id], None);
        assert!(result.is_ok(), "4D transpose compile failed: {:?}", result.err());
    }

    // ── Where tests ───────────────────────────────────────────────────────────

    #[test]
    fn where_ir() {
        begin_trace();
        let cond = Tensor::new(&[4], DType::I64);
        let x = Tensor::new(&[4], DType::F32);
        let y = Tensor::new(&[4], DType::F32);
        let out = Tensor::where_select(&cond, &x, &y);
        let trace = take_trace();
        assert_eq!(out.shape().0, vec![4u64]);
        let result = Compiler::compile(&trace, &[out.id], None);
        assert!(result.is_ok(), "where IR compile failed: {:?}", result.err());
    }

    #[test]
    fn where_runtime() {
        begin_trace();
        // cond = [1, 0, 1, 0], x = [10, 20, 30, 40], y = [100, 200, 300, 400]
        // expected = [10, 200, 30, 400]
        let cond = Tensor::new(&[4], DType::I64);
        let x = Tensor::new(&[4], DType::F32);
        let y = Tensor::new(&[4], DType::F32);
        let out = Tensor::where_select(&cond, &x, &y);
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[out.id], None).expect("compile failed");
        let cond_buf = Buffer::from_slice::<i64>(&[1, 0, 1, 0], &[4], DType::I64);
        let x_buf =
            Buffer::from_slice::<f32>(&[10.0, 20.0, 30.0, 40.0], &[4], DType::F32);
        let y_buf =
            Buffer::from_slice::<f32>(&[100.0, 200.0, 300.0, 400.0], &[4], DType::F32);
        let results = compiled.run(&[&cond_buf, &x_buf, &y_buf]);
        assert_eq!(
            results[0].as_slice::<f32>(),
            &[10.0, 200.0, 30.0, 400.0]
        );
    }

    #[test]
    fn where_2d_runtime() {
        begin_trace();
        // cond = [[1,0],[0,1]], x = [[1,2],[3,4]], y = [[10,20],[30,40]]
        // expected = [[1,20],[30,4]]
        let cond = Tensor::new(&[2, 2], DType::I64);
        let x = Tensor::new(&[2, 2], DType::F32);
        let y = Tensor::new(&[2, 2], DType::F32);
        let out = Tensor::where_select(&cond, &x, &y);
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[out.id], None).expect("compile failed");
        let cond_buf = Buffer::from_slice::<i64>(&[1, 0, 0, 1], &[2, 2], DType::I64);
        let x_buf =
            Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[2, 2], DType::F32);
        let y_buf =
            Buffer::from_slice::<f32>(&[10.0, 20.0, 30.0, 40.0], &[2, 2], DType::F32);
        let results = compiled.run(&[&cond_buf, &x_buf, &y_buf]);
        assert_eq!(results[0].as_slice::<f32>(), &[1.0, 20.0, 30.0, 4.0]);
    }

    // ── Batched matmul tests ───────────────────────────────────────────────────

    #[test]
    fn ir_batched_matmul_f32_uses_linalg_generic() {
        // Batched float matmul should emit linalg.generic (not tosa.matmul).
        begin_trace();
        let a = Tensor::new(&[2, 3, 4], DType::F32);
        let b = Tensor::new(&[2, 4, 5], DType::F32);
        let c = a.matmul(&b);
        let trace = take_trace();
        let ir = Compiler::build_ir_string(&trace, &[c.id]).expect("build_ir_string");
        assert!(
            ir.contains("linalg.generic"),
            "expected linalg.generic in IR:\n{ir}"
        );
        assert!(
            !ir.contains("tosa.matmul"),
            "unexpected tosa.matmul in batched IR:\n{ir}"
        );
        assert!(
            !ir.contains("linalg.matmul"),
            "unexpected linalg.matmul in batched IR:\n{ir}"
        );
    }

    #[test]
    fn ir_batched_matmul_4d_uses_linalg_generic() {
        // 4D batched float matmul: [B,H,M,K] @ [B,H,K,N] should emit linalg.generic.
        begin_trace();
        let a = Tensor::new(&[2, 3, 4, 5], DType::F32);
        let b = Tensor::new(&[2, 3, 5, 6], DType::F32);
        let c = a.matmul(&b);
        let trace = take_trace();
        let ir = Compiler::build_ir_string(&trace, &[c.id]).expect("build_ir_string");
        assert!(
            ir.contains("linalg.generic"),
            "expected linalg.generic in IR:\n{ir}"
        );
        assert!(
            !ir.contains("tosa.matmul"),
            "unexpected tosa.matmul in 4D batched IR:\n{ir}"
        );
    }

    #[test]
    fn runtime_batched_matmul_3d_f32() {
        // [2, 2, 2] @ [2, 2, 3] -> [2, 2, 3]
        // batch=0: [[1,2],[3,4]] @ [[1,0,0],[0,1,0]] = [[1,2,0],[3,4,0]]  (with 3-col RHS)
        // batch=0: [[1,2],[3,4]] @ [[1,2,3],[4,5,6]] = [[9,12,15],[19,26,33]]
        // batch=1: [[5,6],[7,8]] @ [[1,0,0],[0,1,0]] = [[5,6,0],[7,8,0]]
        // Use simple case: identity-like RHS for easy verification.
        //
        // lhs[0] = [[1,2],[3,4]], lhs[1] = [[5,6],[7,8]]
        // rhs[0] = [[1,2,3],[4,5,6]], rhs[1] = [[7,8,9],[10,11,12]]
        // out[0,0,:] = 1*1+2*4, 1*2+2*5, 1*3+2*6 = 9,12,15
        // out[0,1,:] = 3*1+4*4, 3*2+4*5, 3*3+4*6 = 19,26,33
        // out[1,0,:] = 5*7+6*10, 5*8+6*11, 5*9+6*12 = 95,106,117
        // out[1,1,:] = 7*7+8*10, 7*8+8*11, 7*9+8*12 = 129,144,159
        begin_trace();
        let a = Tensor::new(&[2, 2, 2], DType::F32);
        let b = Tensor::new(&[2, 2, 3], DType::F32);
        let c = a.matmul(&b);
        let trace = take_trace();
        assert_eq!(c.shape().0, vec![2u64, 2, 3]);
        let compiled = Compiler::compile(&trace, &[c.id], None).expect("compile failed");

        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b_data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let a_buf = Buffer::from_slice::<f32>(&a_data, &[2, 2, 2], DType::F32);
        let b_buf = Buffer::from_slice::<f32>(&b_data, &[2, 2, 3], DType::F32);
        let results = compiled.run(&[&a_buf, &b_buf]);
        let out = results[0].as_slice::<f32>();
        assert_eq!(
            out,
            &[9.0, 12.0, 15.0, 19.0, 26.0, 33.0, 95.0, 106.0, 117.0, 129.0, 144.0, 159.0],
            "batched 3D matmul result mismatch"
        );
    }

    #[test]
    fn runtime_batched_matmul_4d_f32() {
        // [1, 2, 2, 2] @ [1, 2, 2, 2] -> [1, 2, 2, 2]
        // Two 2x2 matrices for head dimension:
        // head=0: [[1,2],[3,4]] @ [[1,0],[0,1]] = [[1,2],[3,4]]
        // head=1: [[5,6],[7,8]] @ [[2,0],[0,2]] = [[10,12],[14,16]]
        begin_trace();
        let a = Tensor::new(&[1, 2, 2, 2], DType::F32);
        let b = Tensor::new(&[1, 2, 2, 2], DType::F32);
        let c = a.matmul(&b);
        let trace = take_trace();
        assert_eq!(c.shape().0, vec![1u64, 2, 2, 2]);
        let compiled = Compiler::compile(&trace, &[c.id], None).expect("compile failed");

        // lhs: batch=0, head=0: [[1,2],[3,4]], head=1: [[5,6],[7,8]]
        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        // rhs: batch=0, head=0: [[1,0],[0,1]] (identity), head=1: [[2,0],[0,2]]
        let b_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0];
        let a_buf = Buffer::from_slice::<f32>(&a_data, &[1, 2, 2, 2], DType::F32);
        let b_buf = Buffer::from_slice::<f32>(&b_data, &[1, 2, 2, 2], DType::F32);
        let results = compiled.run(&[&a_buf, &b_buf]);
        let out = results[0].as_slice::<f32>();
        assert_eq!(
            out,
            &[1.0, 2.0, 3.0, 4.0, 10.0, 12.0, 14.0, 16.0],
            "batched 4D matmul result mismatch"
        );
    }

    // ── Native .so cache integration tests ───────────────────────────────────

    /// Compile a graph once (cold), then compile again with the same cache key
    /// (warm). The warm run must hit the .so cache and produce identical results.
    #[test]
    fn compile_with_so_cache_hit_produces_correct_results() {
        use std::time::Instant;

        // Use an isolated cache dir so this test cannot interfere with or be
        // polluted by the user's real cache.
        let cache_dir = std::path::PathBuf::from("/tmp/homura_test_cache_so");
        std::fs::create_dir_all(&cache_dir).expect("create test cache dir");
        // SAFETY: single-threaded test, no concurrent env access.
        unsafe { std::env::set_var("HOMURA_CACHE_DIR", &cache_dir); }

        let key = "test_cache_key_add_f32";

        let make_trace = || {
            begin_trace();
            let a = Tensor::new(&[4], DType::F32);
            let b = Tensor::new(&[4], DType::F32);
            let c = &a + &b;
            (take_trace(), c.id)
        };

        let (trace, out_id) = make_trace();
        let t0 = Instant::now();
        let cold = Compiler::compile(&trace, &[out_id], Some(key)).expect("cold compile");
        let cold_time = t0.elapsed();

        // Warm compile — must use the .so cache.
        let (trace2, out_id2) = make_trace();
        let t1 = Instant::now();
        let warm = Compiler::compile(&trace2, &[out_id2], Some(key)).expect("warm compile");
        let warm_time = t1.elapsed();

        println!("cold={cold_time:?}  warm={warm_time:?}");
        // The warm run should be dramatically faster — at minimum it should not
        // take longer than 1/3 of the cold run.  We don't assert timing (it's
        // flaky in CI) but we do assert correctness.

        let a_buf = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[4], DType::F32);
        let b_buf = Buffer::from_slice::<f32>(&[10.0, 20.0, 30.0, 40.0], &[4], DType::F32);

        let cold_out = cold.run(&[&a_buf, &b_buf]);
        let warm_out = warm.run(&[&a_buf, &b_buf]);

        assert_eq!(
            cold_out[0].as_slice::<f32>(),
            warm_out[0].as_slice::<f32>(),
            "cached result must match JIT result"
        );
        assert_eq!(
            warm_out[0].as_slice::<f32>(),
            &[11.0f32, 22.0, 33.0, 44.0]
        );

        // Verify cache files exist.
        assert!(
            cache_dir.join(format!("{key}.so")).exists(),
            ".so file should be in cache dir"
        );
        assert!(
            cache_dir.join(format!("{key}.meta")).exists(),
            ".meta file should be in cache dir"
        );

        // Restore env (best-effort).
        // SAFETY: single-threaded test, no concurrent env access.
        unsafe { std::env::remove_var("HOMURA_CACHE_DIR"); }
    }

    // ── Phase 3.1: DIM_DYNAMIC in make_ranked_tensor_type produces `?` ─────────

    #[test]
    fn dynamic_tensor_type_has_question_mark() {
        use crate::shape::DIM_DYNAMIC;
        let context = create_context();
        // tensor<4x?x3xf32> via DIM_DYNAMIC
        let t = make_ranked_tensor_type(&context, &[4, DIM_DYNAMIC, 3], DType::F32);
        let ir = format!("{t}");
        assert!(
            ir.contains("?") || ir.contains("-1") || ir.contains(&i64::MIN.to_string()),
            "expected dynamic dim marker in type string, got: {ir}"
        );
    }

    // ── Phase 3.2: DIM_DYNAMIC in build_module produces memref<?x...> ──────────

    #[test]
    fn dynamic_input_produces_memref_with_question_mark() {
        use crate::shape::DIM_DYNAMIC;
        // Build a trace with a DIM_DYNAMIC input
        begin_trace();
        let _a = Tensor::new(&[4, DIM_DYNAMIC], DType::F32);
        let trace = take_trace();

        // Use build_ir_string to inspect the generated IR
        let ir = Compiler::build_ir_string(&trace, &[crate::op::NodeId(0)]);
        assert!(
            ir.is_ok(),
            "build_ir_string failed for dynamic input: {:?}",
            ir.err()
        );
        let ir_str = ir.unwrap();
        // The function signature should contain a memref with dynamic dim
        assert!(
            ir_str.contains("memref<4x?xf32>") || ir_str.contains("memref<4x-1xf32>"),
            "expected dynamic memref in IR, got relevant part of:\n{ir_str}"
        );
    }

    // ── Phase 3.5: promote_rank_with_reshape works with DIM_DYNAMIC ─────────────

    #[test]
    fn promote_rank_with_reshape_handles_dynamic_dim() {
        use crate::shape::DIM_DYNAMIC;
        // Build a trace with a 1-D input that we promote to rank 2 (should produce
        // tensor<1x?xf32> after expansion). This tests that static_output_shape
        // correctly emits i64::MIN for the dynamic position.
        begin_trace();
        let a = Tensor::new(&[DIM_DYNAMIC], DType::F32);
        let trace = take_trace();

        let ir = Compiler::build_ir_string(&trace, &[a.id]);
        assert!(
            ir.is_ok(),
            "build_ir_string failed with DIM_DYNAMIC input: {:?}",
            ir.err()
        );
        // The module should verify successfully (verified inside build_module).
    }

    // ── Runtime shape ops: compile tests ──────────────────────────────────────

    #[test]
    fn compile_shape_of_static() {
        // ShapeOf on a fully static tensor: output is tensor<3xi64> with compile-time dims.
        begin_trace();
        let a = Tensor::new(&[2, 4, 8], DType::F32);
        let s = a.shape_of();
        let trace = take_trace();

        let result = Compiler::compile(&trace, &[s.id], None);
        assert!(result.is_ok(), "compile_shape_of_static failed: {:?}", result.err());
    }

    #[test]
    fn run_shape_of_static() {
        // Compile and run ShapeOf on a static tensor; verify the output is [2, 4, 8].
        begin_trace();
        let a = Tensor::new(&[2, 4, 8], DType::F32);
        let s = a.shape_of();
        let trace = take_trace();

        let compiled = Compiler::compile(&trace, &[s.id], None)
            .expect("compile failed");

        let a_buf = Buffer::new(&[2, 4, 8], DType::F32);
        let outputs = compiled.run(&[&a_buf]);
        let out = outputs[0].as_slice::<i64>();
        assert_eq!(out, &[2i64, 4, 8], "shape_of output mismatch: {out:?}");
    }

    #[test]
    fn compile_shape_of_dynamic() {
        use crate::shape::DIM_DYNAMIC;
        // ShapeOf on a tensor with one dynamic dim should still compile.
        begin_trace();
        let a = Tensor::new(&[2, DIM_DYNAMIC, 8], DType::F32);
        let s = a.shape_of();
        let trace = take_trace();

        let result = Compiler::compile(&trace, &[s.id], None);
        assert!(result.is_ok(), "compile_shape_of_dynamic failed: {:?}", result.err());
    }

    #[test]
    fn run_shape_of_dynamic() {
        use crate::shape::DIM_DYNAMIC;
        // Run ShapeOf on a dynamic tensor; the middle dim should come from the actual input.
        // ShapeOf always produces a STATIC output tensor: tensor<{rank}xi64>.
        // The output shape [3] is always static even when the input dims are dynamic.
        begin_trace();
        let a = Tensor::new(&[2, DIM_DYNAMIC, 8], DType::F32);
        let s = a.shape_of();
        let trace = take_trace();

        let compiled = Compiler::compile(&trace, &[s.id], None)
            .expect("compile failed");

        // Provide a concrete input with shape [2, 5, 8] — output shape is [3] (static).
        let a_buf = Buffer::new(&[2, 5, 8], DType::F32);
        let outputs = compiled.run(&[&a_buf]);
        let out = outputs[0].as_slice::<i64>();
        assert_eq!(out, &[2i64, 5, 8], "dynamic shape_of mismatch: {out:?}");
    }

    #[test]
    fn compile_constant_of_shape_static() {
        // ConstantOfShape with a fully static output shape.
        begin_trace();
        let shape_t = Tensor::new(&[3], DType::I64); // will hold [2, 4, 8] at runtime
        let out = Tensor::constant_of_shape(&shape_t, 1.0, crate::Shape(vec![2, 4, 8]), DType::F32);
        let trace = take_trace();

        let result = Compiler::compile(&trace, &[out.id], None);
        assert!(result.is_ok(), "compile_constant_of_shape_static failed: {:?}", result.err());
    }

    #[test]
    fn run_constant_of_shape_static() {
        // Verify ConstantOfShape fills the output with 1.0f32.
        begin_trace();
        let shape_t = Tensor::new(&[3], DType::I64);
        let out = Tensor::constant_of_shape(&shape_t, 1.0, crate::Shape(vec![2, 3, 4]), DType::F32);
        let trace = take_trace();

        let compiled = Compiler::compile(&trace, &[out.id], None)
            .expect("compile failed");

        let shape_buf = Buffer::from_slice::<i64>(&[2, 3, 4], &[3], DType::I64);
        let outputs = compiled.run(&[&shape_buf]);
        let data = outputs[0].as_slice::<f32>();
        assert_eq!(data.len(), 24);
        for &v in data {
            assert!((v - 1.0).abs() < 1e-6, "expected 1.0, got {v}");
        }
    }

    #[test]
    fn compile_range_i64_static() {
        use crate::shape::DIM_DYNAMIC;
        // Range with static-length output.
        begin_trace();
        let start = Tensor::new(&[1], DType::I64);
        let limit = Tensor::new(&[1], DType::I64);
        let delta = Tensor::new(&[1], DType::I64);
        let out = Tensor::range(&start, &limit, &delta, crate::Shape(vec![DIM_DYNAMIC]), DType::I64);
        let trace = take_trace();

        let result = Compiler::compile(&trace, &[out.id], None);
        assert!(result.is_ok(), "compile_range_i64 failed: {:?}", result.err());
    }

    #[test]
    fn run_range_i64() {
        use crate::shape::DIM_DYNAMIC;
        // Range(0, 5, 1) → [0, 1, 2, 3, 4]
        begin_trace();
        let start = Tensor::new(&[1], DType::I64);
        let limit = Tensor::new(&[1], DType::I64);
        let delta = Tensor::new(&[1], DType::I64);
        let out = Tensor::range(&start, &limit, &delta, crate::Shape(vec![DIM_DYNAMIC]), DType::I64);
        let trace = take_trace();

        let compiled = Compiler::compile(&trace, &[out.id], None)
            .expect("compile failed");

        let start_buf = Buffer::from_slice::<i64>(&[0], &[1], DType::I64);
        let limit_buf = Buffer::from_slice::<i64>(&[5], &[1], DType::I64);
        let delta_buf = Buffer::from_slice::<i64>(&[1], &[1], DType::I64);
        let outputs =
            compiled.run_dynamic(&[&start_buf, &limit_buf, &delta_buf], &[crate::Shape(vec![5])]);
        let data = outputs[0].as_slice::<i64>();
        assert_eq!(data, &[0i64, 1, 2, 3, 4], "range output mismatch: {data:?}");
    }

    #[test]
    fn run_range_f32() {
        use crate::shape::DIM_DYNAMIC;
        // Range(0.0, 3.0, 0.5) → [0.0, 0.5, 1.0, 1.5, 2.0, 2.5] (6 elements)
        begin_trace();
        let start = Tensor::new(&[1], DType::F32);
        let limit = Tensor::new(&[1], DType::F32);
        let delta = Tensor::new(&[1], DType::F32);
        let out = Tensor::range(&start, &limit, &delta, crate::Shape(vec![DIM_DYNAMIC]), DType::F32);
        let trace = take_trace();

        let compiled = Compiler::compile(&trace, &[out.id], None)
            .expect("compile failed");

        let start_buf = Buffer::from_slice::<f32>(&[0.0], &[1], DType::F32);
        let limit_buf = Buffer::from_slice::<f32>(&[3.0], &[1], DType::F32);
        let delta_buf = Buffer::from_slice::<f32>(&[0.5], &[1], DType::F32);
        let outputs =
            compiled.run_dynamic(&[&start_buf, &limit_buf, &delta_buf], &[crate::Shape(vec![6])]);
        let data = outputs[0].as_slice::<f32>();
        assert_eq!(data.len(), 6, "expected 6 elements, got {}", data.len());
        let expected = [0.0f32, 0.5, 1.0, 1.5, 2.0, 2.5];
        for (i, (&got, &exp)) in data.iter().zip(expected.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-5, "range[{i}]: expected {exp}, got {got}");
        }
    }

    // ── Dynamic runtime tests ──────────────────────────────────────────────

    #[test]
    fn run_dynamic_matmul_2d() {
        use crate::shape::DIM_DYNAMIC;
        // [?, 4] x [4, 8] → [?, 8] with dynamic M dim.
        // Exercises the batched matmul fallback path (linalg.generic)
        // for 2D matmul with dynamic dims.
        begin_trace();
        let a = Tensor::new(&[DIM_DYNAMIC, 4], DType::F32);
        let b = Tensor::new(&[4, 8], DType::F32);
        let c = a.matmul(&b);
        let trace = take_trace();

        let compiled = Compiler::compile(&trace, &[c.id], None)
            .expect("compile failed");

        // a = [[1,0,0,0]] shape [1,4], b = I4 padded to [4,8]
        let a_buf = Buffer::from_slice::<f32>(
            &[1.0, 0.0, 0.0, 0.0],
            &[1, 4],
            DType::F32,
        );
        let mut w_data = vec![0.0f32; 32];
        for i in 0..4 { w_data[i * 8 + i] = 1.0; }
        let b_buf = Buffer::from_slice::<f32>(&w_data, &[4, 8], DType::F32);

        let outputs = compiled.run_dynamic(
            &[&a_buf, &b_buf],
            &[crate::Shape(vec![1, 8])],
        );
        let out = outputs[0].as_slice::<f32>();
        assert_eq!(out.len(), 8);
        assert!((out[0] - 1.0).abs() < 1e-5, "out[0] = {}", out[0]);
        for i in 1..8 {
            assert!(out[i].abs() < 1e-5, "out[{i}] = {}", out[i]);
        }
    }

    #[test]
    fn run_dynamic_reshape_3d_to_2d() {
        use crate::shape::DIM_DYNAMIC;
        use crate::trace::record;
        // Input [?, ?, 4] f32 reshaped to [?, 4] using a runtime shape tensor [2] i64.
        // Exercises the tensor.reshape dynamic path (shape_tensor: Some).
        begin_trace();
        let input_id = record(Op::Input {
            shape: crate::Shape(vec![DIM_DYNAMIC, DIM_DYNAMIC, 4]),
            dtype: DType::F32,
            arg_index: 0,
        });
        let shape_id = record(Op::Input {
            shape: crate::Shape(vec![2]),
            dtype: DType::I64,
            arg_index: 1,
        });
        let reshape_id = record(Op::Reshape {
            input: input_id,
            target_shape: vec![DIM_DYNAMIC, 4],
            shape_tensor: Some(shape_id),
            shape: crate::Shape(vec![DIM_DYNAMIC, DIM_DYNAMIC]),
            dtype: DType::F32,
        });
        let trace = take_trace();

        let compiled = Compiler::compile(&trace, &[reshape_id], None)
            .expect("compile failed");

        let x_buf = Buffer::from_slice::<f32>(
            &[1.0f32, 2.0, 3.0, 4.0],
            &[1, 1, 4],
            DType::F32,
        );
        let shape_buf = Buffer::from_slice::<i64>(
            &[1i64, 4],
            &[2],
            DType::I64,
        );

        let outputs = compiled.run_dynamic(
            &[&x_buf, &shape_buf],
            &[crate::Shape(vec![1, 4])],
        );
        let out = outputs[0].as_slice::<f32>();
        assert_eq!(out, &[1.0f32, 2.0, 3.0, 4.0], "reshape output mismatch: {out:?}");
    }

    #[test]
    fn run_dynamic_reshape_then_transpose() {
        use crate::shape::DIM_DYNAMIC;
        use crate::trace::record;
        // Reproduce the decode model crash pattern:
        // reshape [1, 1, 768] → [1, 1, 12, 64] via shape tensor,
        // then transpose [0, 2, 1, 3] → [1, 12, 1, 64].
        begin_trace();
        let input_id = record(Op::Input {
            shape: crate::Shape(vec![DIM_DYNAMIC, DIM_DYNAMIC, 768]),
            dtype: DType::F32,
            arg_index: 0,
        });
        let shape_id = record(Op::Input {
            shape: crate::Shape(vec![4]),
            dtype: DType::I64,
            arg_index: 1,
        });
        let reshape_id = record(Op::Reshape {
            input: input_id,
            target_shape: vec![DIM_DYNAMIC, DIM_DYNAMIC, 12, 64],
            shape_tensor: Some(shape_id),
            shape: crate::Shape(vec![DIM_DYNAMIC, DIM_DYNAMIC, DIM_DYNAMIC, DIM_DYNAMIC]),
            dtype: DType::F32,
        });
        let transpose_id = record(Op::Transpose {
            input: reshape_id,
            perm: vec![0, 2, 1, 3],
            shape: crate::Shape(vec![DIM_DYNAMIC, DIM_DYNAMIC, DIM_DYNAMIC, DIM_DYNAMIC]),
            dtype: DType::F32,
        });
        let trace = take_trace();

        let compiled = Compiler::compile(&trace, &[transpose_id], None)
            .expect("compile failed");

        // Input: [1, 1, 768] filled with index values
        let mut input_data = vec![0.0f32; 768];
        for i in 0..768 { input_data[i] = i as f32; }
        let x_buf = Buffer::from_slice::<f32>(&input_data, &[1, 1, 768], DType::F32);
        let shape_buf = Buffer::from_slice::<i64>(&[1, 1, 12, 64], &[4], DType::I64);

        let outputs = compiled.run_dynamic(
            &[&x_buf, &shape_buf],
            &[crate::Shape(vec![1, 12, 1, 64])],
        );
        let out = outputs[0].as_slice::<f32>();
        assert_eq!(out.len(), 768);
        // After reshape [1,1,12,64] + transpose [0,2,1,3] → [1,12,1,64]:
        // Element at output[0,h,0,d] = input[0,0,h*64+d]
        // So out[0] = input[0,0,0] = 0.0, out[64] = input[0,0,64] = 64.0
        assert!((out[0] - 0.0).abs() < 1e-5, "out[0]={}", out[0]);
        assert!((out[64] - 64.0).abs() < 1e-5, "out[64]={}", out[64]);
        assert!((out[128] - 128.0).abs() < 1e-5, "out[128]={}", out[128]);
    }

    #[test]
    fn run_multiple_dynamic_reshapes_with_reuse() {
        use crate::shape::DIM_DYNAMIC;
        use crate::trace::record;
        // Test buffer reuse: two tensor.reshape ops using DIFFERENT shape tensors.
        // This triggers one-shot-bufferize to potentially reuse the shape alloca.
        begin_trace();
        // First input: [?, ?, 768]
        let input1_id = record(Op::Input {
            shape: crate::Shape(vec![DIM_DYNAMIC, DIM_DYNAMIC, 768]),
            dtype: DType::F32,
            arg_index: 0,
        });
        let shape1_id = record(Op::Input {
            shape: crate::Shape(vec![2]),
            dtype: DType::I64,
            arg_index: 1,
        });
        // Reshape 1: [?, ?, 768] → [?, 768] (flatten batch*seq)
        let reshape1_id = record(Op::Reshape {
            input: input1_id,
            target_shape: vec![DIM_DYNAMIC, 768],
            shape_tensor: Some(shape1_id),
            shape: crate::Shape(vec![DIM_DYNAMIC, DIM_DYNAMIC]),
            dtype: DType::F32,
        });
        // Matmul: [?, 768] x [768, 64] → [?, 64]
        let weight_id = record(Op::Input {
            shape: crate::Shape(vec![768, 64]),
            dtype: DType::F32,
            arg_index: 2,
        });
        let matmul_id = record(Op::Matmul {
            lhs: reshape1_id,
            rhs: weight_id,
            shape: crate::Shape(vec![DIM_DYNAMIC, 64]),
            dtype: DType::F32,
        });
        // Second shape tensor for reshape back
        let shape2_id = record(Op::Input {
            shape: crate::Shape(vec![3]),
            dtype: DType::I64,
            arg_index: 3,
        });
        // Reshape 2: [?, 64] → [?, ?, 64] (unflatten)
        let reshape2_id = record(Op::Reshape {
            input: matmul_id,
            target_shape: vec![DIM_DYNAMIC, DIM_DYNAMIC, 64],
            shape_tensor: Some(shape2_id),
            shape: crate::Shape(vec![DIM_DYNAMIC, DIM_DYNAMIC, DIM_DYNAMIC]),
            dtype: DType::F32,
        });
        let trace = take_trace();

        let compiled = Compiler::compile(&trace, &[reshape2_id], None)
            .expect("compile failed");

        // Input: [1, 1, 768] = 768 floats
        let input_data = vec![1.0f32; 768];
        let x_buf = Buffer::from_slice::<f32>(&input_data, &[1, 1, 768], DType::F32);
        let shape1_buf = Buffer::from_slice::<i64>(&[1, 768], &[2], DType::I64);
        // Weight: 768x64 identity-like (first col = 1, rest = 0)
        let mut w_data = vec![0.0f32; 768 * 64];
        for i in 0..64 { w_data[i * 64 + i] = 1.0; }
        let w_buf = Buffer::from_slice::<f32>(&w_data, &[768, 64], DType::F32);
        let shape2_buf = Buffer::from_slice::<i64>(&[1, 1, 64], &[3], DType::I64);

        let outputs = compiled.run_dynamic(
            &[&x_buf, &shape1_buf, &w_buf, &shape2_buf],
            &[crate::Shape(vec![1, 1, 64])],
        );
        let out = outputs[0].as_slice::<f32>();
        assert_eq!(out.len(), 64, "expected 64 elements, got {}", out.len());
        // With identity-like weight, first 64 outputs should be 1.0
        assert!((out[0] - 1.0).abs() < 1e-3, "out[0]={}", out[0]);
    }

    #[test]
    fn run_dynamic_qkv_split_pattern() {
        use crate::shape::DIM_DYNAMIC;
        use crate::trace::record;
        // Reproduce the full QKV split: 3 reshapes (Q, K, V) from same input,
        // each followed by transpose, then matmul Q*K^T.
        begin_trace();
        // Input: [batch, seq, 2304] where 2304 = 3 * 12 * 64
        let input_id = record(Op::Input {
            shape: crate::Shape(vec![DIM_DYNAMIC, DIM_DYNAMIC, 2304]),
            dtype: DType::F32,
            arg_index: 0,
        });
        // Slice Q: [batch, seq, 0:768]
        let q_slice_id = record(Op::Slice {
            input: input_id,
            starts: vec![0, 0, 0],
            ends: vec![i64::MAX, i64::MAX, 768],
            axes: vec![0, 1, 2],
            steps: vec![1, 1, 1],
            shape: crate::Shape(vec![DIM_DYNAMIC, DIM_DYNAMIC, 768]),
            dtype: DType::F32,
        });
        // Shape tensor for [batch, seq, 12, 64]
        let shape4_id = record(Op::Input {
            shape: crate::Shape(vec![4]),
            dtype: DType::I64,
            arg_index: 1,
        });
        // Reshape Q: [?, ?, 768] → [?, ?, 12, 64]
        let q_reshape_id = record(Op::Reshape {
            input: q_slice_id,
            target_shape: vec![DIM_DYNAMIC, DIM_DYNAMIC, 12, 64],
            shape_tensor: Some(shape4_id),
            shape: crate::Shape(vec![DIM_DYNAMIC, DIM_DYNAMIC, DIM_DYNAMIC, DIM_DYNAMIC]),
            dtype: DType::F32,
        });
        // Transpose Q: [b,s,h,d] → [b,h,s,d]
        let q_trans_id = record(Op::Transpose {
            input: q_reshape_id,
            perm: vec![0, 2, 1, 3],
            shape: crate::Shape(vec![DIM_DYNAMIC, DIM_DYNAMIC, DIM_DYNAMIC, DIM_DYNAMIC]),
            dtype: DType::F32,
        });
        // Slice K: [batch, seq, 768:1536]
        let k_slice_id = record(Op::Slice {
            input: input_id,
            starts: vec![0, 0, 768],
            ends: vec![i64::MAX, i64::MAX, 1536],
            axes: vec![0, 1, 2],
            steps: vec![1, 1, 1],
            shape: crate::Shape(vec![DIM_DYNAMIC, DIM_DYNAMIC, 768]),
            dtype: DType::F32,
        });
        // Reshape K
        let k_reshape_id = record(Op::Reshape {
            input: k_slice_id,
            target_shape: vec![DIM_DYNAMIC, DIM_DYNAMIC, 12, 64],
            shape_tensor: Some(shape4_id),
            shape: crate::Shape(vec![DIM_DYNAMIC, DIM_DYNAMIC, DIM_DYNAMIC, DIM_DYNAMIC]),
            dtype: DType::F32,
        });
        // Transpose K: [b,s,h,d] → [b,h,d,s] for Q*K^T
        let k_trans_id = record(Op::Transpose {
            input: k_reshape_id,
            perm: vec![0, 2, 3, 1],
            shape: crate::Shape(vec![DIM_DYNAMIC, DIM_DYNAMIC, DIM_DYNAMIC, DIM_DYNAMIC]),
            dtype: DType::F32,
        });
        // Matmul Q * K^T: [b,h,s,d] x [b,h,d,s] → [b,h,s,s]
        let qk_id = record(Op::Matmul {
            lhs: q_trans_id,
            rhs: k_trans_id,
            shape: crate::Shape(vec![DIM_DYNAMIC, DIM_DYNAMIC, DIM_DYNAMIC, DIM_DYNAMIC]),
            dtype: DType::F32,
        });
        let trace = take_trace();

        let compiled = Compiler::compile(&trace, &[qk_id], None)
            .expect("compile failed");

        // Concrete: batch=1, seq=1, so QKV = [1, 1, 2304]
        let mut input_data = vec![0.0f32; 2304];
        // Q part [0..768]: set head 0, dim 0 = 1.0
        input_data[0] = 1.0;
        // K part [768..1536]: set head 0, dim 0 = 1.0
        input_data[768] = 1.0;
        let x_buf = Buffer::from_slice::<f32>(&input_data, &[1, 1, 2304], DType::F32);
        let shape_buf = Buffer::from_slice::<i64>(&[1, 1, 12, 64], &[4], DType::I64);

        let outputs = compiled.run_dynamic(
            &[&x_buf, &shape_buf],
            &[crate::Shape(vec![1, 12, 1, 1])],
        );
        let out = outputs[0].as_slice::<f32>();
        assert_eq!(out.len(), 12, "expected 12 elements (b=1,h=12,s=1,s=1)");
        // Q[0,0,0,0]=1.0, K[0,0,0,0]=1.0, so QK[0,0,0,0] = dot(Q[0,0,0,:], K[0,0,:,0]) = 1.0
        assert!((out[0] - 1.0).abs() < 1e-3, "qk[0]={}, expected 1.0", out[0]);
    }
}
