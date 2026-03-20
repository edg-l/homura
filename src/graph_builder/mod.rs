use melior::dialect::DialectRegistry;
use melior::{
    Context,
    dialect::func,
    ir::{
        Attribute, Block, Identifier, Location, Module, Region, RegionLike, ShapedTypeLike,
        ValueLike,
        attribute::{StringAttribute, TypeAttribute},
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
    compiler::CompileError,
    runtime::{CompiledGraph, OutputDesc},
};

mod emit_arithmetic;
mod emit_linalg;
mod emit_reshape;

// ── Transform schedule mode ───────────────────────────────────────────────────

/// Controls which transform schedule is attached to a compiled kernel.
#[derive(Clone, Copy, PartialEq)]
pub enum TransformMode {
    /// No transform schedule — lightweight pipeline (elementwise, BatchNorm).
    None,
    /// Full tiling + vectorize + OpenMP schedule.
    Full,
    /// Tile only — no MLIR vectorize, no OpenMP. LLVM auto-vectorizes.
    VectorizeOnly,
    /// Tile + OpenMP parallel on N for large-N matmuls (e.g. LM head 768×50257).
    /// Uses forall on N for multi-threaded memory bandwidth, inner for on N+K
    /// for LLVM auto-vectorization. No MLIR vectorize/pad/outline.
    /// The `n_tile` field controls the forall tile size on the N dimension.
    TileParallel { n_tile: usize },
}

impl TransformMode {
    /// Choose a TileParallel mode with an adaptive forall tile size.
    ///
    /// Given the smallest N dimension across matmuls in the kernel, picks
    /// a tile size that produces at least `available_parallelism` tiles,
    /// rounded down to a power of two (for clean division), clamped to [16, 256].
    pub fn tile_parallel_adaptive(min_n: usize) -> Self {
        let num_cores = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(8);
        // Target at least 1.5× cores worth of tiles for good load balancing.
        let target_tiles = num_cores + num_cores / 2;
        let ideal = min_n / target_tiles;
        // Round down to power of two, clamp to [16, 256].
        let n_tile = if ideal <= 1 {
            1
        } else {
            1usize << (usize::BITS - 1 - ideal.leading_zeros())
        };
        let n_tile = n_tile.max(16).min(256);
        TransformMode::TileParallel { n_tile }
    }
}

// ── Tensor wrapper ────────────────────────────────────────────────────────────

/// A tensor value in a graph being built. Wraps an MLIR `Value` — shape and
/// dtype are read from the MLIR type, not stored separately.
#[derive(Clone, Copy)]
pub struct Tensor<'c> {
    value: melior::ir::Value<'c, 'c>,
}

impl<'c> Tensor<'c> {
    pub(crate) fn from_value(value: melior::ir::Value<'c, 'c>) -> Self {
        Self { value }
    }

    /// The underlying MLIR SSA value.
    pub fn value(&self) -> melior::ir::Value<'c, 'c> {
        self.value
    }

    /// Read the shape from the MLIR `RankedTensorType`. `None` = dynamic dim.
    pub fn shape(&self) -> Vec<Option<u64>> {
        let rtt = self.ranked_tensor_type();
        let rank = rtt.rank();
        (0..rank)
            .map(|i| {
                let raw = unsafe {
                    mlir_sys::mlirShapedTypeGetDimSize(
                        melior::ir::TypeLike::to_raw(&rtt),
                        i as isize,
                    )
                };
                if raw < 0 { None } else { Some(raw as u64) }
            })
            .collect()
    }

    /// Read the dtype from the MLIR element type.
    pub fn dtype(&self) -> DType {
        let rtt = self.ranked_tensor_type();
        mlir_element_type_to_dtype(rtt.element())
    }

    /// Number of dimensions.
    pub fn rank(&self) -> usize {
        self.ranked_tensor_type().rank()
    }

    fn ranked_tensor_type(&self) -> RankedTensorType<'c> {
        let ty = self.value.r#type();
        RankedTensorType::try_from(ty).expect("Tensor value must have RankedTensorType")
    }
}

/// Convert an MLIR element type back to our DType enum.
pub(super) fn mlir_element_type_to_dtype(ty: melior::ir::Type) -> DType {
    let s = ty.to_string();
    match s.as_str() {
        "f32" => DType::F32,
        "f64" => DType::F64,
        "i32" => DType::I32,
        "i64" => DType::I64,
        "bf16" => DType::BF16,
        other => panic!("unsupported MLIR element type: {other}"),
    }
}

// ── GraphBuilder ──────────────────────────────────────────────────────────────

/// Tracks one function argument.
struct ArgInfo {
    _shape: Vec<Option<u64>>,
    _dtype: DType,
    _is_input: bool,
}

/// A completed sub-function ready to be emitted into the MLIR module.
struct CompletedSubFunction<'c> {
    name: String,
    block: Block<'c>,
    arg_types: Vec<melior::ir::Type<'c>>,
    return_types: Vec<melior::ir::Type<'c>>,
}

/// Stashed caller state while a sub-function is being built.
struct SubFunctionBuildState<'c> {
    name: String,
    caller_block: Block<'c>,
    caller_args: Vec<ArgInfo>,
    /// Caller-side tensor values to pass to func.call when finalized.
    caller_arg_values: Vec<melior::ir::Value<'c, 'c>>,
}

/// Handle to a sub-function being built. Returned by `begin_subfunction`.
pub struct SubFunctionHandle {
    pub _index: usize,
}

/// Owns the MLIR Context. Create one, then call `.builder()` to get a
/// `GraphBuilder` that borrows from it.
///
/// ```ignore
/// let ctx = GraphContext::new();
/// let mut gb = ctx.builder();
/// let x = gb.input(&[Some(4)], DType::F32);
/// let graph = gb.compile(&[&x])?;
/// ```
pub struct GraphContext {
    context: Context,
}

impl Default for GraphContext {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphContext {
    pub fn new() -> Self {
        Self {
            context: create_context(),
        }
    }

    pub fn builder(&self) -> GraphBuilder<'_> {
        GraphBuilder::new(&self.context)
    }
}

/// Compile MLIR text to object files only (no linking).
///
/// Returns the .o file paths. The caller is responsible for linking them
/// into a shared library. Used by the per-kernel compilation path where
/// multiple kernels are linked into a single .so.
pub fn compile_to_objects(
    mlir_text: &str,
    label: &str,
    func_name: &str,
    output_dir: &std::path::Path,
) -> Result<Vec<std::path::PathBuf>, CompileError> {
    use melior::ir::Module;
    use melior::pass;

    let context = create_context();

    // Dump pre-pass IR for specific kernels.
    if std::env::var("HOMURA_DUMP_KERNEL").is_ok_and(|k| label.contains(&k)) {
        let _ = std::fs::write("/tmp/homura_kernel_pre.mlir", mlir_text);
        eprintln!("[dump] {label} pre-pass IR → /tmp/homura_kernel_pre.mlir");
    }

    let mut module = Module::parse(&context, mlir_text).ok_or(CompileError::Verification)?;

    let has_schedule = mlir_text.contains("transform.with_named_sequence");
    let vectorize_only = mlir_text.contains("homura.vectorize_only");
    register_all_passes();
    let pass_manager = pass::PassManager::new(&context);

    let pipeline = if has_schedule && !vectorize_only {
        "builtin.module(\
            func.func(canonicalize,cse),\
            transform-interpreter,\
            func.func(canonicalize,cse),\
            func.func(linalg-fuse-elementwise-ops,canonicalize,cse),\
            symbol-dce,\
            func.func(lower-vector-multi-reduction,lower-vector-mask),\
            one-shot-bufferize{bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map unknown-type-conversion=identity-layout-map},\
            func.func(buffer-hoisting,promote-buffers-to-stack{max-alloc-size-in-bytes=4096}),\
            ownership-based-buffer-deallocation,buffer-deallocation-simplification,convert-bufferization-to-memref,canonicalize,\
            scf-forall-to-parallel,\
            fold-memref-alias-ops,\
            convert-vector-to-scf,\
            convert-linalg-to-loops,\
            fold-memref-alias-ops,\
            lower-affine,\
            convert-scf-to-openmp,\
            convert-openmp-to-llvm,\
            convert-scf-to-cf,\
            canonicalize,\
            cse,\
            sccp,\
            convert-vector-to-llvm{vector-contract-lowering=outerproduct},\
            convert-ub-to-llvm,\
            convert-math-to-llvm,\
            expand-strided-metadata,\
            lower-affine,\
            finalize-memref-to-llvm,\
            convert-arith-to-llvm,\
            convert-index-to-llvm,\
            convert-cf-to-llvm,\
            convert-func-to-llvm,\
            convert-openmp-to-llvm,\
            reconcile-unrealized-casts\
        )"
    } else if has_schedule {
        "builtin.module(\
            func.func(canonicalize,cse),\
            transform-interpreter,\
            func.func(canonicalize,cse),\
            func.func(linalg-fuse-elementwise-ops,canonicalize,cse),\
            symbol-dce,\
            func.func(lower-vector-multi-reduction,lower-vector-mask),\
            one-shot-bufferize{bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map unknown-type-conversion=identity-layout-map},\
            func.func(buffer-hoisting,promote-buffers-to-stack{max-alloc-size-in-bytes=4096}),\
            ownership-based-buffer-deallocation,buffer-deallocation-simplification,convert-bufferization-to-memref,canonicalize,\
            fold-memref-alias-ops,\
            convert-vector-to-scf,\
            convert-linalg-to-loops,\
            fold-memref-alias-ops,\
            lower-affine,\
            convert-scf-to-cf,\
            canonicalize,\
            cse,\
            sccp,\
            convert-vector-to-llvm{vector-contract-lowering=outerproduct},\
            convert-ub-to-llvm,\
            convert-math-to-llvm,\
            expand-strided-metadata,\
            lower-affine,\
            finalize-memref-to-llvm,\
            convert-arith-to-llvm,\
            convert-index-to-llvm,\
            convert-cf-to-llvm,\
            convert-func-to-llvm,\
            reconcile-unrealized-casts\
        )"
    } else {
        "builtin.module(\
            func.func(linalg-fuse-elementwise-ops,canonicalize,cse),\
            one-shot-bufferize{bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map unknown-type-conversion=identity-layout-map},\
            func.func(buffer-hoisting,promote-buffers-to-stack{max-alloc-size-in-bytes=4096}),\
            ownership-based-buffer-deallocation,buffer-deallocation-simplification,convert-bufferization-to-memref,canonicalize,\
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
        )"
    };

    parse_pass_pipeline(pass_manager.as_operation_pass_manager(), pipeline)
        .map_err(CompileError::Pass)?;
    pass_manager.run(&mut module).map_err(CompileError::Pass)?;

    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let tid = std::thread::current().id();
    let base_name = format!(
        "homura_{func_name}_{}_{}_{}_{:08x}",
        std::process::id(),
        format!("{:?}", tid)
            .replace("ThreadId(", "")
            .replace(")", ""),
        dur.as_secs(),
        dur.subsec_nanos()
    );

    crate::compiler::emit_object_files_monolithic_pub(
        &module, output_dir, &base_name, label, func_name,
    )
}

// ── Reshape decomposition: collapse_shape / expand_shape ──────────────────────

/// Result of analyzing how to decompose a reshape into collapse/expand ops.
#[derive(Debug, Clone, PartialEq)]
enum ReassocResult {
    /// Same rank, same dims — no-op.
    Identity,
    /// Rank decreases: groups of input dims collapse into output dims.
    /// `inferred_shape` is the result shape computed from input + reassociation
    /// (may be more precise than the caller's out_shape).
    Collapse {
        reassoc: Vec<Vec<usize>>,
        inferred_shape: Vec<Option<u64>>,
    },
    /// Rank increases: each input dim expands into a group of output dims.
    Expand { reassoc: Vec<Vec<usize>> },
    /// General: collapse to intermediate shape, then expand to target.
    CollapseExpand {
        collapse_reassoc: Vec<Vec<usize>>,
        intermediate_shape: Vec<Option<u64>>,
        expand_reassoc: Vec<Vec<usize>>,
    },
}

/// Format a reassociation as an MLIR attribute string, e.g. `[[0, 1], [2]]`.
fn reassoc_to_string(reassoc: &[Vec<usize>]) -> String {
    let groups: Vec<String> = reassoc
        .iter()
        .map(|g| {
            format!(
                "[{}]",
                g.iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
        .collect();
    format!("[{}]", groups.join(", "))
}

/// Compute the reassociation mapping for a reshape from `in_shape` to `out_shape`.
///
/// Returns `None` when the shapes cannot be decomposed into valid
/// collapse_shape/expand_shape ops (e.g., multiple dynamic dims in a group).
///
/// The algorithm greedily matches contiguous dim products left-to-right.
fn compute_reassociation(
    in_shape: &[Option<u64>],
    out_shape: &[Option<u64>],
) -> Option<ReassocResult> {
    let in_rank = in_shape.len();
    let out_rank = out_shape.len();

    // Identity check.
    if in_rank == out_rank && in_shape == out_shape {
        return Some(ReassocResult::Identity);
    }

    if out_rank < in_rank {
        // Collapse: merge groups of input dims into each output dim.
        try_collapse(in_shape, out_shape)
    } else if out_rank > in_rank {
        // Expand: split each input dim into a group of output dims.
        try_expand(in_shape, out_shape)
    } else {
        // Same rank but different shapes — try collapse-then-expand through 1D,
        // or identity with type cast.
        try_collapse_expand(in_shape, out_shape)
    }
}

/// Try to build a collapse reassociation: groups of contiguous input dims
/// whose products match each output dim.
///
/// `tensor.collapse_shape` supports multiple dynamic dims per group —
/// MLIR computes the collapsed dim as the product at runtime.
///
/// Strategy: match static dims from the END first (they anchor the grouping),
/// then assign remaining input dims to the first dynamic output dim.
fn try_collapse(in_shape: &[Option<u64>], out_shape: &[Option<u64>]) -> Option<ReassocResult> {
    let in_rank = in_shape.len();
    let out_rank = out_shape.len();

    if out_rank == 0 || in_rank == 0 {
        return None;
    }

    // Try the forward greedy approach first for fully static cases,
    // then fall back to end-matching for mixed static/dynamic.

    // End-matching: match static output dims from the right, accumulate
    // the rest into the first dynamic group.
    let mut reassoc: Vec<Vec<usize>> = vec![Vec::new(); out_rank];

    // Match from the end: pair output dims with input dims right-to-left.
    let mut in_end = in_rank;
    let mut out_end = out_rank;
    while out_end > 1 && in_end > 0 {
        let out_i = out_end - 1;
        let in_i = in_end - 1;

        match (out_shape[out_i], in_shape[in_i]) {
            (Some(out_val), Some(in_val)) if out_val == in_val => {
                // Static dims match 1:1.
                reassoc[out_i].push(in_i);
                in_end -= 1;
                out_end -= 1;
            }
            (Some(out_val), Some(in_val)) => {
                // Static dims differ — need to match a product of input dims.
                let mut product = in_val;
                let mut group = vec![in_i];
                let mut j = in_i;
                while product < out_val && j > 0 {
                    j -= 1;
                    match in_shape[j] {
                        Some(d) => {
                            product *= d;
                            group.push(j);
                        }
                        None => break,
                    }
                }
                if product == out_val {
                    group.reverse();
                    reassoc[out_i] = group.clone();
                    in_end = *group.first().unwrap();
                    out_end -= 1;
                } else {
                    break;
                }
            }
            (None, None) => {
                // Both dynamic — match 1:1.
                reassoc[out_i].push(in_i);
                in_end -= 1;
                out_end -= 1;
            }
            _ => {
                // Mismatch (one static, one dynamic) — stop matching from end.
                break;
            }
        }
    }

    // Distribute remaining input dims across remaining output groups.
    // The FIRST output group gets all "extra" input dims (ONNX contiguous merge
    // semantics: leading dims merge before trailing dims).
    if in_end > 0 && out_end > 0 {
        let excess = in_end as isize - out_end as isize;
        if excess < 0 {
            return None; // more output groups than input dims
        }
        let mut in_fwd = 0;
        for (out_i, group) in reassoc[..out_end].iter_mut().enumerate() {
            if out_i == 0 {
                // First group: take (1 + excess) input dims.
                let n = 1 + excess as usize;
                for _ in 0..n {
                    if in_fwd < in_end {
                        group.push(in_fwd);
                        in_fwd += 1;
                    }
                }
            } else {
                // Subsequent groups: take one input dim each.
                if in_fwd < in_end {
                    group.push(in_fwd);
                    in_fwd += 1;
                }
            }
        }
        if in_fwd != in_end {
            return None;
        }
    } else if in_end > 0 && out_end == 0 {
        // All output groups matched from the end, but extra input dims remain.
        // Prepend them to the first matched group (smallest output index with a group).
        let first_matched = reassoc.iter().position(|g| !g.is_empty());
        if let Some(idx) = first_matched {
            let extra: Vec<usize> = (0..in_end).collect();
            let mut merged = extra;
            merged.extend_from_slice(&reassoc[idx]);
            reassoc[idx] = merged;
        } else {
            return None;
        }
    } else if in_end > 0 {
        return None;
    }

    // Validate: all output groups must be non-empty and cover all input dims.
    let mut covered = vec![false; in_rank];
    for group in &reassoc {
        if group.is_empty() {
            return None;
        }
        for &i in group {
            if i >= in_rank || covered[i] {
                return None;
            }
            covered[i] = true;
        }
    }
    if covered.iter().any(|&c| !c) {
        return None;
    }

    // Validate contiguity: each group must be contiguous indices.
    for group in &reassoc {
        for w in group.windows(2) {
            if w[1] != w[0] + 1 {
                return None;
            }
        }
    }

    // Compute the correct inferred output shape from input + reassociation.
    // For collapse_shape, each output dim is:
    //   - product of static input dims if ALL dims in the group are static
    //   - dynamic (None) if ANY dim in the group is dynamic
    let inferred_shape: Vec<Option<u64>> = reassoc
        .iter()
        .map(|group| {
            let mut product = 1u64;
            let mut all_static = true;
            for &i in group {
                match in_shape[i] {
                    Some(d) => product *= d,
                    None => {
                        all_static = false;
                        break;
                    }
                }
            }
            if all_static { Some(product) } else { None }
        })
        .collect();

    Some(ReassocResult::Collapse {
        reassoc,
        inferred_shape,
    })
}

/// Try to build an expand reassociation: groups of contiguous output dims
/// whose products match each input dim.
///
/// Strategy: match static dims from the END first (they anchor the grouping),
/// then assign remaining output dims to the first group.
fn try_expand(in_shape: &[Option<u64>], out_shape: &[Option<u64>]) -> Option<ReassocResult> {
    let in_rank = in_shape.len();
    let out_rank = out_shape.len();

    if in_rank == 0 || out_rank == 0 {
        return None;
    }

    let mut reassoc: Vec<Vec<usize>> = vec![Vec::new(); in_rank];

    // Match from the end: pair input dims with output dims right-to-left.
    let mut in_end = in_rank;
    let mut out_end = out_rank;

    while in_end > 1 && out_end > 0 {
        let in_i = in_end - 1;
        let out_i = out_end - 1;

        match (in_shape[in_i], out_shape[out_i]) {
            (Some(in_val), Some(out_val)) if in_val == out_val => {
                // Static dims match 1:1.
                reassoc[in_i].push(out_i);
                in_end -= 1;
                out_end -= 1;
            }
            (Some(in_val), Some(out_val)) => {
                // Static dims differ — accumulate output dims whose product equals input.
                let mut product = out_val;
                let mut group = vec![out_i];
                let mut j = out_i;
                while product < in_val && j > 0 {
                    j -= 1;
                    match out_shape[j] {
                        Some(d) => {
                            product *= d;
                            group.push(j);
                        }
                        None => {
                            // Dynamic output dim — absorbs remainder.
                            group.push(j);
                            break;
                        }
                    }
                }
                if product == in_val || group.iter().any(|&i| out_shape[i].is_none()) {
                    group.reverse();
                    let first_out = *group.first().unwrap();
                    reassoc[in_i] = group;
                    out_end = first_out;
                    in_end -= 1;
                } else {
                    break;
                }
            }
            (None, None) => {
                // Both dynamic — match 1:1.
                reassoc[in_i].push(out_i);
                in_end -= 1;
                out_end -= 1;
            }
            (Some(_in_val), None) => {
                // Static input, dynamic output — output absorbs input.
                reassoc[in_i].push(out_i);
                in_end -= 1;
                out_end -= 1;
            }
            (None, Some(_)) => {
                // Dynamic input, static output — stop matching from end.
                break;
            }
        }
    }

    // Forward match remaining dims.
    if out_end > 0 && in_end > 0 {
        let mut out_fwd = 0;
        for in_i in 0..in_end {
            match in_shape[in_i] {
                Some(source_val) => {
                    // Static input: accumulate output dims whose product matches.
                    let mut product = 1u64;
                    let mut has_dynamic = false;
                    while out_fwd < out_end {
                        match out_shape[out_fwd] {
                            Some(d) => {
                                reassoc[in_i].push(out_fwd);
                                out_fwd += 1;
                                product *= d;
                                if !has_dynamic && product == source_val {
                                    // If last input group, consume remaining output dims.
                                    if in_i == in_end - 1 {
                                        while out_fwd < out_end {
                                            reassoc[in_i].push(out_fwd);
                                            out_fwd += 1;
                                        }
                                    }
                                    break;
                                }
                                if !has_dynamic && product > source_val {
                                    return None;
                                }
                            }
                            None => {
                                reassoc[in_i].push(out_fwd);
                                out_fwd += 1;
                                #[allow(unused_assignments)]
                                {
                                    has_dynamic = true;
                                }
                                // If last input group, consume all remaining.
                                if in_i == in_end - 1 {
                                    while out_fwd < out_end {
                                        reassoc[in_i].push(out_fwd);
                                        out_fwd += 1;
                                    }
                                }
                                break;
                            }
                        }
                    }
                }
                None => {
                    if in_i == in_end - 1 {
                        // Last input: take all remaining output dims.
                        while out_fwd < out_end {
                            reassoc[in_i].push(out_fwd);
                            out_fwd += 1;
                        }
                    } else {
                        if out_fwd < out_end {
                            reassoc[in_i].push(out_fwd);
                            out_fwd += 1;
                        } else {
                            return None;
                        }
                    }
                }
            }
        }
        if out_fwd != out_end {
            return None;
        }
    } else if out_end > 0 {
        return None;
    }

    // Validate: all groups non-empty, all output dims covered.
    let mut covered = vec![false; out_rank];
    for group in &reassoc {
        if group.is_empty() {
            return None;
        }
        for &i in group {
            if i >= out_rank || covered[i] {
                return None;
            }
            covered[i] = true;
        }
    }
    if covered.iter().any(|&c| !c) {
        return None;
    }

    // Validate contiguity.
    for group in &reassoc {
        for w in group.windows(2) {
            if w[1] != w[0] + 1 {
                return None;
            }
        }
    }

    // Validate expand_shape constraint: if a group has any dynamic output dim,
    // the corresponding input dim must also be dynamic (MLIR verifier requires this).
    for (in_i, group) in reassoc.iter().enumerate() {
        let has_dynamic_out = group.iter().any(|&i| out_shape[i].is_none());
        if has_dynamic_out && in_shape[in_i].is_some() {
            return None;
        }
    }

    Some(ReassocResult::Expand { reassoc })
}

/// Try to decompose a same-rank reshape.
///
/// First tries identity reassociation (1:1 dim mapping) — this works when each
/// dim pair is compatible for collapse_shape (static dims must match).
/// Falls back to collapse-to-1D then expand for truly different shapes.
fn try_collapse_expand(
    in_shape: &[Option<u64>],
    out_shape: &[Option<u64>],
) -> Option<ReassocResult> {
    let in_rank = in_shape.len();
    let out_rank = out_shape.len();

    // Try identity: if shapes already match (possibly with dynamic dims),
    // it's a no-op or a safe type refinement (dynamic→same-dynamic).
    // Note: we do NOT use tensor.cast to refine dynamic→static because
    // the static values from const_i64 may be wrong at runtime (computed
    // from a specific input shape, not the actual runtime shape).
    if in_shape == out_shape {
        return Some(ReassocResult::Identity);
    }

    // Collapse to 1D then expand.
    let n_dyn_in = in_shape.iter().filter(|d| d.is_none()).count();
    let n_dyn_out = out_shape.iter().filter(|d| d.is_none()).count();

    if n_dyn_in > 1 || n_dyn_out > 1 {
        return None;
    }

    let collapse_reassoc = vec![(0..in_rank).collect::<Vec<usize>>()];
    let total: Option<u64> = in_shape.iter().try_fold(1u64, |acc, d| d.map(|n| acc * n));
    let intermediate_shape = vec![total];
    let expand_reassoc = vec![(0..out_rank).collect::<Vec<usize>>()];

    Some(ReassocResult::CollapseExpand {
        collapse_reassoc,
        intermediate_shape,
        expand_reassoc,
    })
}

/// Builds an MLIR computation graph by emitting ops directly into an MLIR
/// function body. Borrows the MLIR `Context` from a `GraphContext`.
pub struct GraphBuilder<'c> {
    context: &'c Context,
    block: Block<'c>,
    location: Location<'c>,
    args: Vec<ArgInfo>,
    /// Completed sub-functions to emit into the MLIR module.
    completed_subfunctions: Vec<CompletedSubFunction<'c>>,
    /// Stashed caller state when building a sub-function (no nesting).
    subfunction_build_state: Option<SubFunctionBuildState<'c>>,
}

impl<'c> GraphBuilder<'c> {
    /// Create a new builder borrowing the given context.
    pub fn new(context: &'c Context) -> Self {
        let location = Location::unknown(context);
        let block = Block::new(&[]);
        Self {
            context,
            block,
            location,
            args: Vec::new(),
            completed_subfunctions: Vec::new(),
            subfunction_build_state: None,
        }
    }

    /// Access the MLIR context.
    pub fn context(&self) -> &'c Context {
        self.context
    }

    /// Build the `fastmath = #arith.fastmath<contract>` attribute pair.
    /// Enables FMA fusion (mul+add → fma) without enabling other fast-math transforms.
    fn fastmath_contract_attr(&self) -> (Identifier<'c>, Attribute<'c>) {
        (
            Identifier::new(self.context, "fastmath"),
            Attribute::parse(self.context, "#arith.fastmath<contract>")
                .expect("#arith.fastmath<contract>"),
        )
    }

    /// Access the MLIR location.
    pub fn location(&self) -> Location<'c> {
        self.location
    }

    /// Access the function body block.
    pub fn block(&self) -> &Block<'c> {
        &self.block
    }

    /// Declare a dynamic/static input tensor.
    ///
    /// `shape`: `Some(n)` for static dims, `None` for dynamic (`?`) dims.
    pub fn input(&mut self, shape: &[Option<u64>], dtype: DType) -> Tensor<'c> {
        self.add_arg(shape, dtype, true)
    }

    /// Declare a weight tensor (large constant passed as function argument).
    pub fn add_weight(&mut self, shape: &[Option<u64>], dtype: DType) -> Tensor<'c> {
        self.add_arg(shape, dtype, false)
    }

    // ── Sub-function API ─────────────────────────────────────────────────────

    /// Start building a new sub-function. `args` are caller-side tensors whose
    /// types define the sub-function's parameters.
    ///
    /// Returns a handle and the sub-function's argument tensors (same types as
    /// `args` but in the sub-function's scope). All subsequent `emit_*` calls
    /// go into the sub-function until `end_subfunction` is called.
    pub fn begin_subfunction(
        &mut self,
        name: &str,
        args: &[&Tensor<'c>],
    ) -> (SubFunctionHandle, Vec<Tensor<'c>>) {
        assert!(
            self.subfunction_build_state.is_none(),
            "nested sub-functions not supported"
        );

        let handle = SubFunctionHandle {
            _index: self.completed_subfunctions.len(),
        };

        // New block with tensor-typed arguments matching caller tensors.
        let new_block = Block::new(&[]);
        let mut sub_arg_tensors = Vec::with_capacity(args.len());

        for &arg in args {
            let tensor_type = arg.value().r#type();
            new_block.add_argument(tensor_type, self.location);
            let idx = new_block.argument_count() - 1;
            sub_arg_tensors.push(Tensor::from_value(new_block.argument(idx).unwrap().into()));
        }

        // Stash caller state.
        let caller_arg_values: Vec<_> = args.iter().map(|t| t.value()).collect();
        let caller_block = std::mem::replace(&mut self.block, new_block);
        let caller_args = std::mem::take(&mut self.args);

        self.subfunction_build_state = Some(SubFunctionBuildState {
            name: name.to_string(),
            caller_block,
            caller_args,
            caller_arg_values,
        });

        (handle, sub_arg_tensors)
    }

    /// Dynamically add an argument to the sub-function being built.
    /// `caller_value` is the tensor in the caller's scope. Returns the
    /// corresponding tensor in the sub-function's scope.
    ///
    /// Use this to route weights directly to sub-functions without threading
    /// them through intermediate sub-functions.
    pub fn add_subfunction_arg(&mut self, caller_value: &Tensor<'c>) -> Tensor<'c> {
        let state = self
            .subfunction_build_state
            .as_mut()
            .expect("add_subfunction_arg called outside of a sub-function");
        let tensor_type = caller_value.value().r#type();
        self.block.add_argument(tensor_type, self.location);
        let idx = self.block.argument_count() - 1;
        state.caller_arg_values.push(caller_value.value());
        Tensor::from_value(self.block.argument(idx).unwrap().into())
    }

    /// Returns true if currently building a sub-function.
    pub fn in_subfunction(&self) -> bool {
        self.subfunction_build_state.is_some()
    }

    /// Number of MLIR operations in the current block (approximate op count).
    pub fn block_op_count(&self) -> usize {
        // melior Block doesn't expose op count directly, but we can iterate.
        let mut count = 0;
        let mut maybe_op = self.block.first_operation();
        while let Some(op) = maybe_op {
            count += 1;
            maybe_op = op.next_in_block();
        }
        count
    }

    /// Finalize the sub-function: add `func.return`, store the completed
    /// sub-function, restore the caller block, emit `func.call`, and return
    /// the call results as tensors in the caller's scope.
    pub fn end_subfunction(
        &mut self,
        _handle: SubFunctionHandle,
        returns: &[&Tensor<'c>],
    ) -> Vec<Tensor<'c>> {
        let state = self
            .subfunction_build_state
            .take()
            .expect("end_subfunction called without matching begin_subfunction");

        // func.return in the sub-function.
        let return_values: Vec<melior::ir::Value> = returns.iter().map(|t| t.value()).collect();
        self.block
            .append_operation(func::r#return(&return_values, self.location));

        // Collect types for the FunctionType.
        let arg_types: Vec<melior::ir::Type<'c>> = (0..self.block.argument_count())
            .map(|i| self.block.argument(i).unwrap().r#type())
            .collect();
        let return_types: Vec<melior::ir::Type<'c>> =
            returns.iter().map(|t| t.value().r#type()).collect();

        // Store completed sub-function and restore caller block.
        let sub_block = std::mem::replace(&mut self.block, state.caller_block);
        self.args = state.caller_args;

        let func_name = state.name.clone();
        self.completed_subfunctions.push(CompletedSubFunction {
            name: state.name,
            block: sub_block,
            arg_types,
            return_types: return_types.clone(),
        });

        // Emit func.call in the caller block.
        let callee_attr =
            Attribute::parse(self.context, &format!("@{func_name}")).expect("callee attr");

        let call_op = OperationBuilder::new("func.call", self.location)
            .add_operands(&state.caller_arg_values)
            .add_results(&return_types)
            .add_attributes(&[(Identifier::new(self.context, "callee"), callee_attr)])
            .build()
            .expect("func.call");

        let call_ref = self.block.append_operation(call_op);

        (0..returns.len())
            .map(|i| Tensor::from_value(call_ref.result(i).unwrap().into()))
            .collect()
    }

    fn make_iterator_types(&self, count: usize) -> Attribute<'c> {
        let entries: Vec<&str> = vec!["#linalg.iterator_type<parallel>"; count];
        let attr_str = format!("[{}]", entries.join(", "));
        Attribute::parse(self.context, &attr_str).expect("iterator_types")
    }

    /// Compile the graph. `outputs` are the tensor values to return.
    pub fn compile(self, outputs: &[&Tensor<'c>]) -> Result<CompiledGraph, CompileError> {
        self.compile_with_cache(outputs, None)
    }

    pub fn finalize_to_mlir_named(
        self,
        outputs: &[&Tensor<'c>],
        transform_mode: TransformMode,
        func_name: &str,
    ) -> Result<(String, usize, Vec<OutputDesc>), CompileError> {
        if outputs.is_empty() {
            return Err(CompileError::NoOutputs);
        }

        let context = self.context;
        let location = self.location;
        let num_args = self.args.len();

        // Build output descriptors.
        let output_descs: Vec<OutputDesc> = outputs
            .iter()
            .map(|t| {
                let shape_vec: Vec<u64> = t
                    .shape()
                    .iter()
                    .map(|d| match d {
                        Some(n) => *n,
                        None => crate::shape::DIM_DYNAMIC,
                    })
                    .collect();
                OutputDesc {
                    shape: crate::Shape(shape_vec),
                    dtype: t.dtype(),
                }
            })
            .collect();

        // Add output memref arguments + bufferization.to_buffer + memref.copy.
        for (out_idx, &output_tensor) in outputs.iter().enumerate() {
            let out_shape = output_tensor.shape();
            let out_dtype = output_tensor.dtype();
            let dims: Vec<i64> = out_shape
                .iter()
                .map(|d| match d {
                    Some(n) => *n as i64,
                    None => i64::MIN,
                })
                .collect();
            let out_memref_type: melior::ir::Type<'c> =
                MemRefType::new(out_dtype.to_mlir_type(context), &dims, None, None).into();

            self.block.add_argument(out_memref_type, location);

            let result_memref: melior::ir::Value = self
                .block
                .append_operation(
                    OperationBuilder::new("bufferization.to_buffer", location)
                        .add_operands(&[output_tensor.value()])
                        .add_results(&[out_memref_type])
                        .build()
                        .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                )
                .result(0)
                .unwrap()
                .into();

            let out_arg_idx = num_args + out_idx;
            let out_arg: melior::ir::Value = self.block.argument(out_arg_idx).unwrap().into();
            self.block.append_operation(
                OperationBuilder::new("memref.copy", location)
                    .add_operands(&[result_memref, out_arg])
                    .build()
                    .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
            );
        }

        // func.return
        self.block.append_operation(func::r#return(&[], location));

        // Build function type.
        let all_arg_types: Vec<melior::ir::Type> = (0..self.block.argument_count())
            .map(|i| self.block.argument(i).unwrap().r#type())
            .collect();
        let function_type = FunctionType::new(context, &all_arg_types, &[]);

        // Build module.
        let mut module = Module::new(location);

        let dl_attr = Attribute::parse(
            context,
            "#dlti.dl_spec<\
                index = 64 : i64, \
                i32 = dense<32> : vector<2xi64>, \
                i64 = dense<64> : vector<2xi64>, \
                f32 = dense<32> : vector<2xi64>, \
                f64 = dense<64> : vector<2xi64>, \
                !llvm.ptr = dense<64> : vector<4xi64>\
            >",
        )
        .expect("failed to parse dlti.dl_spec");
        module
            .as_operation_mut()
            .set_attribute("dlti.dl_spec", dl_attr);

        // Emit sub-functions.
        for sub in self.completed_subfunctions {
            let sub_func_type = FunctionType::new(context, &sub.arg_types, &sub.return_types);
            let sub_region = Region::new();
            sub_region.append_block(sub.block);
            let sub_func = func::func(
                context,
                StringAttribute::new(context, &sub.name),
                TypeAttribute::new(sub_func_type.into()),
                sub_region,
                &[(
                    Identifier::new(context, "llvm.emit_c_interface"),
                    Attribute::unit(context),
                )],
                location,
            );
            module.body().append_operation(sub_func);
        }

        // Build @{func_name}.
        let func_region = Region::new();
        func_region.append_block(self.block);
        let function = func::func(
            context,
            StringAttribute::new(context, func_name),
            TypeAttribute::new(function_type.into()),
            func_region,
            &[(
                Identifier::new(context, "llvm.emit_c_interface"),
                Attribute::unit(context),
            )],
            location,
        );
        module.body().append_operation(function);

        // Attach transform schedule based on mode.
        // - Full: tiling + vectorize + OpenMP (for static-M matmuls).
        // - VectorizeOnly: tile only, no OpenMP. LLVM auto-vectorizes.
        // - TileParallel: forall on N (OpenMP) + for on N+K. LLVM auto-vectorizes.
        // - None: skip schedule entirely (elementwise / BatchNorm kernels).
        match transform_mode {
            TransformMode::Full => {
                module
                    .as_operation_mut()
                    .set_attribute("transform.with_named_sequence", Attribute::unit(context));
                crate::compiler::build_transform_schedule(context, &module, location);
            }
            TransformMode::VectorizeOnly => {
                module
                    .as_operation_mut()
                    .set_attribute("transform.with_named_sequence", Attribute::unit(context));
                // Mark the module so compile_to_objects can select the vectorize-only pipeline.
                module
                    .as_operation_mut()
                    .set_attribute("homura.vectorize_only", Attribute::unit(context));
                crate::compiler::build_vectorize_only_schedule(context, &module, location);
            }
            TransformMode::TileParallel { n_tile } => {
                // Uses the Full pipeline (with OpenMP) but a simpler schedule
                // (forall + for, no pad/outline/vectorize).
                module
                    .as_operation_mut()
                    .set_attribute("transform.with_named_sequence", Attribute::unit(context));
                crate::compiler::build_tile_parallel_schedule(context, &module, location, n_tile);
            }
            TransformMode::None => {}
        }

        // Verify.
        if !module.as_operation().verify() {
            let ir = module.as_operation().to_string();
            let _ = std::fs::write("/tmp/homura_gb_failed.mlir", &ir);
            return Err(CompileError::Verification);
        }

        let mlir_text = module.as_operation().to_string();
        Ok((mlir_text, num_args, output_descs))
    }

    /// Compile with optional cache key.
    pub fn compile_with_cache(
        self,
        outputs: &[&Tensor<'c>],
        cache_key: Option<&str>,
    ) -> Result<CompiledGraph, CompileError> {
        if outputs.is_empty() {
            return Err(CompileError::NoOutputs);
        }

        let context = self.context;
        let location = self.location;
        let num_args = self.args.len();

        // Build output descriptors.
        let output_descs: Vec<OutputDesc> = outputs
            .iter()
            .map(|t| {
                let shape_vec: Vec<u64> = t
                    .shape()
                    .iter()
                    .map(|d| match d {
                        Some(n) => *n,
                        None => crate::shape::DIM_DYNAMIC,
                    })
                    .collect();
                OutputDesc {
                    shape: crate::Shape(shape_vec),
                    dtype: t.dtype(),
                }
            })
            .collect();

        // Add output memref arguments + bufferization.to_buffer + memref.copy.
        for (out_idx, &output_tensor) in outputs.iter().enumerate() {
            let out_shape = output_tensor.shape();
            let out_dtype = output_tensor.dtype();
            let dims: Vec<i64> = out_shape
                .iter()
                .map(|d| match d {
                    Some(n) => *n as i64,
                    None => i64::MIN,
                })
                .collect();
            let out_memref_type: melior::ir::Type<'c> =
                MemRefType::new(out_dtype.to_mlir_type(context), &dims, None, None).into();

            self.block.add_argument(out_memref_type, location);

            let result_memref: melior::ir::Value = self
                .block
                .append_operation(
                    OperationBuilder::new("bufferization.to_buffer", location)
                        .add_operands(&[output_tensor.value()])
                        .add_results(&[out_memref_type])
                        .build()
                        .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                )
                .result(0)
                .unwrap()
                .into();

            let out_arg_idx = num_args + out_idx;
            let out_arg: melior::ir::Value = self.block.argument(out_arg_idx).unwrap().into();
            self.block.append_operation(
                OperationBuilder::new("memref.copy", location)
                    .add_operands(&[result_memref, out_arg])
                    .build()
                    .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
            );
        }

        // func.return
        self.block.append_operation(func::r#return(&[], location));

        // Build function type: all block args → void.
        let all_arg_types: Vec<melior::ir::Type> = (0..self.block.argument_count())
            .map(|i| self.block.argument(i).unwrap().r#type())
            .collect();
        let function_type = FunctionType::new(context, &all_arg_types, &[]);

        // Build module.
        let mut module = Module::new(location);

        // Set data layout.
        let dl_attr = Attribute::parse(
            context,
            "#dlti.dl_spec<\
                index = 64 : i64, \
                i32 = dense<32> : vector<2xi64>, \
                i64 = dense<64> : vector<2xi64>, \
                f32 = dense<32> : vector<2xi64>, \
                f64 = dense<64> : vector<2xi64>, \
                !llvm.ptr = dense<64> : vector<4xi64>\
            >",
        )
        .expect("failed to parse dlti.dl_spec");
        module
            .as_operation_mut()
            .set_attribute("dlti.dl_spec", dl_attr);

        // Emit sub-functions before @compute.
        for sub in self.completed_subfunctions {
            let sub_func_type = FunctionType::new(context, &sub.arg_types, &sub.return_types);
            let sub_region = Region::new();
            sub_region.append_block(sub.block);

            let sub_func = func::func(
                context,
                StringAttribute::new(context, &sub.name),
                TypeAttribute::new(sub_func_type.into()),
                sub_region,
                &[(
                    Identifier::new(context, "llvm.emit_c_interface"),
                    Attribute::unit(context),
                )],
                location,
            );
            module.body().append_operation(sub_func);
        }

        // Build func.func @compute.
        let func_region = Region::new();
        func_region.append_block(self.block);

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

        // Attach the transform schedule so transform-interpreter can tile
        // and vectorize linalg.generic contraction ops.
        module
            .as_operation_mut()
            .set_attribute("transform.with_named_sequence", Attribute::unit(context));
        crate::compiler::build_transform_schedule(context, &module, location);

        // Dump pre-pass IR if requested.
        if std::env::var("HOMURA_DUMP_IR").is_ok() {
            let _ = std::fs::write(
                "/tmp/homura_gb_pre_passes.mlir",
                module.as_operation().to_string(),
            );
            log_debug!("GraphBuilder pre-pass IR dumped to /tmp/homura_gb_pre_passes.mlir");
        }

        // Verify.
        if !module.as_operation().verify() {
            let ir = module.as_operation().to_string();
            let _ = std::fs::write("/tmp/homura_gb_failed.mlir", &ir);
            log_warn!(
                "GraphBuilder MLIR verification failed — IR dumped to /tmp/homura_gb_failed.mlir"
            );
            return Err(CompileError::Verification);
        }

        // Cache check.
        if let Some(key) = cache_key {
            let cache = crate::cache::CompilationCache::new();
            if let Some((so_path, meta_path)) = cache.get(key)
                && let Some(meta) = crate::cache::CompilationCache::load_meta(&meta_path)
            {
                match CompiledGraph::load(&so_path, meta.num_inputs, meta.outputs) {
                    Ok(graph) => return Ok(graph),
                    Err(e) => {
                        log_warn!(
                            "cache entry unloadable, recompiling {}: {e}",
                            so_path.display()
                        );
                    }
                }
            }
        }

        // Run lowering passes: transform schedule + vectorization + bufferization.
        let pipeline_start = std::time::Instant::now();
        log_compile!("pipeline", "starting MLIR passes...");
        register_all_passes();
        let pass_manager = pass::PassManager::new(context);
        parse_pass_pipeline(
            pass_manager.as_operation_pass_manager(),
            "builtin.module(\
                func.func(canonicalize,cse),\
                transform-interpreter,\
                func.func(canonicalize,cse),\
                symbol-dce,\
                func.func(lower-vector-multi-reduction,lower-vector-mask),\
                one-shot-bufferize{bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map unknown-type-conversion=identity-layout-map},\
                func.func(buffer-hoisting,promote-buffers-to-stack{max-alloc-size-in-bytes=4096}),\
            ownership-based-buffer-deallocation,buffer-deallocation-simplification,convert-bufferization-to-memref,canonicalize,\
                scf-forall-to-parallel,\
                fold-memref-alias-ops,\
                convert-vector-to-scf,\
                convert-linalg-to-loops,\
                fold-memref-alias-ops,\
                lower-affine,\
                convert-scf-to-openmp,\
                convert-openmp-to-llvm,\
                convert-scf-to-cf,\
                canonicalize,\
                cse,\
                sccp,\
                convert-vector-to-llvm{vector-contract-lowering=outerproduct},\
                convert-ub-to-llvm,\
                convert-math-to-llvm,\
                expand-strided-metadata,\
                lower-affine,\
                finalize-memref-to-llvm,\
                convert-arith-to-llvm,\
                convert-index-to-llvm,\
                convert-cf-to-llvm,\
                convert-func-to-llvm,\
                convert-openmp-to-llvm,\
                reconcile-unrealized-casts\
            )",
        )
        .map_err(CompileError::Pass)?;

        pass_manager.run(&mut module).map_err(CompileError::Pass)?;
        log_compile!(
            "pipeline",
            "MLIR passes done: {}ms",
            pipeline_start.elapsed().as_millis()
        );

        if std::env::var("HOMURA_DUMP_IR").is_ok() {
            let _ = std::fs::write(
                "/tmp/homura_gb_post_passes.mlir",
                module.as_operation().to_string(),
            );
            log_debug!("GraphBuilder post-pass IR dumped to /tmp/homura_gb_post_passes.mlir");
        }

        // AOT compile.
        let tmp_dir = tempfile_dir()
            .ok_or_else(|| CompileError::ObjectEmit("cannot determine temp directory".into()))?;
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos();
        let suffix = format!("{}_{:08x}", std::process::id(), nanos);
        let tmp_so = tmp_dir.join(format!("homura_gb_{suffix}.so"));

        let obj_paths = crate::compiler::emit_object_files_pub(
            &module,
            &tmp_dir,
            &format!("homura_gb_{suffix}"),
            "gb",
            "compute",
        )?;
        crate::compiler::link_shared_lib_pub(&obj_paths, &tmp_so)?;
        for p in &obj_paths {
            std::fs::remove_file(p).ok();
        }

        // Store in cache.
        if let Some(key) = cache_key {
            let cache = crate::cache::CompilationCache::new();
            let meta = crate::cache::CacheMeta {
                num_inputs: num_args,
                outputs: output_descs
                    .iter()
                    .map(|d| OutputDesc {
                        shape: d.shape.clone(),
                        dtype: d.dtype,
                    })
                    .collect(),
            };
            if let Err(e) = cache.store(key, &tmp_so, &meta) {
                log_warn!("cache: failed to write cache entry: {e}");
            }
        }

        let graph = CompiledGraph::load(&tmp_so, num_args, output_descs)
            .map_err(CompileError::ObjectEmit)?;
        std::fs::remove_file(&tmp_so).ok();

        Ok(graph)
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    fn add_arg(&mut self, shape: &[Option<u64>], dtype: DType, is_input: bool) -> Tensor<'c> {
        let dims: Vec<i64> = shape
            .iter()
            .map(|d| match d {
                Some(n) => *n as i64,
                None => i64::MIN,
            })
            .collect();
        let elem_type = dtype.to_mlir_type(self.context);

        // RankedTensorType uses u64 dims where kDynamic = i64::MIN as u64.
        let dims_u64: Vec<u64> = dims.iter().map(|&d| d as u64).collect();
        let tensor_type: melior::ir::Type =
            RankedTensorType::new(&dims_u64, elem_type, None).into();
        let memref_type: melior::ir::Type = MemRefType::new(elem_type, &dims, None, None).into();

        let arg_idx = self.block.argument_count();
        self.block.add_argument(memref_type, self.location);
        let memref_arg = self.block.argument(arg_idx).unwrap();

        // bufferization.to_tensor %memref {restrict}
        let tensor_val = self
            .block
            .append_operation(
                OperationBuilder::new("bufferization.to_tensor", self.location)
                    .add_operands(&[memref_arg.into()])
                    .add_results(&[tensor_type])
                    .add_attributes(&[(
                        Identifier::new(self.context, "restrict"),
                        Attribute::unit(self.context),
                    )])
                    .build()
                    .expect("bufferization.to_tensor"),
            )
            .result(0)
            .unwrap()
            .into();

        self.args.push(ArgInfo {
            _shape: shape.to_vec(),
            _dtype: dtype,
            _is_input: is_input,
        });

        Tensor::from_value(tensor_val)
    }

    // ── Meta ops (shape subgraph) ─────────────────────────────────────────────

    /// Emit `tensor.dim` for each dimension of `input`, returning a list of
    /// runtime `index` values. Static dims become `arith.constant`, dynamic
    /// dims become `tensor.dim` results.
    pub fn emit_shape_of(&mut self, input: &Tensor<'c>) -> Vec<melior::ir::Value<'c, 'c>> {
        let shape = input.shape();
        let index_type = melior::ir::Type::parse(self.context, "index").expect("index type");
        shape
            .iter()
            .enumerate()
            .map(|(i, dim)| match dim {
                Some(n) => {
                    let attr = Attribute::parse(self.context, &format!("{n} : index"))
                        .expect("index constant attr");
                    self.block
                        .append_operation(
                            OperationBuilder::new("arith.constant", self.location)
                                .add_results(&[index_type])
                                .add_attributes(&[(Identifier::new(self.context, "value"), attr)])
                                .build()
                                .expect("arith.constant index"),
                        )
                        .result(0)
                        .unwrap()
                        .into()
                }
                None => self.emit_tensor_dim(input.value(), i),
            })
            .collect()
    }

    /// Create a tensor filled with `fill_value`, with shape given by runtime
    /// `index` values. Equivalent to ONNX ConstantOfShape.
    ///
    /// `shape_vals` is a list of runtime `index` values (from `emit_shape_of`
    /// or `arith.constant`). `fill_value` is the scalar fill.
    pub fn emit_constant_of_shape(
        &mut self,
        shape_vals: &[melior::ir::Value<'c, 'c>],
        static_shape: &[Option<u64>],
        fill_value: f64,
        dtype: DType,
    ) -> Tensor<'c> {
        let elem_type = dtype.to_mlir_type(self.context);
        let tensor_type = self.make_tensor_type(static_shape, dtype);

        // tensor.empty with dynamic operands for None dims.
        let dyn_vals: Vec<melior::ir::Value<'c, 'c>> = static_shape
            .iter()
            .enumerate()
            .filter(|(_, d)| d.is_none())
            .map(|(i, _)| shape_vals[i])
            .collect();

        let init: melior::ir::Value = self
            .block
            .append_operation(
                OperationBuilder::new("tensor.empty", self.location)
                    .add_operands(&dyn_vals)
                    .add_results(&[tensor_type])
                    .build()
                    .expect("tensor.empty constant_of_shape"),
            )
            .result(0)
            .unwrap()
            .into();

        // Scalar fill value.
        let scalar_attr = self.make_scalar_attr(fill_value, dtype);
        let scalar: melior::ir::Value = self
            .block
            .append_operation(
                OperationBuilder::new("arith.constant", self.location)
                    .add_results(&[elem_type])
                    .add_attributes(&[(Identifier::new(self.context, "value"), scalar_attr)])
                    .build()
                    .expect("arith.constant fill"),
            )
            .result(0)
            .unwrap()
            .into();

        // linalg.fill
        let fill_block = Block::new(&[(elem_type, self.location), (elem_type, self.location)]);
        let fill_in: melior::ir::Value = fill_block.argument(0).unwrap().into();
        fill_block.append_operation(
            OperationBuilder::new("linalg.yield", self.location)
                .add_operands(&[fill_in])
                .build()
                .expect("linalg.yield fill"),
        );
        let fill_region = Region::new();
        fill_region.append_block(fill_block);

        let filled: melior::ir::Value = self
            .block
            .append_operation(
                OperationBuilder::new("linalg.fill", self.location)
                    .add_operands(&[scalar, init])
                    .add_results(&[tensor_type])
                    .add_attributes(&[(
                        Identifier::new(self.context, "operandSegmentSizes"),
                        Attribute::parse(self.context, "array<i32: 1, 1>").unwrap(),
                    )])
                    .add_regions([fill_region])
                    .build()
                    .expect("linalg.fill constant_of_shape"),
            )
            .result(0)
            .unwrap()
            .into();

        Tensor::from_value(filled)
    }

    /// Emit a Range tensor: `[start, start+delta, start+2*delta, ...]` up to
    /// (not including) `limit`. All are scalar `index` values.
    ///
    /// Output is a 1-D tensor. If the range size is statically known, the
    /// output has a static dim; otherwise it's dynamic.
    pub fn emit_range(
        &mut self,
        start: melior::ir::Value<'c, 'c>,
        limit: melior::ir::Value<'c, 'c>,
        delta: melior::ir::Value<'c, 'c>,
        output_size: Option<u64>,
        dtype: DType,
    ) -> Tensor<'c> {
        let elem_type = dtype.to_mlir_type(self.context);
        let index_type = melior::ir::Type::parse(self.context, "index").expect("index type");
        let shape = vec![output_size];
        let tensor_type = self.make_tensor_type(&shape, dtype);

        // Compute output size if dynamic: (limit - start + delta - 1) / delta
        // or just use the provided size.
        let mut dyn_vals = Vec::new();
        if output_size.is_none() {
            // Compute runtime size: ceildiv(limit - start, delta)
            let diff: melior::ir::Value = self
                .block
                .append_operation(
                    OperationBuilder::new("arith.subi", self.location)
                        .add_operands(&[limit, start])
                        .add_results(&[index_type])
                        .build()
                        .expect("arith.subi range"),
                )
                .result(0)
                .unwrap()
                .into();
            let size: melior::ir::Value = self
                .block
                .append_operation(
                    OperationBuilder::new("arith.ceildivui", self.location)
                        .add_operands(&[diff, delta])
                        .add_results(&[index_type])
                        .build()
                        .expect("arith.ceildivui range"),
                )
                .result(0)
                .unwrap()
                .into();
            dyn_vals.push(size);
        }

        let init: melior::ir::Value = self
            .block
            .append_operation(
                OperationBuilder::new("tensor.empty", self.location)
                    .add_operands(&dyn_vals)
                    .add_results(&[tensor_type])
                    .build()
                    .expect("tensor.empty range"),
            )
            .result(0)
            .unwrap()
            .into();

        // Cast start and delta from index to element type (they may come from
        // extract_scalar_as_index which returns index values).
        let start_elem = self.emit_index_to_elem(start, dtype);
        let delta_elem = self.emit_index_to_elem(delta, dtype);

        // linalg.generic: body computes start + index * delta
        let identity = identity_map_str(1);
        let indexing_maps =
            Attribute::parse(self.context, &format!("[{identity}]")).expect("range indexing_maps");
        let iterator_types = self.make_iterator_types(1);

        let body_block = Block::new(&[(elem_type, self.location)]);

        // Get iteration index.
        let idx: melior::ir::Value = body_block
            .append_operation(
                OperationBuilder::new("linalg.index", self.location)
                    .add_attributes(&[(
                        Identifier::new(self.context, "dim"),
                        Attribute::parse(self.context, "0 : i64").unwrap(),
                    )])
                    .add_results(&[index_type])
                    .build()
                    .expect("linalg.index"),
            )
            .result(0)
            .unwrap()
            .into();

        // Cast index to element type, then compute start + idx * delta.
        let idx_cast = self.emit_index_to_elem_in_block(&body_block, idx, dtype);
        let is_float = matches!(dtype, DType::F32 | DType::F64);
        let mut mul_builder = OperationBuilder::new(
            if is_float { "arith.mulf" } else { "arith.muli" },
            self.location,
        )
        .add_operands(&[idx_cast, delta_elem])
        .add_results(&[elem_type]);
        if is_float {
            mul_builder = mul_builder.add_attributes(&[self.fastmath_contract_attr()]);
        }
        let prod: melior::ir::Value = body_block
            .append_operation(mul_builder.build().expect("range mul"))
            .result(0)
            .unwrap()
            .into();
        let mut add_builder = OperationBuilder::new(
            if is_float { "arith.addf" } else { "arith.addi" },
            self.location,
        )
        .add_operands(&[start_elem, prod])
        .add_results(&[elem_type]);
        if is_float {
            add_builder = add_builder.add_attributes(&[self.fastmath_contract_attr()]);
        }
        let val: melior::ir::Value = body_block
            .append_operation(add_builder.build().expect("range add"))
            .result(0)
            .unwrap()
            .into();

        body_block.append_operation(
            OperationBuilder::new("linalg.yield", self.location)
                .add_operands(&[val])
                .build()
                .expect("linalg.yield"),
        );
        let body_region = Region::new();
        body_region.append_block(body_block);

        let result: melior::ir::Value = self
            .block
            .append_operation(
                OperationBuilder::new("linalg.generic", self.location)
                    .add_operands(&[init])
                    .add_results(&[tensor_type])
                    .add_attributes(&[
                        (
                            Identifier::new(self.context, "indexing_maps"),
                            indexing_maps,
                        ),
                        (
                            Identifier::new(self.context, "iterator_types"),
                            iterator_types,
                        ),
                        (
                            Identifier::new(self.context, "operandSegmentSizes"),
                            Attribute::parse(self.context, "array<i32: 0, 1>").unwrap(),
                        ),
                    ])
                    .add_regions([body_region])
                    .build()
                    .expect("linalg.generic range"),
            )
            .result(0)
            .unwrap()
            .into();

        Tensor::from_value(result)
    }

    /// Cast an `index` value to the target element type inside a block body.
    /// Cast an `index` value to the target element type (on the main block).
    fn emit_index_to_elem(
        &mut self,
        idx: melior::ir::Value<'c, 'c>,
        dtype: DType,
    ) -> melior::ir::Value<'c, 'c> {
        let elem_type = dtype.to_mlir_type(self.context);
        let i64_type = melior::ir::Type::parse(self.context, "i64").expect("i64 type");
        let as_i64: melior::ir::Value = self
            .block
            .append_operation(
                OperationBuilder::new("arith.index_cast", self.location)
                    .add_operands(&[idx])
                    .add_results(&[i64_type])
                    .build()
                    .expect("arith.index_cast"),
            )
            .result(0)
            .unwrap()
            .into();
        match dtype {
            DType::I64 => as_i64,
            DType::I8 | DType::I16 | DType::I32 => self
                .block
                .append_operation(
                    OperationBuilder::new("arith.trunci", self.location)
                        .add_operands(&[as_i64])
                        .add_results(&[elem_type])
                        .build()
                        .expect("arith.trunci"),
                )
                .result(0)
                .unwrap()
                .into(),
            DType::F16 | DType::F32 | DType::F64 | DType::BF16 => self
                .block
                .append_operation(
                    OperationBuilder::new("arith.sitofp", self.location)
                        .add_operands(&[as_i64])
                        .add_results(&[elem_type])
                        .build()
                        .expect("arith.sitofp"),
                )
                .result(0)
                .unwrap()
                .into(),
            dt => unreachable!("unsupported dtype {:?} for emit_index_to_elem", dt),
        }
    }

    fn emit_index_to_elem_in_block(
        &self,
        block: &Block<'c>,
        idx: melior::ir::Value<'c, 'c>,
        dtype: DType,
    ) -> melior::ir::Value<'c, 'c> {
        let elem_type = dtype.to_mlir_type(self.context);
        let i64_type = melior::ir::Type::parse(self.context, "i64").expect("i64 type");
        // index -> i64
        let as_i64: melior::ir::Value = block
            .append_operation(
                OperationBuilder::new("arith.index_cast", self.location)
                    .add_operands(&[idx])
                    .add_results(&[i64_type])
                    .build()
                    .expect("arith.index_cast"),
            )
            .result(0)
            .unwrap()
            .into();
        match dtype {
            DType::I64 => as_i64,
            DType::I8 | DType::I16 | DType::I32 => block
                .append_operation(
                    OperationBuilder::new("arith.trunci", self.location)
                        .add_operands(&[as_i64])
                        .add_results(&[elem_type])
                        .build()
                        .expect("arith.trunci"),
                )
                .result(0)
                .unwrap()
                .into(),
            DType::F16 | DType::F32 | DType::F64 | DType::BF16 => block
                .append_operation(
                    OperationBuilder::new("arith.sitofp", self.location)
                        .add_operands(&[as_i64])
                        .add_results(&[elem_type])
                        .build()
                        .expect("arith.sitofp"),
                )
                .result(0)
                .unwrap()
                .into(),
            dt => unreachable!("unsupported dtype {:?} for emit_index_to_elem_in_block", dt),
        }
    }

    /// Emit an `arith.constant` scalar tensor value.
    pub fn emit_arith_constant(&mut self, value: f64, dtype: DType) -> Tensor<'c> {
        let dense_str = match dtype {
            DType::F16 => format!("dense<{:.6e}> : tensor<f16>", value),
            DType::F32 => format!("dense<{:.6e}> : tensor<f32>", value),
            DType::F64 => format!("dense<{:.15e}> : tensor<f64>", value),
            DType::BF16 => format!("dense<{:.6e}> : tensor<bf16>", value),
            DType::I8 => format!("dense<{}> : tensor<i8>", value as i8),
            DType::I16 => format!("dense<{}> : tensor<i16>", value as i16),
            DType::I32 => format!("dense<{}> : tensor<i32>", value as i32),
            DType::I64 => format!("dense<{}> : tensor<i64>", value as i64),
            dt => unreachable!("unsupported dtype {:?} for emit_arith_constant", dt),
        };
        let dense_attr = Attribute::parse(self.context, &dense_str).expect("dense constant attr");
        let tensor_type = self.make_tensor_type(&[], dtype);
        let result: melior::ir::Value = self
            .block
            .append_operation(
                OperationBuilder::new("arith.constant", self.location)
                    .add_results(&[tensor_type])
                    .add_attributes(&[(Identifier::new(self.context, "value"), dense_attr)])
                    .build()
                    .expect("arith.constant dense"),
            )
            .result(0)
            .unwrap()
            .into();
        Tensor::from_value(result)
    }

    /// Emit a dense tensor constant from raw data.
    ///
    /// `data_str` is the MLIR dense attribute body, e.g. `"[1.0, 2.0, 3.0]"`
    /// or `"[[1, 2], [3, 4]]"`.
    pub fn emit_dense_constant(
        &mut self,
        data_str: &str,
        shape: &[u64],
        dtype: DType,
    ) -> Tensor<'c> {
        let shape_opt: Vec<Option<u64>> = shape.iter().map(|&d| Some(d)).collect();
        let tensor_type = self.make_tensor_type(&shape_opt, dtype);
        let type_str = tensor_type.to_string();
        let dense_str = format!("dense<{data_str}> : {type_str}");
        let dense_attr = Attribute::parse(self.context, &dense_str)
            .unwrap_or_else(|| panic!("failed to parse dense attr: {dense_str}"));
        let result: melior::ir::Value = self
            .block
            .append_operation(
                OperationBuilder::new("arith.constant", self.location)
                    .add_results(&[tensor_type])
                    .add_attributes(&[(Identifier::new(self.context, "value"), dense_attr)])
                    .build()
                    .expect("arith.constant dense tensor"),
            )
            .result(0)
            .unwrap()
            .into();
        Tensor::from_value(result)
    }

    /// Build a scalar attribute for the given value and dtype.
    fn make_scalar_attr(&self, value: f64, dtype: DType) -> Attribute<'c> {
        let s = match dtype {
            DType::F16 => format!("{:.6e} : f16", value),
            DType::F32 => format!("{:.6e} : f32", value),
            DType::F64 => format!("{:.15e} : f64", value),
            DType::BF16 => format!("{:.6e} : bf16", value),
            DType::I8 => format!("{} : i8", value as i8),
            DType::I16 => format!("{} : i16", value as i16),
            DType::I32 => format!("{} : i32", value as i32),
            DType::I64 => format!("{} : i64", value as i64),
            dt => unreachable!("unsupported dtype {:?} for make_scalar_attr", dt),
        };
        Attribute::parse(self.context, &s).expect("scalar attr")
    }

    // ── Type construction helpers ─────────────────────────────────────────────

    /// Build a `RankedTensorType` from a shape with optional dims.
    fn make_tensor_type(&self, shape: &[Option<u64>], dtype: DType) -> melior::ir::Type<'c> {
        let dims_u64: Vec<u64> = shape
            .iter()
            .map(|d| match d {
                Some(n) => *n,
                None => i64::MIN as u64,
            })
            .collect();
        let elem_type = dtype.to_mlir_type(self.context);
        RankedTensorType::new(&dims_u64, elem_type, None).into()
    }

    /// Read the DType from an MLIR `Value`'s `RankedTensorType`.
    fn value_dtype(&self, val: melior::ir::Value<'c, 'c>) -> DType {
        let rtt = RankedTensorType::try_from(val.r#type())
            .expect("value_dtype: expected RankedTensorType");
        mlir_element_type_to_dtype(rtt.element())
    }

    // ── MLIR emission helpers ─────────────────────────────────────────────────

    /// Emit `arith.constant <n> : index`.
    fn emit_arith_constant_index(&mut self, n: u64) -> melior::ir::Value<'c, 'c> {
        let index_type = melior::ir::Type::parse(self.context, "index").expect("index type");
        let attr =
            Attribute::parse(self.context, &format!("{n} : index")).expect("index const attr");
        self.block
            .append_operation(
                OperationBuilder::new("arith.constant", self.location)
                    .add_results(&[index_type])
                    .add_attributes(&[(Identifier::new(self.context, "value"), attr)])
                    .build()
                    .expect("arith.constant index"),
            )
            .result(0)
            .unwrap()
            .into()
    }

    /// Emit `tensor.cast` for same-rank type refinement (e.g. dynamic→static dims).
    fn emit_tensor_cast(
        &mut self,
        input: melior::ir::Value<'c, 'c>,
        tgt_shape: &[Option<u64>],
    ) -> Tensor<'c> {
        let elem_type = {
            let rtt = RankedTensorType::try_from(input.r#type())
                .expect("tensor.cast input must be RankedTensorType");
            rtt.element()
        };
        let dtype = mlir_element_type_to_dtype(elem_type);
        let out_type = self.make_tensor_type(tgt_shape, dtype);
        let result: melior::ir::Value = self
            .block
            .append_operation(
                OperationBuilder::new("tensor.cast", self.location)
                    .add_operands(&[input])
                    .add_results(&[out_type])
                    .build()
                    .expect("tensor.cast"),
            )
            .result(0)
            .unwrap()
            .into();
        Tensor::from_value(result)
    }

    /// Emit `tensor.empty` for the given shape and dtype.
    ///
    /// For each `None` (dynamic) dim in `shape`, a runtime `index` value is
    /// needed. If `dyn_source` is provided, `tensor.dim` ops are emitted to
    /// read the corresponding dimension from that tensor. The source tensor
    /// must have the same rank and the `None` dims must appear at the same
    /// positions.
    ///
    /// If `dyn_source` is `None` and any dim is dynamic, this panics.
    /// Like `emit_tensor_empty` but with an explicit dynamic-dim source.
    fn emit_tensor_empty_dyn(
        &mut self,
        shape: &[Option<u64>],
        dtype: DType,
        dyn_source: Option<melior::ir::Value<'c, 'c>>,
    ) -> melior::ir::Value<'c, 'c> {
        let tensor_type = self.make_tensor_type(shape, dtype);

        let mut dyn_vals: Vec<melior::ir::Value<'c, 'c>> = Vec::new();
        for (i, dim) in shape.iter().enumerate() {
            if dim.is_none() {
                let src = dyn_source.unwrap_or_else(|| {
                    panic!(
                        "emit_tensor_empty: dim {i} is dynamic but no dyn_source provided (shape: {shape:?})"
                    )
                });
                let dim_val = self.emit_tensor_dim(src, i);
                dyn_vals.push(dim_val);
            }
        }

        self.block
            .append_operation(
                OperationBuilder::new("tensor.empty", self.location)
                    .add_operands(&dyn_vals)
                    .add_results(&[tensor_type])
                    .build()
                    .expect("tensor.empty"),
            )
            .result(0)
            .unwrap()
            .into()
    }

    /// Like `emit_tensor_empty_dyn` but output dim `i` reads source dim `dim_map[i]`.
    fn emit_tensor_empty_with_dim_map(
        &mut self,
        shape: &[Option<u64>],
        dtype: DType,
        source: melior::ir::Value<'c, 'c>,
        dim_map: &[usize],
    ) -> melior::ir::Value<'c, 'c> {
        let tensor_type = self.make_tensor_type(shape, dtype);

        let mut dyn_vals: Vec<melior::ir::Value<'c, 'c>> = Vec::new();
        for (i, dim) in shape.iter().enumerate() {
            if dim.is_none() {
                let src_dim = dim_map[i];
                let dim_val = self.emit_tensor_dim(source, src_dim);
                dyn_vals.push(dim_val);
            }
        }

        self.block
            .append_operation(
                OperationBuilder::new("tensor.empty", self.location)
                    .add_operands(&dyn_vals)
                    .add_results(&[tensor_type])
                    .build()
                    .expect("tensor.empty"),
            )
            .result(0)
            .unwrap()
            .into()
    }

    /// Emit `tensor.dim %tensor, %c_idx : tensor<...>` to get a runtime dim size.
    fn emit_tensor_dim(
        &mut self,
        tensor: melior::ir::Value<'c, 'c>,
        dim_idx: usize,
    ) -> melior::ir::Value<'c, 'c> {
        let index_type = melior::ir::Type::parse(self.context, "index").expect("index type");
        let idx_attr =
            Attribute::parse(self.context, &format!("{dim_idx} : index")).expect("dim index attr");
        let idx_val: melior::ir::Value = self
            .block
            .append_operation(
                OperationBuilder::new("arith.constant", self.location)
                    .add_results(&[index_type])
                    .add_attributes(&[(Identifier::new(self.context, "value"), idx_attr)])
                    .build()
                    .expect("arith.constant index"),
            )
            .result(0)
            .unwrap()
            .into();
        self.block
            .append_operation(
                OperationBuilder::new("tensor.dim", self.location)
                    .add_operands(&[tensor, idx_val])
                    .add_results(&[index_type])
                    .build()
                    .expect("tensor.dim"),
            )
            .result(0)
            .unwrap()
            .into()
    }

    /// Extract a scalar element from a 1-D tensor at a static position.
    ///
    /// Returns the element as an MLIR scalar value with the tensor's element type.
    /// `pos` must be in bounds.
    pub fn emit_tensor_extract_scalar(
        &mut self,
        tensor: &Tensor<'c>,
        pos: usize,
    ) -> melior::ir::Value<'c, 'c> {
        let elem_type = tensor.dtype().to_mlir_type(self.context);
        let index_type = melior::ir::Type::parse(self.context, "index").expect("index type");

        // Build arith.constant for the position index.
        let idx_attr =
            Attribute::parse(self.context, &format!("{pos} : index")).expect("index attr");
        let idx_val: melior::ir::Value<'c, 'c> = self
            .block
            .append_operation(
                OperationBuilder::new("arith.constant", self.location)
                    .add_results(&[index_type])
                    .add_attributes(&[(Identifier::new(self.context, "value"), idx_attr)])
                    .build()
                    .expect("arith.constant index"),
            )
            .result(0)
            .unwrap()
            .into();

        self.block
            .append_operation(
                OperationBuilder::new("tensor.extract", self.location)
                    .add_operands(&[tensor.value(), idx_val])
                    .add_results(&[elem_type])
                    .build()
                    .expect("tensor.extract scalar"),
            )
            .result(0)
            .unwrap()
            .into()
    }
}

// ── Shared utilities ──────────────────────────────────────────────────────────

/// Register an atexit handler that calls `_exit(0)` to skip LLVM shared lib
/// destructors (known double-free bug in teardown). Safe to call multiple
/// times — only registers once.
pub(crate) fn register_force_exit() {
    static REGISTER: std::sync::Once = std::sync::Once::new();
    REGISTER.call_once(|| {
        extern "C" fn force_exit() {
            unsafe { libc::_exit(0) }
        }
        unsafe {
            libc::atexit(force_exit);
        }
    });
}

fn create_context() -> Context {
    register_force_exit();

    let context = Context::new();
    context.attach_diagnostic_handler(|diagnostic| {
        eprintln!("[mlir] {diagnostic}");
        true
    });
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    register_all_llvm_translations(&context);
    context
}

pub(super) fn identity_map_str(rank: usize) -> String {
    let dims: Vec<String> = (0..rank).map(|i| format!("d{i}")).collect();
    let dim_list = dims.join(", ");
    format!("affine_map<({dim_list}) -> ({dim_list})>")
}

pub fn tempfile_dir() -> Option<std::path::PathBuf> {
    std::env::var("TMPDIR")
        .ok()
        .map(std::path::PathBuf::from)
        .or_else(|| Some(std::path::PathBuf::from("/tmp")))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Buffer;

    #[test]
    fn tensor_shape_static() {
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let t = gb.input(&[Some(2), Some(3)], DType::F32);
        assert_eq!(t.shape(), vec![Some(2), Some(3)]);
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.rank(), 2);
    }

    #[test]
    fn tensor_shape_dynamic() {
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let t = gb.input(&[None, Some(12), None, Some(64)], DType::F32);
        assert_eq!(t.shape(), vec![None, Some(12), None, Some(64)]);
        assert_eq!(t.rank(), 4);
    }

    #[test]
    fn identity_graph_compile_and_run() {
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let x = gb.input(&[Some(4)], DType::F32);
        let graph = gb.compile(&[&x]).expect("compile failed");

        let input = Buffer::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], DType::F32);
        let outputs = graph.run(&[&input]);
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].as_slice::<f32>(), &[1.0, 2.0, 3.0, 4.0]);
    }

    // ── Elementwise ops (tasks 2.6, 2.7) ──────────────────────────────────

    #[test]
    fn elementwise_add_static() {
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(4)], DType::F32);
        let b = gb.input(&[Some(4)], DType::F32);
        let c = gb.emit_add(&a, &b);
        let graph = gb.compile(&[&c]).expect("compile add");

        let a_buf = Buffer::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], DType::F32);
        let b_buf = Buffer::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[4], DType::F32);
        let out = graph.run(&[&a_buf, &b_buf]);
        assert_eq!(out[0].as_slice::<f32>(), &[11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn elementwise_sub_mul() {
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(3)], DType::F32);
        let b = gb.input(&[Some(3)], DType::F32);
        let diff = gb.emit_sub(&a, &b);
        let prod = gb.emit_mul(&a, &b);
        let graph = gb.compile(&[&diff, &prod]).expect("compile sub+mul");

        let a_buf = Buffer::from_slice(&[5.0f32, 10.0, 15.0], &[3], DType::F32);
        let b_buf = Buffer::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
        let out = graph.run(&[&a_buf, &b_buf]);
        assert_eq!(out[0].as_slice::<f32>(), &[4.0, 8.0, 12.0]);
        assert_eq!(out[1].as_slice::<f32>(), &[5.0, 20.0, 45.0]);
    }

    #[test]
    fn elementwise_div() {
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(4)], DType::F32);
        let b = gb.input(&[Some(4)], DType::F32);
        let c = gb.emit_div(&a, &b);
        let graph = gb.compile(&[&c]).expect("compile div");

        let a_buf = Buffer::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[4], DType::F32);
        let b_buf = Buffer::from_slice(&[2.0f32, 4.0, 5.0, 8.0], &[4], DType::F32);
        let out = graph.run(&[&a_buf, &b_buf]);
        assert_eq!(out[0].as_slice::<f32>(), &[5.0, 5.0, 6.0, 5.0]);
    }

    #[test]
    fn elementwise_neg_exp_tanh() {
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(2)], DType::F32);
        let neg_a = gb.emit_neg(&a);
        let exp_a = gb.emit_exp(&a);
        let tanh_a = gb.emit_tanh(&a);
        let graph = gb
            .compile(&[&neg_a, &exp_a, &tanh_a])
            .expect("compile unary ops");

        let a_buf = Buffer::from_slice(&[1.0f32, 0.0], &[2], DType::F32);
        let out = graph.run(&[&a_buf]);
        assert_eq!(out[0].as_slice::<f32>(), &[-1.0, 0.0]); // neg
        let exp_vals = out[1].as_slice::<f32>();
        assert!((exp_vals[0] - std::f32::consts::E).abs() < 1e-5); // exp(1) ≈ e
        assert!((exp_vals[1] - 1.0).abs() < 1e-5); // exp(0) = 1
        let tanh_vals = out[2].as_slice::<f32>();
        assert!((tanh_vals[0] - 0.7615942).abs() < 1e-5); // tanh(1)
        assert!((tanh_vals[1] - 0.0).abs() < 1e-5); // tanh(0)
    }

    #[test]
    fn elementwise_relu() {
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(4)], DType::F32);
        let r = gb.emit_relu(&a);
        let graph = gb.compile(&[&r]).expect("compile relu");

        let a_buf = Buffer::from_slice(&[-1.0f32, 0.0, 1.0, -0.5], &[4], DType::F32);
        let out = graph.run(&[&a_buf]);
        assert_eq!(out[0].as_slice::<f32>(), &[0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn elementwise_sqrt_reciprocal_rsqrt() {
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(2)], DType::F32);
        let sq = gb.emit_sqrt(&a);
        let rec = gb.emit_reciprocal(&a);
        let rsq = gb.emit_rsqrt(&a);
        let graph = gb.compile(&[&sq, &rec, &rsq]).expect("compile sqrt ops");

        let a_buf = Buffer::from_slice(&[4.0f32, 9.0], &[2], DType::F32);
        let out = graph.run(&[&a_buf]);
        let sqrt_vals = out[0].as_slice::<f32>();
        assert!((sqrt_vals[0] - 2.0).abs() < 1e-5);
        assert!((sqrt_vals[1] - 3.0).abs() < 1e-5);
        let rec_vals = out[1].as_slice::<f32>();
        assert!((rec_vals[0] - 0.25).abs() < 1e-5);
        assert!((rec_vals[1] - 1.0 / 9.0).abs() < 1e-5);
        let rsq_vals = out[2].as_slice::<f32>();
        assert!((rsq_vals[0] - 0.5).abs() < 1e-5);
        assert!((rsq_vals[1] - 1.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn elementwise_pow() {
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let base = gb.input(&[Some(3)], DType::F32);
        let exp = gb.input(&[Some(3)], DType::F32);
        let result = gb.emit_pow(&base, &exp);
        let graph = gb.compile(&[&result]).expect("compile pow");

        let base_buf = Buffer::from_slice(&[2.0f32, 3.0, 4.0], &[3], DType::F32);
        let exp_buf = Buffer::from_slice(&[3.0f32, 2.0, 0.5], &[3], DType::F32);
        let out = graph.run(&[&base_buf, &exp_buf]);
        let vals = out[0].as_slice::<f32>();
        assert!((vals[0] - 8.0).abs() < 1e-5); // 2^3
        assert!((vals[1] - 9.0).abs() < 1e-5); // 3^2
        assert!((vals[2] - 2.0).abs() < 1e-5); // 4^0.5
    }

    #[test]
    fn elementwise_dynamic_dims_type_inference() {
        // Task 2.9: element-wise ops with `?` dims verify output type.
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[None, Some(768)], DType::F32);
        let b = gb.input(&[None, Some(768)], DType::F32);
        let c = gb.emit_add(&a, &b);
        assert_eq!(c.shape(), vec![None, Some(768)]);
        assert_eq!(c.dtype(), DType::F32);

        let neg_c = gb.emit_neg(&c);
        assert_eq!(neg_c.shape(), vec![None, Some(768)]);

        let exp_c = gb.emit_exp(&c);
        assert_eq!(exp_c.shape(), vec![None, Some(768)]);
    }

    #[test]
    fn broadcast_rank_promotion() {
        // Task 2.10: binary op with [3] + [4,3] => [4,3].
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(3)], DType::F32); // [3]
        let b = gb.input(&[Some(4), Some(3)], DType::F32); // [4, 3]
        let c = gb.emit_add(&a, &b);
        assert_eq!(c.shape(), vec![Some(4), Some(3)]);

        let graph = gb.compile(&[&c]).expect("compile broadcast add");

        // a = [1, 2, 3], b = [[10,20,30],[40,50,60],[70,80,90],[100,110,120]]
        let a_buf = Buffer::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
        let b_buf = Buffer::from_slice(
            &[
                10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0,
            ],
            &[4, 3],
            DType::F32,
        );
        let out = graph.run(&[&a_buf, &b_buf]);
        assert_eq!(
            out[0].as_slice::<f32>(),
            &[
                11.0, 22.0, 33.0, 41.0, 52.0, 63.0, 71.0, 82.0, 93.0, 101.0, 112.0, 123.0
            ]
        );
    }

    // ── Matmul and Gemm (tasks 3.1 – 3.4) ────────────────────────────────

    #[test]
    fn matmul_2d_static() {
        // [2,3] x [3,4] -> [2,4] with known values.
        // A = [[1,2,3],[4,5,6]], B = [[1,0,0,0],[0,1,0,0],[0,0,1,0]]
        // => C = [[1,2,3,0],[4,5,6,0]]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(2), Some(3)], DType::F32);
        let b = gb.input(&[Some(3), Some(4)], DType::F32);
        let c = gb.emit_matmul(&a, &b);
        assert_eq!(c.shape(), vec![Some(2), Some(4)]);
        let graph = gb.compile(&[&c]).expect("compile matmul 2d");

        let a_buf = Buffer::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
        // B is identity-ish: [[1,0,0,0],[0,1,0,0],[0,0,1,0]]
        let b_buf = Buffer::from_slice(
            &[
                1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            ],
            &[3, 4],
            DType::F32,
        );
        let out = graph.run(&[&a_buf, &b_buf]);
        assert_eq!(
            out[0].as_slice::<f32>(),
            &[1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 0.0]
        );
    }

    #[test]
    fn matmul_2d_known_values() {
        // Simple 2x2 @ 2x2
        // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(2), Some(2)], DType::F32);
        let b = gb.input(&[Some(2), Some(2)], DType::F32);
        let c = gb.emit_matmul(&a, &b);
        let graph = gb.compile(&[&c]).expect("compile matmul 2x2");

        let a_buf = Buffer::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], DType::F32);
        let b_buf = Buffer::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], DType::F32);
        let out = graph.run(&[&a_buf, &b_buf]);
        let vals = out[0].as_slice::<f32>();
        assert!((vals[0] - 19.0).abs() < 1e-4);
        assert!((vals[1] - 22.0).abs() < 1e-4);
        assert!((vals[2] - 43.0).abs() < 1e-4);
        assert!((vals[3] - 50.0).abs() < 1e-4);
    }

    #[test]
    fn batch_matmul_3d_static() {
        // [2,3,4] x [2,4,5] -> [2,3,5]
        // Use identity-like B to verify correctness:
        // batch 0: A0 = ones(3,4), B0 = [[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0]]
        //   => C0 = A0 @ B0 = [[1,1,1,1,0],[1,1,1,1,0],[1,1,1,1,0]]
        // batch 1: A1 = eye_3x4, B1 = all_ones(4,5)
        //   => C1 = A1 @ B1 = [[1,1,1,1,1],[1,1,1,1,1],[0,0,0,0,0]]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(2), Some(3), Some(4)], DType::F32);
        let b = gb.input(&[Some(2), Some(4), Some(5)], DType::F32);
        let c = gb.emit_matmul(&a, &b);
        assert_eq!(c.shape(), vec![Some(2), Some(3), Some(5)]);
        let graph = gb.compile(&[&c]).expect("compile batch_matmul 3d");

        // batch 0: A0 = ones(3,4)
        let a0 = vec![1.0f32; 12];
        // batch 0: B0 = partial identity (4x5, first 4 rows = I4 with 5th col=0)
        let b0: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
        ];
        // batch 1: A1 = eye-ish (3x4, first 3 rows partial identity)
        let a1: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        // batch 1: B1 = all ones (4x5)
        let b1 = vec![1.0f32; 20];

        let a_data: Vec<f32> = a0.iter().chain(a1.iter()).copied().collect();
        let b_data: Vec<f32> = b0.iter().chain(b1.iter()).copied().collect();

        let a_buf = Buffer::from_slice(&a_data, &[2, 3, 4], DType::F32);
        let b_buf = Buffer::from_slice(&b_data, &[2, 4, 5], DType::F32);
        let out = graph.run(&[&a_buf, &b_buf]);
        let vals = out[0].as_slice::<f32>();

        // batch 0 result (3x5): ones @ partial_identity = [[1,1,1,1,0],[1,1,1,1,0],[1,1,1,1,0]]
        let expected_b0 = [
            1.0f32, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0,
        ];
        for (i, &exp) in expected_b0.iter().enumerate() {
            assert!(
                (vals[i] - exp).abs() < 1e-4,
                "batch0[{i}]: got {}, expected {}",
                vals[i],
                exp
            );
        }
        // batch 1 result (3x5): eye-ish @ all_ones
        // row0 = [1,0,0,0] @ all_ones = sum of col0 = [1,1,1,1,1]
        // row1 = [0,1,0,0] @ all_ones = [1,1,1,1,1]
        // row2 = [0,0,1,0] @ all_ones = [1,1,1,1,1]
        let expected_b1 = [
            1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ];
        for (i, &exp) in expected_b1.iter().enumerate() {
            let idx = 15 + i;
            assert!(
                (vals[idx] - exp).abs() < 1e-4,
                "batch1[{i}]: got {}, expected {}",
                vals[idx],
                exp
            );
        }
    }

    #[test]
    fn matmul_dynamic_batch_dim() {
        // [?,3,4] x [?,4,5] -> [?,3,5] — compile with dynamic batch dim.
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[None, Some(3), Some(4)], DType::F32);
        let b = gb.input(&[None, Some(4), Some(5)], DType::F32);
        let c = gb.emit_matmul(&a, &b);
        // Output batch dim must be dynamic.
        assert_eq!(c.shape(), vec![None, Some(3), Some(5)]);

        // Compile and run with batch=1 to verify correctness.
        let graph = gb.compile(&[&c]).expect("compile dynamic batch matmul");

        // batch=1: A = ones(1,3,4), B = partial identity (1,4,5)
        let a_data = vec![1.0f32; 12];
        let b_data: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
        ];
        let a_buf = Buffer::from_slice(&a_data, &[1, 3, 4], DType::F32);
        let b_buf = Buffer::from_slice(&b_data, &[1, 4, 5], DType::F32);
        // Output shape is [?,3,5]. Provide concrete shape [1,3,5] to run_dynamic.
        let out = graph.run_dynamic(&[&a_buf, &b_buf], &[crate::Shape(vec![1, 3, 5])]);
        // ones @ partial_identity = [[1,1,1,1,0], ...]
        let vals = out[0].as_slice::<f32>();
        let expected = [
            1.0f32, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0,
        ];
        for (i, &exp) in expected.iter().enumerate() {
            assert!(
                (vals[i] - exp).abs() < 1e-4,
                "[{i}]: got {}, expected {}",
                vals[i],
                exp
            );
        }
    }

    #[test]
    fn gemm_transpose_b() {
        // Gemm with transB=1, alpha=1, beta=1.
        // A = [[1,2],[3,4]]  shape [2,2]
        // B = [[5,7],[6,8]]  shape [2,2] — after transposing: [[5,6],[7,8]]
        // So A @ B^T = [[1,2],[3,4]] @ [[5,6],[7,8]]^T
        //            = [[1,2],[3,4]] @ [[5,7],[6,8]] = [[17,23],[39,53]]
        // Wait — B is stored as [N, K] with transB=1, so we transpose to [K, N].
        // Let B_stored = [[5,6],[7,8]], B_T = [[5,7],[6,8]]
        // A @ B_T = [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
        // C = zeros(2,2), beta=0 so result = A @ B_T + 0*C = [[19,22],[43,50]]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(2), Some(2)], DType::F32);
        let b = gb.input(&[Some(2), Some(2)], DType::F32);
        let c = gb.input(&[Some(2), Some(2)], DType::F32);
        let result = gb.emit_gemm(&a, &b, &c, 1.0, 0.0, false, true);
        let graph = gb.compile(&[&result]).expect("compile gemm transB");

        let a_buf = Buffer::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], DType::F32);
        let b_buf = Buffer::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], DType::F32);
        let c_buf = Buffer::from_slice(&[0.0f32; 4], &[2, 2], DType::F32);
        let out = graph.run(&[&a_buf, &b_buf, &c_buf]);
        let vals = out[0].as_slice::<f32>();
        // A @ B^T: B^T = [[5,7],[6,8]]
        // row0: [1,2] @ [[5,7],[6,8]] = [1*5+2*6, 1*7+2*8] = [17, 23]
        // row1: [3,4] @ [[5,7],[6,8]] = [3*5+4*6, 3*7+4*8] = [39, 53]
        assert!((vals[0] - 17.0).abs() < 1e-4, "vals[0]={}", vals[0]);
        assert!((vals[1] - 23.0).abs() < 1e-4, "vals[1]={}", vals[1]);
        assert!((vals[2] - 39.0).abs() < 1e-4, "vals[2]={}", vals[2]);
        assert!((vals[3] - 53.0).abs() < 1e-4, "vals[3]={}", vals[3]);
    }

    #[test]
    fn gemm_with_alpha_beta_c() {
        // alpha=2, beta=1, no transposes.
        // A = [[1,0],[0,1]], B = [[2,3],[4,5]], C = [[1,1],[1,1]]
        // A@B = [[2,3],[4,5]]
        // 2*(A@B) + 1*C = [[4,6],[8,10]] + [[1,1],[1,1]] = [[5,7],[9,11]]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(2), Some(2)], DType::F32);
        let b = gb.input(&[Some(2), Some(2)], DType::F32);
        let c = gb.input(&[Some(2), Some(2)], DType::F32);
        let result = gb.emit_gemm(&a, &b, &c, 2.0, 1.0, false, false);
        let graph = gb.compile(&[&result]).expect("compile gemm alpha_beta");

        let a_buf = Buffer::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2], DType::F32);
        let b_buf = Buffer::from_slice(&[2.0f32, 3.0, 4.0, 5.0], &[2, 2], DType::F32);
        let c_buf = Buffer::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[2, 2], DType::F32);
        let out = graph.run(&[&a_buf, &b_buf, &c_buf]);
        let vals = out[0].as_slice::<f32>();
        assert!((vals[0] - 5.0).abs() < 1e-4, "vals[0]={}", vals[0]);
        assert!((vals[1] - 7.0).abs() < 1e-4, "vals[1]={}", vals[1]);
        assert!((vals[2] - 9.0).abs() < 1e-4, "vals[2]={}", vals[2]);
        assert!((vals[3] - 11.0).abs() < 1e-4, "vals[3]={}", vals[3]);
    }

    #[test]
    fn broadcast_size_one_expansion() {
        // [4,1] + [4,3] => [4,3] — same rank, size-1 broadcast.
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(4), Some(1)], DType::F32); // [4, 1]
        let b = gb.input(&[Some(4), Some(3)], DType::F32); // [4, 3]
        let c = gb.emit_add(&a, &b);
        assert_eq!(c.shape(), vec![Some(4), Some(3)]);

        let graph = gb.compile(&[&c]).expect("compile size-1 broadcast");

        let a_buf = Buffer::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4, 1], DType::F32);
        let b_buf = Buffer::from_slice(
            &[
                10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0,
            ],
            &[4, 3],
            DType::F32,
        );
        let out = graph.run(&[&a_buf, &b_buf]);
        assert_eq!(
            out[0].as_slice::<f32>(),
            &[
                11.0, 21.0, 31.0, 42.0, 52.0, 62.0, 73.0, 83.0, 93.0, 104.0, 114.0, 124.0
            ]
        );
    }

    // ── Reductions and Softmax (tasks 4.1 – 4.5) ─────────────────────────

    #[test]
    fn reduce_sum_axis0() {
        // tensor<3x4xf32> reduced along axis 0 -> shape [4].
        // Input: rows = [1,2,3,4], [5,6,7,8], [9,10,11,12]
        // Sum along rows: [15, 18, 21, 24]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(3), Some(4)], DType::F32);
        let r = gb.emit_reduce_sum(&a, 0, false);
        assert_eq!(r.shape(), vec![Some(4)]);
        let graph = gb.compile(&[&r]).expect("compile reduce_sum axis0");

        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let a_buf = Buffer::from_slice(&data, &[3, 4], DType::F32);
        let out = graph.run(&[&a_buf]);
        let vals = out[0].as_slice::<f32>();
        assert_eq!(vals.len(), 4);
        assert!((vals[0] - 15.0).abs() < 1e-5, "vals[0]={}", vals[0]);
        assert!((vals[1] - 18.0).abs() < 1e-5, "vals[1]={}", vals[1]);
        assert!((vals[2] - 21.0).abs() < 1e-5, "vals[2]={}", vals[2]);
        assert!((vals[3] - 24.0).abs() < 1e-5, "vals[3]={}", vals[3]);
    }

    #[test]
    fn reduce_sum_keepdim() {
        // tensor<3x4xf32> reduced along axis 0, keepdim=true -> shape [1,4].
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(3), Some(4)], DType::F32);
        let r = gb.emit_reduce_sum(&a, 0, true);
        assert_eq!(r.shape(), vec![Some(1), Some(4)]);
        let graph = gb.compile(&[&r]).expect("compile reduce_sum keepdim");

        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let a_buf = Buffer::from_slice(&data, &[3, 4], DType::F32);
        let out = graph.run(&[&a_buf]);
        let vals = out[0].as_slice::<f32>();
        assert_eq!(vals.len(), 4);
        assert!((vals[0] - 15.0).abs() < 1e-5);
        assert!((vals[1] - 18.0).abs() < 1e-5);
        assert!((vals[2] - 21.0).abs() < 1e-5);
        assert!((vals[3] - 24.0).abs() < 1e-5);
    }

    #[test]
    fn reduce_sum_axis1() {
        // tensor<3x4xf32> reduced along axis 1 -> shape [3].
        // Sum along columns: row0=10, row1=26, row2=42
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(3), Some(4)], DType::F32);
        let r = gb.emit_reduce_sum(&a, 1, false);
        assert_eq!(r.shape(), vec![Some(3)]);
        let graph = gb.compile(&[&r]).expect("compile reduce_sum axis1");

        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let a_buf = Buffer::from_slice(&data, &[3, 4], DType::F32);
        let out = graph.run(&[&a_buf]);
        let vals = out[0].as_slice::<f32>();
        assert_eq!(vals.len(), 3);
        assert!((vals[0] - 10.0).abs() < 1e-5, "vals[0]={}", vals[0]);
        assert!((vals[1] - 26.0).abs() < 1e-5, "vals[1]={}", vals[1]);
        assert!((vals[2] - 42.0).abs() < 1e-5, "vals[2]={}", vals[2]);
    }

    #[test]
    fn reduce_sum_negative_axis() {
        // Negative axis: axis=-1 on [3,4] is equivalent to axis=1.
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(3), Some(4)], DType::F32);
        let r = gb.emit_reduce_sum(&a, -1, false);
        assert_eq!(r.shape(), vec![Some(3)]);
        let graph = gb.compile(&[&r]).expect("compile reduce_sum negative axis");

        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let a_buf = Buffer::from_slice(&data, &[3, 4], DType::F32);
        let out = graph.run(&[&a_buf]);
        let vals = out[0].as_slice::<f32>();
        assert!((vals[0] - 10.0).abs() < 1e-5);
        assert!((vals[1] - 26.0).abs() < 1e-5);
        assert!((vals[2] - 42.0).abs() < 1e-5);
    }

    #[test]
    fn reduce_max_static() {
        // tensor<2x3xf32> reduced along axis 1 -> shape [2].
        // row0 = [3, 1, 4], max = 4
        // row1 = [1, 5, 9], max = 9
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(2), Some(3)], DType::F32);
        let r = gb.emit_reduce_max(&a, 1, false);
        assert_eq!(r.shape(), vec![Some(2)]);
        let graph = gb.compile(&[&r]).expect("compile reduce_max");

        let a_buf = Buffer::from_slice(&[3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0], &[2, 3], DType::F32);
        let out = graph.run(&[&a_buf]);
        let vals = out[0].as_slice::<f32>();
        assert!((vals[0] - 4.0).abs() < 1e-5, "vals[0]={}", vals[0]);
        assert!((vals[1] - 9.0).abs() < 1e-5, "vals[1]={}", vals[1]);
    }

    #[test]
    fn reduce_max_keepdim() {
        // keepdim=true: [2,3] -> [2,1]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(2), Some(3)], DType::F32);
        let r = gb.emit_reduce_max(&a, 1, true);
        assert_eq!(r.shape(), vec![Some(2), Some(1)]);
        let graph = gb.compile(&[&r]).expect("compile reduce_max keepdim");

        let a_buf = Buffer::from_slice(&[3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0], &[2, 3], DType::F32);
        let out = graph.run(&[&a_buf]);
        let vals = out[0].as_slice::<f32>();
        assert_eq!(vals.len(), 2);
        assert!((vals[0] - 4.0).abs() < 1e-5);
        assert!((vals[1] - 9.0).abs() < 1e-5);
    }

    #[test]
    fn reduce_mean_static() {
        // tensor<2x4xf32> mean along axis=1 -> shape [2].
        // row0 = [1,2,3,4], mean = 2.5
        // row1 = [5,6,7,8], mean = 6.5
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(2), Some(4)], DType::F32);
        let r = gb.emit_reduce_mean(&a, &[1], false);
        assert_eq!(r.shape(), vec![Some(2)]);
        let graph = gb.compile(&[&r]).expect("compile reduce_mean");

        let a_buf = Buffer::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 4],
            DType::F32,
        );
        let out = graph.run(&[&a_buf]);
        let vals = out[0].as_slice::<f32>();
        assert_eq!(vals.len(), 2);
        assert!((vals[0] - 2.5).abs() < 1e-5, "vals[0]={}", vals[0]);
        assert!((vals[1] - 6.5).abs() < 1e-5, "vals[1]={}", vals[1]);
    }

    #[test]
    fn reduce_mean_multiple_axes() {
        // tensor<2x3x4xf32> mean along axes [1,2] -> shape [2].
        // All values = 1.0 -> mean = 1.0 for each batch.
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(2), Some(3), Some(4)], DType::F32);
        let r = gb.emit_reduce_mean(&a, &[1, 2], false);
        assert_eq!(r.shape(), vec![Some(2)]);
        let graph = gb.compile(&[&r]).expect("compile reduce_mean multi-axis");

        let data = vec![1.0f32; 24];
        let a_buf = Buffer::from_slice(&data, &[2, 3, 4], DType::F32);
        let out = graph.run(&[&a_buf]);
        let vals = out[0].as_slice::<f32>();
        assert_eq!(vals.len(), 2);
        assert!((vals[0] - 1.0).abs() < 1e-5, "vals[0]={}", vals[0]);
        assert!((vals[1] - 1.0).abs() < 1e-5, "vals[1]={}", vals[1]);
    }

    #[test]
    fn softmax_static() {
        // softmax([1, 2, 3]) ≈ [0.0900, 0.2447, 0.6652]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(1), Some(3)], DType::F32); // [1,3] for axis=1
        let r = gb.emit_softmax(&a, 1);
        assert_eq!(r.shape(), vec![Some(1), Some(3)]);
        let graph = gb.compile(&[&r]).expect("compile softmax");

        let a_buf = Buffer::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3], DType::F32);
        let out = graph.run(&[&a_buf]);
        let vals = out[0].as_slice::<f32>();
        assert_eq!(vals.len(), 3);
        assert!((vals[0] - 0.0900).abs() < 1e-3, "vals[0]={}", vals[0]);
        assert!((vals[1] - 0.2447).abs() < 1e-3, "vals[1]={}", vals[1]);
        assert!((vals[2] - 0.6652).abs() < 1e-3, "vals[2]={}", vals[2]);
        // Also verify they sum to 1.
        let sum: f32 = vals.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum={sum}");
    }

    #[test]
    fn softmax_2d_axis0() {
        // softmax along axis 0 of [[1,1],[2,2]] -> [[sigma,sigma],[1-sigma,1-sigma]]
        // sigma = e^1/(e^1+e^2) ≈ 0.2689
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(2), Some(2)], DType::F32);
        let r = gb.emit_softmax(&a, 0);
        assert_eq!(r.shape(), vec![Some(2), Some(2)]);
        let graph = gb.compile(&[&r]).expect("compile softmax axis0");

        let a_buf = Buffer::from_slice(&[1.0f32, 1.0, 2.0, 2.0], &[2, 2], DType::F32);
        let out = graph.run(&[&a_buf]);
        let vals = out[0].as_slice::<f32>();
        let sigma = 1.0f32 / (1.0 + std::f32::consts::E);
        assert!((vals[0] - sigma).abs() < 1e-5, "vals[0]={}", vals[0]);
        assert!((vals[1] - sigma).abs() < 1e-5, "vals[1]={}", vals[1]);
        assert!(
            (vals[2] - (1.0 - sigma)).abs() < 1e-5,
            "vals[2]={}",
            vals[2]
        );
        assert!(
            (vals[3] - (1.0 - sigma)).abs() < 1e-5,
            "vals[3]={}",
            vals[3]
        );
    }

    #[test]
    fn reduce_sum_dynamic() {
        // tensor<?x4xf32> reduce along axis 0 -> [4].
        // Only checks that compilation and execution succeed with dynamic dim.
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[None, Some(4)], DType::F32);
        let r = gb.emit_reduce_sum(&a, 0, false);
        assert_eq!(r.shape(), vec![Some(4)]);
        let graph = gb.compile(&[&r]).expect("compile reduce_sum dynamic");

        // Run with 3 rows.
        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let a_buf = Buffer::from_slice(&data, &[3, 4], DType::F32);
        let out = graph.run_dynamic(&[&a_buf], &[crate::Shape(vec![4])]);
        let vals = out[0].as_slice::<f32>();
        assert_eq!(vals.len(), 4);
        assert!((vals[0] - 15.0).abs() < 1e-5);
        assert!((vals[1] - 18.0).abs() < 1e-5);
        assert!((vals[2] - 21.0).abs() < 1e-5);
        assert!((vals[3] - 24.0).abs() < 1e-5);
    }

    // ── Phase 5: Shape Manipulation Ops (tasks 5.1 – 5.11) ────────────────

    #[test]
    fn reshape_static() {
        // [2, 6] -> [3, 4]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(2), Some(6)], DType::F32);
        let r = gb.emit_reshape(&a, &[3, 4]);
        assert_eq!(r.shape(), vec![Some(3), Some(4)]);
        let graph = gb.compile(&[&r]).expect("compile reshape static");

        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let a_buf = Buffer::from_slice(&data, &[2, 6], DType::F32);
        let out = graph.run(&[&a_buf]);
        // reshape is a view-like op — elements in same order
        assert_eq!(out[0].as_slice::<f32>(), data.as_slice());
    }

    #[test]
    fn reshape_infer_dim() {
        // [2, 6] -> [-1, 3] — inferred first dim should be 4
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(2), Some(6)], DType::F32);
        let r = gb.emit_reshape(&a, &[-1, 3]);
        // inferred dim is now resolved statically: 12 / 3 = 4
        assert_eq!(r.shape(), vec![Some(4), Some(3)]);
        let graph = gb.compile(&[&r]).expect("compile reshape infer dim");

        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let a_buf = Buffer::from_slice(&data, &[2, 6], DType::F32);
        // output is [4, 3] — now statically inferred
        let out = graph.run(&[&a_buf]);
        assert_eq!(out[0].as_slice::<f32>(), data.as_slice());
    }

    #[test]
    fn reshape_1d_to_3d() {
        // [24] -> [2, 3, 4]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(24)], DType::F32);
        let r = gb.emit_reshape(&a, &[2, 3, 4]);
        assert_eq!(r.shape(), vec![Some(2), Some(3), Some(4)]);
        let graph = gb.compile(&[&r]).expect("compile reshape 1d->3d");

        let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
        let a_buf = Buffer::from_slice(&data, &[24], DType::F32);
        let out = graph.run(&[&a_buf]);
        assert_eq!(out[0].as_slice::<f32>(), data.as_slice());
    }

    #[test]
    fn transpose_2d() {
        // [2, 3] -> [3, 2] with perms [1, 0]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(2), Some(3)], DType::F32);
        let r = gb.emit_transpose(&a, &[1, 0]);
        assert_eq!(r.shape(), vec![Some(3), Some(2)]);
        let graph = gb.compile(&[&r]).expect("compile transpose 2d");

        // [[1, 2, 3], [4, 5, 6]] -> [[1, 4], [2, 5], [3, 6]]
        let a_buf = Buffer::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
        let out = graph.run(&[&a_buf]);
        assert_eq!(out[0].as_slice::<f32>(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn transpose_3d() {
        // [2, 3, 4] -> [4, 2, 3] with perms [2, 0, 1]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(2), Some(3), Some(4)], DType::F32);
        let r = gb.emit_transpose(&a, &[2, 0, 1]);
        assert_eq!(r.shape(), vec![Some(4), Some(2), Some(3)]);

        let graph = gb.compile(&[&r]).expect("compile transpose 3d");

        // Verify shape only — correctness from type checking.
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let a_buf = Buffer::from_slice(&data, &[2, 3, 4], DType::F32);
        let out = graph.run(&[&a_buf]);
        // output[k, i, j] = input[i, j, k]
        // out[0,0,0]=input[0,0,0]=0, out[0,0,1]=input[0,1,0]=4, out[1,0,0]=input[0,0,1]=1
        let vals = out[0].as_slice::<f32>();
        assert!((vals[0] - 0.0).abs() < 1e-5); // out[0,0,0] = input[0,0,0]
        assert!((vals[1] - 4.0).abs() < 1e-5); // out[0,0,1] = input[0,1,0]
        assert!((vals[2] - 8.0).abs() < 1e-5); // out[0,0,2] = input[0,2,0]
    }

    #[test]
    fn concat_axis0() {
        // concat([[1,2,3], [4,5,6]], axis=0) = [1,2,3,4,5,6]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(3)], DType::F32);
        let b = gb.input(&[Some(3)], DType::F32);
        let r = gb.emit_concat(&[a, b], 0);
        assert_eq!(r.shape(), vec![Some(6)]);
        let graph = gb.compile(&[&r]).expect("compile concat axis0");

        let a_buf = Buffer::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
        let b_buf = Buffer::from_slice(&[4.0f32, 5.0, 6.0], &[3], DType::F32);
        let out = graph.run(&[&a_buf, &b_buf]);
        assert_eq!(out[0].as_slice::<f32>(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn concat_axis1_2d() {
        // concat([[1,2],[3,4]], [[5,6],[7,8]], axis=1) = [[1,2,5,6],[3,4,7,8]]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(2), Some(2)], DType::F32);
        let b = gb.input(&[Some(2), Some(2)], DType::F32);
        let r = gb.emit_concat(&[a, b], 1);
        assert_eq!(r.shape(), vec![Some(2), Some(4)]);
        let graph = gb.compile(&[&r]).expect("compile concat axis1 2d");

        let a_buf = Buffer::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], DType::F32);
        let b_buf = Buffer::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], DType::F32);
        let out = graph.run(&[&a_buf, &b_buf]);
        assert_eq!(
            out[0].as_slice::<f32>(),
            &[1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]
        );
    }

    #[test]
    fn slice_static() {
        // slice [0..10] -> [2..7] step 1
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(10)], DType::F32);
        let r = gb.emit_slice(&a, &[2], &[7], &[0], &[1]);
        assert_eq!(r.shape(), vec![Some(5)]);
        let graph = gb.compile(&[&r]).expect("compile slice static");

        let data: Vec<f32> = (0..10).map(|x| x as f32).collect();
        let a_buf = Buffer::from_slice(&data, &[10], DType::F32);
        let out = graph.run(&[&a_buf]);
        assert_eq!(out[0].as_slice::<f32>(), &[2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn slice_with_step() {
        // slice [0..10] -> [0..10] step 2  -> [0, 2, 4, 6, 8]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(10)], DType::F32);
        let r = gb.emit_slice(&a, &[0], &[10], &[0], &[2]);
        assert_eq!(r.shape(), vec![Some(5)]);
        let graph = gb.compile(&[&r]).expect("compile slice step2");

        let data: Vec<f32> = (0..10).map(|x| x as f32).collect();
        let a_buf = Buffer::from_slice(&data, &[10], DType::F32);
        let out = graph.run(&[&a_buf]);
        assert_eq!(out[0].as_slice::<f32>(), &[0.0, 2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn slice_2d_axis1() {
        // 2D slice: [3,5] slice axis=1 from 1 to 4 step 1 -> [3,3]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(3), Some(5)], DType::F32);
        let r = gb.emit_slice(&a, &[1], &[4], &[1], &[1]);
        assert_eq!(r.shape(), vec![Some(3), Some(3)]);
        let graph = gb.compile(&[&r]).expect("compile slice 2d axis1");

        // input: [[0..4], [5..9], [10..14]]
        let data: Vec<f32> = (0..15).map(|x| x as f32).collect();
        let a_buf = Buffer::from_slice(&data, &[3, 5], DType::F32);
        let out = graph.run(&[&a_buf]);
        // rows: [1,2,3], [6,7,8], [11,12,13]
        assert_eq!(
            out[0].as_slice::<f32>(),
            &[1.0, 2.0, 3.0, 6.0, 7.0, 8.0, 11.0, 12.0, 13.0]
        );
    }

    #[test]
    fn gather_axis0() {
        // data = [[1,2],[3,4],[5,6]], indices = [2, 0, 1], axis=0
        // output[i] = data[indices[i]] = [[5,6],[1,2],[3,4]]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let data = gb.input(&[Some(3), Some(2)], DType::F32);
        let indices = gb.input(&[Some(3)], DType::I64);
        let r = gb.emit_gather(&data, &indices, 0);
        // output shape: indices.shape + data.shape[1..] = [3, 2]
        assert_eq!(r.shape(), vec![Some(3), Some(2)]);
        let graph = gb.compile(&[&r]).expect("compile gather axis0");

        let data_buf = Buffer::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], DType::F32);
        let idx_buf = Buffer::from_slice(&[2i64, 0, 1], &[3], DType::I64);
        let out = graph.run(&[&data_buf, &idx_buf]);
        assert_eq!(out[0].as_slice::<f32>(), &[5.0, 6.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn where_select() {
        // cond = [1, 0, 1, 0], x = [10,20,30,40], y = [1,2,3,4]
        // result = [10, 2, 30, 4]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let cond = gb.input(&[Some(4)], DType::I64);
        let x = gb.input(&[Some(4)], DType::F32);
        let y = gb.input(&[Some(4)], DType::F32);
        let r = gb.emit_where(&cond, &x, &y);
        assert_eq!(r.shape(), vec![Some(4)]);
        let graph = gb.compile(&[&r]).expect("compile where");

        let cond_buf = Buffer::from_slice(&[1i64, 0, 1, 0], &[4], DType::I64);
        let x_buf = Buffer::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[4], DType::F32);
        let y_buf = Buffer::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], DType::F32);
        let out = graph.run(&[&cond_buf, &x_buf, &y_buf]);
        assert_eq!(out[0].as_slice::<f32>(), &[10.0, 2.0, 30.0, 4.0]);
    }

    #[test]
    fn cast_i32_to_f32() {
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(4)], DType::I32);
        let r = gb.emit_cast(&a, DType::F32);
        assert_eq!(r.dtype(), DType::F32);
        assert_eq!(r.shape(), vec![Some(4)]);
        let graph = gb.compile(&[&r]).expect("compile cast i32->f32");

        let a_buf = Buffer::from_slice(&[1i32, 2, 3, 4], &[4], DType::I32);
        let out = graph.run(&[&a_buf]);
        assert_eq!(out[0].as_slice::<f32>(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn cast_f32_to_i32() {
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(4)], DType::F32);
        let r = gb.emit_cast(&a, DType::I32);
        assert_eq!(r.dtype(), DType::I32);
        let graph = gb.compile(&[&r]).expect("compile cast f32->i32");

        let a_buf = Buffer::from_slice(&[1.7f32, 2.2, -3.9, 4.0], &[4], DType::F32);
        let out = graph.run(&[&a_buf]);
        // fptosi truncates toward zero
        assert_eq!(out[0].as_slice::<i32>(), &[1, 2, -3, 4]);
    }

    #[test]
    fn cast_i64_to_i32() {
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(3)], DType::I64);
        let r = gb.emit_cast(&a, DType::I32);
        assert_eq!(r.dtype(), DType::I32);
        let graph = gb.compile(&[&r]).expect("compile cast i64->i32");

        let a_buf = Buffer::from_slice(&[100i64, 200, 300], &[3], DType::I64);
        let out = graph.run(&[&a_buf]);
        assert_eq!(out[0].as_slice::<i32>(), &[100, 200, 300]);
    }

    #[test]
    fn unsqueeze_and_squeeze() {
        // unsqueeze [3, 4] at axis 0 -> [1, 3, 4]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(3), Some(4)], DType::F32);
        let u = gb.emit_unsqueeze(&a, &[0]);
        assert_eq!(u.shape(), vec![Some(1), Some(3), Some(4)]);

        // squeeze [1, 3, 4] at axis 0 -> [3, 4]
        let s = gb.emit_squeeze(&u, &[0]);
        assert_eq!(s.shape(), vec![Some(3), Some(4)]);

        let graph = gb.compile(&[&s]).expect("compile unsqueeze+squeeze");

        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let a_buf = Buffer::from_slice(&data, &[3, 4], DType::F32);
        let out = graph.run(&[&a_buf]);
        assert_eq!(out[0].as_slice::<f32>(), data.as_slice());
    }

    #[test]
    fn unsqueeze_middle_axis() {
        // [2, 3] -> unsqueeze axis=1 -> [2, 1, 3]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(2), Some(3)], DType::F32);
        let u = gb.emit_unsqueeze(&a, &[1]);
        assert_eq!(u.shape(), vec![Some(2), Some(1), Some(3)]);
        let graph = gb.compile(&[&u]).expect("compile unsqueeze middle");

        let data: Vec<f32> = (1..=6).map(|x| x as f32).collect();
        let a_buf = Buffer::from_slice(&data, &[2, 3], DType::F32);
        let out = graph.run(&[&a_buf]);
        assert_eq!(out[0].as_slice::<f32>(), data.as_slice());
    }

    #[test]
    fn flatten_axis1() {
        // [2, 3, 4] -> flatten(axis=1) -> [2, 12]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(2), Some(3), Some(4)], DType::F32);
        let f = gb.emit_flatten(&a, 1);
        assert_eq!(f.shape(), vec![Some(2), Some(12)]);
        let graph = gb.compile(&[&f]).expect("compile flatten axis1");

        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let a_buf = Buffer::from_slice(&data, &[2, 3, 4], DType::F32);
        let out = graph.run(&[&a_buf]);
        assert_eq!(out[0].as_slice::<f32>(), data.as_slice());
    }

    #[test]
    fn split_equal_parts() {
        // [6] split into [2, 2, 2] -> three tensors of size 2
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(6)], DType::F32);
        let parts = gb.emit_split(&a, 0, &[2, 2, 2]);
        assert_eq!(parts.len(), 3);
        for p in &parts {
            assert_eq!(p.shape(), vec![Some(2)]);
        }

        let graph = gb
            .compile(&parts.iter().collect::<Vec<_>>())
            .expect("compile split");

        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a_buf = Buffer::from_slice(&data, &[6], DType::F32);
        let out = graph.run(&[&a_buf]);
        assert_eq!(out[0].as_slice::<f32>(), &[1.0, 2.0]);
        assert_eq!(out[1].as_slice::<f32>(), &[3.0, 4.0]);
        assert_eq!(out[2].as_slice::<f32>(), &[5.0, 6.0]);
    }

    #[test]
    fn split_2d_axis0() {
        // [4, 3] split into [1, 3] along axis 0 -> shapes [1,3] and [3,3]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(4), Some(3)], DType::F32);
        let parts = gb.emit_split(&a, 0, &[1, 3]);
        assert_eq!(parts[0].shape(), vec![Some(1), Some(3)]);
        assert_eq!(parts[1].shape(), vec![Some(3), Some(3)]);

        let graph = gb
            .compile(&parts.iter().collect::<Vec<_>>())
            .expect("compile split 2d");

        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let a_buf = Buffer::from_slice(&data, &[4, 3], DType::F32);
        let out = graph.run(&[&a_buf]);
        assert_eq!(out[0].as_slice::<f32>(), &[1.0, 2.0, 3.0]);
        assert_eq!(
            out[1].as_slice::<f32>(),
            &[4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        );
    }

    // ── Spatial ops (task 6) ──────────────────────────────────────────────────

    /// conv2d with no padding: 1x1x4x4 input, 1x1x3x3 kernel, stride=1 → 1x1x2x2 output.
    ///
    /// Input (row-major):
    ///   [[1, 2, 3, 4],
    ///    [5, 6, 7, 8],
    ///    [9,10,11,12],
    ///    [13,14,15,16]]
    ///
    /// Kernel (all-ones 3x3): each output pixel is the sum of a 3x3 window.
    /// out[0,0] = 1+2+3+5+6+7+9+10+11 = 54
    /// out[0,1] = 2+3+4+6+7+8+10+11+12 = 63
    /// out[1,0] = 5+6+7+9+10+11+13+14+15 = 90
    /// out[1,1] = 6+7+8+10+11+12+14+15+16 = 99
    #[test]
    fn conv2d_no_padding() {
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let input = gb.input(&[Some(1), Some(1), Some(4), Some(4)], DType::F32);
        let weight = gb.add_weight(&[Some(1), Some(1), Some(3), Some(3)], DType::F32);
        let out = gb
            .emit_conv2d(&input, &weight, None, [0, 0, 0, 0], [1, 1], [1, 1])
            .expect("emit_conv2d");
        assert_eq!(out.shape(), vec![Some(1), Some(1), Some(2), Some(2)]);
        let graph = gb.compile(&[&out]).expect("compile conv2d_no_padding");

        let input_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let kernel_data = vec![1.0f32; 9];
        let in_buf = Buffer::from_slice(&input_data, &[1, 1, 4, 4], DType::F32);
        let wt_buf = Buffer::from_slice(&kernel_data, &[1, 1, 3, 3], DType::F32);
        let out = graph.run(&[&in_buf, &wt_buf]);
        assert_eq!(out[0].as_slice::<f32>(), &[54.0, 63.0, 90.0, 99.0]);
    }

    /// conv2d with padding [1,1,1,1]: same as above but pad 1 on all sides → 1x1x4x4 output.
    ///
    /// With 1-zero-pad the padded input is 6x6.
    /// Output spatial size = (4 + 1 + 1 - 3) / 1 + 1 = 4.
    /// Top-left corner (0,0): sum over padded window rows 0..2, cols 0..2.
    ///   = 0+0+0 + 0+1+2 + 0+5+6 = 14
    #[test]
    fn conv2d_with_padding() {
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let input = gb.input(&[Some(1), Some(1), Some(4), Some(4)], DType::F32);
        let weight = gb.add_weight(&[Some(1), Some(1), Some(3), Some(3)], DType::F32);
        let out = gb
            .emit_conv2d(&input, &weight, None, [1, 1, 1, 1], [1, 1], [1, 1])
            .expect("emit_conv2d");
        assert_eq!(out.shape(), vec![Some(1), Some(1), Some(4), Some(4)]);
        let graph = gb.compile(&[&out]).expect("compile conv2d_with_padding");

        let input_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let kernel_data = vec![1.0f32; 9];
        let in_buf = Buffer::from_slice(&input_data, &[1, 1, 4, 4], DType::F32);
        let wt_buf = Buffer::from_slice(&kernel_data, &[1, 1, 3, 3], DType::F32);
        let out_data = graph.run(&[&in_buf, &wt_buf]);
        let vals = out_data[0].as_slice::<f32>();
        assert_eq!(vals.len(), 16);
        // Top-left corner: sum of padded 3x3 window = 0+0+0+0+1+2+0+5+6 = 14.
        assert!((vals[0] - 14.0).abs() < 1e-4, "vals[0]={}", vals[0]);
        // Top-right corner [0,3]: 0+0+0+3+4+0+7+8+0 = 22.
        assert!((vals[3] - 22.0).abs() < 1e-4, "vals[3]={}", vals[3]);
    }

    /// max_pool2d: 1x1x4x4 input, 2x2 kernel, stride=2, no padding → 1x1x2x2 output.
    ///
    /// Input:
    ///   [[1, 2, 3, 4],
    ///    [5, 6, 7, 8],
    ///    [9,10,11,12],
    ///    [13,14,15,16]]
    ///
    /// Pooling window max:
    ///   [0,0]: max(1,2,5,6) = 6
    ///   [0,1]: max(3,4,7,8) = 8
    ///   [1,0]: max(9,10,13,14) = 14
    ///   [1,1]: max(11,12,15,16) = 16
    #[test]
    fn max_pool2d_basic() {
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let input = gb.input(&[Some(1), Some(1), Some(4), Some(4)], DType::F32);
        let out = gb
            .emit_max_pool2d(&input, [2, 2], [0, 0, 0, 0], [2, 2], [1, 1])
            .expect("emit_max_pool2d");
        assert_eq!(out.shape(), vec![Some(1), Some(1), Some(2), Some(2)]);
        let graph = gb.compile(&[&out]).expect("compile max_pool2d_basic");

        let input_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let in_buf = Buffer::from_slice(&input_data, &[1, 1, 4, 4], DType::F32);
        let out_data = graph.run(&[&in_buf]);
        assert_eq!(out_data[0].as_slice::<f32>(), &[6.0, 8.0, 14.0, 16.0]);
    }

    /// global_avg_pool: 1x1x2x2 input → 1x1x1x1 average.
    ///
    /// Input: [[1, 2], [3, 4]] → average = (1+2+3+4)/4 = 2.5
    #[test]
    fn global_avg_pool_basic() {
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let input = gb.input(&[Some(1), Some(1), Some(2), Some(2)], DType::F32);
        let out = gb.emit_global_avg_pool(&input);
        assert_eq!(out.shape(), vec![Some(1), Some(1), Some(1), Some(1)]);
        let graph = gb.compile(&[&out]).expect("compile global_avg_pool");

        let in_buf = Buffer::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 2, 2], DType::F32);
        let out_data = graph.run(&[&in_buf]);
        let vals = out_data[0].as_slice::<f32>();
        assert_eq!(vals.len(), 1);
        assert!(
            (vals[0] - 2.5).abs() < 1e-5,
            "expected 2.5, got {}",
            vals[0]
        );
    }

    /// batch_norm with identity parameters: scale=1, bias=0, mean=0, var=1, eps=0.
    ///
    /// Result should equal input: (x - 0) / sqrt(1 + 0) * 1 + 0 = x.
    #[test]
    fn batch_norm_identity() {
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        // Input [1, 2, 2, 2], C=2.
        let input = gb.input(&[Some(1), Some(2), Some(2), Some(2)], DType::F32);
        let scale = gb.add_weight(&[Some(2)], DType::F32);
        let bias = gb.add_weight(&[Some(2)], DType::F32);
        let mean = gb.add_weight(&[Some(2)], DType::F32);
        let var = gb.add_weight(&[Some(2)], DType::F32);
        let out = gb.emit_batch_norm(&input, &scale, &bias, &mean, &var, 1e-5);
        assert_eq!(out.shape(), vec![Some(1), Some(2), Some(2), Some(2)]);
        let graph = gb.compile(&[&out]).expect("compile batch_norm");

        // Input: 8 values.
        let input_data: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let scale_data = vec![1.0f32, 1.0];
        let bias_data = vec![0.0f32, 0.0];
        let mean_data = vec![0.0f32, 0.0];
        let var_data = vec![1.0f32, 1.0];

        let in_buf = Buffer::from_slice(&input_data, &[1, 2, 2, 2], DType::F32);
        let sc_buf = Buffer::from_slice(&scale_data, &[2], DType::F32);
        let bi_buf = Buffer::from_slice(&bias_data, &[2], DType::F32);
        let mn_buf = Buffer::from_slice(&mean_data, &[2], DType::F32);
        let vr_buf = Buffer::from_slice(&var_data, &[2], DType::F32);

        let out_data = graph.run(&[&in_buf, &sc_buf, &bi_buf, &mn_buf, &vr_buf]);
        let vals = out_data[0].as_slice::<f32>();
        // With scale=1, bias=0, mean=0, var=1, eps≈0: out ≈ x / sqrt(1) = x.
        for (i, (&got, &expected)) in vals.iter().zip(input_data.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-4,
                "vals[{i}]: got {got}, expected {expected}"
            );
        }
    }

    // ── Meta ops tests ────────────────────────────────────────────────────

    #[test]
    fn shape_of_static() {
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let x = gb.input(&[Some(3), Some(4)], DType::F32);
        let dims = gb.emit_shape_of(&x);
        assert_eq!(dims.len(), 2);
    }

    #[test]
    fn constant_of_shape_static() {
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let x = gb.input(&[Some(6)], DType::F32); // dummy input to get shape_of values
        let dims = gb.emit_shape_of(&x);
        let filled = gb.emit_constant_of_shape(&dims, &[Some(6)], 5.0, DType::F32);
        let graph = gb.compile(&[&filled]).expect("compile constant_of_shape");

        let dummy = Buffer::from_slice(&[0.0f32; 6], &[6], DType::F32);
        let out = graph.run(&[&dummy]);
        assert_eq!(out[0].as_slice::<f32>(), &[5.0; 6]);
    }

    #[test]
    fn dense_constant() {
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let c = gb.emit_dense_constant("[1.0, 2.0, 3.0]", &[3], DType::F32);
        assert_eq!(c.shape(), vec![Some(3)]);
        let graph = gb.compile(&[&c]).expect("compile dense_constant");
        let out = graph.run(&[]);
        assert_eq!(out[0].as_slice::<f32>(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn arith_constant_scalar() {
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let c = gb.emit_arith_constant(42.0, DType::F32);
        assert_eq!(c.shape(), vec![]);
        assert_eq!(c.dtype(), DType::F32);
    }

    // ── Regression tests ──────────────────────────────────────────────────

    #[test]
    fn regression_broadcast_scalar_to_3d() {
        // Bug: canonicalize fused a broadcast linalg.generic with the binary
        // linalg.generic, producing identity maps on mismatched shapes
        // (tensor<1x1x1> vs tensor<2x3x4>). Fix: inline broadcast maps.
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(2), Some(3), Some(4)], DType::F32);
        let scalar = gb.input(&[Some(1), Some(1), Some(1)], DType::F32);
        let result = gb.emit_add(&a, &scalar);
        assert_eq!(result.shape(), vec![Some(2), Some(3), Some(4)]);

        let graph = gb.compile(&[&result]).expect("compile broadcast scalar");
        let a_buf = Buffer::from_slice(&[1.0f32; 24], &[2, 3, 4], DType::F32);
        let s_buf = Buffer::from_slice(&[10.0f32], &[1, 1, 1], DType::F32);
        let out = graph.run(&[&a_buf, &s_buf]);
        assert!(
            out[0]
                .as_slice::<f32>()
                .iter()
                .all(|&v| (v - 11.0).abs() < 1e-5)
        );
    }

    #[test]
    fn regression_broadcast_1d_to_3d_dynamic() {
        // Same bug but with dynamic dims: tensor<1x1x1> + tensor<?x?x768>
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[None, None, Some(768)], DType::F32);
        let scalar = gb.input(&[Some(1), Some(1), Some(1)], DType::F32);
        let result = gb.emit_sub(&a, &scalar);
        assert_eq!(result.shape(), vec![None, None, Some(768)]);
        // Shape check only — can't compile dynamic without concrete dims at runtime.
    }

    #[test]
    fn regression_reshape_infer_flat() {
        // Bug: emit_reshape panicked on target_shape=[-1] because known_product
        // was None (no known dims to divide by).
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(2), Some(3)], DType::F32);
        let flat = gb.emit_reshape(&a, &[-1]);
        assert_eq!(flat.shape(), vec![Some(6)]); // -1 → statically inferred: 2*3 = 6

        let graph = gb.compile(&[&flat]).expect("compile reshape flat");
        let a_buf = Buffer::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
        let out = graph.run(&[&a_buf]);
        assert_eq!(out[0].as_slice::<f32>(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn regression_matmul_3d_x_2d() {
        // Bug: emit_matmul (3,2) case used emit_expand_shape_1d_to_2d on a 2D
        // tensor, causing an assertion failure. Fix: use emit_unsqueeze instead.
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(2), Some(3), Some(4)], DType::F32); // [B=2, M=3, K=4]
        let b = gb.input(&[Some(4), Some(5)], DType::F32); // [K=4, N=5]
        let c = gb.emit_matmul(&a, &b);
        assert_eq!(c.shape(), vec![Some(2), Some(3), Some(5)]);

        let graph = gb.compile(&[&c]).expect("compile matmul 3dx2d");
        let a_data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let b_data: Vec<f32> = (0..20).map(|x| x as f32 * 0.1).collect();
        let a_buf = Buffer::from_slice(&a_data, &[2, 3, 4], DType::F32);
        let b_buf = Buffer::from_slice(&b_data, &[4, 5], DType::F32);
        let out = graph.run(&[&a_buf, &b_buf]);
        assert_eq!(out[0].shape().0, vec![2, 3, 5]);
        assert!(out[0].as_slice::<f32>().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn regression_matmul_2d_x_3d() {
        // Same fix needed for the (2,3) case.
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(3), Some(4)], DType::F32); // [M=3, K=4]
        let b = gb.input(&[Some(2), Some(4), Some(5)], DType::F32); // [B=2, K=4, N=5]
        let c = gb.emit_matmul(&a, &b);
        assert_eq!(c.shape(), vec![Some(2), Some(3), Some(5)]);

        let graph = gb.compile(&[&c]).expect("compile matmul 2dx3d");
        let a_data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let b_data: Vec<f32> = (0..40).map(|x| x as f32 * 0.1).collect();
        let a_buf = Buffer::from_slice(&a_data, &[3, 4], DType::F32);
        let b_buf = Buffer::from_slice(&b_data, &[2, 4, 5], DType::F32);
        let out = graph.run(&[&a_buf, &b_buf]);
        assert_eq!(out[0].shape().0, vec![2, 3, 5]);
        assert!(out[0].as_slice::<f32>().iter().all(|v| v.is_finite()));
    }

    // ── Sub-function tests (tasks 1.5, 1.6) ─────────────────────────────────

    #[test]
    fn subfunction_simple_matmul() {
        // Task 1.5: input → sub-function(matmul) → return.
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();

        let a = gb.input(&[Some(2), Some(3)], DType::F32);
        let b = gb.input(&[Some(3), Some(4)], DType::F32);

        // Start sub-function that receives a and b.
        let (handle, sub_args) = gb.begin_subfunction("chunk_0", &[&a, &b]);
        let product = gb.emit_matmul(&sub_args[0], &sub_args[1]);
        let results = gb.end_subfunction(handle, &[&product]);

        // results[0] is the matmul result in @compute's scope.
        let graph = gb
            .compile(&[&results[0]])
            .expect("compile with subfunction");

        // a: [[1,2,3],[4,5,6]], b: [[1,0,0,0],[0,1,0,0],[0,0,1,0]]
        let a_buf = Buffer::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
        let b_buf = Buffer::from_slice(
            &[
                1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            ],
            &[3, 4],
            DType::F32,
        );
        let out = graph.run(&[&a_buf, &b_buf]);
        assert_eq!(out[0].shape().0, vec![2, 4]);
        let data = out[0].as_slice::<f32>();
        // With identity-like b, result = [1,2,3,0; 4,5,6,0]
        assert_eq!(data, &[1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 0.0]);
    }

    #[test]
    fn subfunction_two_chained() {
        // Task 1.6: chunk_0 produces value, chunk_1 consumes it.
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();

        let x = gb.input(&[Some(4)], DType::F32);
        let y = gb.input(&[Some(4)], DType::F32);

        // chunk_0: add x + y
        let (h0, args0) = gb.begin_subfunction("chunk_0", &[&x, &y]);
        let sum = gb.emit_add(&args0[0], &args0[1]);
        let r0 = gb.end_subfunction(h0, &[&sum]);

        // chunk_1: mul result * y
        let (h1, args1) = gb.begin_subfunction("chunk_1", &[&r0[0], &y]);
        let product = gb.emit_mul(&args1[0], &args1[1]);
        let r1 = gb.end_subfunction(h1, &[&product]);

        let graph = gb.compile(&[&r1[0]]).expect("compile chained subfunctions");

        let x_buf = Buffer::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], DType::F32);
        let y_buf = Buffer::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[4], DType::F32);
        let out = graph.run(&[&x_buf, &y_buf]);
        let data = out[0].as_slice::<f32>();
        // (x+y)*y = (1+10)*10, (2+20)*20, (3+30)*30, (4+40)*40 = 110, 440, 990, 1760
        assert_eq!(data, &[110.0, 440.0, 990.0, 1760.0]);
    }

    // ── Coverage for previously-untested emit_* methods ──────────────────────

    #[test]
    fn reshape_with_tensor() {
        // Reshape [2,3] → [3,2] using a shape tensor
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let input = gb.input(&[Some(2), Some(3)], DType::F32);
        let shape_tensor = gb.emit_dense_constant("[3, 2]", &[2], DType::I64);
        let out = gb.emit_reshape_with_tensor(&input, shape_tensor.value(), &[Some(3), Some(2)]);
        let graph = gb.compile(&[&out]).expect("compile reshape_with_tensor");

        let buf = Buffer::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
        let result = graph.run(&[&buf]);
        assert_eq!(result[0].shape().0, vec![3, 2]);
        assert_eq!(result[0].as_slice::<f32>(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // Note: emit_dynamic_slice, emit_range, emit_tensor_extract_scalar, and
    // emit_resolve_reshape_dims work with raw MLIR Values at the index type
    // level. They are exercised through the ONNX integration tests (MNIST,
    // ResNet, GPT-2) which provide the correct type context.

    // ── Phase 2 HuggingFace ops ───────────────────────────────────────────────

    #[test]
    fn silu_basic() {
        // silu(x) = x / (1 + exp(-x))
        // Reference values computed as: x * sigmoid(x)
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let x = gb.input(&[Some(4)], DType::F32);
        let out = gb.emit_silu(&x);
        let graph = gb.compile(&[&out]).expect("compile silu");

        let data = [0.0f32, 1.0, -1.0, 2.0];
        let buf = Buffer::from_slice(&data, &[4], DType::F32);
        let result = graph.run(&[&buf]);
        let got = result[0].as_slice::<f32>();

        // silu(x) = x * sigmoid(x)
        let expected: Vec<f32> = data.iter().map(|&x| x / (1.0 + (-x).exp())).collect();
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!(
                (g - e).abs() < 1e-5,
                "silu({}) got {g} expected {e}",
                // recover input from expected: silu is monotone so we just show index
                e
            );
        }
    }

    #[test]
    fn rms_norm_basic() {
        // RMSNorm([1, 2, 3, 4], weight=[1,1,1,1], eps=1e-6)
        // rms = sqrt(mean([1,4,9,16]) + eps) = sqrt(7.5 + eps)
        // normalized = x / rms; output = normalized * weight = normalized
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        // shape: [1, 4]  — batch=1, hidden=4
        let x = gb.input(&[Some(1), Some(4)], DType::F32);
        let w = gb.input(&[Some(1), Some(4)], DType::F32);
        let out = gb.emit_rms_norm(&x, &w, 1e-6);
        let graph = gb.compile(&[&out]).expect("compile rms_norm");

        let x_data = [1.0f32, 2.0, 3.0, 4.0];
        let w_data = [1.0f32, 1.0, 1.0, 1.0];
        let x_buf = Buffer::from_slice(&x_data, &[1, 4], DType::F32);
        let w_buf = Buffer::from_slice(&w_data, &[1, 4], DType::F32);
        let result = graph.run(&[&x_buf, &w_buf]);
        let got = result[0].as_slice::<f32>();

        // Reference: rms = sqrt((1+4+9+16)/4) = sqrt(7.5)
        let rms = (7.5f32 + 1e-6).sqrt();
        let expected: Vec<f32> = x_data.iter().map(|&v| v / rms).collect();
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-4, "rms_norm got {g} expected {e}");
        }
    }

    #[test]
    fn repeat_kv_repeats_2() {
        // Input: [1, 2, 2, 3]  (batch=1, seq=2, kv_heads=2, head_dim=3), repeats=2
        // Expected output: [1, 2, 4, 3]
        // Each KV head is repeated twice: [h0, h0, h1, h1]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let x = gb.input(&[Some(1), Some(2), Some(2), Some(3)], DType::F32);
        let out = gb.emit_repeat_kv(&x, 2);
        assert_eq!(out.shape(), vec![Some(1), Some(2), Some(4), Some(3)]);
        let graph = gb.compile(&[&out]).expect("compile repeat_kv");

        // Input data: head 0 = [1,2,3; 7,8,9], head 1 = [4,5,6; 10,11,12]
        // Layout [1,2,2,3]: batch0, seq0→[h0=[1,2,3], h1=[4,5,6]]; seq1→[h0=[7,8,9], h1=[10,11,12]]
        #[rustfmt::skip]
        let data = [
            1.0f32, 2.0, 3.0,   // [0,0,0,:]
            4.0,    5.0, 6.0,   // [0,0,1,:]
            7.0,    8.0, 9.0,   // [0,1,0,:]
            10.0,   11.0, 12.0, // [0,1,1,:]
        ];
        let buf = Buffer::from_slice(&data, &[1, 2, 2, 3], DType::F32);
        let result = graph.run(&[&buf]);
        assert_eq!(result[0].shape().0, vec![1, 2, 4, 3]);

        // Expected: each kv head appears twice → [h0,h0,h1,h1] per seq pos
        #[rustfmt::skip]
        let expected = [
            1.0f32, 2.0, 3.0,   // [0,0,0,:] = h0
            1.0,    2.0, 3.0,   // [0,0,1,:] = h0 (repeat)
            4.0,    5.0, 6.0,   // [0,0,2,:] = h1
            4.0,    5.0, 6.0,   // [0,0,3,:] = h1 (repeat)
            7.0,    8.0, 9.0,   // [0,1,0,:] = h0
            7.0,    8.0, 9.0,   // [0,1,1,:] = h0 (repeat)
            10.0,   11.0, 12.0, // [0,1,2,:] = h1
            10.0,   11.0, 12.0, // [0,1,3,:] = h1 (repeat)
        ];
        assert_eq!(result[0].as_slice::<f32>(), &expected);
    }

    #[test]
    fn repeat_kv_repeats_1_noop() {
        // repeats=1 should return the input unchanged
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let x = gb.input(&[Some(1), Some(2), Some(2), Some(3)], DType::F32);
        let out = gb.emit_repeat_kv(&x, 1);
        // Same value, same shape
        assert_eq!(out.shape(), vec![Some(1), Some(2), Some(2), Some(3)]);
    }

    #[test]
    fn embedding_basic() {
        // weight: [4, 3]  (vocab=4, hidden=3)
        // indices: [2, 2]  (batch=2, seq=2)
        // output: [2, 2, 3]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let weight = gb.input(&[Some(4), Some(3)], DType::F32);
        let indices = gb.input(&[Some(2), Some(2)], DType::I64);
        let out = gb.emit_embedding(&weight, &indices);
        assert_eq!(out.shape(), vec![Some(2), Some(2), Some(3)]);
        let graph = gb.compile(&[&out]).expect("compile embedding");

        // weight rows: row0=[0,0,0], row1=[1,1,1], row2=[2,2,2], row3=[3,3,3]
        let w_data = [
            0.0f32, 0.0, 0.0, // row 0
            1.0, 1.0, 1.0, // row 1
            2.0, 2.0, 2.0, // row 2
            3.0, 3.0, 3.0, // row 3
        ];
        // indices: [[0, 2], [1, 3]]
        let idx_data = [0i64, 2, 1, 3];
        let w_buf = Buffer::from_slice(&w_data, &[4, 3], DType::F32);
        let i_buf = Buffer::from_slice(&idx_data, &[2, 2], DType::I64);
        let result = graph.run(&[&w_buf, &i_buf]);
        assert_eq!(result[0].shape().0, vec![2, 2, 3]);
        // Expected: row0, row2, row1, row3
        let expected = [
            0.0f32, 0.0, 0.0, // [0,0,:] = row0
            2.0, 2.0, 2.0, // [0,1,:] = row2
            1.0, 1.0, 1.0, // [1,0,:] = row1
            3.0, 3.0, 3.0, // [1,1,:] = row3
        ];
        assert_eq!(result[0].as_slice::<f32>(), &expected);
    }

    #[test]
    fn rope_basic() {
        // x:   [1, 2, 1, 4] — batch=1, seq=2, heads=1, head_dim=4
        // cos: [2, 2]        — seq=2, head_dim/2=2
        // sin: [2, 2]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let x = gb.input(&[Some(1), Some(2), Some(1), Some(4)], DType::F32);
        let cos = gb.input(&[Some(2), Some(2)], DType::F32);
        let sin = gb.input(&[Some(2), Some(2)], DType::F32);
        let out = gb.emit_rope(&x, &cos, &sin);
        assert_eq!(out.shape(), vec![Some(1), Some(2), Some(1), Some(4)]);
        let graph = gb.compile(&[&out]).expect("compile rope");

        // x = [1, 2, 3, 4,  5, 6, 7, 8]  (two positions, head_dim=4)
        // even indices: [1, 3, 5, 7], odd indices: [2, 4, 6, 8]
        let x_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        // cos/sin for position 0 and 1:
        // cos = [[1.0, 0.5], [0.0, 1.0]]
        // sin = [[0.0, 0.5], [1.0, 0.0]]
        let cos_data: Vec<f32> = vec![1.0, 0.5, 0.0, 1.0];
        let sin_data: Vec<f32> = vec![0.0, 0.5, 1.0, 0.0];

        let x_buf = Buffer::from_slice(&x_data, &[1, 2, 1, 4], DType::F32);
        let cos_buf = Buffer::from_slice(&cos_data, &[2, 2], DType::F32);
        let sin_buf = Buffer::from_slice(&sin_data, &[2, 2], DType::F32);
        let result = graph.run(&[&x_buf, &cos_buf, &sin_buf]);
        let out = result[0].as_slice::<f32>();

        // Position 0: cos=[1.0, 0.5], sin=[0.0, 0.5]
        //   x_even=[1, 3], x_odd=[2, 4]
        //   out_even = [1*1.0 - 2*0.0, 3*0.5 - 4*0.5] = [1.0, -0.5]
        //   out_odd  = [1*0.0 + 2*1.0, 3*0.5 + 4*0.5] = [2.0, 3.5]
        //   interleaved: [1.0, 2.0, -0.5, 3.5]
        // Position 1: cos=[0.0, 1.0], sin=[1.0, 0.0]
        //   x_even=[5, 7], x_odd=[6, 8]
        //   out_even = [5*0.0 - 6*1.0, 7*1.0 - 8*0.0] = [-6.0, 7.0]
        //   out_odd  = [5*1.0 + 6*0.0, 7*0.0 + 8*1.0] = [5.0, 8.0]
        //   interleaved: [-6.0, 5.0, 7.0, 8.0]
        let expected: Vec<f32> = vec![1.0, 2.0, -0.5, 3.5, -6.0, 5.0, 7.0, 8.0];
        for (i, (got, want)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-5,
                "rope mismatch at [{i}]: got {got}, want {want}"
            );
        }
    }
}
