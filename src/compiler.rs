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
        operation::{OperationBuilder, OperationLike},
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
    op::{NodeId, Op},
    runtime::CompiledGraph,
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
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Verification => write!(f, "MLIR module verification failed"),
            Self::Pass(e) => write!(f, "lowering pass failed: {e}"),
            Self::EmptyTrace => write!(f, "trace is empty"),
            Self::NoOutputs => write!(f, "no output nodes specified"),
            Self::AttributeParse(s) => write!(f, "failed to parse MLIR attribute: {s}"),
        }
    }
}

impl std::error::Error for CompileError {}

/// Compiles a `Trace` into a JIT-ready `CompiledGraph`.
pub struct Compiler;

impl Compiler {
    /// Compile a trace into an MLIR ExecutionEngine ready to invoke.
    ///
    /// `outputs` is a list of node IDs whose results should become output
    /// memref arguments of the generated function. Currently only a single
    /// output is supported.
    pub fn compile(trace: &Trace, outputs: &[NodeId]) -> Result<CompiledGraph, CompileError> {
        if trace.ops().is_empty() {
            return Err(CompileError::EmptyTrace);
        }
        if outputs.is_empty() {
            return Err(CompileError::NoOutputs);
        }

        let context = create_context();
        let location = Location::unknown(&context);
        let mut module = Module::new(location);
        let (num_inputs, output_shape, output_dtype) =
            build_module(&context, &module, trace, outputs)?;

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
                one-shot-bufferize{function-boundary-type-conversion=identity-layout-map unknown-type-conversion=identity-layout-map},\
                convert-linalg-to-loops,\
                convert-scf-to-cf,\
                lower-affine,\
                convert-math-to-llvm,\
                expand-strided-metadata,\
                finalize-memref-to-llvm,\
                convert-arith-to-llvm,\
                convert-index-to-llvm,\
                convert-cf-to-llvm,\
                convert-func-to-llvm,\
                reconcile-unrealized-casts\
            )",
        )
        .map_err(CompileError::Pass)?;

        pass_manager.run(&mut module).map_err(CompileError::Pass)?;

        // ---- Create ExecutionEngine -------------------------------------------
        // Ensure libmlir_runner_utils.so is loaded globally so the ORC JIT
        // can resolve `memrefCopy` emitted by padded conv2d bufferization.
        ensure_runner_utils_loaded();
        let engine = melior::ExecutionEngine::new(&module, 2, &[], false);

        Ok(CompiledGraph::new(
            engine,
            num_inputs,
            output_shape,
            output_dtype,
        ))
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

/// Ensure `libmlir_c_runner_utils.so` is loaded into the process's global
/// dynamic-linker namespace before the ORC JIT tries to link it.
///
/// LLVM's ORC JIT resolves external symbols via `DynamicLibrarySearchGenerator`
/// which scans `dl_iterate_phdr`. `memrefCopy` — emitted by bufferization
/// lowering for padded conv2d — lives in `libmlir_c_runner_utils.so`.
/// Loading it with `RTLD_GLOBAL` makes its symbols visible to the JIT's
/// `GetForCurrentProcess` generator.
fn ensure_runner_utils_loaded() {
    use std::ffi::CString;
    use std::sync::OnceLock;

    static LOADED: OnceLock<()> = OnceLock::new();
    LOADED.get_or_init(|| {
        const DEFAULT_PATH: &str = "/usr/lib/llvm/21/lib64/libmlir_c_runner_utils.so";
        let lib_path =
            std::env::var("MLIR_RUNNER_UTILS_PATH").unwrap_or_else(|_| DEFAULT_PATH.to_string());
        // SAFETY: dlopen is thread-safe; we call it once via OnceLock and
        // never dlclose, so the symbols remain valid for the process lifetime.
        unsafe {
            let lib_c = CString::new(lib_path.as_str()).unwrap();
            let handle = libc::dlopen(lib_c.as_ptr(), libc::RTLD_LAZY | libc::RTLD_GLOBAL);
            if handle.is_null() {
                eprintln!(
                    "warning: failed to load {lib_path} — conv2d/padded ops may fail at JIT link time. \
                     Set MLIR_RUNNER_UTILS_PATH to override."
                );
            }
        }
    });
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
// verifies it. Returns `(num_inputs, output_shape, output_dtype)`.
//
// `context` and `module` are caller-owned so their lifetimes do not cross a
// function boundary (which would require a self-referential return).

fn build_module<'c>(
    context: &'c Context,
    module: &Module<'c>,
    trace: &Trace,
    outputs: &[NodeId],
) -> Result<(usize, crate::Shape, DType), CompileError> {
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
    let output_id = outputs[0];
    let output_op = trace.get(output_id);
    let output_shape = output_op.shape().clone();
    let output_dtype = output_op.dtype();

    let location = Location::unknown(context);

    {
        let elem_type = output_dtype.to_mlir_type(context);

        let mut arg_types: Vec<(melior::ir::Type, Location)> = Vec::new();
        for (_, op) in &input_ops {
            let Op::Input { shape, dtype, .. } = op else {
                unreachable!()
            };
            let dims: Vec<i64> = shape.0.iter().map(|&d| d as i64).collect();
            let mref = MemRefType::new(dtype.to_mlir_type(context), &dims, None, None);
            arg_types.push((mref.into(), location));
        }
        let out_dims: Vec<i64> = output_shape.0.iter().map(|&d| d as i64).collect();
        let out_mref = MemRefType::new(elem_type, &out_dims, None, None);
        arg_types.push((out_mref.into(), location));

        let func_arg_types: Vec<melior::ir::Type> = arg_types.iter().map(|(t, _)| *t).collect();
        let function_type = FunctionType::new(context, &func_arg_types, &[]);

        let body_block = Block::new(&arg_types);

        emit_tensor_ops(trace, output_id, num_inputs, &body_block, location, context)?;

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

    if !module.as_operation().verify() {
        return Err(CompileError::Verification);
    }

    Ok((num_inputs, output_shape, output_dtype))
}

// ── Helper: build a RankedTensorType for an arbitrary shape ──────────────────

fn make_ranked_tensor_type<'c>(
    context: &'c Context,
    shape: &[u64],
    dtype: DType,
) -> melior::ir::Type<'c> {
    RankedTensorType::new(shape, dtype.to_mlir_type(context), None).into()
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
    let init_val = body_block
        .append_operation(
            OperationBuilder::new("tensor.empty", location)
                .add_results(&[tensor_type])
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        )
        .result(0)
        .unwrap()
        .into();

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
    let n = new_shape.len();
    let dims_str = new_shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let static_shape_attr_str = format!("array<i64: {dims_str}>");
    let static_shape_attr = Attribute::parse(context, &static_shape_attr_str)
        .ok_or_else(|| CompileError::AttributeParse(static_shape_attr_str.clone()))?;
    let _ = n;

    let expanded = body_block
        .append_operation(
            OperationBuilder::new("tensor.expand_shape", location)
                .add_operands(&[val])
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
    let k = lhs_shape[1];
    let n = rhs_shape[1];

    let lhs_3d = emit_tosa_reshape(context, body_block, lhs_val, &[1, m, k], dtype, location)?;
    let rhs_3d = emit_tosa_reshape(context, body_block, rhs_val, &[1, k, n], dtype, location)?;

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

    let result_val = emit_tosa_reshape(context, body_block, out_3d, output_shape, dtype, location)?;
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
    let init_val: melior::ir::Value = body_block
        .append_operation(
            OperationBuilder::new("tensor.empty", location)
                .add_results(&[tensor_type])
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        )
        .result(0)
        .unwrap()
        .into();

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
    let init_val: melior::ir::Value = body_block
        .append_operation(
            OperationBuilder::new("tensor.empty", location)
                .add_results(&[output_tensor_type])
                .build()
                .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
        )
        .result(0)
        .unwrap()
        .into();

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

/// Walk the trace linearly, emitting tensor-level ops into `body_block`.
///
/// For each `Input` op: emits `bufferization.to_tensor` with `restrict`.
/// For each compute op: emits `tensor.empty` + `linalg.generic`.
/// After all ops: emits `bufferization.to_buffer` + `memref.copy` for the output.
fn emit_tensor_ops<'c>(
    trace: &Trace,
    output_id: NodeId,
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
                let result_val = match dtype {
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
                let result_val = match dtype {
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
                let result_val = match dtype {
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
                shape,
                dtype,
            } => {
                let input_val = *values.get(input).expect("input not yet emitted");
                let result_val =
                    emit_tosa_reshape(context, body_block, input_val, &shape.0, *dtype, location)?;
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
                let n = input_shape[0];
                let oh = shape.0[2];
                let ow = shape.0[3];
                let nhwc_out_shape = [n, oh, ow, co];
                let nhwc_out_type = make_ranked_tensor_type(context, &nhwc_out_shape, *dtype);

                // TOSA conv2d pad order: [pad_top, pad_bottom, pad_left, pad_right]
                // Our pads: [pad_top, pad_left, pad_bottom, pad_right]
                let pad_attr_str = format!(
                    "array<i64: {}, {}, {}, {}>",
                    pads[0], pads[2], pads[1], pads[3]
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
                let conv_nhwc: melior::ir::Value = body_block
                    .append_operation(
                        OperationBuilder::new("tosa.conv2d", location)
                            .add_operands(&[input_nhwc, kernel_ohwi, bias_val, input_zp, weight_zp])
                            .add_results(&[nhwc_out_type])
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
        }
    }

    // ---- Output boundary: bufferization.to_buffer + memref.copy ---------------
    let result_tensor = *values.get(&output_id).expect("output node not emitted");

    let output_op = trace.get(output_id);
    let out_elem_type = output_op.dtype().to_mlir_type(context);
    let dims: Vec<i64> = output_op.shape().0.iter().map(|&d| d as i64).collect();
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

    // The output func arg immediately follows all input args.
    let out_arg: melior::ir::Value = body_block.argument(num_inputs).unwrap().into();
    body_block.append_operation(
        OperationBuilder::new("memref.copy", location)
            .add_operands(&[result_memref, out_arg])
            .build()
            .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
    );

    Ok(())
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

        let result = Compiler::compile(&trace, &[c.id]);
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

        let result = Compiler::compile(&trace, &[d.id]);
        assert!(result.is_ok(), "compile failed: {:?}", result.err());
    }

    #[test]
    fn compile_sub() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F32);
        let b = Tensor::new(&[4], DType::F32);
        let c = &a - &b;
        let trace = take_trace();

        let result = Compiler::compile(&trace, &[c.id]);
        assert!(result.is_ok(), "compile failed: {:?}", result.err());
    }

    #[test]
    fn compile_mul() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F32);
        let b = Tensor::new(&[4], DType::F32);
        let c = &a * &b;
        let trace = take_trace();

        let result = Compiler::compile(&trace, &[c.id]);
        assert!(result.is_ok(), "compile failed: {:?}", result.err());
    }

    #[test]
    fn compile_div() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F32);
        let b = Tensor::new(&[4], DType::F32);
        let c = &a / &b;
        let trace = take_trace();

        let result = Compiler::compile(&trace, &[c.id]);
        assert!(result.is_ok(), "compile failed: {:?}", result.err());
    }

    #[test]
    fn compile_neg() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F32);
        let b = -&a;
        let trace = take_trace();

        let result = Compiler::compile(&trace, &[b.id]);
        assert!(result.is_ok(), "compile failed: {:?}", result.err());
    }

    #[test]
    fn compile_relu() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F32);
        let b = a.relu();
        let trace = take_trace();

        let result = Compiler::compile(&trace, &[b.id]);
        assert!(result.is_ok(), "compile failed: {:?}", result.err());
    }

    #[test]
    fn compile_div_i32() {
        begin_trace();
        let a = Tensor::new(&[4], DType::I32);
        let b = Tensor::new(&[4], DType::I32);
        let c = &a / &b;
        let trace = take_trace();

        let result = Compiler::compile(&trace, &[c.id]);
        assert!(result.is_ok(), "compile failed: {:?}", result.err());
    }

    #[test]
    fn compile_neg_i32() {
        begin_trace();
        let a = Tensor::new(&[4], DType::I32);
        let b = -&a;
        let trace = take_trace();

        let result = Compiler::compile(&trace, &[b.id]);
        assert!(result.is_ok(), "compile failed: {:?}", result.err());
    }

    #[test]
    fn compile_relu_i32() {
        begin_trace();
        let a = Tensor::new(&[4], DType::I32);
        let b = a.relu();
        let trace = take_trace();

        let result = Compiler::compile(&trace, &[b.id]);
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

        let result = Compiler::compile(&trace, &[c.id]);
        assert!(result.is_ok(), "compile failed: {:?}", result.err());
    }

    #[test]
    fn compile_add_rank3() {
        begin_trace();
        let a = Tensor::new(&[2, 3, 4], DType::F32);
        let b = Tensor::new(&[2, 3, 4], DType::F32);
        let c = &a + &b;
        let trace = take_trace();

        let result = Compiler::compile(&trace, &[c.id]);
        assert!(result.is_ok(), "compile failed: {:?}", result.err());
    }

    #[test]
    fn compile_neg_rank2() {
        begin_trace();
        let a = Tensor::new(&[2, 3], DType::F32);
        let b = -&a;
        let trace = take_trace();

        let result = Compiler::compile(&trace, &[b.id]);
        assert!(result.is_ok(), "compile failed: {:?}", result.err());
    }

    // ── TOSA spike tests (milestone 2, task 1.2) ──────────────────────────────
    //
    // These tests validate that the TOSA → linalg → LLVM → JIT pipeline works
    // end-to-end. They build MLIR modules by hand (not via the trace compiler)
    // to keep the TOSA lowering path isolated from the existing linalg path.
    //
    // Pass pipeline (TOSA lowering prepended to the existing chain):
    //   tosa-make-broadcastable, tosa-to-linalg-named, tosa-to-linalg,
    //   tosa-to-arith, tosa-to-tensor,
    //   one-shot-bufferize{…}, convert-linalg-to-loops, convert-scf-to-cf,
    //   convert-math-to-llvm, finalize-memref-to-llvm, convert-arith-to-llvm,
    //   convert-index-to-llvm, convert-cf-to-llvm, convert-func-to-llvm,
    //   reconcile-unrealized-casts

    /// Build and run the TOSA lowering pass pipeline on a module.
    ///
    /// Returns `Err(String)` with the pass error message if any pass fails.
    fn run_tosa_pipeline(module: &mut Module, context: &Context) -> Result<(), String> {
        register_all_passes();
        let pm = pass::PassManager::new(context);
        parse_pass_pipeline(
            pm.as_operation_pass_manager(),
            "builtin.module(\
                func.func(\
                    tosa-make-broadcastable,\
                    tosa-to-linalg-named,\
                    tosa-to-linalg,\
                    tosa-to-arith,\
                    tosa-to-tensor\
                ),\
                one-shot-bufferize{function-boundary-type-conversion=identity-layout-map unknown-type-conversion=identity-layout-map},\
                convert-linalg-to-loops,\
                convert-scf-to-cf,\
                lower-affine,\
                convert-math-to-llvm,\
                expand-strided-metadata,\
                finalize-memref-to-llvm,\
                convert-arith-to-llvm,\
                convert-index-to-llvm,\
                convert-cf-to-llvm,\
                convert-func-to-llvm,\
                reconcile-unrealized-casts\
            )",
        )
        .map_err(|e| format!("pipeline parse failed: {e}"))?;
        pm.run(module).map_err(|e| format!("pass failed: {e}"))?;
        Ok(())
    }

    /// Execute a compiled module's `compute` function with the given flat f32
    /// input slices and return the flat f32 output.
    ///
    /// All buffers must have the same flat element count `n`.
    unsafe fn jit_run_f32(
        engine: &melior::ExecutionEngine,
        inputs: &[&[f32]],
        n: usize,
    ) -> Vec<f32> {
        unsafe {
            use crate::runtime::build_memref_descriptor;

            let shape = vec![n as i64];
            let strides = vec![1i64];

            // Build input descriptors (treat &[f32] data as *mut u8).
            let mut input_data_vecs: Vec<Vec<f32>> = inputs.iter().map(|s| s.to_vec()).collect();
            let mut input_descs: Vec<Vec<u8>> = input_data_vecs
                .iter_mut()
                .map(|v| build_memref_descriptor(v.as_mut_ptr() as *mut u8, &shape, &strides))
                .collect();

            // Allocate output buffer.
            let mut output_data: Vec<f32> = vec![0.0f32; n];
            let mut output_desc =
                build_memref_descriptor(output_data.as_mut_ptr() as *mut u8, &shape, &strides);

            // Build args array: &mut ptr_to_desc for each descriptor.
            let mut desc_ptrs: Vec<*mut u8> =
                input_descs.iter_mut().map(|d| d.as_mut_ptr()).collect();
            let mut out_desc_ptr = output_desc.as_mut_ptr();

            let mut args: Vec<*mut ()> = desc_ptrs
                .iter_mut()
                .map(|p| p as *mut *mut u8 as *mut ())
                .collect();
            args.push(&mut out_desc_ptr as *mut *mut u8 as *mut ());

            engine
                .invoke_packed("compute", &mut args)
                .expect("JIT invocation failed");

            output_data
        }
    }

    // ── spike_tosa_add_pipeline ───────────────────────────────────────────────
    //
    // Validates that:
    //   - A hand-built MLIR module using tosa.add compiles through the TOSA
    //     lowering pipeline without errors.
    //   - The JIT-executed result matches the expected element-wise sum.
    //
    // IR shape:
    //   func @compute(%a: memref<4xf32>, %b: memref<4xf32>, %out: memref<4xf32>)
    //     %ta = bufferization.to_tensor %a restrict : memref<4xf32>
    //     %tb = bufferization.to_tensor %b restrict : memref<4xf32>
    //     %tc = tosa.add %ta, %tb : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    //     %mc = bufferization.to_buffer %tc : tensor<4xf32> -> memref<4xf32>
    //     memref.copy %mc, %out : memref<4xf32> to memref<4xf32>
    //     return
    #[test]
    fn spike_tosa_add_pipeline() {
        let context = create_context();
        let location = Location::unknown(&context);
        let mut module = Module::new(location);

        let f32_type = melior::ir::r#type::IntegerType::new(&context, 32);
        // Use f32 — obtain via DType helper.
        let f32_mlir = DType::F32.to_mlir_type(&context);
        let tensor_type: melior::ir::Type = RankedTensorType::new(&[4u64], f32_mlir, None).into();
        let memref_type: melior::ir::Type = MemRefType::new(f32_mlir, &[4i64], None, None).into();
        let _ = f32_type; // silence unused warning

        {
            // Function: (%a: memref<4xf32>, %b: memref<4xf32>, %out: memref<4xf32>) -> ()
            let arg_types = [
                (memref_type, location),
                (memref_type, location),
                (memref_type, location),
            ];
            let function_type =
                FunctionType::new(&context, &[memref_type, memref_type, memref_type], &[]);

            let body_block = Block::new(&arg_types);

            // bufferization.to_tensor %a restrict
            let ta: melior::ir::Value = body_block
                .append_operation(
                    OperationBuilder::new("bufferization.to_tensor", location)
                        .add_operands(&[body_block.argument(0).unwrap().into()])
                        .add_results(&[tensor_type])
                        .add_attributes(&[(
                            Identifier::new(&context, "restrict"),
                            Attribute::unit(&context),
                        )])
                        .build()
                        .expect("bufferization.to_tensor a"),
                )
                .result(0)
                .unwrap()
                .into();

            // bufferization.to_tensor %b restrict
            let tb: melior::ir::Value = body_block
                .append_operation(
                    OperationBuilder::new("bufferization.to_tensor", location)
                        .add_operands(&[body_block.argument(1).unwrap().into()])
                        .add_results(&[tensor_type])
                        .add_attributes(&[(
                            Identifier::new(&context, "restrict"),
                            Attribute::unit(&context),
                        )])
                        .build()
                        .expect("bufferization.to_tensor b"),
                )
                .result(0)
                .unwrap()
                .into();

            // tosa.add %ta, %tb
            let tc: melior::ir::Value = body_block
                .append_operation(
                    OperationBuilder::new("tosa.add", location)
                        .add_operands(&[ta, tb])
                        .add_results(&[tensor_type])
                        .build()
                        .expect("tosa.add"),
                )
                .result(0)
                .unwrap()
                .into();

            // bufferization.to_buffer %tc
            let mc: melior::ir::Value = body_block
                .append_operation(
                    OperationBuilder::new("bufferization.to_buffer", location)
                        .add_operands(&[tc])
                        .add_results(&[memref_type])
                        .build()
                        .expect("bufferization.to_buffer"),
                )
                .result(0)
                .unwrap()
                .into();

            // memref.copy %mc, %out
            body_block.append_operation(
                OperationBuilder::new("memref.copy", location)
                    .add_operands(&[mc, body_block.argument(2).unwrap().into()])
                    .build()
                    .expect("memref.copy"),
            );

            body_block.append_operation(func::r#return(&[], location));

            let region = Region::new();
            region.append_block(body_block);

            let function = func::func(
                &context,
                StringAttribute::new(&context, "compute"),
                TypeAttribute::new(function_type.into()),
                region,
                &[(
                    Identifier::new(&context, "llvm.emit_c_interface"),
                    Attribute::unit(&context),
                )],
                location,
            );
            module.body().append_operation(function);
        }

        assert!(
            module.as_operation().verify(),
            "MLIR module verification failed before passes"
        );

        run_tosa_pipeline(&mut module, &context).expect("TOSA pipeline failed");

        // JIT execute: [1,2,3,4] + [10,20,30,40] = [11,22,33,44]
        let engine = melior::ExecutionEngine::new(&module, 2, &[], false);
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [10.0f32, 20.0, 30.0, 40.0];
        let result = unsafe { jit_run_f32(&engine, &[&a, &b], 4) };
        assert_eq!(result, vec![11.0f32, 22.0, 33.0, 44.0]);
    }

    // ── spike_tosa_mixed_pipeline ─────────────────────────────────────────────
    //
    // Validates that tosa.add and linalg.generic can coexist in the same
    // module and both lower correctly through the mixed pipeline.
    //
    // IR shape:
    //   func @compute(%a: memref<4xf32>, %b: memref<4xf32>,
    //                 %c: memref<4xf32>, %out: memref<4xf32>)
    //     %ta = bufferization.to_tensor %a restrict
    //     %tb = bufferization.to_tensor %b restrict
    //     %tc = bufferization.to_tensor %c restrict
    //     %tsum = tosa.add %ta, %tb          -- tosa op
    //     %tdiv = linalg.generic(%tsum, %tc) -- divide sum by c
    //     %mc = bufferization.to_buffer %tdiv
    //     memref.copy %mc, %out
    //     return
    //
    // Input: a=[2,4,6,8], b=[0,0,0,0] (so tosa.add gives [2,4,6,8]),
    //        c=[1,2,3,4] (linalg div gives [2,2,2,2]).
    #[test]
    fn spike_tosa_mixed_pipeline() {
        let context = create_context();
        let location = Location::unknown(&context);
        let mut module = Module::new(location);

        let f32_mlir = DType::F32.to_mlir_type(&context);
        let tensor_type: melior::ir::Type = RankedTensorType::new(&[4u64], f32_mlir, None).into();
        let memref_type: melior::ir::Type = MemRefType::new(f32_mlir, &[4i64], None, None).into();

        {
            // Function: (%a, %b, %c, %out : all memref<4xf32>) -> ()
            let arg_types = [
                (memref_type, location),
                (memref_type, location),
                (memref_type, location),
                (memref_type, location),
            ];
            let function_type = FunctionType::new(
                &context,
                &[memref_type, memref_type, memref_type, memref_type],
                &[],
            );

            let body_block = Block::new(&arg_types);

            // bufferization.to_tensor for each input
            let to_tensor = |arg_idx: usize| -> melior::ir::Value {
                body_block
                    .append_operation(
                        OperationBuilder::new("bufferization.to_tensor", location)
                            .add_operands(&[body_block.argument(arg_idx).unwrap().into()])
                            .add_results(&[tensor_type])
                            .add_attributes(&[(
                                Identifier::new(&context, "restrict"),
                                Attribute::unit(&context),
                            )])
                            .build()
                            .expect("bufferization.to_tensor"),
                    )
                    .result(0)
                    .unwrap()
                    .into()
            };

            let ta = to_tensor(0);
            let tb = to_tensor(1);
            let tc = to_tensor(2);

            // tosa.add %ta, %tb  -> %tsum
            let tsum: melior::ir::Value = body_block
                .append_operation(
                    OperationBuilder::new("tosa.add", location)
                        .add_operands(&[ta, tb])
                        .add_results(&[tensor_type])
                        .build()
                        .expect("tosa.add"),
                )
                .result(0)
                .unwrap()
                .into();

            // linalg.generic: divf(%tsum, %tc) -> %tdiv
            // tensor.empty for the output slot
            let init_val: melior::ir::Value = body_block
                .append_operation(
                    OperationBuilder::new("tensor.empty", location)
                        .add_results(&[tensor_type])
                        .build()
                        .expect("tensor.empty"),
                )
                .result(0)
                .unwrap()
                .into();

            let identity_map = make_identity_map(&context, 1).expect("identity map");
            let indexing_maps =
                ArrayAttribute::new(&context, &[identity_map, identity_map, identity_map]);
            let iterator_types = make_iterator_types(&context, 1).expect("iterator types");
            let segment_sizes =
                Attribute::parse(&context, "array<i32: 2, 1>").expect("segment sizes");

            let linalg_region = {
                let linalg_block = Block::new(&[
                    (f32_mlir, location),
                    (f32_mlir, location),
                    (f32_mlir, location),
                ]);
                let lhs_elem: melior::ir::Value = linalg_block.argument(0).unwrap().into();
                let rhs_elem: melior::ir::Value = linalg_block.argument(1).unwrap().into();
                let div_val: melior::ir::Value = linalg_block
                    .append_operation(arith::divf(lhs_elem, rhs_elem, location))
                    .result(0)
                    .unwrap()
                    .into();
                linalg_block.append_operation(
                    OperationBuilder::new("linalg.yield", location)
                        .add_operands(&[div_val])
                        .build()
                        .expect("linalg.yield"),
                );
                let r = Region::new();
                r.append_block(linalg_block);
                r
            };

            let tdiv: melior::ir::Value = body_block
                .append_operation(
                    OperationBuilder::new("linalg.generic", location)
                        .add_operands(&[tsum, tc, init_val])
                        .add_results(&[tensor_type])
                        .add_attributes(&[
                            (
                                Identifier::new(&context, "indexing_maps"),
                                indexing_maps.into(),
                            ),
                            (Identifier::new(&context, "iterator_types"), iterator_types),
                            (
                                Identifier::new(&context, "operand_segment_sizes"),
                                segment_sizes,
                            ),
                        ])
                        .add_regions([linalg_region])
                        .build()
                        .expect("linalg.generic"),
                )
                .result(0)
                .unwrap()
                .into();

            // bufferization.to_buffer %tdiv
            let mc: melior::ir::Value = body_block
                .append_operation(
                    OperationBuilder::new("bufferization.to_buffer", location)
                        .add_operands(&[tdiv])
                        .add_results(&[memref_type])
                        .build()
                        .expect("bufferization.to_buffer"),
                )
                .result(0)
                .unwrap()
                .into();

            // memref.copy %mc, %out
            body_block.append_operation(
                OperationBuilder::new("memref.copy", location)
                    .add_operands(&[mc, body_block.argument(3).unwrap().into()])
                    .build()
                    .expect("memref.copy"),
            );

            body_block.append_operation(func::r#return(&[], location));

            let region = Region::new();
            region.append_block(body_block);

            let function = func::func(
                &context,
                StringAttribute::new(&context, "compute"),
                TypeAttribute::new(function_type.into()),
                region,
                &[(
                    Identifier::new(&context, "llvm.emit_c_interface"),
                    Attribute::unit(&context),
                )],
                location,
            );
            module.body().append_operation(function);
        }

        assert!(
            module.as_operation().verify(),
            "MLIR module verification failed before passes"
        );

        run_tosa_pipeline(&mut module, &context).expect("TOSA mixed pipeline failed");

        // JIT execute: a=[2,4,6,8], b=[0,0,0,0] -> tosa.add=[2,4,6,8]
        //              c=[1,2,3,4] -> linalg div=[2,2,2,2]
        let engine = melior::ExecutionEngine::new(&module, 2, &[], false);
        let a = [2.0f32, 4.0, 6.0, 8.0];
        let b = [0.0f32, 0.0, 0.0, 0.0];
        let c = [1.0f32, 2.0, 3.0, 4.0];
        let result = unsafe {
            use crate::runtime::build_memref_descriptor;
            let shape = vec![4i64];
            let strides = vec![1i64];

            let mut a_data = a.to_vec();
            let mut b_data = b.to_vec();
            let mut c_data = c.to_vec();
            let mut out_data = vec![0.0f32; 4];

            let mut a_desc =
                build_memref_descriptor(a_data.as_mut_ptr() as *mut u8, &shape, &strides);
            let mut b_desc =
                build_memref_descriptor(b_data.as_mut_ptr() as *mut u8, &shape, &strides);
            let mut c_desc =
                build_memref_descriptor(c_data.as_mut_ptr() as *mut u8, &shape, &strides);
            let mut out_desc =
                build_memref_descriptor(out_data.as_mut_ptr() as *mut u8, &shape, &strides);

            let mut a_ptr = a_desc.as_mut_ptr();
            let mut b_ptr = b_desc.as_mut_ptr();
            let mut c_ptr = c_desc.as_mut_ptr();
            let mut out_ptr = out_desc.as_mut_ptr();

            let mut args: Vec<*mut ()> = vec![
                &mut a_ptr as *mut *mut u8 as *mut (),
                &mut b_ptr as *mut *mut u8 as *mut (),
                &mut c_ptr as *mut *mut u8 as *mut (),
                &mut out_ptr as *mut *mut u8 as *mut (),
            ];
            engine
                .invoke_packed("compute", &mut args)
                .expect("JIT invocation failed");
            out_data
        };
        assert_eq!(result, vec![2.0f32, 2.0, 2.0, 2.0]);
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
        let compiled = Compiler::compile(&trace, &[c.id]).expect("compile failed");

        let a_buf = Buffer::from_slice::<f32>(a_data, a_shape, DType::F32);
        let b_buf = Buffer::from_slice::<f32>(b_data, b_shape, DType::F32);

        let result = if let Some((bdata, bshape)) = bias_data {
            let bias_buf = Buffer::from_slice::<f32>(bdata, bshape, DType::F32);
            compiled.run(&[&a_buf, &b_buf, &bias_buf])
        } else {
            compiled.run(&[&a_buf, &b_buf])
        };

        result.as_slice::<f32>().to_vec()
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
        let compiled = Compiler::compile(&trace, &[b.id]).expect("compile failed");

        let input = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
        let result = compiled.run(&[&input]);
        assert_eq!(result.as_slice::<f32>(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
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
        let compiled = Compiler::compile(&trace, &[b.id]).expect("compile failed");

        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let input = Buffer::from_slice::<f32>(&data, &[12], DType::F32);
        let result = compiled.run(&[&input]);
        // Values should be unchanged in row-major order
        let out = result.as_slice::<f32>();
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
        let compiled = Compiler::compile(&trace, &[b.id]).expect("compile failed");

        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let input = Buffer::from_slice::<f32>(&data, &[2, 6], DType::F32);
        let result = compiled.run(&[&input]);
        let out = result.as_slice::<f32>();
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
        let compiled = Compiler::compile(&trace, &[c.id]).expect("compile failed");
        let a_buf = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[4], DType::F32);
        let b_buf = Buffer::from_slice::<f32>(&[0.5, 0.5, 0.5, 0.5], &[4], DType::F32);
        let result = compiled.run(&[&a_buf, &b_buf]);
        let out = result.as_slice::<f32>();
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
        let compiled = Compiler::compile(&trace, &[out.id]).expect("compile failed");

        let input_buf = Buffer::from_slice::<f32>(input_data, input_shape, DType::F32);
        let kernel_buf = Buffer::from_slice::<f32>(kernel_data, kernel_shape, DType::F32);

        let result = if let Some((bdata, bshape)) = bias_data {
            let bias_buf = Buffer::from_slice::<f32>(bdata, bshape, DType::F32);
            compiled.run(&[&input_buf, &kernel_buf, &bias_buf])
        } else {
            compiled.run(&[&input_buf, &kernel_buf])
        };
        result.as_slice::<f32>().to_vec()
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
        let result = Compiler::compile(&trace, &[NodeId(0)]);
        assert!(result.is_err());
    }

    #[test]
    fn compile_no_outputs_returns_error() {
        begin_trace();
        let _a = Tensor::new(&[4], DType::F32);
        let trace = take_trace();
        let result = Compiler::compile(&trace, &[]);
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
        let compiled = Compiler::compile(&trace, &[out.id]).expect("compile failed");
        let input_buf = Buffer::from_slice::<f32>(input_data, input_shape, DType::F32);
        compiled.run(&[&input_buf]).as_slice::<f32>().to_vec()
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
}
