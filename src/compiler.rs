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

        // ---- Analyse the trace ------------------------------------------------

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

        // Single output.
        let output_id = outputs[0];
        let output_op = trace.get(output_id);
        let output_shape = output_op.shape().clone();
        let output_dtype = output_op.dtype();

        // ---- Create MLIR context and module ------------------------------------

        let context = create_context();
        let location = Location::unknown(&context);
        let mut module = Module::new(location);

        // ---- Build the func.func @compute -------------------------------------
        {
            let elem_type = output_dtype.to_mlir_type(&context);

            // Build arg types: inputs first, then one output.
            let mut arg_types: Vec<(melior::ir::Type, Location)> = Vec::new();
            for (_, op) in &input_ops {
                let Op::Input { shape, dtype, .. } = op else {
                    unreachable!()
                };
                let dims: Vec<i64> = shape.0.iter().map(|&d| d as i64).collect();
                let mref = MemRefType::new(dtype.to_mlir_type(&context), &dims, None, None);
                arg_types.push((mref.into(), location));
            }
            // Output memref — N-D shape.
            let out_dims: Vec<i64> = output_shape.0.iter().map(|&d| d as i64).collect();
            let out_mref = MemRefType::new(elem_type, &out_dims, None, None);
            arg_types.push((out_mref.into(), location));

            // func.func type: all memrefs -> ()
            let func_arg_types: Vec<melior::ir::Type> = arg_types.iter().map(|(t, _)| *t).collect();
            let function_type = FunctionType::new(&context, &func_arg_types, &[]);

            // Create the function body block.
            let body_block = Block::new(&arg_types);

            // Emit tensor ops.
            emit_tensor_ops(
                trace,
                output_id,
                num_inputs,
                &body_block,
                location,
                &context,
            )?;

            body_block.append_operation(func::r#return(&[], location));

            let func_region = Region::new();
            func_region.append_block(body_block);

            let function = func::func(
                &context,
                StringAttribute::new(&context, "compute"),
                TypeAttribute::new(function_type.into()),
                func_region,
                &[(
                    Identifier::new(&context, "llvm.emit_c_interface"),
                    Attribute::unit(&context),
                )],
                location,
            );

            module.body().append_operation(function);
        }

        // ---- Verify -----------------------------------------------------------
        if !module.as_operation().verify() {
            return Err(CompileError::Verification);
        }

        // ---- Run lowering passes ----------------------------------------------
        register_all_passes();
        let pass_manager = pass::PassManager::new(&context);
        parse_pass_pipeline(
            pass_manager.as_operation_pass_manager(),
            "builtin.module(\
                one-shot-bufferize{function-boundary-type-conversion=identity-layout-map unknown-type-conversion=identity-layout-map},\
                convert-linalg-to-loops,\
                convert-scf-to-cf,\
                convert-math-to-llvm,\
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
        let engine = melior::ExecutionEngine::new(&module, 2, &[], false);

        Ok(CompiledGraph::new(
            engine,
            num_inputs,
            output_shape,
            output_dtype,
        ))
    }

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

fn make_iterator_types<'c>(context: &'c Context, count: usize) -> Result<Attribute<'c>, CompileError> {
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
    F: FnOnce(&Block<'c>, melior::ir::Value<'c, 'c>, melior::ir::Value<'c, 'c>) -> melior::ir::Value<'c, 'c>,
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
                    (Identifier::new(context, "indexing_maps"), indexing_maps.into()),
                    (Identifier::new(context, "iterator_types"), iterator_types),
                    (Identifier::new(context, "operand_segment_sizes"), segment_sizes),
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

// ── Helper: emit a unary element-wise linalg.generic ─────────────────────────

#[allow(clippy::too_many_arguments)]
fn emit_unary_elementwise<'c, F>(
    context: &'c Context,
    body_block: &Block<'c>,
    values: &HashMap<NodeId, melior::ir::Value<'c, 'c>>,
    input: NodeId,
    shape: &[u64],
    dtype: DType,
    location: Location<'c>,
    body_fn: F,
) -> Result<melior::ir::Value<'c, 'c>, CompileError>
where
    F: FnOnce(&Block<'c>, melior::ir::Value<'c, 'c>) -> melior::ir::Value<'c, 'c>,
{
    let elem_type = dtype.to_mlir_type(context);
    let tensor_type = make_ranked_tensor_type(context, shape, dtype);
    let rank = shape.len();

    let input_val = *values.get(&input).expect("input node not yet emitted");

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

    // linalg body: 2 block args (in_elem, out_elem).
    let linalg_region = {
        let linalg_block = Block::new(&[(elem_type, location), (elem_type, location)]);
        let in_elem = linalg_block.argument(0).unwrap().into();

        let result = body_fn(&linalg_block, in_elem);

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

    // 2 identity maps: in, out.
    let identity_map = make_identity_map(context, rank)?;
    let indexing_maps = ArrayAttribute::new(context, &[identity_map, identity_map]);
    let iterator_types = make_iterator_types(context, rank)?;
    let segment_sizes = Attribute::parse(context, "array<i32: 1, 1>").ok_or_else(|| {
        CompileError::AttributeParse("failed to parse operand_segment_sizes".into())
    })?;

    let result_val = body_block
        .append_operation(
            OperationBuilder::new("linalg.generic", location)
                .add_operands(&[input_val, init_val])
                .add_results(&[tensor_type])
                .add_attributes(&[
                    (Identifier::new(context, "indexing_maps"), indexing_maps.into()),
                    (Identifier::new(context, "iterator_types"), iterator_types),
                    (Identifier::new(context, "operand_segment_sizes"), segment_sizes),
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

// ── Helper: emit linalg.matmul for [M,K] x [K,N] -> [M,N] ───────────────────

#[allow(clippy::too_many_arguments)]
fn emit_matmul<'c>(
    context: &'c Context,
    body_block: &Block<'c>,
    values: &HashMap<NodeId, melior::ir::Value<'c, 'c>>,
    lhs: NodeId,
    rhs: NodeId,
    output_shape: &[u64],  // [M, N]
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
    input_shape: &[u64],   // shape of the INPUT tensor
    output_shape: &[u64],  // shape of the OUTPUT tensor
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
    let output_map = Attribute::parse(context, &output_map_str).ok_or_else(|| {
        CompileError::AttributeParse(format!("failed to parse {output_map_str}"))
    })?;

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
                    (Identifier::new(context, "indexing_maps"), indexing_maps.into()),
                    (Identifier::new(context, "iterator_types"), iterator_types),
                    (Identifier::new(context, "operand_segment_sizes"), segment_sizes),
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

            Op::Add { lhs, rhs, shape, dtype } => {
                let lhs_shape = trace.get(*lhs).shape();
                let rhs_shape = trace.get(*rhs).shape();
                let result_val = emit_binary_elementwise(
                    context, body_block, &values, *lhs, *rhs,
                    &lhs_shape.0, &rhs_shape.0, &shape.0, *dtype, location,
                    |block, lhs_elem, rhs_elem| match dtype {
                        DType::F32 | DType::F64 => block
                            .append_operation(arith::addf(lhs_elem, rhs_elem, location))
                            .result(0).unwrap().into(),
                        DType::I32 | DType::I64 => block
                            .append_operation(arith::addi(lhs_elem, rhs_elem, location))
                            .result(0).unwrap().into(),
                    },
                )?;
                values.insert(node_id, result_val);
            }

            Op::Sub { lhs, rhs, shape, dtype } => {
                let lhs_shape = trace.get(*lhs).shape();
                let rhs_shape = trace.get(*rhs).shape();
                let result_val = emit_binary_elementwise(
                    context, body_block, &values, *lhs, *rhs,
                    &lhs_shape.0, &rhs_shape.0, &shape.0, *dtype, location,
                    |block, lhs_elem, rhs_elem| match dtype {
                        DType::F32 | DType::F64 => block
                            .append_operation(arith::subf(lhs_elem, rhs_elem, location))
                            .result(0).unwrap().into(),
                        DType::I32 | DType::I64 => block
                            .append_operation(arith::subi(lhs_elem, rhs_elem, location))
                            .result(0).unwrap().into(),
                    },
                )?;
                values.insert(node_id, result_val);
            }

            Op::Mul { lhs, rhs, shape, dtype } => {
                let lhs_shape = trace.get(*lhs).shape();
                let rhs_shape = trace.get(*rhs).shape();
                let result_val = emit_binary_elementwise(
                    context, body_block, &values, *lhs, *rhs,
                    &lhs_shape.0, &rhs_shape.0, &shape.0, *dtype, location,
                    |block, lhs_elem, rhs_elem| match dtype {
                        DType::F32 | DType::F64 => block
                            .append_operation(arith::mulf(lhs_elem, rhs_elem, location))
                            .result(0).unwrap().into(),
                        DType::I32 | DType::I64 => block
                            .append_operation(arith::muli(lhs_elem, rhs_elem, location))
                            .result(0).unwrap().into(),
                    },
                )?;
                values.insert(node_id, result_val);
            }

            Op::Div { lhs, rhs, shape, dtype } => {
                let lhs_shape = trace.get(*lhs).shape();
                let rhs_shape = trace.get(*rhs).shape();
                let result_val = emit_binary_elementwise(
                    context, body_block, &values, *lhs, *rhs,
                    &lhs_shape.0, &rhs_shape.0, &shape.0, *dtype, location,
                    |block, lhs_elem, rhs_elem| match dtype {
                        DType::F32 | DType::F64 => block
                            .append_operation(arith::divf(lhs_elem, rhs_elem, location))
                            .result(0).unwrap().into(),
                        DType::I32 | DType::I64 => block
                            .append_operation(arith::divsi(lhs_elem, rhs_elem, location))
                            .result(0).unwrap().into(),
                    },
                )?;
                values.insert(node_id, result_val);
            }

            Op::Neg { input, shape, dtype } => {
                let elem_type = dtype.to_mlir_type(context);
                let result_val = emit_unary_elementwise(
                    context, body_block, &values, *input,
                    &shape.0, *dtype, location,
                    |block, in_elem| match dtype {
                        DType::F32 | DType::F64 => block
                            .append_operation(arith::negf(in_elem, location))
                            .result(0).unwrap().into(),
                        DType::I32 | DType::I64 => {
                            let zero = block
                                .append_operation(arith::constant(
                                    context,
                                    IntegerAttribute::new(elem_type, 0).into(),
                                    location,
                                ))
                                .result(0).unwrap().into();
                            block
                                .append_operation(arith::subi(zero, in_elem, location))
                                .result(0).unwrap().into()
                        }
                    },
                )?;
                values.insert(node_id, result_val);
            }

            Op::Exp { input, shape, dtype } => {
                let elem_type = dtype.to_mlir_type(context);
                let result_val = emit_unary_elementwise(
                    context, body_block, &values, *input,
                    &shape.0, *dtype, location,
                    |block, in_elem| {
                        block
                            .append_operation(
                                OperationBuilder::new("math.exp", location)
                                    .add_operands(&[in_elem])
                                    .add_results(&[elem_type])
                                    .build()
                                    .expect("failed to build math.exp"),
                            )
                            .result(0)
                            .unwrap()
                            .into()
                    },
                )?;
                values.insert(node_id, result_val);
            }

            Op::Tanh { input, shape, dtype } => {
                let elem_type = dtype.to_mlir_type(context);
                let result_val = emit_unary_elementwise(
                    context, body_block, &values, *input,
                    &shape.0, *dtype, location,
                    |block, in_elem| {
                        block
                            .append_operation(
                                OperationBuilder::new("math.tanh", location)
                                    .add_operands(&[in_elem])
                                    .add_results(&[elem_type])
                                    .build()
                                    .expect("failed to build math.tanh"),
                            )
                            .result(0)
                            .unwrap()
                            .into()
                    },
                )?;
                values.insert(node_id, result_val);
            }

            Op::Matmul { lhs, rhs, shape, dtype } => {
                let result_val = emit_matmul(
                    context, body_block, &values, *lhs, *rhs, &shape.0, *dtype, location,
                )?;
                values.insert(node_id, result_val);
            }

            Op::Relu { input, shape, dtype } => {
                let elem_type = dtype.to_mlir_type(context);
                let result_val = emit_unary_elementwise(
                    context, body_block, &values, *input,
                    &shape.0, *dtype, location,
                    |block, in_elem| match dtype {
                        DType::F32 | DType::F64 => {
                            let zero = block
                                .append_operation(arith::constant(
                                    context,
                                    FloatAttribute::new(context, elem_type, 0.0).into(),
                                    location,
                                ))
                                .result(0).unwrap().into();
                            block
                                .append_operation(arith::maxnumf(in_elem, zero, location))
                                .result(0).unwrap().into()
                        }
                        DType::I32 | DType::I64 => {
                            let zero = block
                                .append_operation(arith::constant(
                                    context,
                                    IntegerAttribute::new(elem_type, 0).into(),
                                    location,
                                ))
                                .result(0).unwrap().into();
                            block
                                .append_operation(arith::maxsi(in_elem, zero, location))
                                .result(0).unwrap().into()
                        }
                    },
                )?;
                values.insert(node_id, result_val);
            }

            Op::ReduceSum { input, dim, keepdim, shape, dtype } => {
                let input_shape = trace.get(*input).shape();
                let result_val = emit_reduction(
                    context, body_block, &values, *input,
                    &input_shape.0, &shape.0, *dim, *keepdim, *dtype, location,
                    false,
                )?;
                values.insert(node_id, result_val);
            }

            Op::ReduceMax { input, dim, keepdim, shape, dtype } => {
                let input_shape = trace.get(*input).shape();
                let result_val = emit_reduction(
                    context, body_block, &values, *input,
                    &input_shape.0, &shape.0, *dim, *keepdim, *dtype, location,
                    true,
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
}
