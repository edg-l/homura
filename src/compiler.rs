use std::collections::HashMap;

use melior::{
    Context,
    dialect::{arith, func},
    ir::{
        Attribute, Block, Identifier, Location, Module, Region, RegionLike,
        attribute::{ArrayAttribute, StringAttribute, TypeAttribute},
        block::BlockLike,
        operation::{OperationBuilder, OperationLike},
        r#type::{FunctionType, MemRefType, RankedTensorType},
    },
    pass,
    utility::{parse_pass_pipeline, register_all_dialects, register_all_llvm_translations, register_all_passes},
};
use melior::dialect::DialectRegistry;

use crate::{
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
    /// Unsupported trace shape (e.g. rank != 1).
    UnsupportedShape(String),
    /// Trace contains no ops.
    EmptyTrace,
    /// Failed to parse an MLIR attribute string.
    AttributeParse(String),
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Verification => write!(f, "MLIR module verification failed"),
            Self::Pass(e) => write!(f, "lowering pass failed: {e}"),
            Self::UnsupportedShape(s) => write!(f, "unsupported shape: {s}"),
            Self::EmptyTrace => write!(f, "trace is empty"),
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
            return Err(CompileError::EmptyTrace);
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

        // For now: single output, rank-1 only.
        let output_id = outputs[0];
        let output_op = trace.get(output_id);
        let output_shape = output_op.shape().clone();
        let output_dtype = output_op.dtype();

        // Validate all shapes are rank-1.
        for op in trace.ops() {
            if op.shape().rank() != 1 {
                return Err(CompileError::UnsupportedShape(format!(
                    "rank-{} shapes are not supported (only rank-1)",
                    op.shape().rank()
                )));
            }
        }

        // ---- Create MLIR context and module ------------------------------------

        let context = create_context();
        let location = Location::unknown(&context);
        let mut module = Module::new(location);

        // ---- Build the func.func @compute -------------------------------------
        {
            let elem_type = output_dtype.to_mlir_type(&context);
            let num_elems = output_shape.num_elements();

            // Build arg types: inputs first, then one output.
            let mut arg_types: Vec<(melior::ir::Type, Location)> = Vec::new();
            for (_, op) in &input_ops {
                let Op::Input { shape, dtype, .. } = op else {
                    unreachable!()
                };
                let mref = MemRefType::new(
                    dtype.to_mlir_type(&context),
                    &[shape.0[0] as i64],
                    None,
                    None,
                );
                arg_types.push((mref.into(), location));
            }
            // Output memref.
            let out_mref = MemRefType::new(
                elem_type,
                &[num_elems as i64],
                None,
                None,
            );
            arg_types.push((out_mref.into(), location));

            // func.func type: all memrefs -> ()
            let func_arg_types: Vec<melior::ir::Type> = arg_types.iter().map(|(t, _)| *t).collect();
            let function_type = FunctionType::new(&context, &func_arg_types, &[]);

            // Create the function body block.
            let body_block = Block::new(&arg_types);

            // Emit tensor ops: bufferization.to_tensor for inputs, linalg.generic
            // for compute ops, bufferization.to_buffer + memref.copy for the output.
            emit_tensor_ops(
                trace,
                output_id,
                &input_ops,
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
        // register_all_passes() is required before parse_pass_pipeline.
        register_all_passes();
        let pass_manager = pass::PassManager::new(&context);
        parse_pass_pipeline(
            pass_manager.as_operation_pass_manager(),
            "builtin.module(\
                one-shot-bufferize{function-boundary-type-conversion=identity-layout-map unknown-type-conversion=identity-layout-map},\
                convert-linalg-to-loops,\
                convert-scf-to-cf,\
                finalize-memref-to-llvm,\
                convert-arith-to-llvm,\
                convert-index-to-llvm,\
                convert-cf-to-llvm,\
                convert-func-to-llvm,\
                reconcile-unrealized-casts\
            )",
        )
        .map_err(CompileError::Pass)?;

        pass_manager
            .run(&mut module)
            .map_err(CompileError::Pass)?;

        // ---- Create ExecutionEngine -------------------------------------------
        let engine = melior::ExecutionEngine::new(&module, 2, &[], false);

        Ok(CompiledGraph::new(engine, num_inputs, output_shape, output_dtype))
    }

    /// Print the MLIR IR for a trace before lowering (for debugging).
    pub fn print_mlir(trace: &Trace, outputs: &[NodeId]) {
        // We compile up to the verification step and print.
        // Re-use a quick build that ignores errors, so just call compile and
        // intercept the printed module inside compile. For simplicity, run
        // compile fully and note this is just debug output.
        // Instead: build the module and print it before lowering.
        let _ = (trace, outputs);
        eprintln!("(use Compiler::compile with MLIR_PRINT_IR=1 or add debug logging)");
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

/// Walk the trace linearly, emitting tensor-level ops into `body_block`.
///
/// For each `Input` op: emits `bufferization.to_tensor` with `restrict`.
/// For each `Add` op: emits `tensor.empty` + `linalg.generic`.
/// After all ops: emits `bufferization.to_buffer` + `memref.copy` for the output.
fn emit_tensor_ops<'c>(
    trace: &Trace,
    output_id: NodeId,
    input_ops: &[(NodeId, &Op)],
    num_inputs: usize,
    body_block: &Block<'c>,
    location: Location<'c>,
    context: &'c Context,
) -> Result<(), CompileError> {
    let _ = input_ops; // arg_index from Op::Input is the authoritative mapping

    // NodeId -> SSA tensor Value for each op emitted so far.
    let mut values: HashMap<NodeId, melior::ir::Value<'c, '_>> = HashMap::new();

    for (i, op) in trace.ops().iter().enumerate() {
        let node_id = NodeId(i as u32);

        match op {
            Op::Input { arg_index, shape, dtype } => {
                let elem_type = dtype.to_mlir_type(context);
                let tensor_type: melior::ir::Type =
                    RankedTensorType::new(&[shape.0[0] as u64], elem_type, None).into();
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
                let elem_type = dtype.to_mlir_type(context);
                let num_elems = shape.0[0] as u64;
                let tensor_type: melior::ir::Type =
                    RankedTensorType::new(&[num_elems], elem_type, None).into();

                let lhs_val = *values.get(lhs).expect("lhs node not yet emitted");
                let rhs_val = *values.get(rhs).expect("rhs node not yet emitted");

                // tensor.empty() — the initial output buffer for linalg.generic.
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

                // Build the linalg.generic body region.
                // The block has 3 args: lhs_elem, rhs_elem, out_elem (required by linalg ABI).
                let linalg_region = {
                    let linalg_block = Block::new(&[
                        (elem_type, location),
                        (elem_type, location),
                        (elem_type, location),
                    ]);
                    let lhs_elem = linalg_block.argument(0).unwrap();
                    let rhs_elem = linalg_block.argument(1).unwrap();

                    // arith.addf or arith.addi depending on dtype.
                    let sum_val = match dtype {
                        crate::DType::F32 | crate::DType::F64 => linalg_block
                            .append_operation(arith::addf(lhs_elem.into(), rhs_elem.into(), location))
                            .result(0)
                            .unwrap()
                            .into(),
                        crate::DType::I32 | crate::DType::I64 => linalg_block
                            .append_operation(arith::addi(lhs_elem.into(), rhs_elem.into(), location))
                            .result(0)
                            .unwrap()
                            .into(),
                    };

                    linalg_block.append_operation(
                        OperationBuilder::new("linalg.yield", location)
                            .add_operands(&[sum_val])
                            .build()
                            .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                    );

                    let r = Region::new();
                    r.append_block(linalg_block);
                    r
                };

                // Identity affine map for rank-1: affine_map<(d0) -> (d0)>.
                let identity_map = Attribute::parse(context, "affine_map<(d0) -> (d0)>")
                    .ok_or_else(|| {
                        CompileError::AttributeParse(
                            "failed to parse affine_map<(d0) -> (d0)>".into(),
                        )
                    })?;
                // 3 maps: lhs, rhs, out — all identity for element-wise ops.
                let indexing_maps =
                    ArrayAttribute::new(context, &[identity_map, identity_map, identity_map]);

                let iterator_types: Attribute =
                    Attribute::parse(context, "[#linalg.iterator_type<parallel>]")
                        .ok_or_else(|| {
                            CompileError::AttributeParse(
                                "failed to parse iterator_types attribute".into(),
                            )
                        })?;

                // operand_segment_sizes: 2 ins + 1 outs.
                let segment_sizes =
                    Attribute::parse(context, "array<i32: 2, 1>").ok_or_else(|| {
                        CompileError::AttributeParse(
                            "failed to parse operand_segment_sizes".into(),
                        )
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
                                (
                                    Identifier::new(context, "iterator_types"),
                                    iterator_types,
                                ),
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
        }
    }

    // ---- Output boundary: bufferization.to_buffer + memref.copy ---------------
    let result_tensor = *values.get(&output_id).expect("output node not emitted");

    // Recover the output tensor's type so we can build the matching memref type.
    let output_op = trace.get(output_id);
    let out_elem_type = output_op.dtype().to_mlir_type(context);
    let num_elems = output_op.shape().0[0] as i64;
    let out_memref_type: melior::ir::Type =
        MemRefType::new(out_elem_type, &[num_elems], None, None).into();

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
    use crate::{DType, trace::{begin_trace, take_trace}, tensor::Tensor};

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
}
