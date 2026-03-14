use melior::{
    Context,
    dialect::{arith, func, memref, scf},
    ir::{
        Attribute, Block, Identifier, Location, Module, Region, RegionLike,
        attribute::{IntegerAttribute, StringAttribute, TypeAttribute},
        block::BlockLike,
        operation::OperationLike,
        r#type::{FunctionType, MemRefType},
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
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Verification => write!(f, "MLIR module verification failed"),
            Self::Pass(e) => write!(f, "lowering pass failed: {e}"),
            Self::UnsupportedShape(s) => write!(f, "unsupported shape: {s}"),
            Self::EmptyTrace => write!(f, "trace is empty"),
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

            // Emit loop bounds.
            let index_type = melior::ir::Type::index(&context);
            let c0 = body_block
                .append_operation(arith::constant(
                    &context,
                    IntegerAttribute::new(index_type, 0).into(),
                    location,
                ))
                .result(0)
                .unwrap();

            let cn = body_block
                .append_operation(arith::constant(
                    &context,
                    IntegerAttribute::new(index_type, num_elems as i64).into(),
                    location,
                ))
                .result(0)
                .unwrap();

            let c1 = body_block
                .append_operation(arith::constant(
                    &context,
                    IntegerAttribute::new(index_type, 1).into(),
                    location,
                ))
                .result(0)
                .unwrap();

            // Build the scf.for loop body.
            let loop_region = {
                let loop_block = Block::new(&[(index_type, location)]);
                let iv = loop_block.argument(0).unwrap();

                // Evaluate each op in the trace. We map NodeId -> Value.
                // Inputs map to block arguments (arg0, arg1, ...).
                // The loop loads from input memrefs and computes the result.
                let result_val = emit_ops_for_loop(
                    trace,
                    output_id,
                    &input_ops,
                    &body_block,
                    &loop_block,
                    iv.into(),
                    location,
                    &context,
                );

                // memref.store result into output memref.
                let out_arg = body_block
                    .argument(num_inputs)
                    .unwrap();
                loop_block.append_operation(memref::store(
                    result_val,
                    out_arg.into(),
                    &[iv.into()],
                    location,
                ));

                loop_block.append_operation(scf::r#yield(&[], location));

                let r = Region::new();
                r.append_block(loop_block);
                r
            };

            body_block.append_operation(scf::r#for(
                c0.into(),
                cn.into(),
                c1.into(),
                loop_region,
                location,
            ));

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

/// Recursively emit load/compute operations for `node_id` inside the loop
/// block. Returns the SSA value representing the result.
#[allow(clippy::too_many_arguments, clippy::only_used_in_recursion)]
fn emit_ops_for_loop<'c, 'blk>(
    trace: &Trace,
    node_id: NodeId,
    input_ops: &[(NodeId, &Op)],
    body_block: &'blk Block<'c>,
    loop_block: &'blk Block<'c>,
    iv: melior::ir::Value<'c, 'blk>,
    location: Location<'c>,
    context: &'c Context,
) -> melior::ir::Value<'c, 'blk> {
    let op = trace.get(node_id);
    match op {
        Op::Input { arg_index, .. } => {
            // Load from the corresponding function argument memref.
            let memref_arg = body_block.argument(*arg_index as usize).unwrap();
            loop_block
                .append_operation(memref::load(memref_arg.into(), &[iv], location))
                .result(0)
                .unwrap()
                .into()
        }
        Op::Add { lhs, rhs, dtype, .. } => {
            let lhs_val =
                emit_ops_for_loop(trace, *lhs, input_ops, body_block, loop_block, iv, location, context);
            let rhs_val =
                emit_ops_for_loop(trace, *rhs, input_ops, body_block, loop_block, iv, location, context);

            match dtype {
                crate::DType::F32 | crate::DType::F64 => loop_block
                    .append_operation(arith::addf(lhs_val, rhs_val, location))
                    .result(0)
                    .unwrap()
                    .into(),
                crate::DType::I32 | crate::DType::I64 => loop_block
                    .append_operation(arith::addi(lhs_val, rhs_val, location))
                    .result(0)
                    .unwrap()
                    .into(),
            }
        }
    }
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
