use std::collections::{HashMap, HashSet};

use melior::ir::Value;
use melior::ir::block::BlockLike;
use melior::ir::operation::OperationBuilder;
use melior::ir::{Attribute, Identifier};

use crate::{
    DType,
    graph_builder::{GraphBuilder, Tensor},
    runtime::Buffer,
};

use super::parser::{Dim, OnnxAttribute, OnnxError, OnnxModel, OnnxNode};

/// Sentinel for partially-known const_i64 values. Used when a Concat input
/// doesn't have const values — downstream consumers (Reshape) treat this as
/// "dynamic / unknown" while still using the known values from other inputs.
const CONST_I64_UNKNOWN: i64 = i64::MIN;

// ── Value classification ───────────────────────────────────────────────────────

/// An entry in the emitter's value map.
///
/// Most ONNX values are regular ranked tensors. The Shape op is special: it
/// returns a sequence of MLIR `index` values (one per dim of the input), not a
/// packed tensor. When a downstream op needs them as an I64 tensor (e.g.
/// Reshape, ConstantOfShape), we materialize them via `tensor.from_elements`.
enum EmitValue<'c> {
    Tensor(Tensor<'c>),
    /// MLIR `index` SSA values produced by `emit_shape_of`.
    ShapeDims(Vec<Value<'c, 'c>>),
}

impl<'c> EmitValue<'c> {
    fn as_tensor(&self, builder: &mut GraphBuilder<'c>) -> Tensor<'c> {
        match self {
            EmitValue::Tensor(t) => *t,
            EmitValue::ShapeDims(dims) => pack_dims_to_i64_tensor(builder, dims),
        }
    }
}

// ── Index/tensor conversion helpers ──────────────────────────────────────────

/// Pack a slice of MLIR `index` values into a `tensor<Nxi64>` via
/// `tensor.from_elements` (casting each index → i64 first).
fn pack_dims_to_i64_tensor<'c>(
    builder: &mut GraphBuilder<'c>,
    dims: &[Value<'c, 'c>],
) -> Tensor<'c> {
    let ctx = builder.context();
    let loc = builder.location();
    let i64_type = melior::ir::Type::parse(ctx, "i64").expect("i64 type");

    let i64_vals: Vec<Value<'c, 'c>> = dims.iter().map(|&idx| {
        builder.block()
            .append_operation(
                OperationBuilder::new("arith.index_cast", loc)
                    .add_operands(&[idx])
                    .add_results(&[i64_type])
                    .build()
                    .expect("arith.index_cast"),
            )
            .result(0).unwrap().into()
    }).collect();

    let n = dims.len() as u64;
    let tensor_type: melior::ir::Type = melior::ir::r#type::RankedTensorType::new(
        &[n], i64_type, None,
    ).into();

    let result: Value<'c, 'c> = builder.block()
        .append_operation(
            OperationBuilder::new("tensor.from_elements", loc)
                .add_operands(&i64_vals)
                .add_results(&[tensor_type])
                .build()
                .expect("tensor.from_elements"),
        )
        .result(0).unwrap().into();
    Tensor::from_value(result)
}

/// Extract N MLIR `index` values from a 1-D i64/i32 Tensor of known static length.
/// Each element is extracted via `tensor.extract` and cast to `index`.
fn extract_dims_from_i64_tensor<'c>(
    builder: &mut GraphBuilder<'c>,
    tensor: &Tensor<'c>,
) -> Vec<Value<'c, 'c>> {
    let ctx = builder.context();
    let loc = builder.location();
    let index_type = melior::ir::Type::parse(ctx, "index").expect("index type");
    let shape = tensor.shape();
    assert_eq!(shape.len(), 1, "extract_dims_from_i64_tensor: expected 1-D tensor");
    let n = match shape[0] {
        Some(n) => n as usize,
        None => panic!("extract_dims_from_i64_tensor: dynamic-length shape tensor not supported"),
    };
    let elem_type = tensor.dtype().to_mlir_type(ctx);

    (0..n).map(|i| {
        let idx_attr = Attribute::parse(ctx, &format!("{i} : index")).expect("idx attr");
        let idx_val: Value<'c, 'c> = builder.block()
            .append_operation(
                OperationBuilder::new("arith.constant", loc)
                    .add_results(&[index_type])
                    .add_attributes(&[(Identifier::new(ctx, "value"), idx_attr)])
                    .build()
                    .expect("arith.constant idx"),
            )
            .result(0).unwrap().into();

        let elem: Value<'c, 'c> = builder.block()
            .append_operation(
                OperationBuilder::new("tensor.extract", loc)
                    .add_operands(&[tensor.value(), idx_val])
                    .add_results(&[elem_type])
                    .build()
                    .expect("tensor.extract"),
            )
            .result(0).unwrap().into();

        builder.block()
            .append_operation(
                OperationBuilder::new("arith.index_cast", loc)
                    .add_operands(&[elem])
                    .add_results(&[index_type])
                    .build()
                    .expect("arith.index_cast"),
            )
            .result(0).unwrap().into()
    }).collect()
}

/// Extract a single `index` value from a scalar (rank-0) or 1-element (rank-1) tensor.
fn extract_scalar_as_index<'c>(
    builder: &mut GraphBuilder<'c>,
    tensor: &Tensor<'c>,
) -> Value<'c, 'c> {
    let ctx = builder.context();
    let loc = builder.location();
    let index_type = melior::ir::Type::parse(ctx, "index").expect("index type");
    let elem_type = tensor.dtype().to_mlir_type(ctx);

    let scalar: Value<'c, 'c> = if tensor.rank() == 0 {
        builder.block()
            .append_operation(
                OperationBuilder::new("tensor.extract", loc)
                    .add_operands(&[tensor.value()])
                    .add_results(&[elem_type])
                    .build()
                    .expect("tensor.extract scalar"),
            )
            .result(0).unwrap().into()
    } else {
        let c0_attr = Attribute::parse(ctx, "0 : index").expect("0 index");
        let c0: Value<'c, 'c> = builder.block()
            .append_operation(
                OperationBuilder::new("arith.constant", loc)
                    .add_results(&[index_type])
                    .add_attributes(&[(Identifier::new(ctx, "value"), c0_attr)])
                    .build()
                    .expect("arith.constant 0"),
            )
            .result(0).unwrap().into();
        builder.block()
            .append_operation(
                OperationBuilder::new("tensor.extract", loc)
                    .add_operands(&[tensor.value(), c0])
                    .add_results(&[elem_type])
                    .build()
                    .expect("tensor.extract [0]"),
            )
            .result(0).unwrap().into()
    };

    builder.block()
        .append_operation(
            OperationBuilder::new("arith.index_cast", loc)
                .add_operands(&[scalar])
                .add_results(&[index_type])
                .build()
                .expect("arith.index_cast scalar"),
        )
        .result(0).unwrap().into()
}

// ── Static integer constant map ────────────────────────────────────────────────

/// Look up a named value in the static integer constant map.
///
/// This map is populated from:
/// - ONNX initializers (small integer buffers)
/// - ONNX `Constant` nodes that produce integer tensors
///
/// It is used to read compile-time integer values for axes, split sizes, and
/// shape inputs that must be known statically (Unsqueeze axes, Split sizes,
/// Slice axes/steps, ReduceSum/Max/Mean axes, Reshape target shapes).
///
/// It intentionally does NOT track runtime-computed values (Shape → Gather →
/// Unsqueeze chains). Those go through MLIR ops and are handled by the
/// dynamic paths (emit_reshape_with_tensor, emit_dynamic_slice, etc.).
fn lookup_const_i64(const_i64: &HashMap<String, Vec<i64>>, name: &str) -> Option<Vec<i64>> {
    const_i64.get(name).cloned()
}

/// Element-wise op on two const_i64 vectors, with scalar broadcast support.
/// Returns None if sizes are incompatible.
/// Propagates CONST_I64_UNKNOWN: if either operand is unknown, result is unknown.
fn const_i64_elementwise(a: &[i64], b: &[i64], op: fn(i64, i64) -> i64) -> Option<Vec<i64>> {
    let apply = |x: i64, y: i64| -> i64 {
        if x == CONST_I64_UNKNOWN || y == CONST_I64_UNKNOWN {
            CONST_I64_UNKNOWN
        } else {
            op(x, y)
        }
    };
    if a.len() == b.len() {
        Some(a.iter().zip(b.iter()).map(|(&x, &y)| apply(x, y)).collect())
    } else if a.len() == 1 {
        Some(b.iter().map(|&y| apply(a[0], y)).collect())
    } else if b.len() == 1 {
        Some(a.iter().map(|&x| apply(x, b[0])).collect())
    } else {
        None
    }
}

// ── emit_graph ─────────────────────────────────────────────────────────────────

/// Emit an ONNX model into a `GraphBuilder`.
///
/// Returns the output `Tensor`s in graph output order. Pass these to
/// `builder.compile(&refs)` / `builder.compile_with_cache`.
///
/// Also returns the ordered weight buffers (initializers). These must be passed
/// as runtime arguments after the dynamic inputs when running the compiled graph.
/// Default MLIR op-count threshold before starting a new sub-function.
const DEFAULT_SPLIT_THRESHOLD: usize = 100;

pub fn emit_graph<'c>(
    model: &OnnxModel,
    builder: &mut GraphBuilder<'c>,
    keep_dynamic: &HashSet<String>,
) -> Result<(Vec<Tensor<'c>>, Vec<Buffer>), OnnxError> {
    emit_graph_with_split(model, builder, keep_dynamic, DEFAULT_SPLIT_THRESHOLD)
}

pub fn emit_graph_with_split<'c>(
    model: &OnnxModel,
    builder: &mut GraphBuilder<'c>,
    keep_dynamic: &HashSet<String>,
    split_threshold: usize,
) -> Result<(Vec<Tensor<'c>>, Vec<Buffer>), OnnxError> {
    let mut value_map: HashMap<String, EmitValue<'c>> = HashMap::new();

    // ── 1. Dynamic inputs ───────────────────────────────────────────────────────
    for input in &model.dynamic_inputs {
        let shape: Vec<Option<u64>> = input.dims.iter().map(|d| match d {
            Dim::Fixed(v) if *v == crate::shape::DIM_DYNAMIC => None,
            Dim::Fixed(v) => Some(*v),
            Dim::Symbolic(name) if keep_dynamic.contains(name) => None,
            Dim::Symbolic(_) => None, // unresolved — treat as dynamic
        }).collect();
        let t = builder.input(&shape, input.dtype);
        value_map.insert(input.name.clone(), EmitValue::Tensor(t));
    }

    // ── 2. Weights (initializers) ───────────────────────────────────────────────
    let mut const_i64: HashMap<String, Vec<i64>> = HashMap::new();
    let mut weights: Vec<Buffer> = Vec::with_capacity(model.initializers.len());
    let mut weight_names: HashSet<String> = HashSet::new();
    // Store @compute-scope weight tensors for sub-function routing.
    let mut weight_compute_tensors: HashMap<String, Tensor<'c>> = HashMap::new();

    for (name, buffer) in &model.initializers {
        let shape: Vec<Option<u64>> = buffer.shape().0.iter().map(|&d| Some(d)).collect();
        let t = builder.add_weight(&shape, buffer.dtype());
        value_map.insert(name.clone(), EmitValue::Tensor(t));
        weight_names.insert(name.clone());
        weight_compute_tensors.insert(name.clone(), t);
        weights.push(buffer.clone());
        // Seed const_i64 with small integer initializers.
        if buffer.shape().num_elements() <= 64 {
            if let Ok(vals) = read_i64_buffer(buffer) {
                const_i64.insert(name.clone(), vals);
            }
        }
    }

    // ── 2b. Build last_use map (forward scan) ────────────────────────────────────
    let last_use = build_last_use_map(&model.nodes, &model.outputs);
    let splitting_enabled = split_threshold > 0 && model.nodes.len() > 1;

    // ── 3. Walk nodes with automatic splitting ──────────────────────────────────
    let mut chunk_index: usize = 0;
    // Weights already routed into the current sub-function (name -> sub-function tensor).
    let mut weight_remap: HashMap<String, Tensor<'c>> = HashMap::new();

    // Start the first sub-function immediately so @compute only has glue code.
    if splitting_enabled {
        let live = collect_live_values(
            &mut value_map, &weight_names, &last_use, 0, builder,
        );
        begin_chunk(builder, &mut value_map, &live, chunk_index);
        chunk_index += 1;
        weight_remap.clear();
    }

    // ONNX op types that expand dramatically during tiling/vectorization.
    // Force a chunk boundary before each so they become separate functions.
    const HEAVY_OPS: &[&str] = &["Conv", "MatMul", "Gemm", "ConvTranspose"];

    for (node_idx, node) in model.nodes.iter().enumerate() {
        // Split before heavy ops (Conv/MatMul/Gemm) that explode after tiling,
        // or when the current chunk exceeds the op-count threshold.
        let op_count = builder.block_op_count();
        let is_heavy = HEAVY_OPS.contains(&node.op_type.as_str());
        if splitting_enabled && (op_count >= split_threshold || (is_heavy && op_count > 0)) {
            let live = collect_live_values(
                &mut value_map, &weight_names, &last_use, node_idx, builder,
            );
            end_chunk(builder, &mut value_map, &live);
            let live = collect_live_values(
                &mut value_map, &weight_names, &last_use, node_idx, builder,
            );
            begin_chunk(builder, &mut value_map, &live, chunk_index);
            chunk_index += 1;
            weight_remap.clear();
        }

        // Route weights into the current sub-function on demand.
        if builder.in_subfunction() {
            remap_node_weights(
                node, &weight_names, &weight_compute_tensors,
                &mut weight_remap, &mut value_map, builder,
            );
        }

        emit_node(node, &mut value_map, &mut const_i64, builder)?;
    }

    // Close the last sub-function if one is open.
    if builder.in_subfunction() {
        // Collect model output values as the final sub-function's returns.
        let mut final_live: Vec<(String, Tensor<'c>)> = Vec::new();
        for name in &model.outputs {
            let ev = value_map.get(name)
                .ok_or_else(|| OnnxError::MissingEdge(name.clone()))?;
            final_live.push((name.clone(), ev.as_tensor(builder)));
        }
        end_chunk(builder, &mut value_map, &final_live);
    }

    // ── 4. Collect outputs ───────────────────────────────────────────────────────
    let mut outputs: Vec<Tensor<'c>> = Vec::with_capacity(model.outputs.len());
    for name in &model.outputs {
        let ev = value_map.get(name)
            .ok_or_else(|| OnnxError::MissingEdge(name.clone()))?;
        outputs.push(ev.as_tensor(builder));
    }

    if chunk_index > 0 {
        tracing::debug!(
            chunk_index,
            split_threshold,
            "emitter: split graph into sub-functions"
        );
    }

    Ok((outputs, weights))
}

/// Build a map from ONNX value name to the last node index that uses it as input.
/// Model outputs are always considered live (last_use = usize::MAX).
fn build_last_use_map(nodes: &[OnnxNode], model_outputs: &[String]) -> HashMap<String, usize> {
    let mut last_use: HashMap<String, usize> = HashMap::new();
    for (idx, node) in nodes.iter().enumerate() {
        for input_name in &node.inputs {
            if !input_name.is_empty() {
                last_use.insert(input_name.clone(), idx);
            }
        }
    }
    // Model outputs must survive all splits.
    for name in model_outputs {
        last_use.insert(name.clone(), usize::MAX);
    }
    last_use
}

/// Collect live non-weight values from value_map whose last_use >= current node index.
/// Returns (name, tensor) pairs. ShapeDims are materialized into i64 tensors.
fn collect_live_values<'c>(
    value_map: &mut HashMap<String, EmitValue<'c>>,
    weight_names: &HashSet<String>,
    last_use: &HashMap<String, usize>,
    current_node_idx: usize,
    builder: &mut GraphBuilder<'c>,
) -> Vec<(String, Tensor<'c>)> {
    // First pass: identify names and materialize ShapeDims.
    let names_to_materialize: Vec<String> = value_map
        .iter()
        .filter(|(name, ev)| {
            if weight_names.contains(*name) { return false; }
            if let Some(&lu) = last_use.get(*name) {
                if lu >= current_node_idx {
                    matches!(ev, EmitValue::ShapeDims(_))
                } else { false }
            } else { false }
        })
        .map(|(name, _)| name.clone())
        .collect();

    // Materialize ShapeDims → Tensor so they can cross function boundaries.
    for name in &names_to_materialize {
        let tensor = value_map[name].as_tensor(builder);
        value_map.insert(name.clone(), EmitValue::Tensor(tensor));
    }

    // Second pass: collect all live tensors.
    let mut live = Vec::new();
    for (name, ev) in value_map.iter() {
        if weight_names.contains(name) { continue; }
        if let Some(&lu) = last_use.get(name) {
            if lu >= current_node_idx {
                match ev {
                    EmitValue::Tensor(t) => live.push((name.clone(), *t)),
                    EmitValue::ShapeDims(_) => {
                        unreachable!("ShapeDims should have been materialized above");
                    }
                }
            }
        }
    }
    live.sort_by(|a, b| a.0.cmp(&b.0));
    live
}

/// Begin a new sub-function chunk, passing live values as arguments.
fn begin_chunk<'c>(
    builder: &mut GraphBuilder<'c>,
    value_map: &mut HashMap<String, EmitValue<'c>>,
    live: &[(String, Tensor<'c>)],
    chunk_index: usize,
) {
    let tensors: Vec<&Tensor<'c>> = live.iter().map(|(_, t)| t).collect();
    let name = format!("chunk_{chunk_index}");
    let (_handle, sub_args) = builder.begin_subfunction(&name, &tensors);

    // Update value_map to point to sub-function argument tensors.
    for (i, (name, _)) in live.iter().enumerate() {
        value_map.insert(name.clone(), EmitValue::Tensor(sub_args[i]));
    }
}

/// End the current sub-function, returning live values.
fn end_chunk<'c>(
    builder: &mut GraphBuilder<'c>,
    value_map: &mut HashMap<String, EmitValue<'c>>,
    live: &[(String, Tensor<'c>)],
) {
    let returns: Vec<&Tensor<'c>> = live.iter().map(|(_, t)| t).collect();
    let handle = crate::graph_builder::SubFunctionHandle { _index: 0 };
    let results = builder.end_subfunction(handle, &returns);

    // Update value_map to point to @compute-scope call results.
    for (i, (name, _)) in live.iter().enumerate() {
        value_map.insert(name.clone(), EmitValue::Tensor(results[i]));
    }
}

/// Route weight values into the current sub-function on demand.
/// Checks which inputs of `node` are weights and adds them as sub-function args
/// if not already mapped.
fn remap_node_weights<'c>(
    node: &OnnxNode,
    weight_names: &HashSet<String>,
    weight_compute_tensors: &HashMap<String, Tensor<'c>>,
    weight_remap: &mut HashMap<String, Tensor<'c>>,
    value_map: &mut HashMap<String, EmitValue<'c>>,
    builder: &mut GraphBuilder<'c>,
) {
    for input_name in &node.inputs {
        if input_name.is_empty() || !weight_names.contains(input_name) {
            continue;
        }
        if weight_remap.contains_key(input_name) {
            continue; // Already routed into this sub-function.
        }
        // Add weight as a dynamic sub-function argument.
        let compute_tensor = weight_compute_tensors[input_name];
        let sub_tensor = builder.add_subfunction_arg(&compute_tensor);
        weight_remap.insert(input_name.clone(), sub_tensor);
        value_map.insert(input_name.clone(), EmitValue::Tensor(sub_tensor));
    }
}

// ── Node dispatch ──────────────────────────────────────────────────────────────

fn emit_node<'c>(
    node: &OnnxNode,
    value_map: &mut HashMap<String, EmitValue<'c>>,
    const_i64: &mut HashMap<String, Vec<i64>>,
    builder: &mut GraphBuilder<'c>,
) -> Result<(), OnnxError> {
    match node.op_type.as_str() {
        // ── Elementwise binary ─────────────────────────────────────────────────
        "Add" => {
            let a = get_tensor(value_map, builder, &node.inputs[0])?;
            let b = get_tensor(value_map, builder, &node.inputs[1])?;
            let out = builder.emit_add(&a, &b);
            insert_tensor(value_map, &node.outputs[0], out);

            // Propagate const_i64 through Add on integer tensors (with broadcast).
            if let (Some(va), Some(vb)) = (
                lookup_const_i64(const_i64, &node.inputs[0]),
                lookup_const_i64(const_i64, &node.inputs[1]),
            ) {
                if let Some(result) = const_i64_elementwise(&va, &vb, |a, b| a + b) {
                    const_i64.insert(node.outputs[0].clone(), result);
                }
            }
        }
        "Sub" => {
            let a = get_tensor(value_map, builder, &node.inputs[0])?;
            let b = get_tensor(value_map, builder, &node.inputs[1])?;
            let out = builder.emit_sub(&a, &b);
            insert_tensor(value_map, &node.outputs[0], out);

            // Propagate const_i64 through Sub on integer tensors (with broadcast).
            if let (Some(va), Some(vb)) = (
                lookup_const_i64(const_i64, &node.inputs[0]),
                lookup_const_i64(const_i64, &node.inputs[1]),
            ) {
                if let Some(result) = const_i64_elementwise(&va, &vb, |a, b| a - b) {
                    const_i64.insert(node.outputs[0].clone(), result);
                }
            }
        }
        "Mul" => {
            let a = get_tensor(value_map, builder, &node.inputs[0])?;
            let b = get_tensor(value_map, builder, &node.inputs[1])?;
            let out = builder.emit_mul(&a, &b);
            insert_tensor(value_map, &node.outputs[0], out);

            if let (Some(va), Some(vb)) = (
                lookup_const_i64(const_i64, &node.inputs[0]),
                lookup_const_i64(const_i64, &node.inputs[1]),
            ) {
                if let Some(result) = const_i64_elementwise(&va, &vb, |a, b| a * b) {
                    const_i64.insert(node.outputs[0].clone(), result);
                }
            }
        }
        "Div" => {
            let a = get_tensor(value_map, builder, &node.inputs[0])?;
            let b = get_tensor(value_map, builder, &node.inputs[1])?;
            let out = builder.emit_div(&a, &b);
            insert_tensor(value_map, &node.outputs[0], out);

            if let (Some(va), Some(vb)) = (
                lookup_const_i64(const_i64, &node.inputs[0]),
                lookup_const_i64(const_i64, &node.inputs[1]),
            ) {
                if vb.iter().all(|&b| b != 0) {
                    if let Some(result) = const_i64_elementwise(&va, &vb, |a, b| a / b) {
                        const_i64.insert(node.outputs[0].clone(), result);
                    }
                }
            }
        }
        "Neg" => {
            let a = get_tensor(value_map, builder, &node.inputs[0])?;
            let out = builder.emit_neg(&a);
            insert_tensor(value_map, &node.outputs[0], out);
        }
        "Pow" => {
            let base = get_tensor(value_map, builder, &node.inputs[0])?;
            let exp = get_tensor(value_map, builder, &node.inputs[1])?;
            let out = builder.emit_pow(&base, &exp);
            insert_tensor(value_map, &node.outputs[0], out);
        }

        // ── Elementwise unary ──────────────────────────────────────────────────
        "Relu" => {
            let a = get_tensor(value_map, builder, &node.inputs[0])?;
            let out = builder.emit_relu(&a);
            insert_tensor(value_map, &node.outputs[0], out);
        }
        "Exp" => {
            let a = get_tensor(value_map, builder, &node.inputs[0])?;
            let out = builder.emit_exp(&a);
            insert_tensor(value_map, &node.outputs[0], out);
        }
        "Tanh" => {
            let a = get_tensor(value_map, builder, &node.inputs[0])?;
            let out = builder.emit_tanh(&a);
            insert_tensor(value_map, &node.outputs[0], out);
        }
        "Sqrt" => {
            let a = get_tensor(value_map, builder, &node.inputs[0])?;
            let out = builder.emit_sqrt(&a);
            insert_tensor(value_map, &node.outputs[0], out);
        }
        "Clip" => {
            if node.inputs.len() > 1 && node.inputs[1..].iter().any(|s| !s.is_empty()) {
                return Err(OnnxError::UnsupportedOp(
                    "Clip with non-zero bounds (min/max inputs) not supported".to_string(),
                ));
            }
            let a = get_tensor(value_map, builder, &node.inputs[0])?;
            let out = builder.emit_relu(&a);
            insert_tensor(value_map, &node.outputs[0], out);
        }

        // ── Matrix ops ────────────────────────────────────────────────────────
        "MatMul" => {
            let a = get_tensor(value_map, builder, &node.inputs[0])?;
            let b = get_tensor(value_map, builder, &node.inputs[1])?;
            let out = builder.emit_matmul(&a, &b);
            insert_tensor(value_map, &node.outputs[0], out);
        }
        "Gemm" => {
            let a = get_tensor(value_map, builder, &node.inputs[0])?;
            let b = get_tensor(value_map, builder, &node.inputs[1])?;
            let c = if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
                get_tensor(value_map, builder, &node.inputs[2])?
            } else {
                builder.emit_arith_constant(0.0, a.dtype())
            };
            let alpha = get_float_attr(&node.attributes, "alpha", 1.0) as f32;
            let beta = get_float_attr(&node.attributes, "beta", 1.0) as f32;
            let trans_a = get_bool_attr(&node.attributes, "transA");
            let trans_b = get_bool_attr(&node.attributes, "transB");
            let out = builder.emit_gemm(&a, &b, &c, alpha, beta, trans_a, trans_b);
            insert_tensor(value_map, &node.outputs[0], out);
        }

        // ── Softmax ──────────────────────────────────────────────────────────
        "Softmax" => {
            let a = get_tensor(value_map, builder, &node.inputs[0])?;
            let axis = get_int_attr(&node.attributes, "axis", -1);
            let out = builder.emit_softmax(&a, axis);
            insert_tensor(value_map, &node.outputs[0], out);
        }

        // ── Reductions ───────────────────────────────────────────────────────
        "ReduceSum" => {
            let a = get_tensor(value_map, builder, &node.inputs[0])?;
            let keepdim = get_keepdims_attr(&node.attributes, true);
            let axes = get_reduce_axes(node, const_i64, a.rank())?;
            let out = emit_multi_axis_reduce(builder, &a, &axes, keepdim, ReduceOp::Sum);
            insert_tensor(value_map, &node.outputs[0], out);
        }
        "ReduceMax" => {
            let a = get_tensor(value_map, builder, &node.inputs[0])?;
            let keepdim = get_keepdims_attr(&node.attributes, true);
            let axes = get_reduce_axes(node, const_i64, a.rank())?;
            let out = emit_multi_axis_reduce(builder, &a, &axes, keepdim, ReduceOp::Max);
            insert_tensor(value_map, &node.outputs[0], out);
        }
        "ReduceMean" => {
            let a = get_tensor(value_map, builder, &node.inputs[0])?;
            let keepdim = get_keepdims_attr(&node.attributes, true);
            let axes = get_reduce_axes(node, const_i64, a.rank())?;
            let out = builder.emit_reduce_mean(&a, &axes, keepdim);
            insert_tensor(value_map, &node.outputs[0], out);
        }

        // ── Spatial ops ──────────────────────────────────────────────────────
        "Conv" => {
            let group = get_int_attr(&node.attributes, "group", 1);
            if group != 1 {
                return Err(OnnxError::UnsupportedOp(format!(
                    "Conv: group={group} not supported"
                )));
            }
            let x = get_tensor(value_map, builder, &node.inputs[0])?;
            let w = get_tensor(value_map, builder, &node.inputs[1])?;
            let bias = if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
                Some(get_tensor(value_map, builder, &node.inputs[2])?)
            } else {
                None
            };
            let strides = get_ints_attr(&node.attributes, "strides", &[1, 1]);
            let dilations = get_ints_attr(&node.attributes, "dilations", &[1, 1]);
            let auto_pad = get_str_attr(&node.attributes, "auto_pad", "NOTSET");

            let pads = if auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER" {
                let in_h = x.shape()[2].expect("conv input H must be static") as i64;
                let in_w = x.shape()[3].expect("conv input W must be static") as i64;
                let kh = w.shape()[2].expect("conv weight KH must be static") as i64;
                let kw = w.shape()[3].expect("conv weight KW must be static") as i64;
                let sh = strides[0];
                let sw = strides[1];
                let dh = dilations[0];
                let dw = dilations[1];
                let out_h = (in_h + sh - 1) / sh;
                let out_w = (in_w + sw - 1) / sw;
                let pad_h = 0.max((out_h - 1) * sh + dh * (kh - 1) + 1 - in_h);
                let pad_w = 0.max((out_w - 1) * sw + dw * (kw - 1) + 1 - in_w);
                if auto_pad == "SAME_UPPER" {
                    vec![pad_h / 2, pad_w / 2, pad_h - pad_h / 2, pad_w - pad_w / 2]
                } else {
                    vec![pad_h - pad_h / 2, pad_w - pad_w / 2, pad_h / 2, pad_w / 2]
                }
            } else {
                get_ints_attr(&node.attributes, "pads", &[0, 0, 0, 0])
            };

            let out = builder.emit_conv2d(
                &x, &w, bias.as_ref(),
                [pads[0] as u64, pads[1] as u64, pads[2] as u64, pads[3] as u64],
                [strides[0] as u64, strides[1] as u64],
                [dilations[0] as u64, dilations[1] as u64],
            );
            insert_tensor(value_map, &node.outputs[0], out);
        }
        "MaxPool" => {
            let x = get_tensor(value_map, builder, &node.inputs[0])?;
            let kernel_shape = get_ints_attr(&node.attributes, "kernel_shape", &[]);
            if kernel_shape.len() != 2 {
                return Err(OnnxError::UnsupportedOp(
                    "MaxPool: kernel_shape must have exactly 2 elements".to_string(),
                ));
            }
            let strides = get_ints_attr(&node.attributes, "strides", &[1, 1]);
            let pads = get_ints_attr(&node.attributes, "pads", &[0, 0, 0, 0]);
            let dilations = get_ints_attr(&node.attributes, "dilations", &[1, 1]);
            let out = builder.emit_max_pool2d(
                &x,
                [kernel_shape[0] as u64, kernel_shape[1] as u64],
                [pads[0] as u64, pads[1] as u64, pads[2] as u64, pads[3] as u64],
                [strides[0] as u64, strides[1] as u64],
                [dilations[0] as u64, dilations[1] as u64],
            );
            insert_tensor(value_map, &node.outputs[0], out);
        }
        "GlobalAveragePool" => {
            let x = get_tensor(value_map, builder, &node.inputs[0])?;
            let out = builder.emit_global_avg_pool(&x);
            insert_tensor(value_map, &node.outputs[0], out);
        }
        "BatchNormalization" => {
            let x = get_tensor(value_map, builder, &node.inputs[0])?;
            let scale = get_tensor(value_map, builder, &node.inputs[1])?;
            let bias = get_tensor(value_map, builder, &node.inputs[2])?;
            let mean = get_tensor(value_map, builder, &node.inputs[3])?;
            let var = get_tensor(value_map, builder, &node.inputs[4])?;
            let epsilon = get_float_attr(&node.attributes, "epsilon", 1e-5) as f32;
            let out = builder.emit_batch_norm(&x, &scale, &bias, &mean, &var, epsilon);
            insert_tensor(value_map, &node.outputs[0], out);
        }

        // ── Shape manipulation ────────────────────────────────────────────────
        "Reshape" => {
            let x = get_tensor(value_map, builder, &node.inputs[0])?;
            let shape_name = &node.inputs[1];

            // Static path: shape input is fully known (no CONST_I64_UNKNOWN sentinels).
            let static_target = lookup_const_i64(const_i64, shape_name)
                .filter(|vals| vals.iter().all(|&v| v != CONST_I64_UNKNOWN));

            let out = if let Some(mut target_shape) = static_target {
                // In ONNX, `0` means "copy this dimension from the input" (when allowzero=0,
                // the default). Resolve zeros before passing to emit_reshape.
                let allowzero = get_int_attr(&node.attributes, "allowzero", 0);
                if allowzero == 0 {
                    let in_shape = x.shape();
                    for (i, d) in target_shape.iter_mut().enumerate() {
                        if *d == 0 {
                            if i < in_shape.len() {
                                *d = in_shape[i].map(|n| n as i64).unwrap_or(0);
                            }
                        }
                    }
                }
                builder.emit_reshape(&x, &target_shape)
            } else {
                // Dynamic path: shape comes from a runtime tensor (Concat output,
                // ShapeDims, etc.).
                //
                // The shape tensor may contain ONNX special values:
                //   -1 = "infer this dimension" (total_input_elems / product_of_others)
                //    0 = "copy from input dim i" (when allowzero=0, the default)
                //
                // tensor.reshape does NOT handle these — we must resolve them first.
                let shape_val = value_map.get(shape_name)
                    .ok_or_else(|| OnnxError::MissingEdge(shape_name.clone()))?;
                let shape_tensor = shape_val.as_tensor(builder);

                let shape_tensor_shape = shape_tensor.shape();
                let out_rank = shape_tensor_shape.first()
                    .and_then(|d| *d)
                    .unwrap_or_else(|| {
                        panic!("Reshape: shape tensor '{}' must have static rank-1 size", shape_name)
                    }) as usize;

                // Extract individual dim values as index values.
                let dim_indices = extract_dims_from_i64_tensor(builder, &shape_tensor);
                let allowzero = get_int_attr(&node.attributes, "allowzero", 0);

                // Resolve -1 and 0 entries, producing corrected index values.
                let corrected = builder.emit_resolve_reshape_dims(
                    &x, &dim_indices, allowzero != 0,
                );

                // Compute static output shape where possible.
                // If const_i64 propagation (e.g. through Concat) made the shape
                // tensor's values available, use them to resolve dims statically.
                // This prevents the all-dynamic cascade that breaks expand_shape.
                let shape_static = lookup_const_i64(const_i64, shape_name);
                let in_shape = x.shape();
                let allowzero_flag = allowzero != 0;

                let out_shape: Vec<Option<u64>> = (0..out_rank).map(|i| {
                    let raw_val: Option<i64> = shape_static.as_ref()
                        .and_then(|v| v.get(i).copied())
                        .filter(|&v| v != CONST_I64_UNKNOWN);

                    match raw_val {
                        Some(d) if d > 0 => Some(d as u64),
                        Some(0) if !allowzero_flag => {
                            // Copy from input dim i.
                            in_shape.get(i).and_then(|opt| *opt)
                        }
                        Some(-1) => {
                            // Infer: total_input / product_of_known_target.
                            let total_static: Option<u64> = in_shape.iter()
                                .try_fold(1u64, |acc, d| d.map(|n| acc * n));
                            let known_product: Option<u64> = shape_static.as_ref().and_then(|sv| {
                                sv.iter().enumerate().try_fold(1u64, |acc, (j, &d)| {
                                    if j == i { Some(acc) } // skip the -1 dim
                                    else if d > 0 && d != CONST_I64_UNKNOWN { Some(acc * d as u64) }
                                    else if d == 0 && !allowzero_flag {
                                        in_shape.get(j).and_then(|opt| opt.map(|n| acc * n))
                                    }
                                    else { None }
                                })
                            });
                            match (total_static, known_product) {
                                (Some(t), Some(k)) if k > 0 => Some(t / k),
                                _ => None,
                            }
                        }
                        _ => None, // truly dynamic (or CONST_I64_UNKNOWN)
                    }
                }).collect();

                builder.emit_reshape_from_index_dims(&x, &corrected, &out_shape)
            };
            insert_tensor(value_map, &node.outputs[0], out);

            // Propagate const_i64 through Reshape (data values unchanged).
            if let Some(vals) = lookup_const_i64(const_i64, &node.inputs[0]) {
                const_i64.insert(node.outputs[0].clone(), vals);
            }
        }
        "Flatten" => {
            let x = get_tensor(value_map, builder, &node.inputs[0])?;
            let axis = get_int_attr(&node.attributes, "axis", 1) as usize;
            let out = builder.emit_flatten(&x, axis);
            insert_tensor(value_map, &node.outputs[0], out);
        }
        "Transpose" => {
            let x = get_tensor(value_map, builder, &node.inputs[0])?;
            let rank = x.rank();
            let perm: Vec<i64> = if let Some(OnnxAttribute::Ints(v)) = node.attributes.get("perm") {
                v.clone()
            } else {
                (0..rank as i64).rev().collect()
            };
            let out = builder.emit_transpose(&x, &perm);
            insert_tensor(value_map, &node.outputs[0], out);
        }
        "Concat" => {
            let axis = node.attributes.get("axis")
                .and_then(|a| if let OnnxAttribute::Int(v) = a { Some(*v) } else { None })
                .ok_or_else(|| OnnxError::UnsupportedOp(
                    "Concat: missing required `axis` attribute".to_string(),
                ))?;
            let first = get_tensor(value_map, builder, &node.inputs[0])?;
            let axis_usize = normalize_axis(axis, first.rank());
            let mut inputs: Vec<Tensor<'c>> = vec![first];
            for name in &node.inputs[1..] {
                inputs.push(get_tensor(value_map, builder, name)?);
            }
            let out = builder.emit_concat(&inputs, axis_usize);
            insert_tensor(value_map, &node.outputs[0], out);

            // Propagate const_i64 through concatenation so downstream Reshape
            // ops can take the static path. Support PARTIAL propagation:
            // use CONST_I64_UNKNOWN sentinel for inputs without const values,
            // so that known values (like 12, 64, 768) still reach Reshape.
            {
                let mut concat_vals: Vec<i64> = Vec::new();
                let mut has_any = false;
                for name in &node.inputs {
                    if let Some(vals) = const_i64.get(name) {
                        concat_vals.extend(vals);
                        has_any = true;
                    } else {
                        // Unknown — use sentinel. Check the MLIR tensor shape
                        // to determine how many values this input contributes.
                        let n = match value_map.get(name) {
                            Some(EmitValue::Tensor(t)) => {
                                let s = t.shape();
                                if s.len() == 1 { s[0].unwrap_or(1) as usize }
                                else { 1 }
                            }
                            Some(EmitValue::ShapeDims(dims)) => dims.len(),
                            None => 1,
                        };
                        concat_vals.extend(std::iter::repeat(CONST_I64_UNKNOWN).take(n));
                    }
                }
                if has_any {
                    const_i64.insert(node.outputs[0].clone(), concat_vals);
                }
            }
        }
        "Slice" => {
            let data = get_tensor(value_map, builder, &node.inputs[0])?;

            let starts_static = lookup_const_i64(const_i64, &node.inputs[1]);
            let ends_static   = lookup_const_i64(const_i64, &node.inputs[2]);

            // axes and steps are always from initializers or attributes in ONNX.
            let axes: Vec<i64> = if node.inputs.len() > 3 && !node.inputs[3].is_empty() {
                lookup_const_i64(const_i64, &node.inputs[3])
                    .ok_or_else(|| OnnxError::UnsupportedOp(format!(
                        "Slice: axes input '{}' must be a static initializer", node.inputs[3]
                    )))?
            } else {
                Vec::new() // placeholder; filled in below
            };

            if let (Some(starts), Some(ends)) = (starts_static, ends_static) {
                // ── Static path ──────────────────────────────────────────────
                let axes = if axes.is_empty() {
                    (0..starts.len() as i64).collect::<Vec<_>>()
                } else {
                    axes
                };
                let steps: Vec<i64> = if node.inputs.len() > 4 && !node.inputs[4].is_empty() {
                    lookup_const_i64(const_i64, &node.inputs[4])
                        .ok_or_else(|| OnnxError::UnsupportedOp(format!(
                            "Slice: steps input '{}' must be a static initializer", node.inputs[4]
                        )))?
                } else {
                    vec![1i64; starts.len()]
                };
                let out = builder.emit_slice(&data, &starts, &ends, &axes, &steps);
                insert_tensor(value_map, &node.outputs[0], out);

                // Propagate const_i64 through static Slice on 1-D data.
                if let Some(data_vals) = lookup_const_i64(const_i64, &node.inputs[0]) {
                    if axes.len() == 1 && axes[0] == 0 && steps.iter().all(|&s| s == 1) {
                        let n = data_vals.len() as i64;
                        let s = if starts[0] < 0 { (n + starts[0]).max(0) } else { starts[0].min(n) } as usize;
                        let e = if ends[0] < 0 { (n + ends[0]).max(0) } else { ends[0].min(n) } as usize;
                        if s <= e && e <= data_vals.len() {
                            const_i64.insert(node.outputs[0].clone(), data_vals[s..e].to_vec());
                        }
                    }
                }
            } else {
                // ── Dynamic path ─────────────────────────────────────────────
                // starts/ends are runtime 1-D I64 tensors in value_map.
                let starts_t = get_tensor(value_map, builder, &node.inputs[1])?;
                let ends_t   = get_tensor(value_map, builder, &node.inputs[2])?;

                // Number of sliced axes = length of the starts tensor.
                let n_axes = match starts_t.shape().first() {
                    Some(Some(n)) => *n as usize,
                    _ => return Err(OnnxError::UnsupportedOp(format!(
                        "Slice: dynamic starts tensor '{}' must have a static length",
                        node.inputs[1]
                    ))),
                };

                let axes = if axes.is_empty() {
                    (0..n_axes as i64).collect::<Vec<_>>()
                } else {
                    axes
                };
                let steps: Vec<i64> = if node.inputs.len() > 4 && !node.inputs[4].is_empty() {
                    lookup_const_i64(const_i64, &node.inputs[4])
                        .ok_or_else(|| OnnxError::UnsupportedOp(format!(
                            "Slice: steps input '{}' must be a static initializer", node.inputs[4]
                        )))?
                } else {
                    vec![1i64; n_axes]
                };

                // Extract each element from the starts/ends 1-D tensors.
                let start_vals: Vec<Value> = (0..n_axes)
                    .map(|i| builder.emit_tensor_extract_scalar(&starts_t, i))
                    .collect();
                let end_vals: Vec<Value> = (0..n_axes)
                    .map(|i| builder.emit_tensor_extract_scalar(&ends_t, i))
                    .collect();

                let out = builder.emit_dynamic_slice(&data, &start_vals, &end_vals, &axes, &steps);
                insert_tensor(value_map, &node.outputs[0], out);
            }
        }
        "Gather" => {
            let data = get_tensor(value_map, builder, &node.inputs[0])?;
            let indices = get_tensor(value_map, builder, &node.inputs[1])?;
            let axis = get_int_attr(&node.attributes, "axis", 0);
            let axis_usize = normalize_axis(axis, data.rank());
            let out = builder.emit_gather(&data, &indices, axis_usize);
            insert_tensor(value_map, &node.outputs[0], out);

            // Propagate const_i64: if data and indices are both static constants,
            // compute the gathered values so downstream Reshape can use them.
            if let (Some(data_vals), Some(idx_vals)) = (
                lookup_const_i64(const_i64, &node.inputs[0]),
                lookup_const_i64(const_i64, &node.inputs[1]),
            ) {
                if axis_usize == 0 && data.rank() == 1 {
                    let gathered: Vec<i64> = idx_vals.iter().map(|&idx| {
                        let i = if idx < 0 { (data_vals.len() as i64 + idx) as usize } else { idx as usize };
                        data_vals[i]
                    }).collect();
                    const_i64.insert(node.outputs[0].clone(), gathered);
                }
            }
        }
        "Where" => {
            let cond = get_tensor(value_map, builder, &node.inputs[0])?;
            let x = get_tensor(value_map, builder, &node.inputs[1])?;
            let y = get_tensor(value_map, builder, &node.inputs[2])?;
            let cond_i64 = if cond.dtype() != DType::I64 {
                builder.emit_cast(&cond, DType::I64)
            } else {
                cond
            };
            let out = builder.emit_where(&cond_i64, &x, &y);
            insert_tensor(value_map, &node.outputs[0], out);
        }
        "Cast" => {
            let a = get_tensor(value_map, builder, &node.inputs[0])?;
            let to = get_int_attr(&node.attributes, "to", 0);
            let target = onnx_dtype_to_internal(to)
                .ok_or_else(|| OnnxError::UnsupportedOp(format!("Cast: unsupported ONNX dtype {to}")))?;
            let out = builder.emit_cast(&a, target);
            insert_tensor(value_map, &node.outputs[0], out);

            // Propagate const_i64 through Cast (integer values preserved for int→int casts).
            if let Some(vals) = lookup_const_i64(const_i64, &node.inputs[0]) {
                const_i64.insert(node.outputs[0].clone(), vals);
            }
        }
        "Unsqueeze" => {
            let x = get_tensor(value_map, builder, &node.inputs[0])?;
            let axes = get_axes_from_input_or_attr(node, const_i64, "Unsqueeze")?;
            let out = builder.emit_unsqueeze(&x, &axes);
            insert_tensor(value_map, &node.outputs[0], out);

            // Propagate const_i64 through Unsqueeze (values unchanged, just shape).
            if let Some(vals) = lookup_const_i64(const_i64, &node.inputs[0]) {
                const_i64.insert(node.outputs[0].clone(), vals);
            }
        }
        "Squeeze" => {
            let x = get_tensor(value_map, builder, &node.inputs[0])?;
            let axes = if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
                lookup_const_i64(const_i64, &node.inputs[1])
                    .ok_or_else(|| OnnxError::UnsupportedOp(format!(
                        "Squeeze: axes input '{}' must be a static initializer", node.inputs[1]
                    )))?
            } else if let Some(OnnxAttribute::Ints(v)) = node.attributes.get("axes") {
                v.clone()
            } else {
                vec![]
            };
            let out = builder.emit_squeeze(&x, &axes);
            insert_tensor(value_map, &node.outputs[0], out);

            // Propagate const_i64 through Squeeze (values unchanged).
            if let Some(vals) = lookup_const_i64(const_i64, &node.inputs[0]) {
                const_i64.insert(node.outputs[0].clone(), vals);
            }
        }
        "Split" => {
            let input = get_tensor(value_map, builder, &node.inputs[0])?;
            let axis = get_int_attr(&node.attributes, "axis", 0);
            let axis_usize = normalize_axis(axis, input.rank());
            let num_outputs = node.outputs.len();

            let split_sizes: Vec<u64> = if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
                lookup_const_i64(const_i64, &node.inputs[1])
                    .ok_or_else(|| OnnxError::UnsupportedOp(format!(
                        "Split: split input '{}' must be a static initializer", node.inputs[1]
                    )))?
                    .iter().map(|&v| v as u64).collect()
            } else if let Some(OnnxAttribute::Ints(v)) = node.attributes.get("split") {
                v.iter().map(|&v| v as u64).collect()
            } else {
                let ax_dim = input.shape()[axis_usize]
                    .ok_or_else(|| OnnxError::UnsupportedOp(
                        "Split: split axis dimension is dynamic".to_string(),
                    ))?;
                let each = ax_dim / num_outputs as u64;
                vec![each; num_outputs]
            };

            let slices = builder.emit_split(&input, axis_usize, &split_sizes);
            for (i, t) in slices.into_iter().enumerate() {
                insert_tensor(value_map, &node.outputs[i], t);
            }
        }

        // ── Meta / shape subgraph ops ──────────────────────────────────────────
        "Shape" => {
            let input = get_tensor(value_map, builder, &node.inputs[0])?;
            let dims = builder.emit_shape_of(&input);

            // Seed const_i64 with static shape values so downstream Gather/Concat/
            // Reshape chains can resolve dims at compile time.
            let in_shape = input.shape();
            let static_vals: Option<Vec<i64>> = in_shape.iter()
                .map(|d| d.map(|n| n as i64))
                .collect();
            if let Some(vals) = static_vals {
                const_i64.insert(node.outputs[0].clone(), vals);
            }

            // Store as ShapeDims — downstream ops (Reshape, ConstantOfShape) can
            // materialize as a tensor or use the index values directly.
            value_map.insert(node.outputs[0].clone(), EmitValue::ShapeDims(dims));
        }
        "ConstantOfShape" => {
            let shape_name = &node.inputs[0];
            let (shape_vals, static_shape): (Vec<Value<'c, 'c>>, Vec<Option<u64>>) =
                match value_map.get(shape_name)
                    .ok_or_else(|| OnnxError::MissingEdge(shape_name.clone()))?
                {
                    EmitValue::ShapeDims(dims) => {
                        // Each dim is a runtime index value; we don't know statics.
                        let statics: Vec<Option<u64>> = dims.iter().map(|_| None).collect();
                        (dims.clone(), statics)
                    }
                    EmitValue::Tensor(t) => {
                        let t = *t;
                        let n = t.shape()[0].unwrap_or(0) as usize;
                        let statics: Vec<Option<u64>> = vec![None; n];
                        let shape_idx_vals = extract_dims_from_i64_tensor(builder, &t);
                        (shape_idx_vals, statics)
                    }
                };

            let (fill_value, fill_dtype) = if let Some(OnnxAttribute::Tensor(val_buf)) = node.attributes.get("value") {
                let fv = match val_buf.dtype() {
                    DType::F32 => val_buf.as_slice::<f32>()[0] as f64,
                    DType::F64 => val_buf.as_slice::<f64>()[0],
                    DType::I32 => val_buf.as_slice::<i32>()[0] as f64,
                    DType::I64 => val_buf.as_slice::<i64>()[0] as f64,
                };
                (fv, val_buf.dtype())
            } else {
                (0.0, DType::F32)
            };

            let out = builder.emit_constant_of_shape(&shape_vals, &static_shape, fill_value, fill_dtype);
            insert_tensor(value_map, &node.outputs[0], out);
        }
        "Range" => {
            let start_t = get_tensor(value_map, builder, &node.inputs[0])?;
            let limit_t = get_tensor(value_map, builder, &node.inputs[1])?;
            let delta_t = get_tensor(value_map, builder, &node.inputs[2])?;
            let dtype = start_t.dtype();
            let start = extract_scalar_as_index(builder, &start_t);
            let limit = extract_scalar_as_index(builder, &limit_t);
            let delta = extract_scalar_as_index(builder, &delta_t);

            // Try to compute output size statically: ceil((limit - start) / delta).
            let output_size = match (
                lookup_const_i64(const_i64, &node.inputs[0]),
                lookup_const_i64(const_i64, &node.inputs[1]),
                lookup_const_i64(const_i64, &node.inputs[2]),
            ) {
                (Some(s), Some(l), Some(d)) if !s.is_empty() && !l.is_empty() && !d.is_empty() => {
                    let sv = s[0];
                    let lv = l[0];
                    let dv = d[0];
                    if dv != 0 {
                        Some(((lv - sv + dv - 1) / dv).max(0) as u64)
                    } else {
                        None
                    }
                }
                _ => None,
            };

            let out = builder.emit_range(start, limit, delta, output_size, dtype);
            insert_tensor(value_map, &node.outputs[0], out);
        }
        "Constant" => {
            let buf = node.attributes.get("value")
                .and_then(|attr| if let OnnxAttribute::Tensor(b) = attr { Some(b) } else { None })
                .ok_or_else(|| OnnxError::UnsupportedOp(
                    "Constant: missing or non-tensor `value` attribute".to_string(),
                ))?;
            // Seed const_i64 for small integer constants (axes, split sizes, shape inputs).
            if buf.shape().num_elements() <= 64 {
                if let Ok(vals) = read_i64_buffer(buf) {
                    const_i64.insert(node.outputs[0].clone(), vals);
                }
            }
            let out = emit_buffer_as_constant(builder, buf);
            insert_tensor(value_map, &node.outputs[0], out);
        }

        other => {
            return Err(OnnxError::UnsupportedOp(other.to_string()));
        }
    }

    Ok(())
}

// ── Constant emission ─────────────────────────────────────────────────────────

fn emit_buffer_as_constant<'c>(builder: &mut GraphBuilder<'c>, buf: &Buffer) -> Tensor<'c> {
    let shape = &buf.shape().0;
    let dtype = buf.dtype();
    let data_str = buffer_to_dense_str(buf);
    builder.emit_dense_constant(&data_str, shape, dtype)
}

fn buffer_to_dense_str(buf: &Buffer) -> String {
    let shape = &buf.shape().0;
    match buf.dtype() {
        DType::F32 => {
            let vals = buf.as_slice::<f32>();
            format_dense_nd(vals.iter().map(|&v| format!("{v:.6e}")), shape)
        }
        DType::F64 => {
            let vals = buf.as_slice::<f64>();
            format_dense_nd(vals.iter().map(|&v| format!("{v:.15e}")), shape)
        }
        DType::I32 => {
            let vals = buf.as_slice::<i32>();
            format_dense_nd(vals.iter().map(|&v| format!("{v}")), shape)
        }
        DType::I64 => {
            let vals = buf.as_slice::<i64>();
            format_dense_nd(vals.iter().map(|&v| format!("{v}")), shape)
        }
    }
}

fn format_dense_nd<I: Iterator<Item = String>>(vals: I, shape: &[u64]) -> String {
    let flat: Vec<String> = vals.collect();
    if shape.is_empty() {
        return flat.into_iter().next().unwrap_or_else(|| "0".to_string());
    }
    nest_dense(&flat, shape, 0)
}

fn nest_dense(flat: &[String], shape: &[u64], dim: usize) -> String {
    if dim == shape.len() - 1 {
        return format!("[{}]", flat.join(", "));
    }
    let inner_size: usize = shape[dim + 1..].iter().map(|&d| d as usize).product();
    let n = shape[dim] as usize;
    let parts: Vec<String> = (0..n)
        .map(|i| nest_dense(&flat[i * inner_size..(i + 1) * inner_size], shape, dim + 1))
        .collect();
    format!("[{}]", parts.join(", "))
}

// ── Buffer helpers ─────────────────────────────────────────────────────────────

fn read_i64_buffer(buf: &Buffer) -> Result<Vec<i64>, OnnxError> {
    match buf.dtype() {
        DType::I64 => Ok(buf.as_slice::<i64>().to_vec()),
        DType::I32 => Ok(buf.as_slice::<i32>().iter().map(|&v| v as i64).collect()),
        other => Err(OnnxError::UnsupportedOp(format!(
            "expected integer buffer (I32/I64), got {other:?}"
        ))),
    }
}

// ── Multi-axis reduce helper ──────────────────────────────────────────────────

enum ReduceOp { Sum, Max }

fn emit_multi_axis_reduce<'c>(
    builder: &mut GraphBuilder<'c>,
    input: &Tensor<'c>,
    axes: &[i64],
    keepdim: bool,
    op: ReduceOp,
) -> Tensor<'c> {
    if axes.is_empty() {
        let all: Vec<i64> = (0..input.rank() as i64).collect();
        return emit_multi_axis_reduce(builder, input, &all, keepdim, op);
    }
    if axes.len() == 1 {
        return match op {
            ReduceOp::Sum => builder.emit_reduce_sum(input, axes[0], keepdim),
            ReduceOp::Max => builder.emit_reduce_max(input, axes[0], keepdim),
        };
    }
    // Sort descending to avoid index shift as dims are removed.
    let rank = input.rank() as i64;
    let mut norm: Vec<i64> = axes.iter()
        .map(|&a| if a < 0 { a + rank } else { a })
        .collect();
    norm.sort_unstable();
    norm.dedup();

    let mut cur = *input;
    for &ax in norm.iter().rev() {
        cur = match op {
            ReduceOp::Sum => builder.emit_reduce_sum(&cur, ax, keepdim),
            ReduceOp::Max => builder.emit_reduce_max(&cur, ax, keepdim),
        };
    }
    cur
}

// ── Attribute extraction helpers ──────────────────────────────────────────────

fn get_int_attr(attrs: &HashMap<String, OnnxAttribute>, name: &str, default: i64) -> i64 {
    attrs.get(name).and_then(|a| {
        if let OnnxAttribute::Int(v) = a { Some(*v) } else { None }
    }).unwrap_or(default)
}

fn get_float_attr(attrs: &HashMap<String, OnnxAttribute>, name: &str, default: f64) -> f64 {
    attrs.get(name).and_then(|a| {
        if let OnnxAttribute::Float(v) = a { Some(*v as f64) } else { None }
    }).unwrap_or(default)
}

fn get_bool_attr(attrs: &HashMap<String, OnnxAttribute>, name: &str) -> bool {
    attrs.get(name).and_then(|a| {
        if let OnnxAttribute::Int(v) = a { Some(*v != 0) } else { None }
    }).unwrap_or(false)
}

fn get_keepdims_attr(attrs: &HashMap<String, OnnxAttribute>, default: bool) -> bool {
    attrs.get("keepdims").and_then(|a| {
        if let OnnxAttribute::Int(v) = a { Some(*v != 0) } else { None }
    }).unwrap_or(default)
}

fn get_str_attr<'a>(attrs: &'a HashMap<String, OnnxAttribute>, name: &str, default: &'a str) -> &'a str {
    attrs.get(name).and_then(|a| {
        if let OnnxAttribute::String(v) = a { Some(v.as_str()) } else { None }
    }).unwrap_or(default)
}

fn get_ints_attr(attrs: &HashMap<String, OnnxAttribute>, name: &str, default: &[i64]) -> Vec<i64> {
    attrs.get(name).and_then(|a| {
        if let OnnxAttribute::Ints(v) = a { Some(v.clone()) } else { None }
    }).unwrap_or_else(|| default.to_vec())
}

fn onnx_dtype_to_internal(onnx_dtype: i64) -> Option<DType> {
    match onnx_dtype {
        1 => Some(DType::F32),
        6 => Some(DType::I32),
        7 | 9 => Some(DType::I64),
        11 => Some(DType::F64),
        _ => None,
    }
}

fn normalize_axis(axis: i64, rank: usize) -> usize {
    let r = rank as i64;
    let a = if axis < 0 { axis + r } else { axis };
    assert!(a >= 0 && a < r, "axis {axis} out of range for rank {rank}");
    a as usize
}

fn get_axes_from_input_or_attr(
    node: &OnnxNode,
    const_i64: &HashMap<String, Vec<i64>>,
    op_name: &str,
) -> Result<Vec<i64>, OnnxError> {
    if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
        lookup_const_i64(const_i64, &node.inputs[1])
            .ok_or_else(|| OnnxError::UnsupportedOp(format!(
                "{op_name}: axes input '{}' must be a static initializer or Constant node", node.inputs[1]
            )))
    } else if let Some(OnnxAttribute::Ints(v)) = node.attributes.get("axes") {
        Ok(v.clone())
    } else {
        Err(OnnxError::UnsupportedOp(format!("{op_name}: no axes specified")))
    }
}

fn get_reduce_axes(
    node: &OnnxNode,
    const_i64: &HashMap<String, Vec<i64>>,
    rank: usize,
) -> Result<Vec<i64>, OnnxError> {
    if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
        lookup_const_i64(const_i64, &node.inputs[1])
            .ok_or_else(|| OnnxError::UnsupportedOp(format!(
                "{}: axes input '{}' must be a static initializer or Constant node",
                node.op_type, node.inputs[1]
            )))
    } else if let Some(OnnxAttribute::Ints(v)) = node.attributes.get("axes") {
        Ok(v.clone())
    } else {
        Ok((0..rank as i64).collect())
    }
}

// ── Value map helpers ─────────────────────────────────────────────────────────

fn get_tensor<'c>(
    value_map: &mut HashMap<String, EmitValue<'c>>,
    builder: &mut GraphBuilder<'c>,
    name: &str,
) -> Result<Tensor<'c>, OnnxError> {
    let ev = value_map.get(name)
        .ok_or_else(|| OnnxError::MissingEdge(name.to_string()))?;
    Ok(ev.as_tensor(builder))
}

fn insert_tensor<'c>(
    value_map: &mut HashMap<String, EmitValue<'c>>,
    name: &str,
    t: Tensor<'c>,
) {
    value_map.insert(name.to_string(), EmitValue::Tensor(t));
}
