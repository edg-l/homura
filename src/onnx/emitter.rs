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
use super::sym_shapes;

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

    let i64_vals: Vec<Value<'c, 'c>> = dims
        .iter()
        .map(|&idx| {
            builder
                .block()
                .append_operation(
                    OperationBuilder::new("arith.index_cast", loc)
                        .add_operands(&[idx])
                        .add_results(&[i64_type])
                        .build()
                        .expect("arith.index_cast"),
                )
                .result(0)
                .unwrap()
                .into()
        })
        .collect();

    let n = dims.len() as u64;
    let tensor_type: melior::ir::Type =
        melior::ir::r#type::RankedTensorType::new(&[n], i64_type, None).into();

    let result: Value<'c, 'c> = builder
        .block()
        .append_operation(
            OperationBuilder::new("tensor.from_elements", loc)
                .add_operands(&i64_vals)
                .add_results(&[tensor_type])
                .build()
                .expect("tensor.from_elements"),
        )
        .result(0)
        .unwrap()
        .into();
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
    assert_eq!(
        shape.len(),
        1,
        "extract_dims_from_i64_tensor: expected 1-D tensor"
    );
    let n = match shape[0] {
        Some(n) => n as usize,
        None => panic!("extract_dims_from_i64_tensor: dynamic-length shape tensor not supported"),
    };
    let elem_type = tensor.dtype().to_mlir_type(ctx);

    (0..n)
        .map(|i| {
            let idx_attr = Attribute::parse(ctx, &format!("{i} : index")).expect("idx attr");
            let idx_val: Value<'c, 'c> = builder
                .block()
                .append_operation(
                    OperationBuilder::new("arith.constant", loc)
                        .add_results(&[index_type])
                        .add_attributes(&[(Identifier::new(ctx, "value"), idx_attr)])
                        .build()
                        .expect("arith.constant idx"),
                )
                .result(0)
                .unwrap()
                .into();

            let elem: Value<'c, 'c> = builder
                .block()
                .append_operation(
                    OperationBuilder::new("tensor.extract", loc)
                        .add_operands(&[tensor.value(), idx_val])
                        .add_results(&[elem_type])
                        .build()
                        .expect("tensor.extract"),
                )
                .result(0)
                .unwrap()
                .into();

            builder
                .block()
                .append_operation(
                    OperationBuilder::new("arith.index_cast", loc)
                        .add_operands(&[elem])
                        .add_results(&[index_type])
                        .build()
                        .expect("arith.index_cast"),
                )
                .result(0)
                .unwrap()
                .into()
        })
        .collect()
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
        builder
            .block()
            .append_operation(
                OperationBuilder::new("tensor.extract", loc)
                    .add_operands(&[tensor.value()])
                    .add_results(&[elem_type])
                    .build()
                    .expect("tensor.extract scalar"),
            )
            .result(0)
            .unwrap()
            .into()
    } else {
        let c0_attr = Attribute::parse(ctx, "0 : index").expect("0 index");
        let c0: Value<'c, 'c> = builder
            .block()
            .append_operation(
                OperationBuilder::new("arith.constant", loc)
                    .add_results(&[index_type])
                    .add_attributes(&[(Identifier::new(ctx, "value"), c0_attr)])
                    .build()
                    .expect("arith.constant 0"),
            )
            .result(0)
            .unwrap()
            .into();
        builder
            .block()
            .append_operation(
                OperationBuilder::new("tensor.extract", loc)
                    .add_operands(&[tensor.value(), c0])
                    .add_results(&[elem_type])
                    .build()
                    .expect("tensor.extract [0]"),
            )
            .result(0)
            .unwrap()
            .into()
    };

    builder
        .block()
        .append_operation(
            OperationBuilder::new("arith.index_cast", loc)
                .add_operands(&[scalar])
                .add_results(&[index_type])
                .build()
                .expect("arith.index_cast scalar"),
        )
        .result(0)
        .unwrap()
        .into()
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
        let shape: Vec<Option<u64>> = input
            .dims
            .iter()
            .map(|d| match d {
                Dim::Fixed(v) if *v == crate::shape::DIM_DYNAMIC => None,
                Dim::Fixed(v) => Some(*v),
                Dim::Symbolic(name) if keep_dynamic.contains(name) => None,
                Dim::Symbolic(_) => None, // unresolved — treat as dynamic
            })
            .collect();
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
        let live = collect_live_values(&mut value_map, &weight_names, &last_use, 0, builder);
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
            let live =
                collect_live_values(&mut value_map, &weight_names, &last_use, node_idx, builder);
            end_chunk(builder, &mut value_map, &live);
            let live =
                collect_live_values(&mut value_map, &weight_names, &last_use, node_idx, builder);
            begin_chunk(builder, &mut value_map, &live, chunk_index);
            chunk_index += 1;
            weight_remap.clear();
        }

        // Route weights into the current sub-function on demand.
        if builder.in_subfunction() {
            remap_node_weights(
                node,
                &weight_names,
                &weight_compute_tensors,
                &mut weight_remap,
                &mut value_map,
                builder,
            );
        }

        emit_node(node, &mut value_map, &mut const_i64, builder)?;
    }

    // Close the last sub-function if one is open.
    if builder.in_subfunction() {
        // Collect model output values as the final sub-function's returns.
        let mut final_live: Vec<(String, Tensor<'c>)> = Vec::new();
        for name in &model.outputs {
            let ev = value_map
                .get(name)
                .ok_or_else(|| OnnxError::MissingEdge(name.clone()))?;
            final_live.push((name.clone(), ev.as_tensor(builder)));
        }
        end_chunk(builder, &mut value_map, &final_live);
    }

    // ── 4. Collect outputs ───────────────────────────────────────────────────────
    let mut outputs: Vec<Tensor<'c>> = Vec::with_capacity(model.outputs.len());
    for name in &model.outputs {
        let ev = value_map
            .get(name)
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
            if weight_names.contains(*name) {
                return false;
            }
            if let Some(&lu) = last_use.get(*name) {
                if lu >= current_node_idx {
                    matches!(ev, EmitValue::ShapeDims(_))
                } else {
                    false
                }
            } else {
                false
            }
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
        if weight_names.contains(name) {
            continue;
        }
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
                &x,
                &w,
                bias.as_ref(),
                [
                    pads[0] as u64,
                    pads[1] as u64,
                    pads[2] as u64,
                    pads[3] as u64,
                ],
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
                [
                    pads[0] as u64,
                    pads[1] as u64,
                    pads[2] as u64,
                    pads[3] as u64,
                ],
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
                let shape_val = value_map
                    .get(shape_name)
                    .ok_or_else(|| OnnxError::MissingEdge(shape_name.clone()))?;
                let shape_tensor = shape_val.as_tensor(builder);

                let shape_tensor_shape = shape_tensor.shape();
                let out_rank = shape_tensor_shape
                    .first()
                    .and_then(|d| *d)
                    .unwrap_or_else(|| {
                        panic!(
                            "Reshape: shape tensor '{}' must have static rank-1 size",
                            shape_name
                        )
                    }) as usize;

                // Extract individual dim values as index values.
                let dim_indices = extract_dims_from_i64_tensor(builder, &shape_tensor);
                let allowzero = get_int_attr(&node.attributes, "allowzero", 0);

                // Resolve -1 and 0 entries, producing corrected index values.
                let corrected = builder.emit_resolve_reshape_dims(&x, &dim_indices, allowzero != 0);

                // Compute static output shape where possible.
                // If const_i64 propagation (e.g. through Concat) made the shape
                // tensor's values available, use them to resolve dims statically.
                // This prevents the all-dynamic cascade that breaks expand_shape.
                let shape_static = lookup_const_i64(const_i64, shape_name);
                let in_shape = x.shape();
                let allowzero_flag = allowzero != 0;

                let out_shape: Vec<Option<u64>> = (0..out_rank)
                    .map(|i| {
                        let raw_val: Option<i64> = shape_static
                            .as_ref()
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
                                let total_static: Option<u64> =
                                    in_shape.iter().try_fold(1u64, |acc, d| d.map(|n| acc * n));
                                let known_product: Option<u64> =
                                    shape_static.as_ref().and_then(|sv| {
                                        sv.iter().enumerate().try_fold(1u64, |acc, (j, &d)| {
                                            if j == i {
                                                Some(acc)
                                            }
                                            // skip the -1 dim
                                            else if d > 0 && d != CONST_I64_UNKNOWN {
                                                Some(acc * d as u64)
                                            } else if d == 0 && !allowzero_flag {
                                                in_shape.get(j).and_then(|opt| opt.map(|n| acc * n))
                                            } else {
                                                None
                                            }
                                        })
                                    });
                                match (total_static, known_product) {
                                    (Some(t), Some(k)) if k > 0 => Some(t / k),
                                    _ => None,
                                }
                            }
                            _ => None, // truly dynamic (or CONST_I64_UNKNOWN)
                        }
                    })
                    .collect();

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
            let axis = node
                .attributes
                .get("axis")
                .and_then(|a| {
                    if let OnnxAttribute::Int(v) = a {
                        Some(*v)
                    } else {
                        None
                    }
                })
                .ok_or_else(|| {
                    OnnxError::UnsupportedOp(
                        "Concat: missing required `axis` attribute".to_string(),
                    )
                })?;
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
                                if s.len() == 1 {
                                    s[0].unwrap_or(1) as usize
                                } else {
                                    1
                                }
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
            let ends_static = lookup_const_i64(const_i64, &node.inputs[2]);

            // axes and steps are always from initializers or attributes in ONNX.
            let axes: Vec<i64> = if node.inputs.len() > 3 && !node.inputs[3].is_empty() {
                lookup_const_i64(const_i64, &node.inputs[3]).ok_or_else(|| {
                    OnnxError::UnsupportedOp(format!(
                        "Slice: axes input '{}' must be a static initializer",
                        node.inputs[3]
                    ))
                })?
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
                    lookup_const_i64(const_i64, &node.inputs[4]).ok_or_else(|| {
                        OnnxError::UnsupportedOp(format!(
                            "Slice: steps input '{}' must be a static initializer",
                            node.inputs[4]
                        ))
                    })?
                } else {
                    vec![1i64; starts.len()]
                };
                let out = builder.emit_slice(&data, &starts, &ends, &axes, &steps);
                insert_tensor(value_map, &node.outputs[0], out);

                // Propagate const_i64 through static Slice on 1-D data.
                if let Some(data_vals) = lookup_const_i64(const_i64, &node.inputs[0]) {
                    if axes.len() == 1 && axes[0] == 0 && steps.iter().all(|&s| s == 1) {
                        let n = data_vals.len() as i64;
                        let s = if starts[0] < 0 {
                            (n + starts[0]).max(0)
                        } else {
                            starts[0].min(n)
                        } as usize;
                        let e = if ends[0] < 0 {
                            (n + ends[0]).max(0)
                        } else {
                            ends[0].min(n)
                        } as usize;
                        if s <= e && e <= data_vals.len() {
                            const_i64.insert(node.outputs[0].clone(), data_vals[s..e].to_vec());
                        }
                    }
                }
            } else {
                // ── Dynamic path ─────────────────────────────────────────────
                // starts/ends are runtime 1-D I64 tensors in value_map.
                let starts_t = get_tensor(value_map, builder, &node.inputs[1])?;
                let ends_t = get_tensor(value_map, builder, &node.inputs[2])?;

                // Number of sliced axes = length of the starts tensor.
                let n_axes = match starts_t.shape().first() {
                    Some(Some(n)) => *n as usize,
                    _ => {
                        return Err(OnnxError::UnsupportedOp(format!(
                            "Slice: dynamic starts tensor '{}' must have a static length",
                            node.inputs[1]
                        )));
                    }
                };

                let axes = if axes.is_empty() {
                    (0..n_axes as i64).collect::<Vec<_>>()
                } else {
                    axes
                };
                let steps: Vec<i64> = if node.inputs.len() > 4 && !node.inputs[4].is_empty() {
                    lookup_const_i64(const_i64, &node.inputs[4]).ok_or_else(|| {
                        OnnxError::UnsupportedOp(format!(
                            "Slice: steps input '{}' must be a static initializer",
                            node.inputs[4]
                        ))
                    })?
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
                    let gathered: Vec<i64> = idx_vals
                        .iter()
                        .map(|&idx| {
                            let i = if idx < 0 {
                                (data_vals.len() as i64 + idx) as usize
                            } else {
                                idx as usize
                            };
                            data_vals[i]
                        })
                        .collect();
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
            let target = onnx_dtype_to_internal(to).ok_or_else(|| {
                OnnxError::UnsupportedOp(format!("Cast: unsupported ONNX dtype {to}"))
            })?;
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
                lookup_const_i64(const_i64, &node.inputs[1]).ok_or_else(|| {
                    OnnxError::UnsupportedOp(format!(
                        "Squeeze: axes input '{}' must be a static initializer",
                        node.inputs[1]
                    ))
                })?
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
                    .ok_or_else(|| {
                        OnnxError::UnsupportedOp(format!(
                            "Split: split input '{}' must be a static initializer",
                            node.inputs[1]
                        ))
                    })?
                    .iter()
                    .map(|&v| v as u64)
                    .collect()
            } else if let Some(OnnxAttribute::Ints(v)) = node.attributes.get("split") {
                v.iter().map(|&v| v as u64).collect()
            } else {
                let ax_dim = input.shape()[axis_usize].ok_or_else(|| {
                    OnnxError::UnsupportedOp("Split: split axis dimension is dynamic".to_string())
                })?;
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
            let static_vals: Option<Vec<i64>> =
                in_shape.iter().map(|d| d.map(|n| n as i64)).collect();
            if let Some(vals) = static_vals {
                const_i64.insert(node.outputs[0].clone(), vals);
            }

            // Store as ShapeDims — downstream ops (Reshape, ConstantOfShape) can
            // materialize as a tensor or use the index values directly.
            value_map.insert(node.outputs[0].clone(), EmitValue::ShapeDims(dims));
        }
        "ConstantOfShape" => {
            let shape_name = &node.inputs[0];
            let (shape_vals, static_shape): (Vec<Value<'c, 'c>>, Vec<Option<u64>>) = match value_map
                .get(shape_name)
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

            let (fill_value, fill_dtype) =
                if let Some(OnnxAttribute::Tensor(val_buf)) = node.attributes.get("value") {
                    let fv = match val_buf.dtype() {
                        DType::F32 => val_buf.as_slice::<f32>()[0] as f64,
                        DType::F64 => val_buf.as_slice::<f64>()[0],
                        DType::BF16 => val_buf.as_slice::<u16>()[0] as f64,
                        DType::I32 => val_buf.as_slice::<i32>()[0] as f64,
                        DType::I64 => val_buf.as_slice::<i64>()[0] as f64,
                    };
                    (fv, val_buf.dtype())
                } else {
                    (0.0, DType::F32)
                };

            let out =
                builder.emit_constant_of_shape(&shape_vals, &static_shape, fill_value, fill_dtype);
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
            let buf = node
                .attributes
                .get("value")
                .and_then(|attr| {
                    if let OnnxAttribute::Tensor(b) = attr {
                        Some(b)
                    } else {
                        None
                    }
                })
                .ok_or_else(|| {
                    OnnxError::UnsupportedOp(
                        "Constant: missing or non-tensor `value` attribute".to_string(),
                    )
                })?;
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
        DType::BF16 => {
            // BF16 is stored as raw u16 bytes; emit as hex integers so MLIR can
            // parse the dense attribute as bf16 without floating-point conversion.
            let vals = buf.as_slice::<u16>();
            format_dense_nd(vals.iter().map(|&v| format!("{v}")), shape)
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

enum ReduceOp {
    Sum,
    Max,
}

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
    let mut norm: Vec<i64> = axes
        .iter()
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
    attrs
        .get(name)
        .and_then(|a| {
            if let OnnxAttribute::Int(v) = a {
                Some(*v)
            } else {
                None
            }
        })
        .unwrap_or(default)
}

fn get_float_attr(attrs: &HashMap<String, OnnxAttribute>, name: &str, default: f64) -> f64 {
    attrs
        .get(name)
        .and_then(|a| {
            if let OnnxAttribute::Float(v) = a {
                Some(*v as f64)
            } else {
                None
            }
        })
        .unwrap_or(default)
}

fn get_bool_attr(attrs: &HashMap<String, OnnxAttribute>, name: &str) -> bool {
    attrs
        .get(name)
        .and_then(|a| {
            if let OnnxAttribute::Int(v) = a {
                Some(*v != 0)
            } else {
                None
            }
        })
        .unwrap_or(false)
}

fn get_keepdims_attr(attrs: &HashMap<String, OnnxAttribute>, default: bool) -> bool {
    attrs
        .get("keepdims")
        .and_then(|a| {
            if let OnnxAttribute::Int(v) = a {
                Some(*v != 0)
            } else {
                None
            }
        })
        .unwrap_or(default)
}

fn get_str_attr<'a>(
    attrs: &'a HashMap<String, OnnxAttribute>,
    name: &str,
    default: &'a str,
) -> &'a str {
    attrs
        .get(name)
        .and_then(|a| {
            if let OnnxAttribute::String(v) = a {
                Some(v.as_str())
            } else {
                None
            }
        })
        .unwrap_or(default)
}

fn get_ints_attr(attrs: &HashMap<String, OnnxAttribute>, name: &str, default: &[i64]) -> Vec<i64> {
    attrs
        .get(name)
        .and_then(|a| {
            if let OnnxAttribute::Ints(v) = a {
                Some(v.clone())
            } else {
                None
            }
        })
        .unwrap_or_else(|| default.to_vec())
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
        lookup_const_i64(const_i64, &node.inputs[1]).ok_or_else(|| {
            OnnxError::UnsupportedOp(format!(
                "{op_name}: axes input '{}' must be a static initializer or Constant node",
                node.inputs[1]
            ))
        })
    } else if let Some(OnnxAttribute::Ints(v)) = node.attributes.get("axes") {
        Ok(v.clone())
    } else {
        Err(OnnxError::UnsupportedOp(format!(
            "{op_name}: no axes specified"
        )))
    }
}

fn get_reduce_axes(
    node: &OnnxNode,
    const_i64: &HashMap<String, Vec<i64>>,
    rank: usize,
) -> Result<Vec<i64>, OnnxError> {
    if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
        lookup_const_i64(const_i64, &node.inputs[1]).ok_or_else(|| {
            OnnxError::UnsupportedOp(format!(
                "{}: axes input '{}' must be a static initializer or Constant node",
                node.op_type, node.inputs[1]
            ))
        })
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
    let ev = value_map
        .get(name)
        .ok_or_else(|| OnnxError::MissingEdge(name.to_string()))?;
    Ok(ev.as_tensor(builder))
}

fn insert_tensor<'c>(value_map: &mut HashMap<String, EmitValue<'c>>, name: &str, t: Tensor<'c>) {
    value_map.insert(name.to_string(), EmitValue::Tensor(t));
}

// ── Per-kernel compilation ───────────────────────────────────────────────────

/// ONNX op types that expand dramatically during tiling/vectorization.
const HEAVY_OPS: &[&str] = &["Conv", "MatMul", "Gemm", "ConvTranspose"];

/// A group of ONNX nodes that will become one compiled kernel.
struct KernelGroup {
    node_indices: Vec<usize>,
}

/// Returns true if this node is a KV-cache Concat:
/// - op_type == "Concat"
/// - raw axis attribute == -2 (sequence dimension in [B, heads, seq, head_dim])
/// - exactly 2 inputs (past + new)
fn is_kv_concat(node: &OnnxNode) -> bool {
    if node.op_type != "Concat" || node.inputs.len() != 2 {
        return false;
    }
    let axis_raw = node
        .attributes
        .get("axis")
        .and_then(|a| {
            if let OnnxAttribute::Int(v) = a {
                Some(*v)
            } else {
                None
            }
        })
        .unwrap_or(0);
    axis_raw == -2
}

/// Split kernel groups containing KV Concat nodes into sub-groups.
///
/// For each group containing KV Concat(s), produces:
/// - pre_concat: all transitive inputs of the KV Concats within the group
/// - one single-node group per KV Concat (will become NativeOp::Concat steps)
/// - post_concat: all transitive consumers of KV Concat outputs within the group
///
/// Groups without KV Concats pass through unchanged.
///
/// Returns (new_groups, set of KV Concat node indices, ordered list of KV Concat node indices).
fn split_kv_concat_groups(
    nodes: &[OnnxNode],
    groups: Vec<KernelGroup>,
) -> (Vec<KernelGroup>, HashSet<usize>, Vec<usize>) {
    let mut result: Vec<KernelGroup> = Vec::new();
    let mut kv_concat_indices: HashSet<usize> = HashSet::new();
    let mut kv_concat_ordered: Vec<usize> = Vec::new();

    for group in groups {
        // Quick check: does this group contain any KV Concat?
        let kv_concats: Vec<usize> = group
            .node_indices
            .iter()
            .copied()
            .filter(|&ni| is_kv_concat(&nodes[ni]))
            .collect();

        if kv_concats.is_empty() {
            result.push(group);
            continue;
        }

        // Build intra-group DAG.
        // producer: value_name → node_index (only for nodes IN this group)
        let mut producer: HashMap<&str, usize> = HashMap::new();
        for &ni in &group.node_indices {
            for out in &nodes[ni].outputs {
                if !out.is_empty() {
                    producer.insert(out.as_str(), ni);
                }
            }
        }

        // dependencies: node → set of nodes that produce its inputs (within group)
        // consumers: node → set of nodes that consume its outputs (within group)
        let mut consumers: HashMap<usize, HashSet<usize>> = HashMap::new();
        let mut dependencies: HashMap<usize, HashSet<usize>> = HashMap::new();
        for &ni in &group.node_indices {
            for input_name in &nodes[ni].inputs {
                if let Some(&prod_ni) = producer.get(input_name.as_str()) {
                    if prod_ni != ni {
                        consumers.entry(prod_ni).or_default().insert(ni);
                        dependencies.entry(ni).or_default().insert(prod_ni);
                    }
                }
            }
        }

        let kv_concat_set: HashSet<usize> = kv_concats.iter().copied().collect();

        // Walk backwards from KV Concats to find all ancestors (pre-concat nodes).
        let mut pre_concat: HashSet<usize> = HashSet::new();
        {
            let mut stack: Vec<usize> = Vec::new();
            for &ci in &kv_concats {
                if let Some(deps) = dependencies.get(&ci) {
                    for &d in deps {
                        if !kv_concat_set.contains(&d) {
                            stack.push(d);
                        }
                    }
                }
            }
            while let Some(ni) = stack.pop() {
                if !pre_concat.insert(ni) {
                    continue;
                }
                if let Some(deps) = dependencies.get(&ni) {
                    for &d in deps {
                        if !kv_concat_set.contains(&d) && !pre_concat.contains(&d) {
                            stack.push(d);
                        }
                    }
                }
            }
        }

        // Walk forwards from KV Concats to find all descendants (post-concat nodes).
        let mut post_concat: HashSet<usize> = HashSet::new();
        {
            let mut stack: Vec<usize> = Vec::new();
            for &ci in &kv_concats {
                if let Some(cons) = consumers.get(&ci) {
                    for &c in cons {
                        if !kv_concat_set.contains(&c) {
                            stack.push(c);
                        }
                    }
                }
            }
            while let Some(ni) = stack.pop() {
                if !post_concat.insert(ni) {
                    continue;
                }
                if let Some(cons) = consumers.get(&ni) {
                    for &c in cons {
                        if !kv_concat_set.contains(&c) && !post_concat.contains(&c) {
                            stack.push(c);
                        }
                    }
                }
            }
        }

        // Nodes in both pre and post: keep in pre (value crosses boundary via buffer slots).
        post_concat.retain(|ni| !pre_concat.contains(ni));

        // Nodes in neither pre nor post (e.g., Constants not connected to Concat): place in pre.
        for &ni in &group.node_indices {
            if !kv_concat_set.contains(&ni)
                && !pre_concat.contains(&ni)
                && !post_concat.contains(&ni)
            {
                pre_concat.insert(ni);
            }
        }

        // Build sub-groups, maintaining original node order within each.
        let pre_nodes: Vec<usize> = group
            .node_indices
            .iter()
            .copied()
            .filter(|ni| pre_concat.contains(ni))
            .collect();
        let post_nodes: Vec<usize> = group
            .node_indices
            .iter()
            .copied()
            .filter(|ni| post_concat.contains(ni))
            .collect();

        // Emit sub-groups in order: pre, concat(s), post.
        if !pre_nodes.is_empty() {
            result.push(KernelGroup {
                node_indices: pre_nodes,
            });
        }
        for &ci in &kv_concats {
            kv_concat_indices.insert(ci);
            kv_concat_ordered.push(ci);
            result.push(KernelGroup {
                node_indices: vec![ci],
            });
        }
        if !post_nodes.is_empty() {
            result.push(KernelGroup {
                node_indices: post_nodes,
            });
        }
    }

    (result, kv_concat_indices, kv_concat_ordered)
}

/// Partition ONNX nodes into kernel groups: each heavy op becomes its own
/// kernel, lightweight ops between heavy ops are grouped together.
fn partition_nodes(nodes: &[OnnxNode]) -> Vec<KernelGroup> {
    let mut groups: Vec<KernelGroup> = Vec::new();
    let mut current_lightweight: Vec<usize> = Vec::new();

    for (i, node) in nodes.iter().enumerate() {
        if HEAVY_OPS.contains(&node.op_type.as_str()) {
            // Close current lightweight group if non-empty.
            if !current_lightweight.is_empty() {
                groups.push(KernelGroup {
                    node_indices: std::mem::take(&mut current_lightweight),
                });
            }
            // Heavy op is its own kernel.
            groups.push(KernelGroup {
                node_indices: vec![i],
            });
        } else {
            current_lightweight.push(i);
        }
    }

    // Close any remaining lightweight group.
    if !current_lightweight.is_empty() {
        groups.push(KernelGroup {
            node_indices: current_lightweight,
        });
    }

    groups
}

/// Info about a Gemm→Reshape→Add fusion detected in the graph.
struct GemmResidualFusion {
    /// Node index of the Reshape that expands Gemm output to 3D.
    reshape_idx: usize,
    /// Node index of the Add (residual connection).
    add_idx: usize,
    /// ONNX value name of the residual input to the Add (the non-Gemm input).
    residual_name: String,
    /// ONNX value name of the Add output (what downstream nodes consume).
    add_output_name: String,
}

/// Shape-prep op types that appear between Gemm output and Reshape.
const SHAPE_PREP_OPS: &[&str] = &[
    "Constant",
    "Unsqueeze",
    "Concat",
    "Gather",
    "Shape",
    "Slice",
    "Squeeze",
];

/// Detect `Gemm → [shape_prep...] → Reshape → Add(residual)` patterns.
///
/// Returns a map from Gemm group index → fusion info. When detected, the
/// shape_prep + Reshape + Add nodes are absorbed into the Gemm group.
fn detect_and_absorb_gemm_residual(
    nodes: &[OnnxNode],
    groups: &mut Vec<KernelGroup>,
) -> HashMap<usize, GemmResidualFusion> {
    let mut fusions: HashMap<usize, GemmResidualFusion> = HashMap::new();

    // Build: value_name → producing node index.
    let mut producer: HashMap<&str, usize> = HashMap::new();
    for (ni, node) in nodes.iter().enumerate() {
        for out in &node.outputs {
            producer.insert(out.as_str(), ni);
        }
    }

    let gi_count = groups.len();
    let mut gi = 0;
    while gi + 1 < gi_count {
        // Is this group a single Gemm with alpha=1, beta=1, rank-1 bias?
        if groups[gi].node_indices.len() != 1 {
            gi += 1;
            continue;
        }
        let gemm_ni = groups[gi].node_indices[0];
        let gemm_node = &nodes[gemm_ni];
        if gemm_node.op_type != "Gemm" {
            gi += 1;
            continue;
        }
        let alpha = get_float_attr(&gemm_node.attributes, "alpha", 1.0) as f32;
        let beta = get_float_attr(&gemm_node.attributes, "beta", 1.0) as f32;
        if (alpha - 1.0).abs() > f32::EPSILON || (beta - 1.0).abs() > f32::EPSILON {
            gi += 1;
            continue;
        }

        // Look at the next (lightweight) group for the Reshape→Add pattern.
        let next_gi = gi + 1;
        let next_nodes = &groups[next_gi].node_indices;

        // Walk the lightweight group to find: shape_prep... → Reshape(gemm_out) → Add.
        let gemm_out = &gemm_node.outputs[0];
        let mut reshape_idx = None;
        let mut add_idx = None;
        for &ni in next_nodes {
            let node = &nodes[ni];
            if SHAPE_PREP_OPS.contains(&node.op_type.as_str()) {
                // Shape prep — candidate for absorption.
                // Only absorb if it feeds the Reshape (checked indirectly below).
                continue;
            }
            if node.op_type == "Reshape" && reshape_idx.is_none() {
                // Check: does this Reshape consume the Gemm output?
                if node.inputs[0] == *gemm_out {
                    reshape_idx = Some(ni);
                    continue;
                }
            }
            if node.op_type == "Add" && reshape_idx.is_some() && add_idx.is_none() {
                let reshape_out = &nodes[reshape_idx.unwrap()].outputs[0];
                // One Add input must be the Reshape output.
                if node.inputs[0] == *reshape_out || node.inputs[1] == *reshape_out {
                    add_idx = Some(ni);
                    // Find the residual (the other input).
                    let residual = if node.inputs[0] == *reshape_out {
                        &node.inputs[1]
                    } else {
                        &node.inputs[0]
                    };
                    // The residual must NOT be produced within this lightweight group
                    // (it comes from outside — the actual residual connection).
                    let produced_in_group = next_nodes
                        .iter()
                        .any(|&nj| nodes[nj].outputs.contains(residual));
                    if produced_in_group {
                        add_idx = None; // Not a residual pattern
                    }
                }
                break; // Stop after first Add
            }
        }

        if let (Some(r_idx), Some(a_idx)) = (reshape_idx, add_idx) {
            let add_node = &nodes[a_idx];
            let reshape_out = &nodes[r_idx].outputs[0];
            let residual = if add_node.inputs[0] == *reshape_out {
                add_node.inputs[1].clone()
            } else {
                add_node.inputs[0].clone()
            };

            // Determine which shape_prep nodes to absorb: walk transitive
            // dependencies of the Reshape and Add within this lightweight group.
            let mut to_absorb: Vec<usize> = vec![r_idx, a_idx];
            // Iterate until no new nodes are found (transitive closure).
            loop {
                let mut changed = false;
                for &ni in next_nodes {
                    if to_absorb.contains(&ni) {
                        continue;
                    }
                    let node = &nodes[ni];
                    if !SHAPE_PREP_OPS.contains(&node.op_type.as_str()) {
                        continue;
                    }
                    // Check if any output feeds an already-absorbed node.
                    let feeds_absorbed = node
                        .outputs
                        .iter()
                        .any(|out| to_absorb.iter().any(|&nj| nodes[nj].inputs.contains(out)));
                    if feeds_absorbed {
                        to_absorb.push(ni);
                        changed = true;
                    }
                }
                if !changed {
                    break;
                }
            }

            let fusion = GemmResidualFusion {
                reshape_idx: r_idx,
                add_idx: a_idx,
                residual_name: residual,
                add_output_name: add_node.outputs[0].clone(),
            };

            // Absorb into Gemm group, maintaining original graph order.
            groups[gi].node_indices.extend_from_slice(&to_absorb);
            groups[gi].node_indices.sort_unstable();
            groups[next_gi]
                .node_indices
                .retain(|ni| !to_absorb.contains(ni));

            fusions.insert(gemm_ni, fusion);
        }

        gi += 1;
    }

    // Remove empty groups.
    groups.retain(|g| !g.node_indices.is_empty());

    fusions
}

/// I/O mapping for a single kernel: which ONNX value names are inputs/outputs
/// and which buffer pool slot they map to.
struct KernelIO {
    /// (ONNX value name, buffer slot index) — inputs to this kernel, in order.
    input_slots: Vec<(String, usize)>,
    /// (ONNX value name, buffer slot index) — outputs from this kernel, in order.
    output_slots: Vec<(String, usize)>,
}

/// Assign buffer pool slots and compute per-kernel I/O mappings.
///
/// Returns `(kernel_ios, num_slots, input_slots, weight_slots, output_slots, slot_value_names)`
/// where `slot_value_names[i]` is the ONNX value name for slot `i`.
fn assign_buffer_slots(
    model: &OnnxModel,
    groups: &[KernelGroup],
) -> (
    Vec<KernelIO>,
    usize,
    Vec<usize>,
    Vec<usize>,
    Vec<usize>,
    Vec<String>,
) {
    let mut name_to_slot: HashMap<String, usize> = HashMap::new();
    let mut slot_names: Vec<String> = Vec::new();
    let mut next_slot = 0usize;

    let alloc_slot = |name: &str,
                      name_to_slot: &mut HashMap<String, usize>,
                      slot_names: &mut Vec<String>,
                      next: &mut usize|
     -> usize {
        if let Some(&s) = name_to_slot.get(name) {
            s
        } else {
            let s = *next;
            *next += 1;
            name_to_slot.insert(name.to_string(), s);
            slot_names.push(name.to_string());
            s
        }
    };

    // Assign slots for model inputs (in order).
    let mut input_slots: Vec<usize> = Vec::new();
    for input in &model.dynamic_inputs {
        let s = alloc_slot(
            &input.name,
            &mut name_to_slot,
            &mut slot_names,
            &mut next_slot,
        );
        input_slots.push(s);
    }

    // Assign slots for weights (in order).
    let mut weight_slots: Vec<usize> = Vec::new();
    for (name, _) in &model.initializers {
        let s = alloc_slot(name, &mut name_to_slot, &mut slot_names, &mut next_slot);
        weight_slots.push(s);
    }

    // For each group, determine which values are produced inside vs consumed from outside.
    let mut produced_by: HashMap<String, usize> = HashMap::new(); // value name → group index
    for (gi, group) in groups.iter().enumerate() {
        for &ni in &group.node_indices {
            for out_name in &model.nodes[ni].outputs {
                if !out_name.is_empty() {
                    produced_by.insert(out_name.clone(), gi);
                }
            }
        }
    }

    // Build per-kernel I/O.
    let mut kernel_ios: Vec<KernelIO> = Vec::new();
    for group in groups.iter() {
        let mut produced_in_group: HashSet<String> = HashSet::new();
        for &ni in &group.node_indices {
            for out_name in &model.nodes[ni].outputs {
                if !out_name.is_empty() {
                    produced_in_group.insert(out_name.clone());
                }
            }
        }

        // Inputs: consumed by this group but produced outside (or model inputs/weights).
        let mut seen_inputs: HashSet<String> = HashSet::new();
        let mut io_input_slots: Vec<(String, usize)> = Vec::new();
        for &ni in &group.node_indices {
            for in_name in &model.nodes[ni].inputs {
                if in_name.is_empty() || produced_in_group.contains(in_name) {
                    continue;
                }
                if seen_inputs.insert(in_name.clone()) {
                    let s = alloc_slot(in_name, &mut name_to_slot, &mut slot_names, &mut next_slot);
                    io_input_slots.push((in_name.clone(), s));
                }
            }
        }

        // Outputs: produced by this group AND (consumed by a later group OR model output).
        let model_output_set: HashSet<&str> = model.outputs.iter().map(|s| s.as_str()).collect();
        let mut seen_outputs: HashSet<String> = HashSet::new();
        let mut io_output_slots: Vec<(String, usize)> = Vec::new();
        for &ni in &group.node_indices {
            for out_name in &model.nodes[ni].outputs {
                if out_name.is_empty() || !seen_outputs.insert(out_name.clone()) {
                    continue;
                }
                // Is this value used outside this group or is a model output?
                let used_outside = model
                    .nodes
                    .iter()
                    .enumerate()
                    .any(|(other_ni, other_node)| {
                        if group.node_indices.contains(&other_ni) {
                            return false;
                        }
                        other_node.inputs.contains(out_name)
                    })
                    || model_output_set.contains(out_name.as_str());

                if used_outside {
                    let s =
                        alloc_slot(out_name, &mut name_to_slot, &mut slot_names, &mut next_slot);
                    io_output_slots.push((out_name.clone(), s));
                }
            }
        }

        kernel_ios.push(KernelIO {
            input_slots: io_input_slots,
            output_slots: io_output_slots,
        });
    }

    // Model output slots (in order).
    let output_slots: Vec<usize> = model
        .outputs
        .iter()
        .map(|name| {
            *name_to_slot
                .get(name)
                .expect("model output not in slot map")
        })
        .collect();

    (
        kernel_ios,
        next_slot,
        input_slots,
        weight_slots,
        output_slots,
        slot_names,
    )
}

/// Shape + dtype info for a value, recorded after emission for use by downstream kernels.
struct ValueShapeInfo {
    shape: Vec<Option<u64>>,
    dtype: DType,
}

/// Emit and compile all kernels, producing an ExecutionPlan.
///
/// This is the per-kernel replacement for `emit_graph` + `compile_with_cache`.
/// Each kernel group gets its own `GraphContext` + `GraphBuilder`, so MLIR
/// passes and codegen run on small independent modules.
pub fn emit_and_compile_plan(
    model: &OnnxModel,
    inputs: &[&Buffer],
    model_bytes: Option<&[u8]>,
    keep_dynamic: &HashSet<String>,
) -> Result<(crate::runtime::ExecutionPlan, Vec<Buffer>), OnnxError> {
    use crate::Shape;
    use crate::cache::CompilationCache;
    use crate::graph_builder::GraphContext;
    use crate::runtime::{ExecutionPlan, KernelStep, SlotDesc};
    use crate::shape::DIM_DYNAMIC;

    let mut groups = partition_nodes(&model.nodes);
    let gemm_residual_fusions = detect_and_absorb_gemm_residual(&model.nodes, &mut groups);
    if !gemm_residual_fusions.is_empty() {
        eprintln!(
            "[{:>8.2}s] [plan] fused {} Gemm+residual Add pairs",
            crate::log_ts(),
            gemm_residual_fusions.len()
        );
    }
    let (groups, kv_concat_node_indices, kv_concat_ordered) =
        split_kv_concat_groups(&model.nodes, groups);
    if !kv_concat_node_indices.is_empty() {
        eprintln!(
            "[{:>8.2}s] [plan] split {} KV Concat nodes into native ops",
            crate::log_ts(),
            kv_concat_node_indices.len()
        );
    }

    let (kernel_ios, num_slots, input_slots, weight_slots, output_slots, slot_names) =
        assign_buffer_slots(model, &groups);

    // Build KV Concat metadata: map node_index → (layer, is_value).
    // KV Concats come in pairs (K then V) per layer, in model order.
    let mut kv_concat_meta: HashMap<usize, (usize, bool)> = HashMap::new();
    for (i, &ni) in kv_concat_ordered.iter().enumerate() {
        let layer = i / 2;
        let is_value = (i % 2) == 1;
        kv_concat_meta.insert(ni, (layer, is_value));
    }

    // Build model input name set for identifying "past" KV inputs.
    let model_input_names: HashSet<&str> = model
        .dynamic_inputs
        .iter()
        .map(|inp| inp.name.as_str())
        .collect();

    // Track past_kv input slots and present_kv output slots for KvPlanInfo.
    let mut past_kv_input_slots: Vec<usize> = Vec::new();
    let mut present_kv_output_slots: Vec<usize> = Vec::new();

    eprintln!(
        "[{:>8.2}s] [plan] {} nodes → {} kernels, {} buffer slots",
        crate::log_ts(),
        model.nodes.len(),
        groups.len(),
        num_slots
    );

    // Seed shape info from model inputs.
    let mut shape_info: HashMap<String, ValueShapeInfo> = HashMap::new();
    for (spec, buf) in model.dynamic_inputs.iter().zip(inputs.iter()) {
        let shape: Vec<Option<u64>> = spec
            .dims
            .iter()
            .enumerate()
            .map(|(i, d)| match d {
                Dim::Fixed(v) if *v == DIM_DYNAMIC => None,
                Dim::Fixed(v) => Some(*v),
                Dim::Symbolic(name) if keep_dynamic.contains(name) => None,
                Dim::Symbolic(_) => Some(buf.shape().0[i]),
            })
            .collect();
        shape_info.insert(
            spec.name.clone(),
            ValueShapeInfo {
                shape,
                dtype: spec.dtype,
            },
        );
    }

    // Seed shape info from weights.
    for (name, buf) in &model.initializers {
        shape_info.insert(
            name.clone(),
            ValueShapeInfo {
                shape: buf.shape().0.iter().map(|&d| Some(d)).collect(),
                dtype: buf.dtype(),
            },
        );
    }

    // Seed const_i64 from small integer initializers.
    let mut const_i64: HashMap<String, Vec<i64>> = HashMap::new();
    for (name, buf) in &model.initializers {
        if buf.shape().num_elements() <= 64 {
            if let Ok(vals) = read_i64_buffer(buf) {
                const_i64.insert(name.clone(), vals);
            }
        }
    }

    // Seed symbolic shape info from model inputs.
    // DIM_DYNAMIC (sentinel u64::MAX) is NOT stored in sym_shape_info — it would
    // overflow when propagated through symbolic arithmetic.
    let mut sym_shape_info: HashMap<String, crate::shape::SymShape> = HashMap::new();
    for (spec, buf) in model.dynamic_inputs.iter().zip(inputs.iter()) {
        let sym_shape: crate::shape::SymShape = spec
            .dims
            .iter()
            .enumerate()
            .map(|(i, d)| match d {
                Dim::Fixed(v) if *v == DIM_DYNAMIC => {
                    // Truly unknown — represent with a placeholder var so downstream
                    // propagation can still track the dimension symbolically.
                    crate::shape::SymDim::Var(format!("__dyn_{i}"))
                }
                Dim::Fixed(v) => crate::shape::SymDim::Concrete(*v),
                Dim::Symbolic(name) if keep_dynamic.contains(name) => {
                    crate::shape::SymDim::Var(name.clone())
                }
                Dim::Symbolic(_) => crate::shape::SymDim::Concrete(buf.shape().0[i]),
            })
            .collect();
        sym_shape_info.insert(spec.name.clone(), sym_shape);
    }

    // Seed symbolic shape info from initializers (all concrete).
    for (name, buf) in &model.initializers {
        let sym_shape: crate::shape::SymShape = buf
            .shape()
            .0
            .iter()
            .map(|&d| crate::shape::SymDim::Concrete(d))
            .collect();
        sym_shape_info.insert(name.clone(), sym_shape);
    }

    // Seed sym_const_i64 from small non-negative integer initializers (all concrete).
    // Negative values (sentinels like -1 or CONST_I64_UNKNOWN) are excluded to prevent
    // u64 overflow when these propagate through arithmetic.
    let mut sym_const_i64: HashMap<String, Vec<crate::shape::SymDim>> = HashMap::new();
    for (name, buf) in &model.initializers {
        if buf.shape().num_elements() <= 64 {
            if let Ok(vals) = read_i64_buffer(buf) {
                if vals.iter().all(|&v| v >= 0) {
                    sym_const_i64.insert(
                        name.clone(),
                        vals.iter()
                            .map(|&v| crate::shape::SymDim::Concrete(v as u64))
                            .collect(),
                    );
                }
            }
        }
    }

    // Collect weights in initializer order.
    let weights: Vec<Buffer> = model.initializers.iter().map(|(_, b)| b.clone()).collect();

    // Build input_shapes for cache key.
    let input_shapes: Vec<Shape> = model
        .dynamic_inputs
        .iter()
        .zip(inputs.iter())
        .map(|(spec, buf)| {
            let dims: Vec<u64> = spec
                .dims
                .iter()
                .enumerate()
                .map(|(i, d)| match d {
                    Dim::Fixed(v) if *v == DIM_DYNAMIC => DIM_DYNAMIC,
                    Dim::Fixed(v) => *v,
                    Dim::Symbolic(name) if keep_dynamic.contains(name) => DIM_DYNAMIC,
                    Dim::Symbolic(_) => buf.shape().0[i],
                })
                .collect();
            Shape(dims)
        })
        .collect();
    let model_cache_key = model_bytes.map(|b| {
        let shape_refs: Vec<&[u64]> = input_shapes.iter().map(|s| s.0.as_slice()).collect();
        CompilationCache::cache_key(b, &shape_refs)
    });

    // Phase 1: Emit all kernels sequentially (shape propagation requires order).
    // Each kernel produces an MLIR module text + metadata for deferred compilation.
    struct KernelEmitResult {
        mlir_text: String,
        num_inputs: usize,
        output_descs: Vec<crate::runtime::OutputDesc>,
        cache_key: Option<String>,
        group_idx: usize,
        num_in: usize,
        num_out: usize,
        ops_label: String,
    }

    let mut emit_results: Vec<KernelEmitResult> = Vec::new();
    let mut steps: Vec<KernelStep> = Vec::new();
    let mut next_kernel_idx: usize = 0;

    for (gi, (group, io)) in groups.iter().zip(kernel_ios.iter()).enumerate() {
        // Skip MLIR emission for native-op groups (KV Concat).
        if group.node_indices.len() == 1 && kv_concat_node_indices.contains(&group.node_indices[0])
        {
            let node = &model.nodes[group.node_indices[0]];
            let axis_raw = node
                .attributes
                .get("axis")
                .and_then(|a| {
                    if let OnnxAttribute::Int(v) = a {
                        Some(*v)
                    } else {
                        None
                    }
                })
                .unwrap_or(0);
            let rank = 4usize;
            let axis = if axis_raw < 0 {
                (rank as i64 + axis_raw) as usize
            } else {
                axis_raw as usize
            };
            // Still need to propagate shape info for the Concat output.
            // The output shape = input[0] shape with axis dim summed.
            if let (Some(in0_info), Some(in1_info)) = (
                shape_info.get(&node.inputs[0]),
                shape_info.get(&node.inputs[1]),
            ) {
                let mut out_shape = in0_info.shape.clone();
                match (in0_info.shape[axis], in1_info.shape[axis]) {
                    (Some(a), Some(b)) => out_shape[axis] = Some(a + b),
                    _ => out_shape[axis] = None, // dynamic
                }
                shape_info.insert(
                    node.outputs[0].clone(),
                    ValueShapeInfo {
                        shape: out_shape,
                        dtype: in0_info.dtype,
                    },
                );
            }
            // Propagate sym shapes.
            if let Some(sym0) = sym_shape_info.get(&node.inputs[0]) {
                if let Some(sym1) = sym_shape_info.get(&node.inputs[1]) {
                    let mut out_sym = sym0.clone();
                    if axis < out_sym.len() && axis < sym1.len() {
                        out_sym[axis] = sym0[axis].clone().add(sym1[axis].clone());
                    }
                    sym_shape_info.insert(node.outputs[0].clone(), out_sym);
                }
            }
            let ni = group.node_indices[0];
            let all_input_slots: Vec<usize> = io.input_slots.iter().map(|(_, s)| *s).collect();
            let all_output_slots: Vec<usize> = io.output_slots.iter().map(|(_, s)| *s).collect();

            if let Some(&(layer, is_value)) = kv_concat_meta.get(&ni) {
                // Track which input is "past" (model input) and which is "new".
                for (name, slot) in &io.input_slots {
                    if model_input_names.contains(name.as_str()) {
                        past_kv_input_slots.push(*slot);
                    }
                }
                for (_, slot) in &io.output_slots {
                    present_kv_output_slots.push(*slot);
                }

                // KvAppend keeps both inputs so run() can fall back to Concat.
                // run_kv() knows which input is "new" (the non-past one).
                steps.push(KernelStep {
                    kernel_idx: usize::MAX,
                    input_slots: all_input_slots,
                    output_slots: all_output_slots,
                    native_op: Some(crate::runtime::NativeOp::KvAppend { layer, is_value }),
                });
            } else {
                steps.push(KernelStep {
                    kernel_idx: usize::MAX,
                    input_slots: all_input_slots,
                    output_slots: all_output_slots,
                    native_op: Some(crate::runtime::NativeOp::Concat { axis }),
                });
            }
            continue;
        }

        let ctx = GraphContext::new();
        let mut builder = ctx.builder();
        let mut local_value_map: HashMap<String, EmitValue> = HashMap::new();

        // Add inputs to this kernel's builder.
        for (name, _slot) in &io.input_slots {
            let info = shape_info
                .get(name)
                .unwrap_or_else(|| panic!("kernel {gi}: no shape info for input '{name}'"));
            let t = builder.input(&info.shape, info.dtype);
            local_value_map.insert(name.clone(), EmitValue::Tensor(t));
        }

        // Emit all nodes in this kernel group.
        // For fused Gemm+residual groups, emit the fused path.
        for &ni in &group.node_indices {
            let node = &model.nodes[ni];

            // In a fused group, replace the Gemm emission with the fused version
            // and skip the Add node (its output is produced by the fused Gemm).
            if let Some(fusion) = gemm_residual_fusions.get(&group.node_indices[0]) {
                if ni == fusion.add_idx {
                    // Skip — the Add output was already produced by the fused Gemm.
                    // But we still need to propagate sym shapes (handled below).
                    // Alias the Add output to the Reshape output.
                    let reshape_out = &model.nodes[fusion.reshape_idx].outputs[0];
                    let t = get_tensor(&mut local_value_map, &mut builder, reshape_out)?;
                    insert_tensor(&mut local_value_map, &fusion.add_output_name, t);
                    // Fall through to sym shape propagation below.
                } else if node.op_type == "Gemm" && group.node_indices[0] == ni {
                    // Fused Gemm: emit_gemm_with_residual.
                    let a = get_tensor(&mut local_value_map, &mut builder, &node.inputs[0])?;
                    let b = get_tensor(&mut local_value_map, &mut builder, &node.inputs[1])?;
                    let bias = if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
                        get_tensor(&mut local_value_map, &mut builder, &node.inputs[2])?
                    } else {
                        builder.emit_arith_constant(0.0, a.dtype())
                    };
                    // The residual is 3D [B,S,N] — collapse to 2D [M,N] for the matmul.
                    let residual_3d =
                        get_tensor(&mut local_value_map, &mut builder, &fusion.residual_name)?;
                    let residual_2d = if residual_3d.rank() == 3 {
                        builder.emit_flatten_leading(&residual_3d)
                    } else {
                        residual_3d
                    };
                    let trans_a = get_bool_attr(&node.attributes, "transA");
                    let trans_b = get_bool_attr(&node.attributes, "transB");
                    let out = builder.emit_gemm_with_residual(
                        &a,
                        &b,
                        &bias,
                        &residual_2d,
                        trans_a,
                        trans_b,
                    );
                    insert_tensor(&mut local_value_map, &node.outputs[0], out);
                    // Fall through to sym shape propagation.
                } else {
                    // Non-Gemm, non-Add node in the fused group (shape prep, Reshape).
                    emit_node(node, &mut local_value_map, &mut const_i64, &mut builder)?;
                }
            } else {
                emit_node(node, &mut local_value_map, &mut const_i64, &mut builder)?;
            }

            // Propagate symbolic shapes for this node.
            // Use a default empty shape for inputs not in sym_shape_info to preserve
            // positional alignment (e.g., Gemm needs input[0]=data, input[1]=weight).
            let empty_sym: crate::shape::SymShape = vec![];
            let input_sym_refs: Vec<&crate::shape::SymShape> = node
                .inputs
                .iter()
                .map(|name| sym_shape_info.get(name.as_str()).unwrap_or(&empty_sym))
                .collect();
            let (out_sym_shapes, sym_updates) =
                sym_shapes::propagate_sym_shapes(node, &input_sym_refs, &sym_const_i64, &const_i64);
            for (name, val) in sym_updates {
                sym_const_i64.insert(name, val);
            }
            for (out_name, sym_shape) in node.outputs.iter().zip(out_sym_shapes) {
                if !out_name.is_empty() {
                    sym_shape_info.insert(out_name.clone(), sym_shape);
                }
            }
        }

        // Record output shapes for downstream kernels.
        for (name, _slot) in &io.output_slots {
            if let Some(ev) = local_value_map.get(name) {
                let t = ev.as_tensor(&mut builder);
                shape_info.insert(
                    name.clone(),
                    ValueShapeInfo {
                        shape: t.shape(),
                        dtype: t.dtype(),
                    },
                );
            }
        }

        // Collect output tensors and finalize the module (without compiling).
        let output_tensors: Vec<crate::graph_builder::Tensor> = io
            .output_slots
            .iter()
            .map(|(name, _)| {
                local_value_map
                    .get(name)
                    .unwrap_or_else(|| panic!("kernel {gi}: output '{name}' not in value_map"))
                    .as_tensor(&mut builder)
            })
            .collect();
        let output_refs: Vec<&crate::graph_builder::Tensor> = output_tensors.iter().collect();

        // Choose the transform schedule mode for this kernel.
        //
        // For matmul/gemm kernels with a dynamic M dimension (e.g. M=1 during
        // autoregressive decode), use VectorizeOnly: vectorize without tiling
        // or OpenMP. The OpenMP fork/join + 32×32 cache-tiling overhead on a
        // single-row vector-matrix multiply is catastrophic (measured ~100×
        // slower than memory bandwidth allows).
        //
        // For MatMul [.., M, K] × [K, N]: check all dims except the last (K)
        // of the first input. If any is dynamic → VectorizeOnly, else Full.
        // Find the primary heavy op (Gemm/MatMul) in this group.
        let heavy_node = group
            .node_indices
            .iter()
            .find(|&&ni| matches!(model.nodes[ni].op_type.as_str(), "Gemm" | "MatMul"));
        let transform_mode = if let Some(&heavy_ni) = heavy_node {
            let node = &model.nodes[heavy_ni];
            let is_matmul_like = true;
            if is_matmul_like && !node.inputs.is_empty() {
                let first_input = &node.inputs[0];
                let has_dynamic_m = shape_info
                    .get(first_input)
                    .map(|info| {
                        let m_dims = &info.shape[..info.shape.len().saturating_sub(1)];
                        m_dims.iter().any(|d| d.is_none())
                    })
                    .unwrap_or(false);
                if has_dynamic_m {
                    // Gemm kernels benefit from OpenMP parallelism on N.
                    // Small attention MatMuls (dynamic N, tiny at runtime)
                    // stay on VectorizeOnly to avoid fork/join overhead.
                    let is_gemm = node.op_type.as_str() == "Gemm";
                    let has_large_static_n = node
                        .inputs
                        .get(1)
                        .and_then(|inp| {
                            shape_info
                                .get(inp)
                                .and_then(|info| info.shape.last().copied().flatten())
                        })
                        .map_or(false, |n| n > 256);
                    if is_gemm || has_large_static_n {
                        crate::graph_builder::TransformMode::TileParallel
                    } else {
                        crate::graph_builder::TransformMode::VectorizeOnly
                    }
                } else {
                    crate::graph_builder::TransformMode::Full
                }
            } else {
                crate::graph_builder::TransformMode::Full
            }
        } else {
            crate::graph_builder::TransformMode::Full
        };

        // Finalize module to MLIR text for deferred parallel compilation.
        // Use a unique function name per kernel so all .o files can be linked
        // into a single .so without symbol collisions.
        let func_name = format!("k{gi}");
        let (mlir_text, num_inputs, output_descs) = builder
            .finalize_to_mlir_named(&output_refs, transform_mode, &func_name)
            .map_err(|e| OnnxError::CompileError(format!("kernel {gi}: {e}")))?;

        let cache_key = model_cache_key.as_ref().map(|k| format!("pk_{k}_{gi}"));

        // Build a short label for the kernel's op types.
        let ops_label = if group.node_indices.len() == 1 {
            model.nodes[group.node_indices[0]].op_type.clone()
        } else {
            let types: Vec<&str> = group
                .node_indices
                .iter()
                .map(|&ni| model.nodes[ni].op_type.as_str())
                .collect();
            // Deduplicate for compact display.
            let mut unique: Vec<&str> = Vec::new();
            for t in &types {
                if !unique.contains(t) {
                    unique.push(t);
                }
            }
            if unique.len() <= 3 {
                unique.join("+")
            } else {
                format!("{}+...({})", unique[..2].join("+"), types.len())
            }
        };

        emit_results.push(KernelEmitResult {
            mlir_text,
            num_inputs,
            output_descs,
            cache_key,
            group_idx: gi,
            num_in: io.input_slots.len(),
            num_out: io.output_slots.len(),
            ops_label,
        });

        steps.push(KernelStep {
            kernel_idx: next_kernel_idx,
            input_slots: io.input_slots.iter().map(|(_, s)| *s).collect(),
            output_slots: io.output_slots.iter().map(|(_, s)| *s).collect(),
            native_op: None,
        });
        next_kernel_idx += 1;
    }

    assert_eq!(
        emit_results.len(),
        next_kernel_idx,
        "compiled kernel count mismatch"
    );

    // Phase 2: Compile all kernels in parallel.
    eprintln!(
        "[{:>8.2}s] [plan] compiling {} kernels in parallel...",
        crate::log_ts(),
        emit_results.len()
    );
    let compile_start = std::time::Instant::now();

    // Phase 2a: Compile each kernel to .o files in parallel (no linking yet).
    let tmp_dir = crate::graph_builder::tempfile_dir()
        .ok_or_else(|| OnnxError::CompileError("cannot determine temp directory".into()))?;

    let all_obj_paths: Vec<(usize, Vec<std::path::PathBuf>)> = {
        use rayon::prelude::*;
        let results: Vec<Result<(usize, Vec<std::path::PathBuf>), OnnxError>> = emit_results
            .par_iter()
            .map(|er| {
                let t0 = std::time::Instant::now();
                let func_name = format!("k{}", er.group_idx);
                let label = format!("k{}:{}", er.group_idx, er.ops_label);
                let obj_paths = crate::graph_builder::compile_to_objects(
                    &er.mlir_text,
                    &label,
                    &func_name,
                    &tmp_dir,
                )
                .map_err(|e| OnnxError::CompileError(format!("kernel {}: {e}", er.group_idx)))?;
                eprintln!(
                    "[{:>8.2}s] [plan] k{} [{}] ({} in / {} out): {}ms",
                    crate::log_ts(),
                    er.group_idx,
                    er.ops_label,
                    er.num_in,
                    er.num_out,
                    t0.elapsed().as_millis()
                );
                Ok((er.group_idx, obj_paths))
            })
            .collect();
        results.into_iter().collect::<Result<Vec<_>, _>>()?
    };

    // Phase 2b: Link all .o files into a single .so.
    let link_start = std::time::Instant::now();
    let all_objs: Vec<std::path::PathBuf> = all_obj_paths
        .iter()
        .flat_map(|(_, paths)| paths.iter().cloned())
        .collect();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    let unified_so = tmp_dir.join(format!(
        "homura_unified_{}_{:08x}.so",
        std::process::id(),
        nanos
    ));
    crate::compiler::link_shared_lib_pub(&all_objs, &unified_so)
        .map_err(|e| OnnxError::CompileError(format!("unified link: {e}")))?;
    eprintln!(
        "[{:>8.2}s] [plan] unified link ({} .o files): {}ms",
        crate::log_ts(),
        all_objs.len(),
        link_start.elapsed().as_millis()
    );

    // Clean up .o files.
    for p in &all_objs {
        std::fs::remove_file(p).ok();
    }

    // Phase 2c: dlopen once, dlsym each kernel.
    let lib = {
        use std::ffi::CString;
        let path_cstr = CString::new(unified_so.to_str().unwrap()).unwrap();
        let lib = unsafe { libc::dlopen(path_cstr.as_ptr(), libc::RTLD_NOW) };
        if lib.is_null() {
            let err = unsafe {
                let msg = libc::dlerror();
                if msg.is_null() {
                    "unknown dlopen error".to_string()
                } else {
                    std::ffi::CStr::from_ptr(msg).to_string_lossy().into_owned()
                }
            };
            return Err(OnnxError::CompileError(format!(
                "dlopen unified .so failed: {err}"
            )));
        }
        lib
    };

    let kernels: Vec<crate::runtime::CompiledGraph> = emit_results
        .iter()
        .map(|er| {
            let func_name = format!("k{}", er.group_idx);
            crate::runtime::CompiledGraph::load_from_handle(
                lib,
                er.num_inputs,
                er.output_descs.clone(),
                &func_name,
            )
            .map_err(|e| OnnxError::CompileError(format!("kernel {}: {e}", er.group_idx)))
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Clean up the .so file (it's already dlopen'd).
    std::fs::remove_file(&unified_so).ok();

    eprintln!(
        "[{:>8.2}s] [plan] all {} kernels compiled + linked: {}ms total",
        crate::log_ts(),
        kernels.len(),
        compile_start.elapsed().as_millis()
    );

    // Build slot descriptors.
    let slot_descs: Vec<SlotDesc> = slot_names
        .iter()
        .map(|name| {
            let info = shape_info
                .get(name)
                .unwrap_or_else(|| panic!("no shape info for slot '{name}'"));
            let shape = Shape(
                info.shape
                    .iter()
                    .map(|d| d.unwrap_or(DIM_DYNAMIC))
                    .collect(),
            );
            let sym_shape = sym_shape_info.get(name).cloned();
            SlotDesc {
                shape,
                dtype: info.dtype,
                sym_shape,
            }
        })
        .collect();

    let mut plan = ExecutionPlan::new(
        kernels,
        steps,
        num_slots,
        input_slots,
        weight_slots,
        output_slots,
        slot_descs,
    );
    plan.set_shared_lib(lib);

    // Attach KV cache info if KV Concats were detected.
    if !kv_concat_ordered.is_empty() {
        assert_eq!(
            kv_concat_ordered.len() % 2,
            0,
            "expected even number of KV Concat nodes (K+V pairs)"
        );
        let num_layers = kv_concat_ordered.len() / 2;

        // Extract num_heads and head_dim from the first KV Concat's new input shape.
        let first_node = &model.nodes[kv_concat_ordered[0]];
        let new_input_name = if model_input_names.contains(first_node.inputs[0].as_str()) {
            &first_node.inputs[1]
        } else {
            &first_node.inputs[0]
        };
        let new_info = shape_info
            .get(new_input_name)
            .expect("no shape info for KV Concat new input");
        // Shape: [1, num_heads, 1, head_dim]
        let num_heads = new_info.shape[1].expect("KV num_heads must be static") as usize;
        let head_dim = new_info.shape[3].expect("KV head_dim must be static") as usize;

        plan.set_kv_info(crate::runtime::KvPlanInfo {
            num_layers,
            num_heads,
            head_dim,
            past_kv_input_slots,
            present_kv_output_slots,
        });
        eprintln!(
            "[{:>8.2}s] [plan] KV cache: {} layers, {} heads, head_dim={}",
            crate::log_ts(),
            num_layers,
            num_heads,
            head_dim
        );
    }

    Ok((plan, weights))
}
