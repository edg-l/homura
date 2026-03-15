use std::collections::HashMap;

use crate::{
    Tensor,
    op::NodeId,
    runtime::Buffer,
    trace::{Trace, begin_trace, take_trace},
};

use super::parser::{OnnxAttribute, OnnxError, OnnxModel, OnnxNode};

/// Lower an `OnnxModel` into a `Trace` plus the ordered weight buffers.
///
/// Returns `(trace, output_ids, weights)` where:
/// - `trace` contains all recorded ops (inputs then computation).
/// - `output_ids` are the `NodeId`s of the model's outputs, in graph output order.
/// - `weights` are the initializer `Buffer`s in initializer order. At runtime,
///   callers should pass dynamic inputs first (in `model.dynamic_inputs` order)
///   followed by weights (in this vec's order).
pub fn map_graph(model: &OnnxModel) -> Result<(Trace, Vec<NodeId>, Vec<Buffer>), OnnxError> {
    begin_trace();
    match map_graph_inner(model) {
        Ok(result) => Ok(result),
        Err(e) => {
            // Ensure the trace is cleaned up even when an error occurs, so that
            // subsequent calls do not see a stale active trace.
            let _ = take_trace();
            Err(e)
        }
    }
}

fn map_graph_inner(model: &OnnxModel) -> Result<(Trace, Vec<NodeId>, Vec<Buffer>), OnnxError> {
    let mut tensors: HashMap<String, Tensor> = HashMap::new();

    // 1. Dynamic inputs first, in graph.input order.
    for input in &model.dynamic_inputs {
        let t = Tensor::new(&input.shape.0, input.dtype);
        tensors.insert(input.name.clone(), t);
    }

    // 2. Weights (initializers) in initializer order.
    let mut initializer_data: HashMap<String, Buffer> = HashMap::new();
    let mut weights: Vec<Buffer> = Vec::with_capacity(model.initializers.len());
    for (name, buffer) in &model.initializers {
        let t = Tensor::new(&buffer.shape().0, buffer.dtype());
        tensors.insert(name.clone(), t);
        initializer_data.insert(name.clone(), buffer.clone());
        weights.push(buffer.clone());
    }

    // 3. Walk nodes in topological order (guaranteed by ONNX spec).
    for node in &model.nodes {
        map_node(node, &mut tensors, &initializer_data)?;
    }

    // 4. Collect output NodeIds.
    let mut output_ids: Vec<NodeId> = Vec::with_capacity(model.outputs.len());
    for name in &model.outputs {
        let id = tensors
            .get(name)
            .ok_or_else(|| OnnxError::MissingEdge(name.clone()))?
            .id();
        output_ids.push(id);
    }

    let trace = take_trace();
    Ok((trace, output_ids, weights))
}

// ── Node dispatch ─────────────────────────────────────────────────────────────

fn map_node(
    node: &OnnxNode,
    tensors: &mut HashMap<String, Tensor>,
    initializer_data: &HashMap<String, Buffer>,
) -> Result<(), OnnxError> {
    match node.op_type.as_str() {
        "Add" => {
            let a = get_tensor(tensors, &node.inputs[0])?;
            let b = get_tensor(tensors, &node.inputs[1])?;
            let result = &a + &b;
            tensors.insert(node.outputs[0].clone(), result);
        }
        "Sub" => {
            let a = get_tensor(tensors, &node.inputs[0])?;
            let b = get_tensor(tensors, &node.inputs[1])?;
            let result = &a - &b;
            tensors.insert(node.outputs[0].clone(), result);
        }
        "Mul" => {
            let a = get_tensor(tensors, &node.inputs[0])?;
            let b = get_tensor(tensors, &node.inputs[1])?;
            let result = &a * &b;
            tensors.insert(node.outputs[0].clone(), result);
        }
        "Div" => {
            let a = get_tensor(tensors, &node.inputs[0])?;
            let b = get_tensor(tensors, &node.inputs[1])?;
            let result = &a / &b;
            tensors.insert(node.outputs[0].clone(), result);
        }
        "Neg" => {
            let a = get_tensor(tensors, &node.inputs[0])?;
            let result = -&a;
            tensors.insert(node.outputs[0].clone(), result);
        }
        "Relu" => {
            let a = get_tensor(tensors, &node.inputs[0])?;
            let result = a.relu();
            tensors.insert(node.outputs[0].clone(), result);
        }
        "Exp" => {
            let a = get_tensor(tensors, &node.inputs[0])?;
            let result = a.exp();
            tensors.insert(node.outputs[0].clone(), result);
        }
        "Tanh" => {
            let a = get_tensor(tensors, &node.inputs[0])?;
            let result = a.tanh();
            tensors.insert(node.outputs[0].clone(), result);
        }
        "Softmax" => {
            let a = get_tensor(tensors, &node.inputs[0])?;
            // ONNX Softmax `axis` defaults to -1 in opset 13+.
            let axis = node
                .attributes
                .get("axis")
                .and_then(|attr| {
                    if let OnnxAttribute::Int(v) = attr {
                        Some(*v as i32)
                    } else {
                        None
                    }
                })
                .unwrap_or(-1);
            let result = a.softmax(axis);
            tensors.insert(node.outputs[0].clone(), result);
        }
        "MatMul" => {
            let a = get_tensor(tensors, &node.inputs[0])?;
            let b = get_tensor(tensors, &node.inputs[1])?;
            let result = a.matmul(&b);
            tensors.insert(node.outputs[0].clone(), result);
        }
        "Gemm" => {
            let a = get_tensor(tensors, &node.inputs[0])?;
            let b = get_tensor(tensors, &node.inputs[1])?;
            let bias = if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
                Some(get_tensor(tensors, &node.inputs[2])?)
            } else {
                None
            };
            let alpha = node
                .attributes
                .get("alpha")
                .and_then(|attr| {
                    if let OnnxAttribute::Float(v) = attr {
                        Some(*v as f64)
                    } else {
                        None
                    }
                })
                .unwrap_or(1.0);
            let beta = node
                .attributes
                .get("beta")
                .and_then(|attr| {
                    if let OnnxAttribute::Float(v) = attr {
                        Some(*v as f64)
                    } else {
                        None
                    }
                })
                .unwrap_or(1.0);
            let trans_a = node
                .attributes
                .get("transA")
                .and_then(|attr| {
                    if let OnnxAttribute::Int(v) = attr {
                        Some(*v != 0)
                    } else {
                        None
                    }
                })
                .unwrap_or(false);
            let trans_b = node
                .attributes
                .get("transB")
                .and_then(|attr| {
                    if let OnnxAttribute::Int(v) = attr {
                        Some(*v != 0)
                    } else {
                        None
                    }
                })
                .unwrap_or(false);
            let result = a.gemm(&b, bias.as_ref(), alpha, beta, trans_a, trans_b);
            tensors.insert(node.outputs[0].clone(), result);
        }
        "Conv" => {
            let x = get_tensor(tensors, &node.inputs[0])?;
            let w = get_tensor(tensors, &node.inputs[1])?;
            let bias = if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
                Some(get_tensor(tensors, &node.inputs[2])?)
            } else {
                None
            };
            let strides = get_ints_attr(&node.attributes, "strides", &[1, 1]);
            let pads = get_ints_attr(&node.attributes, "pads", &[0, 0, 0, 0]);
            let dilations = get_ints_attr(&node.attributes, "dilations", &[1, 1]);
            let result = x.conv2d(
                &w,
                bias.as_ref(),
                [strides[0] as u64, strides[1] as u64],
                [
                    pads[0] as u64,
                    pads[1] as u64,
                    pads[2] as u64,
                    pads[3] as u64,
                ],
                [dilations[0] as u64, dilations[1] as u64],
            );
            tensors.insert(node.outputs[0].clone(), result);
        }
        "MaxPool" => {
            let x = get_tensor(tensors, &node.inputs[0])?;
            let kernel_shape = get_ints_attr(&node.attributes, "kernel_shape", &[]);
            if kernel_shape.len() != 2 {
                return Err(OnnxError::UnsupportedOp(
                    "MaxPool: kernel_shape must have exactly 2 elements".to_string(),
                ));
            }
            let strides = get_ints_attr(&node.attributes, "strides", &[1, 1]);
            let pads = get_ints_attr(&node.attributes, "pads", &[0, 0, 0, 0]);
            let result = x.max_pool2d(
                [kernel_shape[0] as u64, kernel_shape[1] as u64],
                [strides[0] as u64, strides[1] as u64],
                [
                    pads[0] as u64,
                    pads[1] as u64,
                    pads[2] as u64,
                    pads[3] as u64,
                ],
            );
            tensors.insert(node.outputs[0].clone(), result);
        }
        "Reshape" => {
            let x = get_tensor(tensors, &node.inputs[0])?;
            // ONNX Reshape takes the target shape as a second input (typically an initializer).
            let shape_name = &node.inputs[1];
            let shape_buf = initializer_data.get(shape_name).ok_or_else(|| {
                OnnxError::UnsupportedOp(format!(
                    "Reshape: shape input '{shape_name}' must be a static initializer"
                ))
            })?;
            let target_shape: Vec<i64> = shape_buf.as_slice::<i64>().to_vec();
            let result = x.reshape(&target_shape);
            tensors.insert(node.outputs[0].clone(), result);
        }
        "Flatten" => {
            let x = get_tensor(tensors, &node.inputs[0])?;
            let axis = node
                .attributes
                .get("axis")
                .and_then(|attr| {
                    if let OnnxAttribute::Int(v) = attr {
                        Some(*v)
                    } else {
                        None
                    }
                })
                .unwrap_or(1) as usize;
            // Flatten: merge dims [0..axis) into one, [axis..) into another.
            let shape = &x.shape().0;
            let dim0: u64 = shape[..axis].iter().product();
            let dim1: u64 = shape[axis..].iter().product();
            let result = x.reshape(&[dim0 as i64, dim1 as i64]);
            tensors.insert(node.outputs[0].clone(), result);
        }
        "Clip" => {
            // In opset 11+, min/max are optional inputs (not attributes).
            // Map Clip(x) with exactly 1 input → relu (x clipped to [0, ∞)).
            // Clip with min/max inputs cannot be safely mapped to relu without
            // inspecting the constant values, so reject it for now.
            if node.inputs.len() > 1 && node.inputs[1..].iter().any(|s| !s.is_empty()) {
                return Err(OnnxError::UnsupportedOp(
                    "Clip with non-zero bounds (min/max inputs present)".to_string(),
                ));
            }
            let a = get_tensor(tensors, &node.inputs[0])?;
            let result = a.relu();
            tensors.insert(node.outputs[0].clone(), result);
        }
        other => {
            return Err(OnnxError::UnsupportedOp(other.to_string()));
        }
    }
    Ok(())
}

/// Extract an `Ints` attribute, returning `default` if missing.
fn get_ints_attr(attrs: &HashMap<String, OnnxAttribute>, name: &str, default: &[i64]) -> Vec<i64> {
    attrs
        .get(name)
        .and_then(|attr| {
            if let OnnxAttribute::Ints(v) = attr {
                Some(v.clone())
            } else {
                None
            }
        })
        .unwrap_or_else(|| default.to_vec())
}

/// Clone a `Tensor` handle out of the map by edge name.
///
/// Cloning a `Tensor` is cheap (NodeId + Shape + DType — no data).
fn get_tensor(tensors: &HashMap<String, Tensor>, name: &str) -> Result<Tensor, OnnxError> {
    tensors
        .get(name)
        .cloned()
        .ok_or_else(|| OnnxError::MissingEdge(name.to_string()))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, Shape, onnx::parser::DynamicInput, op::Op};

    // ── Test model builder ────────────────────────────────────────────────────

    fn make_node(
        op_type: &str,
        inputs: &[&str],
        outputs: &[&str],
        attributes: Vec<(&str, OnnxAttribute)>,
    ) -> OnnxNode {
        OnnxNode {
            op_type: op_type.to_string(),
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            outputs: outputs.iter().map(|s| s.to_string()).collect(),
            attributes: attributes
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect(),
        }
    }

    fn make_dynamic(name: &str, shape: &[u64], dtype: DType) -> DynamicInput {
        DynamicInput {
            name: name.to_string(),
            shape: Shape(shape.to_vec()),
            dtype,
        }
    }

    fn make_weight(name: &str, data: &[f32], shape: &[u64]) -> (String, Buffer) {
        let buf = Buffer::from_slice::<f32>(data, shape, DType::F32);
        (name.to_string(), buf)
    }

    // ── Task 7.7: simple Add graph ────────────────────────────────────────────

    #[test]
    fn map_simple_add_graph() {
        let model = OnnxModel {
            nodes: vec![make_node("Add", &["X", "Y"], &["Z"], vec![])],
            initializers: vec![],
            dynamic_inputs: vec![
                make_dynamic("X", &[4], DType::F32),
                make_dynamic("Y", &[4], DType::F32),
            ],
            outputs: vec!["Z".to_string()],
        };

        let (trace, output_ids, weights) = map_graph(&model).expect("map_graph failed");

        assert!(weights.is_empty());
        assert_eq!(output_ids.len(), 1);

        // ops: Input(X), Input(Y), Add → 3 ops total
        assert_eq!(trace.ops().len(), 3);
        assert_eq!(trace.input_count(), 2);

        // The output node must be an Add op.
        let out_id = output_ids[0];
        assert!(
            matches!(trace.get(out_id), Op::Add { .. }),
            "expected Op::Add, got {:?}",
            trace.get(out_id)
        );
    }

    #[test]
    fn map_relu_graph() {
        let model = OnnxModel {
            nodes: vec![make_node("Relu", &["X"], &["Y"], vec![])],
            initializers: vec![],
            dynamic_inputs: vec![make_dynamic("X", &[3], DType::F32)],
            outputs: vec!["Y".to_string()],
        };

        let (trace, output_ids, _) = map_graph(&model).expect("map_graph failed");
        let out_id = output_ids[0];
        assert!(
            matches!(trace.get(out_id), Op::Relu { .. }),
            "expected Op::Relu, got {:?}",
            trace.get(out_id)
        );
    }

    #[test]
    fn map_matmul_graph() {
        let model = OnnxModel {
            nodes: vec![make_node("MatMul", &["A", "B"], &["C"], vec![])],
            initializers: vec![],
            dynamic_inputs: vec![
                make_dynamic("A", &[2, 3], DType::F32),
                make_dynamic("B", &[3, 4], DType::F32),
            ],
            outputs: vec!["C".to_string()],
        };

        let (trace, output_ids, _) = map_graph(&model).expect("map_graph failed");
        let out_id = output_ids[0];
        assert!(
            matches!(trace.get(out_id), Op::Matmul { .. }),
            "expected Op::Matmul, got {:?}",
            trace.get(out_id)
        );
    }

    // ── Task 7.8: diamond graph (residual connection) ─────────────────────────
    //
    // A → B (Relu)
    // A → C (Neg)
    // B + C → D (Add)
    //
    // The edge "A" feeds both B and C. This tests that the value map correctly
    // stores handles that can be reused across multiple consumers.

    #[test]
    fn map_diamond_graph() {
        let model = OnnxModel {
            nodes: vec![
                make_node("Relu", &["A"], &["B"], vec![]),
                make_node("Neg", &["A"], &["C"], vec![]),
                make_node("Add", &["B", "C"], &["D"], vec![]),
            ],
            initializers: vec![],
            dynamic_inputs: vec![make_dynamic("A", &[4], DType::F32)],
            outputs: vec!["D".to_string()],
        };

        let (trace, output_ids, _) = map_graph(&model).expect("map_graph failed");

        // ops: Input(A), Relu(B), Neg(C), Add(D) → 4 ops
        assert_eq!(trace.ops().len(), 4);
        assert_eq!(trace.input_count(), 1);

        let out_id = output_ids[0];
        assert!(
            matches!(trace.get(out_id), Op::Add { .. }),
            "expected final op to be Op::Add, got {:?}",
            trace.get(out_id)
        );
    }

    // ── Task 7.9: deterministic argument ordering ─────────────────────────────
    //
    // Model has 1 dynamic input and 2 weights. Dynamic inputs must appear as
    // Input ops before weight Input ops, matching the runtime calling convention.

    #[test]
    fn dynamic_inputs_before_weights() {
        let model = OnnxModel {
            nodes: vec![
                make_node("Add", &["X", "W1"], &["tmp"], vec![]),
                make_node("Add", &["tmp", "W2"], &["out"], vec![]),
            ],
            initializers: vec![
                make_weight("W1", &[1.0, 2.0, 3.0, 4.0], &[4]),
                make_weight("W2", &[0.1, 0.2, 0.3, 0.4], &[4]),
            ],
            dynamic_inputs: vec![make_dynamic("X", &[4], DType::F32)],
            outputs: vec!["out".to_string()],
        };

        let (trace, _output_ids, weights) = map_graph(&model).expect("map_graph failed");

        // Weights vec should have 2 entries in initializer order.
        assert_eq!(weights.len(), 2);

        // The first Input op (arg_index 0) must be the dynamic input X.
        // The next two (arg_index 1, 2) must be the weight inputs W1 and W2.
        let inputs: Vec<(u32, &Shape, DType)> = trace
            .ops()
            .iter()
            .filter_map(|op| {
                if let Op::Input {
                    arg_index,
                    shape,
                    dtype,
                } = op
                {
                    Some((*arg_index, shape, *dtype))
                } else {
                    None
                }
            })
            .collect();

        assert_eq!(
            inputs.len(),
            3,
            "expected 3 Input ops (1 dynamic + 2 weights)"
        );
        // arg_index must be 0, 1, 2 in order.
        assert_eq!(inputs[0].0, 0);
        assert_eq!(inputs[1].0, 1);
        assert_eq!(inputs[2].0, 2);
        // Dynamic input has shape [4], weights also [4].
        assert_eq!(inputs[0].1, &Shape(vec![4]));
        assert_eq!(inputs[1].1, &Shape(vec![4]));
        assert_eq!(inputs[2].1, &Shape(vec![4]));
    }

    // ── Task 7.10: Softmax axis attribute extraction ──────────────────────────

    #[test]
    fn softmax_default_axis_is_minus_one() {
        // No `axis` attribute → should behave identically to axis=-1.
        let model = OnnxModel {
            nodes: vec![make_node("Softmax", &["X"], &["Y"], vec![])],
            initializers: vec![],
            dynamic_inputs: vec![make_dynamic("X", &[2, 3], DType::F32)],
            outputs: vec!["Y".to_string()],
        };

        let (trace, output_ids, _) = map_graph(&model).expect("map_graph failed");

        // Softmax decomposes into ReduceMax, Sub, Exp, ReduceSum, Div — 6 ops + 1 Input.
        // Input(X), ReduceMax, Sub, Exp, ReduceSum, Div  → 6 ops
        assert_eq!(trace.input_count(), 1);

        // The output should be a Div op (final step of softmax decomposition).
        let out_id = output_ids[0];
        assert!(
            matches!(trace.get(out_id), Op::Div { .. }),
            "expected Op::Div (softmax output), got {:?}",
            trace.get(out_id)
        );
    }

    #[test]
    fn softmax_explicit_axis_one() {
        let model = OnnxModel {
            nodes: vec![make_node(
                "Softmax",
                &["X"],
                &["Y"],
                vec![("axis", OnnxAttribute::Int(1))],
            )],
            initializers: vec![],
            dynamic_inputs: vec![make_dynamic("X", &[2, 3], DType::F32)],
            outputs: vec!["Y".to_string()],
        };

        let (trace, output_ids, _) = map_graph(&model).expect("map_graph failed");

        // Regardless of axis value, output is always a Div.
        let out_id = output_ids[0];
        assert!(
            matches!(trace.get(out_id), Op::Div { .. }),
            "expected Op::Div (softmax output), got {:?}",
            trace.get(out_id)
        );

        // The ReduceMax op should use dim=1 (axis 1 on rank-2 input).
        let reduce_max_op = trace
            .ops()
            .iter()
            .find(|op| matches!(op, Op::ReduceMax { .. }));
        assert!(reduce_max_op.is_some(), "expected a ReduceMax op");
        if let Some(Op::ReduceMax { dim, keepdim, .. }) = reduce_max_op {
            assert_eq!(*dim, 1, "ReduceMax should reduce dim 1");
            assert!(*keepdim, "ReduceMax in softmax must use keepdim=true");
        }
    }

    // ── Task 7.11: unsupported op returns clear error ─────────────────────────

    #[test]
    fn unsupported_op_returns_error() {
        let model = OnnxModel {
            nodes: vec![make_node("LSTM", &["X", "W", "R"], &["Y"], vec![])],
            initializers: vec![],
            dynamic_inputs: vec![make_dynamic("X", &[5, 1, 4], DType::F32)],
            outputs: vec!["Y".to_string()],
        };

        let result = map_graph(&model);
        match result {
            Err(OnnxError::UnsupportedOp(ref op)) if op == "LSTM" => {}
            Err(other) => panic!("expected UnsupportedOp(\"LSTM\"), got Err({other})"),
            Ok(_) => panic!("expected Err, got Ok"),
        }
    }

    #[test]
    fn unsupported_op_error_message_contains_op_name() {
        let model = OnnxModel {
            nodes: vec![make_node("GRU", &["X"], &["Y"], vec![])],
            initializers: vec![],
            dynamic_inputs: vec![make_dynamic("X", &[1, 1, 4, 4], DType::F32)],
            outputs: vec!["Y".to_string()],
        };

        let result = map_graph(&model);
        match result {
            Err(err) => {
                let msg = err.to_string();
                assert!(
                    msg.contains("GRU"),
                    "error message should mention 'GRU', got: {msg}"
                );
            }
            Ok(_) => panic!("expected an error for GRU, got Ok"),
        }
    }

    // ── Gemm attribute handling (Task 7.4) ────────────────────────────────────

    #[test]
    fn map_gemm_no_bias() {
        let model = OnnxModel {
            nodes: vec![make_node("Gemm", &["A", "B"], &["C"], vec![])],
            initializers: vec![],
            dynamic_inputs: vec![
                make_dynamic("A", &[2, 3], DType::F32),
                make_dynamic("B", &[3, 4], DType::F32),
            ],
            outputs: vec!["C".to_string()],
        };

        let (trace, output_ids, _) = map_graph(&model).expect("map_graph failed");
        let out_id = output_ids[0];
        match trace.get(out_id) {
            Op::Gemm {
                alpha,
                beta,
                trans_a,
                trans_b,
                bias,
                ..
            } => {
                assert_eq!(*alpha, 1.0);
                assert_eq!(*beta, 1.0);
                assert!(!trans_a);
                assert!(!trans_b);
                assert!(bias.is_none());
            }
            other => panic!("expected Op::Gemm, got {other:?}"),
        }
    }

    #[test]
    fn map_gemm_with_trans_flags() {
        let model = OnnxModel {
            nodes: vec![make_node(
                "Gemm",
                &["A", "B"],
                &["C"],
                vec![
                    ("transA", OnnxAttribute::Int(1)),
                    ("transB", OnnxAttribute::Int(1)),
                    ("alpha", OnnxAttribute::Float(2.0)),
                    ("beta", OnnxAttribute::Float(0.5)),
                ],
            )],
            initializers: vec![],
            dynamic_inputs: vec![
                make_dynamic("A", &[3, 2], DType::F32), // [K, M], transA → [M, K] = [2, 3]
                make_dynamic("B", &[4, 3], DType::F32), // [N, K], transB → [K, N] = [3, 4]
            ],
            outputs: vec!["C".to_string()],
        };

        let (trace, output_ids, _) = map_graph(&model).expect("map_graph failed");
        let out_id = output_ids[0];
        match trace.get(out_id) {
            Op::Gemm {
                alpha,
                beta,
                trans_a,
                trans_b,
                shape,
                ..
            } => {
                assert_eq!(*alpha, 2.0);
                assert_eq!(*beta, 0.5);
                assert!(*trans_a);
                assert!(*trans_b);
                assert_eq!(shape.0, vec![2, 4]);
            }
            other => panic!("expected Op::Gemm, got {other:?}"),
        }
    }

    // ── Clip maps to relu (Task 7.5) ──────────────────────────────────────────

    #[test]
    fn clip_maps_to_relu() {
        let model = OnnxModel {
            nodes: vec![make_node("Clip", &["X"], &["Y"], vec![])],
            initializers: vec![],
            dynamic_inputs: vec![make_dynamic("X", &[4], DType::F32)],
            outputs: vec!["Y".to_string()],
        };

        let (trace, output_ids, _) = map_graph(&model).expect("map_graph failed");
        let out_id = output_ids[0];
        assert!(
            matches!(trace.get(out_id), Op::Relu { .. }),
            "expected Clip to map to Op::Relu, got {:?}",
            trace.get(out_id)
        );
    }

    // ── Regression: Issue 2 — output edge name not found returns MissingEdge ──

    #[test]
    fn output_edge_not_found_returns_missing_edge() {
        // Model declares "Z" as the output but no node produces "Z".
        let model = OnnxModel {
            nodes: vec![make_node("Relu", &["X"], &["Y"], vec![])],
            initializers: vec![],
            dynamic_inputs: vec![make_dynamic("X", &[4], DType::F32)],
            outputs: vec!["Z".to_string()], // "Z" is never produced
        };

        let result = map_graph(&model);
        match result {
            Err(OnnxError::MissingEdge(ref name)) if name == "Z" => {}
            Err(other) => panic!("expected MissingEdge(\"Z\"), got Err({other})"),
            Ok(_) => panic!("expected Err, got Ok"),
        }
    }

    // ── Regression: Issue 3 — Clip with min/max inputs returns UnsupportedOp ──

    #[test]
    fn clip_with_min_max_inputs_returns_unsupported() {
        // Clip with 3 inputs (tensor, min, max) should not silently map to relu.
        let model = OnnxModel {
            nodes: vec![make_node("Clip", &["X", "min", "max"], &["Y"], vec![])],
            initializers: vec![
                make_weight("min", &[0.0], &[1]),
                make_weight("max", &[6.0], &[1]),
            ],
            dynamic_inputs: vec![make_dynamic("X", &[4], DType::F32)],
            outputs: vec!["Y".to_string()],
        };

        let result = map_graph(&model);
        match result {
            Err(OnnxError::UnsupportedOp(_)) => {}
            Err(other) => panic!("expected UnsupportedOp, got Err({other})"),
            Ok(_) => panic!("expected Err for Clip with bounds, got Ok"),
        }
    }

    // ── Regression: Issue 5 — trace leak: error then success does not panic ───

    #[test]
    fn trace_cleaned_up_after_error_so_second_call_succeeds() {
        // First call: model with an unsupported op → returns error.
        let bad_model = OnnxModel {
            nodes: vec![make_node("LSTM", &["X"], &["Y"], vec![])],
            initializers: vec![],
            dynamic_inputs: vec![make_dynamic("X", &[4], DType::F32)],
            outputs: vec!["Y".to_string()],
        };
        let result = map_graph(&bad_model);
        assert!(result.is_err(), "expected error from bad model");

        // Second call: valid model — must succeed without panicking.
        let good_model = OnnxModel {
            nodes: vec![make_node("Relu", &["X"], &["Y"], vec![])],
            initializers: vec![],
            dynamic_inputs: vec![make_dynamic("X", &[4], DType::F32)],
            outputs: vec!["Y".to_string()],
        };
        let result = map_graph(&good_model);
        assert!(
            result.is_ok(),
            "second call should succeed but got an error"
        );
    }

    // ── Missing op mapping tests ─────────────────────────────────────────────

    #[test]
    fn map_sub_graph() {
        let model = OnnxModel {
            nodes: vec![make_node("Sub", &["X", "Y"], &["Z"], vec![])],
            initializers: vec![],
            dynamic_inputs: vec![
                make_dynamic("X", &[4], DType::F32),
                make_dynamic("Y", &[4], DType::F32),
            ],
            outputs: vec!["Z".to_string()],
        };
        let (trace, output_ids, _) = map_graph(&model).expect("map_graph failed");
        assert!(
            matches!(trace.get(output_ids[0]), Op::Sub { .. }),
            "expected Op::Sub, got {:?}",
            trace.get(output_ids[0])
        );
    }

    #[test]
    fn map_mul_graph() {
        let model = OnnxModel {
            nodes: vec![make_node("Mul", &["X", "Y"], &["Z"], vec![])],
            initializers: vec![],
            dynamic_inputs: vec![
                make_dynamic("X", &[4], DType::F32),
                make_dynamic("Y", &[4], DType::F32),
            ],
            outputs: vec!["Z".to_string()],
        };
        let (trace, output_ids, _) = map_graph(&model).expect("map_graph failed");
        assert!(
            matches!(trace.get(output_ids[0]), Op::Mul { .. }),
            "expected Op::Mul, got {:?}",
            trace.get(output_ids[0])
        );
    }

    #[test]
    fn map_div_graph() {
        let model = OnnxModel {
            nodes: vec![make_node("Div", &["X", "Y"], &["Z"], vec![])],
            initializers: vec![],
            dynamic_inputs: vec![
                make_dynamic("X", &[4], DType::F32),
                make_dynamic("Y", &[4], DType::F32),
            ],
            outputs: vec!["Z".to_string()],
        };
        let (trace, output_ids, _) = map_graph(&model).expect("map_graph failed");
        assert!(
            matches!(trace.get(output_ids[0]), Op::Div { .. }),
            "expected Op::Div, got {:?}",
            trace.get(output_ids[0])
        );
    }

    #[test]
    fn map_neg_graph() {
        let model = OnnxModel {
            nodes: vec![make_node("Neg", &["X"], &["Y"], vec![])],
            initializers: vec![],
            dynamic_inputs: vec![make_dynamic("X", &[4], DType::F32)],
            outputs: vec!["Y".to_string()],
        };
        let (trace, output_ids, _) = map_graph(&model).expect("map_graph failed");
        assert!(
            matches!(trace.get(output_ids[0]), Op::Neg { .. }),
            "expected Op::Neg, got {:?}",
            trace.get(output_ids[0])
        );
    }

    #[test]
    fn map_exp_graph() {
        let model = OnnxModel {
            nodes: vec![make_node("Exp", &["X"], &["Y"], vec![])],
            initializers: vec![],
            dynamic_inputs: vec![make_dynamic("X", &[4], DType::F32)],
            outputs: vec!["Y".to_string()],
        };
        let (trace, output_ids, _) = map_graph(&model).expect("map_graph failed");
        assert!(
            matches!(trace.get(output_ids[0]), Op::Exp { .. }),
            "expected Op::Exp, got {:?}",
            trace.get(output_ids[0])
        );
    }

    #[test]
    fn map_tanh_graph() {
        let model = OnnxModel {
            nodes: vec![make_node("Tanh", &["X"], &["Y"], vec![])],
            initializers: vec![],
            dynamic_inputs: vec![make_dynamic("X", &[4], DType::F32)],
            outputs: vec!["Y".to_string()],
        };
        let (trace, output_ids, _) = map_graph(&model).expect("map_graph failed");
        assert!(
            matches!(trace.get(output_ids[0]), Op::Tanh { .. }),
            "expected Op::Tanh, got {:?}",
            trace.get(output_ids[0])
        );
    }

    // ── Multi-op chain tests ─────────────────────────────────────────────────

    #[test]
    fn map_add_relu_chain() {
        let model = OnnxModel {
            nodes: vec![
                make_node("Add", &["X", "Y"], &["sum"], vec![]),
                make_node("Relu", &["sum"], &["Z"], vec![]),
            ],
            initializers: vec![],
            dynamic_inputs: vec![
                make_dynamic("X", &[4], DType::F32),
                make_dynamic("Y", &[4], DType::F32),
            ],
            outputs: vec!["Z".to_string()],
        };
        let (trace, output_ids, _) = map_graph(&model).expect("map_graph failed");
        // Input(X), Input(Y), Add, Relu → 4 ops
        assert_eq!(trace.ops().len(), 4);
        assert!(matches!(trace.get(output_ids[0]), Op::Relu { .. }));
    }

    #[test]
    fn map_matmul_add_relu_chain() {
        let model = OnnxModel {
            nodes: vec![
                make_node("MatMul", &["X", "W"], &["mm"], vec![]),
                make_node("Add", &["mm", "B"], &["biased"], vec![]),
                make_node("Relu", &["biased"], &["Y"], vec![]),
            ],
            initializers: vec![
                make_weight("W", &[1.0; 12], &[3, 4]),
                make_weight("B", &[0.0; 4], &[4]),
            ],
            dynamic_inputs: vec![make_dynamic("X", &[2, 3], DType::F32)],
            outputs: vec!["Y".to_string()],
        };
        let (trace, output_ids, weights) = map_graph(&model).expect("map_graph failed");
        assert_eq!(weights.len(), 2);
        // Input(X), Input(W), Input(B), Matmul, Add, Relu → 6 ops
        assert_eq!(trace.ops().len(), 6);
        assert!(matches!(trace.get(output_ids[0]), Op::Relu { .. }));
    }

    // ── Conv mapping (task 12.1) ─────────────────────────────────────────────

    #[test]
    fn map_conv_graph() {
        let model = OnnxModel {
            nodes: vec![make_node(
                "Conv",
                &["X", "W"],
                &["Y"],
                vec![
                    ("kernel_shape", OnnxAttribute::Ints(vec![3, 3])),
                    ("strides", OnnxAttribute::Ints(vec![1, 1])),
                    ("pads", OnnxAttribute::Ints(vec![0, 0, 0, 0])),
                    ("dilations", OnnxAttribute::Ints(vec![1, 1])),
                ],
            )],
            initializers: vec![make_weight("W", &[1.0; 9], &[1, 1, 3, 3])],
            dynamic_inputs: vec![make_dynamic("X", &[1, 1, 5, 5], DType::F32)],
            outputs: vec!["Y".to_string()],
        };
        let (trace, output_ids, weights) = map_graph(&model).expect("map_graph failed");
        assert_eq!(weights.len(), 1);
        assert!(
            matches!(trace.get(output_ids[0]), Op::Conv2d { .. }),
            "expected Op::Conv2d, got {:?}",
            trace.get(output_ids[0])
        );
    }

    // ── MaxPool mapping (task 12.2) ──────────────────────────────────────────

    #[test]
    fn map_max_pool_graph() {
        let model = OnnxModel {
            nodes: vec![make_node(
                "MaxPool",
                &["X"],
                &["Y"],
                vec![
                    ("kernel_shape", OnnxAttribute::Ints(vec![2, 2])),
                    ("strides", OnnxAttribute::Ints(vec![2, 2])),
                    ("pads", OnnxAttribute::Ints(vec![0, 0, 0, 0])),
                ],
            )],
            initializers: vec![],
            dynamic_inputs: vec![make_dynamic("X", &[1, 1, 4, 4], DType::F32)],
            outputs: vec!["Y".to_string()],
        };
        let (trace, output_ids, _) = map_graph(&model).expect("map_graph failed");
        assert!(
            matches!(trace.get(output_ids[0]), Op::MaxPool2d { .. }),
            "expected Op::MaxPool2d, got {:?}",
            trace.get(output_ids[0])
        );
    }

    // ── Reshape mapping (task 12.3) ──────────────────────────────────────────

    fn make_i64_weight(name: &str, data: &[i64], shape: &[u64]) -> (String, Buffer) {
        let buf = Buffer::from_slice::<i64>(data, shape, DType::I64);
        (name.to_string(), buf)
    }

    #[test]
    fn map_reshape_graph() {
        let model = OnnxModel {
            nodes: vec![make_node("Reshape", &["X", "shape"], &["Y"], vec![])],
            initializers: vec![make_i64_weight("shape", &[3, 4], &[2])],
            dynamic_inputs: vec![make_dynamic("X", &[2, 6], DType::F32)],
            outputs: vec!["Y".to_string()],
        };
        let (trace, output_ids, _) = map_graph(&model).expect("map_graph failed");
        assert!(
            matches!(trace.get(output_ids[0]), Op::Reshape { .. }),
            "expected Op::Reshape, got {:?}",
            trace.get(output_ids[0])
        );
    }

    // ── Flatten mapping (task 12.3) ──────────────────────────────────────────

    #[test]
    fn map_flatten_graph() {
        let model = OnnxModel {
            nodes: vec![make_node(
                "Flatten",
                &["X"],
                &["Y"],
                vec![("axis", OnnxAttribute::Int(1))],
            )],
            initializers: vec![],
            dynamic_inputs: vec![make_dynamic("X", &[2, 3, 4], DType::F32)],
            outputs: vec!["Y".to_string()],
        };
        let (trace, output_ids, _) = map_graph(&model).expect("map_graph failed");
        let out = trace.get(output_ids[0]);
        assert!(
            matches!(out, Op::Reshape { .. }),
            "expected Op::Reshape, got {:?}",
            out
        );
        // Flatten(axis=1) on [2,3,4] → [2, 12]
        assert_eq!(out.shape().0, vec![2, 12]);
    }

    #[test]
    fn map_flatten_default_axis() {
        let model = OnnxModel {
            nodes: vec![make_node("Flatten", &["X"], &["Y"], vec![])],
            initializers: vec![],
            dynamic_inputs: vec![make_dynamic("X", &[2, 3, 4], DType::F32)],
            outputs: vec!["Y".to_string()],
        };
        let (trace, output_ids, _) = map_graph(&model).expect("map_graph failed");
        let out = trace.get(output_ids[0]);
        // Default axis=1: [2, 3*4] = [2, 12]
        assert_eq!(out.shape().0, vec![2, 12]);
    }
}
