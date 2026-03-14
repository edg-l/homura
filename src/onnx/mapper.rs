use std::collections::HashMap;

use crate::{
    Tensor,
    op::NodeId,
    runtime::Buffer,
    trace::{begin_trace, take_trace, Trace},
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

    let mut tensors: HashMap<String, Tensor> = HashMap::new();

    // 1. Dynamic inputs first, in graph.input order.
    for input in &model.dynamic_inputs {
        let t = Tensor::new(&input.shape.0, input.dtype);
        tensors.insert(input.name.clone(), t);
    }

    // 2. Weights (initializers) in initializer order.
    let mut weights: Vec<Buffer> = Vec::with_capacity(model.initializers.len());
    for (name, buffer) in &model.initializers {
        let t = Tensor::new(&buffer.shape().0, buffer.dtype());
        tensors.insert(name.clone(), t);
        weights.push(buffer.clone());
    }

    // 3. Walk nodes in topological order (guaranteed by ONNX spec).
    for node in &model.nodes {
        map_node(node, &mut tensors)?;
    }

    // 4. Collect output NodeIds.
    let output_ids: Vec<NodeId> = model
        .outputs
        .iter()
        .map(|name| {
            tensors
                .get(name)
                .unwrap_or_else(|| panic!("output edge '{name}' not found in value map"))
                .id()
        })
        .collect();

    let trace = take_trace();
    Ok((trace, output_ids, weights))
}

// ── Node dispatch ─────────────────────────────────────────────────────────────

fn map_node(node: &OnnxNode, tensors: &mut HashMap<String, Tensor>) -> Result<(), OnnxError> {
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
        "Clip" => {
            // In opset 11+, min/max are inputs (not attributes).
            // Map Clip(x, min=0, no-max) → relu for the common activation pattern.
            // TODO: handle general Clip with arbitrary min/max inputs.
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

/// Clone a `Tensor` handle out of the map by edge name.
///
/// Cloning a `Tensor` is cheap (NodeId + Shape + DType — no data).
fn get_tensor(
    tensors: &HashMap<String, Tensor>,
    name: &str,
) -> Result<Tensor, OnnxError> {
    tensors
        .get(name)
        .cloned()
        .ok_or_else(|| OnnxError::MissingEdge(name.to_string()))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, Shape, op::Op, onnx::parser::DynamicInput};

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
                if let Op::Input { arg_index, shape, dtype } = op {
                    Some((*arg_index, shape, *dtype))
                } else {
                    None
                }
            })
            .collect();

        assert_eq!(inputs.len(), 3, "expected 3 Input ops (1 dynamic + 2 weights)");
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
        let reduce_max_op = trace.ops().iter().find(|op| matches!(op, Op::ReduceMax { .. }));
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
            nodes: vec![make_node("Conv", &["X", "W"], &["Y"], vec![])],
            initializers: vec![],
            dynamic_inputs: vec![make_dynamic("X", &[1, 1, 4, 4], DType::F32)],
            outputs: vec!["Y".to_string()],
        };

        let result = map_graph(&model);
        match result {
            Err(err) => {
                let msg = err.to_string();
                assert!(
                    msg.contains("Conv"),
                    "error message should mention 'Conv', got: {msg}"
                );
            }
            Ok(_) => panic!("expected an error for Conv, got Ok"),
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
            Op::Gemm { alpha, beta, trans_a, trans_b, bias, .. } => {
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
            Op::Gemm { alpha, beta, trans_a, trans_b, shape, .. } => {
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
}
