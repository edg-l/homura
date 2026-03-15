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
        let shape = input.concrete_shape().expect(
            "map_graph_inner called with unresolved symbolic dims; resolve them before calling",
        );
        let t = Tensor::new(&shape.0, input.dtype);
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

    // Constant propagation map: seeded with small initializers only (≤64 elements).
    // Large float weight tensors (embedding tables, FC matrices) are excluded to
    // avoid duplicating hundreds of MB. Shape/index tensors (I32/I64) and small
    // scalar constants (used by Range, ConstantOfShape, etc.) are included.
    let mut constant_data: HashMap<String, Buffer> = initializer_data
        .iter()
        .filter(|(_, buf)| buf.shape().num_elements() <= 64)
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    // 3. Walk nodes in topological order (guaranteed by ONNX spec).
    for node in &model.nodes {
        map_node(node, &mut tensors, &mut constant_data, &mut weights)?;
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
    constant_data: &mut HashMap<String, Buffer>,
    weights: &mut Vec<Buffer>,
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
            let group = node
                .attributes
                .get("group")
                .and_then(|a| {
                    if let OnnxAttribute::Int(v) = a {
                        Some(*v)
                    } else {
                        None
                    }
                })
                .unwrap_or(1);
            if group != 1 {
                return Err(OnnxError::UnsupportedOp(format!(
                    "Conv: group={group} (grouped/depthwise conv) not supported"
                )));
            }
            let x = get_tensor(tensors, &node.inputs[0])?;
            let w = get_tensor(tensors, &node.inputs[1])?;
            let bias = if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
                Some(get_tensor(tensors, &node.inputs[2])?)
            } else {
                None
            };
            let strides = get_ints_attr(&node.attributes, "strides", &[1, 1]);
            let dilations = get_ints_attr(&node.attributes, "dilations", &[1, 1]);
            let auto_pad = get_str_attr(&node.attributes, "auto_pad", "NOTSET");
            let pads = if auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER" {
                // Compute padding so output_size = ceil(input_size / stride).
                let in_h = x.shape().0[2] as i64;
                let in_w = x.shape().0[3] as i64;
                let kh = w.shape().0[2] as i64;
                let kw = w.shape().0[3] as i64;
                let sh = strides[0];
                let sw = strides[1];
                let dh = dilations[0];
                let dw = dilations[1];
                let out_h = (in_h + sh - 1) / sh;
                let out_w = (in_w + sw - 1) / sw;
                let pad_h = 0.max((out_h - 1) * sh + dh * (kh - 1) + 1 - in_h);
                let pad_w = 0.max((out_w - 1) * sw + dw * (kw - 1) + 1 - in_w);
                if auto_pad == "SAME_UPPER" {
                    // Extra padding goes to bottom/right.
                    vec![pad_h / 2, pad_w / 2, pad_h - pad_h / 2, pad_w - pad_w / 2]
                } else {
                    // SAME_LOWER: extra padding goes to top/left.
                    vec![pad_h - pad_h / 2, pad_w - pad_w / 2, pad_h / 2, pad_w / 2]
                }
            } else {
                get_ints_attr(&node.attributes, "pads", &[0, 0, 0, 0])
            };
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
            let kh = kernel_shape[0];
            let kw = kernel_shape[1];
            let sh = strides[0];
            let sw = strides[1];
            let result = x.max_pool2d(
                [kh as u64, kw as u64],
                [sh as u64, sw as u64],
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
            // ONNX Reshape takes the target shape as a second input. It may be a
            // static initializer/constant or a value produced by a traced op.
            let shape_name = &node.inputs[1];

            // Check whether the shape is fully static (in constant_data, no sentinels).
            let static_shape_opt = constant_data
                .get(shape_name)
                .filter(|buf| !contains_dynamic_sentinel(buf))
                .and_then(|buf| {
                    match buf.dtype() {
                        crate::DType::I64 => Some(buf.as_slice::<i64>().to_vec()),
                        crate::DType::I32 => Some(
                            buf.as_slice::<i32>().iter().map(|&v| v as i64).collect(),
                        ),
                        _ => None,
                    }
                });

            if let Some(target_shape) = static_shape_opt {
                // Static path: existing constant-fold reshape.
                let result = x.reshape(&target_shape);

                // Constant propagation: if the data input is a known constant,
                // propagate it with the new shape so downstream ops can read it.
                let x_name = &node.inputs[0];
                if let Some(in_buf) = constant_data.get(x_name).cloned() {
                    let new_shape_u64: Vec<u64> =
                        result.shape().0.iter().map(|&d| d).collect();
                    let mut out_buf = Buffer::new(&new_shape_u64, in_buf.dtype());
                    if out_buf.data.len() == in_buf.data.len() {
                        out_buf.data_mut().copy_from_slice(&in_buf.data);
                        constant_data.insert(node.outputs[0].clone(), out_buf);
                    }
                }

                tensors.insert(node.outputs[0].clone(), result);
            } else {
                // Dynamic path: shape comes from a traced tensor.
                // The output rank = number of elements in the shape tensor (a 1-D tensor).
                let shape_tensor = get_tensor(tensors, shape_name)?;
                let out_rank = shape_tensor.shape().0[0] as usize;

                // Try to get partial static dims from the constant buffer even if it
                // contains some sentinel values, so we don't lose static info.
                let output_shape = if let Some(buf) = constant_data.get(shape_name) {
                    if let Ok(dims) = read_i64_buffer(buf) {
                        let v: Vec<u64> = dims
                            .iter()
                            .map(|&d| {
                                if d == -1 || d == i64::MIN {
                                    crate::shape::DIM_DYNAMIC
                                } else {
                                    d as u64
                                }
                            })
                            .collect();
                        crate::Shape(v)
                    } else {
                        crate::Shape(vec![crate::shape::DIM_DYNAMIC; out_rank])
                    }
                } else {
                    crate::Shape(vec![crate::shape::DIM_DYNAMIC; out_rank])
                };

                let result = x.reshape_with_tensor(&shape_tensor, output_shape);
                tensors.insert(node.outputs[0].clone(), result);
            }
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
            assert!(
                axis <= shape.len(),
                "Flatten: axis {axis} out of range for rank {}",
                shape.len()
            );
            // ONNX Flatten always produces a 2D output.
            // axis==0 → [1, total], axis==rank → [total, 1].
            let dim0: u64 = if axis == 0 {
                1
            } else {
                shape[..axis].iter().product()
            };
            let dim1: u64 = if axis == shape.len() {
                1
            } else {
                shape[axis..].iter().product()
            };
            let result = x.reshape(&[dim0 as i64, dim1 as i64]);
            tensors.insert(node.outputs[0].clone(), result);
        }
        "BatchNormalization" => {
            let x = get_tensor(tensors, &node.inputs[0])?;
            let scale = get_tensor(tensors, &node.inputs[1])?;
            let bias = get_tensor(tensors, &node.inputs[2])?;
            let mean = get_tensor(tensors, &node.inputs[3])?;
            let var = get_tensor(tensors, &node.inputs[4])?;
            let epsilon = node
                .attributes
                .get("epsilon")
                .and_then(|attr| {
                    if let OnnxAttribute::Float(v) = attr {
                        Some(*v as f64)
                    } else {
                        None
                    }
                })
                .unwrap_or(1e-5);
            let result = x.batch_norm(&scale, &bias, &mean, &var, epsilon);
            tensors.insert(node.outputs[0].clone(), result);
        }
        "GlobalAveragePool" => {
            let x = get_tensor(tensors, &node.inputs[0])?;
            let result = x.global_avg_pool();
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

        // ── Trace-time constant ops ────────────────────────────────────────────
        //
        // These ops resolve entirely at trace time when shapes are statically known.
        // Each one produces a constant Buffer that is:
        //   1. Added to `weights` (so the JIT receives it as an input argument).
        //   2. Registered as a Tensor Input in the trace.
        //   3. Stored in `constant_data` so downstream nodes can read the values.

        "Constant" => {
            // ONNX Constant node: has a `value` attribute containing a TensorProto.
            let buf = node
                .attributes
                .get("value")
                .and_then(|attr| {
                    if let OnnxAttribute::Tensor(b) = attr {
                        Some(b.clone())
                    } else {
                        None
                    }
                })
                .ok_or_else(|| {
                    OnnxError::UnsupportedOp(
                        "Constant: missing or non-tensor `value` attribute".to_string(),
                    )
                })?;
            let t = register_constant(buf.clone(), constant_data, weights, &node.outputs[0]);
            tensors.insert(node.outputs[0].clone(), t);
        }

        "Shape" => {
            // ONNX Shape: returns the shape of the input tensor as an I64 1-D tensor.
            let input = get_tensor(tensors, &node.inputs[0])?;
            if input.shape().has_dynamic_dims() {
                // Dynamic path: trace a ShapeOf op so the compiler emits tensor.dim
                // at runtime for each dynamic dimension.
                let result = input.shape_of();
                tensors.insert(node.outputs[0].clone(), result);
                // Do NOT insert into constant_data — the values are only known at runtime.
            } else {
                // Static path: constant-fold as before.
                let dims: Vec<i64> = input.shape().0.iter().map(|&d| d as i64).collect();
                let buf = Buffer::from_slice::<i64>(&dims, &[dims.len() as u64], crate::DType::I64);
                let t = register_constant(buf, constant_data, weights, &node.outputs[0]);
                tensors.insert(node.outputs[0].clone(), t);
            }
        }

        "ConstantOfShape" => {
            // ONNX ConstantOfShape: creates a tensor of the given shape filled with `value`.
            // The shape comes from the first input (a 1-D I64 tensor).
            let shape_name = &node.inputs[0];

            // Parse fill value and dtype from the `value` attribute.
            let (fill_value, fill_dtype) =
                if let Some(OnnxAttribute::Tensor(val_buf)) = node.attributes.get("value") {
                    let fv = match val_buf.dtype() {
                        crate::DType::F32 => val_buf.as_slice::<f32>()[0] as f64,
                        crate::DType::F64 => val_buf.as_slice::<f64>()[0],
                        crate::DType::I32 => val_buf.as_slice::<i32>()[0] as f64,
                        crate::DType::I64 => val_buf.as_slice::<i64>()[0] as f64,
                    };
                    (fv, val_buf.dtype())
                } else {
                    (0.0, crate::DType::F32)
                };

            // Determine whether the shape is statically known without sentinels.
            let static_shape_opt = constant_data
                .get(shape_name)
                .filter(|buf| !contains_dynamic_sentinel(buf))
                .and_then(|buf| read_i64_buffer(buf).ok());

            if let Some(shape_dims) = static_shape_opt {
                // Static path: constant-fold as before.
                let shape_u64: Vec<u64> = shape_dims.iter().map(|&d| d as u64).collect();
                let buf = if let Some(OnnxAttribute::Tensor(val_buf)) = node.attributes.get("value") {
                    let dtype = val_buf.dtype();
                    let elem_size = dtype.size_bytes();
                    let total: usize = shape_u64.iter().product::<u64>().max(1) as usize;
                    let mut out = Buffer::new(&shape_u64, dtype);
                    let elem: &[u8] = &val_buf.data[..elem_size];
                    let out_bytes = out.data_mut();
                    for i in 0..total {
                        out_bytes[i * elem_size..(i + 1) * elem_size].copy_from_slice(elem);
                    }
                    out
                } else {
                    Buffer::new(&shape_u64, crate::DType::F32)
                };
                let t = register_constant(buf, constant_data, weights, &node.outputs[0]);
                tensors.insert(node.outputs[0].clone(), t);
            } else {
                // Dynamic path: the shape tensor is not fully static.
                // Determine which output dims are known vs dynamic.
                let shape_tensor = get_tensor(tensors, shape_name)?;
                // Rank of the output = number of elements in the 1-D shape tensor.
                // shape_tensor.shape() = [rank] (it's a 1-D tensor of that many elements).
                let out_rank = shape_tensor.shape().0[0] as usize;

                // Try to get partial shape info from constant_data (even if sentinel).
                // For positions with sentinel values, mark as DIM_DYNAMIC.
                let output_shape = if let Some(buf) = constant_data.get(shape_name) {
                    if let Ok(dims) = read_i64_buffer(buf) {
                        let v: Vec<u64> = dims
                            .iter()
                            .map(|&d| {
                                if d == i64::MIN || d < 0 {
                                    crate::shape::DIM_DYNAMIC
                                } else {
                                    d as u64
                                }
                            })
                            .collect();
                        crate::Shape(v)
                    } else {
                        // Unknown — all dims dynamic.
                        crate::Shape(vec![crate::shape::DIM_DYNAMIC; out_rank])
                    }
                } else {
                    // shape_input is a traced tensor — all output dims are unknown.
                    crate::Shape(vec![crate::shape::DIM_DYNAMIC; out_rank])
                };

                let result = Tensor::constant_of_shape(&shape_tensor, fill_value, output_shape, fill_dtype);
                tensors.insert(node.outputs[0].clone(), result);
            }
        }

        "Range" => {
            // ONNX Range: produces [start, start+delta, ...] up to (but not including) limit.
            let start_name = &node.inputs[0];
            let limit_name = &node.inputs[1];
            let delta_name = &node.inputs[2];

            // Static path: all three inputs are fully-static constants (no sentinel).
            let start_static = constant_data
                .get(start_name)
                .filter(|b| !contains_dynamic_sentinel(b))
                .cloned();
            let limit_static = constant_data
                .get(limit_name)
                .filter(|b| !contains_dynamic_sentinel(b))
                .cloned();
            let delta_static = constant_data
                .get(delta_name)
                .filter(|b| !contains_dynamic_sentinel(b))
                .cloned();

            if let (Some(start_buf), Some(limit_buf), Some(delta_buf)) =
                (start_static, limit_static, delta_static)
            {
                // All inputs share the same dtype (ONNX spec constraint).
                let dtype = start_buf.dtype();
                let buf = match dtype {
                    crate::DType::I32 => {
                        let start = start_buf.as_slice::<i32>()[0];
                        let limit = limit_buf.as_slice::<i32>()[0];
                        let delta = delta_buf.as_slice::<i32>()[0];
                        let values: Vec<i32> = range_values(start, limit, delta);
                        Buffer::from_slice::<i32>(&values, &[values.len() as u64], dtype)
                    }
                    crate::DType::I64 => {
                        let start = start_buf.as_slice::<i64>()[0];
                        let limit = limit_buf.as_slice::<i64>()[0];
                        let delta = delta_buf.as_slice::<i64>()[0];
                        let values: Vec<i64> = range_values(start, limit, delta);
                        Buffer::from_slice::<i64>(&values, &[values.len() as u64], dtype)
                    }
                    crate::DType::F32 => {
                        let start = start_buf.as_slice::<f32>()[0];
                        let limit = limit_buf.as_slice::<f32>()[0];
                        let delta = delta_buf.as_slice::<f32>()[0];
                        let values: Vec<f32> = range_values_f32(start, limit, delta);
                        Buffer::from_slice::<f32>(&values, &[values.len() as u64], dtype)
                    }
                    crate::DType::F64 => {
                        let start = start_buf.as_slice::<f64>()[0];
                        let limit = limit_buf.as_slice::<f64>()[0];
                        let delta = delta_buf.as_slice::<f64>()[0];
                        let values: Vec<f64> = range_values_f64(start, limit, delta);
                        Buffer::from_slice::<f64>(&values, &[values.len() as u64], dtype)
                    }
                };
                let t = register_constant(buf, constant_data, weights, &node.outputs[0]);
                tensors.insert(node.outputs[0].clone(), t);
            } else {
                // Dynamic path: at least one input is non-constant or contains sentinel.
                let start_t = get_tensor(tensors, start_name)?;
                let limit_t = get_tensor(tensors, limit_name)?;
                let delta_t = get_tensor(tensors, delta_name)?;
                let dtype = start_t.dtype();
                let output_shape = crate::Shape(vec![crate::shape::DIM_DYNAMIC]);
                let result = Tensor::range(&start_t, &limit_t, &delta_t, output_shape, dtype);
                tensors.insert(node.outputs[0].clone(), result);
            }
        }

        "Squeeze" => {
            // ONNX Squeeze: removes size-1 dimensions.
            // In opset 13+, axes come from the second input tensor.
            // In older opsets, axes is an attribute.
            let x = get_tensor(tensors, &node.inputs[0])?;
            let in_shape = x.shape().0.clone();
            let rank = in_shape.len() as i64;

            let axes: Option<Vec<i64>> = if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
                // Opset 13+: axes as input tensor.
                let axes_name = &node.inputs[1];
                let axes_buf = constant_data.get(axes_name).ok_or_else(|| {
                    OnnxError::UnsupportedOp(format!(
                        "Squeeze: axes input '{axes_name}' is not a known constant"
                    ))
                })?;
                Some(read_i64_buffer(axes_buf)?)
            } else if let Some(OnnxAttribute::Ints(v)) = node.attributes.get("axes") {
                Some(v.clone())
            } else {
                None // no axes → remove all size-1 dims
            };

            let new_shape: Vec<i64> = match axes {
                None => in_shape
                    .iter()
                    .filter(|&&d| d != 1)
                    .map(|&d| d as i64)
                    .collect(),
                Some(ax) => {
                    // Normalize negative axes.
                    let ax_set: std::collections::HashSet<i64> = ax
                        .iter()
                        .map(|&a| if a < 0 { a + rank } else { a })
                        .collect();
                    in_shape
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| !ax_set.contains(&(*i as i64)))
                        .map(|(_, &d)| d as i64)
                        .collect()
                }
            };

            let result = x.reshape(&new_shape);

            // Constant propagation: if the data input is a known constant,
            // propagate the squeezed buffer so downstream ops can read it.
            let x_name = &node.inputs[0];
            if let Some(in_buf) = constant_data.get(x_name).cloned() {
                let new_shape_u64: Vec<u64> = new_shape.iter().map(|&d| d as u64).collect();
                let mut out_buf = Buffer::new(&new_shape_u64, in_buf.dtype());
                if out_buf.data.len() == in_buf.data.len() {
                    out_buf.data_mut().copy_from_slice(&in_buf.data);
                    constant_data.insert(node.outputs[0].clone(), out_buf);
                }
            }

            tensors.insert(node.outputs[0].clone(), result);
        }

        "Unsqueeze" => {
            // ONNX Unsqueeze: inserts size-1 dimensions at specified axes.
            // In opset 13+, axes come from the second input tensor.
            // In older opsets, axes is an attribute.
            let x = get_tensor(tensors, &node.inputs[0])?;
            let in_shape = x.shape().0.clone();

            let axes: Vec<i64> = if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
                let axes_name = &node.inputs[1];
                let axes_buf = constant_data.get(axes_name).ok_or_else(|| {
                    OnnxError::UnsupportedOp(format!(
                        "Unsqueeze: axes input '{axes_name}' is not a known constant"
                    ))
                })?;
                read_i64_buffer(axes_buf)?
            } else if let Some(OnnxAttribute::Ints(v)) = node.attributes.get("axes") {
                v.clone()
            } else {
                return Err(OnnxError::UnsupportedOp(
                    "Unsqueeze: no axes specified".to_string(),
                ));
            };

            // Output rank = input rank + number of axes.
            let out_rank = (in_shape.len() + axes.len()) as i64;

            // Normalize negative axes against the output rank.
            let mut normalized: Vec<i64> = axes
                .iter()
                .map(|&a| if a < 0 { a + out_rank } else { a })
                .collect();
            normalized.sort_unstable();

            // Build the output shape by inserting 1s at the specified positions.
            let mut new_shape: Vec<i64> = in_shape.iter().map(|&d| d as i64).collect();
            for &ax in &normalized {
                new_shape.insert(ax as usize, 1);
            }

            let result = x.reshape(&new_shape);

            // Constant propagation: if the input is a known constant, the
            // unsqueeze output has the same data with a new shape.
            let x_name = &node.inputs[0];
            if let Some(in_buf) = constant_data.get(x_name).cloned() {
                let new_shape_u64: Vec<u64> = new_shape.iter().map(|&d| d as u64).collect();
                let mut out_buf = Buffer::new(&new_shape_u64, in_buf.dtype());
                out_buf.data_mut().copy_from_slice(&in_buf.data);
                constant_data.insert(node.outputs[0].clone(), out_buf);
            }

            tensors.insert(node.outputs[0].clone(), result);
        }

        "Pow" => {
            let a = get_tensor(tensors, &node.inputs[0])?;
            let b = get_tensor(tensors, &node.inputs[1])?;
            let result = a.pow(&b);
            tensors.insert(node.outputs[0].clone(), result);
        }

        "Sqrt" => {
            let a = get_tensor(tensors, &node.inputs[0])?;
            let result = a.sqrt();
            tensors.insert(node.outputs[0].clone(), result);
        }

        "Cast" => {
            let a = get_tensor(tensors, &node.inputs[0])?;
            let to = node
                .attributes
                .get("to")
                .and_then(|attr| {
                    if let OnnxAttribute::Int(v) = attr {
                        Some(*v)
                    } else {
                        None
                    }
                })
                .ok_or_else(|| {
                    OnnxError::UnsupportedOp("Cast: missing `to` attribute".to_string())
                })?;
            let target_dtype = onnx_dtype_to_internal(to).ok_or_else(|| {
                OnnxError::UnsupportedOp(format!("Cast: unsupported ONNX dtype {to}"))
            })?;
            let result = a.cast(target_dtype);

            // Constant propagation: if the input is a known integer constant,
            // cast its values and store in constant_data so downstream Range/
            // Reshape etc. can read the result.
            let a_name = &node.inputs[0];
            if let Some(in_buf) = constant_data.get(a_name).cloned() {
                if let Some(const_buf) = eval_cast_constant(&in_buf, target_dtype) {
                    constant_data.insert(node.outputs[0].clone(), const_buf);
                }
            }

            tensors.insert(node.outputs[0].clone(), result);
        }

        "ReduceMean" => {
            let a = get_tensor(tensors, &node.inputs[0])?;
            let axes: Vec<i64> = if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
                // Opset 18+: axes as input tensor.
                let axes_name = &node.inputs[1];
                let axes_buf = constant_data.get(axes_name).ok_or_else(|| {
                    OnnxError::UnsupportedOp(format!(
                        "ReduceMean: axes input '{axes_name}' is not a known constant"
                    ))
                })?;
                read_i64_buffer(axes_buf)?
            } else if let Some(OnnxAttribute::Ints(v)) = node.attributes.get("axes") {
                v.clone()
            } else {
                // No axes: reduce over all dimensions.
                (0..a.shape().rank() as i64).collect()
            };
            let keepdim = node
                .attributes
                .get("keepdims")
                .and_then(|attr| {
                    if let OnnxAttribute::Int(v) = attr {
                        Some(*v != 0)
                    } else {
                        None
                    }
                })
                .unwrap_or(true);
            let result = a.reduce_mean(&axes, keepdim);
            tensors.insert(node.outputs[0].clone(), result);
        }

        "Gather" => {
            let data = get_tensor(tensors, &node.inputs[0])?;
            let indices = get_tensor(tensors, &node.inputs[1])?;
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
                .unwrap_or(0);
            let result = data.gather(&indices, axis);

            // Constant propagation: if both inputs are known constants, evaluate
            // the gather at trace time so downstream Reshape/etc. can read the result.
            // Focus on the common pattern: axis=0, scalar I64 index into an I64 tensor.
            let data_name = &node.inputs[0];
            let indices_name = &node.inputs[1];
            if let (Some(data_buf), Some(idx_buf)) = (
                constant_data.get(data_name).cloned(),
                constant_data.get(indices_name).cloned(),
            ) {
                if let Some(const_buf) =
                    eval_gather_constant(&data_buf, &idx_buf, axis)
                {
                    constant_data.insert(node.outputs[0].clone(), const_buf);
                }
            }

            tensors.insert(node.outputs[0].clone(), result);
        }

        "Slice" => {
            let data = get_tensor(tensors, &node.inputs[0])?;
            // ONNX Slice opset 10+: starts, ends, axes, steps are all input tensors.
            let starts_name = &node.inputs[1];
            let ends_name = &node.inputs[2];
            let starts_buf = constant_data.get(starts_name).ok_or_else(|| {
                OnnxError::UnsupportedOp(format!(
                    "Slice: starts input '{starts_name}' must be a static constant"
                ))
            })?;
            let ends_buf = constant_data.get(ends_name).ok_or_else(|| {
                OnnxError::UnsupportedOp(format!(
                    "Slice: ends input '{ends_name}' must be a static constant"
                ))
            })?;
            let starts = read_i64_buffer(starts_buf)?;
            let ends = read_i64_buffer(ends_buf)?;

            // axes is optional (input index 3).
            let axes: Vec<i64> = if node.inputs.len() > 3 && !node.inputs[3].is_empty() {
                let axes_name = &node.inputs[3];
                let axes_buf = constant_data.get(axes_name).ok_or_else(|| {
                    OnnxError::UnsupportedOp(format!(
                        "Slice: axes input '{axes_name}' must be a static constant"
                    ))
                })?;
                read_i64_buffer(axes_buf)?
            } else {
                // Default: axes = [0, 1, ..., starts.len()-1].
                (0..starts.len() as i64).collect()
            };

            // steps is optional (input index 4).
            let steps: Vec<i64> = if node.inputs.len() > 4 && !node.inputs[4].is_empty() {
                let steps_name = &node.inputs[4];
                let steps_buf = constant_data.get(steps_name).ok_or_else(|| {
                    OnnxError::UnsupportedOp(format!(
                        "Slice: steps input '{steps_name}' must be a static constant"
                    ))
                })?;
                read_i64_buffer(steps_buf)?
            } else {
                vec![1i64; starts.len()]
            };

            let result = data.slice(&starts, &ends, &axes, &steps);

            // Constant propagation: if the data input is a known constant
            // (e.g. a shape tensor produced by Shape), evaluate the slice at
            // trace time so downstream Squeeze/Reshape etc. can read the result.
            let data_name = &node.inputs[0];
            if let Some(data_buf) = constant_data.get(data_name).cloned() {
                if let Some(const_buf) =
                    eval_slice_constant(&data_buf, &starts, &ends, &axes, &steps)
                {
                    constant_data.insert(node.outputs[0].clone(), const_buf);
                }
            }

            tensors.insert(node.outputs[0].clone(), result);
        }

        "Concat" => {
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
                .ok_or_else(|| {
                    OnnxError::UnsupportedOp("Concat: missing required `axis` attribute".to_string())
                })?;
            let input_tensors: Result<Vec<Tensor>, OnnxError> = node
                .inputs
                .iter()
                .map(|name| get_tensor(tensors, name))
                .collect();
            let input_tensors = input_tensors?;
            let refs: Vec<&Tensor> = input_tensors.iter().collect();
            let result = Tensor::concat(&refs, axis);

            // Constant propagation: if all inputs are known constants, concatenate
            // their buffers at trace time so downstream Reshape can use the result.
            let input_bufs: Vec<Buffer> = node
                .inputs
                .iter()
                .filter_map(|name| constant_data.get(name).cloned())
                .collect();
            if input_bufs.len() == node.inputs.len() {
                if let Some(const_buf) = eval_concat_constant(&input_bufs, axis) {
                    constant_data.insert(node.outputs[0].clone(), const_buf);
                }
            }

            tensors.insert(node.outputs[0].clone(), result);
        }

        "Split" => {
            // Resolve Split by emitting multiple Slice ops — one per output.
            // The split axis is an attribute.
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
                .unwrap_or(0);
            let input = get_tensor(tensors, &node.inputs[0])?;
            let rank = input.shape().rank() as i64;
            let ax = if axis < 0 { axis + rank } else { axis } as usize;
            let ax_size = input.shape().0[ax] as i64;
            let num_outputs = node.outputs.len();

            // Determine split sizes: either from the `split` input tensor or equal splits.
            let split_sizes: Vec<i64> =
                if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
                    let split_name = &node.inputs[1];
                    let split_buf = constant_data.get(split_name).ok_or_else(|| {
                        OnnxError::UnsupportedOp(format!(
                            "Split: split input '{split_name}' must be a static constant"
                        ))
                    })?;
                    read_i64_buffer(split_buf)?
                } else if let Some(OnnxAttribute::Ints(v)) = node.attributes.get("split") {
                    v.clone()
                } else {
                    // Equal split.
                    assert!(
                        ax_size % num_outputs as i64 == 0,
                        "Split: axis size {ax_size} not divisible by output count {num_outputs}"
                    );
                    vec![ax_size / num_outputs as i64; num_outputs]
                };

            // Emit one Slice per output.
            let input_rank = input.shape().rank();
            let mut offset = 0i64;
            for (out_idx, size) in split_sizes.iter().enumerate() {
                let starts = vec![offset];
                let ends = vec![offset + size];
                let axes_vec = vec![ax as i64];
                let steps = vec![1i64];
                let slice_result = input.slice(&starts, &ends, &axes_vec, &steps);
                tensors.insert(node.outputs[out_idx].clone(), slice_result);
                let _ = input_rank;
                offset += size;
            }
        }

        "Transpose" => {
            let x = get_tensor(tensors, &node.inputs[0])?;
            let rank = x.shape().rank();
            let perm: Vec<i64> = if let Some(OnnxAttribute::Ints(v)) = node.attributes.get("perm")
            {
                v.clone()
            } else {
                // Default: reverse dimensions.
                (0..rank as i64).rev().collect()
            };
            let result = x.transpose(&perm);
            tensors.insert(node.outputs[0].clone(), result);
        }

        "Where" => {
            let condition = get_tensor(tensors, &node.inputs[0])?;
            let x = get_tensor(tensors, &node.inputs[1])?;
            let y = get_tensor(tensors, &node.inputs[2])?;
            // ONNX Where condition is boolean (BOOL dtype = 9). We receive it as I64
            // from a Cast or comparison, but it may arrive as I32. Cast to I64 if needed.
            let cond_i64 = if condition.dtype() != crate::DType::I64 {
                condition.cast(crate::DType::I64)
            } else {
                condition
            };
            let result = Tensor::where_select(&cond_i64, &x, &y);
            tensors.insert(node.outputs[0].clone(), result);
        }

        other => {
            return Err(OnnxError::UnsupportedOp(other.to_string()));
        }
    }

    // General constant folding: after the op is mapped, check if all inputs
    // were constants. If so, try to evaluate the output at trace time so
    // downstream ops (especially arithmetic in shape chains) can read it.
    try_constant_fold(node, constant_data);

    Ok(())
}

/// Convert an ONNX data type integer to the internal DType.
///
/// ONNX data type enum values:
///   1 = FLOAT (f32), 2 = UINT8, 3 = INT8, 4 = UINT16, 5 = INT16,
///   6 = INT32,        7 = INT64, 10 = FLOAT16, 11 = DOUBLE (f64),
///   12 = UINT32,      13 = UINT64
fn onnx_dtype_to_internal(onnx_dtype: i64) -> Option<crate::DType> {
    match onnx_dtype {
        1 => Some(crate::DType::F32),
        6 => Some(crate::DType::I32),
        7 | 9 => Some(crate::DType::I64), // 9 = Bool → treat as I64
        11 => Some(crate::DType::F64),
        _ => None,
    }
}

/// Extract a `String` attribute, returning `default` if missing.
fn get_str_attr<'a>(
    attrs: &'a HashMap<String, OnnxAttribute>,
    name: &str,
    default: &'a str,
) -> &'a str {
    attrs
        .get(name)
        .and_then(|attr| {
            if let OnnxAttribute::String(v) = attr {
                Some(v.as_str())
            } else {
                None
            }
        })
        .unwrap_or(default)
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

/// Register a constant buffer as a new weight input in the trace.
///
/// Adds `buf` to `weights`, creates a `Tensor::new` Input op in the trace,
/// and stores the buffer in `constant_data` under `edge_name`. Returns the
/// new Tensor handle (caller is responsible for inserting it into `tensors`).
fn register_constant(
    buf: Buffer,
    constant_data: &mut HashMap<String, Buffer>,
    weights: &mut Vec<Buffer>,
    edge_name: &str,
) -> Tensor {
    let t = Tensor::new(&buf.shape().0, buf.dtype());
    constant_data.insert(edge_name.to_string(), buf.clone());
    weights.push(buf);
    t
}

/// Read an integer buffer as a `Vec<i64>`, accepting I32 or I64.
fn read_i64_buffer(buf: &Buffer) -> Result<Vec<i64>, OnnxError> {
    match buf.dtype() {
        crate::DType::I64 => Ok(buf.as_slice::<i64>().to_vec()),
        crate::DType::I32 => Ok(buf
            .as_slice::<i32>()
            .iter()
            .map(|&v| v as i64)
            .collect()),
        other => Err(OnnxError::UnsupportedOp(format!(
            "expected integer buffer (I32/I64), got {other:?}"
        ))),
    }
}

/// Returns `true` if an I64 buffer contains `i64::MIN` (the `DIM_DYNAMIC` sentinel).
///
/// When `DIM_DYNAMIC` (u64::MAX, which becomes i64::MIN when cast) leaks into a
/// constant buffer, constant-folding downstream ops would produce garbage results.
/// This guard prevents that.
fn contains_dynamic_sentinel(buf: &Buffer) -> bool {
    if buf.dtype() == crate::DType::I64 {
        buf.as_slice::<i64>().iter().any(|&v| v == i64::MIN)
    } else {
        false
    }
}

/// Compute integer range values: [start, start+delta, ...] stopping before limit.
///
/// Follows ONNX semantics: if delta > 0, produces ascending values < limit;
/// if delta < 0, produces descending values > limit.
fn range_values<T>(start: T, limit: T, delta: T) -> Vec<T>
where
    T: Copy + PartialOrd + std::ops::Add<Output = T> + Default,
{
    let mut values = Vec::new();
    let mut cur = start;
    let zero = T::default();
    if delta > zero {
        while cur < limit {
            values.push(cur);
            cur = cur + delta;
        }
    } else if delta < zero {
        while cur > limit {
            values.push(cur);
            cur = cur + delta;
        }
    }
    // delta == 0 → empty (avoids infinite loop)
    values
}

/// Compute f32 range values.
fn range_values_f32(start: f32, limit: f32, delta: f32) -> Vec<f32> {
    let mut values = Vec::new();
    let mut cur = start;
    if delta > 0.0 {
        while cur < limit {
            values.push(cur);
            cur += delta;
        }
    } else if delta < 0.0 {
        while cur > limit {
            values.push(cur);
            cur += delta;
        }
    }
    values
}

/// Compute f64 range values.
fn range_values_f64(start: f64, limit: f64, delta: f64) -> Vec<f64> {
    let mut values = Vec::new();
    let mut cur = start;
    if delta > 0.0 {
        while cur < limit {
            values.push(cur);
            cur += delta;
        }
    } else if delta < 0.0 {
        while cur > limit {
            values.push(cur);
            cur += delta;
        }
    }
    values
}

/// Evaluate a Slice op at trace time when the data input is a constant.
///
/// Only handles 1-D I64 tensors (shape constants) sliced along axis 0 with
/// step 1. This covers the `Slice(Shape(x), starts, ends)` pattern in GPT-2.
/// Returns `None` if the combination is unsupported.
fn eval_slice_constant(
    data: &Buffer,
    starts: &[i64],
    ends: &[i64],
    axes: &[i64],
    steps: &[i64],
) -> Option<Buffer> {
    // Only handle I64 buffers.
    if data.dtype() != crate::DType::I64 {
        return None;
    }
    // Only handle 1-D case.
    if data.shape().rank() != 1 {
        return None;
    }
    let len = data.shape().0[0] as i64;
    let vals = data.as_slice::<i64>();
    let mut result: Vec<i64> = Vec::new();

    for i in 0..starts.len() {
        let ax = axes[i];
        // Only support axis 0 for 1-D tensors.
        if ax != 0 && ax != -1 {
            return None;
        }
        let step = if i < steps.len() { steps[i] } else { 1 };
        if step != 1 {
            return None;
        }
        // Clamp to valid range per ONNX spec.
        let start = starts[i].clamp(-len, len);
        let start = if start < 0 { start + len } else { start } as usize;
        let end = ends[i].clamp(-len - 1, len);
        let end = if end < 0 { end + len } else { end } as usize;
        let end = end.min(len as usize);
        result.extend_from_slice(&vals[start..end]);
    }

    Some(Buffer::from_slice::<i64>(
        &result,
        &[result.len() as u64],
        crate::DType::I64,
    ))
}

/// Evaluate a Cast op at trace time when the input is a constant.
///
/// Handles integer-to-integer and integer-to-float casts for scalar/1-D
/// constant buffers, enabling downstream Range/Reshape to read the result.
fn eval_cast_constant(input: &Buffer, target: crate::DType) -> Option<Buffer> {
    use crate::DType;
    let shape = &input.shape().0;
    match (input.dtype(), target) {
        // Same dtype — just clone.
        (a, b) if a == b => Some(input.clone()),
        // I32 → I64
        (DType::I32, DType::I64) => {
            let vals: Vec<i64> = input.as_slice::<i32>().iter().map(|&v| v as i64).collect();
            Some(Buffer::from_slice::<i64>(&vals, shape, DType::I64))
        }
        // I64 → I32
        (DType::I64, DType::I32) => {
            let vals: Vec<i32> = input.as_slice::<i64>().iter().map(|&v| v as i32).collect();
            Some(Buffer::from_slice::<i32>(&vals, shape, DType::I32))
        }
        // I64 → F32
        (DType::I64, DType::F32) => {
            let vals: Vec<f32> = input.as_slice::<i64>().iter().map(|&v| v as f32).collect();
            Some(Buffer::from_slice::<f32>(&vals, shape, DType::F32))
        }
        // I32 → F32
        (DType::I32, DType::F32) => {
            let vals: Vec<f32> = input.as_slice::<i32>().iter().map(|&v| v as f32).collect();
            Some(Buffer::from_slice::<f32>(&vals, shape, DType::F32))
        }
        // I64 → F64
        (DType::I64, DType::F64) => {
            let vals: Vec<f64> = input.as_slice::<i64>().iter().map(|&v| v as f64).collect();
            Some(Buffer::from_slice::<f64>(&vals, shape, DType::F64))
        }
        _ => None,
    }
}

/// Evaluate a Gather op at trace time when all inputs are constants.
///
/// Handles the common case: axis=0, scalar I64 index into a 1-D or 2-D I64
/// tensor. Returns `None` if the combination is not supported.
fn eval_gather_constant(data: &Buffer, indices: &Buffer, axis: i64) -> Option<Buffer> {
    // Only handle I64 data — shape tensors are always I64.
    if data.dtype() != crate::DType::I64 || indices.dtype() != crate::DType::I64 {
        return None;
    }
    // Only handle axis=0.
    if axis != 0 {
        return None;
    }
    // Guard: if data contains DIM_DYNAMIC sentinel (i64::MIN), the result would
    // be a literal i64::MIN which would corrupt downstream ops (ConstantOfShape,
    // Range, etc.). Return None so the caller takes the dynamic-trace path.
    if contains_dynamic_sentinel(data) {
        return None;
    }

    let data_vals = data.as_slice::<i64>();
    let data_shape = &data.shape().0;
    let idx_vals = indices.as_slice::<i64>();

    // Scalar index: indices has 0 or 1 elements (rank-0 or rank-1 shape [1]).
    if idx_vals.len() == 1 {
        let idx = idx_vals[0];
        let idx = if idx < 0 {
            (data_shape[0] as i64 + idx) as usize
        } else {
            idx as usize
        };

        if data_shape.len() == 1 {
            // 1-D data, scalar index → scalar result.
            let val = data_vals[idx];
            return Some(Buffer::from_slice::<i64>(&[val], &[1], crate::DType::I64));
        }
        if data_shape.len() == 2 {
            // 2-D data, scalar index → 1-D row.
            let row_len = data_shape[1] as usize;
            let row = &data_vals[idx * row_len..(idx + 1) * row_len];
            return Some(Buffer::from_slice::<i64>(row, &[row_len as u64], crate::DType::I64));
        }
    }

    // Vector index: indices is a 1-D tensor of length N → result is [N, ...].
    if idx_vals.len() > 1 && data_shape.len() == 1 {
        let result: Vec<i64> = idx_vals
            .iter()
            .map(|&i| {
                let i = if i < 0 {
                    (data_shape[0] as i64 + i) as usize
                } else {
                    i as usize
                };
                data_vals[i]
            })
            .collect();
        return Some(Buffer::from_slice::<i64>(
            &result,
            &[result.len() as u64],
            crate::DType::I64,
        ));
    }

    None
}

/// Evaluate a Concat op at trace time when all inputs are I64 constants.
///
/// Handles 1-D I64 tensors (and rank-0 scalars treated as 1-element tensors)
/// concatenated along axis=0. This is the common pattern for building reshape
/// target shapes in GPT-2 graphs (e.g. [batch, seq, 3, 12, 64]).
/// Returns `None` if the combination is unsupported.
fn eval_concat_constant(inputs: &[Buffer], axis: i64) -> Option<Buffer> {
    // Only handle I64 buffers (shape constants).
    if inputs.iter().any(|b| b.dtype() != crate::DType::I64) {
        return None;
    }
    // Only handle axis 0 (shape-building pattern).
    if axis != 0 {
        return None;
    }
    // All inputs must be rank 0 (scalar) or rank 1.
    if inputs.iter().any(|b| b.shape().rank() > 1) {
        return None;
    }

    let mut result: Vec<i64> = Vec::new();
    for buf in inputs {
        result.extend_from_slice(buf.as_slice::<i64>());
    }
    Some(Buffer::from_slice::<i64>(
        &result,
        &[result.len() as u64],
        crate::DType::I64,
    ))
}

// ── General constant folding ──────────────────────────────────────────────────

/// After a node has been mapped (its trace op emitted), check whether all of
/// its inputs are in `constant_data`. If they are, try to evaluate the output
/// buffer eagerly and store it so downstream nodes can use the value at trace
/// time.
///
/// This handles arithmetic ops (Add, Sub, Mul, Div, Pow) and comparison ops
/// (Equal, Less, Greater, ReduceSum, ReduceMax) that appear in GPT-2's shape
/// computation chains. The existing per-op constant propagation in map_node
/// handles "source" ops (Constant, Shape, ConstantOfShape, Range) and
/// reshape-like ops (Squeeze, Unsqueeze, Reshape, Cast, Gather, Slice, Concat)
/// — those are intentionally left in place. This function adds the missing
/// propagation for ops that were previously not constant-folded.
///
/// Skipped if any required input is not in `constant_data` (silent no-op).
fn try_constant_fold(node: &OnnxNode, constant_data: &mut HashMap<String, Buffer>) {
    // Skip if the output is already folded (e.g. by per-op propagation above).
    if node.outputs.is_empty() || constant_data.contains_key(&node.outputs[0]) {
        return;
    }

    // Check that all non-empty inputs are present as constants.
    let all_const = node
        .inputs
        .iter()
        .filter(|name| !name.is_empty())
        .all(|name| constant_data.contains_key(name));

    if !all_const {
        return;
    }

    match node.op_type.as_str() {
        "Add" | "Sub" | "Mul" | "Div" => {
            if node.inputs.len() < 2 {
                return;
            }
            let a = constant_data.get(&node.inputs[0]).cloned().unwrap();
            let b = constant_data.get(&node.inputs[1]).cloned().unwrap();
            if let Some(result) = eval_binary_constant(&a, &b, node.op_type.as_str()) {
                constant_data.insert(node.outputs[0].clone(), result);
            }
        }
        "Pow" => {
            if node.inputs.len() < 2 {
                return;
            }
            let a = constant_data.get(&node.inputs[0]).cloned().unwrap();
            let b = constant_data.get(&node.inputs[1]).cloned().unwrap();
            if let Some(result) = eval_pow_constant(&a, &b) {
                constant_data.insert(node.outputs[0].clone(), result);
            }
        }
        "Equal" | "Less" | "Greater" | "LessOrEqual" | "GreaterOrEqual" => {
            if node.inputs.len() < 2 {
                return;
            }
            let a = constant_data.get(&node.inputs[0]).cloned().unwrap();
            let b = constant_data.get(&node.inputs[1]).cloned().unwrap();
            if let Some(result) = eval_cmp_constant(&a, &b, node.op_type.as_str()) {
                constant_data.insert(node.outputs[0].clone(), result);
            }
        }
        "Where" => {
            if node.inputs.len() < 3 {
                return;
            }
            let cond = constant_data.get(&node.inputs[0]).cloned().unwrap();
            let x = constant_data.get(&node.inputs[1]).cloned().unwrap();
            let y = constant_data.get(&node.inputs[2]).cloned().unwrap();
            if let Some(result) = eval_where_constant(&cond, &x, &y) {
                constant_data.insert(node.outputs[0].clone(), result);
            }
        }
        "ReduceSum" | "ReduceMax" => {
            // Common in shape chains: ReduceSum/ReduceMax over small I64 tensors.
            // axes may be an attribute or the second input.
            let data = constant_data.get(&node.inputs[0]).cloned().unwrap();
            let axes_opt: Option<Vec<i64>> =
                if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
                    constant_data
                        .get(&node.inputs[1])
                        .and_then(|b| read_i64_buffer(b).ok())
                } else if let Some(OnnxAttribute::Ints(v)) = node.attributes.get("axes") {
                    Some(v.clone())
                } else {
                    None // reduce all
                };
            let keepdims = node
                .attributes
                .get("keepdims")
                .and_then(|a| {
                    if let OnnxAttribute::Int(v) = a {
                        Some(*v != 0)
                    } else {
                        None
                    }
                })
                .unwrap_or(true);
            if let Some(result) =
                eval_reduce_constant(&data, axes_opt.as_deref(), keepdims, node.op_type.as_str())
            {
                constant_data.insert(node.outputs[0].clone(), result);
            }
        }
        "Neg" => {
            let a = constant_data.get(&node.inputs[0]).cloned().unwrap();
            if let Some(result) = eval_neg_constant(&a) {
                constant_data.insert(node.outputs[0].clone(), result);
            }
        }
        _ => {} // Not handled here; per-op propagation covers the rest.
    }
}

/// Broadcast two shapes together (NumPy/ONNX semantics).
/// Returns the output shape, or None if shapes are incompatible.
fn broadcast_shapes(a: &[u64], b: &[u64]) -> Option<Vec<u64>> {
    let rank = a.len().max(b.len());
    let mut out = vec![0u64; rank];
    for i in 0..rank {
        let da = if i + a.len() >= rank {
            a[i + a.len() - rank]
        } else {
            1
        };
        let db = if i + b.len() >= rank {
            b[i + b.len() - rank]
        } else {
            1
        };
        if da == db {
            out[i] = da;
        } else if da == 1 {
            out[i] = db;
        } else if db == 1 {
            out[i] = da;
        } else {
            return None; // incompatible
        }
    }
    Some(out)
}

/// For each output element (C-order), compute the flat source index accounting
/// for broadcasting. `src_shape` is the shape of the source buffer (right-aligned
/// against `out_shape`). Returns a Vec of length `product(out_shape)`.
fn broadcast_index_map(src_shape: &[u64], out_shape: &[u64]) -> Vec<usize> {
    // Compute the total number of output elements.
    let total: usize = out_shape.iter().product::<u64>() as usize;
    if total == 0 {
        return vec![];
    }

    let rank = out_shape.len();

    // For each dimension of out_shape, compute the stride in src.
    // src is right-aligned; missing dims are treated as size 1 with stride 0.
    let mut src_strides = vec![0usize; rank];
    {
        let src_rank = src_shape.len();
        // Compute row-major strides for src.
        let mut raw_strides = vec![0usize; src_rank];
        if src_rank > 0 {
            raw_strides[src_rank - 1] = 1;
            for i in (0..src_rank - 1).rev() {
                raw_strides[i] = raw_strides[i + 1] * src_shape[i + 1] as usize;
            }
        }
        for i in 0..rank {
            let src_dim_idx = i + src_rank - rank; // may underflow → treated as broadcast
            if i + src_rank >= rank {
                let d = src_shape[src_dim_idx];
                if d > 1 {
                    src_strides[i] = raw_strides[src_dim_idx];
                } else {
                    src_strides[i] = 0; // broadcast
                }
            }
            // else: src_rank < rank for this dim → stride stays 0 (broadcast)
        }
    }

    // Iterate over all output indices.
    let mut src_indices = vec![0usize; total];
    let mut out_coords = vec![0u64; rank];
    for out_flat in 0..total {
        // Compute src flat index from out_coords and src_strides.
        let mut src_flat = 0usize;
        for i in 0..rank {
            src_flat += out_coords[i] as usize * src_strides[i];
        }
        src_indices[out_flat] = src_flat;

        // Increment out_coords (right-to-left carry).
        for i in (0..rank).rev() {
            out_coords[i] += 1;
            if out_coords[i] < out_shape[i] {
                break;
            }
            out_coords[i] = 0;
        }
    }
    src_indices
}

/// Evaluate a binary element-wise op on constant buffers.
/// Supports I32, I64, F32, F64 with broadcasting.
fn eval_binary_constant(a: &Buffer, b: &Buffer, op: &str) -> Option<Buffer> {
    use crate::DType;
    if a.dtype() != b.dtype() {
        return None;
    }
    let out_shape = broadcast_shapes(&a.shape().0, &b.shape().0)?;

    let a_idx = broadcast_index_map(&a.shape().0, &out_shape);
    let b_idx = broadcast_index_map(&b.shape().0, &out_shape);
    let total = out_shape.iter().product::<u64>() as usize;

    match a.dtype() {
        DType::I64 => {
            let av = a.as_slice::<i64>();
            let bv = b.as_slice::<i64>();
            let result: Vec<i64> = (0..total)
                .map(|i| {
                    let x = av[a_idx[i]];
                    let y = bv[b_idx[i]];
                    match op {
                        "Add" => x + y,
                        "Sub" => x - y,
                        "Mul" => x * y,
                        "Div" => x / y,
                        _ => unreachable!(),
                    }
                })
                .collect();
            Some(Buffer::from_slice::<i64>(&result, &out_shape, DType::I64))
        }
        DType::I32 => {
            let av = a.as_slice::<i32>();
            let bv = b.as_slice::<i32>();
            let result: Vec<i32> = (0..total)
                .map(|i| {
                    let x = av[a_idx[i]];
                    let y = bv[b_idx[i]];
                    match op {
                        "Add" => x + y,
                        "Sub" => x - y,
                        "Mul" => x * y,
                        "Div" => x / y,
                        _ => unreachable!(),
                    }
                })
                .collect();
            Some(Buffer::from_slice::<i32>(&result, &out_shape, DType::I32))
        }
        DType::F32 => {
            let av = a.as_slice::<f32>();
            let bv = b.as_slice::<f32>();
            let result: Vec<f32> = (0..total)
                .map(|i| {
                    let x = av[a_idx[i]];
                    let y = bv[b_idx[i]];
                    match op {
                        "Add" => x + y,
                        "Sub" => x - y,
                        "Mul" => x * y,
                        "Div" => x / y,
                        _ => unreachable!(),
                    }
                })
                .collect();
            Some(Buffer::from_slice::<f32>(&result, &out_shape, DType::F32))
        }
        DType::F64 => {
            let av = a.as_slice::<f64>();
            let bv = b.as_slice::<f64>();
            let result: Vec<f64> = (0..total)
                .map(|i| {
                    let x = av[a_idx[i]];
                    let y = bv[b_idx[i]];
                    match op {
                        "Add" => x + y,
                        "Sub" => x - y,
                        "Mul" => x * y,
                        "Div" => x / y,
                        _ => unreachable!(),
                    }
                })
                .collect();
            Some(Buffer::from_slice::<f64>(&result, &out_shape, DType::F64))
        }
    }
}

/// Evaluate Pow at trace time for constant buffers.
fn eval_pow_constant(base: &Buffer, exp: &Buffer) -> Option<Buffer> {
    use crate::DType;
    if base.dtype() != exp.dtype() {
        return None;
    }
    let out_shape = broadcast_shapes(&base.shape().0, &exp.shape().0)?;
    let a_idx = broadcast_index_map(&base.shape().0, &out_shape);
    let b_idx = broadcast_index_map(&exp.shape().0, &out_shape);
    let total = out_shape.iter().product::<u64>() as usize;
    match base.dtype() {
        DType::F32 => {
            let av = base.as_slice::<f32>();
            let bv = exp.as_slice::<f32>();
            let result: Vec<f32> = (0..total).map(|i| av[a_idx[i]].powf(bv[b_idx[i]])).collect();
            Some(Buffer::from_slice::<f32>(&result, &out_shape, DType::F32))
        }
        DType::F64 => {
            let av = base.as_slice::<f64>();
            let bv = exp.as_slice::<f64>();
            let result: Vec<f64> = (0..total).map(|i| av[a_idx[i]].powf(bv[b_idx[i]])).collect();
            Some(Buffer::from_slice::<f64>(&result, &out_shape, DType::F64))
        }
        DType::I64 => {
            let av = base.as_slice::<i64>();
            let bv = exp.as_slice::<i64>();
            let result: Vec<i64> = (0..total)
                .map(|i| {
                    let e = bv[b_idx[i]];
                    if e >= 0 {
                        av[a_idx[i]].pow(e as u32)
                    } else {
                        0 // integer pow with negative exp → 0 (truncation)
                    }
                })
                .collect();
            Some(Buffer::from_slice::<i64>(&result, &out_shape, DType::I64))
        }
        DType::I32 => {
            let av = base.as_slice::<i32>();
            let bv = exp.as_slice::<i32>();
            let result: Vec<i32> = (0..total)
                .map(|i| {
                    let e = bv[b_idx[i]];
                    if e >= 0 {
                        av[a_idx[i]].pow(e as u32)
                    } else {
                        0
                    }
                })
                .collect();
            Some(Buffer::from_slice::<i32>(&result, &out_shape, DType::I32))
        }
    }
}

/// Evaluate comparison ops (Equal, Less, Greater, LessOrEqual, GreaterOrEqual)
/// at trace time. Output is I64 (0 or 1), matching the Bool→I64 mapping.
fn eval_cmp_constant(a: &Buffer, b: &Buffer, op: &str) -> Option<Buffer> {
    use crate::DType;
    if a.dtype() != b.dtype() {
        return None;
    }
    let out_shape = broadcast_shapes(&a.shape().0, &b.shape().0)?;
    let a_idx = broadcast_index_map(&a.shape().0, &out_shape);
    let b_idx = broadcast_index_map(&b.shape().0, &out_shape);
    let total = out_shape.iter().product::<u64>() as usize;

    // Convert all dtypes to I64 for comparison; output is always I64.
    let av: Vec<i64> = match a.dtype() {
        DType::I64 => a.as_slice::<i64>().to_vec(),
        DType::I32 => a.as_slice::<i32>().iter().map(|&v| v as i64).collect(),
        DType::F32 => a.as_slice::<f32>().iter().map(|&v| v as i64).collect(),
        DType::F64 => a.as_slice::<f64>().iter().map(|&v| v as i64).collect(),
    };
    let bv: Vec<i64> = match b.dtype() {
        DType::I64 => b.as_slice::<i64>().to_vec(),
        DType::I32 => b.as_slice::<i32>().iter().map(|&v| v as i64).collect(),
        DType::F32 => b.as_slice::<f32>().iter().map(|&v| v as i64).collect(),
        DType::F64 => b.as_slice::<f64>().iter().map(|&v| v as i64).collect(),
    };

    let result: Vec<i64> = (0..total)
        .map(|i| {
            let x = av[a_idx[i]];
            let y = bv[b_idx[i]];
            let cond = match op {
                "Equal" => x == y,
                "Less" => x < y,
                "Greater" => x > y,
                "LessOrEqual" => x <= y,
                "GreaterOrEqual" => x >= y,
                _ => unreachable!(),
            };
            if cond { 1i64 } else { 0i64 }
        })
        .collect();
    Some(Buffer::from_slice::<i64>(&result, &out_shape, DType::I64))
}

/// Evaluate Where at trace time when all three inputs are constants.
/// Condition is treated as I64 (non-zero = true), x and y must share dtype.
fn eval_where_constant(cond: &Buffer, x: &Buffer, y: &Buffer) -> Option<Buffer> {
    use crate::DType;
    if x.dtype() != y.dtype() {
        return None;
    }
    // Broadcast all three shapes together.
    let cx_shape = broadcast_shapes(&cond.shape().0, &x.shape().0)?;
    let out_shape = broadcast_shapes(&cx_shape, &y.shape().0)?;

    let c_idx = broadcast_index_map(&cond.shape().0, &out_shape);
    let x_idx = broadcast_index_map(&x.shape().0, &out_shape);
    let y_idx = broadcast_index_map(&y.shape().0, &out_shape);
    let total = out_shape.iter().product::<u64>() as usize;

    // Read condition as i64 (truthy = non-zero).
    let cond_vals: Vec<i64> = match cond.dtype() {
        DType::I64 => cond.as_slice::<i64>().to_vec(),
        DType::I32 => cond.as_slice::<i32>().iter().map(|&v| v as i64).collect(),
        _ => return None,
    };

    match x.dtype() {
        DType::I64 => {
            let xv = x.as_slice::<i64>();
            let yv = y.as_slice::<i64>();
            let result: Vec<i64> = (0..total)
                .map(|i| if cond_vals[c_idx[i]] != 0 { xv[x_idx[i]] } else { yv[y_idx[i]] })
                .collect();
            Some(Buffer::from_slice::<i64>(&result, &out_shape, DType::I64))
        }
        DType::I32 => {
            let xv = x.as_slice::<i32>();
            let yv = y.as_slice::<i32>();
            let result: Vec<i32> = (0..total)
                .map(|i| if cond_vals[c_idx[i]] != 0 { xv[x_idx[i]] } else { yv[y_idx[i]] })
                .collect();
            Some(Buffer::from_slice::<i32>(&result, &out_shape, DType::I32))
        }
        DType::F32 => {
            let xv = x.as_slice::<f32>();
            let yv = y.as_slice::<f32>();
            let result: Vec<f32> = (0..total)
                .map(|i| if cond_vals[c_idx[i]] != 0 { xv[x_idx[i]] } else { yv[y_idx[i]] })
                .collect();
            Some(Buffer::from_slice::<f32>(&result, &out_shape, DType::F32))
        }
        DType::F64 => {
            let xv = x.as_slice::<f64>();
            let yv = y.as_slice::<f64>();
            let result: Vec<f64> = (0..total)
                .map(|i| if cond_vals[c_idx[i]] != 0 { xv[x_idx[i]] } else { yv[y_idx[i]] })
                .collect();
            Some(Buffer::from_slice::<f64>(&result, &out_shape, DType::F64))
        }
    }
}

/// Evaluate ReduceSum or ReduceMax at trace time for constant I64/I32 buffers.
/// Only handles 1-D tensors (the common shape-chain case).
fn eval_reduce_constant(
    data: &Buffer,
    axes: Option<&[i64]>,
    keepdims: bool,
    op: &str,
) -> Option<Buffer> {
    use crate::DType;
    // Only handle 1-D tensors for simplicity (shape chains are always 1-D).
    if data.shape().rank() != 1 {
        return None;
    }
    // Normalize axes: None = reduce all.
    let normalized: Vec<i64> = match axes {
        None => vec![0],
        Some(ax) => ax
            .iter()
            .map(|&a| if a < 0 { a + 1 } else { a }) // rank=1
            .collect(),
    };
    // For a 1-D tensor, axis must be 0.
    if normalized != [0] {
        return None;
    }

    let out_shape: Vec<u64> = if keepdims { vec![1] } else { vec![] };

    match data.dtype() {
        DType::I64 => {
            let vals = data.as_slice::<i64>();
            let val = match op {
                "ReduceSum" => vals.iter().sum::<i64>(),
                "ReduceMax" => *vals.iter().max()?,
                _ => unreachable!(),
            };
            let result = vec![val];
            Some(Buffer::from_slice::<i64>(&result, &out_shape, DType::I64))
        }
        DType::I32 => {
            let vals = data.as_slice::<i32>();
            let val = match op {
                "ReduceSum" => vals.iter().sum::<i32>(),
                "ReduceMax" => *vals.iter().max()?,
                _ => unreachable!(),
            };
            let result = vec![val];
            Some(Buffer::from_slice::<i32>(&result, &out_shape, DType::I32))
        }
        DType::F32 => {
            let vals = data.as_slice::<f32>();
            let val = match op {
                "ReduceSum" => vals.iter().sum::<f32>(),
                "ReduceMax" => vals
                    .iter()
                    .cloned()
                    .reduce(f32::max)?,
                _ => unreachable!(),
            };
            let result = vec![val];
            Some(Buffer::from_slice::<f32>(&result, &out_shape, DType::F32))
        }
        _ => None,
    }
}

/// Evaluate Neg at trace time for constant buffers.
fn eval_neg_constant(a: &Buffer) -> Option<Buffer> {
    use crate::DType;
    match a.dtype() {
        DType::I64 => {
            let vals: Vec<i64> = a.as_slice::<i64>().iter().map(|&v| -v).collect();
            Some(Buffer::from_slice::<i64>(&vals, &a.shape().0, DType::I64))
        }
        DType::I32 => {
            let vals: Vec<i32> = a.as_slice::<i32>().iter().map(|&v| -v).collect();
            Some(Buffer::from_slice::<i32>(&vals, &a.shape().0, DType::I32))
        }
        DType::F32 => {
            let vals: Vec<f32> = a.as_slice::<f32>().iter().map(|&v| -v).collect();
            Some(Buffer::from_slice::<f32>(&vals, &a.shape().0, DType::F32))
        }
        DType::F64 => {
            let vals: Vec<f64> = a.as_slice::<f64>().iter().map(|&v| -v).collect();
            Some(Buffer::from_slice::<f64>(&vals, &a.shape().0, DType::F64))
        }
    }
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
        use crate::onnx::parser::Dim;
        DynamicInput {
            name: name.to_string(),
            dims: shape.iter().map(|&v| Dim::Fixed(v)).collect(),
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

    // ── Regression tests for code review fixes ───────────────────────────────

    #[test]
    fn map_conv_grouped_returns_error() {
        let model = OnnxModel {
            nodes: vec![make_node(
                "Conv",
                &["X", "W"],
                &["Y"],
                vec![
                    ("kernel_shape", OnnxAttribute::Ints(vec![3, 3])),
                    ("group", OnnxAttribute::Int(2)),
                ],
            )],
            initializers: vec![make_weight("W", &[1.0; 18], &[2, 1, 3, 3])],
            dynamic_inputs: vec![make_dynamic("X", &[1, 2, 5, 5], DType::F32)],
            outputs: vec!["Y".to_string()],
        };
        let result = map_graph(&model);
        assert!(result.is_err(), "grouped conv should return error");
        let msg = result.err().unwrap().to_string();
        assert!(
            msg.contains("group=2"),
            "error should mention group, got: {msg}"
        );
    }

    fn make_i32_weight(name: &str, data: &[i32], shape: &[u64]) -> (String, Buffer) {
        let buf = Buffer::from_slice::<i32>(data, shape, DType::I32);
        (name.to_string(), buf)
    }

    #[test]
    fn map_reshape_with_i32_shape_tensor() {
        let model = OnnxModel {
            nodes: vec![make_node("Reshape", &["X", "shape"], &["Y"], vec![])],
            initializers: vec![make_i32_weight("shape", &[3, 4], &[2])],
            dynamic_inputs: vec![make_dynamic("X", &[2, 6], DType::F32)],
            outputs: vec!["Y".to_string()],
        };
        let (trace, output_ids, _) = map_graph(&model).expect("i32 shape should work");
        let out = trace.get(output_ids[0]);
        assert_eq!(out.shape().0, vec![3, 4]);
    }

    #[test]
    fn map_flatten_axis_zero() {
        let model = OnnxModel {
            nodes: vec![make_node(
                "Flatten",
                &["X"],
                &["Y"],
                vec![("axis", OnnxAttribute::Int(0))],
            )],
            initializers: vec![],
            dynamic_inputs: vec![make_dynamic("X", &[2, 3, 4], DType::F32)],
            outputs: vec!["Y".to_string()],
        };
        let (trace, output_ids, _) = map_graph(&model).expect("map_graph failed");
        let out = trace.get(output_ids[0]);
        // axis=0: [1, 2*3*4] = [1, 24]
        assert_eq!(out.shape().0, vec![1, 24]);
    }

    // ── Task 16.1/16.3: BatchNormalization mapping ──────────────────────────

    #[test]
    fn map_batch_normalization() {
        let c = 3u64;
        let ones = vec![1.0f32; c as usize];
        let zeros = vec![0.0f32; c as usize];
        let model = OnnxModel {
            nodes: vec![make_node(
                "BatchNormalization",
                &["X", "scale", "bias", "mean", "var"],
                &["Y"],
                vec![("epsilon", OnnxAttribute::Float(1e-5))],
            )],
            initializers: vec![
                make_weight("scale", &ones, &[c]),
                make_weight("bias", &zeros, &[c]),
                make_weight("mean", &zeros, &[c]),
                make_weight("var", &ones, &[c]),
            ],
            dynamic_inputs: vec![make_dynamic("X", &[1, c, 4, 4], DType::F32)],
            outputs: vec!["Y".to_string()],
        };
        let (trace, output_ids, weights) = map_graph(&model).expect("map_graph failed");
        assert_eq!(weights.len(), 4);
        let out = trace.get(output_ids[0]);
        assert!(
            matches!(out, Op::BatchNorm { .. }),
            "expected Op::BatchNorm, got {:?}",
            out
        );
        // Output shape matches input
        assert_eq!(out.shape().0, vec![1, c, 4, 4]);
    }

    #[test]
    fn map_batch_normalization_default_epsilon() {
        let c = 2u64;
        let ones = vec![1.0f32; c as usize];
        let zeros = vec![0.0f32; c as usize];
        let model = OnnxModel {
            nodes: vec![make_node(
                "BatchNormalization",
                &["X", "scale", "bias", "mean", "var"],
                &["Y"],
                vec![], // no epsilon attr → default 1e-5
            )],
            initializers: vec![
                make_weight("scale", &ones, &[c]),
                make_weight("bias", &zeros, &[c]),
                make_weight("mean", &zeros, &[c]),
                make_weight("var", &ones, &[c]),
            ],
            dynamic_inputs: vec![make_dynamic("X", &[1, c, 2, 2], DType::F32)],
            outputs: vec!["Y".to_string()],
        };
        let (trace, output_ids, _) = map_graph(&model).expect("map_graph failed");
        let out = trace.get(output_ids[0]);
        assert_eq!(out.shape().0, vec![1, c, 2, 2]);
    }

    // ── Task 16.2/16.3: GlobalAveragePool mapping ───────────────────────────

    #[test]
    fn map_global_average_pool() {
        let model = OnnxModel {
            nodes: vec![make_node("GlobalAveragePool", &["X"], &["Y"], vec![])],
            initializers: vec![],
            dynamic_inputs: vec![make_dynamic("X", &[1, 3, 7, 7], DType::F32)],
            outputs: vec!["Y".to_string()],
        };
        let (trace, output_ids, _) = map_graph(&model).expect("map_graph failed");
        let out = trace.get(output_ids[0]);
        assert!(
            matches!(out, Op::GlobalAvgPool { .. }),
            "expected Op::GlobalAvgPool, got {:?}",
            out
        );
        assert_eq!(out.shape().0, vec![1, 3, 1, 1]);
    }

    // ── Task 3.7: tests for mapper-only (trace-time) constant ops ─────────────

    // ── 3.1 Constant ──────────────────────────────────────────────────────────

    #[test]
    fn constant_node_produces_tensor_input() {
        // Constant node with a float value tensor [1.0, 2.0, 3.0].
        let buf = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3], DType::F32);
        let model = OnnxModel {
            nodes: vec![make_node(
                "Constant",
                &[],
                &["c"],
                vec![("value", OnnxAttribute::Tensor(buf))],
            )],
            initializers: vec![],
            dynamic_inputs: vec![],
            outputs: vec!["c".to_string()],
        };

        let (trace, output_ids, weights) = map_graph(&model).expect("map_graph failed");

        // One Input op registered, one weight appended.
        assert_eq!(weights.len(), 1);
        assert_eq!(trace.input_count(), 1);
        assert_eq!(weights[0].as_slice::<f32>(), &[1.0f32, 2.0, 3.0]);
        assert_eq!(weights[0].shape(), &Shape(vec![3]));

        // Output node must be the Input op for the constant.
        assert!(
            matches!(trace.get(output_ids[0]), Op::Input { .. }),
            "expected Op::Input, got {:?}",
            trace.get(output_ids[0])
        );
    }

    #[test]
    fn constant_node_missing_value_attr_returns_error() {
        let model = OnnxModel {
            nodes: vec![make_node("Constant", &[], &["c"], vec![])],
            initializers: vec![],
            dynamic_inputs: vec![],
            outputs: vec!["c".to_string()],
        };
        let result = map_graph(&model);
        assert!(matches!(result, Err(OnnxError::UnsupportedOp(_))));
    }

    // ── 3.2 Shape ─────────────────────────────────────────────────────────────

    #[test]
    fn shape_node_returns_i64_shape_buffer() {
        // Shape([X: shape [2, 3, 4]]) → I64 tensor [2, 3, 4].
        let model = OnnxModel {
            nodes: vec![make_node("Shape", &["X"], &["shape_out"], vec![])],
            initializers: vec![],
            dynamic_inputs: vec![make_dynamic("X", &[2, 3, 4], DType::F32)],
            outputs: vec!["shape_out".to_string()],
        };

        let (trace, output_ids, weights) = map_graph(&model).expect("map_graph failed");

        // 1 dynamic input + 1 constant (shape) = 2 Input ops; 1 weight.
        assert_eq!(weights.len(), 1);
        assert_eq!(trace.input_count(), 2);

        // The shape buffer must be I64 [2, 3, 4].
        assert_eq!(weights[0].dtype(), DType::I64);
        assert_eq!(weights[0].as_slice::<i64>(), &[2i64, 3, 4]);
        assert_eq!(weights[0].shape(), &Shape(vec![3]));

        // Output op is an Input (the constant weight).
        assert!(
            matches!(trace.get(output_ids[0]), Op::Input { .. }),
            "expected Op::Input for Shape output, got {:?}",
            trace.get(output_ids[0])
        );
    }

    // ── 3.3 ConstantOfShape ───────────────────────────────────────────────────

    #[test]
    fn constant_of_shape_default_zeros() {
        // ConstantOfShape with shape [2, 3] and no value attribute → F32 zeros.
        //
        // Graph: shape_const → ConstantOfShape → out
        // shape_const is an I64 initializer [2, 3].
        let shape_buf = Buffer::from_slice::<i64>(&[2, 3], &[2], DType::I64);
        let model = OnnxModel {
            nodes: vec![make_node(
                "ConstantOfShape",
                &["shape_const"],
                &["out"],
                vec![],
            )],
            initializers: vec![("shape_const".to_string(), shape_buf)],
            dynamic_inputs: vec![],
            outputs: vec!["out".to_string()],
        };

        let (_trace, _output_ids, weights) = map_graph(&model).expect("map_graph failed");

        // weights[0] = shape_const initializer, weights[1] = the ConstantOfShape output.
        assert_eq!(weights.len(), 2);
        let out_buf = &weights[1];
        assert_eq!(out_buf.dtype(), DType::F32);
        assert_eq!(out_buf.shape(), &Shape(vec![2, 3]));
        assert_eq!(out_buf.as_slice::<f32>(), &[0.0f32; 6]);
    }

    #[test]
    fn constant_of_shape_with_fill_value() {
        // ConstantOfShape with shape [4] and value = 7.0 f32.
        let shape_buf = Buffer::from_slice::<i64>(&[4], &[1], DType::I64);
        let fill_buf = Buffer::from_slice::<f32>(&[7.0], &[1], DType::F32);
        let model = OnnxModel {
            nodes: vec![make_node(
                "ConstantOfShape",
                &["shape_const"],
                &["out"],
                vec![("value", OnnxAttribute::Tensor(fill_buf))],
            )],
            initializers: vec![("shape_const".to_string(), shape_buf)],
            dynamic_inputs: vec![],
            outputs: vec!["out".to_string()],
        };

        let (_trace, _output_ids, weights) = map_graph(&model).expect("map_graph failed");

        let out_buf = &weights[1];
        assert_eq!(out_buf.dtype(), DType::F32);
        assert_eq!(out_buf.shape(), &Shape(vec![4]));
        assert_eq!(out_buf.as_slice::<f32>(), &[7.0f32, 7.0, 7.0, 7.0]);
    }

    // ── 3.4 Range ─────────────────────────────────────────────────────────────

    #[test]
    fn range_node_i64() {
        // Range(0, 5, 1) → [0, 1, 2, 3, 4].
        let start = Buffer::from_slice::<i64>(&[0], &[1], DType::I64);
        let limit = Buffer::from_slice::<i64>(&[5], &[1], DType::I64);
        let delta = Buffer::from_slice::<i64>(&[1], &[1], DType::I64);
        let model = OnnxModel {
            nodes: vec![make_node(
                "Range",
                &["start", "limit", "delta"],
                &["out"],
                vec![],
            )],
            initializers: vec![
                ("start".to_string(), start),
                ("limit".to_string(), limit),
                ("delta".to_string(), delta),
            ],
            dynamic_inputs: vec![],
            outputs: vec!["out".to_string()],
        };

        let (_trace, _output_ids, weights) = map_graph(&model).expect("map_graph failed");

        // 3 initializer weights + 1 range output.
        assert_eq!(weights.len(), 4);
        let out_buf = &weights[3];
        assert_eq!(out_buf.dtype(), DType::I64);
        assert_eq!(out_buf.as_slice::<i64>(), &[0i64, 1, 2, 3, 4]);
    }

    #[test]
    fn range_node_f32() {
        // Range(0.0, 3.0, 0.5) → [0.0, 0.5, 1.0, 1.5, 2.0, 2.5].
        let start = Buffer::from_slice::<f32>(&[0.0], &[1], DType::F32);
        let limit = Buffer::from_slice::<f32>(&[3.0], &[1], DType::F32);
        let delta = Buffer::from_slice::<f32>(&[0.5], &[1], DType::F32);
        let model = OnnxModel {
            nodes: vec![make_node(
                "Range",
                &["start", "limit", "delta"],
                &["out"],
                vec![],
            )],
            initializers: vec![
                ("start".to_string(), start),
                ("limit".to_string(), limit),
                ("delta".to_string(), delta),
            ],
            dynamic_inputs: vec![],
            outputs: vec!["out".to_string()],
        };

        let (_trace, _output_ids, weights) = map_graph(&model).expect("map_graph failed");

        let out_buf = &weights[3];
        assert_eq!(out_buf.dtype(), DType::F32);
        let values = out_buf.as_slice::<f32>();
        assert_eq!(values.len(), 6);
        for (got, expected) in values.iter().zip([0.0f32, 0.5, 1.0, 1.5, 2.0, 2.5].iter()) {
            assert!((got - expected).abs() < 1e-6, "got {got}, expected {expected}");
        }
    }

    // ── 3.5 Squeeze ───────────────────────────────────────────────────────────

    #[test]
    fn squeeze_removes_specified_axes() {
        // Input [1, 3, 1, 4], squeeze axes [0, 2] → [3, 4].
        // Axes provided as an attribute (older opset style).
        let model = OnnxModel {
            nodes: vec![make_node(
                "Squeeze",
                &["X"],
                &["Y"],
                vec![("axes", OnnxAttribute::Ints(vec![0, 2]))],
            )],
            initializers: vec![],
            dynamic_inputs: vec![make_dynamic("X", &[1, 3, 1, 4], DType::F32)],
            outputs: vec!["Y".to_string()],
        };

        let (trace, output_ids, _) = map_graph(&model).expect("map_graph failed");
        let out = trace.get(output_ids[0]);
        assert_eq!(out.shape().0, vec![3, 4]);
    }

    #[test]
    fn squeeze_no_axes_removes_all_size_one_dims() {
        // Input [1, 2, 1, 3], no axes → [2, 3].
        let model = OnnxModel {
            nodes: vec![make_node("Squeeze", &["X"], &["Y"], vec![])],
            initializers: vec![],
            dynamic_inputs: vec![make_dynamic("X", &[1, 2, 1, 3], DType::F32)],
            outputs: vec!["Y".to_string()],
        };

        let (trace, output_ids, _) = map_graph(&model).expect("map_graph failed");
        let out = trace.get(output_ids[0]);
        assert_eq!(out.shape().0, vec![2, 3]);
    }

    #[test]
    fn squeeze_axes_as_input_tensor() {
        // Input [1, 3, 1, 4], axes = [0, 2] as input tensor (opset 13+ style).
        let axes_buf = Buffer::from_slice::<i64>(&[0, 2], &[2], DType::I64);
        let model = OnnxModel {
            nodes: vec![make_node("Squeeze", &["X", "axes"], &["Y"], vec![])],
            initializers: vec![("axes".to_string(), axes_buf)],
            dynamic_inputs: vec![make_dynamic("X", &[1, 3, 1, 4], DType::F32)],
            outputs: vec!["Y".to_string()],
        };

        let (trace, output_ids, _) = map_graph(&model).expect("map_graph failed");
        let out = trace.get(output_ids[0]);
        assert_eq!(out.shape().0, vec![3, 4]);
    }

    // ── 3.6 Unsqueeze ─────────────────────────────────────────────────────────

    #[test]
    fn unsqueeze_inserts_dims_at_axes() {
        // Input [3, 4], unsqueeze axes [0, 2] → [1, 3, 1, 4].
        let model = OnnxModel {
            nodes: vec![make_node(
                "Unsqueeze",
                &["X"],
                &["Y"],
                vec![("axes", OnnxAttribute::Ints(vec![0, 2]))],
            )],
            initializers: vec![],
            dynamic_inputs: vec![make_dynamic("X", &[3, 4], DType::F32)],
            outputs: vec!["Y".to_string()],
        };

        let (trace, output_ids, _) = map_graph(&model).expect("map_graph failed");
        let out = trace.get(output_ids[0]);
        assert_eq!(out.shape().0, vec![1, 3, 1, 4]);
    }

    #[test]
    fn unsqueeze_negative_axis() {
        // Input [3, 4], unsqueeze axis [-1] → [3, 4, 1].
        // Output rank = 3, axis -1 → 3 - 1 = 2.
        let model = OnnxModel {
            nodes: vec![make_node(
                "Unsqueeze",
                &["X"],
                &["Y"],
                vec![("axes", OnnxAttribute::Ints(vec![-1]))],
            )],
            initializers: vec![],
            dynamic_inputs: vec![make_dynamic("X", &[3, 4], DType::F32)],
            outputs: vec!["Y".to_string()],
        };

        let (trace, output_ids, _) = map_graph(&model).expect("map_graph failed");
        let out = trace.get(output_ids[0]);
        assert_eq!(out.shape().0, vec![3, 4, 1]);
    }

    #[test]
    fn unsqueeze_axes_as_input_tensor() {
        // Input [5], axes = [0] as input tensor → [1, 5].
        let axes_buf = Buffer::from_slice::<i64>(&[0], &[1], DType::I64);
        let model = OnnxModel {
            nodes: vec![make_node("Unsqueeze", &["X", "axes"], &["Y"], vec![])],
            initializers: vec![("axes".to_string(), axes_buf)],
            dynamic_inputs: vec![make_dynamic("X", &[5], DType::F32)],
            outputs: vec!["Y".to_string()],
        };

        let (trace, output_ids, _) = map_graph(&model).expect("map_graph failed");
        let out = trace.get(output_ids[0]);
        assert_eq!(out.shape().0, vec![1, 5]);
    }

    // ── 3.8 Constant propagation: Shape → ConstantOfShape chain ──────────────

    #[test]
    fn shape_feeds_constant_of_shape() {
        // Graph: X [2, 4] → Shape → shape_out → ConstantOfShape → out
        // ConstantOfShape should produce a [2, 4] F32 zeros tensor.
        let model = OnnxModel {
            nodes: vec![
                make_node("Shape", &["X"], &["shape_out"], vec![]),
                make_node("ConstantOfShape", &["shape_out"], &["out"], vec![]),
            ],
            initializers: vec![],
            dynamic_inputs: vec![make_dynamic("X", &[2, 4], DType::F32)],
            outputs: vec!["out".to_string()],
        };

        let (_trace, _output_ids, weights) = map_graph(&model).expect("map_graph failed");

        // weights[0] = shape_out (I64 [2, 4]), weights[1] = out (F32 zeros [2, 4]).
        assert_eq!(weights.len(), 2);
        let shape_w = &weights[0];
        assert_eq!(shape_w.dtype(), DType::I64);
        assert_eq!(shape_w.as_slice::<i64>(), &[2i64, 4]);

        let out_w = &weights[1];
        assert_eq!(out_w.dtype(), DType::F32);
        assert_eq!(out_w.shape(), &Shape(vec![2, 4]));
        assert_eq!(out_w.as_slice::<f32>(), &[0.0f32; 8]);
    }

    // ── Task 4.5: Pow, Sqrt, Cast, ReduceMean mapper tests ────────────────────

    #[test]
    fn map_pow_graph() {
        let model = OnnxModel {
            nodes: vec![make_node("Pow", &["X", "Y"], &["Z"], vec![])],
            initializers: vec![],
            dynamic_inputs: vec![
                make_dynamic("X", &[4], DType::F32),
                make_dynamic("Y", &[4], DType::F32),
            ],
            outputs: vec!["Z".to_string()],
        };

        let (trace, output_ids, _) = map_graph(&model).expect("map_graph failed");
        assert!(
            matches!(trace.get(output_ids[0]), Op::Pow { .. }),
            "expected Op::Pow, got {:?}",
            trace.get(output_ids[0])
        );
    }

    #[test]
    fn map_sqrt_graph() {
        let model = OnnxModel {
            nodes: vec![make_node("Sqrt", &["X"], &["Y"], vec![])],
            initializers: vec![],
            dynamic_inputs: vec![make_dynamic("X", &[4], DType::F32)],
            outputs: vec!["Y".to_string()],
        };

        let (trace, output_ids, _) = map_graph(&model).expect("map_graph failed");
        assert!(
            matches!(trace.get(output_ids[0]), Op::Sqrt { .. }),
            "expected Op::Sqrt, got {:?}",
            trace.get(output_ids[0])
        );
    }

    #[test]
    fn map_cast_i64_to_f32() {
        // ONNX dtype 7 = INT64, dtype 1 = FLOAT.
        let model = OnnxModel {
            nodes: vec![make_node(
                "Cast",
                &["X"],
                &["Y"],
                vec![("to", OnnxAttribute::Int(1))],
            )],
            initializers: vec![],
            dynamic_inputs: vec![make_dynamic("X", &[3], DType::I64)],
            outputs: vec!["Y".to_string()],
        };

        let (trace, output_ids, _) = map_graph(&model).expect("map_graph failed");
        let out_op = trace.get(output_ids[0]);
        match out_op {
            Op::Cast {
                target_dtype,
                dtype,
                ..
            } => {
                assert_eq!(*target_dtype, DType::F32);
                assert_eq!(*dtype, DType::F32);
            }
            other => panic!("expected Op::Cast, got {:?}", other),
        }
    }

    #[test]
    fn map_cast_same_dtype_is_input() {
        // Cast to same dtype → no Cast op, output is the input tensor directly.
        let model = OnnxModel {
            nodes: vec![make_node(
                "Cast",
                &["X"],
                &["Y"],
                vec![("to", OnnxAttribute::Int(1))], // FLOAT
            )],
            initializers: vec![],
            dynamic_inputs: vec![make_dynamic("X", &[3], DType::F32)],
            outputs: vec!["Y".to_string()],
        };

        let (trace, output_ids, _) = map_graph(&model).expect("map_graph failed");
        // Only 1 op: the Input op (Cast was a no-op).
        assert_eq!(trace.ops().len(), 1, "expected only Input op, no Cast");
        assert!(
            matches!(trace.get(output_ids[0]), Op::Input { .. }),
            "expected Op::Input (no-op cast)"
        );
    }

    #[test]
    fn map_reduce_mean_with_axes_attr() {
        let model = OnnxModel {
            nodes: vec![make_node(
                "ReduceMean",
                &["X"],
                &["Y"],
                vec![
                    ("axes", OnnxAttribute::Ints(vec![1])),
                    ("keepdims", OnnxAttribute::Int(0)),
                ],
            )],
            initializers: vec![],
            dynamic_inputs: vec![make_dynamic("X", &[2, 4], DType::F32)],
            outputs: vec!["Y".to_string()],
        };

        let (trace, output_ids, _) = map_graph(&model).expect("map_graph failed");
        let out_op = trace.get(output_ids[0]);
        match out_op {
            Op::ReduceMean {
                axes,
                keepdim,
                shape,
                ..
            } => {
                assert_eq!(*axes, vec![1i64]);
                assert!(!keepdim);
                assert_eq!(shape.0, vec![2u64]);
            }
            other => panic!("expected Op::ReduceMean, got {:?}", other),
        }
    }

    #[test]
    fn map_reduce_mean_keepdims_default_true() {
        // No keepdims attribute → default is 1 (keepdim=true).
        let model = OnnxModel {
            nodes: vec![make_node(
                "ReduceMean",
                &["X"],
                &["Y"],
                vec![("axes", OnnxAttribute::Ints(vec![0]))],
            )],
            initializers: vec![],
            dynamic_inputs: vec![make_dynamic("X", &[3, 4], DType::F32)],
            outputs: vec!["Y".to_string()],
        };

        let (trace, output_ids, _) = map_graph(&model).expect("map_graph failed");
        let out_op = trace.get(output_ids[0]);
        match out_op {
            Op::ReduceMean { keepdim, shape, .. } => {
                assert!(*keepdim, "expected keepdim=true by default");
                assert_eq!(shape.0, vec![1u64, 4]);
            }
            other => panic!("expected Op::ReduceMean, got {:?}", other),
        }
    }

    // ── Gather mapper tests ───────────────────────────────────────────────────

    #[test]
    fn map_gather_axis0() {
        let model = OnnxModel {
            nodes: vec![make_node(
                "Gather",
                &["data", "indices"],
                &["out"],
                vec![("axis", OnnxAttribute::Int(0))],
            )],
            initializers: vec![],
            dynamic_inputs: vec![
                make_dynamic("data", &[10, 4], DType::F32),
                make_dynamic("indices", &[3], DType::I64),
            ],
            outputs: vec!["out".to_string()],
        };
        let (trace, output_ids, _) = map_graph(&model).expect("map_graph failed");
        let out_op = trace.get(output_ids[0]);
        match out_op {
            Op::Gather { axis, shape, dtype, .. } => {
                assert_eq!(*axis, 0);
                assert_eq!(shape.0, vec![3u64, 4]);
                assert_eq!(*dtype, DType::F32);
            }
            other => panic!("expected Op::Gather, got {:?}", other),
        }
    }

    #[test]
    fn map_gather_default_axis() {
        // No axis attribute → default axis=0.
        let model = OnnxModel {
            nodes: vec![make_node("Gather", &["data", "idx"], &["out"], vec![])],
            initializers: vec![],
            dynamic_inputs: vec![
                make_dynamic("data", &[5, 3], DType::F32),
                make_dynamic("idx", &[2], DType::I64),
            ],
            outputs: vec!["out".to_string()],
        };
        let (trace, output_ids, _) = map_graph(&model).expect("map_graph failed");
        match trace.get(output_ids[0]) {
            Op::Gather { axis, shape, .. } => {
                assert_eq!(*axis, 0);
                assert_eq!(shape.0, vec![2u64, 3]);
            }
            other => panic!("expected Op::Gather, got {:?}", other),
        }
    }

    // ── Slice mapper tests ────────────────────────────────────────────────────

    #[test]
    fn map_slice_with_constant_inputs() {
        // Slice opset 10+: starts/ends as input tensors in constant_data.
        let starts_buf = Buffer::from_slice::<i64>(&[1], &[1], DType::I64);
        let ends_buf = Buffer::from_slice::<i64>(&[4], &[1], DType::I64);

        let model = OnnxModel {
            nodes: vec![make_node(
                "Slice",
                &["data", "starts", "ends"],
                &["out"],
                vec![],
            )],
            initializers: vec![
                ("starts".to_string(), starts_buf),
                ("ends".to_string(), ends_buf),
            ],
            dynamic_inputs: vec![make_dynamic("data", &[6], DType::F32)],
            outputs: vec!["out".to_string()],
        };
        let (trace, output_ids, _) = map_graph(&model).expect("map_graph failed");
        match trace.get(output_ids[0]) {
            Op::Slice { starts, ends, shape, .. } => {
                assert_eq!(*starts, vec![1i64]);
                assert_eq!(*ends, vec![4i64]);
                assert_eq!(shape.0, vec![3u64]);
            }
            other => panic!("expected Op::Slice, got {:?}", other),
        }
    }

    // ── Concat mapper tests ───────────────────────────────────────────────────

    #[test]
    fn map_concat_two_inputs() {
        let model = OnnxModel {
            nodes: vec![make_node(
                "Concat",
                &["A", "B"],
                &["out"],
                vec![("axis", OnnxAttribute::Int(0))],
            )],
            initializers: vec![],
            dynamic_inputs: vec![
                make_dynamic("A", &[3, 4], DType::F32),
                make_dynamic("B", &[5, 4], DType::F32),
            ],
            outputs: vec!["out".to_string()],
        };
        let (trace, output_ids, _) = map_graph(&model).expect("map_graph failed");
        match trace.get(output_ids[0]) {
            Op::Concat { axis, shape, dtype, .. } => {
                assert_eq!(*axis, 0);
                assert_eq!(shape.0, vec![8u64, 4]);
                assert_eq!(*dtype, DType::F32);
            }
            other => panic!("expected Op::Concat, got {:?}", other),
        }
    }

    // ── Split mapper tests ────────────────────────────────────────────────────

    #[test]
    fn map_split_equal_chunks() {
        // Split a [6] tensor into 3 equal [2] tensors.
        let model = OnnxModel {
            nodes: vec![make_node(
                "Split",
                &["X"],
                &["a", "b", "c"],
                vec![("axis", OnnxAttribute::Int(0))],
            )],
            initializers: vec![],
            dynamic_inputs: vec![make_dynamic("X", &[6], DType::F32)],
            outputs: vec!["a".to_string(), "b".to_string(), "c".to_string()],
        };
        let (trace, output_ids, _) = map_graph(&model).expect("map_graph failed");
        assert_eq!(output_ids.len(), 3);
        for oid in &output_ids {
            match trace.get(*oid) {
                Op::Slice { shape, .. } => {
                    assert_eq!(shape.0, vec![2u64]);
                }
                other => panic!("expected Op::Slice (from Split), got {:?}", other),
            }
        }
    }

    // ── Transpose mapper tests ────────────────────────────────────────────────

    #[test]
    fn map_transpose_with_perm() {
        let model = OnnxModel {
            nodes: vec![make_node(
                "Transpose",
                &["X"],
                &["Y"],
                vec![("perm", OnnxAttribute::Ints(vec![1, 0]))],
            )],
            initializers: vec![],
            dynamic_inputs: vec![make_dynamic("X", &[3, 5], DType::F32)],
            outputs: vec!["Y".to_string()],
        };
        let (trace, output_ids, _) = map_graph(&model).expect("map_graph failed");
        match trace.get(output_ids[0]) {
            Op::Transpose { perm, shape, .. } => {
                assert_eq!(*perm, vec![1i64, 0]);
                assert_eq!(shape.0, vec![5u64, 3]);
            }
            other => panic!("expected Op::Transpose, got {:?}", other),
        }
    }

    #[test]
    fn map_transpose_default_perm_reversal() {
        // No perm → reverse dims.
        let model = OnnxModel {
            nodes: vec![make_node("Transpose", &["X"], &["Y"], vec![])],
            initializers: vec![],
            dynamic_inputs: vec![make_dynamic("X", &[2, 3, 4], DType::F32)],
            outputs: vec!["Y".to_string()],
        };
        let (trace, output_ids, _) = map_graph(&model).expect("map_graph failed");
        match trace.get(output_ids[0]) {
            Op::Transpose { perm, shape, .. } => {
                assert_eq!(*perm, vec![2i64, 1, 0]);
                assert_eq!(shape.0, vec![4u64, 3, 2]);
            }
            other => panic!("expected Op::Transpose, got {:?}", other),
        }
    }

    // ── Where mapper tests ────────────────────────────────────────────────────

    #[test]
    fn map_where_i64_condition() {
        // Where with I64 condition (already correct dtype).
        let model = OnnxModel {
            nodes: vec![make_node("Where", &["cond", "X", "Y"], &["out"], vec![])],
            initializers: vec![],
            dynamic_inputs: vec![
                make_dynamic("cond", &[4], DType::I64),
                make_dynamic("X", &[4], DType::F32),
                make_dynamic("Y", &[4], DType::F32),
            ],
            outputs: vec!["out".to_string()],
        };
        let (trace, output_ids, _) = map_graph(&model).expect("map_graph failed");
        match trace.get(output_ids[0]) {
            Op::Where { shape, dtype, .. } => {
                assert_eq!(shape.0, vec![4u64]);
                assert_eq!(*dtype, DType::F32);
            }
            other => panic!("expected Op::Where, got {:?}", other),
        }
    }

    #[test]
    fn map_where_i32_condition_cast() {
        // Where with I32 condition: mapper should cast to I64 before Where.
        let model = OnnxModel {
            nodes: vec![make_node("Where", &["cond", "X", "Y"], &["out"], vec![])],
            initializers: vec![],
            dynamic_inputs: vec![
                make_dynamic("cond", &[3], DType::I32),
                make_dynamic("X", &[3], DType::F32),
                make_dynamic("Y", &[3], DType::F32),
            ],
            outputs: vec!["out".to_string()],
        };
        let (trace, output_ids, _) = map_graph(&model).expect("map_graph failed");
        // Output should be Op::Where with F32 dtype.
        let result = match trace.get(output_ids[0]) {
            Op::Where { dtype, shape, .. } => {
                assert_eq!(*dtype, DType::F32);
                assert_eq!(shape.0, vec![3u64]);
                true
            }
            _ => false,
        };
        assert!(result, "expected Op::Where");
    }
}
