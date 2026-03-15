pub mod mapper;
pub mod parser;
pub mod proto;

use std::path::Path;

use crate::{
    Compiler,
    runtime::{Buffer, CompiledGraph},
};
use parser::OnnxError;

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model")
            .field("num_dynamic_inputs", &self.num_dynamic_inputs)
            .field("num_weights", &self.weights.len())
            .finish_non_exhaustive()
    }
}

/// A compiled ONNX model ready for repeated inference.
///
/// Weights are stored alongside the compiled graph. Each call to `run`
/// prepends the caller-supplied dynamic inputs to the stored weights and
/// invokes the JIT function.
pub struct Model {
    compiled: CompiledGraph,
    weights: Vec<Buffer>,
    num_dynamic_inputs: usize,
}

impl Model {
    /// Load and compile an ONNX model from a file path.
    pub fn load(path: impl AsRef<Path>) -> Result<Model, OnnxError> {
        let onnx_model = parser::parse_model(path)?;
        Model::from_onnx(onnx_model)
    }

    /// Load and compile an ONNX model from raw protobuf bytes.
    ///
    /// Useful in tests where models are built in memory.
    pub fn load_bytes(bytes: &[u8]) -> Result<Model, OnnxError> {
        let onnx_model = parser::parse_bytes(bytes)?;
        Model::from_onnx(onnx_model)
    }

    fn from_onnx(onnx_model: parser::OnnxModel) -> Result<Model, OnnxError> {
        if onnx_model.outputs.len() > 1 {
            return Err(OnnxError::MultipleOutputs(onnx_model.outputs.len()));
        }
        let num_dynamic = onnx_model.dynamic_inputs.len();
        let (trace, output_ids, weights) = mapper::map_graph(&onnx_model)?;

        let compiled = Compiler::compile(&trace, &output_ids)
            .map_err(|e| OnnxError::CompileError(e.to_string()))?;

        Ok(Model {
            compiled,
            weights,
            num_dynamic_inputs: num_dynamic,
        })
    }

    /// Run inference with the given dynamic inputs.
    ///
    /// `inputs` must have exactly `num_dynamic_inputs` entries.
    /// The weights stored in the model are appended automatically.
    pub fn run(&self, inputs: &[&Buffer]) -> Result<Buffer, OnnxError> {
        if inputs.len() != self.num_dynamic_inputs {
            return Err(OnnxError::WrongInputCount {
                expected: self.num_dynamic_inputs,
                got: inputs.len(),
            });
        }

        let mut all_args: Vec<&Buffer> = Vec::with_capacity(inputs.len() + self.weights.len());
        all_args.extend_from_slice(inputs);
        for w in &self.weights {
            all_args.push(w);
        }

        let output = self.compiled.run(&all_args);
        Ok(output)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, onnx::proto::*};
    use prost::Message;

    // ── CNN model builder (Task 13.1) ─────────────────────────────────────────

    /// Build a small CNN ONNX model via protobuf:
    ///
    /// Input [1,1,8,8]
    ///   → Conv(kernel=[1,1,3,3], pad=1) → [1,1,8,8]
    ///   → Relu → [1,1,8,8]
    ///   → MaxPool(kernel=2×2, stride=2) → [1,1,4,4]
    ///   → Reshape([1,16]) → [1,16]
    ///   → MatMul(W:[16,4]) → [1,4]
    ///   → Add(bias:[4]) → [1,4]
    ///
    /// All-ones conv kernel: each output pixel = sum of 3×3 neighbourhood.
    /// All-ones FC weight: each of the 4 outputs = sum over the 16 spatial values.
    /// Zero FC bias: output = FC output unchanged.
    fn make_cnn_model_bytes() -> Vec<u8> {
        // Conv kernel: [1,1,3,3] all-ones (CO=1, CI=1, KH=3, KW=3).
        let conv_kernel: Vec<f32> = vec![1.0f32; 1 * 1 * 3 * 3];
        let conv_kernel_raw: Vec<u8> = conv_kernel.iter().flat_map(|v| v.to_le_bytes()).collect();

        // Reshape target shape: [1, 16] encoded as I64.
        let reshape_shape: Vec<i64> = vec![1, 16];
        let reshape_shape_raw: Vec<u8> =
            reshape_shape.iter().flat_map(|v| v.to_le_bytes()).collect();

        // FC weight: [16, 4] all-ones.
        let fc_weight: Vec<f32> = vec![1.0f32; 16 * 4];
        let fc_weight_raw: Vec<u8> = fc_weight.iter().flat_map(|v| v.to_le_bytes()).collect();

        // FC bias: [4] all-zeros.
        let fc_bias: Vec<f32> = vec![0.0f32; 4];
        let fc_bias_raw: Vec<u8> = fc_bias.iter().flat_map(|v| v.to_le_bytes()).collect();

        encode(&ModelProto {
            ir_version: 8,
            graph: Some(GraphProto {
                node: vec![
                    // Conv: X, conv_w → conv_out
                    NodeProto {
                        op_type: "Conv".into(),
                        input: vec!["X".into(), "conv_w".into()],
                        output: vec!["conv_out".into()],
                        attribute: vec![
                            AttributeProto {
                                name: "kernel_shape".into(),
                                r#type: 7, // INTS
                                ints: vec![3, 3],
                                ..Default::default()
                            },
                            AttributeProto {
                                name: "pads".into(),
                                r#type: 7, // INTS
                                ints: vec![1, 1, 1, 1],
                                ..Default::default()
                            },
                            AttributeProto {
                                name: "strides".into(),
                                r#type: 7, // INTS
                                ints: vec![1, 1],
                                ..Default::default()
                            },
                        ],
                        ..Default::default()
                    },
                    // Relu: conv_out → relu_out
                    NodeProto {
                        op_type: "Relu".into(),
                        input: vec!["conv_out".into()],
                        output: vec!["relu_out".into()],
                        ..Default::default()
                    },
                    // MaxPool: relu_out → pool_out
                    NodeProto {
                        op_type: "MaxPool".into(),
                        input: vec!["relu_out".into()],
                        output: vec!["pool_out".into()],
                        attribute: vec![
                            AttributeProto {
                                name: "kernel_shape".into(),
                                r#type: 7, // INTS
                                ints: vec![2, 2],
                                ..Default::default()
                            },
                            AttributeProto {
                                name: "strides".into(),
                                r#type: 7, // INTS
                                ints: vec![2, 2],
                                ..Default::default()
                            },
                        ],
                        ..Default::default()
                    },
                    // Reshape: pool_out, reshape_shape → flat_out
                    NodeProto {
                        op_type: "Reshape".into(),
                        input: vec!["pool_out".into(), "reshape_shape".into()],
                        output: vec!["flat_out".into()],
                        ..Default::default()
                    },
                    // MatMul: flat_out, fc_w → matmul_out
                    NodeProto {
                        op_type: "MatMul".into(),
                        input: vec!["flat_out".into(), "fc_w".into()],
                        output: vec!["matmul_out".into()],
                        ..Default::default()
                    },
                    // Add: matmul_out, fc_b → output
                    NodeProto {
                        op_type: "Add".into(),
                        input: vec!["matmul_out".into(), "fc_b".into()],
                        output: vec!["output".into()],
                        ..Default::default()
                    },
                ],
                initializer: vec![
                    TensorProto {
                        name: "conv_w".into(),
                        dims: vec![1, 1, 3, 3],
                        data_type: 1, // FLOAT
                        raw_data: conv_kernel_raw,
                        ..Default::default()
                    },
                    TensorProto {
                        name: "reshape_shape".into(),
                        dims: vec![2],
                        data_type: 7, // INT64
                        raw_data: reshape_shape_raw,
                        ..Default::default()
                    },
                    TensorProto {
                        name: "fc_w".into(),
                        dims: vec![16, 4],
                        data_type: 1, // FLOAT
                        raw_data: fc_weight_raw,
                        ..Default::default()
                    },
                    TensorProto {
                        name: "fc_b".into(),
                        dims: vec![4],
                        data_type: 1, // FLOAT
                        raw_data: fc_bias_raw,
                        ..Default::default()
                    },
                ],
                input: vec![
                    value_info("X", &[1, 1, 8, 8]),
                    value_info("conv_w", &[1, 1, 3, 3]),
                    value_info("reshape_shape", &[2]),
                    value_info("fc_w", &[16, 4]),
                    value_info("fc_b", &[4]),
                ],
                output: vec![value_info("output", &[1, 4])],
                ..Default::default()
            }),
            ..Default::default()
        })
    }

    // ── Task 13.2 + 13.3: CNN pipeline integration test ───────────────────────

    /// Full CNN pipeline: Conv → Relu → MaxPool → Reshape → MatMul → Add.
    ///
    /// With all-1.0 input and all-ones conv kernel (pad=1):
    ///   - Each conv output pixel = sum of its 3×3 neighbourhood.
    ///   - MaxPool 2×2 stride 2 always picks the maximum of each 2×2 block.
    ///     Every 2×2 block in the 8×8 feature map contains at least one
    ///     interior pixel (value 9.0), so all 16 pool outputs = 9.0.
    ///   - FC weight is all-ones [16,4]: each output = sum(16 × 9.0) = 144.0.
    ///   - Zero bias: final output = [144.0; 4].
    #[test]
    fn load_and_run_cnn_pipeline() {
        let bytes = make_cnn_model_bytes();
        let model = Model::load_bytes(&bytes).expect("load failed");

        let input_data: Vec<f32> = vec![1.0f32; 1 * 1 * 8 * 8];
        let input = Buffer::from_slice::<f32>(&input_data, &[1, 1, 8, 8], DType::F32);
        let output = model.run(&[&input]).expect("run failed");

        let out = output.as_slice::<f32>();

        // Task 13.3: verify shape and values.
        assert_eq!(out.len(), 4, "expected 4 output values");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "output contains NaN or Inf: {out:?}"
        );
        assert!(
            out.iter().all(|v| *v > 0.0),
            "expected all-positive output for all-ones input: {out:?}"
        );
        // Each output neuron = sum over 16 pool values (each 9.0) × all-ones weight = 144.0.
        for (i, &v) in out.iter().enumerate() {
            assert!(
                (v - 144.0f32).abs() < 1e-3,
                "output[{i}] = {v}, expected ~144.0"
            );
        }
    }

    // ── Task 13.1 (optional): real mnist-12.onnx test ─────────────────────────

    #[test]
    fn run_real_mnist_model_if_available() {
        let path = "tests/fixtures/mnist-12.onnx";
        if !std::path::Path::new(path).exists() {
            eprintln!("skipping: {path} not found (download with scripts/download_mnist.sh)");
            return;
        }
        let model = Model::load(path).expect("load failed");
        let input = Buffer::from_slice::<f32>(&vec![0.0f32; 784], &[1, 1, 28, 28], DType::F32);
        let output = model.run(&[&input]).expect("run failed");
        let out = output.as_slice::<f32>();
        assert_eq!(out.len(), 10, "expected 10 output logits");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "output contains NaN or Inf: {out:?}"
        );
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn encode(model: &ModelProto) -> Vec<u8> {
        let mut buf = Vec::new();
        model.encode(&mut buf).unwrap();
        buf
    }

    fn fixed_shape_f32(dims: &[i64]) -> TypeProto {
        use crate::onnx::proto::tensor_shape_proto::Dimension;
        use crate::onnx::proto::type_proto;
        TypeProto {
            value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                elem_type: 1, // FLOAT
                shape: Some(TensorShapeProto {
                    dim: dims
                        .iter()
                        .map(|&d| Dimension {
                            value: Some(
                                crate::onnx::proto::tensor_shape_proto::dimension::Value::DimValue(
                                    d,
                                ),
                            ),
                            ..Default::default()
                        })
                        .collect(),
                }),
            })),
            ..Default::default()
        }
    }

    fn value_info(name: &str, dims: &[i64]) -> ValueInfoProto {
        ValueInfoProto {
            name: name.into(),
            r#type: Some(fixed_shape_f32(dims)),
            ..Default::default()
        }
    }

    /// Build a minimal Add(X, Y) -> Z ONNX model with two f32[4] dynamic inputs.
    fn make_add_model_bytes() -> Vec<u8> {
        encode(&ModelProto {
            ir_version: 8,
            graph: Some(GraphProto {
                node: vec![NodeProto {
                    op_type: "Add".into(),
                    input: vec!["X".into(), "Y".into()],
                    output: vec!["Z".into()],
                    ..Default::default()
                }],
                input: vec![value_info("X", &[4]), value_info("Y", &[4])],
                output: vec![value_info("Z", &[4])],
                ..Default::default()
            }),
            ..Default::default()
        })
    }

    /// Build an Add(X, W) -> Z model where W is a static weight (f32[4]).
    fn make_add_with_weight_bytes(weight_data: &[f32]) -> Vec<u8> {
        let raw: Vec<u8> = weight_data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let dims: Vec<i64> = vec![weight_data.len() as i64];

        encode(&ModelProto {
            ir_version: 8,
            graph: Some(GraphProto {
                node: vec![NodeProto {
                    op_type: "Add".into(),
                    input: vec!["X".into(), "W".into()],
                    output: vec!["Z".into()],
                    ..Default::default()
                }],
                initializer: vec![TensorProto {
                    name: "W".into(),
                    dims: dims.clone(),
                    data_type: 1, // FLOAT
                    raw_data: raw,
                    ..Default::default()
                }],
                // W also listed in graph.input for opset compat — parser must skip it.
                input: vec![value_info("X", &dims), value_info("W", &dims)],
                output: vec![value_info("Z", &dims)],
                ..Default::default()
            }),
            ..Default::default()
        })
    }

    // ── Task 8.4: load/run round-trip ─────────────────────────────────────────

    #[test]
    fn load_and_run_add_two_dynamic_inputs() {
        let bytes = make_add_model_bytes();
        let model = Model::load_bytes(&bytes).expect("load_bytes failed");

        let x = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[4], DType::F32);
        let y = Buffer::from_slice::<f32>(&[10.0, 20.0, 30.0, 40.0], &[4], DType::F32);
        let out = model.run(&[&x, &y]).expect("run failed");

        assert_eq!(out.as_slice::<f32>(), &[11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn load_and_run_add_with_weight() {
        let weights = [1.0f32, 2.0, 3.0, 4.0];
        let bytes = make_add_with_weight_bytes(&weights);
        let model = Model::load_bytes(&bytes).expect("load_bytes failed");

        // Only 1 dynamic input — the weight is baked in.
        let x = Buffer::from_slice::<f32>(&[10.0, 20.0, 30.0, 40.0], &[4], DType::F32);
        let out = model.run(&[&x]).expect("run failed");

        assert_eq!(out.as_slice::<f32>(), &[11.0, 22.0, 33.0, 44.0]);
    }

    // ── Task 8.5: error cases ──────────────────────────────────────────────────

    #[test]
    fn load_nonexistent_file_returns_io_error() {
        let result = Model::load("/tmp/does_not_exist_homura_test.onnx");
        assert!(
            matches!(result, Err(OnnxError::Io(_))),
            "expected Io error, got: {:?}",
            result
        );
    }

    #[test]
    fn run_wrong_input_count_returns_error() {
        let bytes = make_add_model_bytes();
        let model = Model::load_bytes(&bytes).expect("load_bytes failed");

        // Model expects 2 inputs; pass 0.
        let result = model.run(&[]);
        match result {
            Err(OnnxError::WrongInputCount {
                expected: 2,
                got: 0,
            }) => {}
            other => panic!("expected WrongInputCount{{2, 0}}, got: {:?}", other),
        }
    }

    #[test]
    fn run_too_many_inputs_returns_error() {
        let bytes = make_add_model_bytes();
        let model = Model::load_bytes(&bytes).expect("load_bytes failed");

        let x = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[4], DType::F32);
        let y = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[4], DType::F32);
        let z = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[4], DType::F32);
        let result = model.run(&[&x, &y, &z]);
        match result {
            Err(OnnxError::WrongInputCount {
                expected: 2,
                got: 3,
            }) => {}
            other => panic!("expected WrongInputCount{{2, 3}}, got: {:?}", other),
        }
    }

    // ── Task 8.6: multiple run() calls — weights not mutated ──────────────────

    #[test]
    fn multiple_runs_produce_correct_results() {
        let bytes = make_add_model_bytes();
        let model = Model::load_bytes(&bytes).expect("load_bytes failed");

        let x1 = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[4], DType::F32);
        let y1 = Buffer::from_slice::<f32>(&[10.0, 20.0, 30.0, 40.0], &[4], DType::F32);
        let out1 = model.run(&[&x1, &y1]).expect("first run failed");
        assert_eq!(out1.as_slice::<f32>(), &[11.0, 22.0, 33.0, 44.0]);

        let x2 = Buffer::from_slice::<f32>(&[5.0, 6.0, 7.0, 8.0], &[4], DType::F32);
        let y2 = Buffer::from_slice::<f32>(&[50.0, 60.0, 70.0, 80.0], &[4], DType::F32);
        let out2 = model.run(&[&x2, &y2]).expect("second run failed");
        assert_eq!(out2.as_slice::<f32>(), &[55.0, 66.0, 77.0, 88.0]);

        // First result is still correct — no aliasing between runs.
        assert_eq!(out1.as_slice::<f32>(), &[11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn multiple_runs_with_weight_not_mutated() {
        let weights = [0.5f32, 1.0, 1.5, 2.0];
        let bytes = make_add_with_weight_bytes(&weights);
        let model = Model::load_bytes(&bytes).expect("load_bytes failed");

        let x1 = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[4], DType::F32);
        let out1 = model.run(&[&x1]).expect("first run failed");
        assert_eq!(out1.as_slice::<f32>(), &[1.5, 3.0, 4.5, 6.0]);

        // Run again with different input; weight should be unchanged.
        let x2 = Buffer::from_slice::<f32>(&[10.0, 10.0, 10.0, 10.0], &[4], DType::F32);
        let out2 = model.run(&[&x2]).expect("second run failed");
        assert_eq!(out2.as_slice::<f32>(), &[10.5, 11.0, 11.5, 12.0]);

        // Confirm first result is independent.
        assert_eq!(out1.as_slice::<f32>(), &[1.5, 3.0, 4.5, 6.0]);
    }

    // ── Regression: Issue 9 — model with 2 outputs returns MultipleOutputs ────

    #[test]
    fn model_with_two_outputs_returns_error() {
        let bytes = encode(&ModelProto {
            ir_version: 8,
            graph: Some(GraphProto {
                node: vec![NodeProto {
                    op_type: "Add".into(),
                    input: vec!["X".into(), "Y".into()],
                    output: vec!["Z".into()],
                    ..Default::default()
                }],
                input: vec![value_info("X", &[4]), value_info("Y", &[4])],
                // Two outputs: Z and X (both valid edges, but >1 output unsupported).
                output: vec![value_info("Z", &[4]), value_info("X", &[4])],
                ..Default::default()
            }),
            ..Default::default()
        });

        let result = Model::load_bytes(&bytes);
        match result {
            Err(OnnxError::MultipleOutputs(2)) => {}
            Err(other) => panic!("expected MultipleOutputs(2), got Err({other})"),
            Ok(_) => panic!("expected Err for 2-output model, got Ok"),
        }
    }

    // ── Multi-op round-trip tests ────────────────────────────────────────────

    #[test]
    fn load_and_run_relu() {
        let bytes = encode(&ModelProto {
            ir_version: 8,
            graph: Some(GraphProto {
                node: vec![NodeProto {
                    op_type: "Relu".into(),
                    input: vec!["X".into()],
                    output: vec!["Y".into()],
                    ..Default::default()
                }],
                input: vec![value_info("X", &[4])],
                output: vec![value_info("Y", &[4])],
                ..Default::default()
            }),
            ..Default::default()
        });
        let model = Model::load_bytes(&bytes).expect("load failed");
        let x = Buffer::from_slice::<f32>(&[-1.0, 0.0, 3.0, -4.0], &[4], DType::F32);
        let result = model.run(&[&x]).unwrap();
        assert_eq!(result.as_slice::<f32>(), &[0.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn load_and_run_add_relu_chain() {
        let bytes = encode(&ModelProto {
            ir_version: 8,
            graph: Some(GraphProto {
                node: vec![
                    NodeProto {
                        op_type: "Add".into(),
                        input: vec!["X".into(), "Y".into()],
                        output: vec!["sum".into()],
                        ..Default::default()
                    },
                    NodeProto {
                        op_type: "Relu".into(),
                        input: vec!["sum".into()],
                        output: vec!["Z".into()],
                        ..Default::default()
                    },
                ],
                input: vec![value_info("X", &[4]), value_info("Y", &[4])],
                output: vec![value_info("Z", &[4])],
                ..Default::default()
            }),
            ..Default::default()
        });
        let model = Model::load_bytes(&bytes).expect("load failed");
        let x = Buffer::from_slice::<f32>(&[1.0, -2.0, 3.0, -4.0], &[4], DType::F32);
        let y = Buffer::from_slice::<f32>(&[5.0, 6.0, -7.0, 8.0], &[4], DType::F32);
        let result = model.run(&[&x, &y]).unwrap();
        // add=[6,4,-4,4], relu=[6,4,0,4]
        assert_eq!(result.as_slice::<f32>(), &[6.0, 4.0, 0.0, 4.0]);
    }

    #[test]
    fn load_and_run_two_weight_chain() {
        // input → Add(W1) → Add(W2) → output
        let w1_raw: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let w2_raw: Vec<u8> = [10.0f32, 20.0, 30.0, 40.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let bytes = encode(&ModelProto {
            ir_version: 8,
            graph: Some(GraphProto {
                node: vec![
                    NodeProto {
                        op_type: "Add".into(),
                        input: vec!["X".into(), "W1".into()],
                        output: vec!["mid".into()],
                        ..Default::default()
                    },
                    NodeProto {
                        op_type: "Add".into(),
                        input: vec!["mid".into(), "W2".into()],
                        output: vec!["Z".into()],
                        ..Default::default()
                    },
                ],
                initializer: vec![
                    TensorProto {
                        name: "W1".into(),
                        dims: vec![4],
                        data_type: 1,
                        raw_data: w1_raw,
                        ..Default::default()
                    },
                    TensorProto {
                        name: "W2".into(),
                        dims: vec![4],
                        data_type: 1,
                        raw_data: w2_raw,
                        ..Default::default()
                    },
                ],
                input: vec![
                    value_info("X", &[4]),
                    value_info("W1", &[4]),
                    value_info("W2", &[4]),
                ],
                output: vec![value_info("Z", &[4])],
                ..Default::default()
            }),
            ..Default::default()
        });
        let model = Model::load_bytes(&bytes).expect("load failed");
        let x = Buffer::from_slice::<f32>(&[100.0, 200.0, 300.0, 400.0], &[4], DType::F32);
        let result = model.run(&[&x]).unwrap();
        // x + w1 + w2 = [111, 222, 333, 444]
        assert_eq!(result.as_slice::<f32>(), &[111.0, 222.0, 333.0, 444.0]);
    }
}
