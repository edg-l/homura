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
}
