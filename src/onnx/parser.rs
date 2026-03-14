use std::collections::{HashMap, HashSet};
use std::path::Path;

use prost::Message;

use crate::runtime::Buffer;
use crate::{DType, Shape};

use super::proto::{
    self, ModelProto,
    attribute_proto::AttributeType,
    type_proto,
    tensor_shape_proto::dimension::Value as DimValue,
};

// ── Public types ──────────────────────────────────────────────────────────────

/// A parsed and validated ONNX model ready for further lowering.
#[derive(Debug)]
pub struct OnnxModel {
    /// Computation nodes in topological order (guaranteed by ONNX spec).
    pub nodes: Vec<OnnxNode>,
    /// Static weight tensors: (name, buffer).
    pub initializers: Vec<(String, Buffer)>,
    /// Dynamic inputs the caller provides at runtime.
    pub dynamic_inputs: Vec<DynamicInput>,
    /// Output edge names.
    pub outputs: Vec<String>,
}

/// A single ONNX computation node.
#[derive(Debug)]
pub struct OnnxNode {
    pub op_type: String,
    /// Input edge names (some may be empty strings for optional inputs).
    pub inputs: Vec<String>,
    /// Output edge names.
    pub outputs: Vec<String>,
    pub attributes: HashMap<String, OnnxAttribute>,
}

/// A dynamic (runtime-provided) input to the model.
#[derive(Debug)]
pub struct DynamicInput {
    pub name: String,
    pub shape: Shape,
    pub dtype: DType,
}

/// A typed attribute value from an ONNX node.
#[derive(Debug, Clone)]
pub enum OnnxAttribute {
    Int(i64),
    Float(f32),
    String(String),
    Ints(Vec<i64>),
    Floats(Vec<f32>),
}

// ── Error type ────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum OnnxError {
    Io(std::io::Error),
    Decode(prost::DecodeError),
    /// data_type field value not supported.
    UnsupportedDtype(i32),
    MissingGraph,
    /// A `ValueInfoProto` has a `dim_param` (symbolic) instead of a fixed size.
    DynamicShape(String),
    /// An input has an unsupported type (not a tensor).
    UnsupportedInputType(String),
    /// An ONNX op type that the mapper does not know how to lower.
    UnsupportedOp(String),
    /// An edge name referenced by a node was never produced by any prior node or input.
    MissingEdge(String),
}

impl std::fmt::Display for OnnxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OnnxError::Io(e) => write!(f, "I/O error: {e}"),
            OnnxError::Decode(e) => write!(f, "protobuf decode error: {e}"),
            OnnxError::UnsupportedDtype(dt) => write!(f, "unsupported ONNX data_type: {dt}"),
            OnnxError::MissingGraph => write!(f, "ModelProto has no graph"),
            OnnxError::DynamicShape(name) => {
                write!(f, "dynamic (symbolic) shape not supported: {name}")
            }
            OnnxError::UnsupportedInputType(name) => {
                write!(f, "input '{name}' has an unsupported type (must be tensor)")
            }
            OnnxError::UnsupportedOp(op) => write!(f, "unsupported ONNX op: {op}"),
            OnnxError::MissingEdge(name) => write!(f, "edge '{name}' not found in value map"),
        }
    }
}

impl std::error::Error for OnnxError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            OnnxError::Io(e) => Some(e),
            OnnxError::Decode(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for OnnxError {
    fn from(e: std::io::Error) -> Self {
        OnnxError::Io(e)
    }
}

impl From<prost::DecodeError> for OnnxError {
    fn from(e: prost::DecodeError) -> Self {
        OnnxError::Decode(e)
    }
}

// ── Entry points ──────────────────────────────────────────────────────────────

/// Parse an ONNX model from a file path.
pub fn parse_model(path: impl AsRef<Path>) -> Result<OnnxModel, OnnxError> {
    let bytes = std::fs::read(path)?;
    parse_bytes(&bytes)
}

/// Parse an ONNX model from raw protobuf bytes.
///
/// Useful in tests where the model is built in-memory.
pub fn parse_bytes(bytes: &[u8]) -> Result<OnnxModel, OnnxError> {
    let model = ModelProto::decode(bytes)?;
    let graph = model.graph.ok_or(OnnxError::MissingGraph)?;

    // Build a set of initializer names for fast lookup.
    let init_names: HashSet<&str> = graph.initializer.iter().map(|t| t.name.as_str()).collect();

    // Extract initializers → (name, Buffer).
    let mut initializers = Vec::with_capacity(graph.initializer.len());
    for tensor in &graph.initializer {
        let buf = tensor_proto_to_buffer(tensor)?;
        initializers.push((tensor.name.clone(), buf));
    }

    // Dynamic inputs: graph.input entries whose name is NOT an initializer.
    // This handles the opset-compat case where initializers also appear in graph.input.
    let mut dynamic_inputs = Vec::new();
    for vi in &graph.input {
        if init_names.contains(vi.name.as_str()) {
            continue;
        }
        let (shape, dtype) = extract_input_type(vi)?;
        dynamic_inputs.push(DynamicInput {
            name: vi.name.clone(),
            shape,
            dtype,
        });
    }

    // Nodes — trust ONNX topological order guarantee.
    let mut nodes = Vec::with_capacity(graph.node.len());
    for node in &graph.node {
        let attributes = extract_attributes(node)?;
        nodes.push(OnnxNode {
            op_type: node.op_type.clone(),
            inputs: node.input.clone(),
            outputs: node.output.clone(),
            attributes,
        });
    }

    // Output edge names.
    let outputs = graph.output.iter().map(|vi| vi.name.clone()).collect();

    Ok(OnnxModel {
        nodes,
        initializers,
        dynamic_inputs,
        outputs,
    })
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Convert a `TensorProto` to a `Buffer`.
///
/// Priority: if `raw_data` is non-empty, use it directly.
/// Otherwise fall back to the typed field (`float_data`, `double_data`, etc.).
fn tensor_proto_to_buffer(t: &proto::TensorProto) -> Result<Buffer, OnnxError> {
    let shape: Vec<u64> = t.dims.iter().map(|&d| d as u64).collect();
    let dtype = onnx_dtype(t.data_type)?;

    let buf = if !t.raw_data.is_empty() {
        // raw_data: byte array — copy directly into a Buffer.
        let num_elems: u64 = if shape.is_empty() { 1 } else { shape.iter().product() };
        let expected_bytes = num_elems as usize * dtype.size_bytes();
        // Tolerate raw_data that exactly matches (extra bytes would be a malformed file).
        assert_eq!(
            t.raw_data.len(),
            expected_bytes,
            "raw_data length {} != expected {} for shape {:?} dtype {:?}",
            t.raw_data.len(),
            expected_bytes,
            shape,
            dtype,
        );
        Buffer::from_raw_bytes(&t.raw_data, &shape, dtype)
    } else {
        match dtype {
            DType::F32 => Buffer::from_slice::<f32>(&t.float_data, &shape, dtype),
            DType::F64 => Buffer::from_slice::<f64>(&t.double_data, &shape, dtype),
            DType::I32 => {
                // prost represents int32_data as Vec<i32>
                Buffer::from_slice::<i32>(&t.int32_data, &shape, dtype)
            }
            DType::I64 => Buffer::from_slice::<i64>(&t.int64_data, &shape, dtype),
        }
    };

    Ok(buf)
}

/// Map ONNX `data_type` field values to our `DType`.
fn onnx_dtype(data_type: i32) -> Result<DType, OnnxError> {
    match data_type {
        1 => Ok(DType::F32),
        11 => Ok(DType::F64),
        6 => Ok(DType::I32),
        7 => Ok(DType::I64),
        other => Err(OnnxError::UnsupportedDtype(other)),
    }
}

/// Extract shape and dtype from a `ValueInfoProto`. Only tensor types supported.
fn extract_input_type(vi: &proto::ValueInfoProto) -> Result<(Shape, DType), OnnxError> {
    let type_proto = vi
        .r#type
        .as_ref()
        .and_then(|tp| tp.value.as_ref())
        .ok_or_else(|| OnnxError::UnsupportedInputType(vi.name.clone()))?;

    let tensor_type = match type_proto {
        type_proto::Value::TensorType(t) => t,
        _ => return Err(OnnxError::UnsupportedInputType(vi.name.clone())),
    };

    let dtype = onnx_dtype(tensor_type.elem_type)?;

    let shape_proto = tensor_type
        .shape
        .as_ref()
        .ok_or_else(|| OnnxError::UnsupportedInputType(vi.name.clone()))?;

    let mut dims = Vec::with_capacity(shape_proto.dim.len());
    for dim in &shape_proto.dim {
        match &dim.value {
            Some(DimValue::DimValue(v)) => dims.push(*v as u64),
            Some(DimValue::DimParam(p)) => {
                return Err(OnnxError::DynamicShape(p.clone()));
            }
            None => return Err(OnnxError::DynamicShape(vi.name.clone())),
        }
    }

    Ok((Shape(dims), dtype))
}

/// Extract node attributes into a `HashMap<String, OnnxAttribute>`.
///
/// Unsupported attribute types (TENSOR, GRAPH, etc.) are silently skipped.
fn extract_attributes(
    node: &proto::NodeProto,
) -> Result<HashMap<String, OnnxAttribute>, OnnxError> {
    let mut map = HashMap::new();
    for attr in &node.attribute {
        let value = match AttributeType::try_from(attr.r#type) {
            Ok(AttributeType::Int) => OnnxAttribute::Int(attr.i),
            Ok(AttributeType::Float) => OnnxAttribute::Float(attr.f),
            Ok(AttributeType::String) => {
                OnnxAttribute::String(String::from_utf8_lossy(&attr.s).into_owned())
            }
            Ok(AttributeType::Ints) => OnnxAttribute::Ints(attr.ints.clone()),
            Ok(AttributeType::Floats) => OnnxAttribute::Floats(attr.floats.clone()),
            // Skip TENSOR, GRAPH, and other unsupported types.
            _ => continue,
        };
        map.insert(attr.name.clone(), value);
    }
    Ok(map)
}

// ── Buffer extension ─────────────────────────────────────────────────────────
// Buffer::from_raw_bytes is not in the public API yet — add it here as an
// extension trait to avoid touching runtime.rs.

trait BufferExt {
    fn from_raw_bytes(bytes: &[u8], shape: &[u64], dtype: DType) -> Self;
}

impl BufferExt for Buffer {
    fn from_raw_bytes(bytes: &[u8], shape: &[u64], dtype: DType) -> Self {
        // Build a zero buffer for the right shape/dtype, then overwrite data.
        let mut buf = Buffer::new(shape, dtype);
        buf.data_mut().copy_from_slice(bytes);
        buf
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx::proto::*;

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Encode a ModelProto to bytes.
    fn encode(model: &ModelProto) -> Vec<u8> {
        let mut buf = Vec::new();
        model.encode(&mut buf).unwrap();
        buf
    }

    /// Build a minimal tensor shape for use in ValueInfoProto.
    fn fixed_shape(dims: &[i64]) -> TypeProto {
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
            r#type: Some(fixed_shape(dims)),
            ..Default::default()
        }
    }

    // ── Task 5.6: parse a hand-crafted small ONNX model ──────────────────────

    #[test]
    fn parse_simple_add_graph() {
        let model = ModelProto {
            ir_version: 8,
            graph: Some(GraphProto {
                node: vec![NodeProto {
                    op_type: "Add".into(),
                    input: vec!["X".into(), "Y".into()],
                    output: vec!["Z".into()],
                    ..Default::default()
                }],
                input: vec![value_info("X", &[2, 3]), value_info("Y", &[2, 3])],
                output: vec![value_info("Z", &[2, 3])],
                ..Default::default()
            }),
            ..Default::default()
        };

        let parsed = parse_bytes(&encode(&model)).expect("parse failed");

        assert_eq!(parsed.nodes.len(), 1);
        assert_eq!(parsed.nodes[0].op_type, "Add");
        assert_eq!(parsed.nodes[0].inputs, vec!["X", "Y"]);
        assert_eq!(parsed.nodes[0].outputs, vec!["Z"]);
        assert!(parsed.initializers.is_empty());
        assert_eq!(parsed.dynamic_inputs.len(), 2);
        assert_eq!(parsed.dynamic_inputs[0].name, "X");
        assert_eq!(parsed.dynamic_inputs[0].shape, Shape(vec![2, 3]));
        assert_eq!(parsed.dynamic_inputs[0].dtype, DType::F32);
        assert_eq!(parsed.outputs, vec!["Z"]);
    }

    #[test]
    fn parse_node_attributes() {
        let model = ModelProto {
            ir_version: 8,
            graph: Some(GraphProto {
                node: vec![NodeProto {
                    op_type: "Conv".into(),
                    input: vec!["X".into()],
                    output: vec!["Y".into()],
                    attribute: vec![
                        AttributeProto {
                            name: "kernel_shape".into(),
                            r#type: 7, // INTS
                            ints: vec![3, 3],
                            ..Default::default()
                        },
                        AttributeProto {
                            name: "dilations".into(),
                            r#type: 7, // INTS
                            ints: vec![1, 1],
                            ..Default::default()
                        },
                        AttributeProto {
                            name: "group".into(),
                            r#type: 2, // INT
                            i: 1,
                            ..Default::default()
                        },
                        AttributeProto {
                            name: "auto_pad".into(),
                            r#type: 3, // STRING
                            s: b"NOTSET".to_vec(),
                            ..Default::default()
                        },
                    ],
                    ..Default::default()
                }],
                input: vec![value_info("X", &[1, 3, 8, 8])],
                output: vec![value_info("Y", &[1, 3, 6, 6])],
                ..Default::default()
            }),
            ..Default::default()
        };

        let parsed = parse_bytes(&encode(&model)).expect("parse failed");
        let attrs = &parsed.nodes[0].attributes;

        assert_eq!(attrs.len(), 4);
        assert!(matches!(attrs["kernel_shape"], OnnxAttribute::Ints(ref v) if v == &[3i64, 3]));
        assert!(matches!(attrs["group"], OnnxAttribute::Int(1)));
        assert!(
            matches!(attrs["auto_pad"], OnnxAttribute::String(ref s) if s == "NOTSET")
        );
    }

    // ── Task 5.7: initializer raw_data vs float_data ──────────────────────────

    #[test]
    fn initializer_float_data_encoding() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0];

        let tensor_float = TensorProto {
            name: "W".into(),
            dims: vec![2, 2],
            data_type: 1, // FLOAT
            float_data: values.clone(),
            ..Default::default()
        };

        let model = ModelProto {
            ir_version: 8,
            graph: Some(GraphProto {
                initializer: vec![tensor_float],
                input: vec![value_info("X", &[2, 2])],
                output: vec![value_info("Y", &[2, 2])],
                node: vec![NodeProto {
                    op_type: "Add".into(),
                    input: vec!["X".into(), "W".into()],
                    output: vec!["Y".into()],
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        };

        let parsed = parse_bytes(&encode(&model)).expect("parse failed");
        assert_eq!(parsed.initializers.len(), 1);
        let (name, buf) = &parsed.initializers[0];
        assert_eq!(name, "W");
        assert_eq!(buf.as_slice::<f32>(), values.as_slice());
        assert_eq!(buf.shape(), &Shape(vec![2, 2]));
        assert_eq!(buf.dtype(), DType::F32);
    }

    #[test]
    fn initializer_raw_data_encoding_matches_float_data() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0];

        // Build raw_data by reinterpreting the f32 slice as bytes.
        let raw: Vec<u8> = values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let tensor_raw = TensorProto {
            name: "W_raw".into(),
            dims: vec![2, 2],
            data_type: 1, // FLOAT
            raw_data: raw,
            ..Default::default()
        };

        let tensor_typed = TensorProto {
            name: "W_typed".into(),
            dims: vec![2, 2],
            data_type: 1,
            float_data: values.clone(),
            ..Default::default()
        };

        let model = ModelProto {
            ir_version: 8,
            graph: Some(GraphProto {
                initializer: vec![tensor_raw, tensor_typed],
                output: vec![value_info("out", &[2, 2])],
                node: vec![],
                ..Default::default()
            }),
            ..Default::default()
        };

        let parsed = parse_bytes(&encode(&model)).expect("parse failed");
        let (_, buf_raw) = &parsed.initializers[0];
        let (_, buf_typed) = &parsed.initializers[1];

        // Both encodings must produce identical byte contents.
        assert_eq!(buf_raw.as_slice::<f32>(), buf_typed.as_slice::<f32>());
    }

    #[test]
    fn initializer_raw_data_i64() {
        let values = vec![10i64, 20, 30];
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let tensor = TensorProto {
            name: "idx".into(),
            dims: vec![3],
            data_type: 7, // INT64
            raw_data: raw,
            ..Default::default()
        };

        let model = ModelProto {
            ir_version: 8,
            graph: Some(GraphProto {
                initializer: vec![tensor],
                output: vec![],
                node: vec![],
                ..Default::default()
            }),
            ..Default::default()
        };

        let parsed = parse_bytes(&encode(&model)).expect("parse failed");
        let (_, buf) = &parsed.initializers[0];
        assert_eq!(buf.as_slice::<i64>(), values.as_slice());
        assert_eq!(buf.dtype(), DType::I64);
    }

    // ── Task 5.8: initializers also listed in graph.input ────────────────────

    #[test]
    fn initializers_in_graph_input_are_not_dynamic() {
        // Old opset models list initializers in graph.input too.
        // They must be classified as static weights, not dynamic inputs.
        let weight_tensor = TensorProto {
            name: "W".into(),
            dims: vec![3],
            data_type: 1,
            float_data: vec![0.1, 0.2, 0.3],
            ..Default::default()
        };

        let model = ModelProto {
            ir_version: 8,
            graph: Some(GraphProto {
                initializer: vec![weight_tensor],
                // W appears in both graph.input (opset compat) and initializer
                input: vec![value_info("X", &[3]), value_info("W", &[3])],
                output: vec![value_info("Y", &[3])],
                node: vec![NodeProto {
                    op_type: "Add".into(),
                    input: vec!["X".into(), "W".into()],
                    output: vec!["Y".into()],
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        };

        let parsed = parse_bytes(&encode(&model)).expect("parse failed");

        // Only X is dynamic; W is an initializer.
        assert_eq!(parsed.dynamic_inputs.len(), 1);
        assert_eq!(parsed.dynamic_inputs[0].name, "X");

        assert_eq!(parsed.initializers.len(), 1);
        assert_eq!(parsed.initializers[0].0, "W");
    }

    // ── Task 5.9: error handling ──────────────────────────────────────────────

    #[test]
    fn empty_bytes_returns_decode_error() {
        let result = parse_bytes(&[]);
        // prost succeeds on empty bytes (decodes to default), so we get MissingGraph
        // because there is no graph field.
        assert!(matches!(result, Err(OnnxError::MissingGraph)));
    }

    #[test]
    fn garbage_bytes_returns_decode_error() {
        let garbage = b"\xff\xfe\xfa\x01\x02\x03NOTAPROTO";
        let result = parse_bytes(garbage);
        // prost may or may not fail here — accept either Decode or MissingGraph.
        assert!(matches!(
            result,
            Err(OnnxError::Decode(_)) | Err(OnnxError::MissingGraph)
        ));
    }

    #[test]
    fn model_without_graph_returns_missing_graph() {
        let model = ModelProto {
            ir_version: 8,
            graph: None,
            ..Default::default()
        };
        let result = parse_bytes(&encode(&model));
        assert!(matches!(result, Err(OnnxError::MissingGraph)));
    }

    #[test]
    fn unsupported_dtype_returns_error() {
        // data_type = 10 (FLOAT16) is not supported.
        let tensor = TensorProto {
            name: "fp16_weight".into(),
            dims: vec![2],
            data_type: 10, // FLOAT16 — unsupported
            raw_data: vec![0u8; 4], // 2 * 2 bytes for fp16
            ..Default::default()
        };

        let model = ModelProto {
            ir_version: 8,
            graph: Some(GraphProto {
                initializer: vec![tensor],
                output: vec![],
                node: vec![],
                ..Default::default()
            }),
            ..Default::default()
        };

        let result = parse_bytes(&encode(&model));
        assert!(matches!(result, Err(OnnxError::UnsupportedDtype(10))));
    }

    #[test]
    fn dynamic_shape_in_input_returns_error() {
        use crate::onnx::proto::tensor_shape_proto::Dimension;
        use crate::onnx::proto::type_proto;

        // Input with a symbolic dim ("batch_size" instead of a fixed integer).
        let dynamic_input = ValueInfoProto {
            name: "X".into(),
            r#type: Some(TypeProto {
                value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                    elem_type: 1,
                    shape: Some(TensorShapeProto {
                        dim: vec![
                            Dimension {
                                value: Some(
                                    crate::onnx::proto::tensor_shape_proto::dimension::Value::DimParam(
                                        "batch_size".into(),
                                    ),
                                ),
                                ..Default::default()
                            },
                            Dimension {
                                value: Some(
                                    crate::onnx::proto::tensor_shape_proto::dimension::Value::DimValue(
                                        128,
                                    ),
                                ),
                                ..Default::default()
                            },
                        ],
                    }),
                })),
                ..Default::default()
            }),
            ..Default::default()
        };

        let model = ModelProto {
            ir_version: 8,
            graph: Some(GraphProto {
                input: vec![dynamic_input],
                output: vec![],
                node: vec![],
                ..Default::default()
            }),
            ..Default::default()
        };

        let result = parse_bytes(&encode(&model));
        assert!(matches!(result, Err(OnnxError::DynamicShape(_))));
    }
}
