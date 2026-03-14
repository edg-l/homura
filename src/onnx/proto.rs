// Re-export prost-generated ONNX protobuf types.
// The package declaration in onnx.proto is `package onnx;`, so prost generates
// types in a module named `onnx` inside the output file.
include!(concat!(env!("OUT_DIR"), "/onnx.rs"));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_proto_is_usable() {
        let model = ModelProto::default();
        assert_eq!(model.ir_version, 0);
    }

    #[test]
    fn graph_proto_is_usable() {
        let graph = GraphProto::default();
        assert!(graph.node.is_empty());
        assert!(graph.initializer.is_empty());
    }

    #[test]
    fn tensor_proto_is_usable() {
        let tensor = TensorProto::default();
        assert!(tensor.dims.is_empty());
        assert!(tensor.float_data.is_empty());
        assert!(tensor.raw_data.is_empty());
    }
}
