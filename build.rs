fn main() {
    // Use vendored protoc from protobuf-src so we don't require a system protoc.
    // SAFETY: build scripts run single-threaded, so set_var has no data races.
    unsafe { std::env::set_var("PROTOC", protobuf_src::protoc()) };

    prost_build::Config::new()
        .compile_protos(&["proto/onnx.proto"], &["proto/"])
        .expect("Failed to compile ONNX protobuf");
}
