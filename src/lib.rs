pub mod cache;
pub(crate) mod compiler;
pub mod graph_builder;
mod llvm_ffi;
pub mod dtype;
pub mod generate;
pub mod kv_generate;
pub mod onnx;
pub mod op;
pub(crate) mod runtime;
pub mod shape;
pub mod tokenizer;

/// Timestamp prefix for compile-time logging. Returns seconds since process start.
pub(crate) fn log_ts() -> f64 {
    use std::sync::OnceLock;
    use std::time::Instant;
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs_f64()
}

pub use dtype::DType;
pub use graph_builder::{GraphBuilder, GraphContext};
pub use onnx::Model;
pub use op::NodeId;
pub use runtime::{Buffer, CompiledGraph, ExecutionPlan, KernelStep, SlotDesc};
pub use shape::Shape;
