use crate::{DType, Shape};

/// Index into the trace's op list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub(crate) u32);

/// A recorded operation.
#[derive(Debug, Clone)]
pub enum Op {
    /// A placeholder input tensor.
    Input {
        shape: Shape,
        dtype: DType,
        /// Index in the function's argument list.
        arg_index: u32,
    },
    /// Element-wise addition.
    Add {
        lhs: NodeId,
        rhs: NodeId,
        shape: Shape,
        dtype: DType,
    },
    /// Element-wise subtraction.
    Sub {
        lhs: NodeId,
        rhs: NodeId,
        shape: Shape,
        dtype: DType,
    },
    /// Element-wise multiplication.
    Mul {
        lhs: NodeId,
        rhs: NodeId,
        shape: Shape,
        dtype: DType,
    },
    /// Element-wise division.
    Div {
        lhs: NodeId,
        rhs: NodeId,
        shape: Shape,
        dtype: DType,
    },
    /// Element-wise negation.
    Neg {
        input: NodeId,
        shape: Shape,
        dtype: DType,
    },
    /// Element-wise relu (max(0, x)).
    Relu {
        input: NodeId,
        shape: Shape,
        dtype: DType,
    },
    /// Element-wise exponential (e^x).
    Exp {
        input: NodeId,
        shape: Shape,
        dtype: DType,
    },
    /// Element-wise hyperbolic tangent.
    Tanh {
        input: NodeId,
        shape: Shape,
        dtype: DType,
    },
    /// Matrix multiplication: [M, K] x [K, N] -> [M, N].
    Matmul {
        lhs: NodeId,
        rhs: NodeId,
        shape: Shape,
        dtype: DType,
    },
    /// Reduce by summing along a dimension.
    ReduceSum {
        input: NodeId,
        dim: usize,
        keepdim: bool,
        shape: Shape,
        dtype: DType,
    },
    /// Reduce by taking max along a dimension.
    ReduceMax {
        input: NodeId,
        dim: usize,
        keepdim: bool,
        shape: Shape,
        dtype: DType,
    },
    /// General matrix multiply: alpha * (A @ B) + beta * C.
    ///
    /// Equivalent to the ONNX Gemm op. The output shape is [M, N].
    Gemm {
        lhs: NodeId,          // A matrix
        rhs: NodeId,          // B matrix
        bias: Option<NodeId>, // C vector (optional)
        alpha: f64,           // scaling for A@B
        beta: f64,            // scaling for C
        trans_a: bool,
        trans_b: bool,
        shape: Shape,         // output shape [M, N]
        dtype: DType,
    },
}

impl Op {
    pub fn shape(&self) -> &Shape {
        match self {
            Op::Input { shape, .. } => shape,
            Op::Add { shape, .. } => shape,
            Op::Sub { shape, .. } => shape,
            Op::Mul { shape, .. } => shape,
            Op::Div { shape, .. } => shape,
            Op::Neg { shape, .. } => shape,
            Op::Relu { shape, .. } => shape,
            Op::Exp { shape, .. } => shape,
            Op::Tanh { shape, .. } => shape,
            Op::Matmul { shape, .. } => shape,
            Op::ReduceSum { shape, .. } => shape,
            Op::ReduceMax { shape, .. } => shape,
            Op::Gemm { shape, .. } => shape,
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            Op::Input { dtype, .. } => *dtype,
            Op::Add { dtype, .. } => *dtype,
            Op::Sub { dtype, .. } => *dtype,
            Op::Mul { dtype, .. } => *dtype,
            Op::Div { dtype, .. } => *dtype,
            Op::Neg { dtype, .. } => *dtype,
            Op::Relu { dtype, .. } => *dtype,
            Op::Exp { dtype, .. } => *dtype,
            Op::Tanh { dtype, .. } => *dtype,
            Op::Matmul { dtype, .. } => *dtype,
            Op::ReduceSum { dtype, .. } => *dtype,
            Op::ReduceMax { dtype, .. } => *dtype,
            Op::Gemm { dtype, .. } => *dtype,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn op_shape_and_dtype() {
        let op = Op::Input {
            shape: Shape(vec![4]),
            dtype: DType::F32,
            arg_index: 0,
        };
        assert_eq!(op.shape(), &Shape(vec![4]));
        assert_eq!(op.dtype(), DType::F32);
    }
}
