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
    /// Reshape a tensor to a new shape with the same number of elements.
    Reshape {
        input: NodeId,
        /// Fully resolved target shape (no -1 values).
        target_shape: Vec<u64>,
        /// When `Some`, the reshape target shape comes from a traced tensor at
        /// runtime. The compiler emits `tensor.reshape` using this tensor as the
        /// shape operand. When `None`, the static `target_shape` is used.
        shape_tensor: Option<NodeId>,
        shape: Shape,
        dtype: DType,
    },
    /// 2D convolution (NCHW layout).
    Conv2d {
        input: NodeId,        // [N, CI, H, W]
        kernel: NodeId,       // [CO, CI, KH, KW]
        bias: Option<NodeId>, // [CO] (optional)
        strides: [u64; 2],    // [stride_h, stride_w]
        pads: [u64; 4],       // [pad_top, pad_left, pad_bottom, pad_right]
        dilations: [u64; 2],  // [dilation_h, dilation_w]
        shape: Shape,         // output shape [N, CO, OH, OW]
        dtype: DType,
    },
    /// 2D max pooling (NCHW layout).
    MaxPool2d {
        input: NodeId,         // [N, C, H, W]
        kernel_size: [u64; 2], // [KH, KW]
        strides: [u64; 2],     // [stride_h, stride_w]
        pads: [u64; 4],        // [pad_top, pad_left, pad_bottom, pad_right]
        shape: Shape,          // output shape [N, C, OH, OW]
        dtype: DType,
    },
    /// Global average pooling (NCHW layout) — averages all spatial dims.
    ///
    /// Input shape: [N, C, H, W]. Output shape: [N, C, 1, 1].
    GlobalAvgPool {
        input: NodeId, // [N, C, H, W]
        shape: Shape,  // output shape [N, C, 1, 1]
        dtype: DType,
    },
    /// Batch normalization: scale * (x - mean) / sqrt(var + epsilon) + bias.
    ///
    /// All parameter tensors (scale, bias, mean, var) have shape [C] where C is
    /// the channel dimension (dim 1 for NCHW inputs).
    BatchNorm {
        input: NodeId,
        scale: NodeId,
        bias: NodeId,
        mean: NodeId,
        var: NodeId,
        epsilon: f64,
        shape: Shape,
        dtype: DType,
    },
    /// Element-wise power: base ^ exponent (broadcast-compatible, float only).
    Pow {
        lhs: NodeId, // base
        rhs: NodeId, // exponent
        shape: Shape,
        dtype: DType,
    },
    /// Element-wise square root (float only).
    Sqrt {
        input: NodeId,
        shape: Shape,
        dtype: DType,
    },
    /// Cast tensor elements to a different dtype.
    Cast {
        input: NodeId,
        target_dtype: DType,
        shape: Shape,
        dtype: DType, // equals target_dtype
    },
    /// Reduce by computing mean along one or more axes.
    ReduceMean {
        input: NodeId,
        axes: Vec<i64>, // negative axes are relative to rank
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
        shape: Shape, // output shape [M, N]
        dtype: DType,
    },
    /// Gather: index into `input` along `axis` using `indices`.
    ///
    /// output_shape = input.shape[0..axis] + indices.shape + input.shape[axis+1..]
    Gather {
        input: NodeId,   // data tensor
        indices: NodeId, // index tensor (I32 or I64)
        axis: i64,
        shape: Shape,
        dtype: DType,
    },
    /// Slice a tensor along specified axes with optional step.
    ///
    /// `starts`, `ends`, `axes`, and `steps` are all per-axis, pre-normalized.
    Slice {
        input: NodeId,
        starts: Vec<i64>,
        ends: Vec<i64>,
        axes: Vec<i64>,
        steps: Vec<i64>,
        shape: Shape,
        dtype: DType,
    },
    /// Concatenate a list of tensors along `axis`.
    Concat {
        inputs: Vec<NodeId>,
        axis: i64,
        shape: Shape,
        dtype: DType,
    },
    /// Permute tensor dimensions according to `perm`.
    Transpose {
        input: NodeId,
        perm: Vec<i64>,
        shape: Shape,
        dtype: DType,
    },
    /// Element-wise conditional selection: output[i] = x[i] if condition[i] else y[i].
    ///
    /// `condition` dtype is I64 (treated as boolean: non-zero = true).
    Where {
        condition: NodeId,
        x: NodeId,
        y: NodeId,
        shape: Shape,
        dtype: DType,
    },
    /// Extract the runtime shape of a tensor as a 1-D I64 tensor.
    ///
    /// Output shape is always static `[rank]` (rank is known at trace time).
    /// dtype is always I64.
    ShapeOf {
        input: NodeId,
        shape: Shape, // always [rank]
        dtype: DType, // always I64
    },
    /// Create a tensor filled with a constant value; output shape may be dynamic.
    ///
    /// `shape_input` is a traced 1-D I64 tensor describing the output shape at runtime.
    /// `shape` is the statically-known output shape (may contain DIM_DYNAMIC).
    ConstantOfShape {
        shape_input: NodeId,
        fill_value: f64,
        shape: Shape,
        dtype: DType,
    },
    /// Create an arange tensor [start, start+delta, ...) up to (not including) limit.
    ///
    /// `start`, `limit`, `delta` are scalar (1-element) tensors.
    /// Output shape is `[DIM_DYNAMIC]` when any input is non-constant.
    Range {
        start: NodeId,
        limit: NodeId,
        delta: NodeId,
        shape: Shape,
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
            Op::Reshape { shape, .. } => shape,
            Op::Conv2d { shape, .. } => shape,
            Op::MaxPool2d { shape, .. } => shape,
            Op::GlobalAvgPool { shape, .. } => shape,
            Op::BatchNorm { shape, .. } => shape,
            Op::Gemm { shape, .. } => shape,
            Op::Pow { shape, .. } => shape,
            Op::Sqrt { shape, .. } => shape,
            Op::Cast { shape, .. } => shape,
            Op::ReduceMean { shape, .. } => shape,
            Op::Gather { shape, .. } => shape,
            Op::Slice { shape, .. } => shape,
            Op::Concat { shape, .. } => shape,
            Op::Transpose { shape, .. } => shape,
            Op::Where { shape, .. } => shape,
            Op::ShapeOf { shape, .. } => shape,
            Op::ConstantOfShape { shape, .. } => shape,
            Op::Range { shape, .. } => shape,
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
            Op::Reshape { dtype, .. } => *dtype,
            Op::Conv2d { dtype, .. } => *dtype,
            Op::MaxPool2d { dtype, .. } => *dtype,
            Op::GlobalAvgPool { dtype, .. } => *dtype,
            Op::BatchNorm { dtype, .. } => *dtype,
            Op::Gemm { dtype, .. } => *dtype,
            Op::Pow { dtype, .. } => *dtype,
            Op::Sqrt { dtype, .. } => *dtype,
            Op::Cast { dtype, .. } => *dtype,
            Op::ReduceMean { dtype, .. } => *dtype,
            Op::Gather { dtype, .. } => *dtype,
            Op::Slice { dtype, .. } => *dtype,
            Op::Concat { dtype, .. } => *dtype,
            Op::Transpose { dtype, .. } => *dtype,
            Op::Where { dtype, .. } => *dtype,
            Op::ShapeOf { dtype, .. } => *dtype,
            Op::ConstantOfShape { dtype, .. } => *dtype,
            Op::Range { dtype, .. } => *dtype,
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
