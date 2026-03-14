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
}

impl Op {
    pub fn shape(&self) -> &Shape {
        match self {
            Op::Input { shape, .. } => shape,
            Op::Add { shape, .. } => shape,
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            Op::Input { dtype, .. } => *dtype,
            Op::Add { dtype, .. } => *dtype,
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
