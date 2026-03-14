use crate::{
    DType, Shape,
    op::{NodeId, Op},
    trace,
};

/// A handle to a traced tensor. This is not a real tensor — it is just a
/// reference to a node in the active trace.
pub struct Tensor {
    pub(crate) id: NodeId,
    pub(crate) shape: Shape,
    pub(crate) dtype: DType,
}

impl Tensor {
    /// Create an input placeholder tensor in the current trace.
    pub fn new(shape: &[u64], dtype: DType) -> Self {
        let arg_index = trace::current_input_count();
        let shape = Shape(shape.to_vec());
        let id = trace::record(Op::Input {
            shape: shape.clone(),
            dtype,
            arg_index,
        });
        Self { id, shape, dtype }
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn id(&self) -> NodeId {
        self.id
    }
}

impl std::ops::Add for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: &Tensor) -> Tensor {
        assert_eq!(self.shape, rhs.shape, "shape mismatch in Add");
        assert_eq!(self.dtype, rhs.dtype, "dtype mismatch in Add");
        let id = trace::record(Op::Add {
            lhs: self.id,
            rhs: rhs.id,
            shape: self.shape.clone(),
            dtype: self.dtype,
        });
        Tensor {
            id,
            shape: self.shape.clone(),
            dtype: self.dtype,
        }
    }
}

impl std::ops::Add for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Tensor) -> Tensor {
        &self + &rhs
    }
}

impl std::ops::Add<Tensor> for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: Tensor) -> Tensor {
        self + &rhs
    }
}

impl std::ops::Add<&Tensor> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: &Tensor) -> Tensor {
        &self + rhs
    }
}

impl std::ops::Sub for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: &Tensor) -> Tensor {
        assert_eq!(self.shape, rhs.shape, "shape mismatch in Sub");
        assert_eq!(self.dtype, rhs.dtype, "dtype mismatch in Sub");
        let id = trace::record(Op::Sub {
            lhs: self.id,
            rhs: rhs.id,
            shape: self.shape.clone(),
            dtype: self.dtype,
        });
        Tensor {
            id,
            shape: self.shape.clone(),
            dtype: self.dtype,
        }
    }
}

impl std::ops::Sub for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Tensor) -> Tensor {
        &self - &rhs
    }
}

impl std::ops::Sub<Tensor> for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Tensor) -> Tensor {
        self - &rhs
    }
}

impl std::ops::Sub<&Tensor> for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: &Tensor) -> Tensor {
        &self - rhs
    }
}

impl std::ops::Mul for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: &Tensor) -> Tensor {
        assert_eq!(self.shape, rhs.shape, "shape mismatch in Mul");
        assert_eq!(self.dtype, rhs.dtype, "dtype mismatch in Mul");
        let id = trace::record(Op::Mul {
            lhs: self.id,
            rhs: rhs.id,
            shape: self.shape.clone(),
            dtype: self.dtype,
        });
        Tensor {
            id,
            shape: self.shape.clone(),
            dtype: self.dtype,
        }
    }
}

impl std::ops::Mul for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Tensor) -> Tensor {
        &self * &rhs
    }
}

impl std::ops::Mul<Tensor> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Tensor) -> Tensor {
        self * &rhs
    }
}

impl std::ops::Mul<&Tensor> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: &Tensor) -> Tensor {
        &self * rhs
    }
}

impl std::ops::Div for &Tensor {
    type Output = Tensor;

    fn div(self, rhs: &Tensor) -> Tensor {
        assert_eq!(self.shape, rhs.shape, "shape mismatch in Div");
        assert_eq!(self.dtype, rhs.dtype, "dtype mismatch in Div");
        let id = trace::record(Op::Div {
            lhs: self.id,
            rhs: rhs.id,
            shape: self.shape.clone(),
            dtype: self.dtype,
        });
        Tensor {
            id,
            shape: self.shape.clone(),
            dtype: self.dtype,
        }
    }
}

impl std::ops::Div for Tensor {
    type Output = Tensor;

    fn div(self, rhs: Tensor) -> Tensor {
        &self / &rhs
    }
}

impl std::ops::Div<Tensor> for &Tensor {
    type Output = Tensor;

    fn div(self, rhs: Tensor) -> Tensor {
        self / &rhs
    }
}

impl std::ops::Div<&Tensor> for Tensor {
    type Output = Tensor;

    fn div(self, rhs: &Tensor) -> Tensor {
        &self / rhs
    }
}

impl std::ops::Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Tensor {
        let id = trace::record(Op::Neg {
            input: self.id,
            shape: self.shape.clone(),
            dtype: self.dtype,
        });
        Tensor {
            id,
            shape: self.shape.clone(),
            dtype: self.dtype,
        }
    }
}

impl std::ops::Neg for Tensor {
    type Output = Tensor;

    fn neg(self) -> Tensor {
        -&self
    }
}

impl Tensor {
    pub fn relu(&self) -> Tensor {
        let id = trace::record(Op::Relu {
            input: self.id,
            shape: self.shape.clone(),
            dtype: self.dtype,
        });
        Tensor {
            id,
            shape: self.shape.clone(),
            dtype: self.dtype,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::{begin_trace, take_trace};

    #[test]
    fn tensor_new_records_input() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F32);
        let trace = take_trace();
        assert_eq!(a.id.0, 0);
        assert_eq!(trace.input_count(), 1);
        assert_eq!(trace.ops().len(), 1);
    }

    #[test]
    fn add_records_add_op() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F32);
        let b = Tensor::new(&[4], DType::F32);
        let c = &a + &b;
        let trace = take_trace();
        assert_eq!(c.id.0, 2);
        assert_eq!(trace.ops().len(), 3);
        assert_eq!(trace.input_count(), 2);
    }

    #[test]
    #[should_panic(expected = "shape mismatch")]
    fn add_shape_mismatch_panics() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F32);
        let b = Tensor::new(&[8], DType::F32);
        let _ = &a + &b;
        let _ = take_trace();
    }
}
