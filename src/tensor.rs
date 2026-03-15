use crate::{
    DType, Shape,
    op::{NodeId, Op},
    runtime::Buffer,
    trace,
};

/// A handle to a traced tensor. This is not a real tensor — it is just a
/// reference to a node in the active trace.
#[derive(Clone)]
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
        assert_eq!(self.dtype, rhs.dtype, "dtype mismatch in Add");
        let shape = self
            .shape
            .broadcast(&rhs.shape)
            .unwrap_or_else(|e| panic!("broadcast failed in Add: {e}"));
        let id = trace::record(Op::Add {
            lhs: self.id,
            rhs: rhs.id,
            shape: shape.clone(),
            dtype: self.dtype,
        });
        Tensor {
            id,
            shape,
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
        assert_eq!(self.dtype, rhs.dtype, "dtype mismatch in Sub");
        let shape = self
            .shape
            .broadcast(&rhs.shape)
            .unwrap_or_else(|e| panic!("broadcast failed in Sub: {e}"));
        let id = trace::record(Op::Sub {
            lhs: self.id,
            rhs: rhs.id,
            shape: shape.clone(),
            dtype: self.dtype,
        });
        Tensor {
            id,
            shape,
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
        assert_eq!(self.dtype, rhs.dtype, "dtype mismatch in Mul");
        let shape = self
            .shape
            .broadcast(&rhs.shape)
            .unwrap_or_else(|e| panic!("broadcast failed in Mul: {e}"));
        let id = trace::record(Op::Mul {
            lhs: self.id,
            rhs: rhs.id,
            shape: shape.clone(),
            dtype: self.dtype,
        });
        Tensor {
            id,
            shape,
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
        assert_eq!(self.dtype, rhs.dtype, "dtype mismatch in Div");
        let shape = self
            .shape
            .broadcast(&rhs.shape)
            .unwrap_or_else(|e| panic!("broadcast failed in Div: {e}"));
        let id = trace::record(Op::Div {
            lhs: self.id,
            rhs: rhs.id,
            shape: shape.clone(),
            dtype: self.dtype,
        });
        Tensor {
            id,
            shape,
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
    pub fn gemm(
        &self,
        rhs: &Tensor,
        bias: Option<&Tensor>,
        alpha: f64,
        beta: f64,
        trans_a: bool,
        trans_b: bool,
    ) -> Tensor {
        assert_eq!(self.shape.rank(), 2, "gemm requires rank-2 lhs");
        assert_eq!(rhs.shape.rank(), 2, "gemm requires rank-2 rhs");
        assert_eq!(self.dtype, rhs.dtype, "dtype mismatch in gemm");

        // self shape: [M, K] or [K, M] if trans_a
        // rhs shape:  [K, N] or [N, K] if trans_b
        let m = if trans_a {
            self.shape.0[1]
        } else {
            self.shape.0[0]
        };
        let n = if trans_b {
            rhs.shape.0[0]
        } else {
            rhs.shape.0[1]
        };
        let output_shape = Shape(vec![m, n]);

        let op = Op::Gemm {
            lhs: self.id,
            rhs: rhs.id,
            bias: bias.map(|b| b.id),
            alpha,
            beta,
            trans_a,
            trans_b,
            shape: output_shape.clone(),
            dtype: self.dtype,
        };
        let id = trace::record(op);
        Tensor {
            id,
            shape: output_shape,
            dtype: self.dtype,
        }
    }

    pub fn matmul(&self, rhs: &Tensor) -> Tensor {
        assert_eq!(self.shape.rank(), 2, "matmul requires rank-2 tensors");
        assert_eq!(rhs.shape.rank(), 2, "matmul requires rank-2 tensors");
        assert_eq!(self.dtype, rhs.dtype, "dtype mismatch in matmul");
        let m = self.shape.0[0];
        let k = self.shape.0[1];
        let k2 = rhs.shape.0[0];
        let n = rhs.shape.0[1];
        assert_eq!(
            k, k2,
            "inner dimensions mismatch in matmul: [{}, {}] x [{}, {}]",
            m, k, k2, n
        );
        let out_shape = Shape(vec![m, n]);
        let id = trace::record(Op::Matmul {
            lhs: self.id,
            rhs: rhs.id,
            shape: out_shape.clone(),
            dtype: self.dtype,
        });
        Tensor {
            id,
            shape: out_shape,
            dtype: self.dtype,
        }
    }

    /// Reshape the tensor to `target_shape`.
    ///
    /// At most one dimension may be `-1`, in which case it is inferred from the
    /// total element count. All other dimensions must be non-negative. The total
    /// number of elements must not change.
    pub fn reshape(&self, target_shape: &[i64]) -> Tensor {
        let total_elems: u64 = self.shape.0.iter().product();

        // Count and locate the -1 dimension, if any.
        let neg_count = target_shape.iter().filter(|&&d| d == -1).count();
        assert!(
            neg_count <= 1,
            "reshape: at most one dimension can be -1, got {neg_count}"
        );

        let mut resolved: Vec<u64> = Vec::with_capacity(target_shape.len());
        let mut known_product: u64 = 1;
        let mut infer_idx: Option<usize> = None;

        for (i, &d) in target_shape.iter().enumerate() {
            if d == -1 {
                infer_idx = Some(i);
                resolved.push(0); // placeholder
            } else {
                assert!(d > 0, "reshape: dimension must be positive or -1, got {d}");
                let d = d as u64;
                known_product *= d;
                resolved.push(d);
            }
        }

        if let Some(idx) = infer_idx {
            assert!(
                total_elems % known_product == 0,
                "reshape: cannot infer -1 dimension: {total_elems} elements not divisible by {known_product}"
            );
            resolved[idx] = total_elems / known_product;
        } else {
            assert_eq!(
                known_product, total_elems,
                "reshape: element count mismatch: input has {total_elems} elements but target shape has {known_product}"
            );
        }

        let shape = Shape(resolved.clone());
        let id = trace::record(Op::Reshape {
            input: self.id,
            target_shape: resolved,
            shape: shape.clone(),
            dtype: self.dtype,
        });
        Tensor {
            id,
            shape,
            dtype: self.dtype,
        }
    }

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

    pub fn exp(&self) -> Tensor {
        assert!(
            matches!(self.dtype, DType::F32 | DType::F64),
            "exp requires float dtype, got {:?}",
            self.dtype
        );
        let id = trace::record(Op::Exp {
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

    pub fn tanh(&self) -> Tensor {
        assert!(
            matches!(self.dtype, DType::F32 | DType::F64),
            "tanh requires float dtype, got {:?}",
            self.dtype
        );
        let id = trace::record(Op::Tanh {
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

    pub fn reduce_sum(&self, dim: i32, keepdim: bool) -> Tensor {
        let resolved_dim = resolve_dim(dim, self.shape.rank());
        let out_shape = reduce_shape(&self.shape, resolved_dim, keepdim);
        let id = trace::record(Op::ReduceSum {
            input: self.id,
            dim: resolved_dim,
            keepdim,
            shape: out_shape.clone(),
            dtype: self.dtype,
        });
        Tensor {
            id,
            shape: out_shape,
            dtype: self.dtype,
        }
    }

    pub fn reduce_max(&self, dim: i32, keepdim: bool) -> Tensor {
        let resolved_dim = resolve_dim(dim, self.shape.rank());
        let out_shape = reduce_shape(&self.shape, resolved_dim, keepdim);
        let id = trace::record(Op::ReduceMax {
            input: self.id,
            dim: resolved_dim,
            keepdim,
            shape: out_shape.clone(),
            dtype: self.dtype,
        });
        Tensor {
            id,
            shape: out_shape,
            dtype: self.dtype,
        }
    }

    /// Numerically stable softmax along `dim`.
    ///
    /// Decomposes into: exp(x - max(x)) / sum(exp(x - max(x)))
    /// with keepdim=true reductions so the result broadcasts correctly.
    pub fn softmax(&self, dim: i32) -> Tensor {
        let max = self.reduce_max(dim, true);
        let shifted = self - &max;
        let exps = shifted.exp();
        let sum = exps.reduce_sum(dim, true);
        &exps / &sum
    }

    /// One-call API: freeze the active trace, compile, and execute.
    ///
    /// The caller must have called `begin_trace()` before building the
    /// computation graph. `inputs` must be in the same order as the
    /// `Tensor::new` calls that created the graph's input placeholders.
    pub fn eval(&self, inputs: &[Buffer]) -> Buffer {
        let trace = crate::trace::take_trace();
        let compiled =
            crate::Compiler::compile(&trace, &[self.id]).expect("compilation failed in eval");
        compiled.run(&inputs.iter().collect::<Vec<&Buffer>>())
    }
}

fn resolve_dim(dim: i32, rank: usize) -> usize {
    let resolved = if dim < 0 {
        (rank as i32 + dim) as usize
    } else {
        dim as usize
    };
    assert!(resolved < rank, "dim {dim} is out of range for rank {rank}");
    resolved
}

fn reduce_shape(shape: &Shape, dim: usize, keepdim: bool) -> Shape {
    let mut new_dims = shape.0.clone();
    if keepdim {
        new_dims[dim] = 1;
    } else {
        new_dims.remove(dim);
    }
    Shape(new_dims)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Compiler,
        trace::{begin_trace, take_trace},
    };

    #[test]
    fn tensor_new_records_input() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F32);
        let trace = take_trace();
        assert_eq!(a.id.0, 0);
        assert_eq!(trace.input_count(), 1);
        assert_eq!(trace.ops().len(), 1);
    }

    // ── Gemm trace tests (task 6.3) ───────────────────────────────────────────

    #[test]
    fn gemm_records_gemm_op() {
        begin_trace();
        let a = Tensor::new(&[2, 3], DType::F32);
        let b = Tensor::new(&[3, 4], DType::F32);
        let c = a.gemm(&b, None, 1.0, 1.0, false, false);
        let trace = take_trace();
        // ops: Input(a), Input(b), Gemm
        assert_eq!(trace.ops().len(), 3);
        assert_eq!(trace.input_count(), 2);
        let op = &trace.ops()[c.id.0 as usize];
        match op {
            Op::Gemm {
                alpha,
                beta,
                trans_a,
                trans_b,
                bias,
                shape,
                dtype,
                ..
            } => {
                assert_eq!(*alpha, 1.0);
                assert_eq!(*beta, 1.0);
                assert!(!trans_a);
                assert!(!trans_b);
                assert!(bias.is_none());
                assert_eq!(shape.0, vec![2, 4]);
                assert_eq!(*dtype, DType::F32);
            }
            _ => panic!("expected Op::Gemm, got {:?}", op),
        }
    }

    #[test]
    fn gemm_records_trans_flags() {
        begin_trace();
        let a = Tensor::new(&[3, 2], DType::F32); // [K, M], transA makes it [M, K] = [2, 3]
        let b = Tensor::new(&[4, 3], DType::F32); // [N, K], transB makes it [K, N] = [3, 4]
        let c = a.gemm(&b, None, 1.0, 1.0, true, true);
        let trace = take_trace();
        let op = &trace.ops()[c.id.0 as usize];
        match op {
            Op::Gemm {
                trans_a,
                trans_b,
                shape,
                ..
            } => {
                assert!(*trans_a);
                assert!(*trans_b);
                // output [M, N] = [2, 4]
                assert_eq!(shape.0, vec![2, 4]);
            }
            _ => panic!("expected Op::Gemm"),
        }
    }

    #[test]
    fn gemm_records_bias_and_scaling() {
        begin_trace();
        let a = Tensor::new(&[2, 3], DType::F32);
        let b = Tensor::new(&[3, 4], DType::F32);
        let bias = Tensor::new(&[4], DType::F32);
        let c = a.gemm(&b, Some(&bias), 2.0, 0.5, false, false);
        let trace = take_trace();
        let op = &trace.ops()[c.id.0 as usize];
        match op {
            Op::Gemm {
                alpha,
                beta,
                bias: bias_id,
                ..
            } => {
                assert_eq!(*alpha, 2.0);
                assert_eq!(*beta, 0.5);
                assert!(bias_id.is_some());
            }
            _ => panic!("expected Op::Gemm"),
        }
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
    #[should_panic(expected = "broadcast failed")]
    fn add_incompatible_shapes_panics() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F32);
        let b = Tensor::new(&[8], DType::F32);
        let _ = &a + &b;
        let _ = take_trace();
    }

    // ── Reshape trace tests (task 9.3) ───────────────────────────────────────

    #[test]
    fn reshape_records_op_with_resolved_shape() {
        begin_trace();
        let a = Tensor::new(&[2, 6], DType::F32);
        let b = a.reshape(&[3, 4]);
        let trace = take_trace();
        let op = &trace.ops()[b.id.0 as usize];
        match op {
            Op::Reshape {
                target_shape,
                shape,
                dtype,
                ..
            } => {
                assert_eq!(target_shape, &vec![3u64, 4]);
                assert_eq!(shape.0, vec![3u64, 4]);
                assert_eq!(*dtype, DType::F32);
            }
            _ => panic!("expected Op::Reshape, got {:?}", op),
        }
    }

    #[test]
    fn reshape_infers_negative_one_dim() {
        begin_trace();
        let a = Tensor::new(&[2, 6], DType::F32);
        let b = a.reshape(&[-1, 4]);
        let trace = take_trace();
        let op = &trace.ops()[b.id.0 as usize];
        match op {
            Op::Reshape { target_shape, .. } => {
                // 2*6=12 elements, second dim=4, so first dim=3
                assert_eq!(target_shape, &vec![3u64, 4]);
            }
            _ => panic!("expected Op::Reshape, got {:?}", op),
        }
    }

    #[test]
    #[should_panic(expected = "at most one dimension can be -1")]
    fn reshape_multiple_neg_one_panics() {
        begin_trace();
        let a = Tensor::new(&[12], DType::F32);
        let _ = a.reshape(&[-1, -1]);
        let _ = take_trace();
    }

    #[test]
    #[should_panic(expected = "element count mismatch")]
    fn reshape_element_count_mismatch_panics() {
        begin_trace();
        let a = Tensor::new(&[2, 6], DType::F32);
        let _ = a.reshape(&[3, 5]); // 15 != 12
        let _ = take_trace();
    }

    // ── Softmax tests (tasks 7.2, 7.3) ───────────────────────────────────────

    #[test]
    fn run_softmax() {
        begin_trace();
        let a = Tensor::new(&[2, 3], DType::F32);
        let b = a.softmax(-1);
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[b.id]).expect("compile failed");
        // [[1, 2, 3], [1, 2, 3]]
        let a_buf = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3], DType::F32);
        let result = compiled.run(&[&a_buf]);
        let out = result.as_slice::<f32>();
        // Each row should sum to ~1.0
        let row0_sum: f32 = out[0] + out[1] + out[2];
        let row1_sum: f32 = out[3] + out[4] + out[5];
        assert!((row0_sum - 1.0).abs() < 1e-5, "row 0 sum: {row0_sum}");
        assert!((row1_sum - 1.0).abs() < 1e-5, "row 1 sum: {row1_sum}");
        // Values should be monotonically increasing within each row
        assert!(out[0] < out[1] && out[1] < out[2]);
    }

    #[test]
    fn run_softmax_stability() {
        begin_trace();
        let a = Tensor::new(&[3], DType::F32);
        let b = a.softmax(-1);
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[b.id]).expect("compile failed");
        // Large values that would overflow exp() without the max subtraction trick
        let a_buf = Buffer::from_slice::<f32>(&[1000.0, 1000.0, 1000.0], &[3], DType::F32);
        let result = compiled.run(&[&a_buf]);
        let out = result.as_slice::<f32>();
        // All equal inputs -> uniform distribution
        for &v in out {
            assert!((v - 1.0 / 3.0).abs() < 1e-5, "expected ~0.333, got {v}");
        }
    }

    // ── Eval sugar test (task 8.2) ────────────────────────────────────────────

    #[test]
    fn eval_simple_add() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F32);
        let b = Tensor::new(&[4], DType::F32);
        let c = &a + &b;

        let a_buf = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[4], DType::F32);
        let b_buf = Buffer::from_slice::<f32>(&[5.0, 6.0, 7.0, 8.0], &[4], DType::F32);
        let result = c.eval(&[a_buf, b_buf]);
        assert_eq!(result.as_slice::<f32>(), &[6.0, 8.0, 10.0, 12.0]);
    }
}
