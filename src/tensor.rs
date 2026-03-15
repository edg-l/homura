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
    pub fn conv2d(
        &self,
        kernel: &Tensor,
        bias: Option<&Tensor>,
        strides: [u64; 2],
        pads: [u64; 4],
        dilations: [u64; 2],
    ) -> Tensor {
        assert_eq!(self.shape.rank(), 4, "conv2d requires rank-4 input (NCHW)");
        assert_eq!(
            kernel.shape.rank(),
            4,
            "conv2d requires rank-4 kernel (OIHW)"
        );
        assert_eq!(
            self.shape.0[1], kernel.shape.0[1],
            "conv2d: input channels ({}) != kernel input channels ({})",
            self.shape.0[1], kernel.shape.0[1]
        );
        assert_eq!(self.dtype, kernel.dtype, "dtype mismatch in conv2d");

        let n = self.shape.0[0];
        let h = self.shape.0[2];
        let w = self.shape.0[3];
        let co = kernel.shape.0[0];
        let kh = kernel.shape.0[2];
        let kw = kernel.shape.0[3];

        // OH = (H + pad_top + pad_bottom - dilation_h * (KH - 1) - 1) / stride_h + 1
        let eff_kh = dilations[0] * (kh - 1) + 1;
        let eff_kw = dilations[1] * (kw - 1) + 1;
        assert!(
            h + pads[0] + pads[2] >= eff_kh,
            "conv2d: effective kernel height ({eff_kh}) exceeds padded input height ({})",
            h + pads[0] + pads[2]
        );
        assert!(
            w + pads[1] + pads[3] >= eff_kw,
            "conv2d: effective kernel width ({eff_kw}) exceeds padded input width ({})",
            w + pads[1] + pads[3]
        );
        let oh = (h + pads[0] + pads[2] - eff_kh) / strides[0] + 1;
        let ow = (w + pads[1] + pads[3] - eff_kw) / strides[1] + 1;
        let output_shape = Shape(vec![n, co, oh, ow]);

        let id = trace::record(Op::Conv2d {
            input: self.id,
            kernel: kernel.id,
            bias: bias.map(|b| b.id),
            strides,
            pads,
            dilations,
            shape: output_shape.clone(),
            dtype: self.dtype,
        });
        Tensor {
            id,
            shape: output_shape,
            dtype: self.dtype,
        }
    }

    pub fn max_pool2d(&self, kernel_size: [u64; 2], strides: [u64; 2], pads: [u64; 4]) -> Tensor {
        assert_eq!(
            self.shape.rank(),
            4,
            "max_pool2d requires rank-4 input (NCHW)"
        );

        let n = self.shape.0[0];
        let c = self.shape.0[1];
        let h = self.shape.0[2];
        let w = self.shape.0[3];
        let [kh, kw] = kernel_size;
        // OH = (H + pad_top + pad_bottom - KH) / stride_h + 1
        assert!(
            h + pads[0] + pads[2] >= kh,
            "max_pool2d: kernel height ({kh}) exceeds padded input height ({})",
            h + pads[0] + pads[2]
        );
        assert!(
            w + pads[1] + pads[3] >= kw,
            "max_pool2d: kernel width ({kw}) exceeds padded input width ({})",
            w + pads[1] + pads[3]
        );
        let oh = (h + pads[0] + pads[2] - kh) / strides[0] + 1;
        let ow = (w + pads[1] + pads[3] - kw) / strides[1] + 1;
        let output_shape = Shape(vec![n, c, oh, ow]);

        let id = trace::record(Op::MaxPool2d {
            input: self.id,
            kernel_size,
            strides,
            pads,
            shape: output_shape.clone(),
            dtype: self.dtype,
        });
        Tensor {
            id,
            shape: output_shape,
            dtype: self.dtype,
        }
    }

    /// Global average pooling over spatial dims — averages H and W, producing [N, C, 1, 1].
    ///
    /// Input must be rank 4 (NCHW layout).
    pub fn global_avg_pool(&self) -> Tensor {
        assert_eq!(
            self.shape.rank(),
            4,
            "global_avg_pool requires rank-4 input (NCHW)"
        );
        let n = self.shape.0[0];
        let c = self.shape.0[1];
        let output_shape = Shape(vec![n, c, 1, 1]);
        let id = trace::record(Op::GlobalAvgPool {
            input: self.id,
            shape: output_shape.clone(),
            dtype: self.dtype,
        });
        Tensor {
            id,
            shape: output_shape,
            dtype: self.dtype,
        }
    }

    /// Batch normalization: `scale * (x - mean) / sqrt(var + epsilon) + bias`.
    ///
    /// Input must be at least rank 2 (channel dimension is dim 1). All
    /// parameter tensors (`scale`, `bias`, `mean`, `var`) must have the same
    /// dtype as the input and shape `[C]` where `C = input.shape()[1]`.
    pub fn batch_norm(
        &self,
        scale: &Tensor,
        bias: &Tensor,
        mean: &Tensor,
        var: &Tensor,
        epsilon: f64,
    ) -> Tensor {
        assert!(
            self.shape.rank() >= 2,
            "batch_norm requires rank >= 2, got {}",
            self.shape.rank()
        );
        assert_eq!(
            self.dtype, scale.dtype,
            "dtype mismatch in batch_norm (scale)"
        );
        assert_eq!(
            self.dtype, bias.dtype,
            "dtype mismatch in batch_norm (bias)"
        );
        assert_eq!(
            self.dtype, mean.dtype,
            "dtype mismatch in batch_norm (mean)"
        );
        assert_eq!(self.dtype, var.dtype, "dtype mismatch in batch_norm (var)");

        let id = trace::record(Op::BatchNorm {
            input: self.id,
            scale: scale.id,
            bias: bias.id,
            mean: mean.id,
            var: var.id,
            epsilon,
            shape: self.shape.clone(),
            dtype: self.dtype,
        });
        Tensor {
            id,
            shape: self.shape.clone(),
            dtype: self.dtype,
        }
    }

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
        assert!(
            self.shape.rank() >= 2,
            "matmul requires rank-2 tensors (got rank {})",
            self.shape.rank()
        );
        assert!(
            rhs.shape.rank() >= 2,
            "matmul requires rank-2 tensors (got rank {})",
            rhs.shape.rank()
        );
        assert_eq!(self.dtype, rhs.dtype, "dtype mismatch in matmul");

        let lhs_rank = self.shape.rank();
        let rhs_rank = rhs.shape.rank();

        let m = self.shape.0[lhs_rank - 2];
        let k = self.shape.0[lhs_rank - 1];
        let k2 = rhs.shape.0[rhs_rank - 2];
        let n = rhs.shape.0[rhs_rank - 1];
        assert_eq!(
            k, k2,
            "inner dimensions mismatch in matmul: last dims [{}, {}] x [{}, {}]",
            m, k, k2, n
        );

        // Broadcast the batch dimensions (all dims except the last two).
        let lhs_batch = Shape(self.shape.0[..lhs_rank - 2].to_vec());
        let rhs_batch = Shape(rhs.shape.0[..rhs_rank - 2].to_vec());
        let batch_shape = if lhs_batch.0.is_empty() && rhs_batch.0.is_empty() {
            Shape(vec![])
        } else if lhs_batch.0.is_empty() {
            rhs_batch
        } else if rhs_batch.0.is_empty() {
            lhs_batch
        } else {
            lhs_batch
                .broadcast(&rhs_batch)
                .expect("batch dimensions are not broadcast-compatible for matmul")
        };

        let mut out_dims = batch_shape.0;
        out_dims.push(m);
        out_dims.push(n);
        let out_shape = Shape(out_dims);

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
            shape_tensor: None,
            shape: shape.clone(),
            dtype: self.dtype,
        });
        Tensor {
            id,
            shape,
            dtype: self.dtype,
        }
    }

    /// Dynamic reshape: shape comes from a traced tensor at runtime.
    ///
    /// `shape_tensor` is a 1-D I64 tensor whose values are the output dimensions.
    /// `output_shape` is the statically-inferred output shape (may contain DIM_DYNAMIC).
    pub fn reshape_with_tensor(&self, shape_tensor: &Tensor, output_shape: Shape) -> Tensor {
        let id = trace::record(Op::Reshape {
            input: self.id,
            target_shape: output_shape.0.clone(),
            shape_tensor: Some(shape_tensor.id),
            shape: output_shape.clone(),
            dtype: self.dtype,
        });
        Tensor {
            id,
            shape: output_shape,
            dtype: self.dtype,
        }
    }

    /// Extract the runtime shape of this tensor as a 1-D I64 tensor.
    ///
    /// The output shape is always static `[rank]`. Suitable for models with
    /// dynamic dimensions where the shape must be computed at runtime.
    pub fn shape_of(&self) -> Tensor {
        let rank = self.shape.rank() as u64;
        let out_shape = Shape(vec![rank]);
        let id = trace::record(Op::ShapeOf {
            input: self.id,
            shape: out_shape.clone(),
            dtype: DType::I64,
        });
        Tensor {
            id,
            shape: out_shape,
            dtype: DType::I64,
        }
    }

    /// Create a tensor filled with `fill_value` with shape described by `shape_tensor`.
    ///
    /// `shape_tensor` is a 1-D I64 tensor whose elements give the output shape at runtime.
    /// `output_shape` is the statically-inferred output shape (may contain DIM_DYNAMIC).
    /// `dtype` is the element type of the output tensor.
    pub fn constant_of_shape(
        shape_tensor: &Tensor,
        fill_value: f64,
        output_shape: Shape,
        dtype: DType,
    ) -> Tensor {
        let id = trace::record(Op::ConstantOfShape {
            shape_input: shape_tensor.id,
            fill_value,
            shape: output_shape.clone(),
            dtype,
        });
        Tensor {
            id,
            shape: output_shape,
            dtype,
        }
    }

    /// Create an arange tensor: [start, start+delta, ...) up to (not including) limit.
    ///
    /// `start`, `limit`, `delta` are scalar (1-element) tensors with the same dtype.
    /// `output_shape` is typically `[DIM_DYNAMIC]` when inputs are non-constant.
    pub fn range(
        start: &Tensor,
        limit: &Tensor,
        delta: &Tensor,
        output_shape: Shape,
        dtype: DType,
    ) -> Tensor {
        let id = trace::record(Op::Range {
            start: start.id,
            limit: limit.id,
            delta: delta.id,
            shape: output_shape.clone(),
            dtype,
        });
        Tensor {
            id,
            shape: output_shape,
            dtype,
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

    /// Element-wise power: `self ^ exponent` (broadcast-compatible, float only).
    pub fn pow(&self, exponent: &Tensor) -> Tensor {
        assert_eq!(self.dtype, exponent.dtype, "dtype mismatch in pow");
        assert!(
            matches!(self.dtype, DType::F32 | DType::F64),
            "pow requires float dtype, got {:?}",
            self.dtype
        );
        let shape = self
            .shape
            .broadcast(&exponent.shape)
            .unwrap_or_else(|e| panic!("broadcast failed in pow: {e}"));
        let id = trace::record(Op::Pow {
            lhs: self.id,
            rhs: exponent.id,
            shape: shape.clone(),
            dtype: self.dtype,
        });
        Tensor {
            id,
            shape,
            dtype: self.dtype,
        }
    }

    /// Element-wise square root (float only).
    pub fn sqrt(&self) -> Tensor {
        assert!(
            matches!(self.dtype, DType::F32 | DType::F64),
            "sqrt requires float dtype, got {:?}",
            self.dtype
        );
        let id = trace::record(Op::Sqrt {
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

    /// Cast tensor elements to `target_dtype`. If `target_dtype == self.dtype`, returns self unchanged.
    pub fn cast(&self, target_dtype: DType) -> Tensor {
        if self.dtype == target_dtype {
            return self.clone();
        }
        let id = trace::record(Op::Cast {
            input: self.id,
            target_dtype,
            shape: self.shape.clone(),
            dtype: target_dtype,
        });
        Tensor {
            id,
            shape: self.shape.clone(),
            dtype: target_dtype,
        }
    }

    /// Reduce by computing mean along `axes`. Negative axes are relative to rank.
    pub fn reduce_mean(&self, axes: &[i64], keepdim: bool) -> Tensor {
        assert!(!axes.is_empty(), "reduce_mean: axes must not be empty");
        let rank = self.shape.rank();

        // Normalize axes and compute output shape.
        let mut normalized: Vec<usize> = axes
            .iter()
            .map(|&a| {
                let r = if a < 0 {
                    (rank as i64 + a) as usize
                } else {
                    a as usize
                };
                assert!(r < rank, "reduce_mean: axis {a} out of range for rank {rank}");
                r
            })
            .collect();
        normalized.sort_unstable();
        normalized.dedup();

        // Compute output shape after all reductions.
        let mut out_dims = self.shape.0.clone();
        for &ax in normalized.iter().rev() {
            if keepdim {
                out_dims[ax] = 1;
            } else {
                out_dims.remove(ax);
            }
        }
        let out_shape = Shape(out_dims);

        // Store axes as i64 (already validated above).
        let axes_i64: Vec<i64> = normalized.iter().map(|&a| a as i64).collect();

        let id = trace::record(Op::ReduceMean {
            input: self.id,
            axes: axes_i64,
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

    /// Gather: index into `self` along `axis` using `indices`.
    ///
    /// Output shape = self.shape[0..axis] + indices.shape + self.shape[axis+1..].
    /// `indices` must have dtype I32 or I64. `axis` may be negative.
    pub fn gather(&self, indices: &Tensor, axis: i64) -> Tensor {
        let rank = self.shape.rank() as i64;
        let ax = if axis < 0 { axis + rank } else { axis };
        assert!(ax >= 0 && ax < rank, "gather: axis {axis} out of range for rank {rank}");
        let ax = ax as usize;
        assert!(
            matches!(indices.dtype, DType::I32 | DType::I64),
            "gather: indices must have I32 or I64 dtype, got {:?}",
            indices.dtype
        );

        // output_shape = data.shape[0..ax] + indices.shape + data.shape[ax+1..]
        let mut out_dims: Vec<u64> = Vec::new();
        out_dims.extend_from_slice(&self.shape.0[..ax]);
        out_dims.extend_from_slice(&indices.shape.0);
        out_dims.extend_from_slice(&self.shape.0[ax + 1..]);
        let out_shape = Shape(out_dims);

        let id = trace::record(Op::Gather {
            input: self.id,
            indices: indices.id,
            axis: ax as i64,
            shape: out_shape.clone(),
            dtype: self.dtype,
        });
        Tensor {
            id,
            shape: out_shape,
            dtype: self.dtype,
        }
    }

    /// Slice `self` along specified axes.
    ///
    /// `starts`, `ends`, `axes`, and `steps` must all have the same length.
    /// Negative indices and steps of 1 are assumed (all steps must be positive).
    /// Negative starts/ends are normalized against the corresponding axis size.
    pub fn slice(
        &self,
        starts: &[i64],
        ends: &[i64],
        axes: &[i64],
        steps: &[i64],
    ) -> Tensor {
        let rank = self.shape.rank();
        assert_eq!(starts.len(), ends.len(), "slice: starts and ends must have equal length");
        assert_eq!(starts.len(), axes.len(), "slice: starts and axes must have equal length");
        assert_eq!(starts.len(), steps.len(), "slice: starts and steps must have equal length");

        // Normalize negative axes and compute per-axis start/end.
        let mut norm_starts = starts.to_vec();
        let mut norm_ends = ends.to_vec();
        let mut norm_axes = axes.to_vec();
        for i in 0..norm_axes.len() {
            let ax = if norm_axes[i] < 0 {
                norm_axes[i] + rank as i64
            } else {
                norm_axes[i]
            };
            assert!(ax >= 0 && ax < rank as i64, "slice: axis {} out of range", norm_axes[i]);
            norm_axes[i] = ax;
            let dim_size = self.shape.0[ax as usize] as i64;
            // Normalize negative start.
            if norm_starts[i] < 0 {
                norm_starts[i] = (norm_starts[i] + dim_size).max(0);
            }
            // Clamp start to [0, dim_size].
            norm_starts[i] = norm_starts[i].clamp(0, dim_size);
            // Normalize negative end.
            if norm_ends[i] < 0 {
                norm_ends[i] = (norm_ends[i] + dim_size).max(0);
            }
            // Clamp end to [0, dim_size].
            norm_ends[i] = norm_ends[i].clamp(0, dim_size);
        }

        // Compute output shape: default axes to identity, then apply per-axis slices.
        let mut out_dims: Vec<u64> = self.shape.0.clone();
        for i in 0..norm_axes.len() {
            let ax = norm_axes[i] as usize;
            let step = steps[i];
            assert!(step > 0, "slice: only positive steps are supported");
            let len = (norm_ends[i] - norm_starts[i]).max(0);
            out_dims[ax] = ((len + step - 1) / step) as u64;
        }
        let out_shape = Shape(out_dims);

        let id = trace::record(Op::Slice {
            input: self.id,
            starts: norm_starts,
            ends: norm_ends,
            axes: norm_axes,
            steps: steps.to_vec(),
            shape: out_shape.clone(),
            dtype: self.dtype,
        });
        Tensor {
            id,
            shape: out_shape,
            dtype: self.dtype,
        }
    }

    /// Concatenate a list of tensors along `axis`. All tensors must have the same
    /// rank and dtype; all dimensions except `axis` must match.
    ///
    /// `axis` may be negative.
    pub fn concat(tensors: &[&Tensor], axis: i64) -> Tensor {
        assert!(!tensors.is_empty(), "concat: at least one tensor required");
        let dtype = tensors[0].dtype;
        let rank = tensors[0].shape.rank();
        let ax = if axis < 0 {
            (rank as i64 + axis) as usize
        } else {
            axis as usize
        };
        assert!(ax < rank, "concat: axis {axis} out of range for rank {rank}");

        for t in tensors.iter().skip(1) {
            assert_eq!(t.dtype, dtype, "concat: all tensors must have the same dtype");
            assert_eq!(t.shape.rank(), rank, "concat: all tensors must have the same rank");
        }

        // Compute output shape: sum `axis` dimension, check all others match.
        let mut out_dims = tensors[0].shape.0.clone();
        for t in tensors.iter().skip(1) {
            for (d, dim) in out_dims.iter_mut().enumerate() {
                if d == ax {
                    *dim += t.shape.0[d];
                } else {
                    assert_eq!(
                        *dim, t.shape.0[d],
                        "concat: dimension {d} mismatch: {} vs {}",
                        dim, t.shape.0[d]
                    );
                }
            }
        }
        let out_shape = Shape(out_dims);

        let inputs: Vec<NodeId> = tensors.iter().map(|t| t.id).collect();
        let id = trace::record(Op::Concat {
            inputs,
            axis: ax as i64,
            shape: out_shape.clone(),
            dtype,
        });
        Tensor {
            id,
            shape: out_shape,
            dtype,
        }
    }

    /// Permute tensor dimensions. `perm` must be a permutation of 0..rank.
    /// Negative values are not supported (normalize before calling).
    pub fn transpose(&self, perm: &[i64]) -> Tensor {
        let rank = self.shape.rank();
        assert_eq!(perm.len(), rank, "transpose: perm length must match tensor rank");

        let out_dims: Vec<u64> = perm.iter().map(|&p| self.shape.0[p as usize]).collect();
        let out_shape = Shape(out_dims);

        let id = trace::record(Op::Transpose {
            input: self.id,
            perm: perm.to_vec(),
            shape: out_shape.clone(),
            dtype: self.dtype,
        });
        Tensor {
            id,
            shape: out_shape,
            dtype: self.dtype,
        }
    }

    /// Element-wise conditional selection.
    ///
    /// `condition` is treated as boolean (non-zero = true). It must have dtype I64.
    /// `x` and `y` must have the same dtype; the output shape is the broadcast of all three.
    pub fn where_select(condition: &Tensor, x: &Tensor, y: &Tensor) -> Tensor {
        assert_eq!(x.dtype, y.dtype, "where_select: x and y must have the same dtype");
        assert_eq!(
            condition.dtype,
            DType::I64,
            "where_select: condition must be I64, got {:?}",
            condition.dtype
        );

        // Compute broadcast output shape across all three inputs.
        let cond_xy_shape = condition
            .shape
            .broadcast(&x.shape)
            .unwrap_or_else(|e| panic!("broadcast failed in where_select (cond/x): {e}"));
        let out_shape = cond_xy_shape
            .broadcast(&y.shape)
            .unwrap_or_else(|e| panic!("broadcast failed in where_select (cond_x/y): {e}"));

        let id = trace::record(Op::Where {
            condition: condition.id,
            x: x.id,
            y: y.id,
            shape: out_shape.clone(),
            dtype: x.dtype,
        });
        Tensor {
            id,
            shape: out_shape,
            dtype: x.dtype,
        }
    }

    /// One-call API: freeze the active trace, compile, and execute.
    ///
    /// The caller must have called `begin_trace()` before building the
    /// computation graph. `inputs` must be in the same order as the
    /// `Tensor::new` calls that created the graph's input placeholders.
    pub fn eval(&self, inputs: &[Buffer]) -> Buffer {
        let trace = crate::trace::take_trace();
        let compiled =
            crate::Compiler::compile(&trace, &[self.id], None).expect("compilation failed in eval");
        let mut outputs = compiled.run(&inputs.iter().collect::<Vec<&Buffer>>());
        outputs.remove(0)
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

    // ── MaxPool2d trace tests (task 11.3) ────────────────────────────────────

    #[test]
    fn max_pool2d_records_correct_op_and_shape() {
        begin_trace();
        // input: [1, 1, 4, 4], kernel=2x2, stride=2, no pad → output: [1, 1, 2, 2]
        let input = Tensor::new(&[1, 1, 4, 4], DType::F32);
        let out = input.max_pool2d([2, 2], [2, 2], [0, 0, 0, 0]);
        let trace = take_trace();
        assert_eq!(out.shape().0, vec![1u64, 1, 2, 2]);
        let op = &trace.ops()[out.id.0 as usize];
        match op {
            Op::MaxPool2d {
                kernel_size,
                strides,
                pads,
                shape,
                dtype,
                ..
            } => {
                assert_eq!(*kernel_size, [2u64, 2]);
                assert_eq!(*strides, [2u64, 2]);
                assert_eq!(*pads, [0u64, 0, 0, 0]);
                assert_eq!(shape.0, vec![1u64, 1, 2, 2]);
                assert_eq!(*dtype, DType::F32);
            }
            _ => panic!("expected Op::MaxPool2d, got {:?}", op),
        }
    }

    #[test]
    fn max_pool2d_output_shape_with_padding() {
        begin_trace();
        // input: [1, 1, 4, 4], kernel=2x2, stride=1, pad=1 all sides
        // OH = (4 + 1 + 1 - 2) / 1 + 1 = 4/1 + 1 = 5
        let input = Tensor::new(&[1, 1, 4, 4], DType::F32);
        let out = input.max_pool2d([2, 2], [1, 1], [1, 1, 1, 1]);
        let _ = take_trace();
        assert_eq!(out.shape().0, vec![1u64, 1, 5, 5]);
    }

    #[test]
    #[should_panic(expected = "max_pool2d requires rank-4 input")]
    fn max_pool2d_rank3_input_panics() {
        begin_trace();
        let input = Tensor::new(&[1, 4, 4], DType::F32);
        let _ = input.max_pool2d([2, 2], [1, 1], [0, 0, 0, 0]);
        let _ = take_trace();
    }

    // ── BatchNorm trace tests (task 85) ──────────────────────────────────────

    #[test]
    fn batch_norm_records_correct_op() {
        begin_trace();
        // input: [2, 3, 4, 4] — batch=2, channels=3, spatial=4x4
        let input = Tensor::new(&[2, 3, 4, 4], DType::F32);
        let scale = Tensor::new(&[3], DType::F32);
        let bias = Tensor::new(&[3], DType::F32);
        let mean = Tensor::new(&[3], DType::F32);
        let var = Tensor::new(&[3], DType::F32);
        let out = input.batch_norm(&scale, &bias, &mean, &var, 1e-5);
        let trace = take_trace();
        // 5 inputs + 1 batch_norm op
        assert_eq!(trace.ops().len(), 6);
        assert_eq!(trace.input_count(), 5);
        let op = &trace.ops()[out.id.0 as usize];
        match op {
            Op::BatchNorm {
                epsilon,
                shape,
                dtype,
                ..
            } => {
                assert!((epsilon - 1e-5f64).abs() < 1e-10, "epsilon mismatch");
                assert_eq!(shape.0, vec![2u64, 3, 4, 4]);
                assert_eq!(*dtype, DType::F32);
            }
            _ => panic!("expected Op::BatchNorm, got {:?}", op),
        }
    }

    #[test]
    fn batch_norm_output_shape_matches_input() {
        begin_trace();
        let input = Tensor::new(&[1, 4, 8, 8], DType::F32);
        let scale = Tensor::new(&[4], DType::F32);
        let bias = Tensor::new(&[4], DType::F32);
        let mean = Tensor::new(&[4], DType::F32);
        let var = Tensor::new(&[4], DType::F32);
        let out = input.batch_norm(&scale, &bias, &mean, &var, 1e-5);
        let _ = take_trace();
        assert_eq!(out.shape().0, vec![1u64, 4, 8, 8]);
    }

    #[test]
    fn batch_norm_records_five_inputs() {
        begin_trace();
        let input = Tensor::new(&[1, 2, 3, 3], DType::F32);
        let scale = Tensor::new(&[2], DType::F32);
        let bias = Tensor::new(&[2], DType::F32);
        let mean = Tensor::new(&[2], DType::F32);
        let var = Tensor::new(&[2], DType::F32);
        let out = input.batch_norm(&scale, &bias, &mean, &var, 1e-3);
        let trace = take_trace();
        let op = &trace.ops()[out.id.0 as usize];
        match op {
            Op::BatchNorm {
                input: inp_id,
                scale: scale_id,
                bias: bias_id,
                mean: mean_id,
                var: var_id,
                epsilon,
                ..
            } => {
                assert_eq!(inp_id.0, 0);
                assert_eq!(scale_id.0, 1);
                assert_eq!(bias_id.0, 2);
                assert_eq!(mean_id.0, 3);
                assert_eq!(var_id.0, 4);
                assert!((epsilon - 1e-3f64).abs() < 1e-10);
            }
            _ => panic!("expected Op::BatchNorm"),
        }
    }

    #[test]
    #[should_panic(expected = "batch_norm requires rank >= 2")]
    fn batch_norm_rank1_panics() {
        begin_trace();
        let input = Tensor::new(&[4], DType::F32);
        let scale = Tensor::new(&[4], DType::F32);
        let bias = Tensor::new(&[4], DType::F32);
        let mean = Tensor::new(&[4], DType::F32);
        let var = Tensor::new(&[4], DType::F32);
        let _ = input.batch_norm(&scale, &bias, &mean, &var, 1e-5);
        let _ = take_trace();
    }

    #[test]
    #[should_panic(expected = "dtype mismatch in batch_norm")]
    fn batch_norm_dtype_mismatch_panics() {
        begin_trace();
        let input = Tensor::new(&[1, 2, 4, 4], DType::F32);
        let scale = Tensor::new(&[2], DType::I32);
        let bias = Tensor::new(&[2], DType::F32);
        let mean = Tensor::new(&[2], DType::F32);
        let var = Tensor::new(&[2], DType::F32);
        let _ = input.batch_norm(&scale, &bias, &mean, &var, 1e-5);
        let _ = take_trace();
    }

    // ── GlobalAvgPool trace tests (task 91) ──────────────────────────────────

    #[test]
    fn global_avg_pool_records_correct_op() {
        begin_trace();
        let input = Tensor::new(&[2, 4, 8, 8], DType::F32);
        let out = input.global_avg_pool();
        let trace = take_trace();
        // 1 input op + 1 global_avg_pool op
        assert_eq!(trace.ops().len(), 2);
        assert_eq!(trace.input_count(), 1);
        let op = &trace.ops()[out.id.0 as usize];
        match op {
            Op::GlobalAvgPool { shape, dtype, .. } => {
                assert_eq!(shape.0, vec![2u64, 4, 1, 1]);
                assert_eq!(*dtype, DType::F32);
            }
            _ => panic!("expected Op::GlobalAvgPool, got {:?}", op),
        }
    }

    #[test]
    fn global_avg_pool_output_shape_is_n_c_1_1() {
        begin_trace();
        // Arbitrary spatial dimensions should both collapse to 1.
        let input = Tensor::new(&[3, 16, 7, 7], DType::F32);
        let out = input.global_avg_pool();
        let _ = take_trace();
        assert_eq!(out.shape().0, vec![3u64, 16, 1, 1]);
    }

    #[test]
    #[should_panic(expected = "global_avg_pool requires rank-4 input")]
    fn global_avg_pool_rank3_input_panics() {
        begin_trace();
        let input = Tensor::new(&[1, 4, 4], DType::F32);
        let _ = input.global_avg_pool();
        let _ = take_trace();
    }

    // ── Conv2d trace tests (task 10.3) ───────────────────────────────────────

    #[test]
    fn conv2d_records_correct_op_and_shape() {
        begin_trace();
        // input: [1, 1, 4, 4], kernel: [1, 1, 3, 3]
        // no padding, stride=1, dilation=1 → output: [1, 1, 2, 2]
        let input = Tensor::new(&[1, 1, 4, 4], DType::F32);
        let kernel = Tensor::new(&[1, 1, 3, 3], DType::F32);
        let out = input.conv2d(&kernel, None, [1, 1], [0, 0, 0, 0], [1, 1]);
        let trace = take_trace();
        assert_eq!(out.shape().0, vec![1u64, 1, 2, 2]);
        let op = &trace.ops()[out.id.0 as usize];
        match op {
            Op::Conv2d {
                bias,
                strides,
                pads,
                dilations,
                shape,
                dtype,
                ..
            } => {
                assert!(bias.is_none());
                assert_eq!(*strides, [1u64, 1]);
                assert_eq!(*pads, [0u64, 0, 0, 0]);
                assert_eq!(*dilations, [1u64, 1]);
                assert_eq!(shape.0, vec![1u64, 1, 2, 2]);
                assert_eq!(*dtype, DType::F32);
            }
            _ => panic!("expected Op::Conv2d, got {:?}", op),
        }
    }

    #[test]
    fn conv2d_records_bias() {
        begin_trace();
        let input = Tensor::new(&[1, 2, 5, 5], DType::F32);
        let kernel = Tensor::new(&[3, 2, 3, 3], DType::F32);
        let bias = Tensor::new(&[3], DType::F32);
        let out = input.conv2d(&kernel, Some(&bias), [1, 1], [0, 0, 0, 0], [1, 1]);
        let trace = take_trace();
        // output: [1, 3, 3, 3]
        assert_eq!(out.shape().0, vec![1u64, 3, 3, 3]);
        let op = &trace.ops()[out.id.0 as usize];
        match op {
            Op::Conv2d { bias, .. } => {
                assert!(bias.is_some(), "expected bias to be recorded");
            }
            _ => panic!("expected Op::Conv2d"),
        }
    }

    #[test]
    fn conv2d_output_shape_with_padding() {
        begin_trace();
        // input: [1, 1, 4, 4], kernel: [1, 1, 3, 3], pad=1 on all sides, stride=1
        // OH = (4 + 1 + 1 - 1*(3-1) - 1) / 1 + 1 = (4 + 2 - 2 - 1) / 1 + 1 = 3/1 + 1 = 4
        let input = Tensor::new(&[1, 1, 4, 4], DType::F32);
        let kernel = Tensor::new(&[1, 1, 3, 3], DType::F32);
        let out = input.conv2d(&kernel, None, [1, 1], [1, 1, 1, 1], [1, 1]);
        let _ = take_trace();
        assert_eq!(out.shape().0, vec![1u64, 1, 4, 4]);
    }

    #[test]
    fn conv2d_output_shape_with_stride() {
        begin_trace();
        // input: [1, 1, 4, 4], kernel: [1, 1, 3, 3], no pad, stride=2
        // OH = (4 + 0 + 0 - 1*(3-1) - 1) / 2 + 1 = (4 - 2 - 1) / 2 + 1 = 1/2 + 1 = 1
        let input = Tensor::new(&[1, 1, 4, 4], DType::F32);
        let kernel = Tensor::new(&[1, 1, 3, 3], DType::F32);
        let out = input.conv2d(&kernel, None, [2, 2], [0, 0, 0, 0], [1, 1]);
        let _ = take_trace();
        assert_eq!(out.shape().0, vec![1u64, 1, 1, 1]);
    }

    #[test]
    #[should_panic(expected = "conv2d requires rank-4 input")]
    fn conv2d_rank3_input_panics() {
        begin_trace();
        let input = Tensor::new(&[1, 4, 4], DType::F32);
        let kernel = Tensor::new(&[1, 1, 3, 3], DType::F32);
        let _ = input.conv2d(&kernel, None, [1, 1], [0, 0, 0, 0], [1, 1]);
        let _ = take_trace();
    }

    #[test]
    #[should_panic(expected = "input channels")]
    fn conv2d_channel_mismatch_panics() {
        begin_trace();
        let input = Tensor::new(&[1, 2, 4, 4], DType::F32);
        let kernel = Tensor::new(&[1, 3, 3, 3], DType::F32); // CI=3, but input has CI=2
        let _ = input.conv2d(&kernel, None, [1, 1], [0, 0, 0, 0], [1, 1]);
        let _ = take_trace();
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
        let compiled = Compiler::compile(&trace, &[b.id], None).expect("compile failed");
        // [[1, 2, 3], [1, 2, 3]]
        let a_buf = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3], DType::F32);
        let result = compiled.run(&[&a_buf]);
        let out = result[0].as_slice::<f32>();
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
        let compiled = Compiler::compile(&trace, &[b.id], None).expect("compile failed");
        // Large values that would overflow exp() without the max subtraction trick
        let a_buf = Buffer::from_slice::<f32>(&[1000.0, 1000.0, 1000.0], &[3], DType::F32);
        let result = compiled.run(&[&a_buf]);
        let out = result[0].as_slice::<f32>();
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

    // ── Dtype mismatch panic tests ───────────────────────────────────────────

    #[test]
    #[should_panic(expected = "dtype mismatch")]
    fn add_dtype_mismatch_panics() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F32);
        let b = Tensor::new(&[4], DType::I32);
        let _ = &a + &b;
        let _ = take_trace();
    }

    #[test]
    #[should_panic(expected = "dtype mismatch")]
    fn sub_dtype_mismatch_panics() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F32);
        let b = Tensor::new(&[4], DType::I32);
        let _ = &a - &b;
        let _ = take_trace();
    }

    #[test]
    #[should_panic(expected = "dtype mismatch")]
    fn mul_dtype_mismatch_panics() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F32);
        let b = Tensor::new(&[4], DType::I32);
        let _ = &a * &b;
        let _ = take_trace();
    }

    #[test]
    #[should_panic(expected = "dtype mismatch")]
    fn div_dtype_mismatch_panics() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F32);
        let b = Tensor::new(&[4], DType::I32);
        let _ = &a / &b;
        let _ = take_trace();
    }

    #[test]
    #[should_panic(expected = "dtype mismatch")]
    fn matmul_dtype_mismatch_panics() {
        begin_trace();
        let a = Tensor::new(&[2, 3], DType::F32);
        let b = Tensor::new(&[3, 4], DType::I32);
        let _ = a.matmul(&b);
        let _ = take_trace();
    }

    // ── Broadcast incompatibility panic tests ────────────────────────────────

    #[test]
    #[should_panic(expected = "broadcast failed")]
    fn sub_incompatible_shapes_panics() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F32);
        let b = Tensor::new(&[8], DType::F32);
        let _ = &a - &b;
        let _ = take_trace();
    }

    #[test]
    #[should_panic(expected = "broadcast failed")]
    fn mul_incompatible_shapes_panics() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F32);
        let b = Tensor::new(&[8], DType::F32);
        let _ = &a * &b;
        let _ = take_trace();
    }

    #[test]
    #[should_panic(expected = "broadcast failed")]
    fn div_incompatible_shapes_panics() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F32);
        let b = Tensor::new(&[8], DType::F32);
        let _ = &a / &b;
        let _ = take_trace();
    }

    // ── Exp/Tanh integer dtype panic tests ───────────────────────────────────

    #[test]
    #[should_panic(expected = "exp requires float dtype")]
    fn exp_integer_dtype_panics() {
        begin_trace();
        let a = Tensor::new(&[4], DType::I32);
        let _ = a.exp();
        let _ = take_trace();
    }

    #[test]
    #[should_panic(expected = "tanh requires float dtype")]
    fn tanh_integer_dtype_panics() {
        begin_trace();
        let a = Tensor::new(&[4], DType::I32);
        let _ = a.tanh();
        let _ = take_trace();
    }

    // ── Matmul dimension/rank panic tests ────────────────────────────────────

    #[test]
    #[should_panic(expected = "inner dimensions mismatch")]
    fn matmul_inner_dim_mismatch_panics() {
        begin_trace();
        let a = Tensor::new(&[2, 3], DType::F32);
        let b = Tensor::new(&[5, 4], DType::F32);
        let _ = a.matmul(&b);
        let _ = take_trace();
    }

    #[test]
    #[should_panic(expected = "matmul requires rank-2")]
    fn matmul_rank1_panics() {
        begin_trace();
        let a = Tensor::new(&[3], DType::F32);
        let b = Tensor::new(&[3, 4], DType::F32);
        let _ = a.matmul(&b);
        let _ = take_trace();
    }

    // ── Reduce out-of-range panic tests ──────────────────────────────────────

    #[test]
    #[should_panic(expected = "out of range")]
    fn reduce_sum_dim_out_of_range_panics() {
        begin_trace();
        let a = Tensor::new(&[2, 3], DType::F32);
        let _ = a.reduce_sum(10, false);
        let _ = take_trace();
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn reduce_max_dim_out_of_range_panics() {
        begin_trace();
        let a = Tensor::new(&[2, 3], DType::F32);
        let _ = a.reduce_max(5, false);
        let _ = take_trace();
    }

    // ── Gemm rank assertion tests ────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "gemm requires rank-2")]
    fn gemm_rank1_panics() {
        begin_trace();
        let a = Tensor::new(&[6], DType::F32);
        let b = Tensor::new(&[3, 4], DType::F32);
        let _ = a.gemm(&b, None, 1.0, 1.0, false, false);
        let _ = take_trace();
    }

    // ── Reshape edge case tests ──────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "dimension must be positive")]
    fn reshape_zero_dim_panics() {
        begin_trace();
        let a = Tensor::new(&[12], DType::F32);
        let _ = a.reshape(&[0, 12]);
        let _ = take_trace();
    }

    // ── Overflow guard regression tests ──────────────────────────────────────

    #[test]
    #[should_panic(expected = "effective kernel height")]
    fn conv2d_kernel_too_large_panics() {
        begin_trace();
        let input = Tensor::new(&[1, 1, 3, 3], DType::F32);
        let kernel = Tensor::new(&[1, 1, 5, 5], DType::F32);
        let _ = input.conv2d(&kernel, None, [1, 1], [0, 0, 0, 0], [1, 1]);
        let _ = take_trace();
    }

    #[test]
    #[should_panic(expected = "kernel height")]
    fn max_pool2d_kernel_too_large_panics() {
        begin_trace();
        let input = Tensor::new(&[1, 1, 3, 3], DType::F32);
        let _ = input.max_pool2d([5, 5], [1, 1], [0, 0, 0, 0]);
        let _ = take_trace();
    }

    #[test]
    #[should_panic(expected = "effective kernel height")]
    fn conv2d_dilated_kernel_too_large_panics() {
        begin_trace();
        let input = Tensor::new(&[1, 1, 3, 3], DType::F32);
        let kernel = Tensor::new(&[1, 1, 2, 2], DType::F32);
        // dilation=3: effective kernel = 3*(2-1)+1 = 4 > 3
        let _ = input.conv2d(&kernel, None, [1, 1], [0, 0, 0, 0], [3, 3]);
        let _ = take_trace();
    }

    // ── Gather trace tests ────────────────────────────────────────────────────

    #[test]
    fn gather_records_correct_shape() {
        begin_trace();
        // data[6, 4], indices[3, 2], axis=0 → output[3, 2, 4]
        let data = Tensor::new(&[6, 4], DType::F32);
        let indices = Tensor::new(&[3, 2], DType::I64);
        let out = data.gather(&indices, 0);
        let _ = take_trace();
        assert_eq!(out.shape().0, vec![3u64, 2, 4]);
        assert_eq!(out.dtype(), DType::F32);
    }

    #[test]
    fn gather_axis1_records_correct_shape() {
        begin_trace();
        // data[4, 10], indices[7], axis=1 → output[4, 7]
        let data = Tensor::new(&[4, 10], DType::F32);
        let indices = Tensor::new(&[7], DType::I64);
        let out = data.gather(&indices, 1);
        let _ = take_trace();
        assert_eq!(out.shape().0, vec![4u64, 7]);
    }

    #[test]
    fn gather_negative_axis() {
        begin_trace();
        // data[2, 3, 4], indices[5], axis=-1 → output[2, 3, 5]
        let data = Tensor::new(&[2, 3, 4], DType::F32);
        let indices = Tensor::new(&[5], DType::I64);
        let out = data.gather(&indices, -1);
        let _ = take_trace();
        assert_eq!(out.shape().0, vec![2u64, 3, 5]);
    }

    // ── Slice trace tests ─────────────────────────────────────────────────────

    #[test]
    fn slice_records_correct_shape() {
        begin_trace();
        // Slice [0..4] of a 1D length-8 tensor → shape [4]
        let a = Tensor::new(&[8], DType::F32);
        let b = a.slice(&[0], &[4], &[0], &[1]);
        let _ = take_trace();
        assert_eq!(b.shape().0, vec![4u64]);
    }

    #[test]
    fn slice_2d_records_correct_shape() {
        begin_trace();
        // data[5, 6], slice rows [1..4], cols [2..5] → shape [3, 3]
        let a = Tensor::new(&[5, 6], DType::F32);
        let b = a.slice(&[1, 2], &[4, 5], &[0, 1], &[1, 1]);
        let _ = take_trace();
        assert_eq!(b.shape().0, vec![3u64, 3]);
    }

    #[test]
    fn slice_strided_records_correct_shape() {
        begin_trace();
        // data[10], slice [0..10] step 2 → shape [5]
        let a = Tensor::new(&[10], DType::F32);
        let b = a.slice(&[0], &[10], &[0], &[2]);
        let _ = take_trace();
        assert_eq!(b.shape().0, vec![5u64]);
    }

    // ── Concat trace tests ────────────────────────────────────────────────────

    #[test]
    fn concat_records_correct_shape_axis0() {
        begin_trace();
        let a = Tensor::new(&[3, 4], DType::F32);
        let b = Tensor::new(&[5, 4], DType::F32);
        let c = Tensor::concat(&[&a, &b], 0);
        let _ = take_trace();
        assert_eq!(c.shape().0, vec![8u64, 4]);
        assert_eq!(c.dtype(), DType::F32);
    }

    #[test]
    fn concat_records_correct_shape_axis1() {
        begin_trace();
        let a = Tensor::new(&[2, 3], DType::F32);
        let b = Tensor::new(&[2, 7], DType::F32);
        let c = Tensor::concat(&[&a, &b], 1);
        let _ = take_trace();
        assert_eq!(c.shape().0, vec![2u64, 10]);
    }

    // ── Transpose trace tests ─────────────────────────────────────────────────

    #[test]
    fn transpose_records_correct_shape() {
        begin_trace();
        let a = Tensor::new(&[2, 3, 4], DType::F32);
        let b = a.transpose(&[2, 0, 1]);
        let _ = take_trace();
        // perm [2,0,1]: out[0]=a[2]=4, out[1]=a[0]=2, out[2]=a[1]=3
        assert_eq!(b.shape().0, vec![4u64, 2, 3]);
    }

    #[test]
    fn transpose_2d_records_correct_shape() {
        begin_trace();
        let a = Tensor::new(&[5, 7], DType::F32);
        let b = a.transpose(&[1, 0]);
        let _ = take_trace();
        assert_eq!(b.shape().0, vec![7u64, 5]);
    }

    // ── Where trace tests ─────────────────────────────────────────────────────

    #[test]
    fn where_records_correct_shape() {
        begin_trace();
        let cond = Tensor::new(&[4], DType::I64);
        let x = Tensor::new(&[4], DType::F32);
        let y = Tensor::new(&[4], DType::F32);
        let out = Tensor::where_select(&cond, &x, &y);
        let _ = take_trace();
        assert_eq!(out.shape().0, vec![4u64]);
        assert_eq!(out.dtype(), DType::F32);
    }

    #[test]
    fn where_records_op() {
        begin_trace();
        let cond = Tensor::new(&[3], DType::I64);
        let x = Tensor::new(&[3], DType::F32);
        let y = Tensor::new(&[3], DType::F32);
        let out = Tensor::where_select(&cond, &x, &y);
        let trace = take_trace();
        let op = &trace.ops()[out.id.0 as usize];
        assert!(
            matches!(op, Op::Where { .. }),
            "expected Op::Where, got {:?}",
            op
        );
    }

    // ── ShapeOf / ConstantOfShape / Range trace tests ─────────────────────────

    #[test]
    fn shape_of_records_correct_shape_and_dtype() {
        begin_trace();
        let a = Tensor::new(&[3, 4, 5], DType::F32);
        let s = a.shape_of();
        let _ = take_trace();
        // Output shape is always [rank] = [3]
        assert_eq!(s.shape().0, vec![3u64]);
        assert_eq!(s.dtype(), DType::I64);
    }

    #[test]
    fn shape_of_records_op() {
        begin_trace();
        let a = Tensor::new(&[2, 7], DType::F32);
        let s = a.shape_of();
        let trace = take_trace();
        let op = &trace.ops()[s.id.0 as usize];
        assert!(
            matches!(op, Op::ShapeOf { .. }),
            "expected Op::ShapeOf, got {:?}",
            op
        );
    }

    #[test]
    fn constant_of_shape_records_correct_shape_and_dtype() {
        use crate::shape::DIM_DYNAMIC;
        begin_trace();
        let shape_t = Tensor::new(&[4], DType::I64);
        let out = Tensor::constant_of_shape(
            &shape_t,
            1.0,
            crate::Shape(vec![1, 12, DIM_DYNAMIC, 64]),
            DType::F32,
        );
        let _ = take_trace();
        assert_eq!(out.shape().0, vec![1, 12, DIM_DYNAMIC, 64]);
        assert_eq!(out.dtype(), DType::F32);
    }

    #[test]
    fn constant_of_shape_records_op() {
        begin_trace();
        let shape_t = Tensor::new(&[2], DType::I64);
        let out = Tensor::constant_of_shape(&shape_t, 0.0, crate::Shape(vec![4, 4]), DType::F32);
        let trace = take_trace();
        let op = &trace.ops()[out.id.0 as usize];
        assert!(
            matches!(op, Op::ConstantOfShape { .. }),
            "expected Op::ConstantOfShape, got {:?}",
            op
        );
    }

    #[test]
    fn range_records_correct_shape_and_dtype() {
        use crate::shape::DIM_DYNAMIC;
        begin_trace();
        let start = Tensor::new(&[1], DType::I64);
        let limit = Tensor::new(&[1], DType::I64);
        let delta = Tensor::new(&[1], DType::I64);
        let out = Tensor::range(&start, &limit, &delta, crate::Shape(vec![DIM_DYNAMIC]), DType::I64);
        let _ = take_trace();
        assert_eq!(out.shape().0, vec![DIM_DYNAMIC]);
        assert_eq!(out.dtype(), DType::I64);
    }

    #[test]
    fn range_records_op() {
        use crate::shape::DIM_DYNAMIC;
        begin_trace();
        let start = Tensor::new(&[1], DType::I64);
        let limit = Tensor::new(&[1], DType::I64);
        let delta = Tensor::new(&[1], DType::I64);
        let out = Tensor::range(&start, &limit, &delta, crate::Shape(vec![DIM_DYNAMIC]), DType::I64);
        let trace = take_trace();
        let op = &trace.ops()[out.id.0 as usize];
        assert!(
            matches!(op, Op::Range { .. }),
            "expected Op::Range, got {:?}",
            op
        );
    }

    #[test]
    fn reshape_with_tensor_records_op_with_shape_tensor() {
        use crate::shape::DIM_DYNAMIC;
        begin_trace();
        let data = Tensor::new(&[6], DType::F32);
        let shape_t = Tensor::new(&[2], DType::I64);
        let out = data.reshape_with_tensor(&shape_t, crate::Shape(vec![DIM_DYNAMIC, DIM_DYNAMIC]));
        let trace = take_trace();
        let op = &trace.ops()[out.id.0 as usize];
        match op {
            Op::Reshape { shape_tensor, .. } => {
                assert!(shape_tensor.is_some(), "shape_tensor should be Some");
            }
            _ => panic!("expected Op::Reshape, got {:?}", op),
        }
    }
}
