use super::*;

impl<'c> GraphBuilder<'c> {
    /// RMSNorm: x * rsqrt(mean(x², axis=-1, keepdim=true) + eps) * weight.
    ///
    /// `weight` must be broadcastable to `input` (typically shape `[hidden_size]`).
    pub fn emit_rms_norm(
        &mut self,
        input: &Tensor<'c>,
        weight: &Tensor<'c>,
        eps: f32,
    ) -> Tensor<'c> {
        // x² = x * x
        let x_sq = self.emit_mul(input, input);
        // variance = mean(x², axis=-1, keepdim=true)
        let variance = self.emit_reduce_mean(&x_sq, &[-1], true);
        // variance + eps
        let eps_tensor = self.emit_arith_constant(eps as f64, input.dtype());
        let var_eps = self.emit_add(&variance, &eps_tensor);
        // rsqrt(variance + eps)
        let inv_rms = self.emit_rsqrt(&var_eps);
        // x * inv_rms  (inv_rms broadcasts via keepdim shape)
        let normalized = self.emit_mul(input, &inv_rms);
        // normalized * weight  (weight broadcasts on the last dim)
        self.emit_mul(&normalized, weight)
    }

    /// GQA head expansion: repeat each KV head `repeats` times along the heads axis.
    ///
    /// Input shape:  `[batch, seq, kv_heads, head_dim]`
    /// Output shape: `[batch, seq, kv_heads * repeats, head_dim]`
    ///
    /// When `repeats == 1` the input is returned unchanged.
    pub fn emit_repeat_kv(&mut self, input: &Tensor<'c>, repeats: usize) -> Tensor<'c> {
        if repeats == 1 {
            return *input;
        }
        // Unsqueeze: [b, s, kv_h, d] → [b, s, kv_h, 1, d]
        let expanded = self.emit_unsqueeze(input, &[3]);
        // Concat `repeats` copies along axis 3 → [b, s, kv_h, repeats, d]
        let copies: Vec<Tensor<'c>> = (0..repeats).map(|_| expanded).collect();
        let tiled = self.emit_concat(&copies, 3);
        // Collapse dims 2 and 3 → [b, s, kv_h * repeats, d]
        let in_shape = input.shape();
        let batch = in_shape[0];
        let seq = in_shape[1];
        let kv_heads = in_shape[2];
        let head_dim = in_shape[3];
        let q_heads = kv_heads.map(|h| h * repeats as u64);
        let out_shape = [batch, seq, q_heads, head_dim];
        // reassociation: [[0], [1], [2, 3], [4]] merges the kv_heads and repeats dims
        self.emit_collapse_shape_with_reassoc(tiled.value(), &out_shape, "[[0], [1], [2, 3], [4]]")
    }

    /// GQA-aware QK^T: `scores[b,h,m,n] = sum_k Q[b,h,m,k] * K[b, h floordiv gqa, n, k]`
    ///
    /// Q is BHSD: `[B, num_heads, seq, head_dim]`
    /// K is BHSD: `[B, kv_heads, total_seq, head_dim]` (directly from KV cache)
    /// Output:    `[B, num_heads, seq, total_seq]`
    ///
    /// Uses a 5D linalg.generic with `floordiv` indexing to avoid materializing
    /// the GQA head expansion. When `gqa_repeat == 1`, the floordiv is a no-op.
    pub fn emit_gqa_qk_transpose(
        &mut self,
        q: &Tensor<'c>, // [B, num_heads, seq, head_dim]
        k: &Tensor<'c>, // [B, kv_heads, total_seq, head_dim]
        gqa_repeat: usize,
    ) -> Tensor<'c> {
        let q_shape = q.shape();
        let k_shape = k.shape();
        let b = q_shape[0]; // B (static)
        let num_heads = q_shape[1]; // H (static)
        let seq = q_shape[2]; // M (dynamic)
        let total_seq = k_shape[2]; // N (dynamic)

        let out_shape = vec![b, num_heads, seq, total_seq];
        let dtype = self.value_dtype(q.value());

        let mut dyn_sources = Vec::new();
        if seq.is_none() {
            dyn_sources.push((q.value(), 2)); // M from Q dim 2
        }
        if total_seq.is_none() {
            dyn_sources.push((k.value(), 2)); // N from K dim 2
        }
        let filled = self.emit_zero_filled_tensor(&out_shape, dtype, &dyn_sources);

        let out_type = self.make_tensor_type(&out_shape, dtype);
        let segment =
            Attribute::parse(self.context, "array<i32: 2, 1>").expect("gqa qkt segment sizes");
        let matmul_region = self.make_matmul_region(dtype);

        // 5 iteration dims: d0=B, d1=H, d2=M, d3=N, d4=K (reduction)
        let q_map = "affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>";
        let k_map =
            format!("affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 floordiv {gqa_repeat}, d3, d4)>");
        let out_map = "affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>";
        let indexing_maps =
            Attribute::parse(self.context, &format!("[{q_map}, {k_map}, {out_map}]"))
                .expect("gqa qkt indexing maps");
        let iterator_types = Attribute::parse(
            self.context,
            "[#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, \
              #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, \
              #linalg.iterator_type<reduction>]",
        )
        .expect("gqa qkt iterator types");

        let result = self
            .block
            .append_operation(
                OperationBuilder::new("linalg.generic", self.location)
                    .add_operands(&[q.value(), k.value(), filled])
                    .add_results(&[out_type])
                    .add_attributes(&[
                        (
                            Identifier::new(self.context, "indexing_maps"),
                            indexing_maps,
                        ),
                        (
                            Identifier::new(self.context, "iterator_types"),
                            iterator_types,
                        ),
                        (
                            Identifier::new(self.context, "operandSegmentSizes"),
                            segment,
                        ),
                    ])
                    .add_regions([matmul_region])
                    .build()
                    .expect("linalg.generic gqa qkt"),
            )
            .result(0)
            .unwrap()
            .into();
        Tensor::from_value(result)
    }

    /// GQA-aware AV: `out[b,h,m,d] = sum_n weights[b,h,m,n] * V[b, h floordiv gqa, n, d]`
    ///
    /// weights: `[B, num_heads, seq, total_seq]` (softmax output)
    /// V is BHSD: `[B, kv_heads, total_seq, head_dim]` (directly from KV cache)
    /// Output:   `[B, num_heads, seq, head_dim]`
    pub fn emit_gqa_av(
        &mut self,
        weights: &Tensor<'c>, // [B, num_heads, seq, total_seq]
        v: &Tensor<'c>,       // [B, kv_heads, total_seq, head_dim]
        gqa_repeat: usize,
    ) -> Tensor<'c> {
        let w_shape = weights.shape();
        let v_shape = v.shape();
        let b = w_shape[0]; // B (static)
        let num_heads = w_shape[1]; // H (static)
        let seq = w_shape[2]; // M (dynamic)
        let head_dim = v_shape[3]; // D (static)

        let out_shape = vec![b, num_heads, seq, head_dim];
        let dtype = self.value_dtype(weights.value());

        let mut dyn_sources = Vec::new();
        if seq.is_none() {
            dyn_sources.push((weights.value(), 2)); // M from weights dim 2
        }
        let filled = self.emit_zero_filled_tensor(&out_shape, dtype, &dyn_sources);

        let out_type = self.make_tensor_type(&out_shape, dtype);
        let segment =
            Attribute::parse(self.context, "array<i32: 2, 1>").expect("gqa av segment sizes");
        let matmul_region = self.make_matmul_region(dtype);

        // 5 iteration dims: d0=B, d1=H, d2=M, d3=D, d4=N (reduction)
        let w_map = "affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>";
        let v_map =
            format!("affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 floordiv {gqa_repeat}, d4, d3)>");
        let out_map = "affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>";
        let indexing_maps =
            Attribute::parse(self.context, &format!("[{w_map}, {v_map}, {out_map}]"))
                .expect("gqa av indexing maps");
        let iterator_types = Attribute::parse(
            self.context,
            "[#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, \
              #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, \
              #linalg.iterator_type<reduction>]",
        )
        .expect("gqa av iterator types");

        let result = self
            .block
            .append_operation(
                OperationBuilder::new("linalg.generic", self.location)
                    .add_operands(&[weights.value(), v.value(), filled])
                    .add_results(&[out_type])
                    .add_attributes(&[
                        (
                            Identifier::new(self.context, "indexing_maps"),
                            indexing_maps,
                        ),
                        (
                            Identifier::new(self.context, "iterator_types"),
                            iterator_types,
                        ),
                        (
                            Identifier::new(self.context, "operandSegmentSizes"),
                            segment,
                        ),
                    ])
                    .add_regions([matmul_region])
                    .build()
                    .expect("linalg.generic gqa av"),
            )
            .result(0)
            .unwrap()
            .into();
        Tensor::from_value(result)
    }

    /// Embedding lookup: gather rows from `weight` by `indices`.
    ///
    /// `weight`:  `[vocab_size, hidden_size]` (F32)
    /// `indices`: `[batch, seq]` (I64)
    /// Output:    `[batch, seq, hidden_size]`
    pub fn emit_embedding(&mut self, weight: &Tensor<'c>, indices: &Tensor<'c>) -> Tensor<'c> {
        self.emit_gather(weight, indices, 0)
    }

    /// Apply Rotary Position Embeddings (RoPE) to Q or K tensor.
    ///
    /// `x`:   `[batch, seq, heads, head_dim]` — the Q or K tensor
    /// `cos`: `[seq, head_dim/2]` — precomputed cosine table (already gathered by position)
    /// `sin`: `[seq, head_dim/2]` — precomputed sine table (already gathered by position)
    ///
    /// The rotation pairs even/odd elements of head_dim:
    ///   out[..., 2i]   = x[..., 2i] * cos[..., i] - x[..., 2i+1] * sin[..., i]
    ///   out[..., 2i+1] = x[..., 2i] * sin[..., i] + x[..., 2i+1] * cos[..., i]
    ///
    /// Decomposed into slices + elementwise ops. The MLIR pass pipeline fuses
    /// the elementwise chain into a single tiled kernel.
    pub fn emit_rope(&mut self, x: &Tensor<'c>, cos: &Tensor<'c>, sin: &Tensor<'c>) -> Tensor<'c> {
        let shape = x.shape();
        let head_dim = shape[3].expect("head_dim must be static for RoPE");

        // Split into even and odd elements along the last axis (step=2).
        // x_even: [b, s, h, d/2] — elements at indices 0, 2, 4, ...
        // x_odd:  [b, s, h, d/2] — elements at indices 1, 3, 5, ...
        let x_even = self.emit_slice(x, &[0], &[head_dim as i64], &[-1], &[2]);
        let x_odd = self.emit_slice(x, &[1], &[head_dim as i64], &[-1], &[2]);

        // cos/sin are [seq, d/2] — unsqueeze to [1, seq, 1, d/2] for broadcast
        // against x_even/x_odd which are [batch, seq, heads, d/2].
        let cos_4d = self.emit_unsqueeze(cos, &[0, 2]);
        let sin_4d = self.emit_unsqueeze(sin, &[0, 2]);

        // out_even = x_even * cos - x_odd * sin
        let even_cos = self.emit_mul(&x_even, &cos_4d);
        let odd_sin = self.emit_mul(&x_odd, &sin_4d);
        let out_even = self.emit_sub(&even_cos, &odd_sin);

        // out_odd = x_even * sin + x_odd * cos
        let even_sin = self.emit_mul(&x_even, &sin_4d);
        let odd_cos = self.emit_mul(&x_odd, &cos_4d);
        let out_odd = self.emit_add(&even_sin, &odd_cos);

        // Interleave: stack [out_even, out_odd] on a new last dim then flatten.
        // [b, s, h, d/2] → [b, s, h, d/2, 1] for each
        let even_5d = self.emit_unsqueeze(&out_even, &[4]);
        let odd_5d = self.emit_unsqueeze(&out_odd, &[4]);
        // concat on axis 4 → [b, s, h, d/2, 2]
        let interleaved = self.emit_concat(&[even_5d, odd_5d], 4);
        // collapse last two dims → [b, s, h, d]
        let out_shape = [shape[0], shape[1], shape[2], Some(head_dim)];
        self.emit_collapse_shape_with_reassoc(
            interleaved.value(),
            &out_shape,
            "[[0], [1], [2], [3, 4]]",
        )
    }
    /// Apply RoPE using the HuggingFace half-rotation convention.
    ///
    /// `x`:   `[batch, seq, heads, head_dim]`
    /// `cos`: `[seq, head_dim]` -- full head_dim (first half == second half)
    /// `sin`: `[seq, head_dim]` -- full head_dim
    ///
    /// Implements: `x * cos + rotate_half(x) * sin`
    /// where `rotate_half(x) = cat(-x[..., d//2:], x[..., :d//2])`.
    pub fn emit_rope_half(
        &mut self,
        x: &Tensor<'c>,
        cos: &Tensor<'c>,
        sin: &Tensor<'c>,
    ) -> Tensor<'c> {
        let shape = x.shape();
        let head_dim = shape[3].expect("head_dim must be static for RoPE");
        let half = head_dim / 2;

        // Split x into first half and second half along last axis
        // x1 = x[..., :d//2], x2 = x[..., d//2:]
        let x1 = self.emit_slice(x, &[0], &[half as i64], &[-1], &[1]);
        let x2 = self.emit_slice(x, &[half as i64], &[head_dim as i64], &[-1], &[1]);

        // rotate_half(x) = cat(-x2, x1)
        let neg_x2 = self.emit_neg(&x2);
        let rotated = self.emit_concat(&[neg_x2, x1], 3);

        // cos/sin are [seq, head_dim] -- unsqueeze to [1, seq, 1, head_dim]
        let cos_4d = self.emit_unsqueeze(cos, &[0, 2]);
        let sin_4d = self.emit_unsqueeze(sin, &[0, 2]);

        // result = x * cos + rotate_half(x) * sin
        let x_cos = self.emit_mul(x, &cos_4d);
        let rot_sin = self.emit_mul(&rotated, &sin_4d);
        self.emit_add(&x_cos, &rot_sin)
    }

    // ── Reductions ────────────────────────────────────────────────────────────

    /// Reduce along a single axis via summation.
    ///
    /// `axis` supports Python-style negative indexing. If `keepdim` is true,
    /// the reduced axis is reinserted as a size-1 dimension.
    pub fn emit_reduce_sum(&mut self, input: &Tensor<'c>, axis: i64, keepdim: bool) -> Tensor<'c> {
        let dtype = input.dtype();
        let combiner_op = match dtype {
            DType::F32 | DType::F64 | DType::BF16 | DType::F16 => "arith.addf",
            DType::I8 | DType::I16 | DType::I32 | DType::I64 => "arith.addi",
            dt => unreachable!("unsupported dtype {:?} for reduce_sum", dt),
        };
        let init_val = self.make_zero_scalar_attr(dtype);
        self.emit_linalg_reduce_single_axis(input, axis, keepdim, combiner_op, init_val)
    }

    /// Reduce along a single axis via maximum.
    ///
    /// Init value is `-inf` for float or `MIN` for integer types.
    pub fn emit_reduce_max(&mut self, input: &Tensor<'c>, axis: i64, keepdim: bool) -> Tensor<'c> {
        let dtype = input.dtype();
        let combiner_op = match dtype {
            DType::F32 | DType::F64 | DType::BF16 | DType::F16 => "arith.maximumf",
            DType::I8 | DType::I16 | DType::I32 | DType::I64 => "arith.maxsi",
            dt => unreachable!("unsupported dtype {:?} for reduce_max", dt),
        };
        let init_val = self.make_min_scalar_attr(dtype);
        self.emit_linalg_reduce_single_axis(input, axis, keepdim, combiner_op, init_val)
    }

    /// Reduce along multiple axes via sum, then divide by the element count.
    ///
    /// `axes` supports Python-style negative indexing. If `keepdim` is true,
    /// all reduced axes are reinserted as size-1 dimensions.
    pub fn emit_reduce_mean(
        &mut self,
        input: &Tensor<'c>,
        axes: &[i64],
        keepdim: bool,
    ) -> Tensor<'c> {
        assert!(!axes.is_empty(), "emit_reduce_mean: axes must not be empty");
        let input_shape = input.shape();
        let rank = input_shape.len() as i64;

        // Normalize axes and sort.
        let mut norm_axes: Vec<usize> = axes
            .iter()
            .map(|&a| {
                let a = if a < 0 { a + rank } else { a };
                assert!(a >= 0 && a < rank, "axis {a} out of bounds for rank {rank}");
                a as usize
            })
            .collect();
        norm_axes.sort_unstable();
        norm_axes.dedup();

        let dtype = input.dtype();

        // Compute the output shape after reducing all norm_axes (keepdim=false for intermediate).
        let reduce_shape: Vec<Option<u64>> = input_shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !norm_axes.contains(i))
            .map(|(_, d)| *d)
            .collect();

        // Init tensor filled with zero for the sum.
        let non_reduced_indices: Vec<usize> = (0..input_shape.len())
            .filter(|i| !norm_axes.contains(i))
            .collect();
        let filled = self.emit_filled_tensor_for_reduce(
            input.value(),
            &reduce_shape,
            &non_reduced_indices,
            dtype,
            0.0_f64,
        );

        // Emit linalg.reduce with all axes at once.
        let dims_str = norm_axes
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let dimensions_attr = Attribute::parse(self.context, &format!("array<i64: {dims_str}>"))
            .expect("reduce dimensions attr");

        let reduced_type = self.make_tensor_type(&reduce_shape, dtype);
        let elem_type = dtype.to_mlir_type(self.context);

        let body_block = Block::new(&[
            (elem_type, self.location), // in_elem
            (elem_type, self.location), // acc_elem
        ]);
        let in_e: melior::ir::Value = body_block.argument(0).unwrap().into();
        let acc_e: melior::ir::Value = body_block.argument(1).unwrap().into();
        let add_op = match dtype {
            DType::F32 | DType::F64 | DType::BF16 | DType::F16 => "arith.addf",
            DType::I8 | DType::I16 | DType::I32 | DType::I64 => "arith.addi",
            dt => unreachable!("unsupported dtype {:?} for reduce_mean", dt),
        };
        let sum_val: melior::ir::Value = body_block
            .append_operation(
                OperationBuilder::new(add_op, self.location)
                    .add_operands(&[acc_e, in_e])
                    .add_results(&[elem_type])
                    .build()
                    .expect(add_op),
            )
            .result(0)
            .unwrap()
            .into();
        body_block.append_operation(
            OperationBuilder::new("linalg.yield", self.location)
                .add_operands(&[sum_val])
                .build()
                .expect("linalg.yield"),
        );
        let body_region = Region::new();
        body_region.append_block(body_block);

        let sum_val: melior::ir::Value = self
            .block
            .append_operation(
                OperationBuilder::new("linalg.reduce", self.location)
                    .add_operands(&[input.value(), filled])
                    .add_results(&[reduced_type])
                    .add_attributes(&[(
                        Identifier::new(self.context, "dimensions"),
                        dimensions_attr,
                    )])
                    .add_regions([body_region])
                    .build()
                    .expect("linalg.reduce mean-sum"),
            )
            .result(0)
            .unwrap()
            .into();

        // Compute the product of all reduced axis sizes to get the count.
        let count_val = self.emit_reduction_count(&input_shape, &norm_axes, input.value());

        // Divide sum by count element-wise.
        let index_type = melior::ir::Type::parse(self.context, "index").expect("index type");
        let sum_tensor = Tensor::from_value(sum_val);
        let divided = self.emit_div_by_index_scalar(&sum_tensor, count_val, index_type);

        // Optionally reinsert reduced axes as size-1 dims.
        if keepdim {
            self.reinsert_reduced_axes(divided, &input_shape, &norm_axes)
        } else {
            divided
        }
    }

    /// Softmax along the given axis.
    ///
    /// Implemented via manual decomposition (linalg.softmax is an "aggregated"
    /// op that doesn't lower cleanly through convert-linalg-to-loops):
    ///   1. max = reduce_max(input, axis, keepdim=true)
    ///   2. shifted = input - max
    ///   3. exp_shifted = exp(shifted)
    ///   4. sum_exp = reduce_sum(exp_shifted, axis, keepdim=true)
    ///   5. result = exp_shifted / sum_exp
    pub fn emit_softmax(&mut self, input: &Tensor<'c>, axis: i64) -> Tensor<'c> {
        let shape = input.shape();
        let rank = shape.len() as i64;
        let axis = if axis < 0 { axis + rank } else { axis };
        assert!(
            axis >= 0 && axis < rank,
            "softmax axis {axis} out of bounds for rank {rank}"
        );

        // Step 1: max along axis with keepdim.
        let max_val = self.emit_reduce_max(input, axis, true);
        // Step 2: subtract max (numerically stable).
        let shifted = self.emit_sub(input, &max_val);
        // Step 3: exp.
        let exp_shifted = self.emit_exp(&shifted);
        // Step 4: sum of exp with keepdim.
        let sum_exp = self.emit_reduce_sum(&exp_shifted, axis, true);
        // Step 5: divide.
        self.emit_div(&exp_shifted, &sum_exp)
    }

    // ── Reduction helpers ─────────────────────────────────────────────────────

    /// Core single-axis reduce via `linalg.reduce`.
    /// Handles negative axis, dynamic dims, and keepdim.
    fn emit_linalg_reduce_single_axis(
        &mut self,
        input: &Tensor<'c>,
        axis: i64,
        keepdim: bool,
        combiner_op: &str,
        init_scalar_attr: Attribute<'c>,
    ) -> Tensor<'c> {
        let input_shape = input.shape();
        let rank = input_shape.len() as i64;
        let axis = if axis < 0 { axis + rank } else { axis };
        assert!(
            axis >= 0 && axis < rank,
            "axis {axis} out of bounds for rank {rank}"
        );
        let axis = axis as usize;

        let dtype = input.dtype();
        let elem_type = dtype.to_mlir_type(self.context);

        // Output shape: input shape with `axis` removed.
        let reduce_shape: Vec<Option<u64>> = input_shape
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != axis)
            .map(|(_, d)| *d)
            .collect();

        // Emit init tensor filled with the identity value.
        let filled = self.emit_scalar_filled_tensor(
            &reduce_shape,
            dtype,
            init_scalar_attr,
            input.value(),
            &input_shape,
            axis,
        );

        let reduced_type = self.make_tensor_type(&reduce_shape, dtype);
        let dimensions_attr = Attribute::parse(self.context, &format!("array<i64: {axis}>"))
            .expect("reduce dimensions attr");

        // Body: (%in_elem, %acc_elem) -> combiner(%acc, %in) -> yield
        let body_block = Block::new(&[
            (elem_type, self.location), // in_elem
            (elem_type, self.location), // acc_elem
        ]);
        let in_e: melior::ir::Value = body_block.argument(0).unwrap().into();
        let acc_e: melior::ir::Value = body_block.argument(1).unwrap().into();
        let combined: melior::ir::Value = body_block
            .append_operation(
                OperationBuilder::new(combiner_op, self.location)
                    .add_operands(&[acc_e, in_e])
                    .add_results(&[elem_type])
                    .build()
                    .expect(combiner_op),
            )
            .result(0)
            .unwrap()
            .into();
        body_block.append_operation(
            OperationBuilder::new("linalg.yield", self.location)
                .add_operands(&[combined])
                .build()
                .expect("linalg.yield"),
        );
        let body_region = Region::new();
        body_region.append_block(body_block);

        let reduced: melior::ir::Value = self
            .block
            .append_operation(
                OperationBuilder::new("linalg.reduce", self.location)
                    .add_operands(&[input.value(), filled])
                    .add_results(&[reduced_type])
                    .add_attributes(&[(
                        Identifier::new(self.context, "dimensions"),
                        dimensions_attr,
                    )])
                    .add_regions([body_region])
                    .build()
                    .expect("linalg.reduce"),
            )
            .result(0)
            .unwrap()
            .into();

        let reduced_tensor = Tensor::from_value(reduced);

        if keepdim {
            // Reinsert the reduced axis as a size-1 dimension.
            self.reinsert_reduced_axes(reduced_tensor, &input_shape, &[axis])
        } else {
            reduced_tensor
        }
    }

    /// Emit a tensor filled with a scalar constant. Handles dynamic dims by
    /// emitting `tensor.dim` for each dynamic dimension in `shape`.
    ///
    /// `dyn_src` and `full_input_shape` are used to get tensor.dim sources:
    /// for each `None` in `shape`, we find the corresponding non-reduced dim
    /// in `full_input_shape` and emit `tensor.dim dyn_src, <original_dim_idx>`.
    fn emit_scalar_filled_tensor(
        &mut self,
        shape: &[Option<u64>],
        dtype: DType,
        scalar_attr: Attribute<'c>,
        dyn_src: melior::ir::Value<'c, 'c>,
        full_input_shape: &[Option<u64>],
        reduced_axis: usize,
    ) -> melior::ir::Value<'c, 'c> {
        let elem_type = dtype.to_mlir_type(self.context);
        let index_type = melior::ir::Type::parse(self.context, "index").expect("index type");

        // Map output dim index -> original input dim index (skipping reduced_axis).
        let orig_indices: Vec<usize> = (0..full_input_shape.len())
            .filter(|&i| i != reduced_axis)
            .collect();

        let mut dyn_vals: Vec<melior::ir::Value<'c, 'c>> = Vec::new();
        for (out_i, dim) in shape.iter().enumerate() {
            if dim.is_none() {
                let orig_i = orig_indices[out_i];
                let idx_attr = Attribute::parse(self.context, &format!("{orig_i} : index"))
                    .expect("dim index attr");
                let idx_val: melior::ir::Value = self
                    .block
                    .append_operation(
                        OperationBuilder::new("arith.constant", self.location)
                            .add_results(&[index_type])
                            .add_attributes(&[(Identifier::new(self.context, "value"), idx_attr)])
                            .build()
                            .expect("arith.constant index"),
                    )
                    .result(0)
                    .unwrap()
                    .into();
                let dim_val: melior::ir::Value = self
                    .block
                    .append_operation(
                        OperationBuilder::new("tensor.dim", self.location)
                            .add_operands(&[dyn_src, idx_val])
                            .add_results(&[index_type])
                            .build()
                            .expect("tensor.dim"),
                    )
                    .result(0)
                    .unwrap()
                    .into();
                dyn_vals.push(dim_val);
            }
        }

        let tensor_type = self.make_tensor_type(shape, dtype);

        let init: melior::ir::Value = self
            .block
            .append_operation(
                OperationBuilder::new("tensor.empty", self.location)
                    .add_operands(&dyn_vals)
                    .add_results(&[tensor_type])
                    .build()
                    .expect("tensor.empty reduce init"),
            )
            .result(0)
            .unwrap()
            .into();

        let scalar: melior::ir::Value = self
            .block
            .append_operation(
                OperationBuilder::new("arith.constant", self.location)
                    .add_results(&[elem_type])
                    .add_attributes(&[(Identifier::new(self.context, "value"), scalar_attr)])
                    .build()
                    .expect("arith.constant fill value"),
            )
            .result(0)
            .unwrap()
            .into();

        // linalg.fill region.
        let fill_block = Block::new(&[(elem_type, self.location), (elem_type, self.location)]);
        let fill_in: melior::ir::Value = fill_block.argument(0).unwrap().into();
        fill_block.append_operation(
            OperationBuilder::new("linalg.yield", self.location)
                .add_operands(&[fill_in])
                .build()
                .expect("linalg.yield fill"),
        );
        let fill_region = Region::new();
        fill_region.append_block(fill_block);

        self.block
            .append_operation(
                OperationBuilder::new("linalg.fill", self.location)
                    .add_operands(&[scalar, init])
                    .add_results(&[tensor_type])
                    .add_attributes(&[(
                        Identifier::new(self.context, "operandSegmentSizes"),
                        Attribute::parse(self.context, "array<i32: 1, 1>").unwrap(),
                    )])
                    .add_regions([fill_region])
                    .build()
                    .expect("linalg.fill"),
            )
            .result(0)
            .unwrap()
            .into()
    }

    /// Emit a filled init tensor for reduce-mean, handling dynamic dims.
    /// `reduce_shape` is the post-reduction shape (all reduced axes removed).
    fn emit_filled_tensor_for_reduce(
        &mut self,
        input_val: melior::ir::Value<'c, 'c>,
        reduce_shape: &[Option<u64>],
        non_reduced_input_indices: &[usize],
        dtype: DType,
        fill_f64: f64,
    ) -> melior::ir::Value<'c, 'c> {
        let elem_type = dtype.to_mlir_type(self.context);

        let mut dyn_vals = Vec::new();
        for (rs_i, rs_dim) in reduce_shape.iter().enumerate() {
            if rs_dim.is_none() {
                let in_idx = non_reduced_input_indices[rs_i];
                let dim_val = self.emit_tensor_dim(input_val, in_idx);
                dyn_vals.push(dim_val);
            }
        }

        let tensor_type = self.make_tensor_type(reduce_shape, dtype);
        let init: melior::ir::Value = self
            .block
            .append_operation(
                OperationBuilder::new("tensor.empty", self.location)
                    .add_operands(&dyn_vals)
                    .add_results(&[tensor_type])
                    .build()
                    .expect("tensor.empty reduce-mean init"),
            )
            .result(0)
            .unwrap()
            .into();

        // Fill with zero (or given value).
        let fill_attr = match dtype {
            DType::F16 => Attribute::parse(self.context, &format!("{fill_f64:.6e} : f16")),
            DType::F32 => Attribute::parse(self.context, &format!("{fill_f64:.6e} : f32")),
            DType::F64 => Attribute::parse(self.context, &format!("{fill_f64:.6e} : f64")),
            DType::BF16 => Attribute::parse(self.context, &format!("{fill_f64:.6e} : bf16")),
            DType::I8 => Attribute::parse(self.context, &format!("{} : i8", fill_f64 as i8)),
            DType::I16 => Attribute::parse(self.context, &format!("{} : i16", fill_f64 as i16)),
            DType::I32 => Attribute::parse(self.context, &format!("{} : i32", fill_f64 as i64)),
            DType::I64 => Attribute::parse(self.context, &format!("{} : i64", fill_f64 as i64)),
            dt => unreachable!("unsupported dtype {:?} for fill", dt),
        }
        .expect("fill attr");

        let scalar: melior::ir::Value = self
            .block
            .append_operation(
                OperationBuilder::new("arith.constant", self.location)
                    .add_results(&[elem_type])
                    .add_attributes(&[(Identifier::new(self.context, "value"), fill_attr)])
                    .build()
                    .expect("arith.constant fill"),
            )
            .result(0)
            .unwrap()
            .into();

        let fill_block = Block::new(&[(elem_type, self.location), (elem_type, self.location)]);
        let fill_in: melior::ir::Value = fill_block.argument(0).unwrap().into();
        fill_block.append_operation(
            OperationBuilder::new("linalg.yield", self.location)
                .add_operands(&[fill_in])
                .build()
                .expect("linalg.yield fill"),
        );
        let fill_region = Region::new();
        fill_region.append_block(fill_block);

        self.block
            .append_operation(
                OperationBuilder::new("linalg.fill", self.location)
                    .add_operands(&[scalar, init])
                    .add_results(&[tensor_type])
                    .add_attributes(&[(
                        Identifier::new(self.context, "operandSegmentSizes"),
                        Attribute::parse(self.context, "array<i32: 1, 1>").unwrap(),
                    )])
                    .add_regions([fill_region])
                    .build()
                    .expect("linalg.fill"),
            )
            .result(0)
            .unwrap()
            .into()
    }

    /// Compute the product of all reduced axis sizes as an `index` SSA value.
    /// For static dims, this is an `arith.constant`. For dynamic dims, we emit
    /// `tensor.dim` + `arith.muli` to compute the product at runtime.
    fn emit_reduction_count(
        &mut self,
        input_shape: &[Option<u64>],
        axes: &[usize],
        input_val: melior::ir::Value<'c, 'c>,
    ) -> melior::ir::Value<'c, 'c> {
        let index_type = melior::ir::Type::parse(self.context, "index").expect("index type");
        let mut count: Option<melior::ir::Value<'c, 'c>> = None;

        for &ax in axes {
            let dim_size_val: melior::ir::Value<'c, 'c> = match input_shape[ax] {
                Some(n) => {
                    let attr = Attribute::parse(self.context, &format!("{n} : index"))
                        .expect("count constant");
                    self.block
                        .append_operation(
                            OperationBuilder::new("arith.constant", self.location)
                                .add_results(&[index_type])
                                .add_attributes(&[(Identifier::new(self.context, "value"), attr)])
                                .build()
                                .expect("arith.constant count"),
                        )
                        .result(0)
                        .unwrap()
                        .into()
                }
                None => {
                    let ax_attr = Attribute::parse(self.context, &format!("{ax} : index"))
                        .expect("axis index attr");
                    let ax_val: melior::ir::Value = self
                        .block
                        .append_operation(
                            OperationBuilder::new("arith.constant", self.location)
                                .add_results(&[index_type])
                                .add_attributes(&[(
                                    Identifier::new(self.context, "value"),
                                    ax_attr,
                                )])
                                .build()
                                .expect("arith.constant ax index"),
                        )
                        .result(0)
                        .unwrap()
                        .into();
                    self.block
                        .append_operation(
                            OperationBuilder::new("tensor.dim", self.location)
                                .add_operands(&[input_val, ax_val])
                                .add_results(&[index_type])
                                .build()
                                .expect("tensor.dim count"),
                        )
                        .result(0)
                        .unwrap()
                        .into()
                }
            };

            count = Some(match count {
                None => dim_size_val,
                Some(prev) => self
                    .block
                    .append_operation(
                        OperationBuilder::new("arith.muli", self.location)
                            .add_operands(&[prev, dim_size_val])
                            .add_results(&[index_type])
                            .build()
                            .expect("arith.muli count"),
                    )
                    .result(0)
                    .unwrap()
                    .into(),
            });
        }

        count.expect("axes must be non-empty")
    }

    /// Divide every element of `input` (float) by a scalar `count` of type `index`.
    /// Converts `count` from index to f32/f64, then emits a `linalg.generic` divf.
    fn emit_div_by_index_scalar(
        &mut self,
        input: &Tensor<'c>,
        count_idx: melior::ir::Value<'c, 'c>,
        _index_type: melior::ir::Type<'c>,
    ) -> Tensor<'c> {
        let shape = input.shape();
        let dtype = input.dtype();
        let elem_type = dtype.to_mlir_type(self.context);

        // Convert index -> i64 via arith.index_cast, then i64 -> f32/f64 via arith.sitofp.
        // arith.uitofp/sitofp does not accept 'index' type directly.
        let i64_type = melior::ir::Type::parse(self.context, "i64").expect("i64 type");
        let count_i64: melior::ir::Value = self
            .block
            .append_operation(
                OperationBuilder::new("arith.index_cast", self.location)
                    .add_operands(&[count_idx])
                    .add_results(&[i64_type])
                    .build()
                    .expect("arith.index_cast"),
            )
            .result(0)
            .unwrap()
            .into();

        let float_convert_op = match dtype {
            DType::F32 | DType::F64 | DType::BF16 | DType::F16 => "arith.sitofp",
            _ => panic!("emit_div_by_index_scalar: float types only"),
        };
        let count_f: melior::ir::Value = self
            .block
            .append_operation(
                OperationBuilder::new(float_convert_op, self.location)
                    .add_operands(&[count_i64])
                    .add_results(&[elem_type])
                    .build()
                    .expect(float_convert_op),
            )
            .result(0)
            .unwrap()
            .into();

        self.emit_div_tensor_by_scalar_tensor(input, count_f, &shape, dtype)
    }

    /// Divide a tensor by a scalar value `count_f` (already converted to elem_type).
    /// Uses `tensor.from_elements` + `linalg.generic` with a scalar second input.
    fn emit_div_tensor_by_scalar_tensor(
        &mut self,
        input: &Tensor<'c>,
        count_f: melior::ir::Value<'c, 'c>,
        shape: &[Option<u64>],
        dtype: DType,
    ) -> Tensor<'c> {
        let elem_type = dtype.to_mlir_type(self.context);
        let rank = shape.len();

        // Wrap the scalar in a 0-D tensor via tensor.from_elements.
        let scalar_tensor_type: melior::ir::Type = {
            let rtt: melior::ir::Type = RankedTensorType::new(&[], elem_type, None).into();
            rtt
        };
        let scalar_tensor: melior::ir::Value = self
            .block
            .append_operation(
                OperationBuilder::new("tensor.from_elements", self.location)
                    .add_operands(&[count_f])
                    .add_results(&[scalar_tensor_type])
                    .build()
                    .expect("tensor.from_elements scalar"),
            )
            .result(0)
            .unwrap()
            .into();

        let out_type = self.make_tensor_type(shape, dtype);
        let init = self.emit_tensor_empty_dyn(shape, dtype, Some(input.value()));

        // Indexing maps: input uses identity, scalar uses (), output uses identity.
        let dims: Vec<String> = (0..rank).map(|i| format!("d{i}")).collect();
        let dim_list = dims.join(", ");
        let identity_map = format!("affine_map<({dim_list}) -> ({dim_list})>");
        let scalar_map = format!("affine_map<({dim_list}) -> ()>");
        let indexing_maps = Attribute::parse(
            self.context,
            &format!("[{identity_map}, {scalar_map}, {identity_map}]"),
        )
        .expect("div mean indexing_maps");
        let iterator_types = self.make_iterator_types(rank);

        let body_block = Block::new(&[
            (elem_type, self.location), // in element
            (elem_type, self.location), // scalar count element
            (elem_type, self.location), // out element (unused)
        ]);
        let x: melior::ir::Value = body_block.argument(0).unwrap().into();
        let cnt: melior::ir::Value = body_block.argument(1).unwrap().into();
        let divided: melior::ir::Value = body_block
            .append_operation(
                OperationBuilder::new("arith.divf", self.location)
                    .add_operands(&[x, cnt])
                    .add_results(&[elem_type])
                    .build()
                    .expect("arith.divf mean"),
            )
            .result(0)
            .unwrap()
            .into();
        body_block.append_operation(
            OperationBuilder::new("linalg.yield", self.location)
                .add_operands(&[divided])
                .build()
                .expect("linalg.yield"),
        );
        let body_region = Region::new();
        body_region.append_block(body_block);

        let result: melior::ir::Value = self
            .block
            .append_operation(
                OperationBuilder::new("linalg.generic", self.location)
                    .add_operands(&[input.value(), scalar_tensor, init])
                    .add_results(&[out_type])
                    .add_attributes(&[
                        (
                            Identifier::new(self.context, "indexing_maps"),
                            indexing_maps,
                        ),
                        (
                            Identifier::new(self.context, "iterator_types"),
                            iterator_types,
                        ),
                        (
                            Identifier::new(self.context, "operandSegmentSizes"),
                            Attribute::parse(self.context, "array<i32: 2, 1>").unwrap(),
                        ),
                    ])
                    .add_regions([body_region])
                    .build()
                    .expect("linalg.generic div mean"),
            )
            .result(0)
            .unwrap()
            .into();
        Tensor::from_value(result)
    }

    /// Reinsert the reduced axes as size-1 dimensions.
    /// `reduced` is the tensor after reduction (axes removed).
    /// `input_shape` is the shape before reduction.
    /// `reduced_axes` is the sorted list of axes that were removed.
    fn reinsert_reduced_axes(
        &mut self,
        reduced: Tensor<'c>,
        input_shape: &[Option<u64>],
        reduced_axes: &[usize],
    ) -> Tensor<'c> {
        let dtype = reduced.dtype();

        // Build the keepdim output shape: same as input_shape but reduced_axes become Some(1).
        let keepdim_shape: Vec<Option<u64>> = input_shape
            .iter()
            .enumerate()
            .map(|(i, d)| {
                if reduced_axes.contains(&i) {
                    Some(1)
                } else {
                    *d
                }
            })
            .collect();

        // Reassociation for expand_shape:
        // Each target dim either came from one source dim (direct mapping) or is new (size-1).
        // The reduced result has rank = input_rank - reduced_axes.len().
        // We map output dims back to source dims by iterating through output dims and
        // assigning them to source dims in order, grouping consecutive reduced dims
        // with their neighboring non-reduced dims.
        //
        // Example: input [3,4,5], reduce axis=1, keepdim=true:
        //   reduced shape: [3, 5], keepdim shape: [3, 1, 5]
        //   reassociation: [[0], [1], [2]] with source dim 0->0, 1->source not exist (new dim),
        //   Actually for expand_shape: source rank = 2, target rank = 3.
        //   Each source dim maps to one or more consecutive target dims.
        //   Source dim 0 -> target dims [0]     (3 -> 3)
        //   Source dim 1 -> target dims [1, 2]  (5 -> 1, 5) — the reduced axis is inserted
        //
        // Wait: the target shape is [3, 1, 5]. For expand_shape reassociation:
        //   [[0], [1, 2]] — source dim 0 expands to target dim [0], source dim 1 expands to [1,2].
        //
        // General rule:
        // - iterate over target dims
        // - for each target dim, if it was a reduced axis (size 1), group it with the next
        //   non-reduced source dim
        // This gets complex. The simplest approach: for each consecutive run of (size-1 dims,
        // then one non-reduced dim), group them together.
        //
        // Build groups: for each source dim index (0..reduced_rank),
        // collect which target dims it corresponds to.
        let source_rank = input_shape.len() - reduced_axes.len();
        let target_rank = input_shape.len();

        // For each target dim, determine if it's a "reduced" (size-1) or "kept" dim.
        // Map each source dim to the consecutive target dims it should expand into.
        let mut src_to_tgt: Vec<Vec<usize>> = vec![Vec::new(); source_rank];

        // We scan target dims in order. Each "kept" target dim advances source_idx.
        // Reduced target dims are grouped with the *next* kept target dim.
        // Special case: if all trailing dims are reduced (unusual), group with last source.
        let mut src_idx = 0usize;
        let mut pending_reduced: Vec<usize> = Vec::new();
        for tgt_i in 0..target_rank {
            if reduced_axes.contains(&tgt_i) {
                // size-1 dim: accumulate, will attach to next source dim
                pending_reduced.push(tgt_i);
            } else {
                // kept dim
                // flush pending reduced dims to this source dim
                for r in pending_reduced.drain(..) {
                    src_to_tgt[src_idx].push(r);
                }
                src_to_tgt[src_idx].push(tgt_i);
                src_idx += 1;
            }
        }
        // Any trailing pending reduced dims go to the last source dim.
        if src_idx == 0 {
            // Edge case: all dims were reduced (shouldn't happen with keepdim semantics, but handle).
            for r in pending_reduced.drain(..) {
                if src_to_tgt.is_empty() {
                    src_to_tgt.push(vec![r]);
                } else {
                    src_to_tgt[0].push(r);
                }
            }
        } else {
            for r in pending_reduced.drain(..) {
                src_to_tgt[src_idx - 1].push(r);
            }
        }

        // Build reassociation string: [[g0_0, g0_1, ...], [g1_0, ...], ...]
        let groups: Vec<String> = src_to_tgt
            .iter()
            .map(|g| {
                let inner = g
                    .iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("[{inner}]")
            })
            .collect();
        let reassoc_str = format!("[{}]", groups.join(", "));

        self.emit_expand_shape_impl(reduced.value(), &keepdim_shape, dtype, &reassoc_str)
    }

    /// Build a zero scalar `Attribute` for the given dtype.
    fn make_zero_scalar_attr(&self, dtype: DType) -> Attribute<'c> {
        match dtype {
            DType::F16 => Attribute::parse(self.context, "0.0 : f16"),
            DType::F32 => Attribute::parse(self.context, "0.0 : f32"),
            DType::F64 => Attribute::parse(self.context, "0.0 : f64"),
            DType::BF16 => Attribute::parse(self.context, "0.0 : bf16"),
            DType::I8 => Attribute::parse(self.context, "0 : i8"),
            DType::I16 => Attribute::parse(self.context, "0 : i16"),
            DType::I32 => Attribute::parse(self.context, "0 : i32"),
            DType::I64 => Attribute::parse(self.context, "0 : i64"),
            dt => unreachable!("unsupported dtype {:?} for make_zero_scalar_attr", dt),
        }
        .expect("zero scalar attr")
    }

    /// Build a minimum-value scalar `Attribute` for the given dtype.
    /// For float: -infinity (IEEE 754). For int: MIN_VALUE.
    fn make_min_scalar_attr(&self, dtype: DType) -> Attribute<'c> {
        match dtype {
            // 0xFC00 is -inf for f16.
            DType::F16 => Attribute::parse(self.context, "0xFC00 : f16"),
            // 0xFF800000 is -inf for f32; 0xFFF0000000000000 is -inf for f64.
            DType::F32 => Attribute::parse(self.context, "0xFF800000 : f32"),
            DType::F64 => Attribute::parse(self.context, "0xFFF0000000000000 : f64"),
            // 0xFF80 is -inf for bf16.
            DType::BF16 => Attribute::parse(self.context, "0xFF80 : bf16"),
            DType::I8 => Attribute::parse(self.context, "-128 : i8"),
            DType::I16 => Attribute::parse(self.context, "-32768 : i16"),
            DType::I32 => Attribute::parse(self.context, "-2147483648 : i32"),
            DType::I64 => Attribute::parse(self.context, "-9223372036854775808 : i64"),
            dt => unreachable!("unsupported dtype {:?} for make_min_scalar_attr", dt),
        }
        .expect("min scalar attr")
    }

    // ── Matmul and Gemm ───────────────────────────────────────────────────────

    /// Matrix multiply: `lhs @ rhs`.
    ///
    /// Supported shapes (all dtypes):
    /// - 2D `[M, K] x [K, N] -> [M, N]` via `linalg.matmul`
    /// - 3D `[B, M, K] x [B, K, N] -> [B, M, N]` via `linalg.batch_matmul`
    /// - 1D lhs `[K]`: expanded to `[1, K]`, matmul, result collapsed to `[N]`
    /// - 1D rhs `[K]`: rhs expanded to `[K, 1]`, matmul, result collapsed to `[M]`
    /// - 4D+ batched: leading dims collapsed to one batch dim, then 3D path, then expanded back
    ///
    /// Dynamic dims (`None`) propagate automatically through output shapes.
    pub fn emit_matmul(&mut self, lhs: &Tensor<'c>, rhs: &Tensor<'c>) -> Tensor<'c> {
        let lhs_shape = lhs.shape();
        let rhs_shape = rhs.shape();
        let lhs_rank = lhs_shape.len();
        let rhs_rank = rhs_shape.len();

        match (lhs_rank, rhs_rank) {
            (2, 2) => self.emit_matmul_2d(lhs.value(), &lhs_shape, rhs.value(), &rhs_shape),
            (3, 3) => self.emit_batch_matmul_3d(lhs.value(), &lhs_shape, rhs.value(), &rhs_shape),
            (1, 2) => {
                // [K] -> [1, K] via expand_shape, matmul [1,K]x[K,N]->[1,N], collapse to [N]
                let exp_shape = vec![Some(1), lhs_shape[0]];
                let expanded = self.emit_expand_shape_1d_to_2d(lhs.value(), &lhs_shape, &exp_shape);
                let result_2d =
                    self.emit_matmul_2d(expanded.value(), &exp_shape, rhs.value(), &rhs_shape);
                // result is [1, N] -> collapse to [N]
                let n = rhs_shape[1];
                self.emit_collapse_shape_2d_to_1d(result_2d.value(), &[Some(1), n], &[n])
            }
            (2, 1) => {
                // rhs: [K] -> [K, 1], matmul [M,K]x[K,1]->[M,1], collapse to [M]
                let k = rhs_shape[0];
                let exp_shape = vec![k, Some(1)];
                let expanded = self.emit_expand_shape_1d_to_2d(rhs.value(), &rhs_shape, &exp_shape);
                let result_2d =
                    self.emit_matmul_2d(lhs.value(), &lhs_shape, expanded.value(), &exp_shape);
                let m = lhs_shape[0];
                self.emit_collapse_shape_2d_to_1d(result_2d.value(), &[m, Some(1)], &[m])
            }
            _ if lhs_rank >= 4 && rhs_rank >= 4 => {
                // Collapse all leading batch dims into one, do batch_matmul, expand back.
                // lhs: [..., M, K] -> [B_flat, M, K]
                // rhs: [..., K, N] -> [B_flat, K, N]
                let m = lhs_shape[lhs_rank - 2];
                let lhs_k = lhs_shape[lhs_rank - 1];
                let rhs_k = rhs_shape[rhs_rank - 2];
                let n = rhs_shape[rhs_rank - 1];

                // Compute flat batch size per tensor (use each tensor's own batch dims).
                let lhs_batch_dims = &lhs_shape[..lhs_rank - 2];
                let flat_b: Option<u64> = lhs_batch_dims
                    .iter()
                    .try_fold(1u64, |acc, d| d.map(|v| acc * v));
                let rhs_batch_dims = &rhs_shape[..rhs_rank - 2];
                let rhs_flat_b: Option<u64> = rhs_batch_dims
                    .iter()
                    .try_fold(1u64, |acc, d| d.map(|v| acc * v));

                let lhs_3d_shape = vec![flat_b, m, lhs_k];
                let rhs_3d_shape = vec![rhs_flat_b, rhs_k, n];
                let lhs_collapsed =
                    self.emit_collapse_shape_nd_to_3d(lhs.value(), &lhs_shape, &lhs_3d_shape);
                let rhs_collapsed =
                    self.emit_collapse_shape_nd_to_3d(rhs.value(), &rhs_shape, &rhs_3d_shape);

                let result_3d = self.emit_batch_matmul_3d(
                    lhs_collapsed.value(),
                    &lhs_3d_shape,
                    rhs_collapsed.value(),
                    &rhs_3d_shape,
                );

                // Expand batch dim back: [B_flat, M, N] -> [..., M, N]
                let mut out_shape: Vec<Option<u64>> = lhs_batch_dims.to_vec();
                out_shape.push(m);
                out_shape.push(n);
                let result_3d_shape = vec![flat_b, m, n];
                self.emit_expand_shape_3d_to_nd(
                    result_3d.value(),
                    &result_3d_shape,
                    &out_shape,
                    Some((lhs.value(), &lhs_shape)),
                )
            }
            (3, 2) => {
                // [B,M,K] x [K,N] → unsqueeze rhs to [1,K,N], use projected batch dim
                // in the matmul indexing map (rhs batch → 0) to avoid copying the weight.
                let rhs_3d = self.emit_unsqueeze(rhs, &[0]); // [K,N] → [1,K,N]
                let rhs_3d_shape = rhs_3d.shape();
                self.emit_batch_matmul_3d_broadcast_rhs(
                    lhs.value(),
                    &lhs_shape,
                    rhs_3d.value(),
                    &rhs_3d_shape,
                )
            }
            (2, 3) => {
                // [M,K] x [B,K,N] → unsqueeze lhs to [1,M,K], broadcast to [B,M,K], batch_matmul
                let lhs_3d = self.emit_unsqueeze(lhs, &[0]); // [M,K] → [1,M,K]
                let lhs_3d_shape = lhs_3d.shape();
                let b = rhs_shape[0];
                let bcast_shape = vec![b, lhs_3d_shape[1], lhs_3d_shape[2]];
                let bcasted = self.broadcast_to(&lhs_3d, &bcast_shape);
                self.emit_batch_matmul_3d(bcasted.value(), &bcast_shape, rhs.value(), &rhs_shape)
            }
            _ => panic!(
                "emit_matmul: unsupported rank combination {}D x {}D — \
                 supported: 1D-2D, 2D-2D, 2D-3D, 3D-2D, 3D-3D, 4D+-4D+",
                lhs_rank, rhs_rank
            ),
        }
    }

    /// Emit ONNX-style Gemm: `alpha * (A @ B) + beta * C`.
    ///
    /// Handles optional transposes for A and B. `alpha` and `beta` are f32
    /// scalars. `c` is the bias/residual term.
    #[allow(clippy::too_many_arguments)]
    pub fn emit_gemm(
        &mut self,
        a: &Tensor<'c>,
        b: &Tensor<'c>,
        c: &Tensor<'c>,
        alpha: f32,
        beta: f32,
        transpose_a: bool,
        transpose_b: bool,
    ) -> Tensor<'c> {
        // Optionally transpose A and B.
        let a_used = if transpose_a {
            self.emit_linalg_transpose_2d(a)
        } else {
            *a
        };
        let b_used = if transpose_b {
            self.emit_linalg_transpose_2d(b)
        } else {
            *b
        };

        // Fast path: alpha=1, beta=1, C is a 1D bias vector.
        // Fuse the bias into the matmul by broadcasting it as the initial
        // accumulator. Eliminates a separate add pass over the output.
        let is_unit_scale =
            (alpha - 1.0f32).abs() <= f32::EPSILON && (beta - 1.0f32).abs() <= f32::EPSILON;
        if is_unit_scale && c.rank() == 1 {
            return self.emit_matmul_2d_with_bias(
                a_used.value(),
                &a_used.shape(),
                b_used.value(),
                &b_used.shape(),
                c.value(),
                &c.shape(),
            );
        }

        // General path: matmul then scale and add.
        let ab = self.emit_matmul(&a_used, &b_used);

        let ab_scaled = if (alpha - 1.0f32).abs() > f32::EPSILON {
            self.emit_linalg_scale_f32(ab, alpha)
        } else {
            ab
        };

        let c_scaled = if (beta - 1.0f32).abs() > f32::EPSILON {
            self.emit_linalg_scale_f32(*c, beta)
        } else {
            *c
        };

        self.emit_add(&ab_scaled, &c_scaled)
    }

    /// Emit `Y = A @ B + bias + residual` in one fused matmul.
    ///
    /// Combines the 1D bias broadcast and 2D residual add into a single
    /// accumulator init, then runs matmul. Result: two passes (init + matmul)
    /// instead of three (bias_broadcast + matmul + residual_add).
    ///
    /// Requires alpha=1, beta=1 (caller must verify).
    pub fn emit_gemm_with_residual(
        &mut self,
        a: &Tensor<'c>,
        b: &Tensor<'c>,
        bias: &Tensor<'c>,
        residual: &Tensor<'c>,
        transpose_a: bool,
        transpose_b: bool,
    ) -> Tensor<'c> {
        let a_used = if transpose_a {
            self.emit_linalg_transpose_2d(a)
        } else {
            *a
        };
        let b_used = if transpose_b {
            self.emit_linalg_transpose_2d(b)
        } else {
            *b
        };

        self.emit_matmul_2d_with_bias_and_residual(
            a_used.value(),
            &a_used.shape(),
            b_used.value(),
            &b_used.shape(),
            bias.value(),
            residual.value(),
            &residual.shape(),
        )
    }

    // ── Matmul internals ──────────────────────────────────────────────────────

    /// Emit `linalg.generic` matmul for 2D inputs `[M,K] x [K,N] -> [M,N]`.
    /// Handles static and dynamic M, K, N.
    fn emit_matmul_2d(
        &mut self,
        lhs_val: melior::ir::Value<'c, 'c>,
        lhs_shape: &[Option<u64>], // [M, K]
        rhs_val: melior::ir::Value<'c, 'c>,
        rhs_shape: &[Option<u64>], // [K, N]
    ) -> Tensor<'c> {
        let m = lhs_shape[0];
        let n = rhs_shape[1];
        let out_shape = vec![m, n];
        let lhs_dtype = self.value_dtype(lhs_val);
        let rhs_dtype = self.value_dtype(rhs_val);

        // Mixed-precision: if either input is bf16, accumulate in f32.
        let mixed =
            lhs_dtype != rhs_dtype && (lhs_dtype == DType::BF16 || rhs_dtype == DType::BF16);
        let out_dtype = if mixed { DType::F32 } else { lhs_dtype };

        let mut dyn_sources = Vec::new();
        if m.is_none() {
            dyn_sources.push((lhs_val, 0)); // M from lhs dim 0
        }
        if n.is_none() {
            dyn_sources.push((rhs_val, 1)); // N from rhs dim 1
        }
        let filled = self.emit_zero_filled_tensor(&out_shape, out_dtype, &dyn_sources);

        let out_type = self.make_tensor_type(&out_shape, out_dtype);
        let segment =
            Attribute::parse(self.context, "array<i32: 2, 1>").expect("matmul segment sizes");
        let matmul_region = if mixed {
            self.make_mixed_matmul_region(lhs_dtype, rhs_dtype)
        } else {
            self.make_matmul_region(lhs_dtype)
        };

        // 3 iteration dims: d0=M, d1=N, d2=K
        let lhs_map = "affine_map<(d0, d1, d2) -> (d0, d2)>";
        let rhs_map = "affine_map<(d0, d1, d2) -> (d2, d1)>";
        let out_map = "affine_map<(d0, d1, d2) -> (d0, d1)>";
        let indexing_maps =
            Attribute::parse(self.context, &format!("[{lhs_map}, {rhs_map}, {out_map}]"))
                .expect("matmul 2d indexing maps");
        let iterator_types = Attribute::parse(
            self.context,
            "[#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]",
        ).expect("matmul 2d iterator types");

        let result = self
            .block
            .append_operation(
                OperationBuilder::new("linalg.generic", self.location)
                    .add_operands(&[lhs_val, rhs_val, filled])
                    .add_results(&[out_type])
                    .add_attributes(&[
                        (
                            Identifier::new(self.context, "indexing_maps"),
                            indexing_maps,
                        ),
                        (
                            Identifier::new(self.context, "iterator_types"),
                            iterator_types,
                        ),
                        (
                            Identifier::new(self.context, "operandSegmentSizes"),
                            segment,
                        ),
                    ])
                    .add_regions([matmul_region])
                    .build()
                    .expect("linalg.generic matmul 2d"),
            )
            .result(0)
            .unwrap()
            .into();
        Tensor::from_value(result)
    }

    /// Emit `linalg.generic` batch matmul for 3D inputs `[B,M,K] x [B,K,N] -> [B,M,N]`.
    fn emit_batch_matmul_3d(
        &mut self,
        lhs_val: melior::ir::Value<'c, 'c>,
        lhs_shape: &[Option<u64>], // [B, M, K]
        rhs_val: melior::ir::Value<'c, 'c>,
        rhs_shape: &[Option<u64>], // [B, K, N]
    ) -> Tensor<'c> {
        let b = lhs_shape[0];
        let m = lhs_shape[1];
        let n = rhs_shape[2];
        let out_shape = vec![b, m, n];
        let lhs_dtype = self.value_dtype(lhs_val);
        let rhs_dtype = self.value_dtype(rhs_val);
        let mixed =
            lhs_dtype != rhs_dtype && (lhs_dtype == DType::BF16 || rhs_dtype == DType::BF16);
        let out_dtype = if mixed { DType::F32 } else { lhs_dtype };

        let mut dyn_sources = Vec::new();
        if b.is_none() {
            dyn_sources.push((lhs_val, 0)); // B from lhs dim 0
        }
        if m.is_none() {
            dyn_sources.push((lhs_val, 1)); // M from lhs dim 1
        }
        if n.is_none() {
            dyn_sources.push((rhs_val, 2)); // N from rhs dim 2
        }
        let filled = self.emit_zero_filled_tensor(&out_shape, out_dtype, &dyn_sources);

        let out_type = self.make_tensor_type(&out_shape, out_dtype);
        let segment =
            Attribute::parse(self.context, "array<i32: 2, 1>").expect("batch_matmul segment sizes");
        let matmul_region = if mixed {
            self.make_mixed_matmul_region(lhs_dtype, rhs_dtype)
        } else {
            self.make_matmul_region(lhs_dtype)
        };

        // 4 iteration dims: d0=B, d1=M, d2=N, d3=K
        let lhs_map = "affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>";
        let rhs_map = "affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>";
        let out_map = "affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>";
        let indexing_maps =
            Attribute::parse(self.context, &format!("[{lhs_map}, {rhs_map}, {out_map}]"))
                .expect("batch matmul 3d indexing maps");
        let iterator_types = Attribute::parse(
            self.context,
            "[#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]",
        ).expect("batch matmul 3d iterator types");

        let result = self
            .block
            .append_operation(
                OperationBuilder::new("linalg.generic", self.location)
                    .add_operands(&[lhs_val, rhs_val, filled])
                    .add_results(&[out_type])
                    .add_attributes(&[
                        (
                            Identifier::new(self.context, "indexing_maps"),
                            indexing_maps,
                        ),
                        (
                            Identifier::new(self.context, "iterator_types"),
                            iterator_types,
                        ),
                        (
                            Identifier::new(self.context, "operandSegmentSizes"),
                            segment,
                        ),
                    ])
                    .add_regions([matmul_region])
                    .build()
                    .expect("linalg.generic batch matmul 3d"),
            )
            .result(0)
            .unwrap()
            .into();
        Tensor::from_value(result)
    }

    /// Emit `linalg.generic` matmul for 2D inputs with a 1D bias fused in.
    /// `[M,K] x [K,N] + [N] -> [M,N]`.
    /// The bias is broadcast to [M,N] as the initial accumulator, so the
    /// matmul body `out += a * b` starts from `bias[n]` instead of `0`.
    fn emit_matmul_2d_with_bias(
        &mut self,
        lhs_val: melior::ir::Value<'c, 'c>,
        lhs_shape: &[Option<u64>], // [M, K]
        rhs_val: melior::ir::Value<'c, 'c>,
        rhs_shape: &[Option<u64>], // [K, N]
        bias_val: melior::ir::Value<'c, 'c>,
        _bias_shape: &[Option<u64>], // [N]
    ) -> Tensor<'c> {
        let m = lhs_shape[0];
        let n = rhs_shape[1];
        let out_shape = vec![m, n];
        let dtype = self.value_dtype(lhs_val);

        // Broadcast bias [N] to [M, N] using linalg.generic with projected map.
        let out_type = self.make_tensor_type(&out_shape, dtype);
        let mut dyn_sources = Vec::new();
        if m.is_none() {
            dyn_sources.push((lhs_val, 0usize));
        }
        if n.is_none() {
            dyn_sources.push((rhs_val, 1));
        }
        // Use zero-filled tensor as the output for the broadcast generic.
        // The broadcast body overwrites every element with bias[n].
        let empty = self.emit_zero_filled_tensor(&out_shape, dtype, &dyn_sources);

        let bias_map = "affine_map<(d0, d1) -> (d1)>"; // broadcast N
        let out_map_bcast = "affine_map<(d0, d1) -> (d0, d1)>";
        let bcast_indexing =
            Attribute::parse(self.context, &format!("[{bias_map}, {out_map_bcast}]"))
                .expect("bias broadcast indexing maps");
        let bcast_iters = Attribute::parse(
            self.context,
            "[#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>]",
        )
        .expect("bias broadcast iterator types");
        let bcast_segment =
            Attribute::parse(self.context, "array<i32: 1, 1>").expect("bias broadcast segment");

        let bcast_region = {
            let block = Block::new(&[
                (dtype.to_mlir_type(self.context), self.location),
                (dtype.to_mlir_type(self.context), self.location),
            ]);
            let in_val: melior::ir::Value = block.argument(0).unwrap().into();
            let yield_op = OperationBuilder::new("linalg.yield", self.location)
                .add_operands(&[in_val])
                .build()
                .expect("linalg.yield (bias broadcast)");
            block.append_operation(yield_op);
            let region = Region::new();
            region.append_block(block);
            region
        };

        let filled_result = self
            .block
            .append_operation(
                OperationBuilder::new("linalg.generic", self.location)
                    .add_operands(&[bias_val, empty])
                    .add_results(&[out_type])
                    .add_attributes(&[
                        (
                            Identifier::new(self.context, "indexing_maps"),
                            bcast_indexing,
                        ),
                        (Identifier::new(self.context, "iterator_types"), bcast_iters),
                        (
                            Identifier::new(self.context, "operandSegmentSizes"),
                            bcast_segment,
                        ),
                    ])
                    .add_regions([bcast_region])
                    .build()
                    .expect("linalg.generic bias broadcast"),
            )
            .result(0)
            .unwrap()
            .into();

        // Now emit the matmul using the bias-filled tensor as accumulator.
        // The accumulator is always f32 (from bias broadcast above).
        // If either matmul input is bf16, use mixed-precision region.
        let lhs_dtype = self.value_dtype(lhs_val);
        let rhs_dtype = self.value_dtype(rhs_val);
        let mixed =
            lhs_dtype != rhs_dtype && (lhs_dtype == DType::BF16 || rhs_dtype == DType::BF16);

        let out_type2 = self.make_tensor_type(&out_shape, dtype);
        let segment =
            Attribute::parse(self.context, "array<i32: 2, 1>").expect("matmul segment sizes");
        let matmul_region = if mixed {
            self.make_mixed_matmul_region(lhs_dtype, rhs_dtype)
        } else {
            self.make_matmul_region(dtype)
        };

        let lhs_map = "affine_map<(d0, d1, d2) -> (d0, d2)>";
        let rhs_map = "affine_map<(d0, d1, d2) -> (d2, d1)>";
        let out_map = "affine_map<(d0, d1, d2) -> (d0, d1)>";
        let indexing_maps =
            Attribute::parse(self.context, &format!("[{lhs_map}, {rhs_map}, {out_map}]"))
                .expect("matmul 2d with bias indexing maps");
        let iterator_types = Attribute::parse(
            self.context,
            "[#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]",
        ).expect("matmul 2d with bias iterator types");

        let result = self
            .block
            .append_operation(
                OperationBuilder::new("linalg.generic", self.location)
                    .add_operands(&[lhs_val, rhs_val, filled_result])
                    .add_results(&[out_type2])
                    .add_attributes(&[
                        (
                            Identifier::new(self.context, "indexing_maps"),
                            indexing_maps,
                        ),
                        (
                            Identifier::new(self.context, "iterator_types"),
                            iterator_types,
                        ),
                        (
                            Identifier::new(self.context, "operandSegmentSizes"),
                            segment,
                        ),
                    ])
                    .add_regions([matmul_region])
                    .build()
                    .expect("linalg.generic matmul 2d with bias"),
            )
            .result(0)
            .unwrap()
            .into();
        Tensor::from_value(result)
    }

    /// Emit `Y = A @ B + bias + residual` as a single matmul with fused
    /// accumulator.  The init pass computes `bias[n] + residual[m,n]` per
    /// element, then the matmul accumulates `A[m,k]*B[k,n]` on top.
    #[allow(clippy::too_many_arguments)]
    fn emit_matmul_2d_with_bias_and_residual(
        &mut self,
        lhs_val: melior::ir::Value<'c, 'c>,
        lhs_shape: &[Option<u64>], // [M, K]
        rhs_val: melior::ir::Value<'c, 'c>,
        rhs_shape: &[Option<u64>],               // [K, N]
        bias_val: melior::ir::Value<'c, 'c>,     // [N]
        residual_val: melior::ir::Value<'c, 'c>, // [M, N]
        _residual_shape: &[Option<u64>],         // [M, N]
    ) -> Tensor<'c> {
        let m = lhs_shape[0];
        let n = rhs_shape[1];
        let out_shape = vec![m, n];
        let dtype = self.value_dtype(lhs_val);

        // Build init tensor: bias[n] + residual[m,n] for each (m,n).
        let out_type = self.make_tensor_type(&out_shape, dtype);
        let mut dyn_sources = Vec::new();
        if m.is_none() {
            // M is dynamic — get it from the residual's dim 0.
            dyn_sources.push((residual_val, 0usize));
        }
        if n.is_none() {
            dyn_sources.push((rhs_val, 1));
        }
        let empty = self.emit_zero_filled_tensor(&out_shape, dtype, &dyn_sources);

        // linalg.generic: ins(bias[N], residual[M,N]) outs(empty[M,N])
        //   body: yield bias_elem + residual_elem
        let bias_map = "affine_map<(d0, d1) -> (d1)>"; // broadcast N
        let res_map = "affine_map<(d0, d1) -> (d0, d1)>";
        let out_map = "affine_map<(d0, d1) -> (d0, d1)>";
        let init_indexing =
            Attribute::parse(self.context, &format!("[{bias_map}, {res_map}, {out_map}]"))
                .expect("bias+residual init indexing maps");
        let init_iters = Attribute::parse(
            self.context,
            "[#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>]",
        )
        .expect("bias+residual init iterator types");
        let init_segment =
            Attribute::parse(self.context, "array<i32: 2, 1>").expect("bias+residual segment");

        let init_region = {
            let f32_type = dtype.to_mlir_type(self.context);
            let block = Block::new(&[
                (f32_type, self.location),
                (f32_type, self.location),
                (f32_type, self.location),
            ]);
            let bias_elem: melior::ir::Value = block.argument(0).unwrap().into();
            let res_elem: melior::ir::Value = block.argument(1).unwrap().into();
            let sum: melior::ir::Value = block
                .append_operation(
                    OperationBuilder::new("arith.addf", self.location)
                        .add_operands(&[bias_elem, res_elem])
                        .add_results(&[f32_type])
                        .build()
                        .expect("arith.addf (bias+residual)"),
                )
                .result(0)
                .unwrap()
                .into();
            let yield_op = OperationBuilder::new("linalg.yield", self.location)
                .add_operands(&[sum])
                .build()
                .expect("linalg.yield (bias+residual init)");
            block.append_operation(yield_op);
            let region = Region::new();
            region.append_block(block);
            region
        };

        let filled_result = self
            .block
            .append_operation(
                OperationBuilder::new("linalg.generic", self.location)
                    .add_operands(&[bias_val, residual_val, empty])
                    .add_results(&[out_type])
                    .add_attributes(&[
                        (
                            Identifier::new(self.context, "indexing_maps"),
                            init_indexing,
                        ),
                        (Identifier::new(self.context, "iterator_types"), init_iters),
                        (
                            Identifier::new(self.context, "operandSegmentSizes"),
                            init_segment,
                        ),
                    ])
                    .add_regions([init_region])
                    .build()
                    .expect("linalg.generic bias+residual init"),
            )
            .result(0)
            .unwrap()
            .into();

        // Now emit the matmul with the fused init as accumulator.
        let lhs_dtype = self.value_dtype(lhs_val);
        let rhs_dtype = self.value_dtype(rhs_val);
        let mixed =
            lhs_dtype != rhs_dtype && (lhs_dtype == DType::BF16 || rhs_dtype == DType::BF16);

        let out_type2 = self.make_tensor_type(&out_shape, dtype);
        let segment =
            Attribute::parse(self.context, "array<i32: 2, 1>").expect("matmul segment sizes");
        let matmul_region = if mixed {
            self.make_mixed_matmul_region(lhs_dtype, rhs_dtype)
        } else {
            self.make_matmul_region(dtype)
        };

        let lhs_map = "affine_map<(d0, d1, d2) -> (d0, d2)>";
        let rhs_matmul_map = "affine_map<(d0, d1, d2) -> (d2, d1)>";
        let out_matmul_map = "affine_map<(d0, d1, d2) -> (d0, d1)>";
        let indexing_maps = Attribute::parse(
            self.context,
            &format!("[{lhs_map}, {rhs_matmul_map}, {out_matmul_map}]"),
        )
        .expect("matmul 2d with bias+residual indexing maps");
        let iterator_types = Attribute::parse(
            self.context,
            "[#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]",
        ).expect("matmul 2d with bias+residual iterator types");

        let result = self
            .block
            .append_operation(
                OperationBuilder::new("linalg.generic", self.location)
                    .add_operands(&[lhs_val, rhs_val, filled_result])
                    .add_results(&[out_type2])
                    .add_attributes(&[
                        (
                            Identifier::new(self.context, "indexing_maps"),
                            indexing_maps,
                        ),
                        (
                            Identifier::new(self.context, "iterator_types"),
                            iterator_types,
                        ),
                        (
                            Identifier::new(self.context, "operandSegmentSizes"),
                            segment,
                        ),
                    ])
                    .add_regions([matmul_region])
                    .build()
                    .expect("linalg.generic matmul 2d with bias+residual"),
            )
            .result(0)
            .unwrap()
            .into();
        Tensor::from_value(result)
    }

    /// Batch matmul [B,M,K] × [1,K,N] → [B,M,N] with broadcast on rhs batch.
    ///
    /// Like `emit_batch_matmul_3d` but the rhs has batch=1 and we use a
    /// projected indexing map `(d0,d1,d2,d3) -> (0, d3, d2)` instead of
    /// `(d0,d1,d2,d3) -> (d0, d3, d2)`. This avoids a full copy of the
    /// rhs weight matrix when B is dynamic (e.g. LM head 768×50257).
    fn emit_batch_matmul_3d_broadcast_rhs(
        &mut self,
        lhs_val: melior::ir::Value<'c, 'c>,
        lhs_shape: &[Option<u64>], // [B, M, K]
        rhs_val: melior::ir::Value<'c, 'c>,
        _rhs_shape: &[Option<u64>], // [1, K, N] — batch dim is 1
    ) -> Tensor<'c> {
        let b = lhs_shape[0];
        let m = lhs_shape[1];
        let n = _rhs_shape[2];
        let out_shape = vec![b, m, n];
        let lhs_dtype = self.value_dtype(lhs_val);
        let rhs_dtype = self.value_dtype(rhs_val);
        let mixed =
            lhs_dtype != rhs_dtype && (lhs_dtype == DType::BF16 || rhs_dtype == DType::BF16);
        let out_dtype = if mixed { DType::F32 } else { lhs_dtype };

        // Build dyn_sources for each None dim in out_shape [B, M, N].
        let mut dyn_sources = Vec::new();
        if b.is_none() {
            dyn_sources.push((lhs_val, 0usize)); // B from lhs dim 0
        }
        if m.is_none() {
            dyn_sources.push((lhs_val, 1)); // M from lhs dim 1
        }
        if n.is_none() {
            dyn_sources.push((rhs_val, 2)); // N from rhs dim 2
        }
        let filled = self.emit_zero_filled_tensor(&out_shape, out_dtype, &dyn_sources);

        let out_type = self.make_tensor_type(&out_shape, out_dtype);
        let segment =
            Attribute::parse(self.context, "array<i32: 2, 1>").expect("batch_matmul segment sizes");
        let matmul_region = if mixed {
            self.make_mixed_matmul_region(lhs_dtype, rhs_dtype)
        } else {
            self.make_matmul_region(lhs_dtype)
        };

        // 4 iteration dims: d0=B, d1=M, d2=N, d3=K
        // rhs uses 0 for batch dim instead of d0 (broadcast from batch=1).
        let lhs_map = "affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>";
        let rhs_map = "affine_map<(d0, d1, d2, d3) -> (0, d3, d2)>";
        let out_map = "affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>";
        let indexing_maps =
            Attribute::parse(self.context, &format!("[{lhs_map}, {rhs_map}, {out_map}]"))
                .expect("batch matmul 3d broadcast rhs indexing maps");
        let iterator_types = Attribute::parse(
            self.context,
            "[#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]",
        ).expect("batch matmul 3d broadcast rhs iterator types");

        let result = self
            .block
            .append_operation(
                OperationBuilder::new("linalg.generic", self.location)
                    .add_operands(&[lhs_val, rhs_val, filled])
                    .add_results(&[out_type])
                    .add_attributes(&[
                        (
                            Identifier::new(self.context, "indexing_maps"),
                            indexing_maps,
                        ),
                        (
                            Identifier::new(self.context, "iterator_types"),
                            iterator_types,
                        ),
                        (
                            Identifier::new(self.context, "operandSegmentSizes"),
                            segment,
                        ),
                    ])
                    .add_regions([matmul_region])
                    .build()
                    .expect("linalg.generic batch matmul 3d broadcast rhs"),
            )
            .result(0)
            .unwrap()
            .into();
        Tensor::from_value(result)
    }

    /// Emit a zero-filled `tensor.empty` of the given shape.
    ///
    /// `dyn_sources`: `(tensor_value, dim_index)` for each `None` dim in `shape`,
    /// in order. Used to emit `tensor.dim` for dynamic sizes. Static dims don't
    /// need an entry.
    fn emit_zero_filled_tensor(
        &mut self,
        shape: &[Option<u64>],
        dtype: DType,
        dyn_sources: &[(melior::ir::Value<'c, 'c>, usize)],
    ) -> melior::ir::Value<'c, 'c> {
        let elem_type = dtype.to_mlir_type(self.context);
        let tensor_type = self.make_tensor_type(shape, dtype);

        // Collect tensor.dim values for dynamic dims in shape order.
        let index_type = melior::ir::Type::parse(self.context, "index").expect("index type");
        let mut dyn_iter = dyn_sources.iter();
        let mut dyn_vals: Vec<melior::ir::Value<'c, 'c>> = Vec::new();
        for dim in shape.iter() {
            if dim.is_none() {
                let &(src, idx) = dyn_iter
                    .next()
                    .expect("dyn_sources must have an entry for each None dim");
                // Emit arith.constant for the dim index.
                let idx_attr = Attribute::parse(self.context, &format!("{idx} : index"))
                    .expect("dim index attr");
                let idx_val: melior::ir::Value = self
                    .block
                    .append_operation(
                        OperationBuilder::new("arith.constant", self.location)
                            .add_results(&[index_type])
                            .add_attributes(&[(Identifier::new(self.context, "value"), idx_attr)])
                            .build()
                            .expect("arith.constant index"),
                    )
                    .result(0)
                    .unwrap()
                    .into();
                let dim_val: melior::ir::Value = self
                    .block
                    .append_operation(
                        OperationBuilder::new("tensor.dim", self.location)
                            .add_operands(&[src, idx_val])
                            .add_results(&[index_type])
                            .build()
                            .expect("tensor.dim"),
                    )
                    .result(0)
                    .unwrap()
                    .into();
                dyn_vals.push(dim_val);
            }
        }

        // tensor.empty(%dyn0, %dyn1, ...) : tensor<...>
        let init: melior::ir::Value = self
            .block
            .append_operation(
                OperationBuilder::new("tensor.empty", self.location)
                    .add_operands(&dyn_vals)
                    .add_results(&[tensor_type])
                    .build()
                    .expect("tensor.empty"),
            )
            .result(0)
            .unwrap()
            .into();

        // Zero constant.
        let zero_attr = match dtype {
            DType::F16 => Attribute::parse(self.context, "0.0 : f16"),
            DType::F32 => Attribute::parse(self.context, "0.0 : f32"),
            DType::F64 => Attribute::parse(self.context, "0.0 : f64"),
            DType::BF16 => Attribute::parse(self.context, "0.0 : bf16"),
            DType::I8 => Attribute::parse(self.context, "0 : i8"),
            DType::I16 => Attribute::parse(self.context, "0 : i16"),
            DType::I32 => Attribute::parse(self.context, "0 : i32"),
            DType::I64 => Attribute::parse(self.context, "0 : i64"),
            dt => unreachable!("unsupported dtype {:?} for matmul zero fill", dt),
        }
        .expect("zero constant attr");
        let zero: melior::ir::Value = self
            .block
            .append_operation(
                OperationBuilder::new("arith.constant", self.location)
                    .add_results(&[elem_type])
                    .add_attributes(&[(Identifier::new(self.context, "value"), zero_attr)])
                    .build()
                    .expect("arith.constant zero"),
            )
            .result(0)
            .unwrap()
            .into();

        // linalg.fill region: takes (scalar_in, scalar_out), yields scalar_in.
        let fill_block = Block::new(&[(elem_type, self.location), (elem_type, self.location)]);
        let fill_in: melior::ir::Value = fill_block.argument(0).unwrap().into();
        fill_block.append_operation(
            OperationBuilder::new("linalg.yield", self.location)
                .add_operands(&[fill_in])
                .build()
                .expect("linalg.yield fill"),
        );
        let fill_region = Region::new();
        fill_region.append_block(fill_block);

        let segment_fill =
            Attribute::parse(self.context, "array<i32: 1, 1>").expect("fill segment sizes");

        self.block
            .append_operation(
                OperationBuilder::new("linalg.fill", self.location)
                    .add_operands(&[zero, init])
                    .add_results(&[tensor_type])
                    .add_attributes(&[(
                        Identifier::new(self.context, "operandSegmentSizes"),
                        segment_fill,
                    )])
                    .add_regions([fill_region])
                    .build()
                    .expect("linalg.fill"),
            )
            .result(0)
            .unwrap()
            .into()
    }

    /// Build the linalg.matmul/batch_matmul body region.
    /// Block args: (lhs_elem, rhs_elem, acc_elem).
    /// Body: acc + lhs * rhs.
    fn make_matmul_region(&self, dtype: DType) -> Region<'c> {
        let elem_type = dtype.to_mlir_type(self.context);
        let block = Block::new(&[
            (elem_type, self.location),
            (elem_type, self.location),
            (elem_type, self.location),
        ]);
        let lhs_e: melior::ir::Value = block.argument(0).unwrap().into();
        let rhs_e: melior::ir::Value = block.argument(1).unwrap().into();
        let acc_e: melior::ir::Value = block.argument(2).unwrap().into();

        let mul: melior::ir::Value = match dtype {
            DType::F32 | DType::F64 | DType::BF16 | DType::F16 => {
                let (fmid, fmval) = self.fastmath_contract_attr();
                block
                    .append_operation(
                        OperationBuilder::new("arith.mulf", self.location)
                            .add_operands(&[lhs_e, rhs_e])
                            .add_results(&[elem_type])
                            .add_attributes(&[(fmid, fmval)])
                            .build()
                            .expect("arith.mulf"),
                    )
                    .result(0)
                    .unwrap()
                    .into()
            }
            DType::I8 | DType::I16 | DType::I32 | DType::I64 => block
                .append_operation(
                    OperationBuilder::new("arith.muli", self.location)
                        .add_operands(&[lhs_e, rhs_e])
                        .add_results(&[elem_type])
                        .build()
                        .expect("arith.muli"),
                )
                .result(0)
                .unwrap()
                .into(),
            dt => unreachable!("unsupported dtype {:?} for matmul region", dt),
        };
        let add: melior::ir::Value = match dtype {
            DType::F32 | DType::F64 | DType::BF16 | DType::F16 => {
                let (fmid, fmval) = self.fastmath_contract_attr();
                block
                    .append_operation(
                        OperationBuilder::new("arith.addf", self.location)
                            .add_operands(&[acc_e, mul])
                            .add_results(&[elem_type])
                            .add_attributes(&[(fmid, fmval)])
                            .build()
                            .expect("arith.addf"),
                    )
                    .result(0)
                    .unwrap()
                    .into()
            }
            DType::I8 | DType::I16 | DType::I32 | DType::I64 => block
                .append_operation(
                    OperationBuilder::new("arith.addi", self.location)
                        .add_operands(&[acc_e, mul])
                        .add_results(&[elem_type])
                        .build()
                        .expect("arith.addi"),
                )
                .result(0)
                .unwrap()
                .into(),
            dt => unreachable!("unsupported dtype {:?} for matmul region", dt),
        };
        block.append_operation(
            OperationBuilder::new("linalg.yield", self.location)
                .add_operands(&[add])
                .build()
                .expect("linalg.yield matmul"),
        );
        let region = Region::new();
        region.append_block(block);
        region
    }
    /// Build a mixed-precision matmul body for `linalg.generic`.
    ///
    /// Block args: `(lhs_elem: lhs_dtype, rhs_elem: rhs_dtype, acc: f32)`.
    /// Body casts any bf16 input to f32 via `arith.extf`, then `mulf + addf` in f32.
    /// Used for Strategy A: bf16 weights with f32 activations.
    fn make_mixed_matmul_region(&self, lhs_dtype: DType, rhs_dtype: DType) -> Region<'c> {
        let lhs_type = lhs_dtype.to_mlir_type(self.context);
        let rhs_type = rhs_dtype.to_mlir_type(self.context);
        let f32_type = DType::F32.to_mlir_type(self.context);

        let block = Block::new(&[
            (lhs_type, self.location),
            (rhs_type, self.location),
            (f32_type, self.location),
        ]);
        let lhs_e: melior::ir::Value = block.argument(0).unwrap().into();
        let rhs_e: melior::ir::Value = block.argument(1).unwrap().into();
        let acc_e: melior::ir::Value = block.argument(2).unwrap().into();

        // Cast bf16 inputs to f32 (f32 inputs pass through as identity).
        let lhs_f32: melior::ir::Value = if lhs_dtype == DType::BF16 {
            block
                .append_operation(
                    OperationBuilder::new("arith.extf", self.location)
                        .add_operands(&[lhs_e])
                        .add_results(&[f32_type])
                        .build()
                        .expect("arith.extf lhs bf16->f32"),
                )
                .result(0)
                .unwrap()
                .into()
        } else {
            lhs_e
        };
        let rhs_f32: melior::ir::Value = if rhs_dtype == DType::BF16 {
            block
                .append_operation(
                    OperationBuilder::new("arith.extf", self.location)
                        .add_operands(&[rhs_e])
                        .add_results(&[f32_type])
                        .build()
                        .expect("arith.extf rhs bf16->f32"),
                )
                .result(0)
                .unwrap()
                .into()
        } else {
            rhs_e
        };

        let (fmid, fmval) = self.fastmath_contract_attr();
        let mul: melior::ir::Value = block
            .append_operation(
                OperationBuilder::new("arith.mulf", self.location)
                    .add_operands(&[lhs_f32, rhs_f32])
                    .add_results(&[f32_type])
                    .add_attributes(&[(fmid, fmval)])
                    .build()
                    .expect("arith.mulf"),
            )
            .result(0)
            .unwrap()
            .into();

        let (fmid2, fmval2) = self.fastmath_contract_attr();
        let add: melior::ir::Value = block
            .append_operation(
                OperationBuilder::new("arith.addf", self.location)
                    .add_operands(&[acc_e, mul])
                    .add_results(&[f32_type])
                    .add_attributes(&[(fmid2, fmval2)])
                    .build()
                    .expect("arith.addf"),
            )
            .result(0)
            .unwrap()
            .into();

        block.append_operation(
            OperationBuilder::new("linalg.yield", self.location)
                .add_operands(&[add])
                .build()
                .expect("linalg.yield mixed matmul"),
        );
        let region = Region::new();
        region.append_block(block);
        region
    }

    /// Emit `linalg.conv_2d_nchw_fchw` with optional padding and bias.
    ///
    /// `pads`: `[pad_top, pad_left, pad_bottom, pad_right]` in ONNX order.
    /// `strides`: `[stride_h, stride_w]`.
    /// `dilations`: `[dilation_h, dilation_w]`.
    /// `bias`: optional 1D tensor `[CO]` to broadcast-add after conv.
    pub fn emit_conv2d(
        &mut self,
        input: &Tensor<'c>,
        weight: &Tensor<'c>,
        bias: Option<&Tensor<'c>>,
        pads: [u64; 4],
        strides: [u64; 2],
        dilations: [u64; 2],
    ) -> Result<Tensor<'c>, CompileError> {
        let in_shape = input.shape(); // [N, CI, H, W]
        let wt_shape = weight.shape(); // [CO, CI, KH, KW]
        assert_eq!(in_shape.len(), 4, "conv2d: input must be rank-4 (NCHW)");
        assert_eq!(wt_shape.len(), 4, "conv2d: weight must be rank-4 (FCHW)");

        let dtype = input.dtype();
        let [pad_top, pad_left, pad_bottom, pad_right] = pads;
        let [stride_h, stride_w] = strides;
        let [dilation_h, dilation_w] = dilations;

        // Step 1: pad input if needed.
        let input_val = if pad_top > 0 || pad_left > 0 || pad_bottom > 0 || pad_right > 0 {
            let pad_low = [0i64, 0, pad_top as i64, pad_left as i64];
            let pad_high = [0i64, 0, pad_bottom as i64, pad_right as i64];
            let zero_attr = match dtype {
                DType::F16 => Attribute::parse(self.context, "0.0 : f16"),
                DType::F32 => Attribute::parse(self.context, "0.0 : f32"),
                DType::F64 => Attribute::parse(self.context, "0.0 : f64"),
                DType::BF16 => Attribute::parse(self.context, "0.0 : bf16"),
                DType::I8 => Attribute::parse(self.context, "0 : i8"),
                DType::I16 => Attribute::parse(self.context, "0 : i16"),
                DType::I32 => Attribute::parse(self.context, "0 : i32"),
                DType::I64 => Attribute::parse(self.context, "0 : i64"),
                dt => unreachable!("unsupported dtype {:?} for conv2d pad", dt),
            }
            .expect("conv2d pad zero");
            self.emit_tensor_pad(input.value(), &pad_low, &pad_high, zero_attr)
        } else {
            input.value()
        };

        // Get padded input shape.
        let padded_rtt = RankedTensorType::try_from(input_val.r#type())
            .expect("padded input must be RankedTensorType");
        let padded_shape: Vec<Option<u64>> = (0..4)
            .map(|i| {
                let raw = unsafe {
                    mlir_sys::mlirShapedTypeGetDimSize(
                        melior::ir::TypeLike::to_raw(&padded_rtt),
                        i as isize,
                    )
                };
                if raw < 0 { None } else { Some(raw as u64) }
            })
            .collect();

        let n = padded_shape[0];
        let h_padded = padded_shape[2];
        let w_padded = padded_shape[3];
        let co = wt_shape[0];
        let kh = wt_shape[2];
        let kw = wt_shape[3];

        // Step 2: compute output spatial dims.
        // OH = (H_padded - dilation_h * (KH - 1) - 1) / stride_h + 1
        let oh = h_padded
            .zip(kh)
            .map(|(h, k)| (h - dilation_h * (k - 1) - 1) / stride_h + 1);
        let ow = w_padded
            .zip(kw)
            .map(|(w, k)| (w - dilation_w * (k - 1) - 1) / stride_w + 1);
        let out_shape = vec![n, co, oh, ow];

        // Step 3: zero-filled output tensor [N, CO, OH, OW].
        let out_type = self.make_tensor_type(&out_shape, dtype);
        // For dynamic dims, we read from padded input or weight.
        let elem_type = dtype.to_mlir_type(self.context);
        let zero_attr = match dtype {
            DType::F16 => Attribute::parse(self.context, "0.0 : f16"),
            DType::F32 => Attribute::parse(self.context, "0.0 : f32"),
            DType::F64 => Attribute::parse(self.context, "0.0 : f64"),
            DType::BF16 => Attribute::parse(self.context, "0.0 : bf16"),
            DType::I8 => Attribute::parse(self.context, "0 : i8"),
            DType::I16 => Attribute::parse(self.context, "0 : i16"),
            DType::I32 => Attribute::parse(self.context, "0 : i32"),
            DType::I64 => Attribute::parse(self.context, "0 : i64"),
            dt => unreachable!("unsupported dtype {:?} for conv2d output zero", dt),
        }
        .expect("conv2d output zero");
        let zero_scalar: melior::ir::Value = self
            .block
            .append_operation(
                OperationBuilder::new("arith.constant", self.location)
                    .add_results(&[elem_type])
                    .add_attributes(&[(Identifier::new(self.context, "value"), zero_attr)])
                    .build()
                    .expect("arith.constant conv2d zero"),
            )
            .result(0)
            .unwrap()
            .into();

        // Build dyn_vals for tensor.empty.
        let mut dyn_vals: Vec<melior::ir::Value<'c, 'c>> = Vec::new();
        // dim 0 (N) — from padded input dim 0.
        if n.is_none() {
            dyn_vals.push(self.emit_tensor_dim(input_val, 0));
        }
        // dim 1 (CO) — from weight dim 0.
        if co.is_none() {
            dyn_vals.push(self.emit_tensor_dim(weight.value(), 0));
        }
        // dim 2 (OH) and dim 3 (OW) — must be static for now.
        if oh.is_none() || ow.is_none() {
            return Err(CompileError::Shape(
                "conv2d: dynamic output spatial dims not yet supported".into(),
            ));
        }

        let init_empty: melior::ir::Value = self
            .block
            .append_operation(
                OperationBuilder::new("tensor.empty", self.location)
                    .add_operands(&dyn_vals)
                    .add_results(&[out_type])
                    .build()
                    .expect("tensor.empty conv2d"),
            )
            .result(0)
            .unwrap()
            .into();

        let fill_block = Block::new(&[(elem_type, self.location), (elem_type, self.location)]);
        let fill_in: melior::ir::Value = fill_block.argument(0).unwrap().into();
        fill_block.append_operation(
            OperationBuilder::new("linalg.yield", self.location)
                .add_operands(&[fill_in])
                .build()
                .expect("linalg.yield conv2d fill"),
        );
        let fill_region = Region::new();
        fill_region.append_block(fill_block);

        let init_filled: melior::ir::Value = self
            .block
            .append_operation(
                OperationBuilder::new("linalg.fill", self.location)
                    .add_operands(&[zero_scalar, init_empty])
                    .add_results(&[out_type])
                    .add_attributes(&[(
                        Identifier::new(self.context, "operandSegmentSizes"),
                        Attribute::parse(self.context, "array<i32: 1, 1>").unwrap(),
                    )])
                    .add_regions([fill_region])
                    .build()
                    .expect("linalg.fill conv2d"),
            )
            .result(0)
            .unwrap()
            .into();

        // Step 4: emit linalg.conv_2d_nchw_fchw.
        // The named op auto-generates its region (via buildStructuredOp).
        // We must NOT provide a region — the builder generates it.
        let strides_attr = Attribute::parse(
            self.context,
            &format!("dense<[{stride_h}, {stride_w}]> : tensor<2xi64>"),
        )
        .expect("conv2d strides attr");
        let dilations_attr = Attribute::parse(
            self.context,
            &format!("dense<[{dilation_h}, {dilation_w}]> : tensor<2xi64>"),
        )
        .expect("conv2d dilations attr");

        // linalg.conv_2d_nchw_fchw requires an explicit region with body:
        //   ^bb0(%in: f32, %filter: f32, %acc: f32):
        //     %prod = arith.mulf %in, %filter : f32
        //     %sum  = arith.addf %acc, %prod  : f32
        //     linalg.yield %sum : f32
        let conv_region = self.make_conv2d_region(dtype);

        let conv_result: melior::ir::Value = self
            .block
            .append_operation(
                OperationBuilder::new("linalg.conv_2d_nchw_fchw", self.location)
                    .add_operands(&[input_val, weight.value(), init_filled])
                    .add_results(&[out_type])
                    .add_attributes(&[
                        (Identifier::new(self.context, "strides"), strides_attr),
                        (Identifier::new(self.context, "dilations"), dilations_attr),
                        (
                            Identifier::new(self.context, "operandSegmentSizes"),
                            Attribute::parse(self.context, "array<i32: 2, 1>").unwrap(),
                        ),
                    ])
                    .add_regions([conv_region])
                    .build()
                    .expect("linalg.conv_2d_nchw_fchw"),
            )
            .result(0)
            .unwrap()
            .into();

        // Step 5: add bias if provided.
        let conv_tensor = Tensor::from_value(conv_result);
        if let Some(b) = bias {
            // bias shape: [CO] → unsqueeze to [1, CO, 1, 1] → emit_add.
            let bias_4d = self.emit_unsqueeze(b, &[0, 2, 3]);
            Ok(self.emit_add(&conv_tensor, &bias_4d))
        } else {
            Ok(conv_tensor)
        }
    }

    /// Compute padding for `auto_pad=SAME_UPPER` and call `emit_conv2d`.
    ///
    /// SAME_UPPER padding: output spatial size = ceil(H / stride_h).
    /// pad_total = max(0, (OH - 1) * stride + dilation * (K - 1) + 1 - H)
    /// pad_begin = pad_total / 2, pad_end = pad_total - pad_begin.
    pub fn emit_conv2d_same_upper(
        &mut self,
        input: &Tensor<'c>,
        weight: &Tensor<'c>,
        bias: Option<&Tensor<'c>>,
        strides: [u64; 2],
        dilations: [u64; 2],
    ) -> Result<Tensor<'c>, CompileError> {
        let in_shape = input.shape();
        let wt_shape = weight.shape();
        assert_eq!(
            in_shape.len(),
            4,
            "conv2d_same_upper: input must be NCHW rank-4"
        );
        assert_eq!(
            wt_shape.len(),
            4,
            "conv2d_same_upper: weight must be FCHW rank-4"
        );

        let h = in_shape[2].expect("conv2d_same_upper: H must be static");
        let w = in_shape[3].expect("conv2d_same_upper: W must be static");
        let kh = wt_shape[2].expect("conv2d_same_upper: KH must be static");
        let kw = wt_shape[3].expect("conv2d_same_upper: KW must be static");
        let [stride_h, stride_w] = strides;
        let [dilation_h, dilation_w] = dilations;

        let oh = h.div_ceil(stride_h);
        let ow = w.div_ceil(stride_w);

        let pad_total_h = ((oh - 1) * stride_h + dilation_h * (kh - 1) + 1).saturating_sub(h);
        let pad_total_w = ((ow - 1) * stride_w + dilation_w * (kw - 1) + 1).saturating_sub(w);

        let pad_top = pad_total_h / 2;
        let pad_bottom = pad_total_h - pad_top;
        let pad_left = pad_total_w / 2;
        let pad_right = pad_total_w - pad_left;

        self.emit_conv2d(
            input,
            weight,
            bias,
            [pad_top, pad_left, pad_bottom, pad_right],
            strides,
            dilations,
        )
    }

    /// Build the region body for `linalg.conv_2d_nchw_fchw`:
    ///   ^bb0(%in, %filter, %acc):  acc + in * filter
    fn make_conv2d_region(&self, dtype: DType) -> Region<'c> {
        let elem_type = dtype.to_mlir_type(self.context);
        let block = Block::new(&[
            (elem_type, self.location), // in_elem
            (elem_type, self.location), // filter_elem
            (elem_type, self.location), // acc_elem
        ]);
        let in_e: melior::ir::Value = block.argument(0).unwrap().into();
        let filter_e: melior::ir::Value = block.argument(1).unwrap().into();
        let acc_e: melior::ir::Value = block.argument(2).unwrap().into();

        let prod: melior::ir::Value = match dtype {
            DType::F32 | DType::F64 | DType::BF16 | DType::F16 => {
                let (fmid, fmval) = self.fastmath_contract_attr();
                block
                    .append_operation(
                        OperationBuilder::new("arith.mulf", self.location)
                            .add_operands(&[in_e, filter_e])
                            .add_results(&[elem_type])
                            .add_attributes(&[(fmid, fmval)])
                            .build()
                            .expect("arith.mulf conv2d"),
                    )
                    .result(0)
                    .unwrap()
                    .into()
            }
            DType::I8 | DType::I16 | DType::I32 | DType::I64 => block
                .append_operation(
                    OperationBuilder::new("arith.muli", self.location)
                        .add_operands(&[in_e, filter_e])
                        .add_results(&[elem_type])
                        .build()
                        .expect("arith.muli conv2d"),
                )
                .result(0)
                .unwrap()
                .into(),
            dt => unreachable!("unsupported dtype {:?} for conv2d body", dt),
        };
        let sum: melior::ir::Value = match dtype {
            DType::F32 | DType::F64 | DType::BF16 | DType::F16 => {
                let (fmid, fmval) = self.fastmath_contract_attr();
                block
                    .append_operation(
                        OperationBuilder::new("arith.addf", self.location)
                            .add_operands(&[acc_e, prod])
                            .add_results(&[elem_type])
                            .add_attributes(&[(fmid, fmval)])
                            .build()
                            .expect("arith.addf conv2d"),
                    )
                    .result(0)
                    .unwrap()
                    .into()
            }
            DType::I8 | DType::I16 | DType::I32 | DType::I64 => block
                .append_operation(
                    OperationBuilder::new("arith.addi", self.location)
                        .add_operands(&[acc_e, prod])
                        .add_results(&[elem_type])
                        .build()
                        .expect("arith.addi conv2d"),
                )
                .result(0)
                .unwrap()
                .into(),
            dt => unreachable!("unsupported dtype {:?} for conv2d body", dt),
        };
        block.append_operation(
            OperationBuilder::new("linalg.yield", self.location)
                .add_operands(&[sum])
                .build()
                .expect("linalg.yield conv2d"),
        );
        let region = Region::new();
        region.append_block(block);
        region
    }

    /// Emit `linalg.pooling_nchw_max` for max pooling (NCHW layout).
    ///
    /// `kernel_shape`: `[KH, KW]`.
    /// `pads`: `[pad_top, pad_left, pad_bottom, pad_right]`.
    /// `strides`: `[stride_h, stride_w]`.
    /// `dilations`: `[dilation_h, dilation_w]`.
    pub fn emit_max_pool2d(
        &mut self,
        input: &Tensor<'c>,
        kernel_shape: [u64; 2],
        pads: [u64; 4],
        strides: [u64; 2],
        dilations: [u64; 2],
    ) -> Result<Tensor<'c>, CompileError> {
        let in_shape = input.shape(); // [N, C, H, W]
        assert_eq!(in_shape.len(), 4, "max_pool2d: input must be rank-4 (NCHW)");

        let dtype = input.dtype();
        let [kh, kw] = kernel_shape;
        let [pad_top, pad_left, pad_bottom, pad_right] = pads;
        let [stride_h, stride_w] = strides;
        let [dilation_h, dilation_w] = dilations;

        // Step 1: pad input with -inf if needed.
        let input_val = if pad_top > 0 || pad_left > 0 || pad_bottom > 0 || pad_right > 0 {
            let pad_low = [0i64, 0, pad_top as i64, pad_left as i64];
            let pad_high = [0i64, 0, pad_bottom as i64, pad_right as i64];
            let neg_inf_attr = match dtype {
                DType::F32 => Attribute::parse(self.context, "0xFF800000 : f32"),
                DType::F64 => Attribute::parse(self.context, "0xFFF0000000000000 : f64"),
                _ => panic!("max_pool2d: integer pooling not supported"),
            }
            .expect("max_pool2d -inf pad value");
            self.emit_tensor_pad(input.value(), &pad_low, &pad_high, neg_inf_attr)
        } else {
            input.value()
        };

        // Get padded input shape.
        let padded_rtt =
            RankedTensorType::try_from(input_val.r#type()).expect("padded input RankedTensorType");
        let padded_shape: Vec<Option<u64>> = (0..4)
            .map(|i| {
                let raw = unsafe {
                    mlir_sys::mlirShapedTypeGetDimSize(
                        melior::ir::TypeLike::to_raw(&padded_rtt),
                        i as isize,
                    )
                };
                if raw < 0 { None } else { Some(raw as u64) }
            })
            .collect();

        let n = padded_shape[0];
        let c = padded_shape[1];
        let h_padded = padded_shape[2];
        let w_padded = padded_shape[3];

        // Step 2: compute output spatial dims (accounts for dilation).
        let oh = h_padded.map(|h| (h - dilation_h * (kh - 1) - 1) / stride_h + 1);
        let ow = w_padded.map(|w| (w - dilation_w * (kw - 1) - 1) / stride_w + 1);
        let out_shape = vec![n, c, oh, ow];

        // Step 3: output filled with -inf.
        let elem_type = dtype.to_mlir_type(self.context);
        let neg_inf_attr = match dtype {
            DType::F32 => Attribute::parse(self.context, "0xFF800000 : f32"),
            DType::F64 => Attribute::parse(self.context, "0xFFF0000000000000 : f64"),
            _ => panic!("max_pool2d: integer pooling not supported"),
        }
        .expect("max_pool2d output -inf");

        let neg_inf_scalar: melior::ir::Value = self
            .block
            .append_operation(
                OperationBuilder::new("arith.constant", self.location)
                    .add_results(&[elem_type])
                    .add_attributes(&[(Identifier::new(self.context, "value"), neg_inf_attr)])
                    .build()
                    .expect("arith.constant -inf"),
            )
            .result(0)
            .unwrap()
            .into();

        let out_type = self.make_tensor_type(&out_shape, dtype);
        let mut dyn_vals: Vec<melior::ir::Value<'c, 'c>> = Vec::new();
        if n.is_none() {
            dyn_vals.push(self.emit_tensor_dim(input_val, 0));
        }
        if c.is_none() {
            dyn_vals.push(self.emit_tensor_dim(input_val, 1));
        }
        if oh.is_none() || ow.is_none() {
            return Err(CompileError::Shape(
                "max_pool2d: dynamic output spatial dims not yet supported".into(),
            ));
        }

        let init_empty: melior::ir::Value = self
            .block
            .append_operation(
                OperationBuilder::new("tensor.empty", self.location)
                    .add_operands(&dyn_vals)
                    .add_results(&[out_type])
                    .build()
                    .expect("tensor.empty max_pool2d"),
            )
            .result(0)
            .unwrap()
            .into();

        let fill_block = Block::new(&[(elem_type, self.location), (elem_type, self.location)]);
        let fill_in: melior::ir::Value = fill_block.argument(0).unwrap().into();
        fill_block.append_operation(
            OperationBuilder::new("linalg.yield", self.location)
                .add_operands(&[fill_in])
                .build()
                .expect("linalg.yield max_pool2d fill"),
        );
        let fill_region = Region::new();
        fill_region.append_block(fill_block);

        let init_filled: melior::ir::Value = self
            .block
            .append_operation(
                OperationBuilder::new("linalg.fill", self.location)
                    .add_operands(&[neg_inf_scalar, init_empty])
                    .add_results(&[out_type])
                    .add_attributes(&[(
                        Identifier::new(self.context, "operandSegmentSizes"),
                        Attribute::parse(self.context, "array<i32: 1, 1>").unwrap(),
                    )])
                    .add_regions([fill_region])
                    .build()
                    .expect("linalg.fill max_pool2d"),
            )
            .result(0)
            .unwrap()
            .into();

        // Step 4: create empty window tensor [KH, KW].
        let window_shape = vec![Some(kh), Some(kw)];
        let window_type = self.make_tensor_type(&window_shape, dtype);
        let window_empty: melior::ir::Value = self
            .block
            .append_operation(
                OperationBuilder::new("tensor.empty", self.location)
                    .add_operands(&[])
                    .add_results(&[window_type])
                    .build()
                    .expect("tensor.empty window"),
            )
            .result(0)
            .unwrap()
            .into();

        // Step 5: emit linalg.pooling_nchw_max.
        let strides_attr = Attribute::parse(
            self.context,
            &format!("dense<[{stride_h}, {stride_w}]> : tensor<2xi64>"),
        )
        .expect("max_pool2d strides attr");
        let dilations_attr = Attribute::parse(
            self.context,
            &format!("dense<[{dilation_h}, {dilation_w}]> : tensor<2xi64>"),
        )
        .expect("max_pool2d dilations attr");

        // Region body: max(in, acc).
        let pool_region = self.make_pool_max_region(dtype);

        let pool_result: melior::ir::Value = self
            .block
            .append_operation(
                OperationBuilder::new("linalg.pooling_nchw_max", self.location)
                    .add_operands(&[input_val, window_empty, init_filled])
                    .add_results(&[out_type])
                    .add_attributes(&[
                        (Identifier::new(self.context, "strides"), strides_attr),
                        (Identifier::new(self.context, "dilations"), dilations_attr),
                        (
                            Identifier::new(self.context, "operandSegmentSizes"),
                            Attribute::parse(self.context, "array<i32: 2, 1>").unwrap(),
                        ),
                    ])
                    .add_regions([pool_region])
                    .build()
                    .expect("linalg.pooling_nchw_max"),
            )
            .result(0)
            .unwrap()
            .into();

        Ok(Tensor::from_value(pool_result))
    }

    /// Build the region for `linalg.pooling_nchw_max`:
    ///   ^bb0(%in, %window, %acc): arith.maximumf %in, %acc; linalg.yield
    fn make_pool_max_region(&self, dtype: DType) -> Region<'c> {
        let elem_type = dtype.to_mlir_type(self.context);
        let block = Block::new(&[
            (elem_type, self.location), // in_elem
            (elem_type, self.location), // window_elem (unused)
            (elem_type, self.location), // acc_elem
        ]);
        let in_e: melior::ir::Value = block.argument(0).unwrap().into();
        let acc_e: melior::ir::Value = block.argument(2).unwrap().into();

        let max_op = match dtype {
            DType::F32 | DType::F64 | DType::BF16 | DType::F16 => "arith.maximumf",
            DType::I8 | DType::I16 | DType::I32 | DType::I64 => "arith.maxsi",
            dt => unreachable!("unsupported dtype {:?} for pool max", dt),
        };
        let max_val: melior::ir::Value = block
            .append_operation(
                OperationBuilder::new(max_op, self.location)
                    .add_operands(&[in_e, acc_e])
                    .add_results(&[elem_type])
                    .build()
                    .expect("arith.max pool"),
            )
            .result(0)
            .unwrap()
            .into();
        block.append_operation(
            OperationBuilder::new("linalg.yield", self.location)
                .add_operands(&[max_val])
                .build()
                .expect("linalg.yield pool"),
        );
        let region = Region::new();
        region.append_block(block);
        region
    }

    /// Global average pooling over spatial dims H, W (NCHW).
    ///
    /// Output shape: `[N, C, 1, 1]`.
    /// Implemented as: reduce_sum over axes [2, 3] with keepdim, then divide by H*W.
    pub fn emit_global_avg_pool(&mut self, input: &Tensor<'c>) -> Tensor<'c> {
        let in_shape = input.shape(); // [N, C, H, W]
        assert_eq!(
            in_shape.len(),
            4,
            "global_avg_pool: input must be rank-4 NCHW"
        );

        // Sum over H then W (keepdim=true to preserve rank).
        let sum_h = self.emit_reduce_sum(input, 2, true); // [N, C, 1, W]
        let sum_hw = self.emit_reduce_sum(&sum_h, 3, true); // [N, C, 1, 1]

        // Divide by H*W.
        let h = in_shape[2].expect("global_avg_pool: H must be static");
        let w = in_shape[3].expect("global_avg_pool: W must be static");
        let spatial_size = (h * w) as f64;

        let scale = (1.0 / spatial_size) as f32;
        self.emit_linalg_scale_f32(sum_hw, scale)
    }

    /// Batch normalization: `scale * (x - mean) / sqrt(var + eps) + bias`.
    ///
    /// All parameters (scale, bias, mean, var) are 1D tensors of shape `[C]`.
    /// Input is `[N, C, H, W]`.
    pub fn emit_batch_norm(
        &mut self,
        input: &Tensor<'c>,
        scale: &Tensor<'c>,
        bias: &Tensor<'c>,
        mean: &Tensor<'c>,
        var: &Tensor<'c>,
        epsilon: f32,
    ) -> Tensor<'c> {
        let in_shape = input.shape();
        assert_eq!(in_shape.len(), 4, "batch_norm: input must be rank-4 NCHW");

        // Broadcast 1D [C] params to [1, C, 1, 1].
        let mean_4d = self.emit_unsqueeze(mean, &[0, 2, 3]);
        let var_4d = self.emit_unsqueeze(var, &[0, 2, 3]);
        let scale_4d = self.emit_unsqueeze(scale, &[0, 2, 3]);
        let bias_4d = self.emit_unsqueeze(bias, &[0, 2, 3]);

        // x - mean.
        let x_centered = self.emit_sub(input, &mean_4d);

        // var + eps.
        let eps_tensor = self.emit_scalar_broadcast(epsilon, &var_4d);
        let var_plus_eps = self.emit_add(&var_4d, &eps_tensor);

        // rsqrt(var + eps).
        let rsqrt_var = self.emit_rsqrt(&var_plus_eps);

        // (x - mean) * rsqrt(var + eps) * scale + bias.
        let normalized = self.emit_mul(&x_centered, &rsqrt_var);
        let scaled = self.emit_mul(&normalized, &scale_4d);
        self.emit_add(&scaled, &bias_4d)
    }
}
