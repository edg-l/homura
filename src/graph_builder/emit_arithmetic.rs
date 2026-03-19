use super::*;

impl<'c> GraphBuilder<'c> {
    // ── Elementwise binary ops ────────────────────────────────────────────────

    /// Element-wise addition (F32/F64: arith.addf; I32/I64: arith.addi).
    pub fn emit_add(&mut self, lhs: &Tensor<'c>, rhs: &Tensor<'c>) -> Tensor<'c> {
        let op = match lhs.dtype() {
            DType::F32 | DType::F64 | DType::BF16 => "arith.addf",
            DType::I32 | DType::I64 => "arith.addi",
        };
        self.emit_linalg_binary(op, lhs, rhs)
    }

    /// Element-wise subtraction (F32/F64: arith.subf; I32/I64: arith.subi).
    pub fn emit_sub(&mut self, lhs: &Tensor<'c>, rhs: &Tensor<'c>) -> Tensor<'c> {
        let op = match lhs.dtype() {
            DType::F32 | DType::F64 | DType::BF16 => "arith.subf",
            DType::I32 | DType::I64 => "arith.subi",
        };
        self.emit_linalg_binary(op, lhs, rhs)
    }

    /// Element-wise multiplication (F32/F64: arith.mulf; I32/I64: arith.muli).
    pub fn emit_mul(&mut self, lhs: &Tensor<'c>, rhs: &Tensor<'c>) -> Tensor<'c> {
        let op = match lhs.dtype() {
            DType::F32 | DType::F64 | DType::BF16 => "arith.mulf",
            DType::I32 | DType::I64 => "arith.muli",
        };
        self.emit_linalg_binary(op, lhs, rhs)
    }

    /// Element-wise division (F32/F64: arith.divf; I32/I64: arith.divsi).
    pub fn emit_div(&mut self, lhs: &Tensor<'c>, rhs: &Tensor<'c>) -> Tensor<'c> {
        let op = match lhs.dtype() {
            DType::F32 | DType::F64 | DType::BF16 => "arith.divf",
            DType::I32 | DType::I64 => "arith.divsi",
        };
        self.emit_linalg_binary(op, lhs, rhs)
    }

    /// Element-wise power via math.powf (float only).
    pub fn emit_pow(&mut self, base: &Tensor<'c>, exponent: &Tensor<'c>) -> Tensor<'c> {
        self.emit_linalg_binary("math.powf", base, exponent)
    }

    // ── Elementwise unary ops ─────────────────────────────────────────────────

    /// Element-wise negation (F32/F64: arith.negf; I32/I64: arith.negsi not standard —
    /// use arith.subi(0, x) for integers via a special path).
    pub fn emit_neg(&mut self, input: &Tensor<'c>) -> Tensor<'c> {
        match input.dtype() {
            DType::F32 | DType::F64 | DType::BF16 => self.emit_linalg_unary("arith.negf", input),
            DType::I32 | DType::I64 => {
                // Emit 0 - x via linalg.generic with arith.subi where lhs is a zero constant.
                self.emit_linalg_unary_int_neg(input)
            }
        }
    }

    /// Element-wise exp via math.exp (float only).
    pub fn emit_exp(&mut self, input: &Tensor<'c>) -> Tensor<'c> {
        self.emit_linalg_unary("math.exp", input)
    }

    /// Element-wise tanh via math.tanh (float only).
    pub fn emit_tanh(&mut self, input: &Tensor<'c>) -> Tensor<'c> {
        self.emit_linalg_unary("math.tanh", input)
    }

    /// Element-wise relu via arith.maximumf(x, 0.0).
    pub fn emit_relu(&mut self, input: &Tensor<'c>) -> Tensor<'c> {
        self.emit_linalg_unary_relu(input)
    }

    /// Element-wise reciprocal (1/x) via arith.divf(1.0, x).
    pub fn emit_reciprocal(&mut self, input: &Tensor<'c>) -> Tensor<'c> {
        self.emit_linalg_unary_reciprocal(input)
    }

    /// Element-wise reciprocal square root via math.rsqrt.
    pub fn emit_rsqrt(&mut self, input: &Tensor<'c>) -> Tensor<'c> {
        self.emit_linalg_unary("math.rsqrt", input)
    }

    /// Element-wise sqrt via math.sqrt.
    pub fn emit_sqrt(&mut self, input: &Tensor<'c>) -> Tensor<'c> {
        self.emit_linalg_unary("math.sqrt", input)
    }

    /// SiLU activation: x * sigmoid(x) = x / (1 + exp(-x)).
    ///
    /// Decomposed into existing ops; a fused kernel can replace this later.
    pub fn emit_silu(&mut self, input: &Tensor<'c>) -> Tensor<'c> {
        let neg_x = self.emit_neg(input);
        let exp_neg = self.emit_exp(&neg_x);
        let one = self.emit_arith_constant(1.0, input.dtype());
        let denom = self.emit_add(&one, &exp_neg);
        self.emit_div(input, &denom)
    }
    // ── linalg.generic helpers ────────────────────────────────────────────────

    /// Compute the broadcast output shape from two input shapes.
    ///
    /// Returns `None` for dims that are dynamic on at least one side
    /// without a clear static winner. Broadcasting rules:
    /// - Shapes are right-aligned (prepend 1s to the shorter one).
    /// - dim = max(a, b) when one of them is 1; error if both are static and
    ///   neither is 1 and they differ.
    fn compute_broadcast_shape(
        lhs_shape: &[Option<u64>],
        rhs_shape: &[Option<u64>],
    ) -> Vec<Option<u64>> {
        let out_rank = lhs_shape.len().max(rhs_shape.len());
        // Right-align by padding with 1s on the left.
        let lhs_padded: Vec<Option<u64>> = std::iter::repeat_n(Some(1), out_rank - lhs_shape.len())
            .chain(lhs_shape.iter().copied())
            .collect();
        let rhs_padded: Vec<Option<u64>> = std::iter::repeat_n(Some(1), out_rank - rhs_shape.len())
            .chain(rhs_shape.iter().copied())
            .collect();

        lhs_padded
            .iter()
            .zip(rhs_padded.iter())
            .map(|(l, r)| match (l, r) {
                (Some(1), other) | (other, Some(1)) => *other,
                (Some(a), Some(b)) if a == b => Some(*a),
                (None, _) | (_, None) => None,
                (Some(a), Some(b)) => panic!("broadcast shape mismatch: {a} vs {b}"),
            })
            .collect()
    }

    /// Emit a `linalg.broadcast` to expand `input` to the target `out_shape`.
    ///
    /// `broadcast_dims` is the list of output dims that are NEW (not present in
    /// input). E.g. to go from rank-1 [4] to rank-2 [3,4], broadcast_dims=[0].
    fn emit_linalg_broadcast(
        &mut self,
        input: melior::ir::Value<'c, 'c>,
        out_shape: &[Option<u64>],
        dtype: DType,
        broadcast_dims: &[usize],
    ) -> melior::ir::Value<'c, 'c> {
        let init = self.emit_tensor_empty_dyn(out_shape, dtype, Some(input));

        // dimensions attribute: array<i64: d0, d1, ...>
        let dims_str = broadcast_dims
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let dims_attr = Attribute::parse(self.context, &format!("array<i64: {dims_str}>"))
            .expect("broadcast dimensions attr");

        let dims_u64: Vec<u64> = out_shape
            .iter()
            .map(|d| match d {
                Some(n) => *n,
                None => i64::MIN as u64,
            })
            .collect();
        let elem_type = dtype.to_mlir_type(self.context);
        let tensor_type: melior::ir::Type =
            RankedTensorType::new(&dims_u64, elem_type, None).into();

        // linalg.broadcast requires a region with a linalg.yield body.
        let body_block = Block::new(&[(elem_type, self.location), (elem_type, self.location)]);
        let in_val: melior::ir::Value = body_block.argument(0).unwrap().into();
        body_block.append_operation(
            OperationBuilder::new("linalg.yield", self.location)
                .add_operands(&[in_val])
                .build()
                .expect("linalg.yield"),
        );
        let body_region = Region::new();
        body_region.append_block(body_block);

        self.block
            .append_operation(
                OperationBuilder::new("linalg.broadcast", self.location)
                    .add_operands(&[input, init])
                    .add_results(&[tensor_type])
                    .add_attributes(&[(Identifier::new(self.context, "dimensions"), dims_attr)])
                    .add_regions([body_region])
                    .build()
                    .expect("linalg.broadcast"),
            )
            .result(0)
            .unwrap()
            .into()
    }

    /// Promote `input` from its current shape to `target_shape` using
    /// `linalg.broadcast` for added dimensions and projected permutation maps
    /// for same-rank size-1 broadcast dims.
    ///
    /// Returns a new `Tensor` with `target_shape`.
    pub(super) fn broadcast_to(&mut self, input: &Tensor<'c>, target_shape: &[Option<u64>]) -> Tensor<'c> {
        let src_shape = input.shape();
        if src_shape == target_shape {
            return *input;
        }

        let out_rank = target_shape.len();
        let src_rank = src_shape.len();
        let dtype = input.dtype();

        // Step 1: rank promotion (prepend 1-dims) via linalg.broadcast.
        let (rank_promoted_val, promoted_shape) = if src_rank < out_rank {
            let extra = out_rank - src_rank;
            let broadcast_dims: Vec<usize> = (0..extra).collect();
            let promoted_shape: Vec<Option<u64>> = std::iter::repeat_n(Some(1), extra)
                .chain(src_shape.iter().copied())
                .collect();
            let val =
                self.emit_linalg_broadcast(input.value(), &promoted_shape, dtype, &broadcast_dims);
            (val, promoted_shape)
        } else {
            (input.value(), src_shape.clone())
        };

        // Step 2: size-1 dimension expansion via linalg.generic with projected
        // permutation indexing maps.
        let needs_dim_expand = promoted_shape
            .iter()
            .zip(target_shape.iter())
            .any(|(s, t)| *s == Some(1) && *t != Some(1));

        if !needs_dim_expand {
            // Shape already matches target after rank promotion.
            return Tensor::from_value(rank_promoted_val);
        }

        // Build indexing maps with 0 for broadcast (size-1) dims.
        let mut input_exprs = Vec::new();
        let mut output_exprs = Vec::new();
        for (i, (s, _t)) in promoted_shape.iter().zip(target_shape.iter()).enumerate() {
            output_exprs.push(format!("d{i}"));
            if *s == Some(1) {
                input_exprs.push("0".to_string());
            } else {
                input_exprs.push(format!("d{i}"));
            }
        }

        let dim_vars: Vec<String> = (0..out_rank).map(|i| format!("d{i}")).collect();
        let dim_list = dim_vars.join(", ");
        let in_map = format!("affine_map<({dim_list}) -> ({})>", input_exprs.join(", "));
        let out_map = format!("affine_map<({dim_list}) -> ({})>", output_exprs.join(", "));
        let indexing_maps = Attribute::parse(self.context, &format!("[{in_map}, {out_map}]"))
            .expect("broadcast projected maps");

        let iterator_types = self.make_iterator_types(out_rank);
        let elem_type = dtype.to_mlir_type(self.context);

        let dims_u64: Vec<u64> = target_shape
            .iter()
            .map(|d| match d {
                Some(n) => *n,
                None => i64::MIN as u64,
            })
            .collect();
        let tensor_type: melior::ir::Type =
            RankedTensorType::new(&dims_u64, elem_type, None).into();

        let init = self.emit_tensor_empty_dyn(target_shape, dtype, Some(rank_promoted_val));

        // Body: just forward the input element (identity).
        let body_block = Block::new(&[
            (elem_type, self.location), // input element
            (elem_type, self.location), // output element (unused)
        ]);
        let in_val: melior::ir::Value = body_block.argument(0).unwrap().into();
        body_block.append_operation(
            OperationBuilder::new("linalg.yield", self.location)
                .add_operands(&[in_val])
                .build()
                .expect("linalg.yield"),
        );
        let body_region = Region::new();
        body_region.append_block(body_block);

        let result = self
            .block
            .append_operation(
                OperationBuilder::new("linalg.generic", self.location)
                    .add_operands(&[rank_promoted_val, init])
                    .add_results(&[tensor_type])
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
                            Attribute::parse(self.context, "array<i32: 1, 1>").unwrap(),
                        ),
                    ])
                    .add_regions([body_region])
                    .build()
                    .expect("linalg.generic broadcast expand"),
            )
            .result(0)
            .unwrap()
            .into();

        Tensor::from_value(result)
    }

    /// Emit a `linalg.generic` binary elementwise op with broadcast support.
    ///
    /// Uses broadcast-aware indexing maps directly in the linalg.generic
    /// instead of pre-broadcasting inputs. This avoids canonicalize fusion
    /// bugs where a separate broadcast generic gets merged with the binary
    /// generic and produces incorrect identity maps.
    fn emit_linalg_binary(
        &mut self,
        body_op: &str,
        lhs: &Tensor<'c>,
        rhs: &Tensor<'c>,
    ) -> Tensor<'c> {
        let lhs_shape = lhs.shape();
        let rhs_shape = rhs.shape();
        let out_shape = Self::compute_broadcast_shape(&lhs_shape, &rhs_shape);
        let dtype = lhs.dtype();
        let out_rank = out_shape.len();
        let elem_type = dtype.to_mlir_type(self.context);

        // Rank-promote inputs if needed (insert leading size-1 dims).
        let (lhs_val, lhs_padded) = self.rank_promote_for_broadcast(lhs, out_rank);
        let (rhs_val, rhs_padded) = self.rank_promote_for_broadcast(rhs, out_rank);

        // Build broadcast-aware indexing maps.
        let lhs_map = Self::make_broadcast_map(out_rank, &lhs_padded, &out_shape);
        let rhs_map = Self::make_broadcast_map(out_rank, &rhs_padded, &out_shape);
        let out_map = identity_map_str(out_rank);

        let indexing_maps =
            Attribute::parse(self.context, &format!("[{lhs_map}, {rhs_map}, {out_map}]"))
                .expect("broadcast indexing_maps");
        let iterator_types = self.make_iterator_types(out_rank);

        let out_type = self.make_tensor_type(&out_shape, dtype);

        // For tensor.empty, find a source tensor that has the output's dynamic dims.
        // Prefer the operand that is NOT all-broadcast (has matching dims).
        let dyn_source = if lhs_padded == out_shape {
            Some(lhs_val)
        } else if rhs_padded == out_shape {
            Some(rhs_val)
        } else {
            // Neither operand matches fully — need to build dynamic dims from
            // whichever operand provides each dim.
            Some(self.emit_tensor_empty_for_broadcast(
                &out_shape,
                &lhs_padded,
                lhs_val,
                &rhs_padded,
                rhs_val,
                dtype,
            ))
        };
        let init = if let Some(src) = dyn_source {
            // If src is already a tensor.empty Value, use it directly.
            // Otherwise, emit tensor.empty with dims from src.
            if lhs_padded == out_shape || rhs_padded == out_shape {
                self.emit_tensor_empty_dyn(&out_shape, dtype, Some(src))
            } else {
                src // emit_tensor_empty_for_broadcast already returned a tensor.empty
            }
        } else {
            self.emit_tensor_empty_dyn(&out_shape, dtype, None)
        };

        let body_block = Block::new(&[
            (elem_type, self.location),
            (elem_type, self.location),
            (elem_type, self.location),
        ]);
        let a: melior::ir::Value = body_block.argument(0).unwrap().into();
        let b: melior::ir::Value = body_block.argument(1).unwrap().into();
        let mut builder = OperationBuilder::new(body_op, self.location)
            .add_operands(&[a, b])
            .add_results(&[elem_type]);
        if body_op.ends_with('f') {
            // Float arith ops: enable FMA contraction.
            let (id, val) = self.fastmath_contract_attr();
            builder = builder.add_attributes(&[(id, val)]);
        }
        let op_result = body_block
            .append_operation(
                builder
                    .build()
                    .unwrap_or_else(|e| panic!("{body_op} in linalg body: {e}")),
            )
            .result(0)
            .unwrap()
            .into();
        body_block.append_operation(
            OperationBuilder::new("linalg.yield", self.location)
                .add_operands(&[op_result])
                .build()
                .expect("linalg.yield"),
        );
        let body_region = Region::new();
        body_region.append_block(body_block);

        let result = self
            .block
            .append_operation(
                OperationBuilder::new("linalg.generic", self.location)
                    .add_operands(&[lhs_val, rhs_val, init])
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
                    .expect("linalg.generic binary"),
            )
            .result(0)
            .unwrap()
            .into();
        Tensor::from_value(result)
    }

    /// Promote a tensor to the target rank by inserting leading size-1 dims.
    /// Returns the promoted value and its padded shape.
    fn rank_promote_for_broadcast(
        &mut self,
        tensor: &Tensor<'c>,
        target_rank: usize,
    ) -> (melior::ir::Value<'c, 'c>, Vec<Option<u64>>) {
        let shape = tensor.shape();
        if shape.len() == target_rank {
            return (tensor.value(), shape);
        }
        let extra = target_rank - shape.len();
        let axes: Vec<i64> = (0..extra as i64).collect();
        let promoted = self.emit_unsqueeze(tensor, &axes);
        let padded = promoted.shape();
        (promoted.value(), padded)
    }

    /// Build an affine map string for broadcast: `(d0, d1, ...) -> (expr0, expr1, ...)`
    /// where `expr_i = 0` if `operand_shape[i] == Some(1)` and `out_shape[i] != Some(1)`,
    /// otherwise `expr_i = d_i`.
    fn make_broadcast_map(
        out_rank: usize,
        operand_shape: &[Option<u64>],
        out_shape: &[Option<u64>],
    ) -> String {
        assert_eq!(operand_shape.len(), out_rank);
        let dim_vars: Vec<String> = (0..out_rank).map(|i| format!("d{i}")).collect();
        let dim_list = dim_vars.join(", ");

        let result_exprs: Vec<String> = (0..out_rank)
            .map(|i| {
                if operand_shape[i] == Some(1) && out_shape[i] != Some(1) {
                    "0".to_string()
                } else {
                    dim_vars[i].clone()
                }
            })
            .collect();
        let result_str = result_exprs.join(", ");
        format!("affine_map<({dim_list}) -> ({result_str})>")
    }

    /// Emit a `tensor.empty` for broadcast output when neither input fully matches
    /// the output shape. Picks dynamic dim values from whichever operand provides them.
    fn emit_tensor_empty_for_broadcast(
        &mut self,
        out_shape: &[Option<u64>],
        lhs_shape: &[Option<u64>],
        lhs_val: melior::ir::Value<'c, 'c>,
        _rhs_shape: &[Option<u64>],
        rhs_val: melior::ir::Value<'c, 'c>,
        dtype: DType,
    ) -> melior::ir::Value<'c, 'c> {
        let tensor_type = self.make_tensor_type(out_shape, dtype);
        let mut dyn_vals: Vec<melior::ir::Value<'c, 'c>> = Vec::new();

        for (i, dim) in out_shape.iter().enumerate() {
            if dim.is_none() {
                // Pick the operand that is NOT broadcast at this dim.
                let src = if lhs_shape[i] != Some(1) {
                    lhs_val
                } else {
                    rhs_val
                };
                dyn_vals.push(self.emit_tensor_dim(src, i));
            }
        }

        self.block
            .append_operation(
                OperationBuilder::new("tensor.empty", self.location)
                    .add_operands(&dyn_vals)
                    .add_results(&[tensor_type])
                    .build()
                    .expect("tensor.empty broadcast"),
            )
            .result(0)
            .unwrap()
            .into()
    }

    /// Emit a `linalg.generic` unary elementwise op (single-operand body op).
    /// Emit a unary `linalg.generic` op. The `build_body` closure receives the
    /// body block, the input element value, and the element type, and must
    /// append ops to the block and return the result value to yield.
    fn emit_linalg_unary_with_body(
        &mut self,
        input: &Tensor<'c>,
        build_body: impl FnOnce(&Block, melior::ir::Value, melior::ir::Type),
    ) -> Tensor<'c> {
        let shape = input.shape();
        let dtype = input.dtype();
        let rank = shape.len();
        let elem_type = dtype.to_mlir_type(self.context);

        let dims_u64: Vec<u64> = shape
            .iter()
            .map(|d| match d {
                Some(n) => *n,
                None => i64::MIN as u64,
            })
            .collect();
        let tensor_type: melior::ir::Type =
            RankedTensorType::new(&dims_u64, elem_type, None).into();

        let init = self.emit_tensor_empty_dyn(&shape, dtype, Some(input.value()));
        let identity = identity_map_str(rank);
        let indexing_maps = Attribute::parse(self.context, &format!("[{0}, {0}]", identity))
            .expect("indexing_maps");
        let iterator_types = self.make_iterator_types(rank);

        let body_block = Block::new(&[(elem_type, self.location), (elem_type, self.location)]);
        let x: melior::ir::Value = body_block.argument(0).unwrap().into();

        build_body(&body_block, x, elem_type);

        let body_region = Region::new();
        body_region.append_block(body_block);

        let result = self
            .block
            .append_operation(
                OperationBuilder::new("linalg.generic", self.location)
                    .add_operands(&[input.value(), init])
                    .add_results(&[tensor_type])
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
                            Attribute::parse(self.context, "array<i32: 1, 1>").unwrap(),
                        ),
                    ])
                    .add_regions([body_region])
                    .build()
                    .expect("linalg.generic unary"),
            )
            .result(0)
            .unwrap()
            .into();
        Tensor::from_value(result)
    }

    fn emit_linalg_unary(&mut self, body_op: &str, input: &Tensor<'c>) -> Tensor<'c> {
        let op_name = body_op.to_string();
        let loc = self.location;
        self.emit_linalg_unary_with_body(input, |block, x, elem_type| {
            let op_result = block
                .append_operation(
                    OperationBuilder::new(&op_name, loc)
                        .add_operands(&[x])
                        .add_results(&[elem_type])
                        .build()
                        .unwrap_or_else(|e| panic!("{op_name} in linalg body: {e}")),
                )
                .result(0)
                .unwrap()
                .into();
            block.append_operation(
                OperationBuilder::new("linalg.yield", loc)
                    .add_operands(&[op_result])
                    .build()
                    .expect("linalg.yield"),
            );
        })
    }

    /// Integer negation via `0 - x` (arith has no negsi).
    fn emit_linalg_unary_int_neg(&mut self, input: &Tensor<'c>) -> Tensor<'c> {
        let dtype = input.dtype();
        let ctx = self.context;
        let loc = self.location;
        self.emit_linalg_unary_with_body(input, |block, x, elem_type| {
            let zero_attr = match dtype {
                DType::I32 => Attribute::parse(ctx, "0 : i32"),
                DType::I64 => Attribute::parse(ctx, "0 : i64"),
                _ => unreachable!(),
            }
            .expect("zero constant");
            let zero: melior::ir::Value = block
                .append_operation(
                    OperationBuilder::new("arith.constant", loc)
                        .add_results(&[elem_type])
                        .add_attributes(&[(Identifier::new(ctx, "value"), zero_attr)])
                        .build()
                        .expect("arith.constant zero"),
                )
                .result(0)
                .unwrap()
                .into();
            let neg: melior::ir::Value = block
                .append_operation(
                    OperationBuilder::new("arith.subi", loc)
                        .add_operands(&[zero, x])
                        .add_results(&[elem_type])
                        .build()
                        .expect("arith.subi"),
                )
                .result(0)
                .unwrap()
                .into();
            block.append_operation(
                OperationBuilder::new("linalg.yield", loc)
                    .add_operands(&[neg])
                    .build()
                    .expect("linalg.yield"),
            );
        })
    }

    /// ReLU: max(x, 0.0) via `arith.maximumf`.
    fn emit_linalg_unary_relu(&mut self, input: &Tensor<'c>) -> Tensor<'c> {
        let dtype = input.dtype();
        let ctx = self.context;
        let loc = self.location;
        self.emit_linalg_unary_with_body(input, |block, x, elem_type| {
            let zero_attr = match dtype {
                DType::F32 => Attribute::parse(ctx, "0.0 : f32"),
                DType::F64 => Attribute::parse(ctx, "0.0 : f64"),
                DType::BF16 => Attribute::parse(ctx, "0.0 : bf16"),
                DType::I32 => Attribute::parse(ctx, "0 : i32"),
                DType::I64 => Attribute::parse(ctx, "0 : i64"),
            }
            .expect("zero for relu");
            let zero: melior::ir::Value = block
                .append_operation(
                    OperationBuilder::new("arith.constant", loc)
                        .add_results(&[elem_type])
                        .add_attributes(&[(Identifier::new(ctx, "value"), zero_attr)])
                        .build()
                        .expect("arith.constant"),
                )
                .result(0)
                .unwrap()
                .into();
            let relu_op = match dtype {
                DType::F32 | DType::F64 | DType::BF16 => "arith.maximumf",
                DType::I32 | DType::I64 => "arith.maxsi",
            };
            let relu_val: melior::ir::Value = block
                .append_operation(
                    OperationBuilder::new(relu_op, loc)
                        .add_operands(&[x, zero])
                        .add_results(&[elem_type])
                        .build()
                        .expect(relu_op),
                )
                .result(0)
                .unwrap()
                .into();
            block.append_operation(
                OperationBuilder::new("linalg.yield", loc)
                    .add_operands(&[relu_val])
                    .build()
                    .expect("linalg.yield"),
            );
        })
    }

    /// Reciprocal: 1.0 / x via `arith.divf`.
    fn emit_linalg_unary_reciprocal(&mut self, input: &Tensor<'c>) -> Tensor<'c> {
        let dtype = input.dtype();
        let ctx = self.context;
        let loc = self.location;
        self.emit_linalg_unary_with_body(input, |block, x, elem_type| {
            let one_attr = match dtype {
                DType::F32 => Attribute::parse(ctx, "1.0 : f32"),
                DType::F64 => Attribute::parse(ctx, "1.0 : f64"),
                _ => panic!("reciprocal is float-only"),
            }
            .expect("one for reciprocal");
            let one: melior::ir::Value = block
                .append_operation(
                    OperationBuilder::new("arith.constant", loc)
                        .add_results(&[elem_type])
                        .add_attributes(&[(Identifier::new(ctx, "value"), one_attr)])
                        .build()
                        .expect("arith.constant"),
                )
                .result(0)
                .unwrap()
                .into();
            let recip: melior::ir::Value = block
                .append_operation(
                    OperationBuilder::new("arith.divf", loc)
                        .add_operands(&[one, x])
                        .add_results(&[elem_type])
                        .build()
                        .expect("arith.divf reciprocal"),
                )
                .result(0)
                .unwrap()
                .into();
            block.append_operation(
                OperationBuilder::new("linalg.yield", loc)
                    .add_operands(&[recip])
                    .build()
                    .expect("linalg.yield"),
            );
        })
    }
    /// Element-wise `where(cond, x, y)` — select `x` where `cond != 0`, else `y`.
    ///
    /// Broadcasts `cond`, `x`, `y` to a common shape first.
    /// If `cond` dtype is I32/I64, casts to i1 via `arith.cmpi ne, %val, 0`.
    pub fn emit_where(&mut self, cond: &Tensor<'c>, x: &Tensor<'c>, y: &Tensor<'c>) -> Tensor<'c> {
        let x_shape = x.shape();
        let y_shape = y.shape();
        let cond_shape = cond.shape();
        let out_shape = {
            let xy = Self::compute_broadcast_shape(&x_shape, &y_shape);
            Self::compute_broadcast_shape(&cond_shape, &xy)
        };
        let dtype = x.dtype();
        let out_rank = out_shape.len();
        let elem_type = dtype.to_mlir_type(self.context);

        // Rank-promote inputs for broadcasting (insert leading 1-dims).
        let (cond_val, cond_padded) = self.rank_promote_for_broadcast(cond, out_rank);
        let (x_val, x_padded) = self.rank_promote_for_broadcast(x, out_rank);
        let (y_val, y_padded) = self.rank_promote_for_broadcast(y, out_rank);

        // Build broadcast-aware indexing maps (size-1 dims map to constant 0).
        let cond_map = Self::make_broadcast_map(out_rank, &cond_padded, &out_shape);
        let x_map = Self::make_broadcast_map(out_rank, &x_padded, &out_shape);
        let y_map = Self::make_broadcast_map(out_rank, &y_padded, &out_shape);
        let out_map = identity_map_str(out_rank);

        let indexing_maps = Attribute::parse(
            self.context,
            &format!("[{cond_map}, {x_map}, {y_map}, {out_map}]"),
        )
        .expect("where indexing_maps");
        let iterator_types = self.make_iterator_types(out_rank);

        let out_type = self.make_tensor_type(&out_shape, dtype);
        // Pick a dynamic-dim source for the empty tensor.
        // Need a tensor whose dynamic dims at the same positions as out_shape.
        let find_dyn_source = |padded: &[Option<u64>], _val: melior::ir::Value<'c, 'c>| -> bool {
            // The source must have the same rank and its dynamic positions must
            // cover all dynamic positions of out_shape.
            if padded.len() != out_shape.len() {
                return false;
            }
            out_shape
                .iter()
                .enumerate()
                .all(|(i, d)| d.is_some() || padded[i].is_none())
        };
        let init_source = if find_dyn_source(&x_padded, x_val) {
            Some(x_val)
        } else if find_dyn_source(&y_padded, y_val) {
            Some(y_val)
        } else if find_dyn_source(&cond_padded, cond_val) {
            Some(cond_val)
        } else {
            None
        };
        let init = self.emit_tensor_empty_dyn(&out_shape, dtype, init_source);

        let cond_dtype = cond.dtype();
        let cond_elem_type = cond_dtype.to_mlir_type(self.context);
        let i1_type = melior::ir::Type::parse(self.context, "i1").expect("i1 type");

        let body_block = Block::new(&[
            (cond_elem_type, self.location), // cond element
            (elem_type, self.location),      // x element
            (elem_type, self.location),      // y element
            (elem_type, self.location),      // out element (unused)
        ]);
        let cond_e: melior::ir::Value = body_block.argument(0).unwrap().into();
        let x_e: melior::ir::Value = body_block.argument(1).unwrap().into();
        let y_e: melior::ir::Value = body_block.argument(2).unwrap().into();

        // Convert cond to i1 if needed.
        let cond_i1: melior::ir::Value = match cond_dtype {
            DType::I32 | DType::I64 => {
                let zero_attr = match cond_dtype {
                    DType::I32 => Attribute::parse(self.context, "0 : i32"),
                    DType::I64 => Attribute::parse(self.context, "0 : i64"),
                    _ => unreachable!(),
                }
                .expect("zero for cond");
                let zero: melior::ir::Value = body_block
                    .append_operation(
                        OperationBuilder::new("arith.constant", self.location)
                            .add_results(&[cond_elem_type])
                            .add_attributes(&[(Identifier::new(self.context, "value"), zero_attr)])
                            .build()
                            .expect("arith.constant zero cond"),
                    )
                    .result(0)
                    .unwrap()
                    .into();
                // ne predicate = 1 for arith.cmpi.
                let ne_attr = Attribute::parse(self.context, "1 : i64").expect("ne predicate");
                body_block
                    .append_operation(
                        OperationBuilder::new("arith.cmpi", self.location)
                            .add_operands(&[cond_e, zero])
                            .add_results(&[i1_type])
                            .add_attributes(&[(
                                Identifier::new(self.context, "predicate"),
                                ne_attr,
                            )])
                            .build()
                            .expect("arith.cmpi ne"),
                    )
                    .result(0)
                    .unwrap()
                    .into()
            }
            DType::F32 | DType::F64 | DType::BF16 => {
                // Compare float != 0.0.
                let zero_attr = match cond_dtype {
                    DType::F32 => Attribute::parse(self.context, "0.0 : f32"),
                    DType::F64 => Attribute::parse(self.context, "0.0 : f64"),
                    DType::BF16 => Attribute::parse(self.context, "0.0 : bf16"),
                    _ => unreachable!(),
                }
                .expect("zero float cond");
                let zero: melior::ir::Value = body_block
                    .append_operation(
                        OperationBuilder::new("arith.constant", self.location)
                            .add_results(&[cond_elem_type])
                            .add_attributes(&[(Identifier::new(self.context, "value"), zero_attr)])
                            .build()
                            .expect("arith.constant zero float"),
                    )
                    .result(0)
                    .unwrap()
                    .into();
                // une (unordered not equal) predicate = 13 for arith.cmpf.
                let une_attr = Attribute::parse(self.context, "13 : i64").expect("une predicate");
                body_block
                    .append_operation(
                        OperationBuilder::new("arith.cmpf", self.location)
                            .add_operands(&[cond_e, zero])
                            .add_results(&[i1_type])
                            .add_attributes(&[(
                                Identifier::new(self.context, "predicate"),
                                une_attr,
                            )])
                            .build()
                            .expect("arith.cmpf une"),
                    )
                    .result(0)
                    .unwrap()
                    .into()
            }
        };

        let selected: melior::ir::Value = body_block
            .append_operation(
                OperationBuilder::new("arith.select", self.location)
                    .add_operands(&[cond_i1, x_e, y_e])
                    .add_results(&[elem_type])
                    .build()
                    .expect("arith.select"),
            )
            .result(0)
            .unwrap()
            .into();
        body_block.append_operation(
            OperationBuilder::new("linalg.yield", self.location)
                .add_operands(&[selected])
                .build()
                .expect("linalg.yield where"),
        );
        let body_region = Region::new();
        body_region.append_block(body_block);

        let result: melior::ir::Value<'c, 'c> = self
            .block
            .append_operation(
                OperationBuilder::new("linalg.generic", self.location)
                    .add_operands(&[cond_val, x_val, y_val, init])
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
                            Attribute::parse(self.context, "array<i32: 3, 1>").unwrap(),
                        ),
                    ])
                    .add_regions([body_region])
                    .build()
                    .expect("linalg.generic where"),
            )
            .result(0)
            .unwrap()
            .into();
        Tensor::from_value(result)
    }

    /// Cast every element of `input` to `target_dtype` via `linalg.generic`.
    ///
    /// Supported casts:
    /// - F32 ↔ F64: `arith.extf` / `arith.truncf`
    /// - I32 ↔ I64: `arith.extsi` / `arith.trunci`
    /// - I32/I64 → F32/F64: `arith.sitofp`
    /// - F32/F64 → I32/I64: `arith.fptosi`
    pub fn emit_cast(&mut self, input: &Tensor<'c>, target_dtype: DType) -> Tensor<'c> {
        let src_dtype = input.dtype();
        if src_dtype == target_dtype {
            return *input;
        }

        let cast_op = match (src_dtype, target_dtype) {
            (DType::F32, DType::F64) => "arith.extf",
            (DType::F64, DType::F32) => "arith.truncf",
            (DType::I32, DType::I64) => "arith.extsi",
            (DType::I64, DType::I32) => "arith.trunci",
            (DType::I32, DType::F32)
            | (DType::I32, DType::F64)
            | (DType::I64, DType::F32)
            | (DType::I64, DType::F64) => "arith.sitofp",
            (DType::F32, DType::I32)
            | (DType::F32, DType::I64)
            | (DType::F64, DType::I32)
            | (DType::F64, DType::I64) => "arith.fptosi",
            _ => panic!("emit_cast: unsupported cast {src_dtype:?} -> {target_dtype:?}"),
        };

        let shape = input.shape();
        let rank = shape.len();
        let src_elem_type = src_dtype.to_mlir_type(self.context);
        let tgt_elem_type = target_dtype.to_mlir_type(self.context);
        let out_type = self.make_tensor_type(&shape, target_dtype);

        let init = self.emit_tensor_empty_dyn(&shape, target_dtype, Some(input.value()));
        let identity = identity_map_str(rank);
        let indexing_maps = Attribute::parse(self.context, &format!("[{0}, {0}]", identity))
            .expect("cast indexing_maps");
        let iterator_types = self.make_iterator_types(rank);

        let body_block = Block::new(&[
            (src_elem_type, self.location),
            (tgt_elem_type, self.location),
        ]);
        let x: melior::ir::Value = body_block.argument(0).unwrap().into();
        let casted: melior::ir::Value = body_block
            .append_operation(
                OperationBuilder::new(cast_op, self.location)
                    .add_operands(&[x])
                    .add_results(&[tgt_elem_type])
                    .build()
                    .expect(cast_op),
            )
            .result(0)
            .unwrap()
            .into();
        body_block.append_operation(
            OperationBuilder::new("linalg.yield", self.location)
                .add_operands(&[casted])
                .build()
                .expect("linalg.yield cast"),
        );
        let body_region = Region::new();
        body_region.append_block(body_block);

        let result: melior::ir::Value<'c, 'c> = self
            .block
            .append_operation(
                OperationBuilder::new("linalg.generic", self.location)
                    .add_operands(&[input.value(), init])
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
                            Attribute::parse(self.context, "array<i32: 1, 1>").unwrap(),
                        ),
                    ])
                    .add_regions([body_region])
                    .build()
                    .expect("linalg.generic cast"),
            )
            .result(0)
            .unwrap()
            .into();
        Tensor::from_value(result)
    }
    /// Scale every element of a float tensor by a scalar constant (linalg.generic).
    pub(super) fn emit_linalg_scale_f32(&mut self, input: Tensor<'c>, scale: f32) -> Tensor<'c> {
        let shape = input.shape();
        let dtype = input.dtype();
        let elem_type = dtype.to_mlir_type(self.context);
        let rank = shape.len();

        let init = self.emit_tensor_empty_dyn(&shape, dtype, Some(input.value()));
        let out_type = self.make_tensor_type(&shape, dtype);
        let identity = identity_map_str(rank);
        let indexing_maps = Attribute::parse(self.context, &format!("[{0}, {0}]", identity))
            .expect("scale indexing_maps");
        let iterator_types = self.make_iterator_types(rank);

        // MLIR requires a decimal point for float literals.
        let scale_f64 = scale as f64;
        let scale_attr = match dtype {
            DType::F32 => Attribute::parse(self.context, &format!("{scale_f64:.6e} : f32")),
            DType::F64 => Attribute::parse(self.context, &format!("{scale_f64:.6e} : f64")),
            _ => panic!("emit_linalg_scale_f32: float-only"),
        }
        .expect("scale constant attr");

        let body_block = Block::new(&[(elem_type, self.location), (elem_type, self.location)]);
        let x: melior::ir::Value = body_block.argument(0).unwrap().into();
        let sc: melior::ir::Value = body_block
            .append_operation(
                OperationBuilder::new("arith.constant", self.location)
                    .add_results(&[elem_type])
                    .add_attributes(&[(Identifier::new(self.context, "value"), scale_attr)])
                    .build()
                    .expect("arith.constant scale"),
            )
            .result(0)
            .unwrap()
            .into();
        let scaled: melior::ir::Value = body_block
            .append_operation(
                OperationBuilder::new("arith.mulf", self.location)
                    .add_operands(&[x, sc])
                    .add_results(&[elem_type])
                    .add_attributes(&[self.fastmath_contract_attr()])
                    .build()
                    .expect("arith.mulf scale"),
            )
            .result(0)
            .unwrap()
            .into();
        body_block.append_operation(
            OperationBuilder::new("linalg.yield", self.location)
                .add_operands(&[scaled])
                .build()
                .expect("linalg.yield scale"),
        );
        let body_region = Region::new();
        body_region.append_block(body_block);

        let result = self
            .block
            .append_operation(
                OperationBuilder::new("linalg.generic", self.location)
                    .add_operands(&[input.value(), init])
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
                            Attribute::parse(self.context, "array<i32: 1, 1>").unwrap(),
                        ),
                    ])
                    .add_regions([body_region])
                    .build()
                    .expect("linalg.generic scale"),
            )
            .result(0)
            .unwrap()
            .into();
        Tensor::from_value(result)
    }
    /// Broadcast a scalar `f32` constant to the same shape as `like`.
    pub(super) fn emit_scalar_broadcast(&mut self, value: f32, like: &Tensor<'c>) -> Tensor<'c> {
        let shape = like.shape();
        let dtype = like.dtype();
        let elem_type = dtype.to_mlir_type(self.context);

        let value_f64 = value as f64;
        let val_attr = match dtype {
            DType::F32 => Attribute::parse(self.context, &format!("{value_f64:.6e} : f32")),
            DType::F64 => Attribute::parse(self.context, &format!("{value_f64:.6e} : f64")),
            _ => panic!("emit_scalar_broadcast: float only"),
        }
        .expect("scalar broadcast attr");

        let out_type = self.make_tensor_type(&shape, dtype);
        let init = self.emit_tensor_empty_dyn(&shape, dtype, Some(like.value()));

        // linalg.generic with empty inputs: body just yields the constant.
        let identity = identity_map_str(shape.len());
        let indexing_maps = Attribute::parse(self.context, &format!("[{identity}]"))
            .expect("scalar broadcast indexing_maps");
        let iterator_types = self.make_iterator_types(shape.len());

        let body_block = Block::new(&[(elem_type, self.location)]);
        let const_val: melior::ir::Value = body_block
            .append_operation(
                OperationBuilder::new("arith.constant", self.location)
                    .add_results(&[elem_type])
                    .add_attributes(&[(Identifier::new(self.context, "value"), val_attr)])
                    .build()
                    .expect("arith.constant scalar broadcast"),
            )
            .result(0)
            .unwrap()
            .into();
        body_block.append_operation(
            OperationBuilder::new("linalg.yield", self.location)
                .add_operands(&[const_val])
                .build()
                .expect("linalg.yield scalar broadcast"),
        );
        let body_region = Region::new();
        body_region.append_block(body_block);

        let result = self
            .block
            .append_operation(
                OperationBuilder::new("linalg.generic", self.location)
                    .add_operands(&[init])
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
                            Attribute::parse(self.context, "array<i32: 0, 1>").unwrap(),
                        ),
                    ])
                    .add_regions([body_region])
                    .build()
                    .expect("linalg.generic scalar broadcast"),
            )
            .result(0)
            .unwrap()
            .into();
        Tensor::from_value(result)
    }
}
