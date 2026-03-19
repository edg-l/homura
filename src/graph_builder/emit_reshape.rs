use super::*;

impl<'c> GraphBuilder<'c> {
    /// Transpose a 2D tensor `[M, N] -> [N, M]` via `linalg.transpose`.
    pub(super) fn emit_linalg_transpose_2d(&mut self, input: &Tensor<'c>) -> Tensor<'c> {
        let in_shape = input.shape();
        assert_eq!(
            in_shape.len(),
            2,
            "emit_linalg_transpose_2d requires rank-2 tensor"
        );
        let out_shape = vec![in_shape[1], in_shape[0]];
        let dtype = input.dtype();

        let init = self.emit_tensor_empty_with_dim_map(&out_shape, dtype, input.value(), &[1, 0]);
        let out_type = self.make_tensor_type(&out_shape, dtype);

        // permutation = [1, 0]
        let perm_attr =
            Attribute::parse(self.context, "array<i64: 1, 0>").expect("transpose permutation");

        // linalg.transpose region: single-block with 2 args (input_elem, output_elem).
        // Body yields the input element (arg 0).
        let elem_type = dtype.to_mlir_type(self.context);
        let trans_block = Block::new(&[(elem_type, self.location), (elem_type, self.location)]);
        let t_in: melior::ir::Value = trans_block.argument(0).unwrap().into();
        trans_block.append_operation(
            OperationBuilder::new("linalg.yield", self.location)
                .add_operands(&[t_in])
                .build()
                .expect("linalg.yield transpose"),
        );
        let trans_region = Region::new();
        trans_region.append_block(trans_block);

        let result = self
            .block
            .append_operation(
                OperationBuilder::new("linalg.transpose", self.location)
                    .add_operands(&[input.value(), init])
                    .add_results(&[out_type])
                    .add_attributes(&[(Identifier::new(self.context, "permutation"), perm_attr)])
                    .add_regions([trans_region])
                    .build()
                    .expect("linalg.transpose"),
            )
            .result(0)
            .unwrap()
            .into();
        Tensor::from_value(result)
    }

    // ── Phase 5: Shape Manipulation Ops ──────────────────────────────────────

    /// Reshape `input` to `target_shape`.
    ///
    /// `target_shape[i]` semantics:
    /// - positive `n`  → static dim n
    /// - `0`           → keep input dim unchanged (ONNX copy-from-input)
    /// - `-1`          → ONNX infer-this-dim: computed as `total_elements / product_of_known`
    ///
    /// Prefers `tensor.collapse_shape`/`tensor.expand_shape` over `tensor.reshape`
    /// to avoid broken strides from `memref.reshape` after bufferization.
    /// Falls back to `tensor.reshape` only when collapse/expand constraints aren't met.
    pub fn emit_reshape(&mut self, input: &Tensor<'c>, target_shape: &[i64]) -> Tensor<'c> {
        let in_shape = input.shape();

        // Resolve target_shape into Option<u64> (static where possible).
        let out_shape: Vec<Option<u64>> = target_shape
            .iter()
            .enumerate()
            .map(|(i, &d)| {
                if d > 0 {
                    Some(d as u64)
                } else if d == 0 {
                    in_shape.get(i).and_then(|opt| *opt)
                } else {
                    // d == -1: try to infer statically.
                    let total_static: Option<u64> = in_shape
                        .iter()
                        .try_fold(1u64, |acc, dim| dim.map(|n| acc * n));
                    let known_product: Option<u64> =
                        target_shape
                            .iter()
                            .enumerate()
                            .try_fold(1u64, |acc, (j, &td)| {
                                if j == i {
                                    Some(acc)
                                }
                                // skip the -1 dim
                                else if td > 0 {
                                    Some(acc * td as u64)
                                } else if td == 0 {
                                    in_shape.get(j).and_then(|opt| opt.map(|n| acc * n))
                                } else {
                                    None
                                }
                            });
                    match (total_static, known_product) {
                        (Some(t), Some(k)) if k > 0 => Some(t / k),
                        _ => None,
                    }
                }
            })
            .collect();

        // Try collapse_shape/expand_shape decomposition first.
        if let Some(reassoc) = compute_reassociation(&in_shape, &out_shape) {
            match reassoc {
                ReassocResult::Identity => {
                    return Tensor::from_value(input.value());
                }
                ReassocResult::Collapse {
                    reassoc,
                    inferred_shape,
                } => {
                    let reassoc_str = reassoc_to_string(&reassoc);
                    return self.emit_collapse_shape_with_reassoc(
                        input.value(),
                        &inferred_shape,
                        &reassoc_str,
                    );
                }
                ReassocResult::Expand { reassoc } => {
                    let reassoc_str = reassoc_to_string(&reassoc);
                    return self.emit_expand_shape_with_reassoc(
                        input.value(),
                        &out_shape,
                        &reassoc_str,
                    );
                }
                ReassocResult::CollapseExpand {
                    collapse_reassoc,
                    intermediate_shape,
                    expand_reassoc,
                } => {
                    let c_str = reassoc_to_string(&collapse_reassoc);
                    let collapsed = self.emit_collapse_shape_with_reassoc(
                        input.value(),
                        &intermediate_shape,
                        &c_str,
                    );
                    let e_str = reassoc_to_string(&expand_reassoc);
                    return self.emit_expand_shape_with_reassoc(
                        collapsed.value(),
                        &out_shape,
                        &e_str,
                    );
                }
            }
        }

        // Fallback: emit tensor.reshape (the old path).
        log_debug!(
            "emit_reshape: falling back to tensor.reshape (in={:?} out={:?})",
            in_shape,
            out_shape,
        );
        self.emit_reshape_tensor_op(input, &out_shape)
    }

    /// Emit `tensor.reshape` — the legacy path, used as fallback when
    /// collapse_shape/expand_shape constraints aren't met.
    fn emit_reshape_tensor_op(
        &mut self,
        input: &Tensor<'c>,
        out_shape: &[Option<u64>],
    ) -> Tensor<'c> {
        let in_shape = input.shape();
        let dtype = input.dtype();
        let rank = in_shape.len();
        let index_type = melior::ir::Type::parse(self.context, "index").expect("index type");
        let target_rank = out_shape.len();

        let emit_index_const =
            |block: &Block<'c>, ctx: &'c Context, n: u64| -> melior::ir::Value<'c, 'c> {
                let attr =
                    Attribute::parse(ctx, &format!("{n} : index")).expect("index const attr");
                block
                    .append_operation(
                        OperationBuilder::new("arith.constant", Location::unknown(ctx))
                            .add_results(&[melior::ir::Type::parse(ctx, "index").unwrap()])
                            .add_attributes(&[(Identifier::new(ctx, "value"), attr)])
                            .build()
                            .expect("arith.constant index"),
                    )
                    .result(0)
                    .unwrap()
                    .into()
            };

        // Build dim values — for dynamic dims, compute from input.
        let mut dim_vals: Vec<melior::ir::Value<'c, 'c>> = Vec::with_capacity(target_rank);

        // Compute total input elements for dynamic dims.
        let needs_infer = out_shape.iter().any(|d| d.is_none());
        let total_elems: Option<melior::ir::Value<'c, 'c>> = if needs_infer {
            let mut prod: Option<melior::ir::Value<'c, 'c>> = None;
            for (i, &dim) in in_shape[..rank].iter().enumerate() {
                let dim_val = match dim {
                    Some(n) => emit_index_const(&self.block, self.context, n),
                    None => self.emit_tensor_dim(input.value(), i),
                };
                prod = Some(match prod {
                    None => dim_val,
                    Some(prev) => self
                        .block
                        .append_operation(
                            OperationBuilder::new("arith.muli", self.location)
                                .add_operands(&[prev, dim_val])
                                .add_results(&[index_type])
                                .build()
                                .expect("arith.muli total"),
                        )
                        .result(0)
                        .unwrap()
                        .into(),
                });
            }
            prod
        } else {
            None
        };

        for (i, dim) in out_shape.iter().enumerate() {
            let val = match dim {
                Some(n) => emit_index_const(&self.block, self.context, *n),
                None => {
                    // Dynamic dim — use total/known_product or tensor.dim fallback.
                    if let Some(total) = total_elems {
                        let mut known = emit_index_const(&self.block, self.context, 1);
                        for (j, d) in out_shape.iter().enumerate() {
                            if j == i {
                                continue;
                            }
                            if let Some(n) = d {
                                let c = emit_index_const(&self.block, self.context, *n);
                                known = self
                                    .block
                                    .append_operation(
                                        OperationBuilder::new("arith.muli", self.location)
                                            .add_operands(&[known, c])
                                            .add_results(&[index_type])
                                            .build()
                                            .expect("arith.muli known"),
                                    )
                                    .result(0)
                                    .unwrap()
                                    .into();
                            }
                        }
                        self.block
                            .append_operation(
                                OperationBuilder::new("arith.divui", self.location)
                                    .add_operands(&[total, known])
                                    .add_results(&[index_type])
                                    .build()
                                    .expect("arith.divui infer"),
                            )
                            .result(0)
                            .unwrap()
                            .into()
                    } else if i < rank {
                        self.emit_tensor_dim(input.value(), i)
                    } else {
                        emit_index_const(&self.block, self.context, 1)
                    }
                }
            };
            dim_vals.push(val);
        }

        // Build shape tensor.
        let shape_tensor_type: melior::ir::Type = {
            let dims_u64 = vec![target_rank as u64];
            RankedTensorType::new(&dims_u64, index_type, None).into()
        };
        let shape_tensor: melior::ir::Value<'c, 'c> = self
            .block
            .append_operation(
                OperationBuilder::new("tensor.from_elements", self.location)
                    .add_operands(&dim_vals)
                    .add_results(&[shape_tensor_type])
                    .build()
                    .expect("tensor.from_elements shape"),
            )
            .result(0)
            .unwrap()
            .into();

        let out_type = self.make_tensor_type(out_shape, dtype);
        let result: melior::ir::Value<'c, 'c> = self
            .block
            .append_operation(
                OperationBuilder::new("tensor.reshape", self.location)
                    .add_operands(&[input.value(), shape_tensor])
                    .add_results(&[out_type])
                    .build()
                    .expect("tensor.reshape"),
            )
            .result(0)
            .unwrap()
            .into();
        Tensor::from_value(result)
    }

    /// Reshape `input` using a pre-built runtime shape tensor.
    ///
    /// `shape_tensor` is a 1-D `tensor<Nxindex>` value. `out_shape` describes
    /// the result type: `Some(n)` for static dims, `None` for dynamic.
    pub fn emit_reshape_with_tensor(
        &mut self,
        input: &Tensor<'c>,
        shape_tensor: melior::ir::Value<'c, 'c>,
        out_shape: &[Option<u64>],
    ) -> Tensor<'c> {
        let dtype = input.dtype();
        let out_type = self.make_tensor_type(out_shape, dtype);

        let result: melior::ir::Value<'c, 'c> = self
            .block
            .append_operation(
                OperationBuilder::new("tensor.reshape", self.location)
                    .add_operands(&[input.value(), shape_tensor])
                    .add_results(&[out_type])
                    .build()
                    .expect("tensor.reshape"),
            )
            .result(0)
            .unwrap()
            .into();
        Tensor::from_value(result)
    }

    /// Resolve ONNX special values in a runtime reshape shape.
    ///
    /// Given `dim_indices` (index values extracted from the ONNX shape tensor):
    /// - `-1` → inferred dimension = total_input_elems / product_of_other_dims
    /// - `0`  → copy from input dim i (when `allowzero` is false)
    ///
    /// Returns corrected index values ready for `emit_reshape_from_index_dims`.
    pub fn emit_resolve_reshape_dims(
        &mut self,
        input: &Tensor<'c>,
        dim_indices: &[melior::ir::Value<'c, 'c>],
        allowzero: bool,
    ) -> Vec<melior::ir::Value<'c, 'c>> {
        let in_shape = input.shape();
        let in_rank = in_shape.len();
        let index_type = melior::ir::Type::parse(self.context, "index").expect("index type");
        let loc = self.location;
        let ctx = self.context;

        // Constants we'll need.
        let c0 = self.emit_arith_constant_index(0);
        let c1 = self.emit_arith_constant_index(1);
        let c_neg1 = {
            // -1 as index (for comparison). arith.constant doesn't support negative index,
            // so cast from i64.
            let i64_type = melior::ir::Type::parse(ctx, "i64").expect("i64");
            let neg1_attr = Attribute::parse(ctx, "-1 : i64").expect("-1 i64 attr");
            let neg1_i64: melior::ir::Value = self
                .block
                .append_operation(
                    OperationBuilder::new("arith.constant", loc)
                        .add_results(&[i64_type])
                        .add_attributes(&[(Identifier::new(ctx, "value"), neg1_attr)])
                        .build()
                        .expect("arith.constant -1"),
                )
                .result(0)
                .unwrap()
                .into();
            self.block
                .append_operation(
                    OperationBuilder::new("arith.index_cast", loc)
                        .add_operands(&[neg1_i64])
                        .add_results(&[index_type])
                        .build()
                        .expect("arith.index_cast -1"),
                )
                .result(0)
                .unwrap()
                .into()
        };

        // Compute total input elements.
        let mut total: melior::ir::Value<'c, 'c> = c1;
        for (i, &dim) in in_shape[..in_rank].iter().enumerate() {
            let dim_val = match dim {
                Some(n) => self.emit_arith_constant_index(n),
                None => self.emit_tensor_dim(input.value(), i),
            };
            total = self
                .block
                .append_operation(
                    OperationBuilder::new("arith.muli", loc)
                        .add_operands(&[total, dim_val])
                        .add_results(&[index_type])
                        .build()
                        .expect("arith.muli total"),
                )
                .result(0)
                .unwrap()
                .into();
        }

        // First pass: resolve 0 entries (copy from input), leave -1 as-is.
        let resolved_zeros: Vec<melior::ir::Value<'c, 'c>> = dim_indices
            .iter()
            .enumerate()
            .map(|(i, &dim_val)| {
                if allowzero {
                    return dim_val;
                }
                // Check if dim == 0 using arith.cmpi eq.
                let is_zero: melior::ir::Value = self
                    .block
                    .append_operation(
                        OperationBuilder::new("arith.cmpi", loc)
                            .add_operands(&[dim_val, c0])
                            .add_results(&[melior::ir::Type::parse(ctx, "i1").unwrap()])
                            .add_attributes(&[(
                                Identifier::new(ctx, "predicate"),
                                Attribute::parse(ctx, "0 : i64").unwrap(), // eq predicate
                            )])
                            .build()
                            .expect("arith.cmpi eq zero"),
                    )
                    .result(0)
                    .unwrap()
                    .into();

                // If zero, use input dim i; else use dim_val.
                let input_dim = if i < in_rank {
                    match in_shape[i] {
                        Some(n) => self.emit_arith_constant_index(n),
                        None => self.emit_tensor_dim(input.value(), i),
                    }
                } else {
                    c0 // out of range, shouldn't happen in valid ONNX
                };

                self.block
                    .append_operation(
                        OperationBuilder::new("arith.select", loc)
                            .add_operands(&[is_zero, input_dim, dim_val])
                            .add_results(&[index_type])
                            .build()
                            .expect("arith.select zero"),
                    )
                    .result(0)
                    .unwrap()
                    .into()
            })
            .collect();

        // Compute product of known dims (non -1).
        let mut known_product: melior::ir::Value<'c, 'c> = c1;
        for &dim_val in &resolved_zeros {
            // Check if dim == -1.
            let is_neg1: melior::ir::Value = self
                .block
                .append_operation(
                    OperationBuilder::new("arith.cmpi", loc)
                        .add_operands(&[dim_val, c_neg1])
                        .add_results(&[melior::ir::Type::parse(ctx, "i1").unwrap()])
                        .add_attributes(&[(
                            Identifier::new(ctx, "predicate"),
                            Attribute::parse(ctx, "0 : i64").unwrap(), // eq predicate
                        )])
                        .build()
                        .expect("arith.cmpi eq neg1"),
                )
                .result(0)
                .unwrap()
                .into();

            // If -1, contribute 1 to the product; else contribute dim_val.
            let contrib: melior::ir::Value = self
                .block
                .append_operation(
                    OperationBuilder::new("arith.select", loc)
                        .add_operands(&[is_neg1, c1, dim_val])
                        .add_results(&[index_type])
                        .build()
                        .expect("arith.select neg1 contrib"),
                )
                .result(0)
                .unwrap()
                .into();

            known_product = self
                .block
                .append_operation(
                    OperationBuilder::new("arith.muli", loc)
                        .add_operands(&[known_product, contrib])
                        .add_results(&[index_type])
                        .build()
                        .expect("arith.muli known_product"),
                )
                .result(0)
                .unwrap()
                .into();
        }

        // Inferred dim value = total / known_product.
        let inferred: melior::ir::Value<'c, 'c> = self
            .block
            .append_operation(
                OperationBuilder::new("arith.divui", loc)
                    .add_operands(&[total, known_product])
                    .add_results(&[index_type])
                    .build()
                    .expect("arith.divui inferred"),
            )
            .result(0)
            .unwrap()
            .into();

        // Second pass: replace -1 with inferred.
        resolved_zeros
            .iter()
            .map(|&dim_val| {
                let is_neg1: melior::ir::Value = self
                    .block
                    .append_operation(
                        OperationBuilder::new("arith.cmpi", loc)
                            .add_operands(&[dim_val, c_neg1])
                            .add_results(&[melior::ir::Type::parse(ctx, "i1").unwrap()])
                            .add_attributes(&[(
                                Identifier::new(ctx, "predicate"),
                                Attribute::parse(ctx, "0 : i64").unwrap(), // eq predicate
                            )])
                            .build()
                            .expect("arith.cmpi eq neg1 final"),
                    )
                    .result(0)
                    .unwrap()
                    .into();

                self.block
                    .append_operation(
                        OperationBuilder::new("arith.select", loc)
                            .add_operands(&[is_neg1, inferred, dim_val])
                            .add_results(&[index_type])
                            .build()
                            .expect("arith.select replace neg1"),
                    )
                    .result(0)
                    .unwrap()
                    .into()
            })
            .collect()
    }

    /// Reshape `input` using corrected runtime index dim values.
    ///
    /// Tries collapse_shape/expand_shape first when the output shape has enough
    /// static information. Falls back to `tensor.reshape` via shape tensor.
    pub fn emit_reshape_from_index_dims(
        &mut self,
        input: &Tensor<'c>,
        dim_vals: &[melior::ir::Value<'c, 'c>],
        out_shape: &[Option<u64>],
    ) -> Tensor<'c> {
        let in_shape = input.shape();

        // Try collapse/expand decomposition.
        if let Some(reassoc) = compute_reassociation(&in_shape, out_shape) {
            match reassoc {
                ReassocResult::Identity => {
                    return Tensor::from_value(input.value());
                }
                ReassocResult::Collapse {
                    reassoc,
                    inferred_shape,
                } => {
                    let reassoc_str = reassoc_to_string(&reassoc);
                    return self.emit_collapse_shape_with_reassoc(
                        input.value(),
                        &inferred_shape,
                        &reassoc_str,
                    );
                }
                ReassocResult::Expand { reassoc } => {
                    let reassoc_str = reassoc_to_string(&reassoc);
                    // Use the pre-computed dim_vals for dynamic dims instead of
                    // tensor.dim (which only works for direct inheritance).
                    let dyn_vals: Vec<melior::ir::Value<'c, 'c>> = out_shape
                        .iter()
                        .enumerate()
                        .filter(|(_, d)| d.is_none())
                        .map(|(i, _)| dim_vals[i])
                        .collect();
                    return self.emit_expand_shape_impl_with_dyn_vals(
                        input.value(),
                        out_shape,
                        input.dtype(),
                        &reassoc_str,
                        &dyn_vals,
                    );
                }
                ReassocResult::CollapseExpand {
                    collapse_reassoc,
                    intermediate_shape,
                    expand_reassoc,
                } => {
                    let c_str = reassoc_to_string(&collapse_reassoc);
                    let collapsed = self.emit_collapse_shape_with_reassoc(
                        input.value(),
                        &intermediate_shape,
                        &c_str,
                    );
                    let e_str = reassoc_to_string(&expand_reassoc);
                    let dyn_vals: Vec<melior::ir::Value<'c, 'c>> = out_shape
                        .iter()
                        .enumerate()
                        .filter(|(_, d)| d.is_none())
                        .map(|(i, _)| dim_vals[i])
                        .collect();
                    return self.emit_expand_shape_impl_with_dyn_vals(
                        collapsed.value(),
                        out_shape,
                        input.dtype(),
                        &e_str,
                        &dyn_vals,
                    );
                }
            }
        }

        // Second attempt: if expand failed due to static→dynamic constraint,
        // cast the input to make those dims dynamic, then try expand again.
        if out_shape.len() > in_shape.len() {
            // Make all input dims that correspond to dynamic output groups dynamic.
            let cast_shape: Vec<Option<u64>> = in_shape.iter().map(|_| None).collect();
            let cast_val = self.emit_tensor_cast(input.value(), &cast_shape);
            let cast_in_shape = cast_val.shape();
            if let Some(reassoc) = compute_reassociation(&cast_in_shape, out_shape)
                && let ReassocResult::Expand { reassoc } = reassoc
            {
                let reassoc_str = reassoc_to_string(&reassoc);
                let dyn_vals: Vec<melior::ir::Value<'c, 'c>> = out_shape
                    .iter()
                    .enumerate()
                    .filter(|(_, d)| d.is_none())
                    .map(|(i, _)| dim_vals[i])
                    .collect();
                return self.emit_expand_shape_impl_with_dyn_vals(
                    cast_val.value(),
                    out_shape,
                    input.dtype(),
                    &reassoc_str,
                    &dyn_vals,
                );
            }
        }

        // Final fallback: tensor.reshape via shape tensor.
        log_debug!(
            "emit_reshape_from_index_dims: falling back to tensor.reshape (in={:?} out={:?})",
            in_shape,
            out_shape,
        );
        let index_type = melior::ir::Type::parse(self.context, "index").expect("index type");
        let n = dim_vals.len() as u64;
        let shape_tensor_type: melior::ir::Type =
            melior::ir::r#type::RankedTensorType::new(&[n], index_type, None).into();

        let shape_tensor: melior::ir::Value<'c, 'c> = self
            .block
            .append_operation(
                OperationBuilder::new("tensor.from_elements", self.location)
                    .add_operands(dim_vals)
                    .add_results(&[shape_tensor_type])
                    .build()
                    .expect("tensor.from_elements reshape dims"),
            )
            .result(0)
            .unwrap()
            .into();

        self.emit_reshape_with_tensor(input, shape_tensor, out_shape)
    }

    /// Transpose `input` according to `perms` (ONNX-style signed permutation).
    ///
    /// E.g. `perms = [2, 0, 1]` maps output dim i to input dim perms[i].
    /// Emits `linalg.transpose` named op with the given permutation.
    pub fn emit_transpose(&mut self, input: &Tensor<'c>, perms: &[i64]) -> Tensor<'c> {
        let in_shape = input.shape();
        let rank = in_shape.len();
        let dtype = input.dtype();
        assert_eq!(
            perms.len(),
            rank,
            "emit_transpose: perms len must match rank"
        );

        let norm_perms: Vec<usize> = perms
            .iter()
            .map(|&p| {
                let p = if p < 0 { p + rank as i64 } else { p };
                assert!(
                    p >= 0 && (p as usize) < rank,
                    "perm {p} out of bounds for rank {rank}"
                );
                p as usize
            })
            .collect();
        {
            let mut seen = vec![false; rank];
            for &p in &norm_perms {
                assert!(!seen[p], "duplicate index {p} in permutation");
                seen[p] = true;
            }
        }

        // Output shape: out_shape[i] = in_shape[perms[i]]
        let out_shape: Vec<Option<u64>> = norm_perms.iter().map(|&p| in_shape[p]).collect();

        // dim_map[i] = norm_perms[i] — output dim i reads source dim norm_perms[i].
        let dim_map: Vec<usize> = norm_perms.clone();
        let init = self.emit_tensor_empty_with_dim_map(&out_shape, dtype, input.value(), &dim_map);
        let out_type = self.make_tensor_type(&out_shape, dtype);

        let perm_vals: Vec<String> = norm_perms.iter().map(|p| p.to_string()).collect();
        let perm_attr = Attribute::parse(
            self.context,
            &format!("array<i64: {}>", perm_vals.join(", ")),
        )
        .expect("transpose permutation attr");

        let elem_type = dtype.to_mlir_type(self.context);
        let trans_block = Block::new(&[(elem_type, self.location), (elem_type, self.location)]);
        let t_in: melior::ir::Value = trans_block.argument(0).unwrap().into();
        trans_block.append_operation(
            OperationBuilder::new("linalg.yield", self.location)
                .add_operands(&[t_in])
                .build()
                .expect("linalg.yield transpose"),
        );
        let trans_region = Region::new();
        trans_region.append_block(trans_block);

        let result = self
            .block
            .append_operation(
                OperationBuilder::new("linalg.transpose", self.location)
                    .add_operands(&[input.value(), init])
                    .add_results(&[out_type])
                    .add_attributes(&[(Identifier::new(self.context, "permutation"), perm_attr)])
                    .add_regions([trans_region])
                    .build()
                    .expect("linalg.transpose"),
            )
            .result(0)
            .unwrap()
            .into();
        Tensor::from_value(result)
    }

    /// Concatenate `inputs` along `axis`.
    ///
    /// All inputs must have the same rank and the same shape in every dim except `axis`.
    /// Emits a pre-allocated `tensor.empty` plus one `tensor.insert_slice` per input.
    pub fn emit_concat(&mut self, inputs: &[Tensor<'c>], axis: usize) -> Tensor<'c> {
        assert!(!inputs.is_empty(), "emit_concat: inputs must not be empty");
        let rank = inputs[0].rank();
        let dtype = inputs[0].dtype();
        assert!(
            axis < rank,
            "emit_concat: axis {axis} out of bounds for rank {rank}"
        );
        let index_type = melior::ir::Type::parse(self.context, "index").expect("index type");

        let first_shape = inputs[0].shape();

        // Compute total size along the concat axis.
        // If all axis dims are static, sum them. Otherwise build a runtime sum.
        let mut total_axis_static: Option<u64> = Some(0);
        let mut total_axis_val: Option<melior::ir::Value<'c, 'c>> = None;

        for inp in inputs.iter() {
            let s = inp.shape();
            match s[axis] {
                Some(n) => {
                    if let Some(acc) = total_axis_static {
                        total_axis_static = Some(acc + n);
                    }
                    // Always build runtime value for uniformity when we need it.
                }
                None => {
                    total_axis_static = None;
                }
            }
        }

        // Build out_shape.
        let mut out_shape: Vec<Option<u64>> = first_shape.clone();
        out_shape[axis] = total_axis_static;

        // If any axis dim is dynamic, build runtime sum.
        if total_axis_static.is_none() {
            let mut acc: Option<melior::ir::Value<'c, 'c>> = None;
            for inp in inputs.iter() {
                let dim_val = match inp.shape()[axis] {
                    Some(n) => {
                        let attr = Attribute::parse(self.context, &format!("{n} : index"))
                            .expect("index const");
                        self.block
                            .append_operation(
                                OperationBuilder::new("arith.constant", self.location)
                                    .add_results(&[index_type])
                                    .add_attributes(&[(
                                        Identifier::new(self.context, "value"),
                                        attr,
                                    )])
                                    .build()
                                    .expect("arith.constant"),
                            )
                            .result(0)
                            .unwrap()
                            .into()
                    }
                    None => self.emit_tensor_dim(inp.value(), axis),
                };
                acc = Some(match acc {
                    None => dim_val,
                    Some(prev) => self
                        .block
                        .append_operation(
                            OperationBuilder::new("arith.addi", self.location)
                                .add_operands(&[prev, dim_val])
                                .add_results(&[index_type])
                                .build()
                                .expect("arith.addi axis sum"),
                        )
                        .result(0)
                        .unwrap()
                        .into(),
                });
            }
            total_axis_val = acc;
        }

        // Allocate output tensor. For the dynamic axis dim we need a runtime value.
        let out_tensor = if out_shape[axis].is_none() {
            // Build dyn vals array: for each dynamic dim in out_shape, emit tensor.dim.
            let total_val = total_axis_val.expect("total_axis_val");
            let tensor_type = self.make_tensor_type(&out_shape, dtype);
            let mut dyn_vals: Vec<melior::ir::Value<'c, 'c>> = Vec::new();
            for (i, dim) in out_shape.iter().enumerate() {
                if dim.is_none() {
                    if i == axis {
                        dyn_vals.push(total_val);
                    } else {
                        dyn_vals.push(self.emit_tensor_dim(inputs[0].value(), i));
                    }
                }
            }
            self.block
                .append_operation(
                    OperationBuilder::new("tensor.empty", self.location)
                        .add_operands(&dyn_vals)
                        .add_results(&[tensor_type])
                        .build()
                        .expect("tensor.empty concat"),
                )
                .result(0)
                .unwrap()
                .into()
        } else {
            self.emit_tensor_empty_dyn(&out_shape, dtype, Some(inputs[0].value()))
        };

        // Insert each input into the output tensor via tensor.insert_slice.
        // kDynamic sentinel = i64::MIN.
        const K_DYNAMIC: i64 = i64::MIN;
        let out_type = self.make_tensor_type(&out_shape, dtype);

        let mut current = out_tensor;
        // Track cumulative offset along the concat axis.
        let mut offset_static: i64 = 0;
        let mut offset_val: Option<melior::ir::Value<'c, 'c>> = None;

        for inp in inputs.iter() {
            let inp_shape = inp.shape();

            // Build static_offsets: 0 for all dims except axis.
            // For the axis: static if offset_static is known and axis dim static, else dynamic.
            let axis_offset_is_static = total_axis_static.is_some();
            let static_offsets: Vec<i64> = (0..rank)
                .map(|i| {
                    if i == axis {
                        if axis_offset_is_static {
                            offset_static
                        } else {
                            K_DYNAMIC
                        }
                    } else {
                        0
                    }
                })
                .collect();

            // Build static_sizes: size of each input dim.
            let static_sizes: Vec<i64> = (0..rank)
                .map(|i| match inp_shape[i] {
                    Some(n) => n as i64,
                    None => K_DYNAMIC,
                })
                .collect();

            // Strides = all 1.
            let static_strides: Vec<i64> = vec![1; rank];

            // Count dynamic operands needed (offsets, sizes).
            let dyn_offsets: Vec<melior::ir::Value<'c, 'c>> = if !axis_offset_is_static {
                let ov = match offset_val {
                    Some(v) => v,
                    None => {
                        // offset = 0
                        let attr = Attribute::parse(self.context, "0 : index").expect("0 index");
                        self.block
                            .append_operation(
                                OperationBuilder::new("arith.constant", self.location)
                                    .add_results(&[index_type])
                                    .add_attributes(&[(
                                        Identifier::new(self.context, "value"),
                                        attr,
                                    )])
                                    .build()
                                    .expect("arith.constant 0"),
                            )
                            .result(0)
                            .unwrap()
                            .into()
                    }
                };
                vec![ov]
            } else {
                vec![]
            };

            let dyn_sizes: Vec<melior::ir::Value<'c, 'c>> = (0..rank)
                .filter(|&i| inp_shape[i].is_none())
                .map(|i| self.emit_tensor_dim(inp.value(), i))
                .collect();

            let static_offsets_attr = {
                let s = static_offsets
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                Attribute::parse(self.context, &format!("array<i64: {s}>"))
                    .expect("static_offsets attr")
            };
            let static_sizes_attr = {
                let s = static_sizes
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                Attribute::parse(self.context, &format!("array<i64: {s}>"))
                    .expect("static_sizes attr")
            };
            let static_strides_attr = {
                let s = static_strides
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                Attribute::parse(self.context, &format!("array<i64: {s}>"))
                    .expect("static_strides attr")
            };

            let dyn_offsets_len = dyn_offsets.len() as i32;
            let dyn_sizes_len = dyn_sizes.len() as i32;
            let mut operands = vec![inp.value(), current];
            operands.extend(dyn_offsets);
            operands.extend(dyn_sizes);

            // operandSegmentSizes: [src, dest, dyn_offsets, dyn_sizes, dyn_strides]
            let seg_attr = Attribute::parse(
                self.context,
                &format!("array<i32: 1, 1, {dyn_offsets_len}, {dyn_sizes_len}, 0>"),
            )
            .expect("insert_slice operandSegmentSizes");

            current = self
                .block
                .append_operation(
                    OperationBuilder::new("tensor.insert_slice", self.location)
                        .add_operands(&operands)
                        .add_results(&[out_type])
                        .add_attributes(&[
                            (
                                Identifier::new(self.context, "static_offsets"),
                                static_offsets_attr,
                            ),
                            (
                                Identifier::new(self.context, "static_sizes"),
                                static_sizes_attr,
                            ),
                            (
                                Identifier::new(self.context, "static_strides"),
                                static_strides_attr,
                            ),
                            (
                                Identifier::new(self.context, "operandSegmentSizes"),
                                seg_attr,
                            ),
                        ])
                        .build()
                        .expect("tensor.insert_slice"),
                )
                .result(0)
                .unwrap()
                .into();

            // Advance offset.
            if axis_offset_is_static {
                offset_static += inp_shape[axis].expect("static axis dim") as i64;
            } else {
                let step = match inp_shape[axis] {
                    Some(n) => {
                        let attr = Attribute::parse(self.context, &format!("{n} : index"))
                            .expect("index const");
                        self.block
                            .append_operation(
                                OperationBuilder::new("arith.constant", self.location)
                                    .add_results(&[index_type])
                                    .add_attributes(&[(
                                        Identifier::new(self.context, "value"),
                                        attr,
                                    )])
                                    .build()
                                    .expect("arith.constant step"),
                            )
                            .result(0)
                            .unwrap()
                            .into()
                    }
                    None => self.emit_tensor_dim(inp.value(), axis),
                };
                let prev = match offset_val {
                    Some(v) => v,
                    None => {
                        let attr = Attribute::parse(self.context, "0 : index").expect("0 index");
                        self.block
                            .append_operation(
                                OperationBuilder::new("arith.constant", self.location)
                                    .add_results(&[index_type])
                                    .add_attributes(&[(
                                        Identifier::new(self.context, "value"),
                                        attr,
                                    )])
                                    .build()
                                    .expect("arith.constant 0 prev"),
                            )
                            .result(0)
                            .unwrap()
                            .into()
                    }
                };
                offset_val = Some(
                    self.block
                        .append_operation(
                            OperationBuilder::new("arith.addi", self.location)
                                .add_operands(&[prev, step])
                                .add_results(&[index_type])
                                .build()
                                .expect("arith.addi offset"),
                        )
                        .result(0)
                        .unwrap()
                        .into(),
                );
            }
        }

        Tensor::from_value(current)
    }

    /// Extract a slice of `input` along given `axes` (static path).
    ///
    /// All parameters are ONNX-style:
    /// - `starts[i]`   — start index along `axes[i]`
    /// - `ends[i]`     — exclusive end index along `axes[i]`
    /// - `axes[i]`     — which input dim to slice
    /// - `steps[i]`    — step/stride (must be ≥ 1)
    ///
    /// Non-mentioned axes are taken in full.
    pub fn emit_slice(
        &mut self,
        input: &Tensor<'c>,
        starts: &[i64],
        ends: &[i64],
        axes: &[i64],
        steps: &[i64],
    ) -> Tensor<'c> {
        let in_shape = input.shape();
        let rank = in_shape.len();
        let dtype = input.dtype();
        const K_DYNAMIC: i64 = i64::MIN;

        // Normalize axes to usize.
        let norm_axes: Vec<usize> = axes
            .iter()
            .map(|&a| {
                let a = if a < 0 { a + rank as i64 } else { a };
                a as usize
            })
            .collect();

        // Build per-dim (offset, size, stride) — defaulting to full dim for un-mentioned axes.
        let mut static_offsets = vec![0i64; rank];
        let mut static_sizes: Vec<i64> = in_shape
            .iter()
            .map(|d| match d {
                Some(n) => *n as i64,
                None => K_DYNAMIC,
            })
            .collect();
        let mut static_strides = vec![1i64; rank];

        for (j, &ax) in norm_axes.iter().enumerate() {
            let start = starts[j];
            let end = ends[j];
            let step = steps[j];

            // Normalize negative indices.
            let ax_size = match in_shape[ax] {
                Some(n) => n as i64,
                None => {
                    // Dynamic axis — emit K_DYNAMIC for size, use start as offset.
                    // Caller must use emit_dynamic_slice for non-trivial dynamic slicing.
                    let s = if start < 0 { 0 } else { start }; // can't normalize negative against unknown dim
                    static_offsets[ax] = s;
                    static_sizes[ax] = K_DYNAMIC;
                    static_strides[ax] = step;
                    continue;
                }
            };
            let s = if start < 0 { start + ax_size } else { start }.max(0);
            let e = if end < 0 {
                end + ax_size
            } else {
                end.min(ax_size)
            };
            let size = ((e - s + step - 1) / step).max(0);

            static_offsets[ax] = s;
            static_sizes[ax] = size;
            static_strides[ax] = step;
        }

        // Output shape.
        let out_shape: Vec<Option<u64>> = static_sizes
            .iter()
            .map(|&s| if s == K_DYNAMIC { None } else { Some(s as u64) })
            .collect();
        let out_type = self.make_tensor_type(&out_shape, dtype);

        let make_attr = |ctx: &'c Context, vals: &[i64]| -> Attribute<'c> {
            let s = vals
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            Attribute::parse(ctx, &format!("array<i64: {s}>")).expect("slice attr")
        };

        let dyn_sizes: Vec<melior::ir::Value<'c, 'c>> = static_sizes
            .iter()
            .enumerate()
            .filter(|&(_, s)| *s == K_DYNAMIC)
            .map(|(i, _)| self.emit_tensor_dim(input.value(), i))
            .collect();
        let dyn_sizes_len = dyn_sizes.len() as i32;

        let mut operands = vec![input.value()];
        // No dynamic offsets or strides — all static.
        operands.extend(dyn_sizes);

        // operandSegmentSizes: [source, dyn_offsets, dyn_sizes, dyn_strides]
        let seg_attr = Attribute::parse(
            self.context,
            &format!("array<i32: 1, 0, {dyn_sizes_len}, 0>"),
        )
        .expect("operandSegmentSizes");

        let result: melior::ir::Value<'c, 'c> = self
            .block
            .append_operation(
                OperationBuilder::new("tensor.extract_slice", self.location)
                    .add_operands(&operands)
                    .add_results(&[out_type])
                    .add_attributes(&[
                        (
                            Identifier::new(self.context, "static_offsets"),
                            make_attr(self.context, &static_offsets),
                        ),
                        (
                            Identifier::new(self.context, "static_sizes"),
                            make_attr(self.context, &static_sizes),
                        ),
                        (
                            Identifier::new(self.context, "static_strides"),
                            make_attr(self.context, &static_strides),
                        ),
                        (
                            Identifier::new(self.context, "operandSegmentSizes"),
                            seg_attr,
                        ),
                    ])
                    .build()
                    .expect("tensor.extract_slice"),
            )
            .result(0)
            .unwrap()
            .into();
        Tensor::from_value(result)
    }

    /// Extract a slice of `input` along given `axes` with runtime start/end values.
    ///
    /// - `starts[i]` — runtime MLIR Value (i64 or index) for start along `axes[i]`
    /// - `ends[i]`   — runtime MLIR Value (i64 or index) for exclusive end along `axes[i]`
    /// - `axes[i]`   — which input dim to slice (static)
    /// - `steps[i]`  — step/stride for each sliced axis (static, must be ≥ 1)
    ///
    /// Non-mentioned axes are taken in full (offset=0, full size, stride=1).
    /// Sliced axes produce dynamic output dims.
    pub fn emit_dynamic_slice(
        &mut self,
        input: &Tensor<'c>,
        starts: &[melior::ir::Value<'c, 'c>],
        ends: &[melior::ir::Value<'c, 'c>],
        axes: &[i64],
        steps: &[i64],
    ) -> Tensor<'c> {
        let in_shape = input.shape();
        let rank = in_shape.len();
        let dtype = input.dtype();
        const K_DYNAMIC: i64 = i64::MIN;

        let index_type = melior::ir::Type::parse(self.context, "index").expect("index type");

        // Normalize axes to usize.
        let norm_axes: Vec<usize> = axes
            .iter()
            .map(|&a| {
                let a = if a < 0 { a + rank as i64 } else { a };
                a as usize
            })
            .collect();

        // Per-dim static placeholders; sliced axes will be K_DYNAMIC.
        let mut static_offsets = vec![0i64; rank];
        let mut static_sizes: Vec<i64> = in_shape
            .iter()
            .map(|d| match d {
                Some(n) => *n as i64,
                None => K_DYNAMIC,
            })
            .collect();
        let mut static_strides = vec![1i64; rank];

        // Mark sliced axes as dynamic (offsets and sizes come from runtime values).
        for (j, &ax) in norm_axes.iter().enumerate() {
            static_offsets[ax] = K_DYNAMIC;
            static_sizes[ax] = K_DYNAMIC;
            static_strides[ax] = steps[j];
        }

        // Output shape: dynamic for sliced axes, static (or dynamic) for the rest.
        let out_shape: Vec<Option<u64>> = (0..rank)
            .map(|i| {
                if norm_axes.contains(&i) {
                    None
                } else {
                    in_shape[i]
                }
            })
            .collect();
        let out_type = self.make_tensor_type(&out_shape, dtype);

        // Build dynamic offsets and sizes for the sliced axes.
        let mut dyn_offsets: Vec<melior::ir::Value<'c, 'c>> = Vec::new();
        let mut dyn_sizes_per_axis: Vec<melior::ir::Value<'c, 'c>> = Vec::new();

        for (j, _ax) in norm_axes.iter().enumerate() {
            // Cast start/end to index type if they are i64.
            let start_idx: melior::ir::Value<'c, 'c> = if starts[j].r#type() == index_type {
                starts[j]
            } else {
                self.block
                    .append_operation(
                        OperationBuilder::new("arith.index_cast", self.location)
                            .add_operands(&[starts[j]])
                            .add_results(&[index_type])
                            .build()
                            .expect("arith.index_cast start"),
                    )
                    .result(0)
                    .unwrap()
                    .into()
            };
            let end_idx: melior::ir::Value<'c, 'c> = if ends[j].r#type() == index_type {
                ends[j]
            } else {
                self.block
                    .append_operation(
                        OperationBuilder::new("arith.index_cast", self.location)
                            .add_operands(&[ends[j]])
                            .add_results(&[index_type])
                            .build()
                            .expect("arith.index_cast end"),
                    )
                    .result(0)
                    .unwrap()
                    .into()
            };
            dyn_offsets.push(start_idx);

            // size = (end - start) / step
            let diff: melior::ir::Value<'c, 'c> = self
                .block
                .append_operation(
                    OperationBuilder::new("arith.subi", self.location)
                        .add_operands(&[end_idx, start_idx])
                        .add_results(&[index_type])
                        .build()
                        .expect("arith.subi"),
                )
                .result(0)
                .unwrap()
                .into();

            let size = if steps[j] == 1 {
                diff
            } else {
                let step_attr = Attribute::parse(self.context, &format!("{} : index", steps[j]))
                    .expect("step attr");
                let step_val: melior::ir::Value<'c, 'c> = self
                    .block
                    .append_operation(
                        OperationBuilder::new("arith.constant", self.location)
                            .add_results(&[index_type])
                            .add_attributes(&[(Identifier::new(self.context, "value"), step_attr)])
                            .build()
                            .expect("arith.constant step"),
                    )
                    .result(0)
                    .unwrap()
                    .into();
                self.block
                    .append_operation(
                        OperationBuilder::new("arith.divui", self.location)
                            .add_operands(&[diff, step_val])
                            .add_results(&[index_type])
                            .build()
                            .expect("arith.divui"),
                    )
                    .result(0)
                    .unwrap()
                    .into()
            };
            dyn_sizes_per_axis.push(size);
        }

        // Collect all dynamic sizes in dimension order (sliced axes + non-sliced dynamic axes).
        let mut all_dyn_sizes: Vec<melior::ir::Value<'c, 'c>> = Vec::new();
        {
            let mut sliced_j = 0usize;
            for (i, &dim) in in_shape[..rank].iter().enumerate() {
                if norm_axes.contains(&i) {
                    all_dyn_sizes.push(dyn_sizes_per_axis[sliced_j]);
                    sliced_j += 1;
                } else if dim.is_none() {
                    let sz = self.emit_tensor_dim(input.value(), i);
                    all_dyn_sizes.push(sz);
                }
            }
        }

        let n_dyn_offsets = dyn_offsets.len() as i32;
        let n_dyn_sizes = all_dyn_sizes.len() as i32;

        let make_i64_array = |ctx: &'c Context, vals: &[i64]| -> Attribute<'c> {
            let s = vals
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            Attribute::parse(ctx, &format!("array<i64: {s}>")).expect("i64 array attr")
        };

        let seg_attr = Attribute::parse(
            self.context,
            &format!("array<i32: 1, {n_dyn_offsets}, {n_dyn_sizes}, 0>"),
        )
        .expect("operandSegmentSizes");

        let mut operands = vec![input.value()];
        operands.extend(dyn_offsets);
        operands.extend(all_dyn_sizes);

        let ctx = self.context;
        let result: melior::ir::Value<'c, 'c> = self
            .block
            .append_operation(
                OperationBuilder::new("tensor.extract_slice", self.location)
                    .add_operands(&operands)
                    .add_results(&[out_type])
                    .add_attributes(&[
                        (
                            Identifier::new(ctx, "static_offsets"),
                            make_i64_array(ctx, &static_offsets),
                        ),
                        (
                            Identifier::new(ctx, "static_sizes"),
                            make_i64_array(ctx, &static_sizes),
                        ),
                        (
                            Identifier::new(ctx, "static_strides"),
                            make_i64_array(ctx, &static_strides),
                        ),
                        (Identifier::new(ctx, "operandSegmentSizes"), seg_attr),
                    ])
                    .build()
                    .expect("tensor.extract_slice dynamic"),
            )
            .result(0)
            .unwrap()
            .into();
        Tensor::from_value(result)
    }

    /// Gather elements from `data` at positions given by `indices` along `axis`.
    ///
    /// ONNX semantics (axis=0):
    ///   output[i, j, ...] = data[indices[i, j, ...], :]
    ///
    /// Output shape: `data.shape[0..axis] + indices.shape + data.shape[axis+1..]`
    ///
    /// Uses `linalg.generic` with `linalg.index` for iteration and `tensor.extract`
    /// to read from `data`.
    ///
    /// Restrictions: `indices` dtype must be I32 or I64.
    pub fn emit_gather(
        &mut self,
        data: &Tensor<'c>,
        indices: &Tensor<'c>,
        axis: usize,
    ) -> Tensor<'c> {
        let data_shape = data.shape();
        let idx_shape = indices.shape();
        let dtype = data.dtype();
        let data_rank = data_shape.len();
        let idx_rank = idx_shape.len();

        // Output shape: data[0..axis] + indices.shape + data[axis+1..]
        let mut out_shape: Vec<Option<u64>> = Vec::new();
        out_shape.extend_from_slice(&data_shape[..axis]);
        out_shape.extend_from_slice(&idx_shape);
        out_shape.extend_from_slice(&data_shape[axis + 1..]);
        let out_rank = out_shape.len();

        let elem_type = dtype.to_mlir_type(self.context);
        let idx_dtype = indices.dtype();
        let idx_elem_type = idx_dtype.to_mlir_type(self.context);

        // Build the output tensor.empty with correct dynamic dim sources.
        let out_type = self.make_tensor_type(&out_shape, dtype);
        let mut dyn_vals: Vec<melior::ir::Value<'c, 'c>> = Vec::new();
        for (i, dim) in out_shape.iter().enumerate() {
            if dim.is_none() {
                if i < axis {
                    dyn_vals.push(self.emit_tensor_dim(data.value(), i));
                } else if i < axis + idx_rank {
                    dyn_vals.push(self.emit_tensor_dim(indices.value(), i - axis));
                } else {
                    dyn_vals.push(self.emit_tensor_dim(data.value(), i - idx_rank + 1));
                }
            }
        }
        let init_proper: melior::ir::Value<'c, 'c> = self
            .block
            .append_operation(
                OperationBuilder::new("tensor.empty", self.location)
                    .add_operands(&dyn_vals)
                    .add_results(&[out_type])
                    .build()
                    .expect("tensor.empty gather"),
            )
            .result(0)
            .unwrap()
            .into();
        // Build indexing maps.
        // ins: (data, indices), outs: (result)
        // data: d0..d_{axis-1}, linalg.index(axis+idx_rank...) for post-axis data dims,
        //       and the axis dim comes from indices tensor.
        // indices: d_{axis}..d_{axis+idx_rank-1}
        // out: identity d0..d_{out_rank-1}
        //
        // This is complex for a general linalg.generic approach.
        // Use a simpler linalg.generic with linalg.index to get iteration coordinates,
        // then tensor.extract to read from data and indices.
        //
        // Iterator types: all parallel.
        let iterator_types = self.make_iterator_types(out_rank);

        // Indexing maps:
        // - indices: maps out dims [axis..axis+idx_rank] → indices dims [0..idx_rank]
        // - data: maps out dims [0..axis, axis+idx_rank..out_rank] → data dims [0..axis, axis+1..data_rank]
        //   BUT the axis data dim is NOT in the map — we read it via tensor.extract with the index.
        // - out: identity
        //
        // This is a read-only gather, so we can't express the data access pattern as a simple
        // affine map (it's data-dependent). Use linalg.generic with ins=(indices), outs=(result)
        // and read from data using tensor.extract inside the body.
        //
        // Body for each output element at iteration [i0, ..., i_{out_rank-1}]:
        //   - Get index from indices[i_{axis}..i_{axis+idx_rank-1}] via tensor.extract
        //   - Build data indices: [i0..i_{axis-1}, idx, i_{axis+idx_rank}..i_{out_rank-1}]
        //   - Read data element via tensor.extract
        //   - Yield that element

        let dim_vars: Vec<String> = (0..out_rank).map(|i| format!("d{i}")).collect();
        let dim_list = dim_vars.join(", ");

        // indices map: maps output dims [axis..axis+idx_rank] to indices dims [0..idx_rank].
        let idx_dims: Vec<String> = (axis..axis + idx_rank).map(|i| format!("d{i}")).collect();
        let idx_map = format!("affine_map<({dim_list}) -> ({})>", idx_dims.join(", "));

        // out map: identity.
        let out_map = format!("affine_map<({dim_list}) -> ({dim_list})>");

        let indexing_maps = Attribute::parse(self.context, &format!("[{idx_map}, {out_map}]"))
            .expect("gather indexing_maps");

        // Body block: (indices_elem, out_elem)
        let body_block = Block::new(&[
            (idx_elem_type, self.location), // indices element
            (elem_type, self.location),     // out element (unused, destination style)
        ]);

        let idx_elem: melior::ir::Value = body_block.argument(0).unwrap().into();

        // Cast index to index type for tensor.extract.
        let index_type = melior::ir::Type::parse(self.context, "index").expect("index type");
        let gather_idx: melior::ir::Value = body_block
            .append_operation(
                OperationBuilder::new("arith.index_cast", self.location)
                    .add_operands(&[idx_elem])
                    .add_results(&[index_type])
                    .build()
                    .expect("arith.index_cast gather"),
            )
            .result(0)
            .unwrap()
            .into();

        // Emit linalg.index for each output dim to get iteration indices.
        // linalg.index %c : index  where c is an attribute.
        let mut iter_indices: Vec<melior::ir::Value<'c, 'c>> = Vec::with_capacity(out_rank);
        for i in 0..out_rank {
            let idx_attr = Attribute::parse(self.context, &format!("{i} : i64"))
                .expect("linalg.index dim attr");
            let iter_idx: melior::ir::Value = body_block
                .append_operation(
                    OperationBuilder::new("linalg.index", self.location)
                        .add_results(&[index_type])
                        .add_attributes(&[(Identifier::new(self.context, "dim"), idx_attr)])
                        .build()
                        .expect("linalg.index"),
                )
                .result(0)
                .unwrap()
                .into();
            iter_indices.push(iter_idx);
        }

        // Build data access indices: pre-axis dims + gather_idx + post-axis dims.
        let mut data_indices: Vec<melior::ir::Value<'c, 'c>> = Vec::with_capacity(data_rank);
        // pre-axis dims: iter_indices[0..axis]
        data_indices.extend_from_slice(&iter_indices[0..axis]);
        // axis dim: gather_idx
        data_indices.push(gather_idx);
        // post-axis dims: iter_indices[axis+idx_rank..out_rank]
        data_indices.extend_from_slice(&iter_indices[axis + idx_rank..out_rank]);

        // tensor.extract %data[data_indices] : tensor<...>
        let mut extract_operands = vec![data.value()];
        extract_operands.extend(data_indices);
        let extracted: melior::ir::Value = body_block
            .append_operation(
                OperationBuilder::new("tensor.extract", self.location)
                    .add_operands(&extract_operands)
                    .add_results(&[elem_type])
                    .build()
                    .expect("tensor.extract data"),
            )
            .result(0)
            .unwrap()
            .into();

        body_block.append_operation(
            OperationBuilder::new("linalg.yield", self.location)
                .add_operands(&[extracted])
                .build()
                .expect("linalg.yield gather"),
        );
        let body_region = Region::new();
        body_region.append_block(body_block);

        let result: melior::ir::Value<'c, 'c> = self
            .block
            .append_operation(
                OperationBuilder::new("linalg.generic", self.location)
                    .add_operands(&[indices.value(), init_proper])
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
                    .expect("linalg.generic gather"),
            )
            .result(0)
            .unwrap()
            .into();
        Tensor::from_value(result)
    }
    /// Insert size-1 dimensions at each position in `axes` via `tensor.expand_shape`.
    ///
    /// `axes` are the output positions where new dims should be inserted (ONNX-style).
    pub fn emit_unsqueeze(&mut self, input: &Tensor<'c>, axes: &[i64]) -> Tensor<'c> {
        let in_shape = input.shape();
        let src_rank = in_shape.len();
        let tgt_rank = src_rank + axes.len();

        // Normalize axes to output positions.
        let mut norm_axes: Vec<usize> = axes
            .iter()
            .map(|&a| {
                let a = if a < 0 { a + tgt_rank as i64 } else { a };
                a as usize
            })
            .collect();
        norm_axes.sort_unstable();

        // Build target shape: insert Some(1) at each axis position.
        let mut out_shape: Vec<Option<u64>> = Vec::with_capacity(tgt_rank);
        let mut src_dim = 0usize;
        for tgt_i in 0..tgt_rank {
            if norm_axes.contains(&tgt_i) {
                out_shape.push(Some(1));
            } else {
                out_shape.push(in_shape[src_dim]);
                src_dim += 1;
            }
        }

        // Build reassociation: each source dim maps to one or more consecutive target dims.
        // Source dim k maps to the (non-squeezed) target dims that came from it, plus any
        // size-1 dims immediately preceding it.
        // Build src_to_tgt_dims: for each source dim, which target dims does it expand into?
        let mut src_to_tgt: Vec<Vec<usize>> = vec![Vec::new(); src_rank.max(1)];
        let mut src_idx = 0usize;
        let mut pending: Vec<usize> = Vec::new();
        for tgt_i in 0..tgt_rank {
            if norm_axes.contains(&tgt_i) {
                pending.push(tgt_i);
            } else {
                for p in pending.drain(..) {
                    if src_rank == 0 {
                        break;
                    }
                    src_to_tgt[src_idx].push(p);
                }
                if src_rank > 0 {
                    src_to_tgt[src_idx].push(tgt_i);
                    src_idx += 1;
                }
            }
        }
        // Trailing size-1 dims go to the last source dim.
        if !pending.is_empty() && src_rank > 0 {
            for p in pending.drain(..) {
                src_to_tgt[src_rank - 1].push(p);
            }
        }

        // Edge case: 0D input being unsqueezed to ND.
        // tensor.expand_shape requires exactly src_rank groups. For 0D: 0 groups = "[]".
        if src_rank == 0 {
            let reassoc_str = "[]";
            let dtype = input.dtype();
            return self.emit_expand_shape_impl(input.value(), &out_shape, dtype, reassoc_str);
        }

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
        let dtype = input.dtype();
        self.emit_expand_shape_impl(input.value(), &out_shape, dtype, &reassoc_str)
    }

    /// Remove size-1 dimensions at positions `axes` via `tensor.collapse_shape`.
    ///
    /// `axes` are the input positions to remove (must all be size-1 dims).
    pub fn emit_squeeze(&mut self, input: &Tensor<'c>, axes: &[i64]) -> Tensor<'c> {
        let in_shape = input.shape();
        let src_rank = in_shape.len();

        // Normalize axes.
        let mut norm_axes: Vec<usize> = axes
            .iter()
            .map(|&a| {
                let a = if a < 0 { a + src_rank as i64 } else { a };
                a as usize
            })
            .collect();
        norm_axes.sort_unstable();

        // Build output shape: remove the squeezed dims.
        let out_shape: Vec<Option<u64>> = in_shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !norm_axes.contains(i))
            .map(|(_, d)| *d)
            .collect();

        let tgt_rank = out_shape.len();

        // Build reassociation for collapse_shape:
        // Each target dim corresponds to one or more consecutive source dims
        // (the kept dim, plus any adjacent squeezed dims).
        // Group rule: each "kept" source dim takes any immediately preceding squeezed dims.
        let mut tgt_to_src: Vec<Vec<usize>> = vec![Vec::new(); tgt_rank.max(1)];
        let mut tgt_idx = 0usize;
        let mut pending: Vec<usize> = Vec::new();
        for src_i in 0..src_rank {
            if norm_axes.contains(&src_i) {
                pending.push(src_i);
            } else {
                for p in pending.drain(..) {
                    if tgt_rank > 0 {
                        tgt_to_src[tgt_idx].push(p);
                    }
                }
                if tgt_rank > 0 {
                    tgt_to_src[tgt_idx].push(src_i);
                    tgt_idx += 1;
                }
            }
        }
        // Trailing squeezed dims go to last target dim.
        if !pending.is_empty() && tgt_rank > 0 {
            for p in pending.drain(..) {
                tgt_to_src[tgt_rank - 1].push(p);
            }
        }

        // Edge case: squeezing to 0D.
        if tgt_rank == 0 {
            // All dims squeezed — result is a scalar (0D tensor).
            let reassoc_str = "[]";
            return self.emit_collapse_shape_with_reassoc(input.value(), &out_shape, reassoc_str);
        }

        let groups: Vec<String> = tgt_to_src
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
        self.emit_collapse_shape_with_reassoc(input.value(), &out_shape, &reassoc_str)
    }

    /// Flatten `input` into a 2D tensor by collapsing dims `[0..axis)` into dim 0
    /// and dims `[axis..rank)` into dim 1.
    ///
    /// Special cases:
    /// - `axis == 0`: result shape is `[1, total_elements]`
    /// - `axis == rank`: result shape is `[total_elements, 1]`
    pub fn emit_flatten(&mut self, input: &Tensor<'c>, axis: usize) -> Tensor<'c> {
        let in_shape = input.shape();
        let rank = in_shape.len();
        assert!(
            axis <= rank,
            "emit_flatten: axis {axis} out of bounds for rank {rank}"
        );

        // Compute static dims for the two groups if possible.
        let head_size: Option<u64> = if axis == 0 {
            Some(1)
        } else {
            in_shape[..axis]
                .iter()
                .try_fold(1u64, |acc, d| d.map(|v| acc * v))
        };
        let tail_size: Option<u64> = if axis == rank {
            Some(1)
        } else {
            in_shape[axis..]
                .iter()
                .try_fold(1u64, |acc, d| d.map(|v| acc * v))
        };
        let out_shape = vec![head_size, tail_size];

        // Build reassociation: [[0..axis-1], [axis..rank-1]]
        // If axis == 0: head group is empty (not valid) — use [[0], [0..rank-1]] workaround.
        // tensor.collapse_shape doesn't allow empty groups, so use expand trick:
        // For axis==0: prepend a 1-dim then collapse.
        // Easier: emit reshape instead.
        if axis == 0 || axis == rank {
            // One group covers the entire input; use tensor.reshape.
            let target: Vec<i64> = if axis == 0 { vec![1, -1] } else { vec![-1, 1] };
            return self.emit_reshape(input, &target);
        }

        // General case: two non-empty groups.
        let head_indices: Vec<String> = (0..axis).map(|i| i.to_string()).collect();
        let tail_indices: Vec<String> = (axis..rank).map(|i| i.to_string()).collect();
        let reassoc_str = format!(
            "[[{}], [{}]]",
            head_indices.join(", "),
            tail_indices.join(", ")
        );

        self.emit_collapse_shape_with_reassoc(input.value(), &out_shape, &reassoc_str)
    }

    /// Split `input` into multiple tensors along `axis` with sizes given by `split_sizes`.
    ///
    /// `split_sizes` must sum to the axis dimension of `input`.
    /// Returns one `Tensor` per entry in `split_sizes`.
    pub fn emit_split(
        &mut self,
        input: &Tensor<'c>,
        axis: usize,
        split_sizes: &[u64],
    ) -> Vec<Tensor<'c>> {
        let in_shape = input.shape();
        let rank = in_shape.len();
        let dtype = input.dtype();
        assert!(
            axis < rank,
            "emit_split: axis {axis} out of bounds for rank {rank}"
        );
        assert!(
            !split_sizes.is_empty(),
            "emit_split: split_sizes must not be empty"
        );
        const K_DYNAMIC: i64 = i64::MIN;

        let mut results: Vec<Tensor<'c>> = Vec::with_capacity(split_sizes.len());
        let mut offset: i64 = 0;

        for &size in split_sizes {
            // Out shape: same as input except axis dim = size.
            let out_shape: Vec<Option<u64>> = in_shape
                .iter()
                .enumerate()
                .map(|(i, d)| if i == axis { Some(size) } else { *d })
                .collect();
            let out_type = self.make_tensor_type(&out_shape, dtype);

            // static_offsets: 0 for all except axis.
            let static_offsets: Vec<i64> = (0..rank)
                .map(|i| if i == axis { offset } else { 0 })
                .collect();

            // static_sizes: the split size for axis, input dims for others.
            let static_sizes: Vec<i64> = (0..rank)
                .map(|i| {
                    if i == axis {
                        size as i64
                    } else {
                        match in_shape[i] {
                            Some(n) => n as i64,
                            None => K_DYNAMIC,
                        }
                    }
                })
                .collect();

            let static_strides: Vec<i64> = vec![1; rank];

            let dyn_sizes: Vec<melior::ir::Value<'c, 'c>> = static_sizes
                .iter()
                .enumerate()
                .filter(|&(_, s)| *s == K_DYNAMIC)
                .map(|(i, _)| self.emit_tensor_dim(input.value(), i))
                .collect();

            let make_attr = |ctx: &'c Context, vals: &[i64]| -> Attribute<'c> {
                let s = vals
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                Attribute::parse(ctx, &format!("array<i64: {s}>")).expect("split slice attr")
            };

            let dyn_sizes_len = dyn_sizes.len() as i32;
            let mut operands = vec![input.value()];
            operands.extend(dyn_sizes);

            let seg_attr = Attribute::parse(
                self.context,
                &format!("array<i32: 1, 0, {dyn_sizes_len}, 0>"),
            )
            .expect("operandSegmentSizes split");

            let slice: melior::ir::Value<'c, 'c> = self
                .block
                .append_operation(
                    OperationBuilder::new("tensor.extract_slice", self.location)
                        .add_operands(&operands)
                        .add_results(&[out_type])
                        .add_attributes(&[
                            (
                                Identifier::new(self.context, "static_offsets"),
                                make_attr(self.context, &static_offsets),
                            ),
                            (
                                Identifier::new(self.context, "static_sizes"),
                                make_attr(self.context, &static_sizes),
                            ),
                            (
                                Identifier::new(self.context, "static_strides"),
                                make_attr(self.context, &static_strides),
                            ),
                            (
                                Identifier::new(self.context, "operandSegmentSizes"),
                                seg_attr,
                            ),
                        ])
                        .build()
                        .expect("tensor.extract_slice split"),
                )
                .result(0)
                .unwrap()
                .into();

            results.push(Tensor::from_value(slice));
            offset += size as i64;
        }

        results
    }

    /// Expand a 1D tensor `[K]` to 2D `[1, K]` or `[K, 1]` via `tensor.expand_shape`.
    /// Reassociation: `[[0, 1]]` — both target dims come from source dim 0.
    pub(super) fn emit_expand_shape_1d_to_2d(
        &mut self,
        input: melior::ir::Value<'c, 'c>,
        _src_shape: &[Option<u64>],
        tgt_shape: &[Option<u64>],
    ) -> Tensor<'c> {
        debug_assert_eq!(_src_shape.len(), 1);
        debug_assert_eq!(tgt_shape.len(), 2);

        let elem_type = {
            let rtt = RankedTensorType::try_from(input.r#type())
                .expect("expand_shape input must be RankedTensorType");
            rtt.element()
        };
        let dtype = mlir_element_type_to_dtype(elem_type);
        // Reassociation: [[0, 1]] — both target dims come from source dim 0.
        self.emit_expand_shape_impl(input, tgt_shape, dtype, "[[0, 1]]")
    }

    /// Collapse a 2D tensor to 1D via `tensor.collapse_shape`.
    /// Reassociation: `[[0, 1]]` — source dims 0,1 collapse into target dim 0.
    /// Collapse leading dimensions of a tensor: [d0, d1, ..., dN] → [d0*d1, ..., dN].
    /// Used to flatten 3D [B, S, N] → 2D [M, N] for matmul.
    pub fn emit_flatten_leading(&mut self, input: &Tensor<'c>) -> Tensor<'c> {
        let shape = input.shape();
        assert!(shape.len() >= 2, "emit_flatten_leading requires rank >= 2");
        // Merge dims 0..=(rank-2) into one, keep last dim separate.
        let last = *shape.last().unwrap();
        // The merged dim is dynamic if any of the leading dims is dynamic.
        let merged: Option<u64> = shape[..shape.len() - 1]
            .iter()
            .try_fold(1u64, |acc, d| d.map(|v| acc * v));
        let tgt_shape = vec![merged, last];
        // Reassociation: [[0, 1, ...rank-2], [rank-1]]
        let leading: Vec<String> = (0..shape.len() - 1).map(|i| i.to_string()).collect();
        let reassoc = format!("[[{}], [{}]]", leading.join(", "), shape.len() - 1);
        self.emit_collapse_shape_with_reassoc(input.value(), &tgt_shape, &reassoc)
    }

    pub(super) fn emit_collapse_shape_2d_to_1d(
        &mut self,
        input: melior::ir::Value<'c, 'c>,
        _src_shape: &[Option<u64>],
        tgt_shape: &[Option<u64>],
    ) -> Tensor<'c> {
        debug_assert_eq!(_src_shape.len(), 2);
        debug_assert_eq!(tgt_shape.len(), 1);
        self.emit_collapse_shape_with_reassoc(input, tgt_shape, "[[0, 1]]")
    }

    /// Collapse an ND tensor to 3D by merging all leading dims into one batch dim.
    /// E.g. `[A, B, M, K]` -> `[A*B, M, K]` with reassociation `[[0, 1], [2], [3]]`.
    pub(super) fn emit_collapse_shape_nd_to_3d(
        &mut self,
        input: melior::ir::Value<'c, 'c>,
        src_shape: &[Option<u64>],
        tgt_shape: &[Option<u64>], // [B_flat, M, K]
    ) -> Tensor<'c> {
        let src_rank = src_shape.len();
        debug_assert!(src_rank >= 3);
        debug_assert_eq!(tgt_shape.len(), 3);

        // tensor.collapse_shape supports multiple dynamic dims per group —
        // MLIR computes the collapsed dim as the product at runtime.
        let batch_indices: Vec<String> = (0..src_rank - 2).map(|i| i.to_string()).collect();
        let batch_group = format!("[{}]", batch_indices.join(", "));
        let m_group = format!("[{}]", src_rank - 2);
        let k_group = format!("[{}]", src_rank - 1);
        let reassoc_str = format!("[{batch_group}, {m_group}, {k_group}]");
        self.emit_collapse_shape_with_reassoc(input, tgt_shape, &reassoc_str)
    }

    /// Expand a 3D `[B_flat, M, N]` result back to ND `[..., M, N]`.
    /// Reassociation: first group expands the batch dim, then identity for M and N.
    /// Expand a 3D `[B_flat, M, N]` result back to ND `[..., M, N]`.
    /// Reassociation: first group expands the batch dim, then identity for M and N.
    ///
    /// `batch_ref` is an optional (value, shape) of the original ND tensor whose
    /// batch dims should be restored. Used to get dynamic batch dim values via
    /// `tensor.dim` when the batch group has multiple dynamic dims.
    pub(super) fn emit_expand_shape_3d_to_nd(
        &mut self,
        input: melior::ir::Value<'c, 'c>,
        src_shape: &[Option<u64>], // [B_flat, M, N]
        tgt_shape: &[Option<u64>], // [..., M, N]
        batch_ref: Option<(melior::ir::Value<'c, 'c>, &[Option<u64>])>,
    ) -> Tensor<'c> {
        let tgt_rank = tgt_shape.len();
        debug_assert_eq!(src_shape.len(), 3);
        debug_assert!(tgt_rank >= 3);

        let batch_indices: Vec<String> = (0..tgt_rank - 2).map(|i| i.to_string()).collect();
        let batch_group = format!("[{}]", batch_indices.join(", "));
        let m_group = format!("[{}]", tgt_rank - 2);
        let n_group = format!("[{}]", tgt_rank - 1);
        let reassoc_str = format!("[{batch_group}, {m_group}, {n_group}]");

        // Collect dynamic dim values for the output_shape operands.
        let mut dyn_vals: Vec<melior::ir::Value<'c, 'c>> = Vec::new();
        for (i, dim) in tgt_shape.iter().enumerate() {
            if dim.is_none() {
                // Get the dynamic dim value from the batch reference tensor
                // if available, otherwise from the input.
                let val = if let Some((ref_val, _)) = batch_ref {
                    if i < tgt_rank - 2 {
                        // Batch dim — get from the original ND tensor.
                        self.emit_tensor_dim(ref_val, i)
                    } else {
                        // M or N dim — get from the 3D input.
                        self.emit_tensor_dim(input, i - (tgt_rank - 3))
                    }
                } else {
                    let in_dim = self.find_input_dim_for_expand_output(&reassoc_str, i);
                    self.emit_tensor_dim(input, in_dim)
                };
                dyn_vals.push(val);
            }
        }

        let elem_type = {
            let rtt = RankedTensorType::try_from(input.r#type())
                .expect("expand_shape input must be RankedTensorType");
            rtt.element()
        };
        let dtype = mlir_element_type_to_dtype(elem_type);
        self.emit_expand_shape_impl_with_dyn_vals(input, tgt_shape, dtype, &reassoc_str, &dyn_vals)
    }

    pub(super) fn emit_collapse_shape_with_reassoc(
        &mut self,
        input: melior::ir::Value<'c, 'c>,
        tgt_shape: &[Option<u64>],
        reassoc_str: &str,
    ) -> Tensor<'c> {
        let elem_type = {
            let rtt = RankedTensorType::try_from(input.r#type())
                .expect("collapse_shape input must be RankedTensorType");
            rtt.element()
        };
        let dtype = mlir_element_type_to_dtype(elem_type);
        let out_type = self.make_tensor_type(tgt_shape, dtype);

        let reassoc_attr =
            Attribute::parse(self.context, reassoc_str).expect("collapse_shape reassociation");

        let result: melior::ir::Value = self
            .block
            .append_operation(
                OperationBuilder::new("tensor.collapse_shape", self.location)
                    .add_operands(&[input])
                    .add_results(&[out_type])
                    .add_attributes(&[(
                        Identifier::new(self.context, "reassociation"),
                        reassoc_attr,
                    )])
                    .build()
                    .expect("tensor.collapse_shape"),
            )
            .result(0)
            .unwrap()
            .into();
        Tensor::from_value(result)
    }

    fn emit_expand_shape_with_reassoc(
        &mut self,
        input: melior::ir::Value<'c, 'c>,
        tgt_shape: &[Option<u64>],
        reassoc_str: &str,
    ) -> Tensor<'c> {
        let elem_type = {
            let rtt = RankedTensorType::try_from(input.r#type())
                .expect("expand_shape input must be RankedTensorType");
            rtt.element()
        };
        let dtype = mlir_element_type_to_dtype(elem_type);
        self.emit_expand_shape_impl(input, tgt_shape, dtype, reassoc_str)
    }

    /// Core `tensor.expand_shape` emission with `static_output_shape` attribute.
    pub(super) fn emit_expand_shape_impl(
        &mut self,
        input: melior::ir::Value<'c, 'c>,
        tgt_shape: &[Option<u64>],
        dtype: DType,
        reassoc_str: &str,
    ) -> Tensor<'c> {
        let out_type = self.make_tensor_type(tgt_shape, dtype);

        let reassoc_attr =
            Attribute::parse(self.context, reassoc_str).expect("expand_shape reassociation");

        // static_output_shape: kDynamic sentinel = i64::MIN for dynamic dims.
        let static_shape_vals: Vec<i64> = tgt_shape
            .iter()
            .map(|d| match d {
                Some(n) => *n as i64,
                None => i64::MIN,
            })
            .collect();
        let static_shape_str = static_shape_vals
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let static_output_shape_attr =
            Attribute::parse(self.context, &format!("array<i64: {static_shape_str}>"))
                .expect("static_output_shape attr");

        // For each dynamic dim in tgt_shape, emit tensor.dim on the input
        // to get the runtime size. For expand_shape, a dynamic output dim
        // inherits its runtime size from the input dim in its reassociation group.
        // We parse the reassoc to find which input dim each output dim belongs to.
        let mut dyn_output_shape_vals: Vec<melior::ir::Value<'c, 'c>> = Vec::new();
        for (out_i, dim) in tgt_shape.iter().enumerate() {
            if dim.is_none() {
                // Find which input dim this output dim belongs to by checking
                // the reassociation. For expand_shape [[0,1],[2]], output dim 0
                // and 1 come from input dim 0, output dim 2 comes from input dim 1.
                // We need tensor.dim on the input at the corresponding input dim index.
                // Since this is expand_shape, each group has exactly one input dim.
                // The output dim's runtime value comes from the input dim if it's
                // the only dynamic dim in the group, or needs computation otherwise.
                // For simplicity, use tensor.dim on the input for the group's input dim.
                let in_dim = self.find_input_dim_for_expand_output(reassoc_str, out_i);
                let dim_val = self.emit_tensor_dim(input, in_dim);
                dyn_output_shape_vals.push(dim_val);
            }
        }

        let mut operands = vec![input];
        operands.extend(dyn_output_shape_vals);

        let result: melior::ir::Value = self
            .block
            .append_operation(
                OperationBuilder::new("tensor.expand_shape", self.location)
                    .add_operands(&operands)
                    .add_results(&[out_type])
                    .add_attributes(&[
                        (Identifier::new(self.context, "reassociation"), reassoc_attr),
                        (
                            Identifier::new(self.context, "static_output_shape"),
                            static_output_shape_attr,
                        ),
                    ])
                    .build()
                    .expect("tensor.expand_shape"),
            )
            .result(0)
            .unwrap()
            .into();
        Tensor::from_value(result)
    }

    /// Like `emit_expand_shape_impl` but uses pre-computed dynamic dim values
    /// instead of `tensor.dim`. Used when dynamic dims come from shape tensor
    /// extraction (e.g., ONNX Reshape's -1 inference) rather than direct
    /// inheritance from the input tensor.
    fn emit_expand_shape_impl_with_dyn_vals(
        &mut self,
        input: melior::ir::Value<'c, 'c>,
        tgt_shape: &[Option<u64>],
        dtype: DType,
        reassoc_str: &str,
        dyn_vals: &[melior::ir::Value<'c, 'c>],
    ) -> Tensor<'c> {
        let out_type = self.make_tensor_type(tgt_shape, dtype);

        let reassoc_attr =
            Attribute::parse(self.context, reassoc_str).expect("expand_shape reassociation");

        let static_shape_vals: Vec<i64> = tgt_shape
            .iter()
            .map(|d| match d {
                Some(n) => *n as i64,
                None => i64::MIN,
            })
            .collect();
        let static_shape_str = static_shape_vals
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let static_output_shape_attr =
            Attribute::parse(self.context, &format!("array<i64: {static_shape_str}>"))
                .expect("static_output_shape attr");

        let mut operands = vec![input];
        operands.extend_from_slice(dyn_vals);

        let result: melior::ir::Value = self
            .block
            .append_operation(
                OperationBuilder::new("tensor.expand_shape", self.location)
                    .add_operands(&operands)
                    .add_results(&[out_type])
                    .add_attributes(&[
                        (Identifier::new(self.context, "reassociation"), reassoc_attr),
                        (
                            Identifier::new(self.context, "static_output_shape"),
                            static_output_shape_attr,
                        ),
                    ])
                    .build()
                    .expect("tensor.expand_shape"),
            )
            .result(0)
            .unwrap()
            .into();
        Tensor::from_value(result)
    }

    /// Given a reassociation string like "[[0, 1], [2, 3]]" and an output dim
    /// index, find which input dim (group index) it belongs to.
    fn find_input_dim_for_expand_output(&self, reassoc_str: &str, out_dim: usize) -> usize {
        // Parse the reassociation. Each group is an array of output dim indices.
        // The group index IS the input dim index.
        // E.g. "[[0, 1], [2]]" means group 0 → output dims {0,1}, group 1 → output dim {2}.
        let trimmed = reassoc_str.trim();
        let inner = &trimmed[1..trimmed.len() - 1]; // strip outer []
        let mut depth = 0;
        let mut group_start = None;
        let mut group_idx = 0;
        for (i, ch) in inner.char_indices() {
            match ch {
                '[' => {
                    depth += 1;
                    if depth == 1 {
                        group_start = Some(i + 1);
                    }
                }
                ']' => {
                    depth -= 1;
                    if depth == 0
                        && let Some(start) = group_start
                    {
                        let group_content = &inner[start..i];
                        for num_str in group_content.split(',') {
                            let num: usize = num_str.trim().parse().unwrap();
                            if num == out_dim {
                                return group_idx;
                            }
                        }
                        group_idx += 1;
                    }
                }
                _ => {}
            }
        }
        panic!("output dim {out_dim} not found in reassociation {reassoc_str}");
    }

    // ── Spatial Ops (task 6) ──────────────────────────────────────────────────

    /// Emit `tensor.pad` on the input with static low/high padding values.
    ///
    /// `pad_low[i]` and `pad_high[i]` are the padding amounts for each
    /// dimension. Dimensions with zero padding are passed through unchanged.
    /// The region yields `pad_value`.
    pub(super) fn emit_tensor_pad(
        &mut self,
        input: melior::ir::Value<'c, 'c>,
        pad_low: &[i64],
        pad_high: &[i64],
        pad_value_attr: Attribute<'c>,
    ) -> melior::ir::Value<'c, 'c> {
        let rtt = RankedTensorType::try_from(input.r#type())
            .expect("tensor.pad: input must be RankedTensorType");
        let rank = rtt.rank();
        assert_eq!(pad_low.len(), rank);
        assert_eq!(pad_high.len(), rank);

        let elem_type = rtt.element();
        let dtype = mlir_element_type_to_dtype(elem_type);

        // Compute result shape.
        let in_shape: Vec<Option<u64>> = (0..rank)
            .map(|i| {
                let raw = unsafe {
                    mlir_sys::mlirShapedTypeGetDimSize(
                        melior::ir::TypeLike::to_raw(&rtt),
                        i as isize,
                    )
                };
                if raw < 0 { None } else { Some(raw as u64) }
            })
            .collect();
        let out_shape: Vec<Option<u64>> = in_shape
            .iter()
            .enumerate()
            .map(|(i, d)| d.map(|v| v + pad_low[i] as u64 + pad_high[i] as u64))
            .collect();

        let out_type = self.make_tensor_type(&out_shape, dtype);

        // static_low and static_high as DenseI64ArrayAttr.
        let low_str = pad_low
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let high_str = pad_high
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let static_low_attr = Attribute::parse(self.context, &format!("array<i64: {low_str}>"))
            .expect("static_low attr");
        let static_high_attr = Attribute::parse(self.context, &format!("array<i64: {high_str}>"))
            .expect("static_high attr");

        // Padding region: block args are indices (one per rank dim), yields the pad value.
        let index_type = melior::ir::Type::parse(self.context, "index").expect("index type");
        let pad_block_args: Vec<(melior::ir::Type, Location)> =
            (0..rank).map(|_| (index_type, self.location)).collect();
        let pad_block = Block::new(&pad_block_args);

        // Emit pad_value constant inside the region block.
        let pad_const: melior::ir::Value = pad_block
            .append_operation(
                OperationBuilder::new("arith.constant", self.location)
                    .add_results(&[elem_type])
                    .add_attributes(&[(Identifier::new(self.context, "value"), pad_value_attr)])
                    .build()
                    .expect("arith.constant pad_value"),
            )
            .result(0)
            .unwrap()
            .into();
        pad_block.append_operation(
            OperationBuilder::new("tensor.yield", self.location)
                .add_operands(&[pad_const])
                .build()
                .expect("tensor.yield pad"),
        );
        let pad_region = Region::new();
        pad_region.append_block(pad_block);

        self.block
            .append_operation(
                OperationBuilder::new("tensor.pad", self.location)
                    .add_operands(&[input])
                    .add_results(&[out_type])
                    .add_attributes(&[
                        (Identifier::new(self.context, "static_low"), static_low_attr),
                        (
                            Identifier::new(self.context, "static_high"),
                            static_high_attr,
                        ),
                        // AttrSizedOperandSegments: source(1), low_dynamic(0), high_dynamic(0).
                        (
                            Identifier::new(self.context, "operandSegmentSizes"),
                            Attribute::parse(self.context, "array<i32: 1, 0, 0>").unwrap(),
                        ),
                    ])
                    .add_regions([pad_region])
                    .build()
                    .expect("tensor.pad"),
            )
            .result(0)
            .unwrap()
            .into()
    }
}
