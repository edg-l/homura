use melior::dialect::DialectRegistry;
use melior::{
    Context,
    dialect::func,
    ir::{
        Attribute, Block, Identifier, Location, Module, Region, RegionLike,
        ShapedTypeLike, ValueLike,
        attribute::{StringAttribute, TypeAttribute},
        block::BlockLike,
        operation::{OperationBuilder, OperationLike, OperationMutLike},
        r#type::{FunctionType, MemRefType, RankedTensorType},
    },
    pass,
    utility::{
        parse_pass_pipeline, register_all_dialects, register_all_llvm_translations,
        register_all_passes,
    },
};

use crate::{
    DType,
    compiler::CompileError,
    runtime::{CompiledGraph, OutputDesc},
};

// ── Tensor wrapper ────────────────────────────────────────────────────────────

/// A tensor value in a graph being built. Wraps an MLIR `Value` — shape and
/// dtype are read from the MLIR type, not stored separately.
#[derive(Clone, Copy)]
pub struct Tensor<'c> {
    value: melior::ir::Value<'c, 'c>,
}

impl<'c> Tensor<'c> {
    pub(crate) fn from_value(value: melior::ir::Value<'c, 'c>) -> Self {
        Self { value }
    }

    /// The underlying MLIR SSA value.
    pub fn value(&self) -> melior::ir::Value<'c, 'c> {
        self.value
    }

    /// Read the shape from the MLIR `RankedTensorType`. `None` = dynamic dim.
    pub fn shape(&self) -> Vec<Option<u64>> {
        let rtt = self.ranked_tensor_type();
        let rank = rtt.rank();
        (0..rank)
            .map(|i| {
                let raw = unsafe {
                    mlir_sys::mlirShapedTypeGetDimSize(
                        melior::ir::TypeLike::to_raw(&rtt),
                        i as isize,
                    )
                };
                if raw < 0 { None } else { Some(raw as u64) }
            })
            .collect()
    }

    /// Read the dtype from the MLIR element type.
    pub fn dtype(&self) -> DType {
        let rtt = self.ranked_tensor_type();
        mlir_element_type_to_dtype(rtt.element())
    }

    /// Number of dimensions.
    pub fn rank(&self) -> usize {
        self.ranked_tensor_type().rank()
    }

    fn ranked_tensor_type(&self) -> RankedTensorType<'c> {
        let ty = self.value.r#type();
        RankedTensorType::try_from(ty)
            .expect("Tensor value must have RankedTensorType")
    }
}

/// Convert an MLIR element type back to our DType enum.
fn mlir_element_type_to_dtype(ty: melior::ir::Type) -> DType {
    let s = ty.to_string();
    match s.as_str() {
        "f32" => DType::F32,
        "f64" => DType::F64,
        "i32" => DType::I32,
        "i64" => DType::I64,
        other => panic!("unsupported MLIR element type: {other}"),
    }
}

// ── GraphBuilder ──────────────────────────────────────────────────────────────

/// Tracks one function argument.
struct ArgInfo {
    shape: Vec<Option<u64>>,
    dtype: DType,
    is_input: bool,
}

/// A completed sub-function ready to be emitted into the MLIR module.
struct CompletedSubFunction<'c> {
    name: String,
    block: Block<'c>,
    arg_types: Vec<melior::ir::Type<'c>>,
    return_types: Vec<melior::ir::Type<'c>>,
}

/// Stashed caller state while a sub-function is being built.
struct SubFunctionBuildState<'c> {
    name: String,
    caller_block: Block<'c>,
    caller_args: Vec<ArgInfo>,
    /// Caller-side tensor values to pass to func.call when finalized.
    caller_arg_values: Vec<melior::ir::Value<'c, 'c>>,
}

/// Handle to a sub-function being built. Returned by `begin_subfunction`.
pub struct SubFunctionHandle {
    pub _index: usize,
}

/// Owns the MLIR Context. Create one, then call `.builder()` to get a
/// `GraphBuilder` that borrows from it.
///
/// ```ignore
/// let ctx = GraphContext::new();
/// let mut gb = ctx.builder();
/// let x = gb.input(&[Some(4)], DType::F32);
/// let graph = gb.compile(&[&x])?;
/// ```
pub struct GraphContext {
    context: Context,
}

impl GraphContext {
    pub fn new() -> Self {
        Self { context: create_context() }
    }

    pub fn builder(&self) -> GraphBuilder<'_> {
        GraphBuilder::new(&self.context)
    }
}

/// Builds an MLIR computation graph by emitting ops directly into an MLIR
/// function body. Borrows the MLIR `Context` from a `GraphContext`.
pub struct GraphBuilder<'c> {
    context: &'c Context,
    block: Block<'c>,
    location: Location<'c>,
    args: Vec<ArgInfo>,
    /// Completed sub-functions to emit into the MLIR module.
    completed_subfunctions: Vec<CompletedSubFunction<'c>>,
    /// Stashed caller state when building a sub-function (no nesting).
    subfunction_build_state: Option<SubFunctionBuildState<'c>>,
}

impl<'c> GraphBuilder<'c> {
    /// Create a new builder borrowing the given context.
    pub fn new(context: &'c Context) -> Self {
        let location = Location::unknown(context);
        let block = Block::new(&[]);
        Self {
            context, block, location,
            args: Vec::new(),
            completed_subfunctions: Vec::new(),
            subfunction_build_state: None,
        }
    }

    /// Access the MLIR context.
    pub fn context(&self) -> &'c Context {
        self.context
    }

    /// Access the MLIR location.
    pub fn location(&self) -> Location<'c> {
        self.location
    }

    /// Access the function body block.
    pub fn block(&self) -> &Block<'c> {
        &self.block
    }

    /// Declare a dynamic/static input tensor.
    ///
    /// `shape`: `Some(n)` for static dims, `None` for dynamic (`?`) dims.
    pub fn input(&mut self, shape: &[Option<u64>], dtype: DType) -> Tensor<'c> {
        self.add_arg(shape, dtype, true)
    }

    /// Declare a weight tensor (large constant passed as function argument).
    pub fn add_weight(&mut self, shape: &[Option<u64>], dtype: DType) -> Tensor<'c> {
        self.add_arg(shape, dtype, false)
    }

    // ── Sub-function API ─────────────────────────────────────────────────────

    /// Start building a new sub-function. `args` are caller-side tensors whose
    /// types define the sub-function's parameters.
    ///
    /// Returns a handle and the sub-function's argument tensors (same types as
    /// `args` but in the sub-function's scope). All subsequent `emit_*` calls
    /// go into the sub-function until `end_subfunction` is called.
    pub fn begin_subfunction(
        &mut self,
        name: &str,
        args: &[&Tensor<'c>],
    ) -> (SubFunctionHandle, Vec<Tensor<'c>>) {
        assert!(
            self.subfunction_build_state.is_none(),
            "nested sub-functions not supported"
        );

        let handle = SubFunctionHandle {
            _index: self.completed_subfunctions.len(),
        };

        // New block with tensor-typed arguments matching caller tensors.
        let new_block = Block::new(&[]);
        let mut sub_arg_tensors = Vec::with_capacity(args.len());

        for &arg in args {
            let tensor_type = arg.value().r#type();
            new_block.add_argument(tensor_type, self.location);
            let idx = new_block.argument_count() - 1;
            sub_arg_tensors.push(Tensor::from_value(
                new_block.argument(idx).unwrap().into(),
            ));
        }

        // Stash caller state.
        let caller_arg_values: Vec<_> = args.iter().map(|t| t.value()).collect();
        let caller_block = std::mem::replace(&mut self.block, new_block);
        let caller_args = std::mem::take(&mut self.args);

        self.subfunction_build_state = Some(SubFunctionBuildState {
            name: name.to_string(),
            caller_block,
            caller_args,
            caller_arg_values,
        });

        (handle, sub_arg_tensors)
    }

    /// Dynamically add an argument to the sub-function being built.
    /// `caller_value` is the tensor in the caller's scope. Returns the
    /// corresponding tensor in the sub-function's scope.
    ///
    /// Use this to route weights directly to sub-functions without threading
    /// them through intermediate sub-functions.
    pub fn add_subfunction_arg(&mut self, caller_value: &Tensor<'c>) -> Tensor<'c> {
        let state = self.subfunction_build_state.as_mut().expect(
            "add_subfunction_arg called outside of a sub-function"
        );
        let tensor_type = caller_value.value().r#type();
        self.block.add_argument(tensor_type, self.location);
        let idx = self.block.argument_count() - 1;
        state.caller_arg_values.push(caller_value.value());
        Tensor::from_value(self.block.argument(idx).unwrap().into())
    }

    /// Returns true if currently building a sub-function.
    pub fn in_subfunction(&self) -> bool {
        self.subfunction_build_state.is_some()
    }

    /// Number of MLIR operations in the current block (approximate op count).
    pub fn block_op_count(&self) -> usize {
        // melior Block doesn't expose op count directly, but we can iterate.
        let mut count = 0;
        let mut maybe_op = self.block.first_operation();
        while let Some(op) = maybe_op {
            count += 1;
            maybe_op = op.next_in_block();
        }
        count
    }

    /// Finalize the sub-function: add `func.return`, store the completed
    /// sub-function, restore the caller block, emit `func.call`, and return
    /// the call results as tensors in the caller's scope.
    pub fn end_subfunction(
        &mut self,
        _handle: SubFunctionHandle,
        returns: &[&Tensor<'c>],
    ) -> Vec<Tensor<'c>> {
        let state = self.subfunction_build_state.take().expect(
            "end_subfunction called without matching begin_subfunction",
        );

        // func.return in the sub-function.
        let return_values: Vec<melior::ir::Value> =
            returns.iter().map(|t| t.value()).collect();
        self.block
            .append_operation(func::r#return(&return_values, self.location));

        // Collect types for the FunctionType.
        let arg_types: Vec<melior::ir::Type<'c>> = (0..self.block.argument_count())
            .map(|i| self.block.argument(i).unwrap().r#type())
            .collect();
        let return_types: Vec<melior::ir::Type<'c>> =
            returns.iter().map(|t| t.value().r#type()).collect();

        // Store completed sub-function and restore caller block.
        let sub_block = std::mem::replace(&mut self.block, state.caller_block);
        self.args = state.caller_args;

        let func_name = state.name.clone();
        self.completed_subfunctions.push(CompletedSubFunction {
            name: state.name,
            block: sub_block,
            arg_types,
            return_types: return_types.clone(),
        });

        // Emit func.call in the caller block.
        let callee_attr = Attribute::parse(
            self.context,
            &format!("@{func_name}"),
        )
        .expect("callee attr");

        let call_op = OperationBuilder::new("func.call", self.location)
            .add_operands(&state.caller_arg_values)
            .add_results(&return_types)
            .add_attributes(&[(
                Identifier::new(self.context, "callee"),
                callee_attr,
            )])
            .build()
            .expect("func.call");

        let call_ref = self.block.append_operation(call_op);

        (0..returns.len())
            .map(|i| Tensor::from_value(call_ref.result(i).unwrap().into()))
            .collect()
    }

    // ── Elementwise binary ops ────────────────────────────────────────────────

    /// Element-wise addition (F32/F64: arith.addf; I32/I64: arith.addi).
    pub fn emit_add(&mut self, lhs: &Tensor<'c>, rhs: &Tensor<'c>) -> Tensor<'c> {
        let op = match lhs.dtype() {
            DType::F32 | DType::F64 => "arith.addf",
            DType::I32 | DType::I64 => "arith.addi",
        };
        self.emit_linalg_binary(op, lhs, rhs)
    }

    /// Element-wise subtraction (F32/F64: arith.subf; I32/I64: arith.subi).
    pub fn emit_sub(&mut self, lhs: &Tensor<'c>, rhs: &Tensor<'c>) -> Tensor<'c> {
        let op = match lhs.dtype() {
            DType::F32 | DType::F64 => "arith.subf",
            DType::I32 | DType::I64 => "arith.subi",
        };
        self.emit_linalg_binary(op, lhs, rhs)
    }

    /// Element-wise multiplication (F32/F64: arith.mulf; I32/I64: arith.muli).
    pub fn emit_mul(&mut self, lhs: &Tensor<'c>, rhs: &Tensor<'c>) -> Tensor<'c> {
        let op = match lhs.dtype() {
            DType::F32 | DType::F64 => "arith.mulf",
            DType::I32 | DType::I64 => "arith.muli",
        };
        self.emit_linalg_binary(op, lhs, rhs)
    }

    /// Element-wise division (F32/F64: arith.divf; I32/I64: arith.divsi).
    pub fn emit_div(&mut self, lhs: &Tensor<'c>, rhs: &Tensor<'c>) -> Tensor<'c> {
        let op = match lhs.dtype() {
            DType::F32 | DType::F64 => "arith.divf",
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
            DType::F32 | DType::F64 => self.emit_linalg_unary("arith.negf", input),
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
        let lhs_padded: Vec<Option<u64>> = std::iter::repeat(Some(1))
            .take(out_rank - lhs_shape.len())
            .chain(lhs_shape.iter().copied())
            .collect();
        let rhs_padded: Vec<Option<u64>> = std::iter::repeat(Some(1))
            .take(out_rank - rhs_shape.len())
            .chain(rhs_shape.iter().copied())
            .collect();

        lhs_padded
            .iter()
            .zip(rhs_padded.iter())
            .map(|(l, r)| match (l, r) {
                (Some(1), other) | (other, Some(1)) => *other,
                (Some(a), Some(b)) if a == b => Some(*a),
                (None, _) | (_, None) => None,
                (Some(a), Some(b)) => panic!(
                    "broadcast shape mismatch: {a} vs {b}"
                ),
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
        let dims_attr = Attribute::parse(
            self.context,
            &format!("array<i64: {dims_str}>"),
        ).expect("broadcast dimensions attr");

        let dims_u64: Vec<u64> = out_shape
            .iter()
            .map(|d| match d {
                Some(n) => *n as u64,
                None => i64::MIN as u64,
            })
            .collect();
        let elem_type = dtype.to_mlir_type(self.context);
        let tensor_type: melior::ir::Type =
            RankedTensorType::new(&dims_u64, elem_type, None).into();

        // linalg.broadcast requires a region with a linalg.yield body.
        let body_block = Block::new(&[
            (elem_type, self.location),
            (elem_type, self.location),
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

        self.block
            .append_operation(
                OperationBuilder::new("linalg.broadcast", self.location)
                    .add_operands(&[input, init])
                    .add_results(&[tensor_type])
                    .add_attributes(&[
                        (Identifier::new(self.context, "dimensions"), dims_attr),
                    ])
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
    fn broadcast_to(
        &mut self,
        input: &Tensor<'c>,
        target_shape: &[Option<u64>],
    ) -> Tensor<'c> {
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
            let promoted_shape: Vec<Option<u64>> = std::iter::repeat(Some(1))
                .take(extra)
                .chain(src_shape.iter().copied())
                .collect();
            let val = self.emit_linalg_broadcast(
                input.value(),
                &promoted_shape,
                dtype,
                &broadcast_dims,
            );
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
        let indexing_maps = Attribute::parse(
            self.context,
            &format!("[{in_map}, {out_map}]"),
        ).expect("broadcast projected maps");

        let iterator_types = self.make_iterator_types(out_rank);
        let elem_type = dtype.to_mlir_type(self.context);

        let dims_u64: Vec<u64> = target_shape
            .iter()
            .map(|d| match d {
                Some(n) => *n as u64,
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

        let result = self.block
            .append_operation(
                OperationBuilder::new("linalg.generic", self.location)
                    .add_operands(&[rank_promoted_val, init])
                    .add_results(&[tensor_type])
                    .add_attributes(&[
                        (Identifier::new(self.context, "indexing_maps"), indexing_maps),
                        (Identifier::new(self.context, "iterator_types"), iterator_types),
                        (Identifier::new(self.context, "operandSegmentSizes"),
                         Attribute::parse(self.context, "array<i32: 1, 1>").unwrap()),
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

        let indexing_maps = Attribute::parse(
            self.context,
            &format!("[{lhs_map}, {rhs_map}, {out_map}]"),
        ).expect("broadcast indexing_maps");
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
            Some(self.emit_tensor_empty_for_broadcast(&out_shape, &lhs_padded, lhs_val, &rhs_padded, rhs_val, dtype))
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
        let op_result = body_block
            .append_operation(
                OperationBuilder::new(body_op, self.location)
                    .add_operands(&[a, b])
                    .add_results(&[elem_type])
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

        let result = self.block
            .append_operation(
                OperationBuilder::new("linalg.generic", self.location)
                    .add_operands(&[lhs_val, rhs_val, init])
                    .add_results(&[out_type])
                    .add_attributes(&[
                        (Identifier::new(self.context, "indexing_maps"), indexing_maps),
                        (Identifier::new(self.context, "iterator_types"), iterator_types),
                        (Identifier::new(self.context, "operandSegmentSizes"),
                         Attribute::parse(self.context, "array<i32: 2, 1>").unwrap()),
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
        rhs_shape: &[Option<u64>],
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
    fn emit_linalg_unary(&mut self, body_op: &str, input: &Tensor<'c>) -> Tensor<'c> {
        let shape = input.shape();
        let dtype = input.dtype();
        let rank = shape.len();
        let elem_type = dtype.to_mlir_type(self.context);

        let dims_u64: Vec<u64> = shape
            .iter()
            .map(|d| match d {
                Some(n) => *n as u64,
                None => i64::MIN as u64,
            })
            .collect();
        let tensor_type: melior::ir::Type =
            RankedTensorType::new(&dims_u64, elem_type, None).into();

        let init = self.emit_tensor_empty_dyn(&shape, dtype, Some(input.value()));
        let identity = identity_map_str(rank);
        let indexing_maps = Attribute::parse(
            self.context,
            &format!("[{0}, {0}]", identity),
        ).expect("indexing_maps");
        let iterator_types = self.make_iterator_types(rank);

        let body_block = Block::new(&[
            (elem_type, self.location),
            (elem_type, self.location),
        ]);
        let a: melior::ir::Value = body_block.argument(0).unwrap().into();
        let op_result = body_block
            .append_operation(
                OperationBuilder::new(body_op, self.location)
                    .add_operands(&[a])
                    .add_results(&[elem_type])
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

        let result = self.block
            .append_operation(
                OperationBuilder::new("linalg.generic", self.location)
                    .add_operands(&[input.value(), init])
                    .add_results(&[tensor_type])
                    .add_attributes(&[
                        (Identifier::new(self.context, "indexing_maps"), indexing_maps),
                        (Identifier::new(self.context, "iterator_types"), iterator_types),
                        (Identifier::new(self.context, "operandSegmentSizes"),
                         Attribute::parse(self.context, "array<i32: 1, 1>").unwrap()),
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

    /// Integer negation via `0 - x` (arith has no negsi).
    fn emit_linalg_unary_int_neg(&mut self, input: &Tensor<'c>) -> Tensor<'c> {
        let shape = input.shape();
        let dtype = input.dtype();
        let rank = shape.len();
        let elem_type = dtype.to_mlir_type(self.context);

        let dims_u64: Vec<u64> = shape
            .iter()
            .map(|d| match d {
                Some(n) => *n as u64,
                None => i64::MIN as u64,
            })
            .collect();
        let tensor_type: melior::ir::Type =
            RankedTensorType::new(&dims_u64, elem_type, None).into();

        let init = self.emit_tensor_empty_dyn(&shape, dtype, Some(input.value()));
        let identity = identity_map_str(rank);
        let indexing_maps = Attribute::parse(
            self.context,
            &format!("[{0}, {0}]", identity),
        ).expect("indexing_maps");
        let iterator_types = self.make_iterator_types(rank);

        let body_block = Block::new(&[
            (elem_type, self.location),
            (elem_type, self.location),
        ]);
        let x: melior::ir::Value = body_block.argument(0).unwrap().into();

        // Emit zero constant of matching integer type.
        let zero_attr = match dtype {
            DType::I32 => Attribute::parse(self.context, "0 : i32"),
            DType::I64 => Attribute::parse(self.context, "0 : i64"),
            _ => unreachable!(),
        }.expect("zero constant");
        let zero = body_block
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

        let neg = body_block
            .append_operation(
                OperationBuilder::new("arith.subi", self.location)
                    .add_operands(&[zero, x])
                    .add_results(&[elem_type])
                    .build()
                    .expect("arith.subi"),
            )
            .result(0)
            .unwrap()
            .into();
        body_block.append_operation(
            OperationBuilder::new("linalg.yield", self.location)
                .add_operands(&[neg])
                .build()
                .expect("linalg.yield"),
        );
        let body_region = Region::new();
        body_region.append_block(body_block);

        let result = self.block
            .append_operation(
                OperationBuilder::new("linalg.generic", self.location)
                    .add_operands(&[input.value(), init])
                    .add_results(&[tensor_type])
                    .add_attributes(&[
                        (Identifier::new(self.context, "indexing_maps"), indexing_maps),
                        (Identifier::new(self.context, "iterator_types"), iterator_types),
                        (Identifier::new(self.context, "operandSegmentSizes"),
                         Attribute::parse(self.context, "array<i32: 1, 1>").unwrap()),
                    ])
                    .add_regions([body_region])
                    .build()
                    .expect("linalg.generic int neg"),
            )
            .result(0)
            .unwrap()
            .into();
        Tensor::from_value(result)
    }

    /// ReLU: max(x, 0.0) via `arith.maximumf`.
    fn emit_linalg_unary_relu(&mut self, input: &Tensor<'c>) -> Tensor<'c> {
        let shape = input.shape();
        let dtype = input.dtype();
        let rank = shape.len();
        let elem_type = dtype.to_mlir_type(self.context);

        let dims_u64: Vec<u64> = shape
            .iter()
            .map(|d| match d {
                Some(n) => *n as u64,
                None => i64::MIN as u64,
            })
            .collect();
        let tensor_type: melior::ir::Type =
            RankedTensorType::new(&dims_u64, elem_type, None).into();

        let init = self.emit_tensor_empty_dyn(&shape, dtype, Some(input.value()));
        let identity = identity_map_str(rank);
        let indexing_maps = Attribute::parse(
            self.context,
            &format!("[{0}, {0}]", identity),
        ).expect("indexing_maps");
        let iterator_types = self.make_iterator_types(rank);

        let body_block = Block::new(&[
            (elem_type, self.location),
            (elem_type, self.location),
        ]);
        let x: melior::ir::Value = body_block.argument(0).unwrap().into();

        let zero_attr = match dtype {
            DType::F32 => Attribute::parse(self.context, "0.0 : f32"),
            DType::F64 => Attribute::parse(self.context, "0.0 : f64"),
            DType::I32 => Attribute::parse(self.context, "0 : i32"),
            DType::I64 => Attribute::parse(self.context, "0 : i64"),
        }.expect("zero for relu");
        let zero = body_block
            .append_operation(
                OperationBuilder::new("arith.constant", self.location)
                    .add_results(&[elem_type])
                    .add_attributes(&[(Identifier::new(self.context, "value"), zero_attr)])
                    .build()
                    .expect("arith.constant"),
            )
            .result(0)
            .unwrap()
            .into();

        let relu_op = match dtype {
            DType::F32 | DType::F64 => "arith.maximumf",
            DType::I32 | DType::I64 => "arith.maxsi",
        };
        let relu_val = body_block
            .append_operation(
                OperationBuilder::new(relu_op, self.location)
                    .add_operands(&[x, zero])
                    .add_results(&[elem_type])
                    .build()
                    .expect(relu_op),
            )
            .result(0)
            .unwrap()
            .into();
        body_block.append_operation(
            OperationBuilder::new("linalg.yield", self.location)
                .add_operands(&[relu_val])
                .build()
                .expect("linalg.yield"),
        );
        let body_region = Region::new();
        body_region.append_block(body_block);

        let result = self.block
            .append_operation(
                OperationBuilder::new("linalg.generic", self.location)
                    .add_operands(&[input.value(), init])
                    .add_results(&[tensor_type])
                    .add_attributes(&[
                        (Identifier::new(self.context, "indexing_maps"), indexing_maps),
                        (Identifier::new(self.context, "iterator_types"), iterator_types),
                        (Identifier::new(self.context, "operandSegmentSizes"),
                         Attribute::parse(self.context, "array<i32: 1, 1>").unwrap()),
                    ])
                    .add_regions([body_region])
                    .build()
                    .expect("linalg.generic relu"),
            )
            .result(0)
            .unwrap()
            .into();
        Tensor::from_value(result)
    }

    /// Reciprocal: 1.0 / x via `arith.divf`.
    fn emit_linalg_unary_reciprocal(&mut self, input: &Tensor<'c>) -> Tensor<'c> {
        let shape = input.shape();
        let dtype = input.dtype();
        let rank = shape.len();
        let elem_type = dtype.to_mlir_type(self.context);

        let dims_u64: Vec<u64> = shape
            .iter()
            .map(|d| match d {
                Some(n) => *n as u64,
                None => i64::MIN as u64,
            })
            .collect();
        let tensor_type: melior::ir::Type =
            RankedTensorType::new(&dims_u64, elem_type, None).into();

        let init = self.emit_tensor_empty_dyn(&shape, dtype, Some(input.value()));
        let identity = identity_map_str(rank);
        let indexing_maps = Attribute::parse(
            self.context,
            &format!("[{0}, {0}]", identity),
        ).expect("indexing_maps");
        let iterator_types = self.make_iterator_types(rank);

        let body_block = Block::new(&[
            (elem_type, self.location),
            (elem_type, self.location),
        ]);
        let x: melior::ir::Value = body_block.argument(0).unwrap().into();

        let one_attr = match dtype {
            DType::F32 => Attribute::parse(self.context, "1.0 : f32"),
            DType::F64 => Attribute::parse(self.context, "1.0 : f64"),
            _ => panic!("reciprocal is float-only"),
        }.expect("one for reciprocal");
        let one = body_block
            .append_operation(
                OperationBuilder::new("arith.constant", self.location)
                    .add_results(&[elem_type])
                    .add_attributes(&[(Identifier::new(self.context, "value"), one_attr)])
                    .build()
                    .expect("arith.constant"),
            )
            .result(0)
            .unwrap()
            .into();

        let recip = body_block
            .append_operation(
                OperationBuilder::new("arith.divf", self.location)
                    .add_operands(&[one, x])
                    .add_results(&[elem_type])
                    .build()
                    .expect("arith.divf reciprocal"),
            )
            .result(0)
            .unwrap()
            .into();
        body_block.append_operation(
            OperationBuilder::new("linalg.yield", self.location)
                .add_operands(&[recip])
                .build()
                .expect("linalg.yield"),
        );
        let body_region = Region::new();
        body_region.append_block(body_block);

        let result = self.block
            .append_operation(
                OperationBuilder::new("linalg.generic", self.location)
                    .add_operands(&[input.value(), init])
                    .add_results(&[tensor_type])
                    .add_attributes(&[
                        (Identifier::new(self.context, "indexing_maps"), indexing_maps),
                        (Identifier::new(self.context, "iterator_types"), iterator_types),
                        (Identifier::new(self.context, "operandSegmentSizes"),
                         Attribute::parse(self.context, "array<i32: 1, 1>").unwrap()),
                    ])
                    .add_regions([body_region])
                    .build()
                    .expect("linalg.generic reciprocal"),
            )
            .result(0)
            .unwrap()
            .into();
        Tensor::from_value(result)
    }

    // ── Reductions ────────────────────────────────────────────────────────────

    /// Reduce along a single axis via summation.
    ///
    /// `axis` supports Python-style negative indexing. If `keepdim` is true,
    /// the reduced axis is reinserted as a size-1 dimension.
    pub fn emit_reduce_sum(&mut self, input: &Tensor<'c>, axis: i64, keepdim: bool) -> Tensor<'c> {
        let dtype = input.dtype();
        let combiner_op = match dtype {
            DType::F32 | DType::F64 => "arith.addf",
            DType::I32 | DType::I64 => "arith.addi",
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
            DType::F32 | DType::F64 => "arith.maximumf",
            DType::I32 | DType::I64 => "arith.maxsi",
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
        let filled = self.emit_filled_tensor_for_reduce(input.value(), &reduce_shape, &non_reduced_indices, dtype, 0.0_f64);

        // Emit linalg.reduce with all axes at once.
        let dims_str = norm_axes.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ");
        let dimensions_attr = Attribute::parse(
            self.context,
            &format!("array<i64: {dims_str}>"),
        ).expect("reduce dimensions attr");

        let reduced_type = self.make_tensor_type(&reduce_shape, dtype);
        let elem_type = dtype.to_mlir_type(self.context);

        let body_block = Block::new(&[
            (elem_type, self.location), // in_elem
            (elem_type, self.location), // acc_elem
        ]);
        let in_e: melior::ir::Value = body_block.argument(0).unwrap().into();
        let acc_e: melior::ir::Value = body_block.argument(1).unwrap().into();
        let add_op = match dtype {
            DType::F32 | DType::F64 => "arith.addf",
            DType::I32 | DType::I64 => "arith.addi",
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

        let sum_val: melior::ir::Value = self.block
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
        assert!(axis >= 0 && axis < rank, "softmax axis {axis} out of bounds for rank {rank}");

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
        assert!(axis >= 0 && axis < rank, "axis {axis} out of bounds for rank {rank}");
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
        let filled = self.emit_scalar_filled_tensor(&reduce_shape, dtype, init_scalar_attr, input.value(), &input_shape, axis);

        let reduced_type = self.make_tensor_type(&reduce_shape, dtype);
        let dimensions_attr = Attribute::parse(
            self.context,
            &format!("array<i64: {axis}>"),
        ).expect("reduce dimensions attr");

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

        let reduced: melior::ir::Value = self.block
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
                let idx_val: melior::ir::Value = self.block
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
                let dim_val: melior::ir::Value = self.block
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

        let init: melior::ir::Value = self.block
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

        let scalar: melior::ir::Value = self.block
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
        let init: melior::ir::Value = self.block
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
            DType::F32 => Attribute::parse(self.context, &format!("{fill_f64:.6e} : f32")),
            DType::F64 => Attribute::parse(self.context, &format!("{fill_f64:.6e} : f64")),
            DType::I32 => Attribute::parse(self.context, &format!("{} : i32", fill_f64 as i64)),
            DType::I64 => Attribute::parse(self.context, &format!("{} : i64", fill_f64 as i64)),
        }.expect("fill attr");

        let scalar: melior::ir::Value = self.block
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
                    let ax_val: melior::ir::Value = self.block
                        .append_operation(
                            OperationBuilder::new("arith.constant", self.location)
                                .add_results(&[index_type])
                                .add_attributes(&[(Identifier::new(self.context, "value"), ax_attr)])
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
                Some(prev) => self.block
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
        let count_i64: melior::ir::Value = self.block
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
            DType::F32 | DType::F64 => "arith.sitofp",
            DType::I32 | DType::I64 => panic!("emit_div_by_index_scalar: float types only"),
        };
        let count_f: melior::ir::Value = self.block
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
        let scalar_tensor: melior::ir::Value = self.block
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
        ).expect("div mean indexing_maps");
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

        let result: melior::ir::Value = self.block
            .append_operation(
                OperationBuilder::new("linalg.generic", self.location)
                    .add_operands(&[input.value(), scalar_tensor, init])
                    .add_results(&[out_type])
                    .add_attributes(&[
                        (Identifier::new(self.context, "indexing_maps"), indexing_maps),
                        (Identifier::new(self.context, "iterator_types"), iterator_types),
                        (Identifier::new(self.context, "operandSegmentSizes"),
                         Attribute::parse(self.context, "array<i32: 2, 1>").unwrap()),
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
                if reduced_axes.contains(&i) { Some(1) } else { *d }
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
                let inner = g.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(", ");
                format!("[{inner}]")
            })
            .collect();
        let reassoc_str = format!("[{}]", groups.join(", "));

        self.emit_expand_shape_impl(reduced.value(), &keepdim_shape, dtype, &reassoc_str)
    }

    /// Build a zero scalar `Attribute` for the given dtype.
    fn make_zero_scalar_attr(&self, dtype: DType) -> Attribute<'c> {
        match dtype {
            DType::F32 => Attribute::parse(self.context, "0.0 : f32"),
            DType::F64 => Attribute::parse(self.context, "0.0 : f64"),
            DType::I32 => Attribute::parse(self.context, "0 : i32"),
            DType::I64 => Attribute::parse(self.context, "0 : i64"),
        }.expect("zero scalar attr")
    }

    /// Build a minimum-value scalar `Attribute` for the given dtype.
    /// For float: -infinity (IEEE 754). For int: MIN_VALUE.
    fn make_min_scalar_attr(&self, dtype: DType) -> Attribute<'c> {
        match dtype {
            // 0xFF800000 is -inf for f32; 0xFFF0000000000000 is -inf for f64.
            DType::F32 => Attribute::parse(self.context, "0xFF800000 : f32"),
            DType::F64 => Attribute::parse(self.context, "0xFFF0000000000000 : f64"),
            DType::I32 => Attribute::parse(self.context, "-2147483648 : i32"),
            DType::I64 => Attribute::parse(self.context, "-9223372036854775808 : i64"),
        }.expect("min scalar attr")
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
            (3, 3) => {
                self.emit_batch_matmul_3d(lhs.value(), &lhs_shape, rhs.value(), &rhs_shape)
            }
            (1, 2) => {
                // [K] -> [1, K] via expand_shape, matmul [1,K]x[K,N]->[1,N], collapse to [N]
                let exp_shape = vec![Some(1), lhs_shape[0]];
                let expanded = self.emit_expand_shape_1d_to_2d(lhs.value(), &lhs_shape, &exp_shape);
                let result_2d = self.emit_matmul_2d(
                    expanded.value(),
                    &exp_shape,
                    rhs.value(),
                    &rhs_shape,
                );
                // result is [1, N] -> collapse to [N]
                let n = rhs_shape[1];
                self.emit_collapse_shape_2d_to_1d(result_2d.value(), &[Some(1), n], &[n])
            }
            (2, 1) => {
                // rhs: [K] -> [K, 1], matmul [M,K]x[K,1]->[M,1], collapse to [M]
                let k = rhs_shape[0];
                let exp_shape = vec![k, Some(1)];
                let expanded = self.emit_expand_shape_1d_to_2d(rhs.value(), &rhs_shape, &exp_shape);
                let result_2d = self.emit_matmul_2d(
                    lhs.value(),
                    &lhs_shape,
                    expanded.value(),
                    &exp_shape,
                );
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
                let flat_b: Option<u64> = lhs_batch_dims.iter().try_fold(1u64, |acc, d| {
                    d.map(|v| acc * v)
                });
                let rhs_batch_dims = &rhs_shape[..rhs_rank - 2];
                let rhs_flat_b: Option<u64> = rhs_batch_dims.iter().try_fold(1u64, |acc, d| {
                    d.map(|v| acc * v)
                });

                let lhs_3d_shape = vec![flat_b, m, lhs_k];
                let rhs_3d_shape = vec![rhs_flat_b, rhs_k, n];
                let lhs_collapsed = self.emit_collapse_shape_nd_to_3d(
                    lhs.value(), &lhs_shape, &lhs_3d_shape,
                );
                let rhs_collapsed = self.emit_collapse_shape_nd_to_3d(
                    rhs.value(), &rhs_shape, &rhs_3d_shape,
                );

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
                self.emit_expand_shape_3d_to_nd(result_3d.value(), &result_3d_shape, &out_shape)
            }
            (3, 2) => {
                // [B,M,K] x [K,N] → unsqueeze rhs to [1,K,N], broadcast to [B,K,N], batch_matmul
                let rhs_3d = self.emit_unsqueeze(rhs, &[0]); // [K,N] → [1,K,N]
                let rhs_3d_shape = rhs_3d.shape();
                let b = lhs_shape[0];
                let bcast_shape = vec![b, rhs_3d_shape[1], rhs_3d_shape[2]];
                let bcasted = self.broadcast_to(&rhs_3d, &bcast_shape);
                self.emit_batch_matmul_3d(lhs.value(), &lhs_shape, bcasted.value(), &bcast_shape)
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

        // Compute A @ B via 2D matmul.
        let ab = self.emit_matmul(&a_used, &b_used);

        // Scale by alpha if != 1.0.
        let ab_scaled = if (alpha - 1.0f32).abs() > f32::EPSILON {
            self.emit_linalg_scale_f32(ab, alpha)
        } else {
            ab
        };

        // Scale C by beta if != 1.0.
        let c_scaled = if (beta - 1.0f32).abs() > f32::EPSILON {
            self.emit_linalg_scale_f32(*c, beta)
        } else {
            *c
        };

        // Add: (alpha * A@B) + (beta * C).
        self.emit_add(&ab_scaled, &c_scaled)
    }

    // ── Matmul internals ──────────────────────────────────────────────────────

    /// Emit `linalg.matmul` for 2D inputs `[M,K] x [K,N] -> [M,N]`.
    /// Handles static and dynamic M, K, N.
    fn emit_matmul_2d(
        &mut self,
        lhs_val: melior::ir::Value<'c, 'c>,
        lhs_shape: &[Option<u64>],  // [M, K]
        rhs_val: melior::ir::Value<'c, 'c>,
        rhs_shape: &[Option<u64>],  // [K, N]
    ) -> Tensor<'c> {
        let m = lhs_shape[0];
        let n = rhs_shape[1];
        let out_shape = vec![m, n];
        let dtype = self.value_dtype(lhs_val);

        let filled = self.emit_zero_filled_tensor(&out_shape, dtype, &[
            (lhs_val, 0), // M from lhs dim 0
            (rhs_val, 1), // N from rhs dim 1
        ]);

        let out_type = self.make_tensor_type(&out_shape, dtype);
        let segment = Attribute::parse(self.context, "array<i32: 2, 1>")
            .expect("matmul segment sizes");
        let matmul_region = self.make_matmul_region(dtype);

        let result = self.block
            .append_operation(
                OperationBuilder::new("linalg.matmul", self.location)
                    .add_operands(&[lhs_val, rhs_val, filled])
                    .add_results(&[out_type])
                    .add_attributes(&[(
                        Identifier::new(self.context, "operandSegmentSizes"),
                        segment,
                    )])
                    .add_regions([matmul_region])
                    .build()
                    .expect("linalg.matmul"),
            )
            .result(0)
            .unwrap()
            .into();
        Tensor::from_value(result)
    }

    /// Emit `linalg.batch_matmul` for 3D inputs `[B,M,K] x [B,K,N] -> [B,M,N]`.
    fn emit_batch_matmul_3d(
        &mut self,
        lhs_val: melior::ir::Value<'c, 'c>,
        lhs_shape: &[Option<u64>],  // [B, M, K]
        rhs_val: melior::ir::Value<'c, 'c>,
        rhs_shape: &[Option<u64>],  // [B, K, N]
    ) -> Tensor<'c> {
        let b = lhs_shape[0];
        let m = lhs_shape[1];
        let n = rhs_shape[2];
        let out_shape = vec![b, m, n];
        let dtype = self.value_dtype(lhs_val);

        let filled = self.emit_zero_filled_tensor(&out_shape, dtype, &[
            (lhs_val, 0), // B from lhs dim 0
            (lhs_val, 1), // M from lhs dim 1
            (rhs_val, 2), // N from rhs dim 2
        ]);

        let out_type = self.make_tensor_type(&out_shape, dtype);
        let segment = Attribute::parse(self.context, "array<i32: 2, 1>")
            .expect("batch_matmul segment sizes");
        let matmul_region = self.make_matmul_region(dtype);

        let result = self.block
            .append_operation(
                OperationBuilder::new("linalg.batch_matmul", self.location)
                    .add_operands(&[lhs_val, rhs_val, filled])
                    .add_results(&[out_type])
                    .add_attributes(&[(
                        Identifier::new(self.context, "operandSegmentSizes"),
                        segment,
                    )])
                    .add_regions([matmul_region])
                    .build()
                    .expect("linalg.batch_matmul"),
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
        let index_type = melior::ir::Type::parse(self.context, "index")
            .expect("index type");
        let mut dyn_iter = dyn_sources.iter();
        let mut dyn_vals: Vec<melior::ir::Value<'c, 'c>> = Vec::new();
        for dim in shape.iter() {
            if dim.is_none() {
                let &(src, idx) = dyn_iter.next()
                    .expect("dyn_sources must have an entry for each None dim");
                // Emit arith.constant for the dim index.
                let idx_attr = Attribute::parse(
                    self.context,
                    &format!("{idx} : index"),
                ).expect("dim index attr");
                let idx_val: melior::ir::Value = self.block
                    .append_operation(
                        OperationBuilder::new("arith.constant", self.location)
                            .add_results(&[index_type])
                            .add_attributes(&[(
                                Identifier::new(self.context, "value"),
                                idx_attr,
                            )])
                            .build()
                            .expect("arith.constant index"),
                    )
                    .result(0)
                    .unwrap()
                    .into();
                let dim_val: melior::ir::Value = self.block
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
        let init: melior::ir::Value = self.block
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
            DType::F32 => Attribute::parse(self.context, "0.0 : f32"),
            DType::F64 => Attribute::parse(self.context, "0.0 : f64"),
            DType::I32 => Attribute::parse(self.context, "0 : i32"),
            DType::I64 => Attribute::parse(self.context, "0 : i64"),
        }.expect("zero constant attr");
        let zero: melior::ir::Value = self.block
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

        let segment_fill = Attribute::parse(self.context, "array<i32: 1, 1>")
            .expect("fill segment sizes");

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
            DType::F32 | DType::F64 => block
                .append_operation(
                    OperationBuilder::new("arith.mulf", self.location)
                        .add_operands(&[lhs_e, rhs_e])
                        .add_results(&[elem_type])
                        .build()
                        .expect("arith.mulf"),
                )
                .result(0)
                .unwrap()
                .into(),
            DType::I32 | DType::I64 => block
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
        };
        let add: melior::ir::Value = match dtype {
            DType::F32 | DType::F64 => block
                .append_operation(
                    OperationBuilder::new("arith.addf", self.location)
                        .add_operands(&[acc_e, mul])
                        .add_results(&[elem_type])
                        .build()
                        .expect("arith.addf"),
                )
                .result(0)
                .unwrap()
                .into(),
            DType::I32 | DType::I64 => block
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

    /// Transpose a 2D tensor `[M, N] -> [N, M]` via `linalg.transpose`.
    fn emit_linalg_transpose_2d(&mut self, input: &Tensor<'c>) -> Tensor<'c> {
        let in_shape = input.shape();
        assert_eq!(in_shape.len(), 2, "emit_linalg_transpose_2d requires rank-2 tensor");
        let out_shape = vec![in_shape[1], in_shape[0]];
        let dtype = input.dtype();

        let init = self.emit_tensor_empty_with_dim_map(
            &out_shape, dtype, input.value(), &[1, 0],
        );
        let out_type = self.make_tensor_type(&out_shape, dtype);

        // permutation = [1, 0]
        let perm_attr = Attribute::parse(self.context, "array<i64: 1, 0>")
            .expect("transpose permutation");

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

        let result = self.block
            .append_operation(
                OperationBuilder::new("linalg.transpose", self.location)
                    .add_operands(&[input.value(), init])
                    .add_results(&[out_type])
                    .add_attributes(&[(
                        Identifier::new(self.context, "permutation"),
                        perm_attr,
                    )])
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
    /// - `0`           → keep input dim unchanged (not supported, caller must expand)
    /// - `-1`          → ONNX infer-this-dim: computed as `total_elements / product_of_known`
    ///
    /// Emits `tensor.reshape %input(%shape_tensor) : (...) -> tensor<...>`.
    /// The shape tensor is built from `arith.constant` / `tensor.dim` values via
    /// `tensor.from_elements`.
    pub fn emit_reshape(&mut self, input: &Tensor<'c>, target_shape: &[i64]) -> Tensor<'c> {
        let in_shape = input.shape();
        let dtype = input.dtype();
        let rank = in_shape.len();
        let index_type = melior::ir::Type::parse(self.context, "index").expect("index type");

        let infer_axis = target_shape.iter().position(|&d| d == -1);
        let target_rank = target_shape.len();

        // Helper: emit an `arith.constant <n> : index`.
        let emit_index_const = |block: &Block<'c>, ctx: &'c Context, n: u64| -> melior::ir::Value<'c, 'c> {
            let attr = Attribute::parse(ctx, &format!("{n} : index")).expect("index const attr");
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

        // Build a Value for each target dim.
        let mut dim_vals: Vec<melior::ir::Value<'c, 'c>> = Vec::with_capacity(target_rank);

        // Compute total input elements (needed only when -1 is present).
        let total_elems: Option<melior::ir::Value<'c, 'c>> = if infer_axis.is_some() {
            let mut prod: Option<melior::ir::Value<'c, 'c>> = None;
            for i in 0..rank {
                let dim_val = match in_shape[i] {
                    Some(n) => emit_index_const(&self.block, self.context, n),
                    None => self.emit_tensor_dim(input.value(), i),
                };
                prod = Some(match prod {
                    None => dim_val,
                    Some(prev) => self.block
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

        // Compute product of known (non -1) target dims.
        let known_product: Option<melior::ir::Value<'c, 'c>> = if infer_axis.is_some() {
            let mut prod: Option<melior::ir::Value<'c, 'c>> = None;
            for (i, &d) in target_shape.iter().enumerate() {
                if d == -1 { continue; }
                let dim_val = if d > 0 {
                    emit_index_const(&self.block, self.context, d as u64)
                } else if d == 0 {
                    // ONNX `0` means "copy from input dim i".
                    match in_shape.get(i).and_then(|d| *d) {
                        Some(n) => emit_index_const(&self.block, self.context, n),
                        None => self.emit_tensor_dim(input.value(), i),
                    }
                } else {
                    panic!("emit_reshape: invalid target_shape[{i}] = {d}");
                };
                prod = Some(match prod {
                    None => dim_val,
                    Some(prev) => self.block
                        .append_operation(
                            OperationBuilder::new("arith.muli", self.location)
                                .add_operands(&[prev, dim_val])
                                .add_results(&[index_type])
                                .build()
                                .expect("arith.muli known_product"),
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

        // Build per-dim values.
        for (i, &d) in target_shape.iter().enumerate() {
            let val = if d == -1 {
                // inferred dim = total / known_product
                let total = total_elems.expect("total_elems must exist for -1 dim");
                match known_product {
                    Some(known) => {
                        self.block
                            .append_operation(
                                OperationBuilder::new("arith.divui", self.location)
                                    .add_operands(&[total, known])
                                    .add_results(&[index_type])
                                    .build()
                                    .expect("arith.divui infer dim"),
                            )
                            .result(0)
                            .unwrap()
                            .into()
                    }
                    None => total, // only dim is -1, so it equals total_elems
                }
            } else if d > 0 {
                emit_index_const(&self.block, self.context, d as u64)
            } else if d == 0 {
                // ONNX `0` means "copy from input dim i".
                match in_shape.get(i).and_then(|d| *d) {
                    Some(n) => emit_index_const(&self.block, self.context, n),
                    None => self.emit_tensor_dim(input.value(), i),
                }
            } else {
                panic!("emit_reshape: invalid target_shape[{i}] = {d}");
            };
            dim_vals.push(val);
        }

        // Build shape tensor: tensor<target_rank x index> from elements.
        let shape_tensor_type: melior::ir::Type = {
            let dims_u64 = vec![target_rank as u64];
            RankedTensorType::new(&dims_u64, index_type, None).into()
        };
        let shape_tensor: melior::ir::Value<'c, 'c> = self.block
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

        // Compute result shape: static where possible.
        // - d > 0: static
        // - d == 0: copy from input dim i
        // - d == -1: infer from total_input / product_of_known_target (static if computable)
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
                    let total_static: Option<u64> = in_shape.iter()
                        .try_fold(1u64, |acc, dim| dim.map(|n| acc * n));
                    let known_product: Option<u64> = target_shape.iter().enumerate()
                        .try_fold(1u64, |acc, (j, &td)| {
                            if j == i { Some(acc) } // skip the -1 dim
                            else if td > 0 { Some(acc * td as u64) }
                            else if td == 0 {
                                in_shape.get(j).and_then(|opt| opt.map(|n| acc * n))
                            }
                            else { None }
                        });
                    match (total_static, known_product) {
                        (Some(t), Some(k)) if k > 0 => Some(t / k),
                        _ => None,
                    }
                }
            })
            .collect();
        let out_type = self.make_tensor_type(&out_shape, dtype);

        let result: melior::ir::Value<'c, 'c> = self.block
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

        let result: melior::ir::Value<'c, 'c> = self.block
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
            let neg1_i64: melior::ir::Value = self.block
                .append_operation(
                    OperationBuilder::new("arith.constant", loc)
                        .add_results(&[i64_type])
                        .add_attributes(&[(Identifier::new(ctx, "value"), neg1_attr)])
                        .build()
                        .expect("arith.constant -1"),
                )
                .result(0).unwrap().into();
            self.block
                .append_operation(
                    OperationBuilder::new("arith.index_cast", loc)
                        .add_operands(&[neg1_i64])
                        .add_results(&[index_type])
                        .build()
                        .expect("arith.index_cast -1"),
                )
                .result(0).unwrap().into()
        };

        // Compute total input elements.
        let mut total: melior::ir::Value<'c, 'c> = c1;
        for i in 0..in_rank {
            let dim_val = match in_shape[i] {
                Some(n) => self.emit_arith_constant_index(n),
                None => self.emit_tensor_dim(input.value(), i),
            };
            total = self.block
                .append_operation(
                    OperationBuilder::new("arith.muli", loc)
                        .add_operands(&[total, dim_val])
                        .add_results(&[index_type])
                        .build()
                        .expect("arith.muli total"),
                )
                .result(0).unwrap().into();
        }

        // First pass: resolve 0 entries (copy from input), leave -1 as-is.
        let resolved_zeros: Vec<melior::ir::Value<'c, 'c>> = dim_indices.iter().enumerate().map(|(i, &dim_val)| {
            if allowzero {
                return dim_val;
            }
            // Check if dim == 0 using arith.cmpi eq.
            let is_zero: melior::ir::Value = self.block
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
                .result(0).unwrap().into();

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
                .result(0).unwrap().into()
        }).collect();

        // Compute product of known dims (non -1).
        let mut known_product: melior::ir::Value<'c, 'c> = c1;
        for &dim_val in &resolved_zeros {
            // Check if dim == -1.
            let is_neg1: melior::ir::Value = self.block
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
                .result(0).unwrap().into();

            // If -1, contribute 1 to the product; else contribute dim_val.
            let contrib: melior::ir::Value = self.block
                .append_operation(
                    OperationBuilder::new("arith.select", loc)
                        .add_operands(&[is_neg1, c1, dim_val])
                        .add_results(&[index_type])
                        .build()
                        .expect("arith.select neg1 contrib"),
                )
                .result(0).unwrap().into();

            known_product = self.block
                .append_operation(
                    OperationBuilder::new("arith.muli", loc)
                        .add_operands(&[known_product, contrib])
                        .add_results(&[index_type])
                        .build()
                        .expect("arith.muli known_product"),
                )
                .result(0).unwrap().into();
        }

        // Inferred dim value = total / known_product.
        let inferred: melior::ir::Value<'c, 'c> = self.block
            .append_operation(
                OperationBuilder::new("arith.divui", loc)
                    .add_operands(&[total, known_product])
                    .add_results(&[index_type])
                    .build()
                    .expect("arith.divui inferred"),
            )
            .result(0).unwrap().into();

        // Second pass: replace -1 with inferred.
        resolved_zeros.iter().map(|&dim_val| {
            let is_neg1: melior::ir::Value = self.block
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
                .result(0).unwrap().into();

            self.block
                .append_operation(
                    OperationBuilder::new("arith.select", loc)
                        .add_operands(&[is_neg1, inferred, dim_val])
                        .add_results(&[index_type])
                        .build()
                        .expect("arith.select replace neg1"),
                )
                .result(0).unwrap().into()
        }).collect()
    }

    /// Reshape `input` using corrected runtime index dim values.
    ///
    /// Builds a `tensor<Nxindex>` shape tensor from the provided index values,
    /// then calls `tensor.reshape`.
    pub fn emit_reshape_from_index_dims(
        &mut self,
        input: &Tensor<'c>,
        dim_vals: &[melior::ir::Value<'c, 'c>],
        out_shape: &[Option<u64>],
    ) -> Tensor<'c> {
        let index_type = melior::ir::Type::parse(self.context, "index").expect("index type");
        let n = dim_vals.len() as u64;
        let shape_tensor_type: melior::ir::Type =
            melior::ir::r#type::RankedTensorType::new(&[n], index_type, None).into();

        let shape_tensor: melior::ir::Value<'c, 'c> = self.block
            .append_operation(
                OperationBuilder::new("tensor.from_elements", self.location)
                    .add_operands(dim_vals)
                    .add_results(&[shape_tensor_type])
                    .build()
                    .expect("tensor.from_elements reshape dims"),
            )
            .result(0).unwrap().into();

        self.emit_reshape_with_tensor(input, shape_tensor, out_shape)
    }

    /// Emit `arith.constant <n> : index`.
    fn emit_arith_constant_index(&mut self, n: u64) -> melior::ir::Value<'c, 'c> {
        let index_type = melior::ir::Type::parse(self.context, "index").expect("index type");
        let attr = Attribute::parse(self.context, &format!("{n} : index"))
            .expect("index const attr");
        self.block
            .append_operation(
                OperationBuilder::new("arith.constant", self.location)
                    .add_results(&[index_type])
                    .add_attributes(&[(Identifier::new(self.context, "value"), attr)])
                    .build()
                    .expect("arith.constant index"),
            )
            .result(0).unwrap().into()
    }

    /// Transpose `input` according to `perms` (ONNX-style signed permutation).
    ///
    /// E.g. `perms = [2, 0, 1]` maps output dim i to input dim perms[i].
    /// Emits `linalg.transpose` named op with the given permutation.
    pub fn emit_transpose(&mut self, input: &Tensor<'c>, perms: &[i64]) -> Tensor<'c> {
        let in_shape = input.shape();
        let rank = in_shape.len();
        let dtype = input.dtype();
        assert_eq!(perms.len(), rank, "emit_transpose: perms len must match rank");

        let norm_perms: Vec<usize> = perms.iter().map(|&p| {
            let p = if p < 0 { p + rank as i64 } else { p };
            assert!(p >= 0 && (p as usize) < rank, "perm {p} out of bounds for rank {rank}");
            p as usize
        }).collect();
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
        ).expect("transpose permutation attr");

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

        let result = self.block
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
        assert!(axis < rank, "emit_concat: axis {axis} out of bounds for rank {rank}");
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
                                    .add_attributes(&[(Identifier::new(self.context, "value"), attr)])
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
                    Some(prev) => self.block
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
                .map(|i| if i == axis {
                    if axis_offset_is_static { offset_static } else { K_DYNAMIC }
                } else {
                    0
                })
                .collect();

            // Build static_sizes: size of each input dim.
            let static_sizes: Vec<i64> = (0..rank)
                .map(|i| match inp_shape[i] { Some(n) => n as i64, None => K_DYNAMIC })
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
                                    .add_attributes(&[(Identifier::new(self.context, "value"), attr)])
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
                let s = static_offsets.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ");
                Attribute::parse(self.context, &format!("array<i64: {s}>"))
                    .expect("static_offsets attr")
            };
            let static_sizes_attr = {
                let s = static_sizes.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ");
                Attribute::parse(self.context, &format!("array<i64: {s}>"))
                    .expect("static_sizes attr")
            };
            let static_strides_attr = {
                let s = static_strides.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ");
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
            ).expect("insert_slice operandSegmentSizes");

            current = self.block
                .append_operation(
                    OperationBuilder::new("tensor.insert_slice", self.location)
                        .add_operands(&operands)
                        .add_results(&[out_type])
                        .add_attributes(&[
                            (Identifier::new(self.context, "static_offsets"), static_offsets_attr),
                            (Identifier::new(self.context, "static_sizes"), static_sizes_attr),
                            (Identifier::new(self.context, "static_strides"), static_strides_attr),
                            (Identifier::new(self.context, "operandSegmentSizes"), seg_attr),
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
                                    .add_attributes(&[(Identifier::new(self.context, "value"), attr)])
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
                                    .add_attributes(&[(Identifier::new(self.context, "value"), attr)])
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
                        .into()
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
        let norm_axes: Vec<usize> = axes.iter().map(|&a| {
            let a = if a < 0 { a + rank as i64 } else { a };
            a as usize
        }).collect();

        // Build per-dim (offset, size, stride) — defaulting to full dim for un-mentioned axes.
        let mut static_offsets = vec![0i64; rank];
        let mut static_sizes: Vec<i64> = in_shape.iter()
            .map(|d| match d { Some(n) => *n as i64, None => K_DYNAMIC })
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
            let e = if end < 0 { end + ax_size } else { end.min(ax_size) };
            let size = ((e - s + step - 1) / step).max(0);

            static_offsets[ax] = s;
            static_sizes[ax] = size;
            static_strides[ax] = step;
        }

        // Output shape.
        let out_shape: Vec<Option<u64>> = static_sizes.iter()
            .map(|&s| if s == K_DYNAMIC { None } else { Some(s as u64) })
            .collect();
        let out_type = self.make_tensor_type(&out_shape, dtype);

        let make_attr = |ctx: &'c Context, vals: &[i64]| -> Attribute<'c> {
            let s = vals.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ");
            Attribute::parse(ctx, &format!("array<i64: {s}>")).expect("slice attr")
        };

        let dyn_sizes: Vec<melior::ir::Value<'c, 'c>> = static_sizes.iter().enumerate()
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
        ).expect("operandSegmentSizes");

        let result: melior::ir::Value<'c, 'c> = self.block
            .append_operation(
                OperationBuilder::new("tensor.extract_slice", self.location)
                    .add_operands(&operands)
                    .add_results(&[out_type])
                    .add_attributes(&[
                        (Identifier::new(self.context, "static_offsets"),
                         make_attr(self.context, &static_offsets)),
                        (Identifier::new(self.context, "static_sizes"),
                         make_attr(self.context, &static_sizes)),
                        (Identifier::new(self.context, "static_strides"),
                         make_attr(self.context, &static_strides)),
                        (Identifier::new(self.context, "operandSegmentSizes"), seg_attr),
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
        let norm_axes: Vec<usize> = axes.iter().map(|&a| {
            let a = if a < 0 { a + rank as i64 } else { a };
            a as usize
        }).collect();

        // Per-dim static placeholders; sliced axes will be K_DYNAMIC.
        let mut static_offsets = vec![0i64; rank];
        let mut static_sizes: Vec<i64> = in_shape.iter()
            .map(|d| match d { Some(n) => *n as i64, None => K_DYNAMIC })
            .collect();
        let mut static_strides = vec![1i64; rank];

        // Mark sliced axes as dynamic (offsets and sizes come from runtime values).
        for (j, &ax) in norm_axes.iter().enumerate() {
            static_offsets[ax] = K_DYNAMIC;
            static_sizes[ax] = K_DYNAMIC;
            static_strides[ax] = steps[j];
        }

        // Output shape: dynamic for sliced axes, static (or dynamic) for the rest.
        let out_shape: Vec<Option<u64>> = (0..rank).map(|i| {
            if norm_axes.contains(&i) {
                None
            } else {
                in_shape[i]
            }
        }).collect();
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
            let diff: melior::ir::Value<'c, 'c> = self.block
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
                let step_attr = Attribute::parse(
                    self.context,
                    &format!("{} : index", steps[j]),
                ).expect("step attr");
                let step_val: melior::ir::Value<'c, 'c> = self.block
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
            for i in 0..rank {
                if norm_axes.contains(&i) {
                    all_dyn_sizes.push(dyn_sizes_per_axis[sliced_j]);
                    sliced_j += 1;
                } else if in_shape[i].is_none() {
                    let sz = self.emit_tensor_dim(input.value(), i);
                    all_dyn_sizes.push(sz);
                }
            }
        }

        let n_dyn_offsets = dyn_offsets.len() as i32;
        let n_dyn_sizes = all_dyn_sizes.len() as i32;

        let make_i64_array = |ctx: &'c Context, vals: &[i64]| -> Attribute<'c> {
            let s = vals.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ");
            Attribute::parse(ctx, &format!("array<i64: {s}>")).expect("i64 array attr")
        };

        let seg_attr = Attribute::parse(
            self.context,
            &format!("array<i32: 1, {n_dyn_offsets}, {n_dyn_sizes}, 0>"),
        ).expect("operandSegmentSizes");

        let mut operands = vec![input.value()];
        operands.extend(dyn_offsets);
        operands.extend(all_dyn_sizes);

        let ctx = self.context;
        let result: melior::ir::Value<'c, 'c> = self.block
            .append_operation(
                OperationBuilder::new("tensor.extract_slice", self.location)
                    .add_operands(&operands)
                    .add_results(&[out_type])
                    .add_attributes(&[
                        (Identifier::new(ctx, "static_offsets"),
                         make_i64_array(ctx, &static_offsets)),
                        (Identifier::new(ctx, "static_sizes"),
                         make_i64_array(ctx, &static_sizes)),
                        (Identifier::new(ctx, "static_strides"),
                         make_i64_array(ctx, &static_strides)),
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
    pub fn emit_gather(&mut self, data: &Tensor<'c>, indices: &Tensor<'c>, axis: usize) -> Tensor<'c> {
        let data_shape = data.shape();
        let idx_shape = indices.shape();
        let dtype = data.dtype();
        let data_rank = data_shape.len();
        let idx_rank = idx_shape.len();

        // Output shape: data[0..axis] + indices.shape + data[axis+1..]
        let mut out_shape: Vec<Option<u64>> = Vec::new();
        out_shape.extend_from_slice(&data_shape[..axis]);
        out_shape.extend_from_slice(&idx_shape);
        out_shape.extend_from_slice(&data_shape[axis+1..]);
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
        let init_proper: melior::ir::Value<'c, 'c> = self.block
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
        let idx_dims: Vec<String> = (axis..axis+idx_rank).map(|i| format!("d{i}")).collect();
        let idx_map = format!("affine_map<({dim_list}) -> ({})>", idx_dims.join(", "));

        // out map: identity.
        let out_map = format!("affine_map<({dim_list}) -> ({dim_list})>");

        let indexing_maps = Attribute::parse(
            self.context,
            &format!("[{idx_map}, {out_map}]"),
        ).expect("gather indexing_maps");

        // Body block: (indices_elem, out_elem)
        let body_block = Block::new(&[
            (idx_elem_type, self.location),  // indices element
            (elem_type, self.location),      // out element (unused, destination style)
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
        data_indices.extend_from_slice(&iter_indices[axis+idx_rank..out_rank]);

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

        let result: melior::ir::Value<'c, 'c> = self.block
            .append_operation(
                OperationBuilder::new("linalg.generic", self.location)
                    .add_operands(&[indices.value(), init_proper])
                    .add_results(&[out_type])
                    .add_attributes(&[
                        (Identifier::new(self.context, "indexing_maps"), indexing_maps),
                        (Identifier::new(self.context, "iterator_types"), iterator_types),
                        (Identifier::new(self.context, "operandSegmentSizes"),
                         Attribute::parse(self.context, "array<i32: 1, 1>").unwrap()),
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
        ).expect("where indexing_maps");
        let iterator_types = self.make_iterator_types(out_rank);

        let out_type = self.make_tensor_type(&out_shape, dtype);
        // Pick a dynamic-dim source: prefer the operand that matches the output shape.
        let init_source = if x_padded == out_shape {
            Some(x_val)
        } else if y_padded == out_shape {
            Some(y_val)
        } else {
            None
        };
        let init = self.emit_tensor_empty_dyn(&out_shape, dtype, init_source);

        let cond_dtype = cond.dtype();
        let cond_elem_type = cond_dtype.to_mlir_type(self.context);
        let i1_type = melior::ir::Type::parse(self.context, "i1").expect("i1 type");

        let body_block = Block::new(&[
            (cond_elem_type, self.location),  // cond element
            (elem_type, self.location),       // x element
            (elem_type, self.location),       // y element
            (elem_type, self.location),       // out element (unused)
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
                }.expect("zero for cond");
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
                            .add_attributes(&[(Identifier::new(self.context, "predicate"), ne_attr)])
                            .build()
                            .expect("arith.cmpi ne"),
                    )
                    .result(0)
                    .unwrap()
                    .into()
            }
            DType::F32 | DType::F64 => {
                // Compare float != 0.0.
                let zero_attr = match cond_dtype {
                    DType::F32 => Attribute::parse(self.context, "0.0 : f32"),
                    DType::F64 => Attribute::parse(self.context, "0.0 : f64"),
                    _ => unreachable!(),
                }.expect("zero float cond");
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
                            .add_attributes(&[(Identifier::new(self.context, "predicate"), une_attr)])
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

        let result: melior::ir::Value<'c, 'c> = self.block
            .append_operation(
                OperationBuilder::new("linalg.generic", self.location)
                    .add_operands(&[cond_val, x_val, y_val, init])
                    .add_results(&[out_type])
                    .add_attributes(&[
                        (Identifier::new(self.context, "indexing_maps"), indexing_maps),
                        (Identifier::new(self.context, "iterator_types"), iterator_types),
                        (Identifier::new(self.context, "operandSegmentSizes"),
                         Attribute::parse(self.context, "array<i32: 3, 1>").unwrap()),
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
            (DType::I32, DType::F32) | (DType::I32, DType::F64)
            | (DType::I64, DType::F32) | (DType::I64, DType::F64) => "arith.sitofp",
            (DType::F32, DType::I32) | (DType::F32, DType::I64)
            | (DType::F64, DType::I32) | (DType::F64, DType::I64) => "arith.fptosi",
            _ => panic!("emit_cast: unsupported cast {src_dtype:?} -> {target_dtype:?}"),
        };

        let shape = input.shape();
        let rank = shape.len();
        let src_elem_type = src_dtype.to_mlir_type(self.context);
        let tgt_elem_type = target_dtype.to_mlir_type(self.context);
        let out_type = self.make_tensor_type(&shape, target_dtype);

        let init = self.emit_tensor_empty_dyn(&shape, target_dtype, Some(input.value()));
        let identity = identity_map_str(rank);
        let indexing_maps = Attribute::parse(
            self.context,
            &format!("[{0}, {0}]", identity),
        ).expect("cast indexing_maps");
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

        let result: melior::ir::Value<'c, 'c> = self.block
            .append_operation(
                OperationBuilder::new("linalg.generic", self.location)
                    .add_operands(&[input.value(), init])
                    .add_results(&[out_type])
                    .add_attributes(&[
                        (Identifier::new(self.context, "indexing_maps"), indexing_maps),
                        (Identifier::new(self.context, "iterator_types"), iterator_types),
                        (Identifier::new(self.context, "operandSegmentSizes"),
                         Attribute::parse(self.context, "array<i32: 1, 1>").unwrap()),
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

    /// Insert size-1 dimensions at each position in `axes` via `tensor.expand_shape`.
    ///
    /// `axes` are the output positions where new dims should be inserted (ONNX-style).
    pub fn emit_unsqueeze(&mut self, input: &Tensor<'c>, axes: &[i64]) -> Tensor<'c> {
        let in_shape = input.shape();
        let src_rank = in_shape.len();
        let tgt_rank = src_rank + axes.len();

        // Normalize axes to output positions.
        let mut norm_axes: Vec<usize> = axes.iter().map(|&a| {
            let a = if a < 0 { a + tgt_rank as i64 } else { a };
            a as usize
        }).collect();
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
                    if src_rank == 0 { break; }
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

        let groups: Vec<String> = src_to_tgt.iter()
            .map(|g| {
                let inner = g.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(", ");
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
        let mut norm_axes: Vec<usize> = axes.iter().map(|&a| {
            let a = if a < 0 { a + src_rank as i64 } else { a };
            a as usize
        }).collect();
        norm_axes.sort_unstable();

        // Build output shape: remove the squeezed dims.
        let out_shape: Vec<Option<u64>> = in_shape.iter().enumerate()
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

        let groups: Vec<String> = tgt_to_src.iter()
            .map(|g| {
                let inner = g.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(", ");
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
        assert!(axis <= rank, "emit_flatten: axis {axis} out of bounds for rank {rank}");

        // Compute static dims for the two groups if possible.
        let head_size: Option<u64> = if axis == 0 {
            Some(1)
        } else {
            in_shape[..axis].iter().try_fold(1u64, |acc, d| d.map(|v| acc * v))
        };
        let tail_size: Option<u64> = if axis == rank {
            Some(1)
        } else {
            in_shape[axis..].iter().try_fold(1u64, |acc, d| d.map(|v| acc * v))
        };
        let out_shape = vec![head_size, tail_size];

        // Build reassociation: [[0..axis-1], [axis..rank-1]]
        // If axis == 0: head group is empty (not valid) — use [[0], [0..rank-1]] workaround.
        // tensor.collapse_shape doesn't allow empty groups, so use expand trick:
        // For axis==0: prepend a 1-dim then collapse.
        // Easier: emit reshape instead.
        if axis == 0 || axis == rank {
            // One group covers the entire input; use tensor.reshape.
            let target: Vec<i64> = if axis == 0 {
                vec![1, -1]
            } else {
                vec![-1, 1]
            };
            return self.emit_reshape(input, &target);
        }

        // General case: two non-empty groups.
        let head_indices: Vec<String> = (0..axis).map(|i| i.to_string()).collect();
        let tail_indices: Vec<String> = (axis..rank).map(|i| i.to_string()).collect();
        let reassoc_str = format!("[[{}], [{}]]", head_indices.join(", "), tail_indices.join(", "));

        self.emit_collapse_shape_with_reassoc(input.value(), &out_shape, &reassoc_str)
    }

    /// Split `input` into multiple tensors along `axis` with sizes given by `split_sizes`.
    ///
    /// `split_sizes` must sum to the axis dimension of `input`.
    /// Returns one `Tensor` per entry in `split_sizes`.
    pub fn emit_split(&mut self, input: &Tensor<'c>, axis: usize, split_sizes: &[u64]) -> Vec<Tensor<'c>> {
        let in_shape = input.shape();
        let rank = in_shape.len();
        let dtype = input.dtype();
        assert!(axis < rank, "emit_split: axis {axis} out of bounds for rank {rank}");
        assert!(!split_sizes.is_empty(), "emit_split: split_sizes must not be empty");
        const K_DYNAMIC: i64 = i64::MIN;

        let mut results: Vec<Tensor<'c>> = Vec::with_capacity(split_sizes.len());
        let mut offset: i64 = 0;

        for &size in split_sizes {
            // Out shape: same as input except axis dim = size.
            let out_shape: Vec<Option<u64>> = in_shape.iter().enumerate()
                .map(|(i, d)| if i == axis { Some(size) } else { *d })
                .collect();
            let out_type = self.make_tensor_type(&out_shape, dtype);

            // static_offsets: 0 for all except axis.
            let static_offsets: Vec<i64> = (0..rank)
                .map(|i| if i == axis { offset } else { 0 })
                .collect();

            // static_sizes: the split size for axis, input dims for others.
            let static_sizes: Vec<i64> = (0..rank)
                .map(|i| if i == axis {
                    size as i64
                } else {
                    match in_shape[i] { Some(n) => n as i64, None => K_DYNAMIC }
                })
                .collect();

            let static_strides: Vec<i64> = vec![1; rank];

            let dyn_sizes: Vec<melior::ir::Value<'c, 'c>> = static_sizes.iter().enumerate()
                .filter(|&(_, s)| *s == K_DYNAMIC)
                .map(|(i, _)| self.emit_tensor_dim(input.value(), i))
                .collect();

            let make_attr = |ctx: &'c Context, vals: &[i64]| -> Attribute<'c> {
                let s = vals.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ");
                Attribute::parse(ctx, &format!("array<i64: {s}>")).expect("split slice attr")
            };

            let dyn_sizes_len = dyn_sizes.len() as i32;
            let mut operands = vec![input.value()];
            operands.extend(dyn_sizes);

            let seg_attr = Attribute::parse(
                self.context,
                &format!("array<i32: 1, 0, {dyn_sizes_len}, 0>"),
            ).expect("operandSegmentSizes split");

            let slice: melior::ir::Value<'c, 'c> = self.block
                .append_operation(
                    OperationBuilder::new("tensor.extract_slice", self.location)
                        .add_operands(&operands)
                        .add_results(&[out_type])
                        .add_attributes(&[
                            (Identifier::new(self.context, "static_offsets"),
                             make_attr(self.context, &static_offsets)),
                            (Identifier::new(self.context, "static_sizes"),
                             make_attr(self.context, &static_sizes)),
                            (Identifier::new(self.context, "static_strides"),
                             make_attr(self.context, &static_strides)),
                            (Identifier::new(self.context, "operandSegmentSizes"), seg_attr),
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

    /// Scale every element of a float tensor by a scalar constant (linalg.generic).
    fn emit_linalg_scale_f32(&mut self, input: Tensor<'c>, scale: f32) -> Tensor<'c> {
        let shape = input.shape();
        let dtype = input.dtype();
        let elem_type = dtype.to_mlir_type(self.context);
        let rank = shape.len();

        let init = self.emit_tensor_empty_dyn(&shape, dtype, Some(input.value()));
        let out_type = self.make_tensor_type(&shape, dtype);
        let identity = identity_map_str(rank);
        let indexing_maps = Attribute::parse(
            self.context,
            &format!("[{0}, {0}]", identity),
        ).expect("scale indexing_maps");
        let iterator_types = self.make_iterator_types(rank);

        // MLIR requires a decimal point for float literals.
        let scale_f64 = scale as f64;
        let scale_attr = match dtype {
            DType::F32 => Attribute::parse(self.context, &format!("{scale_f64:.6e} : f32")),
            DType::F64 => Attribute::parse(self.context, &format!("{scale_f64:.6e} : f64")),
            _ => panic!("emit_linalg_scale_f32: float-only"),
        }.expect("scale constant attr");

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

        let result = self.block
            .append_operation(
                OperationBuilder::new("linalg.generic", self.location)
                    .add_operands(&[input.value(), init])
                    .add_results(&[out_type])
                    .add_attributes(&[
                        (Identifier::new(self.context, "indexing_maps"), indexing_maps),
                        (Identifier::new(self.context, "iterator_types"), iterator_types),
                        (Identifier::new(self.context, "operandSegmentSizes"),
                         Attribute::parse(self.context, "array<i32: 1, 1>").unwrap()),
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

    /// Expand a 1D tensor `[K]` to 2D `[1, K]` or `[K, 1]` via `tensor.expand_shape`.
    /// Reassociation: `[[0, 1]]` — both target dims come from source dim 0.
    fn emit_expand_shape_1d_to_2d(
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
    fn emit_collapse_shape_2d_to_1d(
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
    fn emit_collapse_shape_nd_to_3d(
        &mut self,
        input: melior::ir::Value<'c, 'c>,
        src_shape: &[Option<u64>],
        tgt_shape: &[Option<u64>],   // [B_flat, M, K]
    ) -> Tensor<'c> {
        let src_rank = src_shape.len();
        debug_assert!(src_rank >= 3);
        debug_assert_eq!(tgt_shape.len(), 3);

        // Use tensor.collapse_shape when safe (the batch group has at most one
        // dynamic dim, so MLIR can infer the collapsed type correctly).
        // Fall back to emit_reshape only when the batch group would have
        // multiple dynamic dims.
        let batch_dims = &src_shape[..src_rank - 2];
        let n_dynamic_in_batch = batch_dims.iter().filter(|d| d.is_none()).count();

        if n_dynamic_in_batch <= 1 {
            // Safe for collapse_shape.
            let batch_indices: Vec<String> = (0..src_rank - 2).map(|i| i.to_string()).collect();
            let batch_group = format!("[{}]", batch_indices.join(", "));
            let m_group = format!("[{}]", src_rank - 2);
            let k_group = format!("[{}]", src_rank - 1);
            let reassoc_str = format!("[{batch_group}, {m_group}, {k_group}]");
            return self.emit_collapse_shape_with_reassoc(input, tgt_shape, &reassoc_str);
        }

        // Multiple dynamic batch dims — fall back to emit_reshape.
        let target_i64: Vec<i64> = tgt_shape.iter().map(|d| match d {
            Some(n) => *n as i64,
            None => -1,
        }).collect();
        let input_tensor = Tensor::from_value(input);
        self.emit_reshape(&input_tensor, &target_i64)
    }

    /// Expand a 3D `[B_flat, M, N]` result back to ND `[..., M, N]`.
    /// Reassociation: first group expands the batch dim, then identity for M and N.
    fn emit_expand_shape_3d_to_nd(
        &mut self,
        input: melior::ir::Value<'c, 'c>,
        src_shape: &[Option<u64>],   // [B_flat, M, N]
        tgt_shape: &[Option<u64>],   // [..., M, N]
    ) -> Tensor<'c> {
        let tgt_rank = tgt_shape.len();
        debug_assert_eq!(src_shape.len(), 3);
        debug_assert!(tgt_rank >= 3);

        // Use tensor.expand_shape when safe: the batch group (dims 0..tgt_rank-2)
        // must have at most one dynamic dim, otherwise expand_shape can't infer
        // the correct output dims.
        let batch_tgt = &tgt_shape[..tgt_rank - 2];
        let n_dynamic_in_batch = batch_tgt.iter().filter(|d| d.is_none()).count();

        if n_dynamic_in_batch <= 1 {
            // Safe for expand_shape.
            let batch_indices: Vec<String> = (0..tgt_rank - 2).map(|i| i.to_string()).collect();
            let batch_group = format!("[{}]", batch_indices.join(", "));
            let m_group = format!("[{}]", tgt_rank - 2);
            let n_group = format!("[{}]", tgt_rank - 1);
            let reassoc_str = format!("[{batch_group}, {m_group}, {n_group}]");
            return self.emit_expand_shape_with_reassoc(input, tgt_shape, &reassoc_str);
        }

        // Multiple dynamic batch dims — fall back to emit_reshape.
        let target_i64: Vec<i64> = tgt_shape.iter().map(|d| match d {
            Some(n) => *n as i64,
            None => -1,
        }).collect();
        let input_tensor = Tensor::from_value(input);
        self.emit_reshape(&input_tensor, &target_i64)
    }

    fn emit_collapse_shape_with_reassoc(
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

        let reassoc_attr = Attribute::parse(self.context, reassoc_str)
            .expect("collapse_shape reassociation");

        let result: melior::ir::Value = self.block
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
    fn emit_expand_shape_impl(
        &mut self,
        input: melior::ir::Value<'c, 'c>,
        tgt_shape: &[Option<u64>],
        dtype: DType,
        reassoc_str: &str,
    ) -> Tensor<'c> {
        let out_type = self.make_tensor_type(tgt_shape, dtype);

        let reassoc_attr = Attribute::parse(self.context, reassoc_str)
            .expect("expand_shape reassociation");

        // static_output_shape: kDynamic sentinel = i64::MIN for dynamic dims.
        let static_shape_vals: Vec<i64> = tgt_shape.iter().map(|d| match d {
            Some(n) => *n as i64,
            None => i64::MIN,
        }).collect();
        let static_shape_str = static_shape_vals.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ");
        let static_output_shape_attr = Attribute::parse(
            self.context,
            &format!("array<i64: {static_shape_str}>"),
        ).expect("static_output_shape attr");

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

        let result: melior::ir::Value = self.block
            .append_operation(
                OperationBuilder::new("tensor.expand_shape", self.location)
                    .add_operands(&operands)
                    .add_results(&[out_type])
                    .add_attributes(&[
                        (Identifier::new(self.context, "reassociation"), reassoc_attr),
                        (Identifier::new(self.context, "static_output_shape"), static_output_shape_attr),
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
        let inner = &trimmed[1..trimmed.len()-1]; // strip outer []
        let mut depth = 0;
        let mut group_start = None;
        let mut group_idx = 0;
        for (i, ch) in inner.char_indices() {
            match ch {
                '[' => {
                    depth += 1;
                    if depth == 1 { group_start = Some(i + 1); }
                }
                ']' => {
                    depth -= 1;
                    if depth == 0 {
                        if let Some(start) = group_start {
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
                }
                _ => {}
            }
        }
        panic!("output dim {out_dim} not found in reassociation {reassoc_str}");
    }

    // ── Type construction helpers ─────────────────────────────────────────────

    /// Build a `RankedTensorType` from a shape with optional dims.
    fn make_tensor_type(&self, shape: &[Option<u64>], dtype: DType) -> melior::ir::Type<'c> {
        let dims_u64: Vec<u64> = shape
            .iter()
            .map(|d| match d {
                Some(n) => *n,
                None => i64::MIN as u64,
            })
            .collect();
        let elem_type = dtype.to_mlir_type(self.context);
        RankedTensorType::new(&dims_u64, elem_type, None).into()
    }

    /// Read the DType from an MLIR `Value`'s `RankedTensorType`.
    fn value_dtype(&self, val: melior::ir::Value<'c, 'c>) -> DType {
        let rtt = RankedTensorType::try_from(val.r#type())
            .expect("value_dtype: expected RankedTensorType");
        mlir_element_type_to_dtype(rtt.element())
    }

    // ── MLIR emission helpers ─────────────────────────────────────────────────

    /// Emit `tensor.empty` for the given shape/dtype.
    /// Emit `tensor.empty` for the given shape and dtype.
    ///
    /// For each `None` (dynamic) dim in `shape`, a runtime `index` value is
    /// needed. If `dyn_source` is provided, `tensor.dim` ops are emitted to
    /// read the corresponding dimension from that tensor. The source tensor
    /// must have the same rank and the `None` dims must appear at the same
    /// positions.
    ///
    /// If `dyn_source` is `None` and any dim is dynamic, this panics.

    /// Like `emit_tensor_empty` but with an explicit dynamic-dim source.
    fn emit_tensor_empty_dyn(
        &mut self,
        shape: &[Option<u64>],
        dtype: DType,
        dyn_source: Option<melior::ir::Value<'c, 'c>>,
    ) -> melior::ir::Value<'c, 'c> {
        let tensor_type = self.make_tensor_type(shape, dtype);

        let mut dyn_vals: Vec<melior::ir::Value<'c, 'c>> = Vec::new();
        for (i, dim) in shape.iter().enumerate() {
            if dim.is_none() {
                let src = dyn_source.unwrap_or_else(|| {
                    panic!(
                        "emit_tensor_empty: dim {i} is dynamic but no dyn_source provided (shape: {shape:?})"
                    )
                });
                let dim_val = self.emit_tensor_dim(src, i);
                dyn_vals.push(dim_val);
            }
        }

        self.block
            .append_operation(
                OperationBuilder::new("tensor.empty", self.location)
                    .add_operands(&dyn_vals)
                    .add_results(&[tensor_type])
                    .build()
                    .expect("tensor.empty"),
            )
            .result(0)
            .unwrap()
            .into()
    }

    /// Like `emit_tensor_empty_dyn` but output dim `i` reads source dim `dim_map[i]`.
    fn emit_tensor_empty_with_dim_map(
        &mut self,
        shape: &[Option<u64>],
        dtype: DType,
        source: melior::ir::Value<'c, 'c>,
        dim_map: &[usize],
    ) -> melior::ir::Value<'c, 'c> {
        let tensor_type = self.make_tensor_type(shape, dtype);

        let mut dyn_vals: Vec<melior::ir::Value<'c, 'c>> = Vec::new();
        for (i, dim) in shape.iter().enumerate() {
            if dim.is_none() {
                let src_dim = dim_map[i];
                let dim_val = self.emit_tensor_dim(source, src_dim);
                dyn_vals.push(dim_val);
            }
        }

        self.block
            .append_operation(
                OperationBuilder::new("tensor.empty", self.location)
                    .add_operands(&dyn_vals)
                    .add_results(&[tensor_type])
                    .build()
                    .expect("tensor.empty"),
            )
            .result(0)
            .unwrap()
            .into()
    }

    /// Emit `tensor.dim %tensor, %c_idx : tensor<...>` to get a runtime dim size.
    fn emit_tensor_dim(
        &mut self,
        tensor: melior::ir::Value<'c, 'c>,
        dim_idx: usize,
    ) -> melior::ir::Value<'c, 'c> {
        let index_type = melior::ir::Type::parse(self.context, "index").expect("index type");
        let idx_attr = Attribute::parse(self.context, &format!("{dim_idx} : index"))
            .expect("dim index attr");
        let idx_val: melior::ir::Value = self.block
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
        self.block
            .append_operation(
                OperationBuilder::new("tensor.dim", self.location)
                    .add_operands(&[tensor, idx_val])
                    .add_results(&[index_type])
                    .build()
                    .expect("tensor.dim"),
            )
            .result(0)
            .unwrap()
            .into()
    }

    /// Extract a scalar element from a 1-D tensor at a static position.
    ///
    /// Returns the element as an MLIR scalar value with the tensor's element type.
    /// `pos` must be in bounds.
    pub fn emit_tensor_extract_scalar(
        &mut self,
        tensor: &Tensor<'c>,
        pos: usize,
    ) -> melior::ir::Value<'c, 'c> {
        let elem_type = tensor.dtype().to_mlir_type(self.context);
        let index_type = melior::ir::Type::parse(self.context, "index").expect("index type");

        // Build arith.constant for the position index.
        let idx_attr = Attribute::parse(self.context, &format!("{pos} : index"))
            .expect("index attr");
        let idx_val: melior::ir::Value<'c, 'c> = self.block
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

        self.block
            .append_operation(
                OperationBuilder::new("tensor.extract", self.location)
                    .add_operands(&[tensor.value(), idx_val])
                    .add_results(&[elem_type])
                    .build()
                    .expect("tensor.extract scalar"),
            )
            .result(0)
            .unwrap()
            .into()
    }

    // ── Spatial Ops (task 6) ──────────────────────────────────────────────────

    /// Emit `tensor.pad` on the input with static low/high padding values.
    ///
    /// `pad_low[i]` and `pad_high[i]` are the padding amounts for each
    /// dimension. Dimensions with zero padding are passed through unchanged.
    /// The region yields `pad_value`.
    fn emit_tensor_pad(
        &mut self,
        input: melior::ir::Value<'c, 'c>,
        pad_low: &[i64],
        pad_high: &[i64],
        pad_value_attr: Attribute<'c>,
    ) -> melior::ir::Value<'c, 'c> {
        let rtt = RankedTensorType::try_from(input.r#type())
            .expect("tensor.pad: input must be RankedTensorType");
        let rank = rtt.rank() as usize;
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
        let low_str = pad_low.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ");
        let high_str = pad_high.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ");
        let static_low_attr = Attribute::parse(
            self.context, &format!("array<i64: {low_str}>"),
        ).expect("static_low attr");
        let static_high_attr = Attribute::parse(
            self.context, &format!("array<i64: {high_str}>"),
        ).expect("static_high attr");

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
                        (Identifier::new(self.context, "static_high"), static_high_attr),
                        // AttrSizedOperandSegments: source(1), low_dynamic(0), high_dynamic(0).
                        (Identifier::new(self.context, "operandSegmentSizes"),
                         Attribute::parse(self.context, "array<i32: 1, 0, 0>").unwrap()),
                    ])
                    .add_regions([pad_region])
                    .build()
                    .expect("tensor.pad"),
            )
            .result(0)
            .unwrap()
            .into()
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
    ) -> Tensor<'c> {
        let in_shape = input.shape();   // [N, CI, H, W]
        let wt_shape = weight.shape();  // [CO, CI, KH, KW]
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
                DType::F32 => Attribute::parse(self.context, "0.0 : f32"),
                DType::F64 => Attribute::parse(self.context, "0.0 : f64"),
                DType::I32 => Attribute::parse(self.context, "0 : i32"),
                DType::I64 => Attribute::parse(self.context, "0 : i64"),
            }.expect("conv2d pad zero");
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
        let oh = h_padded.zip(kh).map(|(h, k)| {
            (h - dilation_h * (k - 1) - 1) / stride_h + 1
        });
        let ow = w_padded.zip(kw).map(|(w, k)| {
            (w - dilation_w * (k - 1) - 1) / stride_w + 1
        });
        let out_shape = vec![n, co, oh, ow];

        // Step 3: zero-filled output tensor [N, CO, OH, OW].
        let out_type = self.make_tensor_type(&out_shape, dtype);
        // For dynamic dims, we read from padded input or weight.
        let elem_type = dtype.to_mlir_type(self.context);
        let zero_attr = match dtype {
            DType::F32 => Attribute::parse(self.context, "0.0 : f32"),
            DType::F64 => Attribute::parse(self.context, "0.0 : f64"),
            DType::I32 => Attribute::parse(self.context, "0 : i32"),
            DType::I64 => Attribute::parse(self.context, "0 : i64"),
        }.expect("conv2d output zero");
        let zero_scalar: melior::ir::Value = self.block
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
        assert!(oh.is_some() && ow.is_some(),
            "conv2d: dynamic output spatial dims not yet supported");

        let init_empty: melior::ir::Value = self.block
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

        let init_filled: melior::ir::Value = self.block
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
        ).expect("conv2d strides attr");
        let dilations_attr = Attribute::parse(
            self.context,
            &format!("dense<[{dilation_h}, {dilation_w}]> : tensor<2xi64>"),
        ).expect("conv2d dilations attr");

        // linalg.conv_2d_nchw_fchw requires an explicit region with body:
        //   ^bb0(%in: f32, %filter: f32, %acc: f32):
        //     %prod = arith.mulf %in, %filter : f32
        //     %sum  = arith.addf %acc, %prod  : f32
        //     linalg.yield %sum : f32
        let conv_region = self.make_conv2d_region(dtype);

        let conv_result: melior::ir::Value = self.block
            .append_operation(
                OperationBuilder::new("linalg.conv_2d_nchw_fchw", self.location)
                    .add_operands(&[input_val, weight.value(), init_filled])
                    .add_results(&[out_type])
                    .add_attributes(&[
                        (Identifier::new(self.context, "strides"), strides_attr),
                        (Identifier::new(self.context, "dilations"), dilations_attr),
                        (Identifier::new(self.context, "operandSegmentSizes"),
                         Attribute::parse(self.context, "array<i32: 2, 1>").unwrap()),
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
            self.emit_add(&conv_tensor, &bias_4d)
        } else {
            conv_tensor
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
    ) -> Tensor<'c> {
        let in_shape = input.shape();
        let wt_shape = weight.shape();
        assert_eq!(in_shape.len(), 4, "conv2d_same_upper: input must be NCHW rank-4");
        assert_eq!(wt_shape.len(), 4, "conv2d_same_upper: weight must be FCHW rank-4");

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

        self.emit_conv2d(input, weight, bias, [pad_top, pad_left, pad_bottom, pad_right], strides, dilations)
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
            DType::F32 | DType::F64 => block
                .append_operation(
                    OperationBuilder::new("arith.mulf", self.location)
                        .add_operands(&[in_e, filter_e])
                        .add_results(&[elem_type])
                        .build()
                        .expect("arith.mulf conv2d"),
                )
                .result(0).unwrap().into(),
            DType::I32 | DType::I64 => block
                .append_operation(
                    OperationBuilder::new("arith.muli", self.location)
                        .add_operands(&[in_e, filter_e])
                        .add_results(&[elem_type])
                        .build()
                        .expect("arith.muli conv2d"),
                )
                .result(0).unwrap().into(),
        };
        let sum: melior::ir::Value = match dtype {
            DType::F32 | DType::F64 => block
                .append_operation(
                    OperationBuilder::new("arith.addf", self.location)
                        .add_operands(&[acc_e, prod])
                        .add_results(&[elem_type])
                        .build()
                        .expect("arith.addf conv2d"),
                )
                .result(0).unwrap().into(),
            DType::I32 | DType::I64 => block
                .append_operation(
                    OperationBuilder::new("arith.addi", self.location)
                        .add_operands(&[acc_e, prod])
                        .add_results(&[elem_type])
                        .build()
                        .expect("arith.addi conv2d"),
                )
                .result(0).unwrap().into(),
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
    ) -> Tensor<'c> {
        let in_shape = input.shape();   // [N, C, H, W]
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
            }.expect("max_pool2d -inf pad value");
            self.emit_tensor_pad(input.value(), &pad_low, &pad_high, neg_inf_attr)
        } else {
            input.value()
        };

        // Get padded input shape.
        let padded_rtt = RankedTensorType::try_from(input_val.r#type())
            .expect("padded input RankedTensorType");
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
        }.expect("max_pool2d output -inf");

        let neg_inf_scalar: melior::ir::Value = self.block
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
        if n.is_none() { dyn_vals.push(self.emit_tensor_dim(input_val, 0)); }
        if c.is_none() { dyn_vals.push(self.emit_tensor_dim(input_val, 1)); }
        assert!(oh.is_some() && ow.is_some(), "max_pool2d: dynamic output spatial dims not yet supported");

        let init_empty: melior::ir::Value = self.block
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

        let init_filled: melior::ir::Value = self.block
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
        let window_empty: melior::ir::Value = self.block
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
        ).expect("max_pool2d strides attr");
        let dilations_attr = Attribute::parse(
            self.context,
            &format!("dense<[{dilation_h}, {dilation_w}]> : tensor<2xi64>"),
        ).expect("max_pool2d dilations attr");

        // Region body: max(in, acc).
        let pool_region = self.make_pool_max_region(dtype);

        let pool_result: melior::ir::Value = self.block
            .append_operation(
                OperationBuilder::new("linalg.pooling_nchw_max", self.location)
                    .add_operands(&[input_val, window_empty, init_filled])
                    .add_results(&[out_type])
                    .add_attributes(&[
                        (Identifier::new(self.context, "strides"), strides_attr),
                        (Identifier::new(self.context, "dilations"), dilations_attr),
                        (Identifier::new(self.context, "operandSegmentSizes"),
                         Attribute::parse(self.context, "array<i32: 2, 1>").unwrap()),
                    ])
                    .add_regions([pool_region])
                    .build()
                    .expect("linalg.pooling_nchw_max"),
            )
            .result(0)
            .unwrap()
            .into();

        Tensor::from_value(pool_result)
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
            DType::F32 | DType::F64 => "arith.maximumf",
            DType::I32 | DType::I64 => "arith.maxsi",
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
        let in_shape = input.shape();  // [N, C, H, W]
        assert_eq!(in_shape.len(), 4, "global_avg_pool: input must be rank-4 NCHW");

        // Sum over H then W (keepdim=true to preserve rank).
        let sum_h = self.emit_reduce_sum(input, 2, true);    // [N, C, 1, W]
        let sum_hw = self.emit_reduce_sum(&sum_h, 3, true);  // [N, C, 1, 1]

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

    /// Broadcast a scalar `f32` constant to the same shape as `like`.
    fn emit_scalar_broadcast(&mut self, value: f32, like: &Tensor<'c>) -> Tensor<'c> {
        let shape = like.shape();
        let dtype = like.dtype();
        let elem_type = dtype.to_mlir_type(self.context);

        let value_f64 = value as f64;
        let val_attr = match dtype {
            DType::F32 => Attribute::parse(self.context, &format!("{value_f64:.6e} : f32")),
            DType::F64 => Attribute::parse(self.context, &format!("{value_f64:.6e} : f64")),
            _ => panic!("emit_scalar_broadcast: float only"),
        }.expect("scalar broadcast attr");

        let out_type = self.make_tensor_type(&shape, dtype);
        let init = self.emit_tensor_empty_dyn(&shape, dtype, Some(like.value()));

        // linalg.generic with empty inputs: body just yields the constant.
        let identity = identity_map_str(shape.len());
        let indexing_maps = Attribute::parse(
            self.context,
            &format!("[{identity}]"),
        ).expect("scalar broadcast indexing_maps");
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

        let result = self.block
            .append_operation(
                OperationBuilder::new("linalg.generic", self.location)
                    .add_operands(&[init])
                    .add_results(&[out_type])
                    .add_attributes(&[
                        (Identifier::new(self.context, "indexing_maps"), indexing_maps),
                        (Identifier::new(self.context, "iterator_types"), iterator_types),
                        (Identifier::new(self.context, "operandSegmentSizes"),
                         Attribute::parse(self.context, "array<i32: 0, 1>").unwrap()),
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

    fn make_iterator_types(&self, count: usize) -> Attribute<'c> {
        let entries: Vec<&str> = vec!["#linalg.iterator_type<parallel>"; count];
        let attr_str = format!("[{}]", entries.join(", "));
        Attribute::parse(self.context, &attr_str).expect("iterator_types")
    }

    /// Compile the graph. `outputs` are the tensor values to return.
    pub fn compile(self, outputs: &[&Tensor<'c>]) -> Result<CompiledGraph, CompileError> {
        self.compile_with_cache(outputs, None)
    }

    /// Compile with optional cache key.
    pub fn compile_with_cache(
        self,
        outputs: &[&Tensor<'c>],
        cache_key: Option<&str>,
    ) -> Result<CompiledGraph, CompileError> {
        if outputs.is_empty() {
            return Err(CompileError::NoOutputs);
        }

        let context = self.context;
        let location = self.location;
        let num_args = self.args.len();

        // Build output descriptors.
        let output_descs: Vec<OutputDesc> = outputs
            .iter()
            .map(|t| {
                let shape_vec: Vec<u64> = t.shape().iter().map(|d| match d {
                    Some(n) => *n,
                    None => crate::shape::DIM_DYNAMIC,
                }).collect();
                OutputDesc {
                    shape: crate::Shape(shape_vec),
                    dtype: t.dtype(),
                }
            })
            .collect();

        // Add output memref arguments + bufferization.to_buffer + memref.copy.
        for (out_idx, &output_tensor) in outputs.iter().enumerate() {
            let out_shape = output_tensor.shape();
            let out_dtype = output_tensor.dtype();
            let dims: Vec<i64> = out_shape.iter().map(|d| match d {
                Some(n) => *n as i64,
                None => i64::MIN,
            }).collect();
            let out_memref_type: melior::ir::Type<'c> =
                MemRefType::new(out_dtype.to_mlir_type(context), &dims, None, None).into();

            self.block.add_argument(out_memref_type, location);

            let result_memref: melior::ir::Value = self.block
                .append_operation(
                    OperationBuilder::new("bufferization.to_buffer", location)
                        .add_operands(&[output_tensor.value()])
                        .add_results(&[out_memref_type])
                        .build()
                        .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
                )
                .result(0)
                .unwrap()
                .into();

            let out_arg_idx = num_args + out_idx;
            let out_arg: melior::ir::Value = self.block.argument(out_arg_idx).unwrap().into();
            self.block.append_operation(
                OperationBuilder::new("memref.copy", location)
                    .add_operands(&[result_memref, out_arg])
                    .build()
                    .map_err(|e| CompileError::AttributeParse(e.to_string()))?,
            );
        }

        // func.return
        self.block.append_operation(func::r#return(&[], location));

        // Build function type: all block args → void.
        let all_arg_types: Vec<melior::ir::Type> = (0..self.block.argument_count())
            .map(|i| self.block.argument(i).unwrap().r#type())
            .collect();
        let function_type = FunctionType::new(context, &all_arg_types, &[]);

        // Build module.
        let mut module = Module::new(location);

        // Set data layout.
        let dl_attr = Attribute::parse(context,
            "#dlti.dl_spec<\
                index = 64 : i64, \
                i32 = dense<32> : vector<2xi64>, \
                i64 = dense<64> : vector<2xi64>, \
                f32 = dense<32> : vector<2xi64>, \
                f64 = dense<64> : vector<2xi64>, \
                !llvm.ptr = dense<64> : vector<4xi64>\
            >"
        ).expect("failed to parse dlti.dl_spec");
        module.as_operation_mut().set_attribute("dlti.dl_spec", dl_attr);

        // Emit sub-functions before @compute.
        for sub in self.completed_subfunctions {
            let sub_func_type =
                FunctionType::new(context, &sub.arg_types, &sub.return_types);
            let sub_region = Region::new();
            sub_region.append_block(sub.block);

            let sub_func = func::func(
                context,
                StringAttribute::new(context, &sub.name),
                TypeAttribute::new(sub_func_type.into()),
                sub_region,
                &[(
                    Identifier::new(context, "llvm.emit_c_interface"),
                    Attribute::unit(context),
                )],
                location,
            );
            module.body().append_operation(sub_func);
        }

        // Build func.func @compute.
        let func_region = Region::new();
        func_region.append_block(self.block);

        let function = func::func(
            context,
            StringAttribute::new(context, "compute"),
            TypeAttribute::new(function_type.into()),
            func_region,
            &[(
                Identifier::new(context, "llvm.emit_c_interface"),
                Attribute::unit(context),
            )],
            location,
        );

        module.body().append_operation(function);

        // Dump pre-pass IR if requested.
        if std::env::var("HOMURA_DUMP_IR").is_ok() {
            let _ = std::fs::write("/tmp/homura_gb_pre_passes.mlir", module.as_operation().to_string());
            eprintln!("[homura] GraphBuilder pre-pass IR dumped to /tmp/homura_gb_pre_passes.mlir");
        }

        // Verify.
        if !module.as_operation().verify() {
            let ir = module.as_operation().to_string();
            let _ = std::fs::write("/tmp/homura_gb_failed.mlir", &ir);
            eprintln!("[homura] GraphBuilder MLIR verification failed — IR dumped to /tmp/homura_gb_failed.mlir");
            return Err(CompileError::Verification);
        }

        // Cache check.
        if let Some(key) = cache_key {
            let cache = crate::cache::CompilationCache::new();
            if let Some((so_path, meta_path)) = cache.get(key) {
                if let Some(meta) = crate::cache::CompilationCache::load_meta(&meta_path) {
                    match CompiledGraph::load(&so_path, meta.num_inputs, meta.outputs) {
                        Ok(graph) => return Ok(graph),
                        Err(e) => {
                            eprintln!(
                                "homura cache: failed to load {}: {e}, recompiling",
                                so_path.display()
                            );
                        }
                    }
                }
            }
        }

        // Run lowering passes (linalg-direct pipeline, no TOSA).
        register_all_passes();
        let pass_manager = pass::PassManager::new(context);
        parse_pass_pipeline(
            pass_manager.as_operation_pass_manager(),
            "builtin.module(\
                func.func(canonicalize,cse),\
                one-shot-bufferize{bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map unknown-type-conversion=identity-layout-map},\
                func.func(buffer-hoisting,promote-buffers-to-stack{max-alloc-size-in-bytes=4096}),\
                fold-memref-alias-ops,\
                func.func(linalg-generalize-named-ops),\
                convert-linalg-to-loops,\
                fold-memref-alias-ops,\
                lower-affine,\
                convert-scf-to-cf,\
                canonicalize,\
                cse,\
                sccp,\
                convert-math-to-llvm,\
                expand-strided-metadata,\
                lower-affine,\
                finalize-memref-to-llvm,\
                convert-arith-to-llvm,\
                convert-index-to-llvm,\
                convert-cf-to-llvm,\
                convert-func-to-llvm,\
                reconcile-unrealized-casts\
            )",
        )
        .map_err(CompileError::Pass)?;

        pass_manager.run(&mut module).map_err(CompileError::Pass)?;

        if std::env::var("HOMURA_DUMP_IR").is_ok() {
            let _ = std::fs::write("/tmp/homura_gb_post_passes.mlir", module.as_operation().to_string());
            eprintln!("[homura] GraphBuilder post-pass IR dumped to /tmp/homura_gb_post_passes.mlir");
        }

        // AOT compile.
        let tmp_dir = tempfile_dir().ok_or_else(|| {
            CompileError::ObjectEmit("cannot determine temp directory".into())
        })?;
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos();
        let suffix = format!("{}_{:08x}", std::process::id(), nanos);
        let tmp_obj = tmp_dir.join(format!("homura_gb_{suffix}.o"));
        let tmp_so = tmp_dir.join(format!("homura_gb_{suffix}.so"));

        crate::compiler::emit_object_file_pub(&module, &tmp_obj)?;
        crate::compiler::link_shared_lib_pub(&tmp_obj, &tmp_so)?;
        std::fs::remove_file(&tmp_obj).ok();

        // Store in cache.
        if let Some(key) = cache_key {
            let cache = crate::cache::CompilationCache::new();
            let meta = crate::cache::CacheMeta {
                num_inputs: num_args,
                outputs: output_descs
                    .iter()
                    .map(|d| OutputDesc { shape: d.shape.clone(), dtype: d.dtype })
                    .collect(),
            };
            if let Err(e) = cache.store(key, &tmp_so, &meta) {
                eprintln!("homura cache: failed to write cache entry: {e}");
            }
        }

        let graph = CompiledGraph::load(&tmp_so, num_args, output_descs)
            .map_err(CompileError::ObjectEmit)?;
        std::fs::remove_file(&tmp_so).ok();

        Ok(graph)
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    fn add_arg(&mut self, shape: &[Option<u64>], dtype: DType, is_input: bool) -> Tensor<'c> {
        let dims: Vec<i64> = shape.iter().map(|d| match d {
            Some(n) => *n as i64,
            None => i64::MIN,
        }).collect();
        let elem_type = dtype.to_mlir_type(self.context);

        // RankedTensorType uses u64 dims where kDynamic = i64::MIN as u64.
        let dims_u64: Vec<u64> = dims.iter().map(|&d| d as u64).collect();
        let tensor_type: melior::ir::Type =
            RankedTensorType::new(&dims_u64, elem_type, None).into();
        let memref_type: melior::ir::Type =
            MemRefType::new(elem_type, &dims, None, None).into();

        let arg_idx = self.block.argument_count();
        self.block.add_argument(memref_type, self.location);
        let memref_arg = self.block.argument(arg_idx).unwrap();

        // bufferization.to_tensor %memref {restrict}
        let tensor_val = self.block
            .append_operation(
                OperationBuilder::new("bufferization.to_tensor", self.location)
                    .add_operands(&[memref_arg.into()])
                    .add_results(&[tensor_type])
                    .add_attributes(&[(
                        Identifier::new(self.context, "restrict"),
                        Attribute::unit(self.context),
                    )])
                    .build()
                    .expect("bufferization.to_tensor"),
            )
            .result(0)
            .unwrap()
            .into();

        self.args.push(ArgInfo { shape: shape.to_vec(), dtype, is_input });

        Tensor::from_value(tensor_val)
    }

    // ── Meta ops (shape subgraph) ─────────────────────────────────────────────

    /// Emit `tensor.dim` for each dimension of `input`, returning a list of
    /// runtime `index` values. Static dims become `arith.constant`, dynamic
    /// dims become `tensor.dim` results.
    pub fn emit_shape_of(&mut self, input: &Tensor<'c>) -> Vec<melior::ir::Value<'c, 'c>> {
        let shape = input.shape();
        let index_type = melior::ir::Type::parse(self.context, "index").expect("index type");
        shape.iter().enumerate().map(|(i, dim)| {
            match dim {
                Some(n) => {
                    let attr = Attribute::parse(self.context, &format!("{n} : index"))
                        .expect("index constant attr");
                    self.block
                        .append_operation(
                            OperationBuilder::new("arith.constant", self.location)
                                .add_results(&[index_type])
                                .add_attributes(&[(Identifier::new(self.context, "value"), attr)])
                                .build()
                                .expect("arith.constant index"),
                        )
                        .result(0)
                        .unwrap()
                        .into()
                }
                None => self.emit_tensor_dim(input.value(), i),
            }
        }).collect()
    }

    /// Create a tensor filled with `fill_value`, with shape given by runtime
    /// `index` values. Equivalent to ONNX ConstantOfShape.
    ///
    /// `shape_vals` is a list of runtime `index` values (from `emit_shape_of`
    /// or `arith.constant`). `fill_value` is the scalar fill.
    pub fn emit_constant_of_shape(
        &mut self,
        shape_vals: &[melior::ir::Value<'c, 'c>],
        static_shape: &[Option<u64>],
        fill_value: f64,
        dtype: DType,
    ) -> Tensor<'c> {
        let elem_type = dtype.to_mlir_type(self.context);
        let tensor_type = self.make_tensor_type(static_shape, dtype);

        // tensor.empty with dynamic operands for None dims.
        let dyn_vals: Vec<melior::ir::Value<'c, 'c>> = static_shape
            .iter()
            .enumerate()
            .filter(|(_, d)| d.is_none())
            .map(|(i, _)| shape_vals[i])
            .collect();

        let init: melior::ir::Value = self.block
            .append_operation(
                OperationBuilder::new("tensor.empty", self.location)
                    .add_operands(&dyn_vals)
                    .add_results(&[tensor_type])
                    .build()
                    .expect("tensor.empty constant_of_shape"),
            )
            .result(0)
            .unwrap()
            .into();

        // Scalar fill value.
        let scalar_attr = self.make_scalar_attr(fill_value, dtype);
        let scalar: melior::ir::Value = self.block
            .append_operation(
                OperationBuilder::new("arith.constant", self.location)
                    .add_results(&[elem_type])
                    .add_attributes(&[(Identifier::new(self.context, "value"), scalar_attr)])
                    .build()
                    .expect("arith.constant fill"),
            )
            .result(0)
            .unwrap()
            .into();

        // linalg.fill
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

        let filled: melior::ir::Value = self.block
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
                    .expect("linalg.fill constant_of_shape"),
            )
            .result(0)
            .unwrap()
            .into();

        Tensor::from_value(filled)
    }

    /// Emit a Range tensor: `[start, start+delta, start+2*delta, ...]` up to
    /// (not including) `limit`. All are scalar `index` values.
    ///
    /// Output is a 1-D tensor. If the range size is statically known, the
    /// output has a static dim; otherwise it's dynamic.
    pub fn emit_range(
        &mut self,
        start: melior::ir::Value<'c, 'c>,
        limit: melior::ir::Value<'c, 'c>,
        delta: melior::ir::Value<'c, 'c>,
        output_size: Option<u64>,
        dtype: DType,
    ) -> Tensor<'c> {
        let elem_type = dtype.to_mlir_type(self.context);
        let index_type = melior::ir::Type::parse(self.context, "index").expect("index type");
        let shape = vec![output_size];
        let tensor_type = self.make_tensor_type(&shape, dtype);

        // Compute output size if dynamic: (limit - start + delta - 1) / delta
        // or just use the provided size.
        let mut dyn_vals = Vec::new();
        if output_size.is_none() {
            // Compute runtime size: ceildiv(limit - start, delta)
            let diff: melior::ir::Value = self.block
                .append_operation(
                    OperationBuilder::new("arith.subi", self.location)
                        .add_operands(&[limit, start])
                        .add_results(&[index_type])
                        .build()
                        .expect("arith.subi range"),
                )
                .result(0).unwrap().into();
            let size: melior::ir::Value = self.block
                .append_operation(
                    OperationBuilder::new("arith.ceildivui", self.location)
                        .add_operands(&[diff, delta])
                        .add_results(&[index_type])
                        .build()
                        .expect("arith.ceildivui range"),
                )
                .result(0).unwrap().into();
            dyn_vals.push(size);
        }

        let init: melior::ir::Value = self.block
            .append_operation(
                OperationBuilder::new("tensor.empty", self.location)
                    .add_operands(&dyn_vals)
                    .add_results(&[tensor_type])
                    .build()
                    .expect("tensor.empty range"),
            )
            .result(0).unwrap().into();

        // Cast start and delta from index to element type (they may come from
        // extract_scalar_as_index which returns index values).
        let start_elem = self.emit_index_to_elem(start, dtype);
        let delta_elem = self.emit_index_to_elem(delta, dtype);

        // linalg.generic: body computes start + index * delta
        let identity = identity_map_str(1);
        let indexing_maps = Attribute::parse(
            self.context,
            &format!("[{identity}]"),
        ).expect("range indexing_maps");
        let iterator_types = self.make_iterator_types(1);

        let body_block = Block::new(&[(elem_type, self.location)]);

        // Get iteration index.
        let idx: melior::ir::Value = body_block
            .append_operation(
                OperationBuilder::new("linalg.index", self.location)
                    .add_attributes(&[(
                        Identifier::new(self.context, "dim"),
                        Attribute::parse(self.context, "0 : i64").unwrap(),
                    )])
                    .add_results(&[index_type])
                    .build()
                    .expect("linalg.index"),
            )
            .result(0).unwrap().into();

        // Cast index to element type, then compute start + idx * delta.
        let idx_cast = self.emit_index_to_elem_in_block(&body_block, idx, dtype);
        let prod: melior::ir::Value = body_block
            .append_operation(
                OperationBuilder::new(
                    if matches!(dtype, DType::F32 | DType::F64) { "arith.mulf" } else { "arith.muli" },
                    self.location,
                )
                    .add_operands(&[idx_cast, delta_elem])
                    .add_results(&[elem_type])
                    .build()
                    .expect("range mul"),
            )
            .result(0).unwrap().into();
        let val: melior::ir::Value = body_block
            .append_operation(
                OperationBuilder::new(
                    if matches!(dtype, DType::F32 | DType::F64) { "arith.addf" } else { "arith.addi" },
                    self.location,
                )
                    .add_operands(&[start_elem, prod])
                    .add_results(&[elem_type])
                    .build()
                    .expect("range add"),
            )
            .result(0).unwrap().into();

        body_block.append_operation(
            OperationBuilder::new("linalg.yield", self.location)
                .add_operands(&[val])
                .build()
                .expect("linalg.yield"),
        );
        let body_region = Region::new();
        body_region.append_block(body_block);

        let result: melior::ir::Value = self.block
            .append_operation(
                OperationBuilder::new("linalg.generic", self.location)
                    .add_operands(&[init])
                    .add_results(&[tensor_type])
                    .add_attributes(&[
                        (Identifier::new(self.context, "indexing_maps"), indexing_maps),
                        (Identifier::new(self.context, "iterator_types"), iterator_types),
                        (Identifier::new(self.context, "operandSegmentSizes"),
                         Attribute::parse(self.context, "array<i32: 0, 1>").unwrap()),
                    ])
                    .add_regions([body_region])
                    .build()
                    .expect("linalg.generic range"),
            )
            .result(0).unwrap().into();

        Tensor::from_value(result)
    }

    /// Cast an `index` value to the target element type inside a block body.
    /// Cast an `index` value to the target element type (on the main block).
    fn emit_index_to_elem(
        &mut self,
        idx: melior::ir::Value<'c, 'c>,
        dtype: DType,
    ) -> melior::ir::Value<'c, 'c> {
        let elem_type = dtype.to_mlir_type(self.context);
        let i64_type = melior::ir::Type::parse(self.context, "i64").expect("i64 type");
        let as_i64: melior::ir::Value = self.block
            .append_operation(
                OperationBuilder::new("arith.index_cast", self.location)
                    .add_operands(&[idx])
                    .add_results(&[i64_type])
                    .build()
                    .expect("arith.index_cast"),
            )
            .result(0).unwrap().into();
        match dtype {
            DType::I64 => as_i64,
            DType::I32 => {
                self.block
                    .append_operation(
                        OperationBuilder::new("arith.trunci", self.location)
                            .add_operands(&[as_i64])
                            .add_results(&[elem_type])
                            .build()
                            .expect("arith.trunci"),
                    )
                    .result(0).unwrap().into()
            }
            DType::F32 | DType::F64 => {
                self.block
                    .append_operation(
                        OperationBuilder::new("arith.sitofp", self.location)
                            .add_operands(&[as_i64])
                            .add_results(&[elem_type])
                            .build()
                            .expect("arith.sitofp"),
                    )
                    .result(0).unwrap().into()
            }
        }
    }

    fn emit_index_to_elem_in_block(
        &self,
        block: &Block<'c>,
        idx: melior::ir::Value<'c, 'c>,
        dtype: DType,
    ) -> melior::ir::Value<'c, 'c> {
        let elem_type = dtype.to_mlir_type(self.context);
        let i64_type = melior::ir::Type::parse(self.context, "i64").expect("i64 type");
        // index -> i64
        let as_i64: melior::ir::Value = block
            .append_operation(
                OperationBuilder::new("arith.index_cast", self.location)
                    .add_operands(&[idx])
                    .add_results(&[i64_type])
                    .build()
                    .expect("arith.index_cast"),
            )
            .result(0).unwrap().into();
        match dtype {
            DType::I64 => as_i64,
            DType::I32 => {
                block
                    .append_operation(
                        OperationBuilder::new("arith.trunci", self.location)
                            .add_operands(&[as_i64])
                            .add_results(&[elem_type])
                            .build()
                            .expect("arith.trunci"),
                    )
                    .result(0).unwrap().into()
            }
            DType::F32 | DType::F64 => {
                block
                    .append_operation(
                        OperationBuilder::new("arith.sitofp", self.location)
                            .add_operands(&[as_i64])
                            .add_results(&[elem_type])
                            .build()
                            .expect("arith.sitofp"),
                    )
                    .result(0).unwrap().into()
            }
        }
    }

    /// Emit an `arith.constant` scalar tensor value.
    pub fn emit_arith_constant(&mut self, value: f64, dtype: DType) -> Tensor<'c> {
        let dense_str = match dtype {
            DType::F32 => format!("dense<{:.6e}> : tensor<f32>", value),
            DType::F64 => format!("dense<{:.15e}> : tensor<f64>", value),
            DType::I32 => format!("dense<{}> : tensor<i32>", value as i32),
            DType::I64 => format!("dense<{}> : tensor<i64>", value as i64),
        };
        let dense_attr = Attribute::parse(self.context, &dense_str)
            .expect("dense constant attr");
        let tensor_type = self.make_tensor_type(&[], dtype);
        let result: melior::ir::Value = self.block
            .append_operation(
                OperationBuilder::new("arith.constant", self.location)
                    .add_results(&[tensor_type])
                    .add_attributes(&[(Identifier::new(self.context, "value"), dense_attr)])
                    .build()
                    .expect("arith.constant dense"),
            )
            .result(0).unwrap().into();
        Tensor::from_value(result)
    }

    /// Emit a dense tensor constant from raw data.
    ///
    /// `data_str` is the MLIR dense attribute body, e.g. `"[1.0, 2.0, 3.0]"`
    /// or `"[[1, 2], [3, 4]]"`.
    pub fn emit_dense_constant(
        &mut self,
        data_str: &str,
        shape: &[u64],
        dtype: DType,
    ) -> Tensor<'c> {
        let shape_opt: Vec<Option<u64>> = shape.iter().map(|&d| Some(d)).collect();
        let tensor_type = self.make_tensor_type(&shape_opt, dtype);
        let type_str = tensor_type.to_string();
        let dense_str = format!("dense<{data_str}> : {type_str}");
        let dense_attr = Attribute::parse(self.context, &dense_str)
            .unwrap_or_else(|| panic!("failed to parse dense attr: {dense_str}"));
        let result: melior::ir::Value = self.block
            .append_operation(
                OperationBuilder::new("arith.constant", self.location)
                    .add_results(&[tensor_type])
                    .add_attributes(&[(Identifier::new(self.context, "value"), dense_attr)])
                    .build()
                    .expect("arith.constant dense tensor"),
            )
            .result(0).unwrap().into();
        Tensor::from_value(result)
    }

    /// Build a scalar attribute for the given value and dtype.
    fn make_scalar_attr(&self, value: f64, dtype: DType) -> Attribute<'c> {
        let s = match dtype {
            DType::F32 => format!("{:.6e} : f32", value),
            DType::F64 => format!("{:.15e} : f64", value),
            DType::I32 => format!("{} : i32", value as i32),
            DType::I64 => format!("{} : i64", value as i64),
        };
        Attribute::parse(self.context, &s).expect("scalar attr")
    }
}

// ── Shared utilities ──────────────────────────────────────────────────────────

fn create_context() -> Context {
    let context = Context::new();
    context.attach_diagnostic_handler(|diagnostic| {
        eprintln!("{diagnostic}");
        true
    });
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    register_all_llvm_translations(&context);
    context
}

fn identity_map_str(rank: usize) -> String {
    let dims: Vec<String> = (0..rank).map(|i| format!("d{i}")).collect();
    let dim_list = dims.join(", ");
    format!("affine_map<({dim_list}) -> ({dim_list})>")
}

fn tempfile_dir() -> Option<std::path::PathBuf> {
    std::env::var("TMPDIR")
        .ok()
        .map(std::path::PathBuf::from)
        .or_else(|| Some(std::path::PathBuf::from("/tmp")))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Buffer;

    #[test]
    fn tensor_shape_static() {
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let t = gb.input(&[Some(2), Some(3)], DType::F32);
        assert_eq!(t.shape(), vec![Some(2), Some(3)]);
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.rank(), 2);
    }

    #[test]
    fn tensor_shape_dynamic() {
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let t = gb.input(&[None, Some(12), None, Some(64)], DType::F32);
        assert_eq!(t.shape(), vec![None, Some(12), None, Some(64)]);
        assert_eq!(t.rank(), 4);
    }

    #[test]
    fn identity_graph_compile_and_run() {
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let x = gb.input(&[Some(4)], DType::F32);
        let graph = gb.compile(&[&x]).expect("compile failed");

        let input = Buffer::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], DType::F32);
        let outputs = graph.run(&[&input]);
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].as_slice::<f32>(), &[1.0, 2.0, 3.0, 4.0]);
    }

    // ── Elementwise ops (tasks 2.6, 2.7) ──────────────────────────────────

    #[test]
    fn elementwise_add_static() {
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(4)], DType::F32);
        let b = gb.input(&[Some(4)], DType::F32);
        let c = gb.emit_add(&a, &b);
        let graph = gb.compile(&[&c]).expect("compile add");

        let a_buf = Buffer::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], DType::F32);
        let b_buf = Buffer::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[4], DType::F32);
        let out = graph.run(&[&a_buf, &b_buf]);
        assert_eq!(out[0].as_slice::<f32>(), &[11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn elementwise_sub_mul() {
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(3)], DType::F32);
        let b = gb.input(&[Some(3)], DType::F32);
        let diff = gb.emit_sub(&a, &b);
        let prod = gb.emit_mul(&a, &b);
        let graph = gb.compile(&[&diff, &prod]).expect("compile sub+mul");

        let a_buf = Buffer::from_slice(&[5.0f32, 10.0, 15.0], &[3], DType::F32);
        let b_buf = Buffer::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
        let out = graph.run(&[&a_buf, &b_buf]);
        assert_eq!(out[0].as_slice::<f32>(), &[4.0, 8.0, 12.0]);
        assert_eq!(out[1].as_slice::<f32>(), &[5.0, 20.0, 45.0]);
    }

    #[test]
    fn elementwise_div() {
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(4)], DType::F32);
        let b = gb.input(&[Some(4)], DType::F32);
        let c = gb.emit_div(&a, &b);
        let graph = gb.compile(&[&c]).expect("compile div");

        let a_buf = Buffer::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[4], DType::F32);
        let b_buf = Buffer::from_slice(&[2.0f32, 4.0, 5.0, 8.0], &[4], DType::F32);
        let out = graph.run(&[&a_buf, &b_buf]);
        assert_eq!(out[0].as_slice::<f32>(), &[5.0, 5.0, 6.0, 5.0]);
    }

    #[test]
    fn elementwise_neg_exp_tanh() {
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(2)], DType::F32);
        let neg_a = gb.emit_neg(&a);
        let exp_a = gb.emit_exp(&a);
        let tanh_a = gb.emit_tanh(&a);
        let graph = gb.compile(&[&neg_a, &exp_a, &tanh_a]).expect("compile unary ops");

        let a_buf = Buffer::from_slice(&[1.0f32, 0.0], &[2], DType::F32);
        let out = graph.run(&[&a_buf]);
        assert_eq!(out[0].as_slice::<f32>(), &[-1.0, 0.0]); // neg
        let exp_vals = out[1].as_slice::<f32>();
        assert!((exp_vals[0] - std::f32::consts::E).abs() < 1e-5); // exp(1) ≈ e
        assert!((exp_vals[1] - 1.0).abs() < 1e-5); // exp(0) = 1
        let tanh_vals = out[2].as_slice::<f32>();
        assert!((tanh_vals[0] - 0.7615942).abs() < 1e-5); // tanh(1)
        assert!((tanh_vals[1] - 0.0).abs() < 1e-5); // tanh(0)
    }

    #[test]
    fn elementwise_relu() {
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(4)], DType::F32);
        let r = gb.emit_relu(&a);
        let graph = gb.compile(&[&r]).expect("compile relu");

        let a_buf = Buffer::from_slice(&[-1.0f32, 0.0, 1.0, -0.5], &[4], DType::F32);
        let out = graph.run(&[&a_buf]);
        assert_eq!(out[0].as_slice::<f32>(), &[0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn elementwise_sqrt_reciprocal_rsqrt() {
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(2)], DType::F32);
        let sq = gb.emit_sqrt(&a);
        let rec = gb.emit_reciprocal(&a);
        let rsq = gb.emit_rsqrt(&a);
        let graph = gb.compile(&[&sq, &rec, &rsq]).expect("compile sqrt ops");

        let a_buf = Buffer::from_slice(&[4.0f32, 9.0], &[2], DType::F32);
        let out = graph.run(&[&a_buf]);
        let sqrt_vals = out[0].as_slice::<f32>();
        assert!((sqrt_vals[0] - 2.0).abs() < 1e-5);
        assert!((sqrt_vals[1] - 3.0).abs() < 1e-5);
        let rec_vals = out[1].as_slice::<f32>();
        assert!((rec_vals[0] - 0.25).abs() < 1e-5);
        assert!((rec_vals[1] - 1.0/9.0).abs() < 1e-5);
        let rsq_vals = out[2].as_slice::<f32>();
        assert!((rsq_vals[0] - 0.5).abs() < 1e-5);
        assert!((rsq_vals[1] - 1.0/3.0).abs() < 1e-5);
    }

    #[test]
    fn elementwise_pow() {
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let base = gb.input(&[Some(3)], DType::F32);
        let exp = gb.input(&[Some(3)], DType::F32);
        let result = gb.emit_pow(&base, &exp);
        let graph = gb.compile(&[&result]).expect("compile pow");

        let base_buf = Buffer::from_slice(&[2.0f32, 3.0, 4.0], &[3], DType::F32);
        let exp_buf = Buffer::from_slice(&[3.0f32, 2.0, 0.5], &[3], DType::F32);
        let out = graph.run(&[&base_buf, &exp_buf]);
        let vals = out[0].as_slice::<f32>();
        assert!((vals[0] - 8.0).abs() < 1e-5);  // 2^3
        assert!((vals[1] - 9.0).abs() < 1e-5);  // 3^2
        assert!((vals[2] - 2.0).abs() < 1e-5);  // 4^0.5
    }

    #[test]
    fn elementwise_dynamic_dims_type_inference() {
        // Task 2.9: element-wise ops with `?` dims verify output type.
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[None, Some(768)], DType::F32);
        let b = gb.input(&[None, Some(768)], DType::F32);
        let c = gb.emit_add(&a, &b);
        assert_eq!(c.shape(), vec![None, Some(768)]);
        assert_eq!(c.dtype(), DType::F32);

        let neg_c = gb.emit_neg(&c);
        assert_eq!(neg_c.shape(), vec![None, Some(768)]);

        let exp_c = gb.emit_exp(&c);
        assert_eq!(exp_c.shape(), vec![None, Some(768)]);
    }

    #[test]
    fn broadcast_rank_promotion() {
        // Task 2.10: binary op with [3] + [4,3] => [4,3].
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(3)], DType::F32);            // [3]
        let b = gb.input(&[Some(4), Some(3)], DType::F32);   // [4, 3]
        let c = gb.emit_add(&a, &b);
        assert_eq!(c.shape(), vec![Some(4), Some(3)]);

        let graph = gb.compile(&[&c]).expect("compile broadcast add");

        // a = [1, 2, 3], b = [[10,20,30],[40,50,60],[70,80,90],[100,110,120]]
        let a_buf = Buffer::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
        let b_buf = Buffer::from_slice(
            &[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0],
            &[4, 3],
            DType::F32,
        );
        let out = graph.run(&[&a_buf, &b_buf]);
        assert_eq!(
            out[0].as_slice::<f32>(),
            &[11.0, 22.0, 33.0, 41.0, 52.0, 63.0, 71.0, 82.0, 93.0, 101.0, 112.0, 123.0]
        );
    }

    // ── Matmul and Gemm (tasks 3.1 – 3.4) ────────────────────────────────

    #[test]
    fn matmul_2d_static() {
        // [2,3] x [3,4] -> [2,4] with known values.
        // A = [[1,2,3],[4,5,6]], B = [[1,0,0,0],[0,1,0,0],[0,0,1,0]]
        // => C = [[1,2,3,0],[4,5,6,0]]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(2), Some(3)], DType::F32);
        let b = gb.input(&[Some(3), Some(4)], DType::F32);
        let c = gb.emit_matmul(&a, &b);
        assert_eq!(c.shape(), vec![Some(2), Some(4)]);
        let graph = gb.compile(&[&c]).expect("compile matmul 2d");

        let a_buf = Buffer::from_slice(
            &[1.0f32, 2.0, 3.0,
              4.0, 5.0, 6.0],
            &[2, 3],
            DType::F32,
        );
        // B is identity-ish: [[1,0,0,0],[0,1,0,0],[0,0,1,0]]
        let b_buf = Buffer::from_slice(
            &[1.0f32, 0.0, 0.0, 0.0,
              0.0, 1.0, 0.0, 0.0,
              0.0, 0.0, 1.0, 0.0],
            &[3, 4],
            DType::F32,
        );
        let out = graph.run(&[&a_buf, &b_buf]);
        assert_eq!(out[0].as_slice::<f32>(), &[1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 0.0]);
    }

    #[test]
    fn matmul_2d_known_values() {
        // Simple 2x2 @ 2x2
        // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(2), Some(2)], DType::F32);
        let b = gb.input(&[Some(2), Some(2)], DType::F32);
        let c = gb.emit_matmul(&a, &b);
        let graph = gb.compile(&[&c]).expect("compile matmul 2x2");

        let a_buf = Buffer::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], DType::F32);
        let b_buf = Buffer::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], DType::F32);
        let out = graph.run(&[&a_buf, &b_buf]);
        let vals = out[0].as_slice::<f32>();
        assert!((vals[0] - 19.0).abs() < 1e-4);
        assert!((vals[1] - 22.0).abs() < 1e-4);
        assert!((vals[2] - 43.0).abs() < 1e-4);
        assert!((vals[3] - 50.0).abs() < 1e-4);
    }

    #[test]
    fn batch_matmul_3d_static() {
        // [2,3,4] x [2,4,5] -> [2,3,5]
        // Use identity-like B to verify correctness:
        // batch 0: A0 = ones(3,4), B0 = [[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0]]
        //   => C0 = A0 @ B0 = [[1,1,1,1,0],[1,1,1,1,0],[1,1,1,1,0]]
        // batch 1: A1 = eye_3x4, B1 = all_ones(4,5)
        //   => C1 = A1 @ B1 = [[1,1,1,1,1],[1,1,1,1,1],[0,0,0,0,0]]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(2), Some(3), Some(4)], DType::F32);
        let b = gb.input(&[Some(2), Some(4), Some(5)], DType::F32);
        let c = gb.emit_matmul(&a, &b);
        assert_eq!(c.shape(), vec![Some(2), Some(3), Some(5)]);
        let graph = gb.compile(&[&c]).expect("compile batch_matmul 3d");

        // batch 0: A0 = ones(3,4)
        let a0 = vec![1.0f32; 12];
        // batch 0: B0 = partial identity (4x5, first 4 rows = I4 with 5th col=0)
        let b0: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.0,
        ];
        // batch 1: A1 = eye-ish (3x4, first 3 rows partial identity)
        let a1: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ];
        // batch 1: B1 = all ones (4x5)
        let b1 = vec![1.0f32; 20];

        let a_data: Vec<f32> = a0.iter().chain(a1.iter()).copied().collect();
        let b_data: Vec<f32> = b0.iter().chain(b1.iter()).copied().collect();

        let a_buf = Buffer::from_slice(&a_data, &[2, 3, 4], DType::F32);
        let b_buf = Buffer::from_slice(&b_data, &[2, 4, 5], DType::F32);
        let out = graph.run(&[&a_buf, &b_buf]);
        let vals = out[0].as_slice::<f32>();

        // batch 0 result (3x5): ones @ partial_identity = [[1,1,1,1,0],[1,1,1,1,0],[1,1,1,1,0]]
        let expected_b0 = [1.0f32, 1.0, 1.0, 1.0, 0.0,
                            1.0,   1.0, 1.0, 1.0, 0.0,
                            1.0,   1.0, 1.0, 1.0, 0.0];
        for (i, &exp) in expected_b0.iter().enumerate() {
            assert!((vals[i] - exp).abs() < 1e-4, "batch0[{i}]: got {}, expected {}", vals[i], exp);
        }
        // batch 1 result (3x5): eye-ish @ all_ones
        // row0 = [1,0,0,0] @ all_ones = sum of col0 = [1,1,1,1,1]
        // row1 = [0,1,0,0] @ all_ones = [1,1,1,1,1]
        // row2 = [0,0,1,0] @ all_ones = [1,1,1,1,1]
        let expected_b1 = [1.0f32, 1.0, 1.0, 1.0, 1.0,
                            1.0,   1.0, 1.0, 1.0, 1.0,
                            1.0,   1.0, 1.0, 1.0, 1.0];
        for (i, &exp) in expected_b1.iter().enumerate() {
            let idx = 15 + i;
            assert!((vals[idx] - exp).abs() < 1e-4, "batch1[{i}]: got {}, expected {}", vals[idx], exp);
        }
    }

    #[test]
    fn matmul_dynamic_batch_dim() {
        // [?,3,4] x [?,4,5] -> [?,3,5] — compile with dynamic batch dim.
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[None, Some(3), Some(4)], DType::F32);
        let b = gb.input(&[None, Some(4), Some(5)], DType::F32);
        let c = gb.emit_matmul(&a, &b);
        // Output batch dim must be dynamic.
        assert_eq!(c.shape(), vec![None, Some(3), Some(5)]);

        // Compile and run with batch=1 to verify correctness.
        let graph = gb.compile(&[&c]).expect("compile dynamic batch matmul");

        // batch=1: A = ones(1,3,4), B = partial identity (1,4,5)
        let a_data = vec![1.0f32; 12];
        let b_data: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.0,
        ];
        let a_buf = Buffer::from_slice(&a_data, &[1, 3, 4], DType::F32);
        let b_buf = Buffer::from_slice(&b_data, &[1, 4, 5], DType::F32);
        // Output shape is [?,3,5]. Provide concrete shape [1,3,5] to run_dynamic.
        let out = graph.run_dynamic(&[&a_buf, &b_buf], &[crate::Shape(vec![1, 3, 5])]);
        // ones @ partial_identity = [[1,1,1,1,0], ...]
        let vals = out[0].as_slice::<f32>();
        let expected = [1.0f32, 1.0, 1.0, 1.0, 0.0,
                         1.0,   1.0, 1.0, 1.0, 0.0,
                         1.0,   1.0, 1.0, 1.0, 0.0];
        for (i, &exp) in expected.iter().enumerate() {
            assert!((vals[i] - exp).abs() < 1e-4, "[{i}]: got {}, expected {}", vals[i], exp);
        }
    }

    #[test]
    fn gemm_transpose_b() {
        // Gemm with transB=1, alpha=1, beta=1.
        // A = [[1,2],[3,4]]  shape [2,2]
        // B = [[5,7],[6,8]]  shape [2,2] — after transposing: [[5,6],[7,8]]
        // So A @ B^T = [[1,2],[3,4]] @ [[5,6],[7,8]]^T
        //            = [[1,2],[3,4]] @ [[5,7],[6,8]] = [[17,23],[39,53]]
        // Wait — B is stored as [N, K] with transB=1, so we transpose to [K, N].
        // Let B_stored = [[5,6],[7,8]], B_T = [[5,7],[6,8]]
        // A @ B_T = [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
        // C = zeros(2,2), beta=0 so result = A @ B_T + 0*C = [[19,22],[43,50]]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(2), Some(2)], DType::F32);
        let b = gb.input(&[Some(2), Some(2)], DType::F32);
        let c = gb.input(&[Some(2), Some(2)], DType::F32);
        let result = gb.emit_gemm(&a, &b, &c, 1.0, 0.0, false, true);
        let graph = gb.compile(&[&result]).expect("compile gemm transB");

        let a_buf = Buffer::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], DType::F32);
        let b_buf = Buffer::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], DType::F32);
        let c_buf = Buffer::from_slice(&[0.0f32; 4], &[2, 2], DType::F32);
        let out = graph.run(&[&a_buf, &b_buf, &c_buf]);
        let vals = out[0].as_slice::<f32>();
        // A @ B^T: B^T = [[5,7],[6,8]]
        // row0: [1,2] @ [[5,7],[6,8]] = [1*5+2*6, 1*7+2*8] = [17, 23]
        // row1: [3,4] @ [[5,7],[6,8]] = [3*5+4*6, 3*7+4*8] = [39, 53]
        assert!((vals[0] - 17.0).abs() < 1e-4, "vals[0]={}", vals[0]);
        assert!((vals[1] - 23.0).abs() < 1e-4, "vals[1]={}", vals[1]);
        assert!((vals[2] - 39.0).abs() < 1e-4, "vals[2]={}", vals[2]);
        assert!((vals[3] - 53.0).abs() < 1e-4, "vals[3]={}", vals[3]);
    }

    #[test]
    fn gemm_with_alpha_beta_c() {
        // alpha=2, beta=1, no transposes.
        // A = [[1,0],[0,1]], B = [[2,3],[4,5]], C = [[1,1],[1,1]]
        // A@B = [[2,3],[4,5]]
        // 2*(A@B) + 1*C = [[4,6],[8,10]] + [[1,1],[1,1]] = [[5,7],[9,11]]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(2), Some(2)], DType::F32);
        let b = gb.input(&[Some(2), Some(2)], DType::F32);
        let c = gb.input(&[Some(2), Some(2)], DType::F32);
        let result = gb.emit_gemm(&a, &b, &c, 2.0, 1.0, false, false);
        let graph = gb.compile(&[&result]).expect("compile gemm alpha_beta");

        let a_buf = Buffer::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2], DType::F32);
        let b_buf = Buffer::from_slice(&[2.0f32, 3.0, 4.0, 5.0], &[2, 2], DType::F32);
        let c_buf = Buffer::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[2, 2], DType::F32);
        let out = graph.run(&[&a_buf, &b_buf, &c_buf]);
        let vals = out[0].as_slice::<f32>();
        assert!((vals[0] - 5.0).abs() < 1e-4, "vals[0]={}", vals[0]);
        assert!((vals[1] - 7.0).abs() < 1e-4, "vals[1]={}", vals[1]);
        assert!((vals[2] - 9.0).abs() < 1e-4, "vals[2]={}", vals[2]);
        assert!((vals[3] - 11.0).abs() < 1e-4, "vals[3]={}", vals[3]);
    }

    #[test]
    fn broadcast_size_one_expansion() {
        // [4,1] + [4,3] => [4,3] — same rank, size-1 broadcast.
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(4), Some(1)], DType::F32);   // [4, 1]
        let b = gb.input(&[Some(4), Some(3)], DType::F32);   // [4, 3]
        let c = gb.emit_add(&a, &b);
        assert_eq!(c.shape(), vec![Some(4), Some(3)]);

        let graph = gb.compile(&[&c]).expect("compile size-1 broadcast");

        let a_buf = Buffer::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4, 1], DType::F32);
        let b_buf = Buffer::from_slice(
            &[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0],
            &[4, 3],
            DType::F32,
        );
        let out = graph.run(&[&a_buf, &b_buf]);
        assert_eq!(
            out[0].as_slice::<f32>(),
            &[11.0, 21.0, 31.0, 42.0, 52.0, 62.0, 73.0, 83.0, 93.0, 104.0, 114.0, 124.0]
        );
    }

    // ── Reductions and Softmax (tasks 4.1 – 4.5) ─────────────────────────

    #[test]
    fn reduce_sum_axis0() {
        // tensor<3x4xf32> reduced along axis 0 -> shape [4].
        // Input: rows = [1,2,3,4], [5,6,7,8], [9,10,11,12]
        // Sum along rows: [15, 18, 21, 24]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(3), Some(4)], DType::F32);
        let r = gb.emit_reduce_sum(&a, 0, false);
        assert_eq!(r.shape(), vec![Some(4)]);
        let graph = gb.compile(&[&r]).expect("compile reduce_sum axis0");

        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let a_buf = Buffer::from_slice(&data, &[3, 4], DType::F32);
        let out = graph.run(&[&a_buf]);
        let vals = out[0].as_slice::<f32>();
        assert_eq!(vals.len(), 4);
        assert!((vals[0] - 15.0).abs() < 1e-5, "vals[0]={}", vals[0]);
        assert!((vals[1] - 18.0).abs() < 1e-5, "vals[1]={}", vals[1]);
        assert!((vals[2] - 21.0).abs() < 1e-5, "vals[2]={}", vals[2]);
        assert!((vals[3] - 24.0).abs() < 1e-5, "vals[3]={}", vals[3]);
    }

    #[test]
    fn reduce_sum_keepdim() {
        // tensor<3x4xf32> reduced along axis 0, keepdim=true -> shape [1,4].
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(3), Some(4)], DType::F32);
        let r = gb.emit_reduce_sum(&a, 0, true);
        assert_eq!(r.shape(), vec![Some(1), Some(4)]);
        let graph = gb.compile(&[&r]).expect("compile reduce_sum keepdim");

        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let a_buf = Buffer::from_slice(&data, &[3, 4], DType::F32);
        let out = graph.run(&[&a_buf]);
        let vals = out[0].as_slice::<f32>();
        assert_eq!(vals.len(), 4);
        assert!((vals[0] - 15.0).abs() < 1e-5);
        assert!((vals[1] - 18.0).abs() < 1e-5);
        assert!((vals[2] - 21.0).abs() < 1e-5);
        assert!((vals[3] - 24.0).abs() < 1e-5);
    }

    #[test]
    fn reduce_sum_axis1() {
        // tensor<3x4xf32> reduced along axis 1 -> shape [3].
        // Sum along columns: row0=10, row1=26, row2=42
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(3), Some(4)], DType::F32);
        let r = gb.emit_reduce_sum(&a, 1, false);
        assert_eq!(r.shape(), vec![Some(3)]);
        let graph = gb.compile(&[&r]).expect("compile reduce_sum axis1");

        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let a_buf = Buffer::from_slice(&data, &[3, 4], DType::F32);
        let out = graph.run(&[&a_buf]);
        let vals = out[0].as_slice::<f32>();
        assert_eq!(vals.len(), 3);
        assert!((vals[0] - 10.0).abs() < 1e-5, "vals[0]={}", vals[0]);
        assert!((vals[1] - 26.0).abs() < 1e-5, "vals[1]={}", vals[1]);
        assert!((vals[2] - 42.0).abs() < 1e-5, "vals[2]={}", vals[2]);
    }

    #[test]
    fn reduce_sum_negative_axis() {
        // Negative axis: axis=-1 on [3,4] is equivalent to axis=1.
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(3), Some(4)], DType::F32);
        let r = gb.emit_reduce_sum(&a, -1, false);
        assert_eq!(r.shape(), vec![Some(3)]);
        let graph = gb.compile(&[&r]).expect("compile reduce_sum negative axis");

        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let a_buf = Buffer::from_slice(&data, &[3, 4], DType::F32);
        let out = graph.run(&[&a_buf]);
        let vals = out[0].as_slice::<f32>();
        assert!((vals[0] - 10.0).abs() < 1e-5);
        assert!((vals[1] - 26.0).abs() < 1e-5);
        assert!((vals[2] - 42.0).abs() < 1e-5);
    }

    #[test]
    fn reduce_max_static() {
        // tensor<2x3xf32> reduced along axis 1 -> shape [2].
        // row0 = [3, 1, 4], max = 4
        // row1 = [1, 5, 9], max = 9
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(2), Some(3)], DType::F32);
        let r = gb.emit_reduce_max(&a, 1, false);
        assert_eq!(r.shape(), vec![Some(2)]);
        let graph = gb.compile(&[&r]).expect("compile reduce_max");

        let a_buf = Buffer::from_slice(&[3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0], &[2, 3], DType::F32);
        let out = graph.run(&[&a_buf]);
        let vals = out[0].as_slice::<f32>();
        assert!((vals[0] - 4.0).abs() < 1e-5, "vals[0]={}", vals[0]);
        assert!((vals[1] - 9.0).abs() < 1e-5, "vals[1]={}", vals[1]);
    }

    #[test]
    fn reduce_max_keepdim() {
        // keepdim=true: [2,3] -> [2,1]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(2), Some(3)], DType::F32);
        let r = gb.emit_reduce_max(&a, 1, true);
        assert_eq!(r.shape(), vec![Some(2), Some(1)]);
        let graph = gb.compile(&[&r]).expect("compile reduce_max keepdim");

        let a_buf = Buffer::from_slice(&[3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0], &[2, 3], DType::F32);
        let out = graph.run(&[&a_buf]);
        let vals = out[0].as_slice::<f32>();
        assert_eq!(vals.len(), 2);
        assert!((vals[0] - 4.0).abs() < 1e-5);
        assert!((vals[1] - 9.0).abs() < 1e-5);
    }

    #[test]
    fn reduce_mean_static() {
        // tensor<2x4xf32> mean along axis=1 -> shape [2].
        // row0 = [1,2,3,4], mean = 2.5
        // row1 = [5,6,7,8], mean = 6.5
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(2), Some(4)], DType::F32);
        let r = gb.emit_reduce_mean(&a, &[1], false);
        assert_eq!(r.shape(), vec![Some(2)]);
        let graph = gb.compile(&[&r]).expect("compile reduce_mean");

        let a_buf = Buffer::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 4],
            DType::F32,
        );
        let out = graph.run(&[&a_buf]);
        let vals = out[0].as_slice::<f32>();
        assert_eq!(vals.len(), 2);
        assert!((vals[0] - 2.5).abs() < 1e-5, "vals[0]={}", vals[0]);
        assert!((vals[1] - 6.5).abs() < 1e-5, "vals[1]={}", vals[1]);
    }

    #[test]
    fn reduce_mean_multiple_axes() {
        // tensor<2x3x4xf32> mean along axes [1,2] -> shape [2].
        // All values = 1.0 -> mean = 1.0 for each batch.
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(2), Some(3), Some(4)], DType::F32);
        let r = gb.emit_reduce_mean(&a, &[1, 2], false);
        assert_eq!(r.shape(), vec![Some(2)]);
        let graph = gb.compile(&[&r]).expect("compile reduce_mean multi-axis");

        let data = vec![1.0f32; 24];
        let a_buf = Buffer::from_slice(&data, &[2, 3, 4], DType::F32);
        let out = graph.run(&[&a_buf]);
        let vals = out[0].as_slice::<f32>();
        assert_eq!(vals.len(), 2);
        assert!((vals[0] - 1.0).abs() < 1e-5, "vals[0]={}", vals[0]);
        assert!((vals[1] - 1.0).abs() < 1e-5, "vals[1]={}", vals[1]);
    }

    #[test]
    fn softmax_static() {
        // softmax([1, 2, 3]) ≈ [0.0900, 0.2447, 0.6652]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(1), Some(3)], DType::F32); // [1,3] for axis=1
        let r = gb.emit_softmax(&a, 1);
        assert_eq!(r.shape(), vec![Some(1), Some(3)]);
        let graph = gb.compile(&[&r]).expect("compile softmax");

        let a_buf = Buffer::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3], DType::F32);
        let out = graph.run(&[&a_buf]);
        let vals = out[0].as_slice::<f32>();
        assert_eq!(vals.len(), 3);
        assert!((vals[0] - 0.0900).abs() < 1e-3, "vals[0]={}", vals[0]);
        assert!((vals[1] - 0.2447).abs() < 1e-3, "vals[1]={}", vals[1]);
        assert!((vals[2] - 0.6652).abs() < 1e-3, "vals[2]={}", vals[2]);
        // Also verify they sum to 1.
        let sum: f32 = vals.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum={sum}");
    }

    #[test]
    fn softmax_2d_axis0() {
        // softmax along axis 0 of [[1,1],[2,2]] -> [[sigma,sigma],[1-sigma,1-sigma]]
        // sigma = e^1/(e^1+e^2) ≈ 0.2689
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(2), Some(2)], DType::F32);
        let r = gb.emit_softmax(&a, 0);
        assert_eq!(r.shape(), vec![Some(2), Some(2)]);
        let graph = gb.compile(&[&r]).expect("compile softmax axis0");

        let a_buf = Buffer::from_slice(&[1.0f32, 1.0, 2.0, 2.0], &[2, 2], DType::F32);
        let out = graph.run(&[&a_buf]);
        let vals = out[0].as_slice::<f32>();
        let sigma = 1.0f32 / (1.0 + std::f32::consts::E);
        assert!((vals[0] - sigma).abs() < 1e-5, "vals[0]={}", vals[0]);
        assert!((vals[1] - sigma).abs() < 1e-5, "vals[1]={}", vals[1]);
        assert!((vals[2] - (1.0 - sigma)).abs() < 1e-5, "vals[2]={}", vals[2]);
        assert!((vals[3] - (1.0 - sigma)).abs() < 1e-5, "vals[3]={}", vals[3]);
    }

    #[test]
    fn reduce_sum_dynamic() {
        // tensor<?x4xf32> reduce along axis 0 -> [4].
        // Only checks that compilation and execution succeed with dynamic dim.
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[None, Some(4)], DType::F32);
        let r = gb.emit_reduce_sum(&a, 0, false);
        assert_eq!(r.shape(), vec![Some(4)]);
        let graph = gb.compile(&[&r]).expect("compile reduce_sum dynamic");

        // Run with 3 rows.
        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let a_buf = Buffer::from_slice(&data, &[3, 4], DType::F32);
        let out = graph.run_dynamic(&[&a_buf], &[crate::Shape(vec![4])]);
        let vals = out[0].as_slice::<f32>();
        assert_eq!(vals.len(), 4);
        assert!((vals[0] - 15.0).abs() < 1e-5);
        assert!((vals[1] - 18.0).abs() < 1e-5);
        assert!((vals[2] - 21.0).abs() < 1e-5);
        assert!((vals[3] - 24.0).abs() < 1e-5);
    }

    // ── Phase 5: Shape Manipulation Ops (tasks 5.1 – 5.11) ────────────────

    #[test]
    fn reshape_static() {
        // [2, 6] -> [3, 4]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(2), Some(6)], DType::F32);
        let r = gb.emit_reshape(&a, &[3, 4]);
        assert_eq!(r.shape(), vec![Some(3), Some(4)]);
        let graph = gb.compile(&[&r]).expect("compile reshape static");

        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let a_buf = Buffer::from_slice(&data, &[2, 6], DType::F32);
        let out = graph.run(&[&a_buf]);
        // reshape is a view-like op — elements in same order
        assert_eq!(out[0].as_slice::<f32>(), data.as_slice());
    }

    #[test]
    fn reshape_infer_dim() {
        // [2, 6] -> [-1, 3] — inferred first dim should be 4
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(2), Some(6)], DType::F32);
        let r = gb.emit_reshape(&a, &[-1, 3]);
        // inferred dim is now resolved statically: 12 / 3 = 4
        assert_eq!(r.shape(), vec![Some(4), Some(3)]);
        let graph = gb.compile(&[&r]).expect("compile reshape infer dim");

        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let a_buf = Buffer::from_slice(&data, &[2, 6], DType::F32);
        // output is [4, 3] — now statically inferred
        let out = graph.run(&[&a_buf]);
        assert_eq!(out[0].as_slice::<f32>(), data.as_slice());
    }

    #[test]
    fn reshape_1d_to_3d() {
        // [24] -> [2, 3, 4]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(24)], DType::F32);
        let r = gb.emit_reshape(&a, &[2, 3, 4]);
        assert_eq!(r.shape(), vec![Some(2), Some(3), Some(4)]);
        let graph = gb.compile(&[&r]).expect("compile reshape 1d->3d");

        let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
        let a_buf = Buffer::from_slice(&data, &[24], DType::F32);
        let out = graph.run(&[&a_buf]);
        assert_eq!(out[0].as_slice::<f32>(), data.as_slice());
    }

    #[test]
    fn transpose_2d() {
        // [2, 3] -> [3, 2] with perms [1, 0]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(2), Some(3)], DType::F32);
        let r = gb.emit_transpose(&a, &[1, 0]);
        assert_eq!(r.shape(), vec![Some(3), Some(2)]);
        let graph = gb.compile(&[&r]).expect("compile transpose 2d");

        // [[1, 2, 3], [4, 5, 6]] -> [[1, 4], [2, 5], [3, 6]]
        let a_buf = Buffer::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
        let out = graph.run(&[&a_buf]);
        assert_eq!(out[0].as_slice::<f32>(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn transpose_3d() {
        // [2, 3, 4] -> [4, 2, 3] with perms [2, 0, 1]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(2), Some(3), Some(4)], DType::F32);
        let r = gb.emit_transpose(&a, &[2, 0, 1]);
        assert_eq!(r.shape(), vec![Some(4), Some(2), Some(3)]);

        let graph = gb.compile(&[&r]).expect("compile transpose 3d");

        // Verify shape only — correctness from type checking.
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let a_buf = Buffer::from_slice(&data, &[2, 3, 4], DType::F32);
        let out = graph.run(&[&a_buf]);
        // output[k, i, j] = input[i, j, k]
        // out[0,0,0]=input[0,0,0]=0, out[0,0,1]=input[0,1,0]=4, out[1,0,0]=input[0,0,1]=1
        let vals = out[0].as_slice::<f32>();
        assert!((vals[0] - 0.0).abs() < 1e-5);  // out[0,0,0] = input[0,0,0]
        assert!((vals[1] - 4.0).abs() < 1e-5);  // out[0,0,1] = input[0,1,0]
        assert!((vals[2] - 8.0).abs() < 1e-5);  // out[0,0,2] = input[0,2,0]
    }

    #[test]
    fn concat_axis0() {
        // concat([[1,2,3], [4,5,6]], axis=0) = [1,2,3,4,5,6]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(3)], DType::F32);
        let b = gb.input(&[Some(3)], DType::F32);
        let r = gb.emit_concat(&[a, b], 0);
        assert_eq!(r.shape(), vec![Some(6)]);
        let graph = gb.compile(&[&r]).expect("compile concat axis0");

        let a_buf = Buffer::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
        let b_buf = Buffer::from_slice(&[4.0f32, 5.0, 6.0], &[3], DType::F32);
        let out = graph.run(&[&a_buf, &b_buf]);
        assert_eq!(out[0].as_slice::<f32>(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn concat_axis1_2d() {
        // concat([[1,2],[3,4]], [[5,6],[7,8]], axis=1) = [[1,2,5,6],[3,4,7,8]]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(2), Some(2)], DType::F32);
        let b = gb.input(&[Some(2), Some(2)], DType::F32);
        let r = gb.emit_concat(&[a, b], 1);
        assert_eq!(r.shape(), vec![Some(2), Some(4)]);
        let graph = gb.compile(&[&r]).expect("compile concat axis1 2d");

        let a_buf = Buffer::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], DType::F32);
        let b_buf = Buffer::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], DType::F32);
        let out = graph.run(&[&a_buf, &b_buf]);
        assert_eq!(out[0].as_slice::<f32>(), &[1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
    }

    #[test]
    fn slice_static() {
        // slice [0..10] -> [2..7] step 1
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(10)], DType::F32);
        let r = gb.emit_slice(&a, &[2], &[7], &[0], &[1]);
        assert_eq!(r.shape(), vec![Some(5)]);
        let graph = gb.compile(&[&r]).expect("compile slice static");

        let data: Vec<f32> = (0..10).map(|x| x as f32).collect();
        let a_buf = Buffer::from_slice(&data, &[10], DType::F32);
        let out = graph.run(&[&a_buf]);
        assert_eq!(out[0].as_slice::<f32>(), &[2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn slice_with_step() {
        // slice [0..10] -> [0..10] step 2  -> [0, 2, 4, 6, 8]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(10)], DType::F32);
        let r = gb.emit_slice(&a, &[0], &[10], &[0], &[2]);
        assert_eq!(r.shape(), vec![Some(5)]);
        let graph = gb.compile(&[&r]).expect("compile slice step2");

        let data: Vec<f32> = (0..10).map(|x| x as f32).collect();
        let a_buf = Buffer::from_slice(&data, &[10], DType::F32);
        let out = graph.run(&[&a_buf]);
        assert_eq!(out[0].as_slice::<f32>(), &[0.0, 2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn slice_2d_axis1() {
        // 2D slice: [3,5] slice axis=1 from 1 to 4 step 1 -> [3,3]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(3), Some(5)], DType::F32);
        let r = gb.emit_slice(&a, &[1], &[4], &[1], &[1]);
        assert_eq!(r.shape(), vec![Some(3), Some(3)]);
        let graph = gb.compile(&[&r]).expect("compile slice 2d axis1");

        // input: [[0..4], [5..9], [10..14]]
        let data: Vec<f32> = (0..15).map(|x| x as f32).collect();
        let a_buf = Buffer::from_slice(&data, &[3, 5], DType::F32);
        let out = graph.run(&[&a_buf]);
        // rows: [1,2,3], [6,7,8], [11,12,13]
        assert_eq!(out[0].as_slice::<f32>(), &[1.0, 2.0, 3.0, 6.0, 7.0, 8.0, 11.0, 12.0, 13.0]);
    }

    #[test]
    fn gather_axis0() {
        // data = [[1,2],[3,4],[5,6]], indices = [2, 0, 1], axis=0
        // output[i] = data[indices[i]] = [[5,6],[1,2],[3,4]]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let data = gb.input(&[Some(3), Some(2)], DType::F32);
        let indices = gb.input(&[Some(3)], DType::I64);
        let r = gb.emit_gather(&data, &indices, 0);
        // output shape: indices.shape + data.shape[1..] = [3, 2]
        assert_eq!(r.shape(), vec![Some(3), Some(2)]);
        let graph = gb.compile(&[&r]).expect("compile gather axis0");

        let data_buf = Buffer::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[3, 2],
            DType::F32,
        );
        let idx_buf = Buffer::from_slice(&[2i64, 0, 1], &[3], DType::I64);
        let out = graph.run(&[&data_buf, &idx_buf]);
        assert_eq!(out[0].as_slice::<f32>(), &[5.0, 6.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn where_select() {
        // cond = [1, 0, 1, 0], x = [10,20,30,40], y = [1,2,3,4]
        // result = [10, 2, 30, 4]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let cond = gb.input(&[Some(4)], DType::I64);
        let x = gb.input(&[Some(4)], DType::F32);
        let y = gb.input(&[Some(4)], DType::F32);
        let r = gb.emit_where(&cond, &x, &y);
        assert_eq!(r.shape(), vec![Some(4)]);
        let graph = gb.compile(&[&r]).expect("compile where");

        let cond_buf = Buffer::from_slice(&[1i64, 0, 1, 0], &[4], DType::I64);
        let x_buf = Buffer::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[4], DType::F32);
        let y_buf = Buffer::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], DType::F32);
        let out = graph.run(&[&cond_buf, &x_buf, &y_buf]);
        assert_eq!(out[0].as_slice::<f32>(), &[10.0, 2.0, 30.0, 4.0]);
    }

    #[test]
    fn cast_i32_to_f32() {
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(4)], DType::I32);
        let r = gb.emit_cast(&a, DType::F32);
        assert_eq!(r.dtype(), DType::F32);
        assert_eq!(r.shape(), vec![Some(4)]);
        let graph = gb.compile(&[&r]).expect("compile cast i32->f32");

        let a_buf = Buffer::from_slice(&[1i32, 2, 3, 4], &[4], DType::I32);
        let out = graph.run(&[&a_buf]);
        assert_eq!(out[0].as_slice::<f32>(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn cast_f32_to_i32() {
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(4)], DType::F32);
        let r = gb.emit_cast(&a, DType::I32);
        assert_eq!(r.dtype(), DType::I32);
        let graph = gb.compile(&[&r]).expect("compile cast f32->i32");

        let a_buf = Buffer::from_slice(&[1.7f32, 2.2, -3.9, 4.0], &[4], DType::F32);
        let out = graph.run(&[&a_buf]);
        // fptosi truncates toward zero
        assert_eq!(out[0].as_slice::<i32>(), &[1, 2, -3, 4]);
    }

    #[test]
    fn cast_i64_to_i32() {
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(3)], DType::I64);
        let r = gb.emit_cast(&a, DType::I32);
        assert_eq!(r.dtype(), DType::I32);
        let graph = gb.compile(&[&r]).expect("compile cast i64->i32");

        let a_buf = Buffer::from_slice(&[100i64, 200, 300], &[3], DType::I64);
        let out = graph.run(&[&a_buf]);
        assert_eq!(out[0].as_slice::<i32>(), &[100, 200, 300]);
    }

    #[test]
    fn unsqueeze_and_squeeze() {
        // unsqueeze [3, 4] at axis 0 -> [1, 3, 4]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(3), Some(4)], DType::F32);
        let u = gb.emit_unsqueeze(&a, &[0]);
        assert_eq!(u.shape(), vec![Some(1), Some(3), Some(4)]);

        // squeeze [1, 3, 4] at axis 0 -> [3, 4]
        let s = gb.emit_squeeze(&u, &[0]);
        assert_eq!(s.shape(), vec![Some(3), Some(4)]);

        let graph = gb.compile(&[&s]).expect("compile unsqueeze+squeeze");

        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let a_buf = Buffer::from_slice(&data, &[3, 4], DType::F32);
        let out = graph.run(&[&a_buf]);
        assert_eq!(out[0].as_slice::<f32>(), data.as_slice());
    }

    #[test]
    fn unsqueeze_middle_axis() {
        // [2, 3] -> unsqueeze axis=1 -> [2, 1, 3]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(2), Some(3)], DType::F32);
        let u = gb.emit_unsqueeze(&a, &[1]);
        assert_eq!(u.shape(), vec![Some(2), Some(1), Some(3)]);
        let graph = gb.compile(&[&u]).expect("compile unsqueeze middle");

        let data: Vec<f32> = (1..=6).map(|x| x as f32).collect();
        let a_buf = Buffer::from_slice(&data, &[2, 3], DType::F32);
        let out = graph.run(&[&a_buf]);
        assert_eq!(out[0].as_slice::<f32>(), data.as_slice());
    }

    #[test]
    fn flatten_axis1() {
        // [2, 3, 4] -> flatten(axis=1) -> [2, 12]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(2), Some(3), Some(4)], DType::F32);
        let f = gb.emit_flatten(&a, 1);
        assert_eq!(f.shape(), vec![Some(2), Some(12)]);
        let graph = gb.compile(&[&f]).expect("compile flatten axis1");

        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let a_buf = Buffer::from_slice(&data, &[2, 3, 4], DType::F32);
        let out = graph.run(&[&a_buf]);
        assert_eq!(out[0].as_slice::<f32>(), data.as_slice());
    }

    #[test]
    fn split_equal_parts() {
        // [6] split into [2, 2, 2] -> three tensors of size 2
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(6)], DType::F32);
        let parts = gb.emit_split(&a, 0, &[2, 2, 2]);
        assert_eq!(parts.len(), 3);
        for p in &parts { assert_eq!(p.shape(), vec![Some(2)]); }

        let graph = gb.compile(&parts.iter().collect::<Vec<_>>())
            .expect("compile split");

        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a_buf = Buffer::from_slice(&data, &[6], DType::F32);
        let out = graph.run(&[&a_buf]);
        assert_eq!(out[0].as_slice::<f32>(), &[1.0, 2.0]);
        assert_eq!(out[1].as_slice::<f32>(), &[3.0, 4.0]);
        assert_eq!(out[2].as_slice::<f32>(), &[5.0, 6.0]);
    }

    #[test]
    fn split_2d_axis0() {
        // [4, 3] split into [1, 3] along axis 0 -> shapes [1,3] and [3,3]
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(4), Some(3)], DType::F32);
        let parts = gb.emit_split(&a, 0, &[1, 3]);
        assert_eq!(parts[0].shape(), vec![Some(1), Some(3)]);
        assert_eq!(parts[1].shape(), vec![Some(3), Some(3)]);

        let graph = gb.compile(&parts.iter().collect::<Vec<_>>())
            .expect("compile split 2d");

        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let a_buf = Buffer::from_slice(&data, &[4, 3], DType::F32);
        let out = graph.run(&[&a_buf]);
        assert_eq!(out[0].as_slice::<f32>(), &[1.0, 2.0, 3.0]);
        assert_eq!(out[1].as_slice::<f32>(), &[4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    }

    // ── Spatial ops (task 6) ──────────────────────────────────────────────────

    /// conv2d with no padding: 1x1x4x4 input, 1x1x3x3 kernel, stride=1 → 1x1x2x2 output.
    ///
    /// Input (row-major):
    ///   [[1, 2, 3, 4],
    ///    [5, 6, 7, 8],
    ///    [9,10,11,12],
    ///    [13,14,15,16]]
    ///
    /// Kernel (all-ones 3x3): each output pixel is the sum of a 3x3 window.
    /// out[0,0] = 1+2+3+5+6+7+9+10+11 = 54
    /// out[0,1] = 2+3+4+6+7+8+10+11+12 = 63
    /// out[1,0] = 5+6+7+9+10+11+13+14+15 = 90
    /// out[1,1] = 6+7+8+10+11+12+14+15+16 = 99
    #[test]
    fn conv2d_no_padding() {
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let input = gb.input(&[Some(1), Some(1), Some(4), Some(4)], DType::F32);
        let weight = gb.add_weight(&[Some(1), Some(1), Some(3), Some(3)], DType::F32);
        let out = gb.emit_conv2d(&input, &weight, None, [0, 0, 0, 0], [1, 1], [1, 1]);
        assert_eq!(out.shape(), vec![Some(1), Some(1), Some(2), Some(2)]);
        let graph = gb.compile(&[&out]).expect("compile conv2d_no_padding");

        let input_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let kernel_data = vec![1.0f32; 9];
        let in_buf = Buffer::from_slice(&input_data, &[1, 1, 4, 4], DType::F32);
        let wt_buf = Buffer::from_slice(&kernel_data, &[1, 1, 3, 3], DType::F32);
        let out = graph.run(&[&in_buf, &wt_buf]);
        assert_eq!(out[0].as_slice::<f32>(), &[54.0, 63.0, 90.0, 99.0]);
    }

    /// conv2d with padding [1,1,1,1]: same as above but pad 1 on all sides → 1x1x4x4 output.
    ///
    /// With 1-zero-pad the padded input is 6x6.
    /// Output spatial size = (4 + 1 + 1 - 3) / 1 + 1 = 4.
    /// Top-left corner (0,0): sum over padded window rows 0..2, cols 0..2.
    ///   = 0+0+0 + 0+1+2 + 0+5+6 = 14
    #[test]
    fn conv2d_with_padding() {
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let input = gb.input(&[Some(1), Some(1), Some(4), Some(4)], DType::F32);
        let weight = gb.add_weight(&[Some(1), Some(1), Some(3), Some(3)], DType::F32);
        let out = gb.emit_conv2d(&input, &weight, None, [1, 1, 1, 1], [1, 1], [1, 1]);
        assert_eq!(out.shape(), vec![Some(1), Some(1), Some(4), Some(4)]);
        let graph = gb.compile(&[&out]).expect("compile conv2d_with_padding");

        let input_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let kernel_data = vec![1.0f32; 9];
        let in_buf = Buffer::from_slice(&input_data, &[1, 1, 4, 4], DType::F32);
        let wt_buf = Buffer::from_slice(&kernel_data, &[1, 1, 3, 3], DType::F32);
        let out_data = graph.run(&[&in_buf, &wt_buf]);
        let vals = out_data[0].as_slice::<f32>();
        assert_eq!(vals.len(), 16);
        // Top-left corner: sum of padded 3x3 window = 0+0+0+0+1+2+0+5+6 = 14.
        assert!((vals[0] - 14.0).abs() < 1e-4, "vals[0]={}", vals[0]);
        // Top-right corner [0,3]: 0+0+0+3+4+0+7+8+0 = 22.
        assert!((vals[3] - 22.0).abs() < 1e-4, "vals[3]={}", vals[3]);
    }

    /// max_pool2d: 1x1x4x4 input, 2x2 kernel, stride=2, no padding → 1x1x2x2 output.
    ///
    /// Input:
    ///   [[1, 2, 3, 4],
    ///    [5, 6, 7, 8],
    ///    [9,10,11,12],
    ///    [13,14,15,16]]
    ///
    /// Pooling window max:
    ///   [0,0]: max(1,2,5,6) = 6
    ///   [0,1]: max(3,4,7,8) = 8
    ///   [1,0]: max(9,10,13,14) = 14
    ///   [1,1]: max(11,12,15,16) = 16
    #[test]
    fn max_pool2d_basic() {
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let input = gb.input(&[Some(1), Some(1), Some(4), Some(4)], DType::F32);
        let out = gb.emit_max_pool2d(&input, [2, 2], [0, 0, 0, 0], [2, 2], [1, 1]);
        assert_eq!(out.shape(), vec![Some(1), Some(1), Some(2), Some(2)]);
        let graph = gb.compile(&[&out]).expect("compile max_pool2d_basic");

        let input_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let in_buf = Buffer::from_slice(&input_data, &[1, 1, 4, 4], DType::F32);
        let out_data = graph.run(&[&in_buf]);
        assert_eq!(out_data[0].as_slice::<f32>(), &[6.0, 8.0, 14.0, 16.0]);
    }

    /// global_avg_pool: 1x1x2x2 input → 1x1x1x1 average.
    ///
    /// Input: [[1, 2], [3, 4]] → average = (1+2+3+4)/4 = 2.5
    #[test]
    fn global_avg_pool_basic() {
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let input = gb.input(&[Some(1), Some(1), Some(2), Some(2)], DType::F32);
        let out = gb.emit_global_avg_pool(&input);
        assert_eq!(out.shape(), vec![Some(1), Some(1), Some(1), Some(1)]);
        let graph = gb.compile(&[&out]).expect("compile global_avg_pool");

        let in_buf = Buffer::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 2, 2], DType::F32);
        let out_data = graph.run(&[&in_buf]);
        let vals = out_data[0].as_slice::<f32>();
        assert_eq!(vals.len(), 1);
        assert!((vals[0] - 2.5).abs() < 1e-5, "expected 2.5, got {}", vals[0]);
    }

    /// batch_norm with identity parameters: scale=1, bias=0, mean=0, var=1, eps=0.
    ///
    /// Result should equal input: (x - 0) / sqrt(1 + 0) * 1 + 0 = x.
    #[test]
    fn batch_norm_identity() {
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        // Input [1, 2, 2, 2], C=2.
        let input = gb.input(&[Some(1), Some(2), Some(2), Some(2)], DType::F32);
        let scale = gb.add_weight(&[Some(2)], DType::F32);
        let bias = gb.add_weight(&[Some(2)], DType::F32);
        let mean = gb.add_weight(&[Some(2)], DType::F32);
        let var = gb.add_weight(&[Some(2)], DType::F32);
        let out = gb.emit_batch_norm(&input, &scale, &bias, &mean, &var, 1e-5);
        assert_eq!(out.shape(), vec![Some(1), Some(2), Some(2), Some(2)]);
        let graph = gb.compile(&[&out]).expect("compile batch_norm");

        // Input: 8 values.
        let input_data: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let scale_data = vec![1.0f32, 1.0];
        let bias_data = vec![0.0f32, 0.0];
        let mean_data = vec![0.0f32, 0.0];
        let var_data = vec![1.0f32, 1.0];

        let in_buf = Buffer::from_slice(&input_data, &[1, 2, 2, 2], DType::F32);
        let sc_buf = Buffer::from_slice(&scale_data, &[2], DType::F32);
        let bi_buf = Buffer::from_slice(&bias_data, &[2], DType::F32);
        let mn_buf = Buffer::from_slice(&mean_data, &[2], DType::F32);
        let vr_buf = Buffer::from_slice(&var_data, &[2], DType::F32);

        let out_data = graph.run(&[&in_buf, &sc_buf, &bi_buf, &mn_buf, &vr_buf]);
        let vals = out_data[0].as_slice::<f32>();
        // With scale=1, bias=0, mean=0, var=1, eps≈0: out ≈ x / sqrt(1) = x.
        for (i, (&got, &expected)) in vals.iter().zip(input_data.iter()).enumerate() {
            assert!((got - expected).abs() < 1e-4, "vals[{i}]: got {got}, expected {expected}");
        }
    }

    // ── Meta ops tests ────────────────────────────────────────────────────

    #[test]
    fn shape_of_static() {
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let x = gb.input(&[Some(3), Some(4)], DType::F32);
        let dims = gb.emit_shape_of(&x);
        assert_eq!(dims.len(), 2);
    }

    #[test]
    fn constant_of_shape_static() {
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let x = gb.input(&[Some(6)], DType::F32); // dummy input to get shape_of values
        let dims = gb.emit_shape_of(&x);
        let filled = gb.emit_constant_of_shape(&dims, &[Some(6)], 5.0, DType::F32);
        let graph = gb.compile(&[&filled]).expect("compile constant_of_shape");

        let dummy = Buffer::from_slice(&[0.0f32; 6], &[6], DType::F32);
        let out = graph.run(&[&dummy]);
        assert_eq!(out[0].as_slice::<f32>(), &[5.0; 6]);
    }

    #[test]
    fn dense_constant() {
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let c = gb.emit_dense_constant("[1.0, 2.0, 3.0]", &[3], DType::F32);
        assert_eq!(c.shape(), vec![Some(3)]);
        let graph = gb.compile(&[&c]).expect("compile dense_constant");
        let out = graph.run(&[]);
        assert_eq!(out[0].as_slice::<f32>(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn arith_constant_scalar() {
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let c = gb.emit_arith_constant(42.0, DType::F32);
        assert_eq!(c.shape(), vec![]);
        assert_eq!(c.dtype(), DType::F32);
    }

    // ── Regression tests ──────────────────────────────────────────────────

    #[test]
    fn regression_broadcast_scalar_to_3d() {
        // Bug: canonicalize fused a broadcast linalg.generic with the binary
        // linalg.generic, producing identity maps on mismatched shapes
        // (tensor<1x1x1> vs tensor<2x3x4>). Fix: inline broadcast maps.
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(2), Some(3), Some(4)], DType::F32);
        let scalar = gb.input(&[Some(1), Some(1), Some(1)], DType::F32);
        let result = gb.emit_add(&a, &scalar);
        assert_eq!(result.shape(), vec![Some(2), Some(3), Some(4)]);

        let graph = gb.compile(&[&result]).expect("compile broadcast scalar");
        let a_buf = Buffer::from_slice(&[1.0f32; 24], &[2, 3, 4], DType::F32);
        let s_buf = Buffer::from_slice(&[10.0f32], &[1, 1, 1], DType::F32);
        let out = graph.run(&[&a_buf, &s_buf]);
        assert!(out[0].as_slice::<f32>().iter().all(|&v| (v - 11.0).abs() < 1e-5));
    }

    #[test]
    fn regression_broadcast_1d_to_3d_dynamic() {
        // Same bug but with dynamic dims: tensor<1x1x1> + tensor<?x?x768>
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[None, None, Some(768)], DType::F32);
        let scalar = gb.input(&[Some(1), Some(1), Some(1)], DType::F32);
        let result = gb.emit_sub(&a, &scalar);
        assert_eq!(result.shape(), vec![None, None, Some(768)]);
        // Shape check only — can't compile dynamic without concrete dims at runtime.
    }

    #[test]
    fn regression_reshape_infer_flat() {
        // Bug: emit_reshape panicked on target_shape=[-1] because known_product
        // was None (no known dims to divide by).
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(2), Some(3)], DType::F32);
        let flat = gb.emit_reshape(&a, &[-1]);
        assert_eq!(flat.shape(), vec![Some(6)]); // -1 → statically inferred: 2*3 = 6

        let graph = gb.compile(&[&flat]).expect("compile reshape flat");
        let a_buf = Buffer::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
        let out = graph.run(&[&a_buf]);
        assert_eq!(out[0].as_slice::<f32>(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn regression_matmul_3d_x_2d() {
        // Bug: emit_matmul (3,2) case used emit_expand_shape_1d_to_2d on a 2D
        // tensor, causing an assertion failure. Fix: use emit_unsqueeze instead.
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(2), Some(3), Some(4)], DType::F32); // [B=2, M=3, K=4]
        let b = gb.input(&[Some(4), Some(5)], DType::F32);          // [K=4, N=5]
        let c = gb.emit_matmul(&a, &b);
        assert_eq!(c.shape(), vec![Some(2), Some(3), Some(5)]);

        let graph = gb.compile(&[&c]).expect("compile matmul 3dx2d");
        let a_data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let b_data: Vec<f32> = (0..20).map(|x| x as f32 * 0.1).collect();
        let a_buf = Buffer::from_slice(&a_data, &[2, 3, 4], DType::F32);
        let b_buf = Buffer::from_slice(&b_data, &[4, 5], DType::F32);
        let out = graph.run(&[&a_buf, &b_buf]);
        assert_eq!(out[0].shape().0, vec![2, 3, 5]);
        assert!(out[0].as_slice::<f32>().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn regression_matmul_2d_x_3d() {
        // Same fix needed for the (2,3) case.
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let a = gb.input(&[Some(3), Some(4)], DType::F32);          // [M=3, K=4]
        let b = gb.input(&[Some(2), Some(4), Some(5)], DType::F32); // [B=2, K=4, N=5]
        let c = gb.emit_matmul(&a, &b);
        assert_eq!(c.shape(), vec![Some(2), Some(3), Some(5)]);

        let graph = gb.compile(&[&c]).expect("compile matmul 2dx3d");
        let a_data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let b_data: Vec<f32> = (0..40).map(|x| x as f32 * 0.1).collect();
        let a_buf = Buffer::from_slice(&a_data, &[3, 4], DType::F32);
        let b_buf = Buffer::from_slice(&b_data, &[2, 4, 5], DType::F32);
        let out = graph.run(&[&a_buf, &b_buf]);
        assert_eq!(out[0].shape().0, vec![2, 3, 5]);
        assert!(out[0].as_slice::<f32>().iter().all(|v| v.is_finite()));
    }

    // ── Sub-function tests (tasks 1.5, 1.6) ─────────────────────────────────

    #[test]
    fn subfunction_simple_matmul() {
        // Task 1.5: input → sub-function(matmul) → return.
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();

        let a = gb.input(&[Some(2), Some(3)], DType::F32);
        let b = gb.input(&[Some(3), Some(4)], DType::F32);

        // Start sub-function that receives a and b.
        let (handle, sub_args) = gb.begin_subfunction("chunk_0", &[&a, &b]);
        let product = gb.emit_matmul(&sub_args[0], &sub_args[1]);
        let results = gb.end_subfunction(handle, &[&product]);

        // results[0] is the matmul result in @compute's scope.
        let graph = gb.compile(&[&results[0]]).expect("compile with subfunction");

        // a: [[1,2,3],[4,5,6]], b: [[1,0,0,0],[0,1,0,0],[0,0,1,0]]
        let a_buf = Buffer::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
            DType::F32,
        );
        let b_buf = Buffer::from_slice(
            &[1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            &[3, 4],
            DType::F32,
        );
        let out = graph.run(&[&a_buf, &b_buf]);
        assert_eq!(out[0].shape().0, vec![2, 4]);
        let data = out[0].as_slice::<f32>();
        // With identity-like b, result = [1,2,3,0; 4,5,6,0]
        assert_eq!(data, &[1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 0.0]);
    }

    #[test]
    fn subfunction_two_chained() {
        // Task 1.6: chunk_0 produces value, chunk_1 consumes it.
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();

        let x = gb.input(&[Some(4)], DType::F32);
        let y = gb.input(&[Some(4)], DType::F32);

        // chunk_0: add x + y
        let (h0, args0) = gb.begin_subfunction("chunk_0", &[&x, &y]);
        let sum = gb.emit_add(&args0[0], &args0[1]);
        let r0 = gb.end_subfunction(h0, &[&sum]);

        // chunk_1: mul result * y
        let (h1, args1) = gb.begin_subfunction("chunk_1", &[&r0[0], &y]);
        let product = gb.emit_mul(&args1[0], &args1[1]);
        let r1 = gb.end_subfunction(h1, &[&product]);

        let graph = gb.compile(&[&r1[0]]).expect("compile chained subfunctions");

        let x_buf = Buffer::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], DType::F32);
        let y_buf = Buffer::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[4], DType::F32);
        let out = graph.run(&[&x_buf, &y_buf]);
        let data = out[0].as_slice::<f32>();
        // (x+y)*y = (1+10)*10, (2+20)*20, (3+30)*30, (4+40)*40 = 110, 440, 990, 1760
        assert_eq!(data, &[110.0, 440.0, 990.0, 1760.0]);
    }
}
