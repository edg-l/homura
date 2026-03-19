pub mod emitter;
pub mod parser;
pub mod proto;
pub(crate) mod sym_shapes;

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Mutex;

use crate::{
    Shape,
    runtime::{Buffer, ExecutionPlan, OutputDesc},
    shape::DIM_DYNAMIC,
};
use parser::{Dim, OnnxError, OnnxModel};

// ── Compiled state ─────────────────────────────────────────────────────────────

struct CompiledState {
    plan: ExecutionPlan,
    weights: Vec<Buffer>,
    /// Input shapes used during compilation (for shape-change detection).
    /// Dynamic dims are represented as `DIM_DYNAMIC` so the shapes-changed
    /// check skips them.
    input_shapes: Vec<Shape>,
}

// ── Model ──────────────────────────────────────────────────────────────────────

/// An ONNX model that is either eagerly compiled (no symbolic dims) or lazily
/// compiled on the first `run()` call (symbolic dims resolved from input shapes).
///
/// Thread-safety: `run()` takes `&self`; the inner `Mutex` serialises the
/// one-time compilation for models with symbolic dims.
pub struct Model {
    num_dynamic_inputs: usize,
    /// Stored for lazy compilation. `None` after eager compilation.
    parsed: Option<OnnxModel>,
    /// `None` until compiled (lazy). `Some` after eager or first lazy compile.
    state: Mutex<Option<CompiledState>>,
    /// Raw protobuf bytes of the model, used for cache key computation.
    /// `None` when the model was created via `load_bytes` with no bytes stored,
    /// but in practice always `Some` for models loaded from file or bytes.
    model_bytes: Option<Vec<u8>>,
    /// Symbolic dim names that should remain as `DIM_DYNAMIC` in the compiled
    /// code rather than being resolved to concrete values at trace time. This
    /// allows a single compiled artifact to accept varying values for these
    /// dims at runtime (e.g., `past_sequence_length` for KV-cache decoding).
    keep_dynamic: HashSet<String>,
}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let compiled = self.state.lock().unwrap().is_some();
        f.debug_struct("Model")
            .field("num_dynamic_inputs", &self.num_dynamic_inputs)
            .field("compiled", &compiled)
            .finish_non_exhaustive()
    }
}

impl Model {
    /// Load an ONNX model from a file path.
    ///
    /// Models with all-static shapes are compiled immediately. Models with
    /// symbolic dimensions defer compilation to the first `run()` call.
    pub fn load(path: impl AsRef<Path>) -> Result<Model, OnnxError> {
        let bytes = std::fs::read(path).map_err(OnnxError::Io)?;
        let onnx_model = parser::parse_bytes(&bytes)?;
        Model::from_onnx(onnx_model, Some(bytes), HashSet::new())
    }

    /// Load an ONNX model from raw protobuf bytes.
    ///
    /// Useful in tests where models are built in memory.
    pub fn load_bytes(bytes: &[u8]) -> Result<Model, OnnxError> {
        let onnx_model = parser::parse_bytes(bytes)?;
        Model::from_onnx(onnx_model, Some(bytes.to_vec()), HashSet::new())
    }

    /// Load an ONNX model, keeping the specified symbolic dim names as dynamic
    /// in the compiled code.
    ///
    /// Dims in `keep_dynamic` are represented as `DIM_DYNAMIC` throughout the
    /// trace/compiler so that a single compiled artifact works for any runtime
    /// value of those dims (e.g., `past_sequence_length` for KV-cache decoding).
    /// The `run()` method will compute concrete output shapes from the actual
    /// input buffer shapes and allocate output buffers accordingly.
    pub fn load_with_dynamic_dims(
        path: impl AsRef<Path>,
        keep_dynamic: HashSet<String>,
    ) -> Result<Model, OnnxError> {
        let bytes = std::fs::read(path).map_err(OnnxError::Io)?;
        let onnx_model = parser::parse_bytes(&bytes)?;
        Model::from_onnx(onnx_model, Some(bytes), keep_dynamic)
    }

    fn from_onnx(
        onnx_model: OnnxModel,
        model_bytes: Option<Vec<u8>>,
        keep_dynamic: HashSet<String>,
    ) -> Result<Model, OnnxError> {
        let num_dynamic = onnx_model.dynamic_inputs.len();

        if onnx_model.has_symbolic_dims() {
            // Defer compilation — we don't know concrete shapes yet.
            Ok(Model {
                num_dynamic_inputs: num_dynamic,
                parsed: Some(onnx_model),
                state: Mutex::new(None),
                model_bytes,
                keep_dynamic,
            })
        } else {
            // All shapes concrete: compile eagerly.
            let dummy_inputs: Vec<Buffer> = onnx_model
                .dynamic_inputs
                .iter()
                .map(|di| {
                    let shape = di.concrete_shape().expect("all dims are concrete");
                    Buffer::new(&shape.0, di.dtype)
                })
                .collect();
            let dummy_refs: Vec<&Buffer> = dummy_inputs.iter().collect();
            let state = compile_model_emitter(
                &onnx_model,
                &dummy_refs,
                model_bytes.as_deref(),
                &keep_dynamic,
            )?;
            Ok(Model {
                num_dynamic_inputs: num_dynamic,
                parsed: None,
                state: Mutex::new(Some(state)),
                model_bytes,
                keep_dynamic,
            })
        }
    }

    /// Run inference with the given dynamic inputs.
    ///
    /// For models with symbolic dimensions, the first call resolves dims from
    /// the actual input buffer shapes and triggers JIT compilation.
    /// `inputs` must have exactly `num_dynamic_inputs` entries.
    pub fn run(&self, inputs: &[&Buffer]) -> Result<Vec<Buffer>, OnnxError> {
        if inputs.len() != self.num_dynamic_inputs {
            return Err(OnnxError::WrongInputCount {
                expected: self.num_dynamic_inputs,
                got: inputs.len(),
            });
        }

        let mut guard = self.state.lock().unwrap();

        if guard.is_none() {
            // Lazy compile path: resolve symbolic dims from actual input shapes.
            let parsed = self
                .parsed
                .as_ref()
                .expect("parsed model must be present when state is None");

            let resolved = resolve_symbolic_dims(parsed, inputs, &self.keep_dynamic)?;
            *guard = Some(self.do_compile(&resolved, inputs)?);
        }

        // For models with symbolic dims, recompile only if non-dynamic dims changed.
        if let Some(parsed) = self.parsed.as_ref() {
            let shapes_changed =
                {
                    let state = guard.as_ref().unwrap();
                    inputs.iter().enumerate().any(|(i, buf)| {
                        let compiled_shape = &state.input_shapes[i];
                        buf.shape().0.iter().zip(compiled_shape.0.iter()).any(
                            |(actual, compiled)| {
                                // Skip dims that are DIM_DYNAMIC in the compiled shape —
                                // those are intentionally dynamic and should not trigger recompilation.
                                *compiled != DIM_DYNAMIC && actual != compiled
                            },
                        )
                    })
                };
            if shapes_changed {
                let resolved = resolve_symbolic_dims(parsed, inputs, &self.keep_dynamic)?;
                *guard = Some(self.do_compile(&resolved, inputs)?);
            }
        }

        let state = guard.as_ref().unwrap();
        Ok(state.plan.run(inputs, &state.weights))
    }

    fn do_compile(
        &self,
        model: &OnnxModel,
        inputs: &[&Buffer],
    ) -> Result<CompiledState, OnnxError> {
        compile_model_emitter(
            model,
            inputs,
            self.model_bytes.as_deref(),
            &self.keep_dynamic,
        )
    }

    /// Returns the compiled output descriptors, or `None` if not yet compiled.
    pub fn output_descs(&self) -> Option<Vec<OutputDesc>> {
        let guard = self.state.lock().unwrap();
        guard.as_ref().map(|s| {
            s.plan
                .output_slot_descs()
                .iter()
                .map(|sd| OutputDesc {
                    shape: sd.shape.clone(),
                    dtype: sd.dtype,
                })
                .collect()
        })
    }

    /// Whether the compiled plan has KV cache support.
    pub fn has_kv_cache(&self) -> bool {
        let guard = self.state.lock().unwrap();
        guard.as_ref().is_some_and(|s| s.plan.has_kv_cache())
    }

    /// Run inference using the internal KV cache.
    ///
    /// `inputs` should contain only non-KV model inputs (e.g., input_ids + attention_mask).
    /// Past KV is supplied by the cache; present KV is fed back automatically.
    /// Returns only non-KV outputs (e.g., logits).
    pub fn run_kv(&self, inputs: &[&Buffer], max_seq_len: usize) -> Result<Vec<Buffer>, OnnxError> {
        let mut guard = self.state.lock().unwrap();

        if guard.is_none() {
            // Trigger compilation with dummy KV inputs. The KV dims are
            // keep_dynamic so their concrete size doesn't matter — use 0.
            let parsed = self
                .parsed
                .as_ref()
                .expect("parsed model must be present when state is None");
            let full_inputs = self.build_full_inputs_for_compile(parsed, inputs);
            let full_refs: Vec<&Buffer> = full_inputs.iter().collect();
            let resolved = resolve_symbolic_dims(parsed, &full_refs, &self.keep_dynamic)?;
            *guard = Some(self.do_compile(&resolved, &full_refs)?);
        }

        let state = guard.as_mut().unwrap();
        Ok(state.plan.run_kv(inputs, &state.weights, max_seq_len))
    }

    /// Build a full input list for compilation by inserting dummy KV buffers
    /// at the positions expected by the model's dynamic_inputs.
    ///
    /// `non_kv_inputs` are the caller-provided inputs (e.g., input_ids + mask).
    /// KV inputs (those with "past_key_values" in their name) get zero-filled
    /// dummy buffers with the symbolic sequence dim set to 0.
    fn build_full_inputs_for_compile(
        &self,
        parsed: &OnnxModel,
        non_kv_inputs: &[&Buffer],
    ) -> Vec<Buffer> {
        let mut full: Vec<Buffer> = Vec::with_capacity(parsed.dynamic_inputs.len());
        let mut ext_idx = 0;
        for di in &parsed.dynamic_inputs {
            if di.name.contains("past_key_values") {
                // Dummy KV: use fixed dims from the spec, 0 for symbolic (dynamic) dims.
                let shape: Vec<u64> = di
                    .dims
                    .iter()
                    .map(|d| match d {
                        parser::Dim::Fixed(v) => *v,
                        parser::Dim::Symbolic(_) => 0,
                    })
                    .collect();
                full.push(Buffer::new(&shape, di.dtype));
            } else {
                assert!(
                    ext_idx < non_kv_inputs.len(),
                    "run_kv: not enough non-KV inputs (expected more, got {})",
                    non_kv_inputs.len()
                );
                full.push(non_kv_inputs[ext_idx].clone());
                ext_idx += 1;
            }
        }
        full
    }

    /// Initialize the KV cache from prefill output buffers.
    ///
    /// `kv_buffers` should contain `num_layers * 2` buffers (K, V alternating per layer).
    pub fn init_kv_cache(
        &self,
        kv_buffers: &[crate::runtime::Buffer],
        max_seq_len: usize,
    ) -> Result<(), OnnxError> {
        let mut guard = self.state.lock().unwrap();

        if guard.is_none() {
            // Need to compile first. Use the KV buffers themselves to build
            // a full input set (non-KV inputs get dummies — only shapes matter).
            let parsed = self
                .parsed
                .as_ref()
                .expect("parsed model must be present when state is None");
            let dummy_non_kv: Vec<Buffer> = parsed
                .dynamic_inputs
                .iter()
                .filter(|di| !di.name.contains("past_key_values"))
                .map(|di| {
                    let shape: Vec<u64> = di
                        .dims
                        .iter()
                        .map(|d| match d {
                            parser::Dim::Fixed(v) => *v,
                            parser::Dim::Symbolic(_) => 1,
                        })
                        .collect();
                    Buffer::new(&shape, di.dtype)
                })
                .collect();
            let mut full: Vec<Buffer> = Vec::with_capacity(parsed.dynamic_inputs.len());
            let mut kv_idx = 0;
            let mut non_kv_idx = 0;
            for di in &parsed.dynamic_inputs {
                if di.name.contains("past_key_values") {
                    full.push(kv_buffers[kv_idx].clone());
                    kv_idx += 1;
                } else {
                    full.push(dummy_non_kv[non_kv_idx].clone());
                    non_kv_idx += 1;
                }
            }
            let full_refs: Vec<&Buffer> = full.iter().collect();
            let resolved = resolve_symbolic_dims(parsed, &full_refs, &self.keep_dynamic)?;
            *guard = Some(self.do_compile(&resolved, &full_refs)?);
        }

        let state = guard.as_mut().unwrap();
        state.plan.init_kv_cache(kv_buffers, max_seq_len);
        Ok(())
    }

    /// Reset the KV cache for a new sequence.
    pub fn reset_kv_cache(&self) {
        let mut guard = self.state.lock().unwrap();
        if let Some(state) = guard.as_mut() {
            state.plan.reset_kv_cache();
        }
    }

    /// Current KV cache sequence length.
    pub fn kv_cache_len(&self) -> usize {
        let guard = self.state.lock().unwrap();
        guard.as_ref().map(|s| s.plan.kv_cache_len()).unwrap_or(0)
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────────

/// Compile an `OnnxModel` using per-kernel compilation.
///
/// Each heavy op (Conv, MatMul, Gemm) becomes its own compiled kernel with
/// its own MLIR context + pass pipeline. Buffer routing between kernels is
/// handled by the returned `ExecutionPlan`.
fn compile_model_emitter(
    model: &OnnxModel,
    inputs: &[&Buffer],
    model_bytes: Option<&[u8]>,
    keep_dynamic: &HashSet<String>,
) -> Result<CompiledState, OnnxError> {
    // Build input_shapes for recompilation detection.
    let input_shapes: Vec<Shape> = model
        .dynamic_inputs
        .iter()
        .zip(inputs.iter())
        .map(|(spec, buf)| {
            let dims: Vec<u64> = spec
                .dims
                .iter()
                .enumerate()
                .map(|(i, d)| match d {
                    Dim::Fixed(v) if *v == DIM_DYNAMIC => DIM_DYNAMIC,
                    Dim::Fixed(v) => *v,
                    Dim::Symbolic(name) if keep_dynamic.contains(name) => DIM_DYNAMIC,
                    Dim::Symbolic(_) => buf.shape().0[i],
                })
                .collect();
            Shape(dims)
        })
        .collect();

    let (plan, weights) = emitter::emit_and_compile_plan(model, inputs, model_bytes, keep_dynamic)?;

    Ok(CompiledState {
        plan,
        weights,
        input_shapes,
    })
}

/// Build a resolved copy of `model` where every symbolic dim in
/// `dynamic_inputs` is replaced with either:
/// - `DIM_DYNAMIC` if the dim name is in `keep_dynamic`, or
/// - the concrete size read from `inputs` otherwise.
///
/// Returns `ConflictingSymbolicDim` if two inputs disagree on the same name
/// (for non-keep_dynamic dims only).
/// Returns `UnresolvedSymbolicDim` if a name is not in `keep_dynamic` and
/// no input covers it.
fn resolve_symbolic_dims(
    model: &OnnxModel,
    inputs: &[&Buffer],
    keep_dynamic: &HashSet<String>,
) -> Result<OnnxModel, OnnxError> {
    // Build name → value map from the provided buffers (for non-keep_dynamic dims).
    let mut sym_map: HashMap<String, u64> = HashMap::new();

    for (input_spec, buffer) in model.dynamic_inputs.iter().zip(inputs.iter()) {
        let buf_shape = buffer.shape();
        for (idx, dim) in input_spec.dims.iter().enumerate() {
            if let Dim::Symbolic(name) = dim {
                if keep_dynamic.contains(name) {
                    // This dim stays dynamic — don't resolve from input buffer.
                    continue;
                }
                if let Some(&concrete) = buf_shape.0.get(idx) {
                    match sym_map.get(name) {
                        None => {
                            sym_map.insert(name.clone(), concrete);
                        }
                        Some(&existing) if existing == concrete => {} // consistent
                        Some(&existing) => {
                            return Err(OnnxError::ConflictingSymbolicDim {
                                name: name.clone(),
                                first: existing,
                                second: concrete,
                            });
                        }
                    }
                }
            }
        }
    }

    // Verify all non-keep_dynamic symbolic dims are resolved.
    for input_spec in &model.dynamic_inputs {
        for dim in &input_spec.dims {
            if let Dim::Symbolic(name) = dim
                && !keep_dynamic.contains(name)
                && !sym_map.contains_key(name)
            {
                return Err(OnnxError::UnresolvedSymbolicDim(name.clone()));
            }
        }
    }

    // Build a resolved OnnxModel clone.
    // keep_dynamic dims stay as Dim::Symbolic so the emitter preserves their
    // original name (e.g. "sequence_length" vs "past_sequence_length") for
    // correct sym_shape tracking.  The emitter's DIM_DYNAMIC branch is only
    // a fallback for truly anonymous dims.
    let resolved_inputs = model
        .dynamic_inputs
        .iter()
        .map(|inp| {
            let resolved_dims = inp
                .dims
                .iter()
                .map(|d| match d {
                    Dim::Fixed(v) => Dim::Fixed(*v),
                    Dim::Symbolic(name) if keep_dynamic.contains(name) => {
                        Dim::Symbolic(name.clone())
                    }
                    Dim::Symbolic(name) => Dim::Fixed(*sym_map.get(name).unwrap()),
                })
                .collect();
            parser::DynamicInput {
                name: inp.name.clone(),
                dims: resolved_dims,
                dtype: inp.dtype,
            }
        })
        .collect();

    Ok(OnnxModel {
        nodes: model.nodes.clone(),
        initializers: model.initializers.clone(),
        dynamic_inputs: resolved_inputs,
        outputs: model.outputs.clone(),
        output_shapes: model.output_shapes.clone(),
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, onnx::proto::*};
    use prost::Message;

    // ── CNN model builder (Task 13.1) ─────────────────────────────────────────

    /// Build a small CNN ONNX model via protobuf:
    ///
    /// Input [1,1,8,8]
    ///   → Conv(kernel=[1,1,3,3], pad=1) → [1,1,8,8]
    ///   → Relu → [1,1,8,8]
    ///   → MaxPool(kernel=2×2, stride=2) → [1,1,4,4]
    ///   → Reshape([1,16]) → [1,16]
    ///   → MatMul(W:[16,4]) → [1,4]
    ///   → Add(bias:[4]) → [1,4]
    ///
    /// All-ones conv kernel: each output pixel = sum of 3×3 neighbourhood.
    /// All-ones FC weight: each of the 4 outputs = sum over the 16 spatial values.
    /// Zero FC bias: output = FC output unchanged.
    fn make_cnn_model_bytes() -> Vec<u8> {
        // Conv kernel: [1,1,3,3] all-ones (CO=1, CI=1, KH=3, KW=3).
        let conv_kernel: Vec<f32> = vec![1.0f32; 1 * 1 * 3 * 3];
        let conv_kernel_raw: Vec<u8> = conv_kernel.iter().flat_map(|v| v.to_le_bytes()).collect();

        // Reshape target shape: [1, 16] encoded as I64.
        let reshape_shape: Vec<i64> = vec![1, 16];
        let reshape_shape_raw: Vec<u8> =
            reshape_shape.iter().flat_map(|v| v.to_le_bytes()).collect();

        // FC weight: [16, 4] all-ones.
        let fc_weight: Vec<f32> = vec![1.0f32; 16 * 4];
        let fc_weight_raw: Vec<u8> = fc_weight.iter().flat_map(|v| v.to_le_bytes()).collect();

        // FC bias: [4] all-zeros.
        let fc_bias: Vec<f32> = vec![0.0f32; 4];
        let fc_bias_raw: Vec<u8> = fc_bias.iter().flat_map(|v| v.to_le_bytes()).collect();

        encode(&ModelProto {
            ir_version: 8,
            graph: Some(GraphProto {
                node: vec![
                    // Conv: X, conv_w → conv_out
                    NodeProto {
                        op_type: "Conv".into(),
                        input: vec!["X".into(), "conv_w".into()],
                        output: vec!["conv_out".into()],
                        attribute: vec![
                            AttributeProto {
                                name: "kernel_shape".into(),
                                r#type: 7, // INTS
                                ints: vec![3, 3],
                                ..Default::default()
                            },
                            AttributeProto {
                                name: "pads".into(),
                                r#type: 7, // INTS
                                ints: vec![1, 1, 1, 1],
                                ..Default::default()
                            },
                            AttributeProto {
                                name: "strides".into(),
                                r#type: 7, // INTS
                                ints: vec![1, 1],
                                ..Default::default()
                            },
                        ],
                        ..Default::default()
                    },
                    // Relu: conv_out → relu_out
                    NodeProto {
                        op_type: "Relu".into(),
                        input: vec!["conv_out".into()],
                        output: vec!["relu_out".into()],
                        ..Default::default()
                    },
                    // MaxPool: relu_out → pool_out
                    NodeProto {
                        op_type: "MaxPool".into(),
                        input: vec!["relu_out".into()],
                        output: vec!["pool_out".into()],
                        attribute: vec![
                            AttributeProto {
                                name: "kernel_shape".into(),
                                r#type: 7, // INTS
                                ints: vec![2, 2],
                                ..Default::default()
                            },
                            AttributeProto {
                                name: "strides".into(),
                                r#type: 7, // INTS
                                ints: vec![2, 2],
                                ..Default::default()
                            },
                        ],
                        ..Default::default()
                    },
                    // Reshape: pool_out, reshape_shape → flat_out
                    NodeProto {
                        op_type: "Reshape".into(),
                        input: vec!["pool_out".into(), "reshape_shape".into()],
                        output: vec!["flat_out".into()],
                        ..Default::default()
                    },
                    // MatMul: flat_out, fc_w → matmul_out
                    NodeProto {
                        op_type: "MatMul".into(),
                        input: vec!["flat_out".into(), "fc_w".into()],
                        output: vec!["matmul_out".into()],
                        ..Default::default()
                    },
                    // Add: matmul_out, fc_b → output
                    NodeProto {
                        op_type: "Add".into(),
                        input: vec!["matmul_out".into(), "fc_b".into()],
                        output: vec!["output".into()],
                        ..Default::default()
                    },
                ],
                initializer: vec![
                    TensorProto {
                        name: "conv_w".into(),
                        dims: vec![1, 1, 3, 3],
                        data_type: 1, // FLOAT
                        raw_data: conv_kernel_raw,
                        ..Default::default()
                    },
                    TensorProto {
                        name: "reshape_shape".into(),
                        dims: vec![2],
                        data_type: 7, // INT64
                        raw_data: reshape_shape_raw,
                        ..Default::default()
                    },
                    TensorProto {
                        name: "fc_w".into(),
                        dims: vec![16, 4],
                        data_type: 1, // FLOAT
                        raw_data: fc_weight_raw,
                        ..Default::default()
                    },
                    TensorProto {
                        name: "fc_b".into(),
                        dims: vec![4],
                        data_type: 1, // FLOAT
                        raw_data: fc_bias_raw,
                        ..Default::default()
                    },
                ],
                input: vec![
                    value_info("X", &[1, 1, 8, 8]),
                    value_info("conv_w", &[1, 1, 3, 3]),
                    value_info("reshape_shape", &[2]),
                    value_info("fc_w", &[16, 4]),
                    value_info("fc_b", &[4]),
                ],
                output: vec![value_info("output", &[1, 4])],
                ..Default::default()
            }),
            ..Default::default()
        })
    }

    // ── Task 13.2 + 13.3: CNN pipeline integration test ───────────────────────

    /// Full CNN pipeline: Conv → Relu → MaxPool → Reshape → MatMul → Add.
    ///
    /// With all-1.0 input and all-ones conv kernel (pad=1):
    ///   - Each conv output pixel = sum of its 3×3 neighbourhood.
    ///   - MaxPool 2×2 stride 2 always picks the maximum of each 2×2 block.
    ///     Every 2×2 block in the 8×8 feature map contains at least one
    ///     interior pixel (value 9.0), so all 16 pool outputs = 9.0.
    ///   - FC weight is all-ones [16,4]: each output = sum(16 × 9.0) = 144.0.
    ///   - Zero bias: final output = [144.0; 4].
    #[test]
    fn load_and_run_cnn_pipeline() {
        let bytes = make_cnn_model_bytes();
        let model = Model::load_bytes(&bytes).expect("load failed");

        let input_data: Vec<f32> = vec![1.0f32; 1 * 1 * 8 * 8];
        let input = Buffer::from_slice::<f32>(&input_data, &[1, 1, 8, 8], DType::F32);
        let outputs = model.run(&[&input]).expect("run failed");

        let out = outputs[0].as_slice::<f32>();

        // Task 13.3: verify shape and values.
        assert_eq!(out.len(), 4, "expected 4 output values");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "output contains NaN or Inf: {out:?}"
        );
        assert!(
            out.iter().all(|v| *v > 0.0),
            "expected all-positive output for all-ones input: {out:?}"
        );
        // Each output neuron = sum over 16 pool values (each 9.0) × all-ones weight = 144.0.
        for (i, &v) in out.iter().enumerate() {
            assert!(
                (v - 144.0f32).abs() < 1e-3,
                "output[{i}] = {v}, expected ~144.0"
            );
        }
    }

    // ── Task 13.1 (optional): real mnist-12.onnx test ─────────────────────────

    #[test]
    fn load_and_run_mnist_12() {
        let model = Model::load("tests/fixtures/mnist-12.onnx").expect("load failed");
        let input = Buffer::from_slice::<f32>(&vec![0.0f32; 784], &[1, 1, 28, 28], DType::F32);
        let outputs = model.run(&[&input]).expect("run failed");
        let out = outputs[0].as_slice::<f32>();
        assert_eq!(out.len(), 10, "expected 10 output logits");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "output contains NaN or Inf: {out:?}"
        );
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn encode(model: &ModelProto) -> Vec<u8> {
        let mut buf = Vec::new();
        model.encode(&mut buf).unwrap();
        buf
    }

    fn fixed_shape_f32(dims: &[i64]) -> TypeProto {
        use crate::onnx::proto::tensor_shape_proto::Dimension;
        use crate::onnx::proto::type_proto;
        TypeProto {
            value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                elem_type: 1, // FLOAT
                shape: Some(TensorShapeProto {
                    dim: dims
                        .iter()
                        .map(|&d| Dimension {
                            value: Some(
                                crate::onnx::proto::tensor_shape_proto::dimension::Value::DimValue(
                                    d,
                                ),
                            ),
                            ..Default::default()
                        })
                        .collect(),
                }),
            })),
            ..Default::default()
        }
    }

    fn value_info(name: &str, dims: &[i64]) -> ValueInfoProto {
        ValueInfoProto {
            name: name.into(),
            r#type: Some(fixed_shape_f32(dims)),
            ..Default::default()
        }
    }

    /// Build a minimal Add(X, Y) -> Z ONNX model with two f32[4] dynamic inputs.
    fn make_add_model_bytes() -> Vec<u8> {
        encode(&ModelProto {
            ir_version: 8,
            graph: Some(GraphProto {
                node: vec![NodeProto {
                    op_type: "Add".into(),
                    input: vec!["X".into(), "Y".into()],
                    output: vec!["Z".into()],
                    ..Default::default()
                }],
                input: vec![value_info("X", &[4]), value_info("Y", &[4])],
                output: vec![value_info("Z", &[4])],
                ..Default::default()
            }),
            ..Default::default()
        })
    }

    /// Build an Add(X, W) -> Z model where W is a static weight (f32[4]).
    fn make_add_with_weight_bytes(weight_data: &[f32]) -> Vec<u8> {
        let raw: Vec<u8> = weight_data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let dims: Vec<i64> = vec![weight_data.len() as i64];

        encode(&ModelProto {
            ir_version: 8,
            graph: Some(GraphProto {
                node: vec![NodeProto {
                    op_type: "Add".into(),
                    input: vec!["X".into(), "W".into()],
                    output: vec!["Z".into()],
                    ..Default::default()
                }],
                initializer: vec![TensorProto {
                    name: "W".into(),
                    dims: dims.clone(),
                    data_type: 1, // FLOAT
                    raw_data: raw,
                    ..Default::default()
                }],
                // W also listed in graph.input for opset compat — parser must skip it.
                input: vec![value_info("X", &dims), value_info("W", &dims)],
                output: vec![value_info("Z", &dims)],
                ..Default::default()
            }),
            ..Default::default()
        })
    }

    // ── Task 8.4: load/run round-trip ─────────────────────────────────────────

    #[test]
    fn load_and_run_add_two_dynamic_inputs() {
        let bytes = make_add_model_bytes();
        let model = Model::load_bytes(&bytes).expect("load_bytes failed");

        let x = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[4], DType::F32);
        let y = Buffer::from_slice::<f32>(&[10.0, 20.0, 30.0, 40.0], &[4], DType::F32);
        let outs = model.run(&[&x, &y]).expect("run failed");

        assert_eq!(outs[0].as_slice::<f32>(), &[11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn load_and_run_add_with_weight() {
        let weights = [1.0f32, 2.0, 3.0, 4.0];
        let bytes = make_add_with_weight_bytes(&weights);
        let model = Model::load_bytes(&bytes).expect("load_bytes failed");

        // Only 1 dynamic input — the weight is baked in.
        let x = Buffer::from_slice::<f32>(&[10.0, 20.0, 30.0, 40.0], &[4], DType::F32);
        let outs = model.run(&[&x]).expect("run failed");

        assert_eq!(outs[0].as_slice::<f32>(), &[11.0, 22.0, 33.0, 44.0]);
    }

    // ── Task 8.5: error cases ──────────────────────────────────────────────────

    #[test]
    fn load_nonexistent_file_returns_io_error() {
        let result = Model::load("/tmp/does_not_exist_homura_test.onnx");
        assert!(
            matches!(result, Err(OnnxError::Io(_))),
            "expected Io error, got: {:?}",
            result
        );
    }

    #[test]
    fn run_wrong_input_count_returns_error() {
        let bytes = make_add_model_bytes();
        let model = Model::load_bytes(&bytes).expect("load_bytes failed");

        // Model expects 2 inputs; pass 0.
        let result = model.run(&[]);
        match result {
            Err(OnnxError::WrongInputCount {
                expected: 2,
                got: 0,
            }) => {}
            other => panic!("expected WrongInputCount{{2, 0}}, got: {:?}", other),
        }
    }

    #[test]
    fn run_too_many_inputs_returns_error() {
        let bytes = make_add_model_bytes();
        let model = Model::load_bytes(&bytes).expect("load_bytes failed");

        let x = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[4], DType::F32);
        let y = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[4], DType::F32);
        let z = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[4], DType::F32);
        let result = model.run(&[&x, &y, &z]);
        match result {
            Err(OnnxError::WrongInputCount {
                expected: 2,
                got: 3,
            }) => {}
            other => panic!("expected WrongInputCount{{2, 3}}, got: {:?}", other),
        }
    }

    // ── Task 8.6: multiple run() calls — weights not mutated ──────────────────

    #[test]
    fn multiple_runs_produce_correct_results() {
        let bytes = make_add_model_bytes();
        let model = Model::load_bytes(&bytes).expect("load_bytes failed");

        let x1 = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[4], DType::F32);
        let y1 = Buffer::from_slice::<f32>(&[10.0, 20.0, 30.0, 40.0], &[4], DType::F32);
        let outs1 = model.run(&[&x1, &y1]).expect("first run failed");
        assert_eq!(outs1[0].as_slice::<f32>(), &[11.0, 22.0, 33.0, 44.0]);

        let x2 = Buffer::from_slice::<f32>(&[5.0, 6.0, 7.0, 8.0], &[4], DType::F32);
        let y2 = Buffer::from_slice::<f32>(&[50.0, 60.0, 70.0, 80.0], &[4], DType::F32);
        let outs2 = model.run(&[&x2, &y2]).expect("second run failed");
        assert_eq!(outs2[0].as_slice::<f32>(), &[55.0, 66.0, 77.0, 88.0]);

        // First result is still correct — no aliasing between runs.
        assert_eq!(outs1[0].as_slice::<f32>(), &[11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn multiple_runs_with_weight_not_mutated() {
        let weights = [0.5f32, 1.0, 1.5, 2.0];
        let bytes = make_add_with_weight_bytes(&weights);
        let model = Model::load_bytes(&bytes).expect("load_bytes failed");

        let x1 = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[4], DType::F32);
        let outs1 = model.run(&[&x1]).expect("first run failed");
        assert_eq!(outs1[0].as_slice::<f32>(), &[1.5, 3.0, 4.5, 6.0]);

        // Run again with different input; weight should be unchanged.
        let x2 = Buffer::from_slice::<f32>(&[10.0, 10.0, 10.0, 10.0], &[4], DType::F32);
        let outs2 = model.run(&[&x2]).expect("second run failed");
        assert_eq!(outs2[0].as_slice::<f32>(), &[10.5, 11.0, 11.5, 12.0]);

        // Confirm first result is independent.
        assert_eq!(outs1[0].as_slice::<f32>(), &[1.5, 3.0, 4.5, 6.0]);
    }

    // ── Task 2.6: multi-output model loads and returns all outputs ────────────

    #[test]
    fn model_with_two_outputs_succeeds() {
        // Add(X, Y) -> Z; graph outputs are Z and X.
        // Both Z (= X + Y) and X should be returned.
        let bytes = encode(&ModelProto {
            ir_version: 8,
            graph: Some(GraphProto {
                node: vec![NodeProto {
                    op_type: "Add".into(),
                    input: vec!["X".into(), "Y".into()],
                    output: vec!["Z".into()],
                    ..Default::default()
                }],
                input: vec![value_info("X", &[4]), value_info("Y", &[4])],
                // Two outputs: Z (add result) and X (passthrough of first input).
                output: vec![value_info("Z", &[4]), value_info("X", &[4])],
                ..Default::default()
            }),
            ..Default::default()
        });

        let model = Model::load_bytes(&bytes).expect("load must succeed for 2-output model");
        let x = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[4], DType::F32);
        let y = Buffer::from_slice::<f32>(&[10.0, 20.0, 30.0, 40.0], &[4], DType::F32);
        let outputs = model.run(&[&x, &y]).expect("run must succeed");

        assert_eq!(outputs.len(), 2, "expected 2 output buffers");
        // First output: Z = X + Y
        assert_eq!(
            outputs[0].as_slice::<f32>(),
            &[11.0, 22.0, 33.0, 44.0],
            "first output (Z = X + Y) is wrong"
        );
        // Second output: X (passthrough)
        assert_eq!(
            outputs[1].as_slice::<f32>(),
            &[1.0, 2.0, 3.0, 4.0],
            "second output (X passthrough) is wrong"
        );
    }

    // ── Multi-op round-trip tests ────────────────────────────────────────────

    #[test]
    fn load_and_run_relu() {
        let bytes = encode(&ModelProto {
            ir_version: 8,
            graph: Some(GraphProto {
                node: vec![NodeProto {
                    op_type: "Relu".into(),
                    input: vec!["X".into()],
                    output: vec!["Y".into()],
                    ..Default::default()
                }],
                input: vec![value_info("X", &[4])],
                output: vec![value_info("Y", &[4])],
                ..Default::default()
            }),
            ..Default::default()
        });
        let model = Model::load_bytes(&bytes).expect("load failed");
        let x = Buffer::from_slice::<f32>(&[-1.0, 0.0, 3.0, -4.0], &[4], DType::F32);
        let results = model.run(&[&x]).unwrap();
        assert_eq!(results[0].as_slice::<f32>(), &[0.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn load_and_run_add_relu_chain() {
        let bytes = encode(&ModelProto {
            ir_version: 8,
            graph: Some(GraphProto {
                node: vec![
                    NodeProto {
                        op_type: "Add".into(),
                        input: vec!["X".into(), "Y".into()],
                        output: vec!["sum".into()],
                        ..Default::default()
                    },
                    NodeProto {
                        op_type: "Relu".into(),
                        input: vec!["sum".into()],
                        output: vec!["Z".into()],
                        ..Default::default()
                    },
                ],
                input: vec![value_info("X", &[4]), value_info("Y", &[4])],
                output: vec![value_info("Z", &[4])],
                ..Default::default()
            }),
            ..Default::default()
        });
        let model = Model::load_bytes(&bytes).expect("load failed");
        let x = Buffer::from_slice::<f32>(&[1.0, -2.0, 3.0, -4.0], &[4], DType::F32);
        let y = Buffer::from_slice::<f32>(&[5.0, 6.0, -7.0, 8.0], &[4], DType::F32);
        let results = model.run(&[&x, &y]).unwrap();
        // add=[6,4,-4,4], relu=[6,4,0,4]
        assert_eq!(results[0].as_slice::<f32>(), &[6.0, 4.0, 0.0, 4.0]);
    }

    #[test]
    fn load_and_run_two_weight_chain() {
        // input → Add(W1) → Add(W2) → output
        let w1_raw: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let w2_raw: Vec<u8> = [10.0f32, 20.0, 30.0, 40.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let bytes = encode(&ModelProto {
            ir_version: 8,
            graph: Some(GraphProto {
                node: vec![
                    NodeProto {
                        op_type: "Add".into(),
                        input: vec!["X".into(), "W1".into()],
                        output: vec!["mid".into()],
                        ..Default::default()
                    },
                    NodeProto {
                        op_type: "Add".into(),
                        input: vec!["mid".into(), "W2".into()],
                        output: vec!["Z".into()],
                        ..Default::default()
                    },
                ],
                initializer: vec![
                    TensorProto {
                        name: "W1".into(),
                        dims: vec![4],
                        data_type: 1,
                        raw_data: w1_raw,
                        ..Default::default()
                    },
                    TensorProto {
                        name: "W2".into(),
                        dims: vec![4],
                        data_type: 1,
                        raw_data: w2_raw,
                        ..Default::default()
                    },
                ],
                input: vec![
                    value_info("X", &[4]),
                    value_info("W1", &[4]),
                    value_info("W2", &[4]),
                ],
                output: vec![value_info("Z", &[4])],
                ..Default::default()
            }),
            ..Default::default()
        });
        let model = Model::load_bytes(&bytes).expect("load failed");
        let x = Buffer::from_slice::<f32>(&[100.0, 200.0, 300.0, 400.0], &[4], DType::F32);
        let results = model.run(&[&x]).unwrap();
        // x + w1 + w2 = [111, 222, 333, 444]
        assert_eq!(results[0].as_slice::<f32>(), &[111.0, 222.0, 333.0, 444.0]);
    }

    // ── Task 1.6: symbolic-dim model tests ────────────────────────────────────

    /// Build Add(X, Y) -> Z where X has a symbolic first dim `batch` and fixed
    /// second dim `width`. Returns bytes for `Model::load_bytes`.
    fn make_symbolic_add_model_bytes(sym_name: &str, fixed_dim: i64) -> Vec<u8> {
        use crate::onnx::proto::tensor_shape_proto::Dimension;
        use crate::onnx::proto::type_proto;

        let sym_dim = Dimension {
            value: Some(
                crate::onnx::proto::tensor_shape_proto::dimension::Value::DimParam(sym_name.into()),
            ),
            ..Default::default()
        };
        let fixed_dim_proto = Dimension {
            value: Some(
                crate::onnx::proto::tensor_shape_proto::dimension::Value::DimValue(fixed_dim),
            ),
            ..Default::default()
        };

        let make_tensor_type = || TypeProto {
            value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                elem_type: 1, // FLOAT
                shape: Some(TensorShapeProto {
                    dim: vec![sym_dim.clone(), fixed_dim_proto.clone()],
                }),
            })),
            ..Default::default()
        };

        encode(&ModelProto {
            ir_version: 8,
            graph: Some(GraphProto {
                node: vec![NodeProto {
                    op_type: "Add".into(),
                    input: vec!["X".into(), "Y".into()],
                    output: vec!["Z".into()],
                    ..Default::default()
                }],
                input: vec![
                    ValueInfoProto {
                        name: "X".into(),
                        r#type: Some(make_tensor_type()),
                        ..Default::default()
                    },
                    ValueInfoProto {
                        name: "Y".into(),
                        r#type: Some(make_tensor_type()),
                        ..Default::default()
                    },
                ],
                output: vec![ValueInfoProto {
                    name: "Z".into(),
                    r#type: Some(make_tensor_type()),
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        })
    }

    #[test]
    fn model_with_symbolic_dims_defers_compilation() {
        let bytes = make_symbolic_add_model_bytes("batch", 4);
        let model = Model::load_bytes(&bytes).expect("load must succeed for symbolic dims");

        // Compilation should have been deferred: state is None until run().
        let guard = model.state.lock().unwrap();
        assert!(
            guard.is_none(),
            "compilation should be deferred for symbolic-dim models"
        );
    }

    #[test]
    fn symbolic_dims_resolved_on_first_run() {
        let bytes = make_symbolic_add_model_bytes("batch", 4);
        let model = Model::load_bytes(&bytes).expect("load failed");

        // Concrete shape: batch=2, fixed=4 → [2, 4]
        let x = Buffer::from_slice::<f32>(&vec![1.0f32; 8], &[2, 4], DType::F32);
        let y = Buffer::from_slice::<f32>(&vec![2.0f32; 8], &[2, 4], DType::F32);
        let outs = model.run(&[&x, &y]).expect("first run failed");

        let result = outs[0].as_slice::<f32>();
        assert_eq!(result.len(), 8, "output should have 8 elements");
        assert!(
            result.iter().all(|&v| (v - 3.0f32).abs() < 1e-6),
            "expected all 3.0, got {result:?}"
        );
    }

    #[test]
    fn conflicting_symbolic_dim_returns_error() {
        // Build a model where X and Y share the same symbolic dim name "batch"
        // but we pass inputs with different sizes in that position.
        let bytes = make_symbolic_add_model_bytes("batch", 4);
        let model = Model::load_bytes(&bytes).expect("load failed");

        // X has batch=2, Y has batch=3 — conflict on "batch".
        let x = Buffer::from_slice::<f32>(&vec![1.0f32; 8], &[2, 4], DType::F32);
        let y = Buffer::from_slice::<f32>(&vec![1.0f32; 12], &[3, 4], DType::F32);
        let result = model.run(&[&x, &y]);

        assert!(
            matches!(result, Err(OnnxError::ConflictingSymbolicDim { .. })),
            "expected ConflictingSymbolicDim, got {result:?}"
        );
    }

    #[test]
    fn unresolved_symbolic_dim_returns_error() {
        // Build a model with a symbolic dim that is NOT covered by the provided
        // input buffer shapes. We do this by giving an input whose symbolic dim
        // position has the wrong rank (0-dim buffer vs expected 2-dim).
        // More directly: build a model with a symbolic dim and pass a scalar
        // that has no dims to cover the symbolic slot.
        //
        // We construct this by building a 1-input model with shape [sym, 4] and
        // passing a buffer whose shape has only 1 dim, which will panic in the
        // indexing. Instead, we test via a model where the second input covers a
        // *different* symbolic dim name than the first, leaving the first's sym
        // unresolved.
        use crate::onnx::proto::tensor_shape_proto::Dimension;
        use crate::onnx::proto::type_proto;

        // X has dim "sym_a", Y has dim "sym_b" — two different symbolic names.
        // We only pass buffers that cover sym_a (batch=2), leaving sym_b
        // unresolved (because we give Y a different batch size, which would
        // conflict first). We need a model where a sym dim simply never appears
        // in any input buffer position.
        //
        // Simplest: build a model with ONE input whose shape is [sym, 4] but
        // pass zero inputs (wrong count) → WrongInputCount, not what we want.
        //
        // Instead: use two inputs, X=[sym_x, 4] and Y=[sym_y, 4], then provide
        // X but Y uses a different sym name whose position we zero out. Actually
        // the resolution loop iterates both inputs, so both names get resolved.
        //
        // The cleanest way: call resolve_symbolic_dims directly with a model
        // that has a sym dim but empty inputs list.
        let sym_dim = Dimension {
            value: Some(
                crate::onnx::proto::tensor_shape_proto::dimension::Value::DimParam(
                    "unresolvable".into(),
                ),
            ),
            ..Default::default()
        };
        let make_type = || TypeProto {
            value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                elem_type: 1,
                shape: Some(TensorShapeProto {
                    dim: vec![sym_dim.clone()],
                }),
            })),
            ..Default::default()
        };
        let bytes = encode(&ModelProto {
            ir_version: 8,
            graph: Some(GraphProto {
                node: vec![NodeProto {
                    op_type: "Relu".into(),
                    input: vec!["X".into()],
                    output: vec!["Y".into()],
                    ..Default::default()
                }],
                input: vec![ValueInfoProto {
                    name: "X".into(),
                    r#type: Some(make_type()),
                    ..Default::default()
                }],
                output: vec![ValueInfoProto {
                    name: "Y".into(),
                    r#type: Some(make_type()),
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        });

        // Load deferred model (not used directly; we test via model2 below).
        let _model = Model::load_bytes(&bytes).expect("load failed");

        // Pass an input with 0 dims (scalar-like). The sym_dim "unresolvable"
        // sits at index 0 in the input's dims vec, but the actual buffer has
        // shape [4], so index 0 IS resolved to 4 — that's fine.
        //
        // True unresolved dim: build a model where input has 2 dims [fixed, sym]
        // but the buffer passed has only 1 dim — buffer.shape().0[1] would
        // panic. So we test unresolved by using resolve_symbolic_dims with an
        // empty inputs slice via the internal helper instead.
        //
        // Since resolve_symbolic_dims is private, we trigger it through run()
        // by constructing a buffer whose shape length is shorter than the dims
        // vec. This would panic in the zip, not reach the unresolved check.
        //
        // The cleanest approach: use a model where NO input carries the symbolic
        // dim at all — i.e., a zero-input model (only initializers) where some
        // internal-only value has a symbolic shape. But ONNX doesn't model that.
        //
        // Alternative: expose the path via public OnnxModel → test at the Model
        // level by passing a buffer too small. But that causes an index panic.
        //
        // Simplest testable case: build the model where the single input has
        // 2 dims ([fixed=4, sym="unresolvable"]) but pass a 1-D buffer of len
        // 4. The zip stops at 1 element (the fixed dim), and Symbolic at index
        // 1 is never visited → UnresolvedSymbolicDim.
        use crate::onnx::proto::tensor_shape_proto::Dimension as D2;
        let fixed_dim = D2 {
            value: Some(crate::onnx::proto::tensor_shape_proto::dimension::Value::DimValue(4)),
            ..Default::default()
        };
        let sym2 = D2 {
            value: Some(
                crate::onnx::proto::tensor_shape_proto::dimension::Value::DimParam(
                    "unresolved_sym".into(),
                ),
            ),
            ..Default::default()
        };
        let make_type2 = || TypeProto {
            value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                elem_type: 1,
                shape: Some(TensorShapeProto {
                    dim: vec![fixed_dim.clone(), sym2.clone()],
                }),
            })),
            ..Default::default()
        };
        let bytes2 = encode(&ModelProto {
            ir_version: 8,
            graph: Some(GraphProto {
                node: vec![NodeProto {
                    op_type: "Relu".into(),
                    input: vec!["X".into()],
                    output: vec!["Y".into()],
                    ..Default::default()
                }],
                input: vec![ValueInfoProto {
                    name: "X".into(),
                    r#type: Some(make_type2()),
                    ..Default::default()
                }],
                output: vec![ValueInfoProto {
                    name: "Y".into(),
                    r#type: Some(make_type2()),
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        });
        let model2 = Model::load_bytes(&bytes2).expect("load failed");
        // Pass a 1-D buffer: zip covers only dim[0]=Fixed(4), dim[1]=Symbolic stays unresolved.
        let x = Buffer::from_slice::<f32>(&[1.0f32; 4], &[4], DType::F32);
        let result = model2.run(&[&x]);
        assert!(
            matches!(result, Err(OnnxError::UnresolvedSymbolicDim(_))),
            "expected UnresolvedSymbolicDim, got {result:?}"
        );
    }

    #[test]
    fn static_model_still_compiles_eagerly() {
        // A model with all-fixed dims must compile during load(), not lazily.
        let bytes = make_add_model_bytes();
        let model = Model::load_bytes(&bytes).expect("load failed");

        // Compilation must have happened eagerly: state is Some.
        let guard = model.state.lock().unwrap();
        assert!(
            guard.is_some(),
            "static model should compile eagerly at load time"
        );
    }
}
