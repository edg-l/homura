use std::collections::HashMap;

use crate::DType;
use crate::hf::config::TransformerConfig;
use crate::runtime::Buffer;

/// Weights for a single transformer layer.
pub struct LayerWeights {
    pub input_layernorm_weight: Buffer,
    pub q_proj_weight: Buffer, // transposed to [in, out] at load
    pub q_proj_bias: Option<Buffer>,
    pub k_proj_weight: Buffer, // transposed
    pub k_proj_bias: Option<Buffer>,
    pub v_proj_weight: Buffer, // transposed
    pub v_proj_bias: Option<Buffer>,
    /// QK-norm weights (Qwen3-style): RMSNorm on Q/K per-head, shape [head_dim].
    pub q_norm_weight: Option<Buffer>,
    pub k_norm_weight: Option<Buffer>,
    pub o_proj_weight: Buffer, // transposed
    pub post_attention_layernorm_weight: Buffer,
    pub gate_proj_weight: Buffer, // transposed
    pub up_proj_weight: Buffer,   // transposed
    pub down_proj_weight: Buffer, // transposed
}

/// All weights for a decoder-only transformer model.
pub struct TransformerWeights {
    pub embed_tokens_weight: Buffer,
    pub layers: Vec<LayerWeights>,
    pub final_norm_weight: Buffer,
    /// None when tie_word_embeddings=true (reuse embed_tokens_weight).
    pub lm_head_weight: Option<Buffer>,
}

impl TransformerWeights {
    /// Returns true if this model has QK-norm weights (Qwen3-style).
    pub fn has_qk_norm(&self) -> bool {
        self.layers
            .first()
            .map(|l| l.q_norm_weight.is_some())
            .unwrap_or(false)
    }

    /// Returns all weight buffers as a flat Vec<Buffer> in a deterministic order.
    ///
    /// `weight_dtype` controls the dtype for tied lm_head: when BF16, the
    /// transposed embed_tokens copy is truncated to bf16 for the LM head matmul.
    ///
    /// Order:
    ///   - embed_tokens_weight
    ///   - For each layer i (0..num_layers):
    ///     - input_layernorm_weight
    ///     - q_proj_weight, [q_proj_bias], k_proj_weight, [k_proj_bias],
    ///       v_proj_weight, [v_proj_bias]
    ///     - [q_norm_weight, k_norm_weight] (only if has_qk_norm)
    ///     - o_proj_weight
    ///     - post_attention_layernorm_weight
    ///     - gate_proj_weight, up_proj_weight, down_proj_weight
    ///   - final_norm_weight
    ///   - [lm_head_weight] (only if not tied)
    ///
    /// Biases and QK-norms are only included when they exist (inferred from the first layer).
    pub fn to_slot_buffers(&self, weight_dtype: DType) -> Vec<Buffer> {
        let has_bias = self
            .layers
            .first()
            .map(|l| l.q_proj_bias.is_some())
            .unwrap_or(false);
        let has_qk_norm = self.has_qk_norm();

        let mut bufs = Vec::new();
        bufs.push(self.embed_tokens_weight.clone());

        for layer in &self.layers {
            bufs.push(layer.input_layernorm_weight.clone());
            bufs.push(layer.q_proj_weight.clone());
            if has_bias {
                bufs.push(layer.q_proj_bias.as_ref().unwrap().clone());
            }
            bufs.push(layer.k_proj_weight.clone());
            if has_bias {
                bufs.push(layer.k_proj_bias.as_ref().unwrap().clone());
            }
            bufs.push(layer.v_proj_weight.clone());
            if has_bias {
                bufs.push(layer.v_proj_bias.as_ref().unwrap().clone());
            }
            if has_qk_norm {
                bufs.push(layer.q_norm_weight.as_ref().unwrap().clone());
                bufs.push(layer.k_norm_weight.as_ref().unwrap().clone());
            }
            bufs.push(layer.o_proj_weight.clone());
            bufs.push(layer.post_attention_layernorm_weight.clone());
            bufs.push(layer.gate_proj_weight.clone());
            bufs.push(layer.up_proj_weight.clone());
            bufs.push(layer.down_proj_weight.clone());
        }

        bufs.push(self.final_norm_weight.clone());

        // lm_head weight: always [hidden, vocab] (transposed).
        // For tied embeddings, transpose embed_tokens [vocab, hidden] -> [hidden, vocab].
        // When using bf16 weights, convert tied lm_head to bf16 for bandwidth savings.
        if let Some(lm_head) = &self.lm_head_weight {
            bufs.push(lm_head.clone());
        } else {
            let transposed = transpose_2d(&self.embed_tokens_weight);
            if weight_dtype == DType::BF16 && transposed.dtype() == DType::F32 {
                bufs.push(f32_buf_to_bf16(&transposed));
            } else {
                bufs.push(transposed);
            }
        }

        bufs
    }

    /// Returns all weight buffers for the quantized execution plan.
    ///
    /// Quantized projection weights are passed as flat 1D I8 byte buffers (no transpose).
    /// Norms and biases remain f32. Embedding is f32. Tied lm_head is transposed f32.
    ///
    /// Order matches `assign_transformer_slots_quant()` exactly:
    ///   - embed_tokens_weight (f32, as-is)
    ///   - For each layer i (0..num_layers):
    ///     - input_layernorm_weight (f32)
    ///     - q_proj_weight (I8 flat), [q_proj_bias (f32)]
    ///     - k_proj_weight (I8 flat), [k_proj_bias (f32)]
    ///     - v_proj_weight (I8 flat), [v_proj_bias (f32)]
    ///     - [q_norm_weight (f32), k_norm_weight (f32)] (only if has_qk_norm)
    ///     - o_proj_weight (I8 flat)
    ///     - post_attention_layernorm_weight (f32)
    ///     - gate_proj_weight (I8 flat), up_proj_weight (I8 flat), down_proj_weight (I8 flat)
    ///   - final_norm_weight (f32)
    ///   - lm_head_weight: transposed f32 (from embed_tokens if tied, else f32 copy of lm_head)
    pub fn to_slot_buffers_quant(&self) -> Vec<Buffer> {
        let has_bias = self
            .layers
            .first()
            .map(|l| l.q_proj_bias.is_some())
            .unwrap_or(false);
        let has_qk_norm = self.has_qk_norm();

        let mut bufs = Vec::new();
        bufs.push(self.embed_tokens_weight.clone());

        for layer in &self.layers {
            bufs.push(layer.input_layernorm_weight.clone());
            bufs.push(reinterpret_as_flat_i8(&layer.q_proj_weight));
            if has_bias {
                bufs.push(layer.q_proj_bias.as_ref().unwrap().clone());
            }
            bufs.push(reinterpret_as_flat_i8(&layer.k_proj_weight));
            if has_bias {
                bufs.push(layer.k_proj_bias.as_ref().unwrap().clone());
            }
            bufs.push(reinterpret_as_flat_i8(&layer.v_proj_weight));
            if has_bias {
                bufs.push(layer.v_proj_bias.as_ref().unwrap().clone());
            }
            if has_qk_norm {
                bufs.push(layer.q_norm_weight.as_ref().unwrap().clone());
                bufs.push(layer.k_norm_weight.as_ref().unwrap().clone());
            }
            bufs.push(reinterpret_as_flat_i8(&layer.o_proj_weight));
            bufs.push(layer.post_attention_layernorm_weight.clone());
            bufs.push(reinterpret_as_flat_i8(&layer.gate_proj_weight));
            bufs.push(reinterpret_as_flat_i8(&layer.up_proj_weight));
            bufs.push(reinterpret_as_flat_i8(&layer.down_proj_weight));
        }

        bufs.push(self.final_norm_weight.clone());

        // lm_head: always f32 [hidden, vocab] for the quant path.
        // If tied, transpose embed_tokens (already f32) to [hidden, vocab].
        // If untied, ensure f32 (dequant for simplicity in this first pass).
        if let Some(lm_head) = &self.lm_head_weight {
            bufs.push(ensure_f32(lm_head.clone()));
        } else {
            bufs.push(transpose_2d(&self.embed_tokens_weight));
        }

        bufs
    }
}

/// Reinterpret a weight buffer's raw bytes as a flat 1D I8 buffer.
///
/// Used for quantized projection weights: the raw quantized bytes are kept
/// as-is and presented to the dequant-matmul kernel as a flat byte array.
fn reinterpret_as_flat_i8(buf: &Buffer) -> Buffer {
    let bytes = buf.as_bytes().to_vec();
    let len = bytes.len() as u64;
    Buffer::from_raw_bytes(bytes, &[len], DType::I8)
}

/// Convert a bf16 buffer to f32.
fn bf16_buf_to_f32(buf: &Buffer) -> Buffer {
    let src = buf.as_slice::<u16>();
    let mut f32_data = Vec::with_capacity(src.len());
    for &bits in src {
        f32_data.push(f32::from_bits((bits as u32) << 16));
    }
    Buffer::from_slice::<f32>(&f32_data, &buf.shape().0, DType::F32)
}

/// Convert an f32 buffer to bf16 (truncate).
fn f32_buf_to_bf16(buf: &Buffer) -> Buffer {
    let src = buf.as_slice::<f32>();
    let mut bf16_data = Vec::with_capacity(src.len());
    for &val in src {
        bf16_data.push((val.to_bits() >> 16) as u16);
    }
    Buffer::from_slice::<u16>(&bf16_data, &buf.shape().0, DType::BF16)
}

/// Ensure a buffer is f32 (convert from bf16 if needed).
fn ensure_f32(buf: Buffer) -> Buffer {
    if buf.dtype() == DType::BF16 {
        bf16_buf_to_f32(&buf)
    } else {
        buf
    }
}

/// Transpose a 2D buffer from [rows, cols] to [cols, rows].
fn transpose_2d(buf: &Buffer) -> Buffer {
    match buf.dtype() {
        DType::F32 => transpose_2d_f32(buf),
        DType::BF16 => transpose_2d_u16(buf, DType::BF16),
        _ => panic!("transpose_2d: unsupported dtype {:?}", buf.dtype()),
    }
}

fn transpose_2d_f32(buf: &Buffer) -> Buffer {
    let shape = buf.shape().0.clone();
    let rows = shape[0] as usize;
    let cols = shape[1] as usize;
    let src = buf.as_slice::<f32>();
    let mut dst = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
    Buffer::from_slice::<f32>(&dst, &[cols as u64, rows as u64], DType::F32)
}

/// Transpose 2D for 2-byte element types (bf16). Works on raw u16 values.
fn transpose_2d_u16(buf: &Buffer, dtype: DType) -> Buffer {
    let shape = buf.shape().0.clone();
    let rows = shape[0] as usize;
    let cols = shape[1] as usize;
    let src = buf.as_slice::<u16>();
    let mut dst = vec![0u16; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
    Buffer::from_slice::<u16>(&dst, &[cols as u64, rows as u64], dtype)
}

/// Load and organize transformer weights from a flat safetensors map.
///
/// All linear weight matrices are transposed from HF layout [out, in] to
/// matmul layout [in, out]. Embedding and layernorm weights are left as-is.
pub fn load_transformer_weights(
    config: &TransformerConfig,
    mut tensors: HashMap<String, Buffer>,
) -> Result<TransformerWeights, Box<dyn std::error::Error>> {
    // Detect whether biases exist by checking the first layer.
    let has_bias = tensors.contains_key("model.layers.0.self_attn.q_proj.bias");
    // Detect QK-norm (Qwen3-style) by checking for the first layer's q_norm weight.
    let has_qk_norm = tensors.contains_key("model.layers.0.self_attn.q_norm.weight");

    let take = |tensors: &mut HashMap<String, Buffer>,
                key: &str|
     -> Result<Buffer, Box<dyn std::error::Error>> {
        tensors
            .remove(key)
            .ok_or_else(|| format!("missing weight: {key}").into())
    };

    let embed_tokens_weight = {
        let w = take(&mut tensors, "model.embed_tokens.weight")?;
        // Embedding is a gather (not matmul), must be f32.
        if w.dtype() == DType::BF16 {
            bf16_buf_to_f32(&w)
        } else {
            w
        }
    };

    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    let pb = crate::progress::load_progress(config.num_hidden_layers);
    for i in 0..config.num_hidden_layers {
        // Layernorm, biases, QK-norm must be f32 (elementwise ops, not matmul).
        let input_layernorm_weight = ensure_f32(take(
            &mut tensors,
            &format!("model.layers.{i}.input_layernorm.weight"),
        )?);

        // Projection weights: keep whatever dtype they are (f32 or bf16).
        let q_proj_weight = transpose_2d(&take(
            &mut tensors,
            &format!("model.layers.{i}.self_attn.q_proj.weight"),
        )?);
        let q_proj_bias = if has_bias {
            Some(ensure_f32(take(
                &mut tensors,
                &format!("model.layers.{i}.self_attn.q_proj.bias"),
            )?))
        } else {
            None
        };

        let k_proj_weight = transpose_2d(&take(
            &mut tensors,
            &format!("model.layers.{i}.self_attn.k_proj.weight"),
        )?);
        let k_proj_bias = if has_bias {
            Some(ensure_f32(take(
                &mut tensors,
                &format!("model.layers.{i}.self_attn.k_proj.bias"),
            )?))
        } else {
            None
        };

        let v_proj_weight = transpose_2d(&take(
            &mut tensors,
            &format!("model.layers.{i}.self_attn.v_proj.weight"),
        )?);
        let v_proj_bias = if has_bias {
            Some(ensure_f32(take(
                &mut tensors,
                &format!("model.layers.{i}.self_attn.v_proj.bias"),
            )?))
        } else {
            None
        };

        let q_norm_weight = if has_qk_norm {
            Some(ensure_f32(take(
                &mut tensors,
                &format!("model.layers.{i}.self_attn.q_norm.weight"),
            )?))
        } else {
            None
        };
        let k_norm_weight = if has_qk_norm {
            Some(ensure_f32(take(
                &mut tensors,
                &format!("model.layers.{i}.self_attn.k_norm.weight"),
            )?))
        } else {
            None
        };

        let o_proj_weight = transpose_2d(&take(
            &mut tensors,
            &format!("model.layers.{i}.self_attn.o_proj.weight"),
        )?);

        let post_attention_layernorm_weight = ensure_f32(take(
            &mut tensors,
            &format!("model.layers.{i}.post_attention_layernorm.weight"),
        )?);

        let gate_proj_weight = transpose_2d(&take(
            &mut tensors,
            &format!("model.layers.{i}.mlp.gate_proj.weight"),
        )?);
        let up_proj_weight = transpose_2d(&take(
            &mut tensors,
            &format!("model.layers.{i}.mlp.up_proj.weight"),
        )?);
        let down_proj_weight = transpose_2d(&take(
            &mut tensors,
            &format!("model.layers.{i}.mlp.down_proj.weight"),
        )?);

        crate::progress::update_load(&pb, i + 1, &format!("layer {i}"));
        layers.push(LayerWeights {
            input_layernorm_weight,
            q_proj_weight,
            q_proj_bias,
            k_proj_weight,
            k_proj_bias,
            v_proj_weight,
            v_proj_bias,
            q_norm_weight,
            k_norm_weight,
            o_proj_weight,
            post_attention_layernorm_weight,
            gate_proj_weight,
            up_proj_weight,
            down_proj_weight,
        });
    }
    crate::progress::finish_load(&pb);

    let final_norm_weight = ensure_f32(take(&mut tensors, "model.norm.weight")?);

    let lm_head_weight = if config.tie_word_embeddings {
        None
    } else {
        Some(transpose_2d(&take(&mut tensors, "lm_head.weight")?))
    };

    Ok(TransformerWeights {
        embed_tokens_weight,
        layers,
        final_norm_weight,
        lm_head_weight,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hf::config::TransformerConfig;

    #[test]
    fn transpose_2d_basic() {
        // [2, 3] -> [3, 2]
        let buf =
            Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], crate::DType::F32);
        let t = transpose_2d(&buf);
        assert_eq!(t.shape().0, vec![3, 2]);
        assert_eq!(t.as_slice::<f32>(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn load_real_qwen2_weights() {
        let model_dir = std::path::Path::new(concat!(
            env!("HOME"),
            "/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/",
            "snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
        ));
        let config_path = model_dir.join("config.json");
        let safetensors_path = model_dir.join("model.safetensors");
        if !config_path.exists() || !safetensors_path.exists() {
            eprintln!("skipping: Qwen2.5-0.5B not found");
            return;
        }

        let config = TransformerConfig::load(&config_path).unwrap();
        let tensors = crate::hf::safetensors::load_safetensors(&safetensors_path).unwrap();
        let weights = load_transformer_weights(&config, tensors).unwrap();

        assert_eq!(weights.layers.len(), 24);
        // embed_tokens: [vocab=151936, hidden=896] - NOT transposed
        assert_eq!(weights.embed_tokens_weight.shape().0, vec![151936, 896]);
        // q_proj: originally [896, 896], transposed to [896, 896] (square, same shape)
        assert_eq!(weights.layers[0].q_proj_weight.shape().0, vec![896, 896]);
        // k_proj: originally [128, 896], transposed to [896, 128]
        assert_eq!(weights.layers[0].k_proj_weight.shape().0, vec![896, 128]);
        // v_proj: same as k_proj
        assert_eq!(weights.layers[0].v_proj_weight.shape().0, vec![896, 128]);
        // biases exist for Qwen2
        assert!(weights.layers[0].q_proj_bias.is_some());
        // gate_proj: originally [4864, 896], transposed to [896, 4864]
        assert_eq!(
            weights.layers[0].gate_proj_weight.shape().0,
            vec![896, 4864]
        );
        // lm_head is tied
        assert!(weights.lm_head_weight.is_none());
        // final norm
        assert_eq!(weights.final_norm_weight.shape().0, vec![896]);

        // Check flat buffers count.
        // Per layer: 9 weights (input_ln, q, k, v, o, post_attn_ln, gate, up, down)
        //           + 3 biases (q, k, v) when present = 12 total with biases, 9 without.
        let has_bias = weights.layers[0].q_proj_bias.is_some();
        let per_layer = if has_bias { 12 } else { 9 };
        let expected = 1 /* embed */ + 24 * per_layer + 1 /* final_norm */ + 1 /* lm_head (always present, transposed for tied) */;
        let flat = weights.to_slot_buffers(DType::F32);
        assert_eq!(flat.len(), expected);
    }
}
