//! Top-level HF model: load from safetensors + config.json, run inference.

use std::path::Path;
use std::sync::Mutex;

use crate::DType;
use crate::hf::config::TransformerConfig;
use crate::hf::precompute::{build_causal_mask, precompute_rope_cos_sin, slice_rope_for_positions};
use crate::hf::weights::{TransformerWeights, load_transformer_weights};
use crate::runtime::{Buffer, ExecutionPlan};

/// A compiled HuggingFace transformer model.
pub struct HfModel {
    config: TransformerConfig,
    weights: TransformerWeights,
    /// Full RoPE cos/sin tables [max_seq_len, head_dim/2].
    rope_cos: Buffer,
    rope_sin: Buffer,
    /// Compiled execution plan (lazy: built on first run).
    state: Mutex<Option<CompiledState>>,
}

struct CompiledState {
    plan: ExecutionPlan,
    weight_bufs: Vec<Buffer>,
}

impl HfModel {
    /// Load a model from a HuggingFace model directory.
    ///
    /// Expects: `config.json`, `model.safetensors` (or sharded), `tokenizer.json`.
    pub fn load(model_dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let config = TransformerConfig::load(&model_dir.join("config.json"))?;
        let tensors =
            crate::hf::safetensors::load_safetensors(&model_dir.join("model.safetensors"))?;

        log_compile!("hf", "loaded {} tensors from safetensors", tensors.len());

        let weights = load_transformer_weights(&config, tensors)?;

        // Precompute RoPE tables for max_position_embeddings.
        let (rope_cos, rope_sin) = precompute_rope_cos_sin(
            config.head_dim(),
            config.max_position_embeddings,
            config.rope_theta,
        );

        Ok(HfModel {
            config,
            weights,
            rope_cos,
            rope_sin,
            state: Mutex::new(None),
        })
    }

    /// Compile the model if not already compiled.
    fn ensure_compiled(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut state = self.state.lock().unwrap();
        if state.is_some() {
            return Ok(());
        }

        let (plan, weight_bufs) =
            crate::hf::emitter::emit_transformer_plan(&self.config, &self.weights)
                .map_err(|e| format!("compilation failed: {e}"))?;

        *state = Some(CompiledState { plan, weight_bufs });
        Ok(())
    }

    /// Run prefill: process a full sequence of input tokens.
    ///
    /// `input_ids`: token IDs, shape `[1, seq_len]` I64.
    /// Returns logits `[1, seq_len, vocab_size]` F32.
    pub fn run(&self, input_ids: &Buffer) -> Result<Vec<Buffer>, Box<dyn std::error::Error>> {
        self.ensure_compiled()?;

        let seq_len = input_ids.shape().0[1] as usize;

        let mask = build_causal_mask(seq_len, 0);
        let positions: Vec<usize> = (0..seq_len).collect();
        let (cos, sin) = slice_rope_for_positions(&self.rope_cos, &self.rope_sin, &positions);

        // Build past_kv dummy buffers (empty: [1, kv_heads, 0, head_dim])
        let kv_heads = self.config.kv_heads() as u64;
        let head_dim = self.config.head_dim() as u64;
        let num_layers = self.config.num_hidden_layers;
        let mut past_kv: Vec<Buffer> = Vec::new();
        for _ in 0..num_layers {
            past_kv.push(Buffer::new(&[1, kv_heads, 0, head_dim], DType::F32));
            past_kv.push(Buffer::new(&[1, kv_heads, 0, head_dim], DType::F32));
        }

        let state = self.state.lock().unwrap();
        let cs = state.as_ref().unwrap();

        // Inputs: [input_ids, mask, cos, sin, past_k_0, past_v_0, ..., past_k_N, past_v_N]
        let mut inputs: Vec<&Buffer> = vec![input_ids, &mask, &cos, &sin];
        for buf in &past_kv {
            inputs.push(buf);
        }

        let outputs = cs.plan.run(&inputs, &cs.weight_bufs);
        Ok(outputs)
    }

    /// Run KV-cached decode.
    ///
    /// `input_ids`: shape `[1, seq_len]` I64.
    /// `max_seq_len`: maximum sequence length for KV cache allocation.
    /// Returns logits `[1, seq_len, vocab_size]` F32.
    pub fn run_kv(
        &self,
        input_ids: &Buffer,
        max_seq_len: usize,
    ) -> Result<Vec<Buffer>, Box<dyn std::error::Error>> {
        self.ensure_compiled()?;

        let mut state = self.state.lock().unwrap();
        let cs = state.as_mut().unwrap();

        let past_len = cs.plan.kv_cache_len();
        let seq_len = input_ids.shape().0[1] as usize;

        let mask = build_causal_mask(seq_len, past_len);
        let positions: Vec<usize> = (past_len..past_len + seq_len).collect();
        let (cos, sin) = slice_rope_for_positions(&self.rope_cos, &self.rope_sin, &positions);

        // run_kv expects only non-KV inputs (it fills past_kv from cache)
        let inputs: Vec<&Buffer> = vec![input_ids, &mask, &cos, &sin];
        let outputs = cs.plan.run_kv(&inputs, &cs.weight_bufs, max_seq_len);
        Ok(outputs)
    }

    /// Initialize KV cache from prefill KV outputs.
    pub fn init_kv_cache(
        &self,
        kv_buffers: &[Buffer],
        max_seq_len: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut state = self.state.lock().unwrap();
        let cs = state.as_mut().unwrap();
        cs.plan.init_kv_cache(kv_buffers, max_seq_len);
        Ok(())
    }

    /// Reset the KV cache (start a new sequence).
    pub fn reset_kv_cache(&self) {
        let mut state = self.state.lock().unwrap();
        if let Some(cs) = state.as_mut() {
            cs.plan.reset_kv_cache();
        }
    }

    /// Current KV cache sequence length.
    pub fn kv_cache_len(&self) -> usize {
        let state = self.state.lock().unwrap();
        state.as_ref().map(|cs| cs.plan.kv_cache_len()).unwrap_or(0)
    }

    /// Access the model config.
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }
}
