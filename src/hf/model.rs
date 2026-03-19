//! Top-level HF model: load from safetensors + config.json, run inference.

use std::path::Path;
use std::sync::Mutex;

use crate::DType;
use crate::generate::{
    GenerativeModel, PrefillOutput, SamplingConfig, argmax_at_position, generate_streaming,
};
use crate::hf::config::TransformerConfig;
use crate::hf::precompute::{build_causal_mask, precompute_rope_cos_sin, slice_rope_for_positions};
use crate::hf::tokenizer::HfTokenizer;
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
    /// Expects: `config.json`, `model.safetensors`, `tokenizer.json`.
    pub fn load(model_dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let config = TransformerConfig::load(&model_dir.join("config.json"))?;
        let tensors =
            crate::hf::safetensors::load_safetensors(&model_dir.join("model.safetensors"))?;

        log_compile!("hf", "loaded {} tensors from safetensors", tensors.len());

        let weights = load_transformer_weights(&config, tensors)?;

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
    /// Returns all outputs: `[logits, present_k_0, present_v_0, ..., present_k_N, present_v_N]`.
    pub fn run(&self, input_ids: &Buffer) -> Result<Vec<Buffer>, Box<dyn std::error::Error>> {
        self.ensure_compiled()?;

        let seq_len = input_ids.shape().0[1] as usize;

        let mask = build_causal_mask(seq_len, 0);
        let positions: Vec<usize> = (0..seq_len).collect();
        let (cos, sin) = slice_rope_for_positions(&self.rope_cos, &self.rope_sin, &positions);

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

        let mut inputs: Vec<&Buffer> = vec![input_ids, &mask, &cos, &sin];
        for buf in &past_kv {
            inputs.push(buf);
        }

        let outputs = cs.plan.run(&inputs, &cs.weight_bufs);
        Ok(outputs)
    }

    /// Run KV-cached decode step.
    ///
    /// Returns logits only (KV cache updated internally).
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

    /// Generate text with KV-cached decoding.
    pub fn generate(
        &self,
        tokenizer: &HfTokenizer,
        prompt: &str,
        max_new_tokens: usize,
        max_seq_len: usize,
        sampling: &SamplingConfig,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let mut ctx = HfGenerationContext {
            model: self,
            tokenizer,
            max_seq_len,
            eos_override: None,
        };
        generate_streaming(&mut ctx, prompt, max_new_tokens, sampling)
    }
}

// ── HfGenerationContext ───────────────────────────────────────────────────────

/// Wraps HfModel + HfTokenizer for use with the shared `generate_streaming` loop.
///
/// For chat mode, create with `new_chat` which sets an EOS override and keeps
/// the KV cache alive across calls to `generate_streaming`.
pub struct HfGenerationContext<'a> {
    model: &'a HfModel,
    tokenizer: &'a HfTokenizer,
    max_seq_len: usize,
    /// Override the EOS token for chat mode (e.g. `<|im_end|>` instead of `<|endoftext|>`).
    eos_override: Option<u32>,
}

impl<'a> HfGenerationContext<'a> {
    /// Create a context for chat mode with a custom EOS token.
    pub fn new_chat(
        model: &'a HfModel,
        tokenizer: &'a HfTokenizer,
        max_seq_len: usize,
        eos_override: Option<u32>,
    ) -> Self {
        HfGenerationContext {
            model,
            tokenizer,
            max_seq_len,
            eos_override,
        }
    }
}

impl<'a> GenerativeModel for HfGenerationContext<'a> {
    fn encode(&self, prompt: &str) -> Vec<i64> {
        self.tokenizer
            .encode(prompt)
            .iter()
            .map(|&id| id as i64)
            .collect()
    }

    fn decode_tokens(&self, ids: &[u32]) -> String {
        self.tokenizer.decode(ids)
    }

    fn prefill(&mut self, token_ids: &[i64]) -> Result<PrefillOutput, Box<dyn std::error::Error>> {
        let seq_len = token_ids.len();
        let vocab_size = self.model.config.vocab_size;

        let step_start = std::time::Instant::now();
        let input_ids = Buffer::from_slice::<i64>(token_ids, &[1, seq_len as u64], DType::I64);

        if self.model.kv_cache_len() > 0 {
            // Incremental prefill: KV cache already exists from a prior turn.
            // Feed new tokens through run_kv to extend the cache.
            let prior_len = self.model.kv_cache_len(); // snapshot before run_kv advances cache
            let outputs = self.model.run_kv(&input_ids, self.max_seq_len)?;
            let logits = &outputs[0];
            let first_token = argmax_at_position(logits, seq_len - 1, vocab_size);
            let real_pos = prior_len + seq_len; // correct: prior cache len + new tokens
            let prefill_time = step_start.elapsed();

            Ok(PrefillOutput {
                first_token,
                prompt_len: seq_len,
                real_pos,
                prefill_time,
            })
        } else {
            // Full prefill: first turn, no KV cache yet.
            let num_kv = self.model.config.num_hidden_layers * 2;
            let outputs = self.model.run(&input_ids)?;

            let logits = &outputs[0];
            let first_token = argmax_at_position(logits, seq_len - 1, vocab_size);

            let kv_cache: Vec<Buffer> = outputs[1..1 + num_kv].to_vec();
            self.model.init_kv_cache(&kv_cache, self.max_seq_len)?;

            let prefill_time = step_start.elapsed();

            Ok(PrefillOutput {
                first_token,
                prompt_len: seq_len,
                real_pos: seq_len,
                prefill_time,
            })
        }
    }

    fn decode_step(
        &mut self,
        token: u32,
        _real_pos: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let vocab_size = self.model.config.vocab_size;
        let input_ids = Buffer::from_slice::<i64>(&[token as i64], &[1, 1], DType::I64);
        let outputs = self.model.run_kv(&input_ids, self.max_seq_len)?;

        let logits = &outputs[0];
        let logits_data = logits.as_slice::<f32>();
        Ok(logits_data[..vocab_size].to_vec())
    }

    fn eos_token_id(&self) -> u32 {
        self.eos_override
            .or(self.model.config.eos_token_id)
            .unwrap_or(u32::MAX)
    }

    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
}
