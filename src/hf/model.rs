//! Top-level HF model: load from safetensors + config.json, run inference.

use std::path::Path;
use std::sync::Mutex;

use crate::DType;
use crate::generate::{
    GenerationStats, Rng, SamplingConfig, argmax_at_position, escape_token_text, sample_token,
};
use crate::hf::config::TransformerConfig;
use crate::hf::precompute::{build_causal_mask, precompute_rope_cos_sin, slice_rope_for_positions};
use crate::hf::tokenizer::HfTokenizer;
use crate::hf::weights::{TransformerWeights, load_transformer_weights};
use crate::log::{BOLD, BOLD_MAGENTA, CYAN, DIM, GREEN, RESET, YELLOW};
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
        let token_ids: Vec<i64> = tokenizer
            .encode(prompt)
            .iter()
            .map(|&id| id as i64)
            .collect();

        if token_ids.is_empty() || max_new_tokens == 0 {
            return Ok(String::new());
        }

        let eos_token_id = self.config.eos_token_id.unwrap_or(u32::MAX);
        let vocab_size = self.config.vocab_size;
        let num_kv = self.config.num_hidden_layers * 2;

        log_info!(
            "starting generation: {} prompt tokens, max_new_tokens={}",
            token_ids.len(),
            max_new_tokens,
        );

        // Prefill
        let step_start = std::time::Instant::now();
        let seq_len = token_ids.len();
        let input_ids = Buffer::from_slice::<i64>(&token_ids, &[1, seq_len as u64], DType::I64);
        let outputs = self.run(&input_ids)?;

        let logits = &outputs[0];
        let next_token = argmax_at_position(logits, seq_len - 1, vocab_size);

        // Extract KV cache from prefill outputs (outputs[1..1+num_kv])
        let kv_cache: Vec<Buffer> = outputs[1..1 + num_kv].to_vec();
        self.init_kv_cache(&kv_cache, max_seq_len)?;

        let prefill_s = step_start.elapsed().as_secs_f64();
        log_info!("prefill complete in {prefill_s:.2}s ({seq_len} tokens)");

        let mut generated_ids: Vec<u32> = Vec::with_capacity(max_new_tokens);
        let mut rng = Rng::from_optional_seed(sampling.seed);
        let mut decode_times: Vec<std::time::Duration> = Vec::with_capacity(max_new_tokens);
        let verbose = crate::log::enabled(crate::log::Level::Debug);
        let use_stdout = atty::is(atty::Stream::Stdout);

        if next_token == eos_token_id {
            log_info!("EOS after prefill");
            return Ok(String::new());
        }
        generated_ids.push(next_token);
        let token_text = tokenizer.decode(&[next_token]);
        // Stream to stdout
        if use_stdout {
            print!("{token_text}");
            let _ = std::io::Write::flush(&mut std::io::stdout());
        }
        if verbose {
            let token_display = escape_token_text(&token_text);
            eprintln!(
                "  {CYAN}[1/{max_new_tokens}]{RESET} {BOLD}{token_display}{RESET} {DIM}(prefill){RESET}"
            );
        }

        // Decode loop
        let mut current_token = next_token;
        for step in 1..max_new_tokens {
            if seq_len + step >= max_seq_len {
                log_warn!("context limit reached (max_seq_len={})", max_seq_len);
                break;
            }

            let step_start = std::time::Instant::now();
            let input_ids = Buffer::from_slice::<i64>(&[current_token as i64], &[1, 1], DType::I64);
            let outputs = self.run_kv(&input_ids, max_seq_len)?;

            let logits = &outputs[0];
            let logits_data = logits.as_slice::<f32>();
            let logits_vec = logits_data[..vocab_size].to_vec();
            current_token = sample_token(&logits_vec, sampling, &generated_ids, &mut rng);

            let step_elapsed = step_start.elapsed();
            decode_times.push(step_elapsed);

            let token_text = tokenizer.decode(&[current_token]);
            // Stream to stdout
            if use_stdout {
                print!("{token_text}");
                let _ = std::io::Write::flush(&mut std::io::stdout());
            }
            if verbose {
                let tok_s = 1.0 / step_elapsed.as_secs_f64();
                let token_display = escape_token_text(&token_text);
                eprintln!(
                    "  {CYAN}[{}/{max_new_tokens}]{RESET} {BOLD}{token_display}{RESET}  \
                     {YELLOW}{:.0}ms{RESET}  {GREEN}{tok_s:.1} tok/s{RESET}",
                    step + 1,
                    step_elapsed.as_secs_f64() * 1000.0,
                );
            }

            if current_token == eos_token_id {
                break;
            }
            generated_ids.push(current_token);

            if !sampling.stop_sequences.is_empty() {
                let text_so_far = tokenizer.decode(&generated_ids);
                if sampling
                    .stop_sequences
                    .iter()
                    .any(|s| text_so_far.contains(s.as_str()))
                {
                    break;
                }
            }
        }

        if use_stdout {
            println!();
        }
        let prefill_time = std::time::Duration::from_secs_f64(prefill_s);
        let stats = GenerationStats {
            prompt_tokens: seq_len,
            generated_tokens: generated_ids.len(),
            prefill_time,
            decode_times,
            seed: sampling.seed,
        };
        eprintln!(
            "\n  {BOLD_MAGENTA}-- done --{RESET} {}",
            stats.format_summary()
        );

        Ok(tokenizer.decode(&generated_ids))
    }
}
