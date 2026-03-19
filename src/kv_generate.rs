use std::path::{Path, PathBuf};

use crate::log::{BOLD, BOLD_MAGENTA, CYAN, DIM, GREEN, RESET, YELLOW};
use crate::{
    DType,
    generate::{
        GenerationStats, Rng, SamplingConfig, argmax_at_position, escape_token_text, sample_token,
    },
    onnx::parser::Dim,
    onnx::{Model, parser},
    runtime::Buffer,
    tokenizer::Tokenizer,
};

// ── ModelConfig ───────────────────────────────────────────────────────────────

/// Architecture parameters auto-detected from the with-past ONNX model.
///
/// These are extracted from the KV cache input shapes so that the generator
/// works with any causal LM — not just GPT-2.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Number of KV tensors (layers × 2 for key+value). E.g. 24 for GPT-2 (12 layers).
    pub num_kv_tensors: usize,
    /// Number of attention heads. Extracted from KV input shape[1].
    pub num_heads: u64,
    /// Per-head dimension. Extracted from KV input shape[3].
    pub head_dim: u64,
    /// Fixed past sequence length used for the decode model's KV cache.
    pub max_seq_len: usize,
    /// Token ID that signals end-of-sequence. Stop generation on this token.
    pub eos_token_id: u32,
}

impl ModelConfig {
    /// Detect architecture from a with-past ONNX model file.
    ///
    /// Looks for dynamic inputs whose name contains `"past_key_values"` and
    /// reads `num_heads` (dim[1]) and `head_dim` (dim[3]) from the first match.
    pub fn from_onnx_model(
        path: &Path,
        max_seq_len: usize,
        eos_token_id: u32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let onnx = parser::parse_model(path)?;

        let kv_inputs: Vec<&parser::DynamicInput> = onnx
            .dynamic_inputs
            .iter()
            .filter(|inp| inp.name.contains("past_key_values"))
            .collect();

        if kv_inputs.is_empty() {
            return Err(format!("no 'past_key_values' inputs found in {}", path.display()).into());
        }

        let num_kv_tensors = kv_inputs.len();

        // KV shape: [batch, heads, past_seq_len, head_dim]
        let first_kv = kv_inputs[0];
        if first_kv.dims.len() < 4 {
            return Err(format!(
                "KV input '{}' has {} dims, expected 4",
                first_kv.name,
                first_kv.dims.len()
            )
            .into());
        }

        let num_heads = match &first_kv.dims[1] {
            Dim::Fixed(v) => *v,
            Dim::Symbolic(s) => {
                return Err(format!(
                    "KV input '{}' dim[1] (heads) is symbolic '{}', expected fixed",
                    first_kv.name, s
                )
                .into());
            }
        };

        let head_dim = match &first_kv.dims[3] {
            Dim::Fixed(v) => *v,
            Dim::Symbolic(s) => {
                return Err(format!(
                    "KV input '{}' dim[3] (head_dim) is symbolic '{}', expected fixed",
                    first_kv.name, s
                )
                .into());
            }
        };

        Ok(ModelConfig {
            num_kv_tensors,
            num_heads,
            head_dim,
            max_seq_len,
            eos_token_id,
        })
    }
}

// ── UnifiedKvGenerator ────────────────────────────────────────────────────

/// Single-model KV cache text generator for causal language models.
///
/// Uses a single unified ONNX model (extracted from the merged "If" wrapper)
/// for both prefill and decode phases. All sequence-length dims are compiled
/// as dynamic so the model compiles once and accepts any shapes at runtime.
///
/// Prefill passes empty KV tensors `[1, heads, 0, head_dim]`; the zero-dim
/// guard in `ExecutionPlan` skips kernels whose *outputs* are zero-element,
/// while Concat produces the correct non-zero output shape.
pub struct UnifiedKvGenerator {
    model: Model,
    tokenizer: Tokenizer,
    config: ModelConfig,
}

impl UnifiedKvGenerator {
    /// Load a unified KV cache generator from a directory.
    ///
    /// Looks for model files (first match wins):
    /// - `decoder_model_merged.onnx` (optimum export with If wrapper — auto-extracted)
    /// - `gpt2_decoder_model_merged.onnx`
    /// - `gpt2_unified_model.onnx` (pre-extracted then_branch)
    pub fn load(
        model_dir: &str,
        max_seq_len: usize,
        eos_token_id: u32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let dir = Path::new(model_dir);

        let model_path = find_unified_model(dir).ok_or_else(|| {
            format!(
                "no unified model found in {} \
                 (tried decoder_model_merged.onnx, gpt2_decoder_model_merged.onnx, \
                 gpt2_unified_model.onnx)",
                dir.display()
            )
        })?;

        log_info!("loading unified model from {}", model_path.display());
        let keep_dynamic: std::collections::HashSet<String> = [
            "sequence_length",
            "past_sequence_length",
            "attention_mask_sequence_length",
            // Output dim expressions from HF export annotations
            "past_sequence_length + 1",
            "past_sequence_length + sequence_length",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let model = Model::load_with_dynamic_dims(&model_path, keep_dynamic)?;

        log_info!("detecting model config from {}", model_path.display());
        let config = ModelConfig::from_onnx_model(&model_path, max_seq_len, eos_token_id)?;
        log_info!(
            "model config: num_kv_tensors={} num_heads={} head_dim={} max_seq_len={}",
            config.num_kv_tensors,
            config.num_heads,
            config.head_dim,
            config.max_seq_len,
        );

        let vocab_path = dir.join("vocab.json");
        let merges_path = dir.join("merges.txt");
        log_info!("loading tokenizer");
        let tokenizer = Tokenizer::from_files(
            vocab_path
                .to_str()
                .ok_or("vocab.json path is not valid UTF-8")?,
            merges_path
                .to_str()
                .ok_or("merges.txt path is not valid UTF-8")?,
        )?;

        Ok(UnifiedKvGenerator {
            model,
            tokenizer,
            config,
        })
    }

    /// Generate text using the unified model for both prefill and decode.
    ///
    /// Returns only the generated text (prompt excluded).
    pub fn generate(&self, prompt: &str, max_new_tokens: usize) -> String {
        self.generate_with_sampling(prompt, max_new_tokens, &SamplingConfig::default())
    }

    /// Generate text with explicit sampling configuration.
    pub fn generate_with_sampling(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        sampling: &SamplingConfig,
    ) -> String {
        let mut token_ids: Vec<i64> = self
            .tokenizer
            .encode(prompt)
            .into_iter()
            .map(|id| id as i64)
            .collect();

        if token_ids.is_empty() || max_new_tokens == 0 {
            return String::new();
        }

        let max_prompt = self.config.max_seq_len - 1;
        if token_ids.len() > max_prompt {
            log_warn!(
                "truncating prompt from {} to {} tokens",
                token_ids.len(),
                max_prompt
            );
            token_ids.truncate(max_prompt);
        }

        log_info!(
            "starting unified generation: {} prompt tokens, max_new_tokens={}",
            token_ids.len(),
            max_new_tokens,
        );

        // Prefill
        let step_start = std::time::Instant::now();
        let (mut next_token, kv_cache, mut real_pos) = match self.prefill(&token_ids) {
            Ok(r) => r,
            Err(e) => {
                log_error!("prefill failed: {e}");
                return String::new();
            }
        };
        let prefill_s = step_start.elapsed().as_secs_f64();
        log_info!("prefill complete in {prefill_s:.2}s");

        // Initialize persistent KV cache from prefill outputs.
        if let Err(e) = self.model.init_kv_cache(&kv_cache, self.config.max_seq_len) {
            log_error!("init_kv_cache failed: {e}");
            return String::new();
        }
        log_info!("KV cache initialized from prefill ({real_pos} positions)");

        let mut generated_ids: Vec<u32> = Vec::with_capacity(max_new_tokens);
        let mut rng = Rng::from_optional_seed(sampling.seed);
        let mut decode_times: Vec<std::time::Duration> = Vec::with_capacity(max_new_tokens);

        if next_token == self.config.eos_token_id {
            log_info!("EOS after prefill");
            return String::new();
        }
        generated_ids.push(next_token);
        let token_display = escape_token_text(&self.tokenizer.decode(&[next_token]));
        eprintln!(
            "  {CYAN}[1/{max_new_tokens}]{RESET} {BOLD}{token_display}{RESET} \
             {DIM}(prefill){RESET}"
        );

        // Decode loop
        for step in 1..max_new_tokens {
            if real_pos >= self.config.max_seq_len {
                log_warn!(
                    "context limit reached (max_seq_len={})",
                    self.config.max_seq_len
                );
                break;
            }

            let step_start = std::time::Instant::now();
            let logits = match self.decode_step_logits(next_token, real_pos) {
                Ok(l) => l,
                Err(e) => {
                    log_error!("decode step failed: {e}");
                    break;
                }
            };
            next_token = sample_token(&logits, sampling, &generated_ids, &mut rng);
            real_pos += 1;

            let step_elapsed = step_start.elapsed();
            decode_times.push(step_elapsed);
            let tok_s = 1.0 / step_elapsed.as_secs_f64();
            let token_display = escape_token_text(&self.tokenizer.decode(&[next_token]));
            eprintln!(
                "  {CYAN}[{}/{max_new_tokens}]{RESET} {BOLD}{token_display}{RESET}  \
                 {YELLOW}{:.0}ms{RESET}  {GREEN}{tok_s:.1} tok/s{RESET}",
                step + 1,
                step_elapsed.as_secs_f64() * 1000.0,
            );

            if next_token == self.config.eos_token_id {
                log_info!("EOS token reached");
                break;
            }
            generated_ids.push(next_token);

            // Check stop sequences against the generated text so far.
            if !sampling.stop_sequences.is_empty() {
                let text_so_far = self.tokenizer.decode(&generated_ids);
                if let Some(seq) = sampling
                    .stop_sequences
                    .iter()
                    .find(|s| text_so_far.contains(s.as_str()))
                {
                    log_info!("stop sequence {:?} reached", seq);
                    // Trim the generated text at the stop sequence.
                    if let Some(pos) = text_so_far.find(seq.as_str()) {
                        let trimmed_text = &text_so_far[..pos];
                        // Re-encode to get the right token count for the trimmed output.
                        let trimmed_ids: Vec<u32> = self.tokenizer.encode(trimmed_text);
                        generated_ids = trimmed_ids;
                    }
                    break;
                }
            }
        }

        let prefill_time = std::time::Duration::from_secs_f64(prefill_s);
        let stats = GenerationStats {
            prompt_tokens: token_ids.len(),
            generated_tokens: generated_ids.len(),
            prefill_time,
            decode_times,
            seed: sampling.seed,
        };
        eprintln!(
            "  {BOLD_MAGENTA}── done ──{RESET} {}",
            stats.format_summary()
        );

        self.tokenizer.decode(&generated_ids)
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Run the unified model in prefill mode: full prompt, empty KV cache.
    fn prefill(
        &self,
        token_ids: &[i64],
    ) -> Result<(u32, Vec<Buffer>, usize), Box<dyn std::error::Error>> {
        let seq_len = token_ids.len();

        // input_ids: [1, seq_len] — exact length, no bucket padding needed
        let input_ids = Buffer::from_slice::<i64>(token_ids, &[1, seq_len as u64], DType::I64);

        // attention_mask: [1, seq_len] — all ones
        let mask = vec![1i64; seq_len];
        let attention_mask = Buffer::from_slice::<i64>(&mask, &[1, seq_len as u64], DType::I64);

        // Empty KV cache: [1, heads, 0, head_dim] per tensor
        let kv_inputs: Vec<Buffer> = (0..self.config.num_kv_tensors)
            .map(|_| {
                Buffer::new(
                    &[1, self.config.num_heads, 0, self.config.head_dim],
                    DType::F32,
                )
            })
            .collect();

        // Args order must match ONNX model inputs:
        // [input_ids, attention_mask, kv[0], kv[1], ..., kv[23]]
        let mut args: Vec<&Buffer> = Vec::with_capacity(2 + self.config.num_kv_tensors);
        args.push(&input_ids);
        args.push(&attention_mask);
        for kv in &kv_inputs {
            args.push(kv);
        }

        let outputs = self.model.run(&args)?;

        // outputs[0] = logits [1, seq_len, vocab_size]
        let logits = &outputs[0];
        let vocab_size = logits.shape().0[2] as usize;
        let next_token = argmax_at_position(logits, seq_len - 1, vocab_size);

        // outputs[1..] = present KV [1, heads, 0+seq_len, head_dim]
        let kv_count = self.config.num_kv_tensors;
        if outputs.len() < 1 + kv_count {
            return Err(format!(
                "unified model returned {} outputs, expected at least {}",
                outputs.len(),
                1 + kv_count
            )
            .into());
        }
        let kv_cache: Vec<Buffer> = outputs[1..1 + kv_count].to_vec();

        Ok((next_token, kv_cache, seq_len))
    }

    /// Run one decode step using the persistent KV cache.
    ///
    /// Returns the raw logit vector for position 0 (single-token decode).
    fn decode_step_logits(
        &self,
        next_token: u32,
        real_pos: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let input_ids = Buffer::from_slice::<i64>(&[next_token as i64], &[1, 1], DType::I64);
        let mask_len = real_pos + 1;
        let mask_data = vec![1i64; mask_len];
        let attention_mask =
            Buffer::from_slice::<i64>(&mask_data, &[1, mask_len as u64], DType::I64);

        let outputs = self
            .model
            .run_kv(&[&input_ids, &attention_mask], self.config.max_seq_len)?;

        let logits = &outputs[0];
        let vocab_size = logits.shape().0[2] as usize;
        let data = logits.as_slice::<f32>();
        Ok(data[..vocab_size].to_vec())
    }
}

// ── Free functions ────────────────────────────────────────────────────────────

/// Return the first existing path from `candidates` under `dir`.
fn find_file(dir: &Path, candidates: &[&str]) -> Option<PathBuf> {
    for &name in candidates {
        let p = dir.join(name);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// Return the path to a unified/merged model file in the directory, if present.
fn find_unified_model(dir: &Path) -> Option<PathBuf> {
    find_file(
        dir,
        &[
            "decoder_model_merged.onnx",
            "gpt2_decoder_model_merged.onnx",
            "gpt2_unified_model.onnx",
        ],
    )
}

/// Return `true` if the given directory contains a unified/merged model file.
pub fn has_unified_model(model_dir: &Path) -> bool {
    find_unified_model(model_dir).is_some()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_config_field_values() {
        let config = ModelConfig {
            num_kv_tensors: 24,
            num_heads: 12,
            head_dim: 64,
            max_seq_len: 1024,
            eos_token_id: 50256,
        };
        assert_eq!(config.num_kv_tensors, 24);
        assert_eq!(config.num_heads, 12);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.max_seq_len, 1024);
        assert_eq!(config.eos_token_id, 50256);
    }

    #[test]
    fn has_unified_model_returns_false_for_missing_dir() {
        let dir = Path::new("/nonexistent_dir_that_does_not_exist_xyz");
        assert!(!has_unified_model(dir));
    }

    // ── Slow / integration tests ──────────────────────────────────────────────

    /// Load the unified GPT-2 ONNX model and generate 3 tokens.
    ///
    /// Requires `tests/fixtures/gpt2_unified_model.onnx` (or
    /// `gpt2_decoder_model_merged.onnx`), `vocab.json`, `merges.txt`.
    ///
    /// Run with: cargo test unified_kv_generate_produces_tokens -- --ignored --nocapture
    #[test]
    #[ignore]
    fn unified_kv_generate_produces_tokens() {
        let generator = UnifiedKvGenerator::load("tests/fixtures", 1024, 50256)
            .expect("failed to load UnifiedKvGenerator from tests/fixtures");
        let text = generator.generate("Hello", 3);
        assert!(!text.is_empty(), "should generate at least one token");
        eprintln!("Generated (unified): {text:?}");
    }
}
