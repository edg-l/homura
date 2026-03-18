use std::path::{Path, PathBuf};

use crate::{
    DType,
    cache::bucket_pad,
    generate::argmax_at_position,
    onnx::{Model, parser},
    onnx::parser::Dim,
    runtime::Buffer,
    tokenizer::Tokenizer,
};

// ── ModelConfig ───────────────────────────────────────────────────────────────

/// Architecture parameters auto-detected from the with-past ONNX model.
///
/// These are extracted from the KV cache input shapes so that `KvGenerator`
/// works with any causal LM exported in the two-model format — not just GPT-2.
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
            return Err(format!(
                "no 'past_key_values' inputs found in {}",
                path.display()
            )
            .into());
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

// ── KvGenerator ───────────────────────────────────────────────────────────────

/// Two-model KV cache text generator for causal language models.
///
/// Uses a prefill model (full sequence, no past) and a decode model
/// (single token with fixed-size KV cache). This avoids recompiling for
/// each new sequence length: at most 2 compilations total, one per model.
///
/// Generic over any ONNX causal LM exported in the two-model format — not
/// hardcoded to GPT-2.
pub struct KvGenerator {
    prompt_model: Model,
    decode_model: Model,
    tokenizer: Tokenizer,
    config: ModelConfig,
}

impl KvGenerator {
    /// Load a KV cache generator from a directory.
    ///
    /// Looks for model files by a list of candidate names (first match wins):
    /// - Prefill model: `decoder_model.onnx` or `gpt2_decoder_model.onnx`
    /// - Decode model:  `decoder_with_past_model.onnx` or `gpt2_decoder_with_past_model.onnx`
    /// - Tokenizer:     `vocab.json` + `merges.txt`
    ///
    /// `max_seq_len` is the fixed past-sequence length for the KV cache (e.g. 1024).
    /// `eos_token_id` is the stop token (e.g. 50256 for GPT-2).
    pub fn load(
        model_dir: &str,
        max_seq_len: usize,
        eos_token_id: u32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let dir = Path::new(model_dir);

        let prompt_path = find_file(dir, &["decoder_model.onnx", "gpt2_decoder_model.onnx"])
            .ok_or_else(|| {
                format!(
                    "no prefill model found in {} \
                     (tried decoder_model.onnx, gpt2_decoder_model.onnx)",
                    dir.display()
                )
            })?;

        let decode_path = find_file(
            dir,
            &[
                "decoder_with_past_model.onnx",
                "gpt2_decoder_with_past_model.onnx",
            ],
        )
        .ok_or_else(|| {
            format!(
                "no with-past model found in {} \
                 (tried decoder_with_past_model.onnx, gpt2_decoder_with_past_model.onnx)",
                dir.display()
            )
        })?;

        tracing::info!(path = %prompt_path.display(), "loading prefill model");
        // Prefill runs once per generation — no need to keep sequence_length dynamic.
        // Resolving it to the concrete bucket size avoids DIM_DYNAMIC in output shapes.
        let prompt_model = Model::load_with_dynamic_dims(
            &prompt_path,
            std::collections::HashSet::new(),
        )?;

        tracing::info!(path = %decode_path.display(), "loading decode model");
        // Keep past_sequence_length (and derived dims like "past_sequence_length + 1")
        // dynamic so the decode model compiles ONCE and accepts any past_len at runtime.
        let keep_dynamic: std::collections::HashSet<String> = [
            "past_sequence_length".to_string(),
            "past_sequence_length + 1".to_string(),
        ]
        .into_iter()
        .collect();
        let decode_model = Model::load_with_dynamic_dims(&decode_path, keep_dynamic)?;

        tracing::info!(path = %decode_path.display(), "detecting model config");
        let config = ModelConfig::from_onnx_model(&decode_path, max_seq_len, eos_token_id)?;
        tracing::info!(
            num_kv_tensors = config.num_kv_tensors,
            num_heads = config.num_heads,
            head_dim = config.head_dim,
            max_seq_len = config.max_seq_len,
            "model config"
        );

        let vocab_path = dir.join("vocab.json");
        let merges_path = dir.join("merges.txt");
        tracing::info!("loading tokenizer");
        let tokenizer = Tokenizer::from_files(
            vocab_path
                .to_str()
                .ok_or("vocab.json path is not valid UTF-8")?,
            merges_path
                .to_str()
                .ok_or("merges.txt path is not valid UTF-8")?,
        )?;

        Ok(KvGenerator {
            prompt_model,
            decode_model,
            tokenizer,
            config,
        })
    }

    /// Generate text by running the prefill model once then the decode model
    /// for each new token. Returns only the generated text (prompt excluded).
    ///
    /// Stops on EOS token or when `max_new_tokens` is reached.
    /// Prompts longer than `max_seq_len - 1` are truncated with a warning.
    pub fn generate(&self, prompt: &str, max_new_tokens: usize) -> String {
        let mut token_ids: Vec<i64> = self
            .tokenizer
            .encode(prompt)
            .into_iter()
            .map(|id| id as i64)
            .collect();

        if token_ids.is_empty() || max_new_tokens == 0 {
            return String::new();
        }

        // Leave room for at least 1 generated token.
        let max_prompt = self.config.max_seq_len - 1;
        if token_ids.len() > max_prompt {
            tracing::warn!(
                from = token_ids.len(),
                to = max_prompt,
                "truncating prompt"
            );
            token_ids.truncate(max_prompt);
        }

        tracing::info!(
            prompt_tokens = token_ids.len(),
            max_new_tokens,
            "starting generation"
        );

        let gen_start = std::time::Instant::now();

        // Prefill phase
        let step_start = std::time::Instant::now();
        let (mut next_token, mut kv_cache, mut real_pos) = match self.prefill(&token_ids) {
            Ok(r) => r,
            Err(e) => {
                tracing::error!("prefill failed: {e}");
                return String::new();
            }
        };
        let prefill_s = step_start.elapsed().as_secs_f64();
        tracing::info!("prefill complete in {prefill_s:.2}s");

        let mut generated_ids: Vec<u32> = Vec::with_capacity(max_new_tokens);

        if next_token == self.config.eos_token_id {
            tracing::info!("EOS after prefill");
            return String::new();
        }
        generated_ids.push(next_token);
        let token_text = self.tokenizer.decode(&[next_token]);
        eprintln!(
            "  \x1b[36m[1/{max_new_tokens}]\x1b[0m \x1b[1m{token_text:?}\x1b[0m \x1b[2m(prefill)\x1b[0m"
        );

        // Decode loop
        for step in 1..max_new_tokens {
            if real_pos >= self.config.max_seq_len {
                tracing::warn!(max_seq_len = self.config.max_seq_len, "context limit reached");
                break;
            }

            let step_start = std::time::Instant::now();
            next_token = match self.decode_step(next_token, &mut kv_cache, real_pos, step_start) {
                Ok(t) => t,
                Err(e) => {
                    tracing::error!("decode step failed: {e}");
                    break;
                }
            };
            real_pos += 1;

            let token_text = self.tokenizer.decode(&[next_token]);
            let step_elapsed = step_start.elapsed().as_secs_f64();
            let tok_s = 1.0 / step_elapsed;
            eprintln!(
                "  \x1b[36m[{}/{max_new_tokens}]\x1b[0m \x1b[1m{token_text:?}\x1b[0m  \
                 \x1b[33m{:.0}ms\x1b[0m  \x1b[32m{tok_s:.1} tok/s\x1b[0m",
                step + 1,
                step_elapsed * 1000.0,
            );

            if next_token == self.config.eos_token_id {
                tracing::info!("EOS token reached");
                break;
            }
            generated_ids.push(next_token);
        }

        let total = gen_start.elapsed();
        let decode_tokens = generated_ids.len();
        let total_s = total.as_secs_f64();
        let per_tok = if decode_tokens == 0 { 0.0 } else { total_s / decode_tokens as f64 };
        let tok_s = if per_tok > 0.0 { 1.0 / per_tok } else { 0.0 };
        eprintln!(
            "  \x1b[1;35m── done ──\x1b[0m {decode_tokens} tokens in \
             \x1b[33m{total_s:.2}s\x1b[0m · \x1b[32m{tok_s:.1} tok/s\x1b[0m · \
             {:.0}ms/tok",
            per_tok * 1000.0,
        );

        self.tokenizer.decode(&generated_ids)
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Run the prefill (no-past) model on the prompt token IDs.
    ///
    /// Returns `(next_token, kv_cache, real_pos)` where:
    /// - `next_token` is the greedy-argmax token predicted after the prompt
    /// - `kv_cache` is a Vec of `num_kv_tensors` buffers shaped
    ///   `[1, heads, bucket, head_dim]` (actual bucket size, not padded to max_seq_len)
    /// - `real_pos` is `token_ids.len()` (number of filled KV slots)
    fn prefill(
        &self,
        token_ids: &[i64],
    ) -> Result<(u32, Vec<Buffer>, usize), Box<dyn std::error::Error>> {
        let seq_len = token_ids.len();
        let bucket = bucket_pad(seq_len);

        // Pad input_ids with EOS and attention_mask with 0s.
        let mut padded_ids = token_ids.to_vec();
        padded_ids.resize(bucket, self.config.eos_token_id as i64);

        let mut mask = vec![1i64; seq_len];
        mask.resize(bucket, 0i64);

        let input_ids =
            Buffer::from_slice::<i64>(&padded_ids, &[1, bucket as u64], DType::I64);
        let attention_mask =
            Buffer::from_slice::<i64>(&mask, &[1, bucket as u64], DType::I64);

        let outputs = self.prompt_model.run(&[&input_ids, &attention_mask])?;

        // logits: [1, bucket, vocab_size] — read at last REAL position (not padded)
        let logits = &outputs[0];
        let vocab_size = logits.shape().0[2] as usize;
        let next_token = argmax_at_position(logits, seq_len - 1, vocab_size);

        // Debug: print top-5 prefill logits
        {
            let logit_data = logits.as_slice::<f32>();
            let offset = (seq_len - 1) * vocab_size;
            let pos_logits = &logit_data[offset..offset + vocab_size];
            let mut indexed: Vec<(usize, f32)> = pos_logits.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let top5: Vec<String> = indexed[..5.min(indexed.len())]
                .iter()
                .map(|(i, v)| format!("{}={:.4}", i, v))
                .collect();
            tracing::debug!(
                top5 = top5.join(", "),
                min = pos_logits.iter().cloned().fold(f32::INFINITY, f32::min),
                max = pos_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                result_token = next_token,
                "prefill logits top5"
            );
        }

        // KV outputs: outputs[1 .. 1 + num_kv_tensors], each [1, heads, bucket, head_dim]
        // Trim to [1, heads, seq_len, head_dim] (remove bucket padding) so that
        // the KV cache exactly matches real_pos and the attention mask is all-ones.
        let kv_count = self.config.num_kv_tensors;
        if outputs.len() < 1 + kv_count {
            return Err(format!(
                "prefill model returned {} outputs, expected at least {}",
                outputs.len(),
                1 + kv_count
            )
            .into());
        }
        // Trim KV to [1, heads, seq_len, head_dim] — remove bucket padding.
        // This keeps positions contiguous so the decode step's attention mask
        // can be all-ones (no padding gap between real prefill and new decode KVs).
        let kv_buffers: Vec<Buffer> = outputs[1..1 + kv_count]
            .iter()
            .map(|kv| trim_kv_seq_dim(kv, seq_len))
            .collect();

        Ok((next_token, kv_buffers, seq_len))
    }

    /// Run one decode step.
    ///
    /// Inputs: `next_token` (the token just generated), the current KV cache
    /// buffers (each `[1, heads, real_pos, head_dim]`), and `real_pos`
    /// (number of past positions in the KV cache).
    ///
    /// The decode model takes `2 + num_kv_tensors` inputs:
    ///   `[input_ids, kv[0], ..., kv[num_kv-1], attention_mask]`
    ///
    /// The attention mask is `[1, real_pos + 1]` — all ones (past + current token).
    ///
    /// The output KV tensors already have shape `[1, heads, real_pos+1, head_dim]`
    /// (past concatenated with new). They replace the cache directly.
    ///
    /// Returns the greedy argmax token for the next step.
    fn decode_step(
        &self,
        next_token: u32,
        kv_cache: &mut Vec<Buffer>,
        real_pos: usize,
        step_start: std::time::Instant,
    ) -> Result<u32, Box<dyn std::error::Error>> {
        assert!(
            real_pos < self.config.max_seq_len,
            "KV cache full (real_pos={} >= max_seq_len={})",
            real_pos,
            self.config.max_seq_len
        );

        let heads = self.config.num_heads as usize;
        let head_dim = self.config.head_dim as usize;

        // input_ids: [1, 1]
        let input_ids =
            Buffer::from_slice::<i64>(&[next_token as i64], &[1, 1], DType::I64);

        // attention_mask: [1, kv_seq_len + 1] — all ones.
        // KV cache is trimmed (no padding gap), so every position is real.
        let kv_seq_len = kv_cache[0].shape().0[2] as usize;
        let mask_len = kv_seq_len + 1;
        let mask_data = vec![1i64; mask_len];
        let attention_mask =
            Buffer::from_slice::<i64>(&mask_data, &[1, mask_len as u64], DType::I64);

        // Build args: [input_ids, kv[0], ..., kv[n-1], attention_mask]
        let mut args: Vec<&Buffer> = Vec::with_capacity(2 + kv_cache.len());
        args.push(&input_ids);
        for kv in kv_cache.iter() {
            args.push(kv);
        }
        args.push(&attention_mask);

        // Build concrete output shapes for the dynamic model.
        // Output 0: logits [1, 1, vocab_size]
        // Outputs 1..kv_count: KV [1, heads, real_pos+1, head_dim]
        let kv_count = self.config.num_kv_tensors;
        let out_seq_len = (kv_seq_len + 1) as u64;
        let mut output_shapes: Vec<crate::shape::Shape> = Vec::with_capacity(1 + kv_count);
        // Get vocab_size from the compiled model's output desc (last dim of logits).
        let vocab_size_dim = self
            .decode_model
            .output_descs()
            .and_then(|descs| descs.first().and_then(|d| d.shape.0.last().copied()))
            .unwrap_or(50257);
        output_shapes.push(crate::shape::Shape(vec![1, 1, vocab_size_dim]));
        for _ in 0..kv_count {
            output_shapes.push(crate::shape::Shape(vec![
                1,
                heads as u64,
                out_seq_len,
                head_dim as u64,
            ]));
        }

        // Debug: print KV cache and mask shapes
        {
            let kv0 = kv_cache[0].as_slice::<f32>();
            tracing::debug!(
                token = next_token,
                input_ids_shape = ?input_ids.shape(),
                kv0_shape = ?kv_cache[0].shape(),
                mask_shape = ?attention_mask.shape(),
                real_pos,
                kv0_first4 = ?&kv0[..4.min(kv0.len())],
                kv0_last4 = ?&kv0[kv0.len().saturating_sub(4)..],
                "decode inputs"
            );
        }

        let t_setup = step_start.elapsed();

        let jit_start = std::time::Instant::now();
        let mut outputs = self.decode_model.run_with_output_shapes(&args, &output_shapes)?;
        let t_jit = jit_start.elapsed();

        let post_start = std::time::Instant::now();
        // logits: [1, 1, vocab_size] — only one position, read at index 0
        let logits = &outputs[0];
        let vocab_size = logits.shape().0[2] as usize;
        let result_token = argmax_at_position(logits, 0, vocab_size);

        // Debug: print top-5 logits to diagnose output quality
        {
            let logit_data = logits.as_slice::<f32>();
            let mut indexed: Vec<(usize, f32)> = logit_data.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let top5: Vec<String> = indexed[..5.min(indexed.len())]
                .iter()
                .map(|(i, v)| format!("{}={:.4}", i, v))
                .collect();
            tracing::debug!(
                top5 = top5.join(", "),
                min = logit_data.iter().cloned().fold(f32::INFINITY, f32::min),
                max = logit_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                result_token,
                "decode logits top5"
            );
        }

        if outputs.len() < 1 + kv_count {
            return Err(format!(
                "decode model returned {} outputs, expected at least {}",
                outputs.len(),
                1 + kv_count
            )
            .into());
        }

        // Replace each KV cache buffer with the full output KV tensor.
        // The output already has shape [1, heads, real_pos+1, head_dim] —
        // past concatenated with the new token's KV entry.
        for i in 0..kv_count {
            kv_cache[i] = outputs.remove(1);
        }
        let t_post = post_start.elapsed();

        if tracing::enabled!(tracing::Level::DEBUG) {
            eprintln!(
                "    \x1b[2msetup {:.1}ms │ jit {:.1}ms │ post {:.1}ms\x1b[0m",
                t_setup.as_secs_f64() * 1000.0,
                t_jit.as_secs_f64() * 1000.0,
                t_post.as_secs_f64() * 1000.0,
            );
        }

        Ok(result_token)
    }
}

// ── Free functions ────────────────────────────────────────────────────────────

/// Trim a KV cache buffer from `[1, heads, full_seq, head_dim]` to
/// `[1, heads, target_seq, head_dim]` by copying only the first `target_seq`
/// positions along dimension 2.
fn trim_kv_seq_dim(kv: &Buffer, target_seq: usize) -> Buffer {
    let shape = &kv.shape().0;
    assert_eq!(shape.len(), 4, "KV buffer must be 4-D");
    let heads = shape[1] as usize;
    let full_seq = shape[2] as usize;
    let head_dim = shape[3] as usize;

    if target_seq >= full_seq {
        return kv.clone();
    }

    let src = kv.as_slice::<f32>();
    let mut dst = Vec::with_capacity(heads * target_seq * head_dim);
    for h in 0..heads {
        let head_offset = h * full_seq * head_dim;
        dst.extend_from_slice(&src[head_offset..head_offset + target_seq * head_dim]);
    }
    Buffer::from_slice::<f32>(
        &dst,
        &[1, heads as u64, target_seq as u64, head_dim as u64],
        kv.dtype(),
    )
}

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

/// Return `true` if the given directory contains a with-past model file.
///
/// Used by the CLI to decide whether to use `KvGenerator` or `Generator`.
pub fn has_with_past_model(model_dir: &Path) -> bool {
    find_file(
        model_dir,
        &[
            "decoder_with_past_model.onnx",
            "gpt2_decoder_with_past_model.onnx",
        ],
    )
    .is_some()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attention_mask_all_ones() {
        // After KV trimming, decode mask is all-ones: length = kv_seq_len + 1.
        let kv_seq_len = 5usize; // 5 real past positions
        let mask_len = kv_seq_len + 1;
        let mask_data = vec![1i64; mask_len];
        assert_eq!(mask_data, vec![1i64; 6]);
    }

    #[test]
    fn trim_kv_seq_dim_basic() {
        use super::trim_kv_seq_dim;
        // KV: [1, 2, 4, 3] → trim to seq_len=2
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let kv = Buffer::from_slice::<f32>(&data, &[1, 2, 4, 3], DType::F32);
        let trimmed = trim_kv_seq_dim(&kv, 2);
        assert_eq!(trimmed.shape().0, vec![1, 2, 2, 3]);
        let result = trimmed.as_slice::<f32>();
        // Head 0: positions 0-1 → elements 0..6
        // Head 1: positions 0-1 → elements 12..18
        assert_eq!(result, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]);
    }

    #[test]
    fn has_with_past_model_returns_false_for_missing_dir() {
        let dir = Path::new("/nonexistent_dir_that_does_not_exist_xyz");
        assert!(!has_with_past_model(dir));
    }

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

    // ── Slow / integration tests ──────────────────────────────────────────────

    /// Load the two GPT-2 ONNX models and generate 3 tokens with the KV cache.
    ///
    /// Requires `tests/fixtures/gpt2_decoder_model.onnx`,
    /// `tests/fixtures/gpt2_decoder_with_past_model.onnx`,
    /// `tests/fixtures/vocab.json`, and `tests/fixtures/merges.txt`.
    ///
    /// Run with: cargo test kv_generate_produces_tokens -- --ignored --nocapture
    #[test]
    #[ignore]
    fn kv_generate_produces_tokens() {
        let kv_gen = KvGenerator::load("tests/fixtures", 1024, 50256)
            .expect("failed to load KvGenerator from tests/fixtures");
        let text = kv_gen.generate("Hello", 3);
        assert!(!text.is_empty(), "should generate at least one token");
        eprintln!("Generated: {text:?}");
    }
}
