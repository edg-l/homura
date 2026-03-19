use std::path::Path;

use crate::{DType, onnx::Model, runtime::Buffer, tokenizer::Tokenizer};

const EOS_TOKEN_ID: u32 = 50256;

/// GPT-2 text generator using a single decoder ONNX model (full-recompute per step).
///
/// Each generation step runs the full sequence through the model, which
/// recompiles for each new sequence length. This is correct but slow without
/// a compilation cache. Use `max_new_tokens` small for interactive use until
/// the compilation cache (Phase 9) is in place.
pub struct Generator {
    model: Model,
    tokenizer: Tokenizer,
}

impl Generator {
    /// Load from a directory containing:
    ///   - `decoder_model.onnx` (or `gpt2_decoder_model.onnx`)
    ///   - `vocab.json`
    ///   - `merges.txt`
    pub fn load(model_dir: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let dir = Path::new(model_dir);

        // Try both naming conventions.
        let model_path = if dir.join("decoder_model.onnx").exists() {
            dir.join("decoder_model.onnx")
        } else {
            dir.join("gpt2_decoder_model.onnx")
        };

        let vocab_path = dir.join("vocab.json");
        let merges_path = dir.join("merges.txt");

        log_info!("loading model from {}", model_path.display());
        let model = Model::load(&model_path)?;
        log_info!("loading tokenizer");
        let tokenizer =
            Tokenizer::from_files(vocab_path.to_str().unwrap(), merges_path.to_str().unwrap())?;

        Ok(Generator { model, tokenizer })
    }

    /// Generate up to `max_new_tokens` tokens after the prompt using greedy
    /// (argmax) sampling. Returns the generated text (prompt not included).
    ///
    /// Stops early on EOS token (token ID 50256).
    pub fn generate(&self, prompt: &str, max_new_tokens: usize) -> String {
        use std::time::Instant;

        let prompt_ids: Vec<i64> = self
            .tokenizer
            .encode(prompt)
            .into_iter()
            .map(|id| id as i64)
            .collect();

        if prompt_ids.is_empty() || max_new_tokens == 0 {
            return String::new();
        }

        log_info!(
            "starting generation: {} prompt tokens, max_new_tokens={}",
            prompt_ids.len(),
            max_new_tokens
        );

        let prompt_len = prompt_ids.len();
        let mut token_ids = prompt_ids;
        let mut generated_ids: Vec<u32> = Vec::with_capacity(max_new_tokens);
        let gen_start = Instant::now();

        for step in 0..max_new_tokens {
            let seq_len = token_ids.len();
            let step_start = Instant::now();

            let input_ids = Buffer::from_slice::<i64>(&token_ids, &[1, seq_len as u64], DType::I64);
            let mask: Vec<i64> = vec![1i64; seq_len];
            let attention_mask = Buffer::from_slice::<i64>(&mask, &[1, seq_len as u64], DType::I64);

            let outputs = match self.model.run(&[&input_ids, &attention_mask]) {
                Ok(o) => o,
                Err(e) => {
                    log_error!("model run failed (seq_len={}): {e}", seq_len);
                    break;
                }
            };

            // logits: [1, seq_len, 50257]
            let logits = &outputs[0];
            let next_token = argmax_last_token(logits, seq_len);

            let token_text = self.tokenizer.decode(&[next_token]);
            let elapsed = step_start.elapsed();
            log_info!(
                "step {}/{}: token={:?} seq_len={} elapsed={:.3}s",
                step + 1,
                max_new_tokens,
                token_text,
                seq_len,
                elapsed.as_secs_f64(),
            );

            if next_token == EOS_TOKEN_ID {
                log_info!("EOS token reached");
                break;
            }

            generated_ids.push(next_token);
            token_ids.push(next_token as i64);
        }

        let total = gen_start.elapsed();
        log_info!(
            "generation complete: {} tokens in {:.2}s ({:.3}s/tok)",
            generated_ids.len(),
            total.as_secs_f64(),
            if generated_ids.is_empty() {
                0.0
            } else {
                total.as_secs_f64() / generated_ids.len() as f64
            },
        );

        // Drop the prompt tokens, decode only the generated portion.
        let _ = prompt_len;
        self.tokenizer.decode(&generated_ids)
    }
}

/// Return the argmax over the vocab dimension at the last sequence position.
///
/// `logits` buffer has shape `[1, seq_len, vocab_size]` with F32 dtype.
/// Returns the token ID with the highest logit value.
fn argmax_last_token(logits: &Buffer, seq_len: usize) -> u32 {
    const VOCAB_SIZE: usize = 50257;
    argmax_at_position(logits, seq_len - 1, VOCAB_SIZE)
}

/// Return the argmax over the vocab dimension at a specific sequence position.
///
/// `logits` buffer has shape `[1, seq_len, vocab_size]` with F32 dtype.
/// `pos` is the zero-based sequence position to read from.
/// `vocab_size` is the size of the vocabulary dimension.
/// Returns the token ID with the highest logit value.
pub fn argmax_at_position(logits: &Buffer, pos: usize, vocab_size: usize) -> u32 {
    let data = logits.as_slice::<f32>();
    let offset = pos * vocab_size;
    let slice = &data[offset..offset + vocab_size];

    slice
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

// ── Sampling ──────────────────────────────────────────────────────────────────

/// Configuration for token sampling.
#[derive(Clone, Debug)]
pub struct SamplingConfig {
    /// Temperature: 1.0 = neutral, <1 sharper, >1 flatter. 0 = greedy.
    pub temperature: f32,
    /// Nucleus sampling: keep smallest set of tokens with cumulative prob >= top_p.
    pub top_p: f32,
    /// Repetition penalty: logits of already-generated tokens are divided by this.
    pub repetition_penalty: f32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            repetition_penalty: 1.1,
        }
    }
}

/// Sample a token from a logit slice using temperature, repetition penalty, and top-p.
///
/// `logits` is the raw logit vector for a single position (length = vocab_size).
/// `generated_ids` is the list of already-generated token IDs (for repetition penalty).
pub fn sample_token(logits: &[f32], config: &SamplingConfig, generated_ids: &[u32]) -> u32 {
    let mut logits = logits.to_vec();

    // Repetition penalty: divide logits of previously generated tokens.
    if config.repetition_penalty != 1.0 {
        for &id in generated_ids {
            let idx = id as usize;
            if idx < logits.len() {
                if logits[idx] > 0.0 {
                    logits[idx] /= config.repetition_penalty;
                } else {
                    logits[idx] *= config.repetition_penalty;
                }
            }
        }
    }

    // Temperature = 0 means greedy.
    if config.temperature <= 0.0 {
        return logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }

    // Apply temperature.
    let inv_temp = 1.0 / config.temperature;
    for l in &mut logits {
        *l *= inv_temp;
    }

    // Softmax.
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= sum;
    }

    // Top-p (nucleus) filtering.
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumulative = 0.0;
    let mut cutoff_idx = indexed.len();
    for (i, &(_, p)) in indexed.iter().enumerate() {
        cumulative += p;
        if cumulative >= config.top_p {
            cutoff_idx = i + 1;
            break;
        }
    }
    let candidates = &indexed[..cutoff_idx];

    // Renormalize and sample.
    let total: f32 = candidates.iter().map(|(_, p)| p).sum();
    let r = simple_random() * total;
    let mut accum = 0.0;
    for &(token_id, p) in candidates {
        accum += p;
        if accum >= r {
            return token_id as u32;
        }
    }
    candidates.last().map(|(id, _)| *id as u32).unwrap_or(0)
}

/// Simple pseudo-random f32 in [0, 1) using thread-local xorshift64.
fn simple_random() -> f32 {
    use std::cell::Cell;
    thread_local! {
        static STATE: Cell<u64> = Cell::new(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64 | 1
        );
    }
    STATE.with(|s| {
        let mut x = s.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        (x >> 40) as f32 / (1u64 << 24) as f32
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Slow test: loads decoder_model.onnx and generates 3 tokens.
    /// Each token triggers a recompilation (~34s each without cache).
    /// Run explicitly with: cargo test generate -- --ignored --nocapture
    #[test]
    #[ignore]
    fn generate_produces_tokens() {
        let generator = Generator::load("tests/fixtures")
            .expect("failed to load generator from tests/fixtures");
        let text = generator.generate("Hello", 3);
        assert!(!text.is_empty(), "should generate at least one token");
        println!("Generated: {text:?}");
    }
}
