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

        tracing::info!(path = %model_path.display(), "loading model");
        let model = Model::load(&model_path)?;
        tracing::info!("loading tokenizer");
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

        tracing::info!(
            prompt_tokens = prompt_ids.len(),
            max_new_tokens,
            "starting generation"
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
                    tracing::error!(seq_len, "model run failed: {e}");
                    break;
                }
            };

            // logits: [1, seq_len, 50257]
            let logits = &outputs[0];
            let next_token = argmax_last_token(logits, seq_len);

            let token_text = self.tokenizer.decode(&[next_token]);
            let elapsed = step_start.elapsed();
            tracing::info!(
                step = step + 1,
                max = max_new_tokens,
                token = ?token_text,
                seq_len,
                elapsed_s = elapsed.as_secs_f64(),
            );

            if next_token == EOS_TOKEN_ID {
                tracing::info!("EOS token reached");
                break;
            }

            generated_ids.push(next_token);
            token_ids.push(next_token as i64);
        }

        let total = gen_start.elapsed();
        tracing::info!(
            tokens = generated_ids.len(),
            total_s = total.as_secs_f64(),
            per_token_s = if generated_ids.is_empty() {
                0.0
            } else {
                total.as_secs_f64() / generated_ids.len() as f64
            },
            "generation complete"
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
