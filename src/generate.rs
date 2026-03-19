use std::path::Path;

use crate::{DType, onnx::Model, runtime::Buffer, tokenizer::Tokenizer};

const EOS_TOKEN_ID: u32 = 50256;

// ── GenerativeModel trait ─────────────────────────────────────────────────────

/// Output from the prefill phase.
pub struct PrefillOutput {
    pub first_token: u32,
    pub prompt_len: usize,
    /// Position counter after prefill (= prompt_len).
    pub real_pos: usize,
    pub prefill_time: std::time::Duration,
}

/// Backend-agnostic interface for KV-cached causal language model generation.
pub trait GenerativeModel {
    fn encode(&self, prompt: &str) -> Vec<i64>;
    fn decode_tokens(&self, ids: &[u32]) -> String;
    fn prefill(&mut self, token_ids: &[i64]) -> Result<PrefillOutput, Box<dyn std::error::Error>>;
    fn decode_step(
        &mut self,
        token: u32,
        real_pos: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>>;
    fn eos_token_id(&self) -> u32;
    fn max_seq_len(&self) -> usize;
}

/// Shared streaming generation loop for any `GenerativeModel`.
///
/// Handles prefill, decode loop, streaming to stdout, verbose token logging,
/// stop sequences, and the stats summary line.
///
/// If `show_prompt` is false, the prompt text is not echoed to stdout
/// (used in chat mode where the prompt is template markup, not user text).
pub fn generate_streaming(
    model: &mut impl GenerativeModel,
    prompt: &str,
    max_new_tokens: usize,
    sampling: &SamplingConfig,
) -> Result<String, Box<dyn std::error::Error>> {
    let think = ThinkConfig {
        token_ids: None,
        style_content: false,
    };
    let token_ids = model.encode(prompt);
    generate_streaming_core(model, &token_ids, max_new_tokens, sampling, true, think)
}

/// Configuration for think-block handling during generation.
#[derive(Clone, Copy)]
pub struct ThinkConfig {
    /// Token IDs for `<think>` and `</think>`. Both tags are always hidden.
    pub token_ids: Option<(u32, u32)>,
    /// Whether to style the content between tags (dim gray). When false,
    /// the content is hidden entirely (for non-thinking mode where the model
    /// emits empty `<think></think>` blocks).
    pub style_content: bool,
}

/// Like `generate_streaming` but suppresses the prompt echo.
pub fn generate_streaming_no_echo(
    model: &mut impl GenerativeModel,
    prompt: &str,
    max_new_tokens: usize,
    sampling: &SamplingConfig,
    think: ThinkConfig,
) -> Result<String, Box<dyn std::error::Error>> {
    let token_ids = model.encode(prompt);
    generate_streaming_from_ids(model, &token_ids, max_new_tokens, sampling, false, think)
}

/// Like `generate_streaming_no_echo` but takes pre-tokenized IDs directly,
/// avoiding a decode→re-encode round-trip that loses special tokens.
pub fn generate_streaming_from_ids(
    model: &mut impl GenerativeModel,
    token_ids: &[i64],
    max_new_tokens: usize,
    sampling: &SamplingConfig,
    show_prompt: bool,
    think: ThinkConfig,
) -> Result<String, Box<dyn std::error::Error>> {
    generate_streaming_core(
        model,
        token_ids,
        max_new_tokens,
        sampling,
        show_prompt,
        think,
    )
}

fn generate_streaming_core(
    model: &mut impl GenerativeModel,
    token_ids: &[i64],
    max_new_tokens: usize,
    sampling: &SamplingConfig,
    show_prompt: bool,
    think: ThinkConfig,
) -> Result<String, Box<dyn std::error::Error>> {
    use crate::log::{BOLD, CYAN, DIM, GREEN, RESET, YELLOW};
    use std::io::Write;

    if token_ids.is_empty() || max_new_tokens == 0 {
        return Ok(String::new());
    }

    log_info!(
        "starting generation: {} prompt tokens, max_new_tokens={}",
        token_ids.len(),
        max_new_tokens,
    );

    let prefill_out = model.prefill(token_ids)?;
    log_info!(
        "prefill complete in {:.2}s",
        prefill_out.prefill_time.as_secs_f64()
    );

    let prompt_len = prefill_out.prompt_len;
    let mut real_pos = prefill_out.real_pos;
    let first_token = prefill_out.first_token;

    let mut generated_ids: Vec<u32> = Vec::with_capacity(max_new_tokens);
    let mut rng = Rng::from_optional_seed(sampling.seed);
    let mut decode_times: Vec<std::time::Duration> = Vec::with_capacity(max_new_tokens);
    let mut generated_text = String::new();
    let verbose = crate::log::enabled(crate::log::Level::Debug);
    let use_stdout = atty::is(atty::Stream::Stdout);
    // Only show decode progress bar when stdout is NOT a terminal (piped output),
    // so it doesn't fight with streaming token text on the same terminal.
    let decode_pb = if !verbose && !use_stdout {
        let pb = crate::progress::decode_progress(max_new_tokens);
        Some(pb)
    } else {
        None
    };
    let mut in_think_block = false;
    let mut skip_ws_after_think = false;
    let think_tokens = think.token_ids;

    // Print prompt to stdout after all log lines, before tokens stream.
    if use_stdout && show_prompt {
        let prompt_ids: Vec<u32> = token_ids.iter().map(|&id| id as u32).collect();
        let prompt_text = model.decode_tokens(&prompt_ids);
        print!("{prompt_text}");
        let _ = std::io::stdout().flush();
    }

    if first_token == model.eos_token_id() {
        log_info!("EOS after prefill");
        return Ok(String::new());
    }
    generated_ids.push(first_token);
    let token_text = model.decode_tokens(&[first_token]);
    if use_stdout {
        print_token_styled(
            first_token,
            &token_text,
            think_tokens,
            think.style_content,
            &mut in_think_block,
            &mut skip_ws_after_think,
        );
    }
    if verbose {
        let token_display = escape_token_text(&token_text);
        eprintln!(
            "  {CYAN}[1/{max_new_tokens}]{RESET} {BOLD}{token_display}{RESET} \
             {DIM}(prefill){RESET}"
        );
    } else if let Some(ref pb) = decode_pb {
        crate::progress::update_decode(pb, 1, 0.0, 0.0);
    }

    // Decode loop
    let mut current_token = first_token;
    for step in 1..max_new_tokens {
        if real_pos >= model.max_seq_len() {
            log_warn!(
                "context limit reached (max_seq_len={})",
                model.max_seq_len()
            );
            break;
        }

        let step_start = std::time::Instant::now();
        let logits = model.decode_step(current_token, real_pos)?;
        current_token = sample_token(&logits, sampling, &generated_ids, &mut rng);
        real_pos += 1;

        let step_elapsed = step_start.elapsed();
        decode_times.push(step_elapsed);

        let token_text = model.decode_tokens(&[current_token]);
        if use_stdout {
            print_token_styled(
                current_token,
                &token_text,
                think_tokens,
                think.style_content,
                &mut in_think_block,
                &mut skip_ws_after_think,
            );
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
        } else if let Some(ref pb) = decode_pb {
            let tok_s = 1.0 / step_elapsed.as_secs_f64();
            let ms = step_elapsed.as_secs_f64() * 1000.0;
            crate::progress::update_decode(pb, step + 1, tok_s, ms);
        }

        if current_token == model.eos_token_id() {
            break;
        }
        generated_ids.push(current_token);

        // Check stop sequences against the generated text so far.
        if !sampling.stop_sequences.is_empty() {
            generated_text.push_str(&token_text);
            if let Some(seq) = sampling
                .stop_sequences
                .iter()
                .find(|s| generated_text.contains(s.as_str()))
            {
                log_info!("stop sequence {:?} reached", seq);
                if let Some(pos) = generated_text.find(seq.as_str()) {
                    let trimmed_text = &generated_text[..pos];
                    let trimmed_ids: Vec<u32> = model
                        .encode(trimmed_text)
                        .into_iter()
                        .map(|id| id as u32)
                        .collect();
                    generated_ids = trimmed_ids;
                }
                break;
            }
        }
    }

    if use_stdout {
        // Reset style if we ended mid-think-block.
        if in_think_block {
            print!("\x1b[0m");
        }
        println!();
    }

    if let Some(ref pb) = decode_pb {
        crate::progress::finish_decode(pb);
    }

    let stats = GenerationStats {
        prompt_tokens: prompt_len,
        generated_tokens: generated_ids.len(),
        prefill_time: prefill_out.prefill_time,
        decode_times,
        seed: sampling.seed,
    };
    let output_text = model.decode_tokens(&generated_ids);
    crate::progress::print_stats(&stats, &output_text);

    Ok(output_text)
}

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

// ── Think token styling ──────────────────────────────────────────────────────

/// Dark gray + italic for thinking content. The gray is universally supported;
/// italic renders on terminals that support it and is ignored on others.
const THINK_STYLE: &str = "\x1b[90;3m";
const STYLE_RESET: &str = "\x1b[0m";

/// Print a token to stdout, handling think-block styling by token ID.
///
/// - Tag tokens (open/close) are always hidden.
/// - When `style_content` is true, content between tags is shown in gray.
/// - When `style_content` is false, content between tags is hidden entirely
///   (for non-thinking mode where the template inserts empty think blocks).
/// State: true after exiting a hidden think block, until non-whitespace appears.
/// Used to swallow the trailing `\n\n` after `</think>` in non-thinking mode.
fn print_token_styled(
    token_id: u32,
    token_text: &str,
    think_tokens: Option<(u32, u32)>,
    style_content: bool,
    in_think: &mut bool,
    skip_whitespace: &mut bool,
) {
    use std::io::Write;

    if let Some((open_id, close_id)) = think_tokens {
        if token_id == open_id {
            *in_think = true;
            if style_content {
                print!("{THINK_STYLE}");
                let _ = std::io::stdout().flush();
            }
            return;
        }
        if token_id == close_id {
            *in_think = false;
            if style_content {
                print!("{STYLE_RESET}");
            } else {
                // Skip whitespace tokens that follow a hidden think block.
                *skip_whitespace = true;
            }
            let _ = std::io::stdout().flush();
            return;
        }
        // Inside think block: show styled or hide entirely.
        if *in_think && !style_content {
            return;
        }
    }

    // After a hidden think block, skip whitespace-only tokens.
    if *skip_whitespace {
        if token_text.trim().is_empty() {
            return;
        }
        *skip_whitespace = false;
    }

    print!("{token_text}");
    let _ = std::io::stdout().flush();
}

// ── Display helpers ──────────────────────────────────────────────────────────

/// Escape control characters in token text for single-line display.
/// Newlines become `\n`, tabs become `\t`, other control chars become `\xNN`.
pub fn escape_token_text(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => {
                for b in c.to_string().bytes() {
                    out.push_str(&format!("\\x{b:02x}"));
                }
            }
            c => out.push(c),
        }
    }
    out
}

// ── RNG ──────────────────────────────────────────────────────────────────────

/// Seedable xorshift64 RNG for token sampling.
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed | 1 } // ensure nonzero (xorshift64 fixpoint)
    }

    pub fn from_system_time() -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        Self::new(seed)
    }

    /// Construct from an optional seed: use the seed if provided, else system time.
    pub fn from_optional_seed(seed: Option<u64>) -> Self {
        match seed {
            Some(s) => Self::new(s),
            None => Self::from_system_time(),
        }
    }

    /// Returns f32 in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state >> 40) as f32 / (1u64 << 24) as f32
    }
}

// ── Sampling ──────────────────────────────────────────────────────────────────

/// Configuration for token sampling.
#[derive(Clone, Debug)]
pub struct SamplingConfig {
    /// Temperature: 1.0 = neutral, <1 sharper, >1 flatter. 0 = greedy.
    pub temperature: f32,
    /// Top-k: keep only the k most likely tokens before sampling. 0 = disabled.
    pub top_k: usize,
    /// Nucleus sampling: keep smallest set of tokens with cumulative prob >= top_p.
    pub top_p: f32,
    /// min_p: filter tokens with prob < min_p * max_prob. 0 = disabled.
    pub min_p: f32,
    /// Repetition penalty: logits of already-generated tokens are divided by this.
    pub repetition_penalty: f32,
    /// Frequency penalty: subtract freq_penalty * count(token) from logits. 0 = off.
    pub frequency_penalty: f32,
    /// Presence penalty: subtract presence_penalty for any previously seen token. 0 = off.
    pub presence_penalty: f32,
    /// Stop generation when any of these strings appear in the output.
    pub stop_sequences: Vec<String>,
    /// RNG seed for reproducible sampling. None = use system time.
    pub seed: Option<u64>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            min_p: 0.0,
            repetition_penalty: 1.1,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            stop_sequences: vec![],
            seed: None,
        }
    }
}

/// Sample a token from a logit slice using temperature, penalties, and filtering.
///
/// Sampling order:
/// 1. Repetition penalty (multiplicative, existing tokens)
/// 2. Frequency penalty (additive, proportional to count)
/// 3. Presence penalty (additive, flat for any seen token)
/// 4. Greedy early-return if temperature=0
/// 5. Temperature scaling
/// 6. Softmax
/// 7. min_p filtering
/// 8. top_k
/// 9. top_p
/// 10. Sample from distribution
pub fn sample_token(
    logits: &[f32],
    config: &SamplingConfig,
    generated_ids: &[u32],
    rng: &mut Rng,
) -> u32 {
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

    // Frequency and presence penalties (additive).
    if config.frequency_penalty != 0.0 || config.presence_penalty != 0.0 {
        let mut counts: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for &id in generated_ids {
            *counts.entry(id as usize).or_insert(0) += 1;
        }
        for (&idx, &count) in &counts {
            if idx < logits.len() {
                logits[idx] -= config.frequency_penalty * count as f32;
                logits[idx] -= config.presence_penalty;
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

    // min_p: filter tokens with prob < min_p * max_prob.
    if config.min_p > 0.0 {
        let max_prob = probs.iter().cloned().fold(0.0f32, f32::max);
        let threshold = config.min_p * max_prob;
        for p in &mut probs {
            if *p < threshold {
                *p = 0.0;
            }
        }
    }

    // Sort by probability descending for top-k and top-p filtering.
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Top-k: hard cutoff at k most likely tokens.
    if config.top_k > 0 && config.top_k < indexed.len() {
        indexed.truncate(config.top_k);
    }

    // Top-p (nucleus) filtering.
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
    let r = rng.next_f32() * total;
    let mut accum = 0.0;
    for &(token_id, p) in candidates {
        accum += p;
        if accum >= r {
            return token_id as u32;
        }
    }
    candidates.last().map(|(id, _)| *id as u32).unwrap_or(0)
}

// ── Generation stats ─────────────────────────────────────────────────────────

/// Statistics collected during generation for the summary line.
pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_time: std::time::Duration,
    pub decode_times: Vec<std::time::Duration>,
    pub seed: Option<u64>,
}

impl GenerationStats {
    pub fn format_summary(&self) -> String {
        let decode_total: f64 = self.decode_times.iter().map(|d| d.as_secs_f64()).sum();
        let total = self.prefill_time.as_secs_f64() + decode_total;

        let n = self.decode_times.len();
        if n == 0 {
            return format!(
                "prefill {:.2}s ({} tokens) | 0 decode tokens | total {:.2}s",
                self.prefill_time.as_secs_f64(),
                self.prompt_tokens,
                total,
            );
        }

        let tok_s: Vec<f64> = self
            .decode_times
            .iter()
            .map(|d| 1.0 / d.as_secs_f64())
            .collect();
        let avg_tok_s = tok_s.iter().sum::<f64>() / n as f64;
        let min_tok_s = tok_s.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_tok_s = tok_s.iter().cloned().fold(0.0f64, f64::max);
        let avg_ms = decode_total * 1000.0 / n as f64;

        let seed_str = match self.seed {
            Some(s) => format!(" | seed {s}"),
            None => String::new(),
        };

        format!(
            "prefill {:.2}s ({} tok) | \
             decode {} tok in {:.2}s ({:.1} avg, {:.1} min, {:.1} max tok/s | {:.0}ms/tok) | \
             total {:.2}s{seed_str}",
            self.prefill_time.as_secs_f64(),
            self.prompt_tokens,
            n,
            decode_total,
            avg_tok_s,
            min_tok_s,
            max_tok_s,
            avg_ms,
            total,
        )
    }
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

    #[test]
    fn rng_is_deterministic() {
        let mut rng1 = Rng::new(42);
        let mut rng2 = Rng::new(42);
        let vals1: Vec<f32> = (0..10).map(|_| rng1.next_f32()).collect();
        let vals2: Vec<f32> = (0..10).map(|_| rng2.next_f32()).collect();
        assert_eq!(vals1, vals2);
    }

    #[test]
    fn sample_token_greedy() {
        let logits = vec![0.0, 1.0, 0.5, 0.2];
        let config = SamplingConfig {
            temperature: 0.0,
            ..Default::default()
        };
        let mut rng = Rng::new(1);
        assert_eq!(sample_token(&logits, &config, &[], &mut rng), 1);
    }

    #[test]
    fn think_token_styling_hides_tags() {
        // Token IDs: 100 = <think>, 101 = </think>, others are content.
        let think = Some((100u32, 101u32));
        let mut in_think = false;
        let mut skip_ws = false;

        print_token_styled(100, "<think>", think, true, &mut in_think, &mut skip_ws);
        assert!(in_think);

        print_token_styled(42, "reasoning", think, true, &mut in_think, &mut skip_ws);
        assert!(in_think);

        print_token_styled(101, "</think>", think, true, &mut in_think, &mut skip_ws);
        assert!(!in_think);

        print_token_styled(43, "answer", think, true, &mut in_think, &mut skip_ws);
        assert!(!in_think);
    }

    #[test]
    fn think_token_styling_disabled() {
        let mut in_think = false;
        let mut skip_ws = false;
        print_token_styled(100, "<think>", None, false, &mut in_think, &mut skip_ws);
        assert!(!in_think);
    }

    #[test]
    fn think_token_hide_without_style() {
        let think = Some((100u32, 101u32));
        let mut in_think = false;
        let mut skip_ws = false;

        print_token_styled(100, "<think>", think, false, &mut in_think, &mut skip_ws);
        assert!(in_think);
        print_token_styled(42, "hidden", think, false, &mut in_think, &mut skip_ws);
        assert!(in_think);
        print_token_styled(101, "</think>", think, false, &mut in_think, &mut skip_ws);
        assert!(!in_think);
        assert!(skip_ws); // should skip trailing whitespace
        // Whitespace after close is swallowed.
        print_token_styled(44, "\n\n", think, false, &mut in_think, &mut skip_ws);
        assert!(skip_ws); // still skipping (was whitespace-only)
        // Real content stops the skip.
        print_token_styled(43, "visible", think, false, &mut in_think, &mut skip_ws);
        assert!(!skip_ws);
        assert!(!in_think);
    }

    #[test]
    fn min_p_filters_low_prob() {
        // With min_p=0.5, only tokens with prob >= 0.5 * max_prob survive.
        let logits = vec![10.0, 0.0, 0.0, 0.0]; // token 0 dominates
        let config = SamplingConfig {
            temperature: 1.0,
            min_p: 0.5,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            ..Default::default()
        };
        let mut rng = Rng::new(42);
        // With such skewed logits + min_p=0.5, should always pick token 0.
        for _ in 0..10 {
            assert_eq!(sample_token(&logits, &config, &[], &mut rng), 0);
        }
    }
}
