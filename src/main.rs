use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::time::Instant;

use clap::{Args, Parser, Subcommand};
use homura::generate::Generator;
use homura::kv_generate::{UnifiedKvGenerator, has_unified_model};
use homura::log;
use homura::onnx::parser;
use homura::{Buffer, DType, Model};

// Ops that homura currently supports.
const SUPPORTED_OPS: &[&str] = &[
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Neg",
    "Relu",
    "Exp",
    "Tanh",
    "Softmax",
    "MatMul",
    "Gemm",
    "Conv",
    "MaxPool",
    "Reshape",
    "Flatten",
    "Clip",
    "BatchNormalization",
    "GlobalAveragePool",
];

#[derive(Parser)]
#[command(name = "homura", about = "Rust ML inference engine")]
struct Cli {
    /// Show compilation progress (-v) or full memref dumps (-vv)
    #[arg(long, short, global = true, action = clap::ArgAction::Count)]
    verbose: u8,

    #[command(subcommand)]
    command: Commands,
}

/// Shared sampling parameters for text generation.
#[derive(Args, Clone, Debug)]
struct SamplingArgs {
    /// Sampling temperature (0 = greedy, default: 0.7)
    #[arg(long, default_value = "0.7")]
    temperature: f32,
    /// Top-p nucleus sampling threshold (default: 0.9)
    #[arg(long, default_value = "0.9")]
    top_p: f32,
    /// Repetition penalty (1.0 = off, default: 1.1)
    #[arg(long, default_value = "1.1")]
    repetition_penalty: f32,
    /// Top-k filtering (0 = disabled, default: 50)
    #[arg(long, default_value = "50")]
    top_k: usize,
    /// min_p sampling: filter tokens with prob < min_p * max_prob (default: 0, disabled)
    #[arg(long, default_value = "0.0")]
    min_p: f32,
    /// Frequency penalty: subtract freq * count(token) from logits (default: 0, off)
    #[arg(long, default_value = "0.0")]
    frequency_penalty: f32,
    /// Presence penalty: subtract penalty for any previously seen token (default: 0, off)
    #[arg(long, default_value = "0.0")]
    presence_penalty: f32,
    /// RNG seed for reproducible sampling
    #[arg(long)]
    seed: Option<u64>,
    /// Stop sequences (comma-separated, e.g. "\n\n,Posted by")
    #[arg(long)]
    stop: Option<String>,
}

impl SamplingArgs {
    fn to_config(&self) -> homura::generate::SamplingConfig {
        let stop_sequences = self
            .stop
            .as_ref()
            .map(|s| s.split(',').map(|s| s.to_string()).collect())
            .unwrap_or_default();
        homura::generate::SamplingConfig {
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            min_p: self.min_p,
            repetition_penalty: self.repetition_penalty,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            stop_sequences,
            seed: self.seed,
        }
    }
}

#[derive(Subcommand)]
enum Commands {
    /// Print model graph info (inputs, outputs, ops) without compiling
    Info {
        /// Path to the ONNX model file
        model: PathBuf,
    },
    /// Clear the compilation cache (~/.cache/homura/ or HOMURA_CACHE_DIR)
    CleanCache,
    /// Run inference or text generation
    Run {
        /// Path to the ONNX model file or directory (for text generation)
        model: PathBuf,
        /// Raw binary f32 input file, or - for stdin. Uses all-zeros if omitted.
        #[arg(long)]
        input: Option<String>,
        /// Input shape as comma-separated dims, e.g. 1,1,28,28. Required with --input.
        #[arg(long)]
        shape: Option<String>,
        /// Write raw f32 output to this file. Prints as text if omitted.
        #[arg(long)]
        output: Option<PathBuf>,
        /// Text prompt for generation (enables text generation mode)
        #[arg(long)]
        prompt: Option<String>,
        /// Maximum number of tokens to generate (default: 100)
        #[arg(long, default_value = "100")]
        max_tokens: usize,
        /// Maximum context length for KV cache (default: 2048, capped by model limit)
        #[arg(long = "ctx", default_value = "2048")]
        context_len: usize,
        /// Weight dtype: auto (detect CPU), f32, bf16
        #[arg(long, default_value = "auto")]
        dtype: String,
        #[command(flatten)]
        sampling: SamplingArgs,
    },
    /// Interactive multi-turn chat with a HuggingFace model
    Chat {
        /// HuggingFace model (e.g. Qwen/Qwen2.5-0.5B) or local directory
        model: PathBuf,
        /// System prompt
        #[arg(long, default_value = "You are a helpful assistant.")]
        system: String,
        /// Maximum tokens to generate per turn (default: 512)
        #[arg(long, default_value = "512")]
        max_tokens: usize,
        /// Maximum context length for KV cache (default: 2048, capped by model limit)
        #[arg(long = "ctx", default_value = "2048")]
        context_len: usize,
        /// Enable thinking/reasoning output (for models like Qwen3 that support it)
        #[arg(long)]
        think: bool,
        /// Weight dtype: auto (detect CPU), f32, bf16
        #[arg(long, default_value = "auto")]
        dtype: String,
        #[command(flatten)]
        sampling: SamplingArgs,
    },
}

fn main() {
    // Register an atexit handler that calls _exit() to skip C++ global
    // destructors. LLVM has a static initialization order fiasco
    // (llvm/llvm-project#154528) that causes double-free/segfault during
    // __cxa_finalize when MLIR dialects have been loaded. This catches
    // all exit paths including clap's --help/--version which call
    // std::process::exit() before we reach our own _exit().
    extern "C" fn force_exit() {
        let _ = std::io::Write::flush(&mut std::io::stdout());
        let _ = std::io::Write::flush(&mut std::io::stderr());
        unsafe { libc::_exit(0) }
    }
    unsafe { libc::atexit(force_exit) };

    let cli = Cli::parse();
    match cli.verbose {
        1 => log::set_level(log::Level::Debug),
        2.. => log::set_level(log::Level::Trace),
        _ => {}
    }
    homura::cpu_affinity::pin_to_single_ccd();
    let code = match run(cli) {
        Ok(()) => 0,
        Err(e) => {
            log::error!("{e}");
            1
        }
    };
    let _ = io::stdout().flush();
    let _ = io::stderr().flush();
    unsafe { libc::_exit(code) }
}

fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    match cli.command {
        Commands::Info { model } => cmd_info(&model),
        Commands::CleanCache => cmd_clean_cache(),
        Commands::Run {
            model,
            input,
            shape,
            output,
            prompt,
            max_tokens,
            context_len,
            dtype,
            sampling,
        } => {
            if let Some(prompt_text) = prompt {
                let sampling = sampling.to_config();
                let weight_dtype = resolve_weight_dtype(&dtype);
                // Resolve model path: HF repo ID, local directory, or file.
                let model_dir = resolve_model_path(&model)?;
                // Check for GGUF file first
                if let Some(gguf_path) = find_gguf_file(&model_dir) {
                    cmd_generate_gguf(
                        &gguf_path,
                        &model_dir,
                        &prompt_text,
                        max_tokens,
                        context_len,
                        &sampling,
                    )
                } else {
                    let has_config = model_dir.join("config.json").exists();
                    let has_weights = model_dir.join("model.safetensors").exists()
                        || model_dir.join("model.safetensors.index.json").exists();
                    if has_config && has_weights {
                        cmd_generate_hf(
                            &model_dir,
                            &prompt_text,
                            max_tokens,
                            context_len,
                            weight_dtype,
                            &sampling,
                        )
                    } else {
                        cmd_generate(&model_dir, &prompt_text, max_tokens, &sampling)
                    }
                }
            } else {
                cmd_run(
                    &model,
                    input.as_deref(),
                    shape.as_deref(),
                    output.as_deref(),
                )
            }
        }
        Commands::Chat {
            model,
            system,
            max_tokens,
            context_len,
            think,
            dtype,
            sampling,
        } => {
            let model_dir = resolve_model_path(&model)?;
            let model_name = model.to_string_lossy();
            if let Some(gguf_path) = find_gguf_file(&model_dir) {
                return cmd_chat_gguf(
                    &gguf_path,
                    &model_dir,
                    &model_name,
                    &system,
                    max_tokens,
                    context_len,
                    think,
                    &sampling.to_config(),
                );
            }
            let weight_dtype = resolve_weight_dtype(&dtype);
            cmd_chat(
                &model_dir,
                &model_name,
                &system,
                max_tokens,
                context_len,
                think,
                weight_dtype,
                &sampling,
            )
        }
    }
}

// ── dtype resolution ─────────────────────────────────────────────────────────

fn resolve_weight_dtype(s: &str) -> DType {
    match s {
        "auto" => {
            let caps = homura::cpu_caps::CpuCaps::get();
            if caps.supports_bf16_compute() {
                log::info!("auto-detected bf16 support ({})", caps);
                DType::BF16
            } else {
                log::info!("no bf16 hardware support, using f32 ({})", caps);
                DType::F32
            }
        }
        "bf16" => {
            let caps = homura::cpu_caps::CpuCaps::get();
            if !caps.supports_bf16_compute() {
                log::warn!(
                    "bf16 requested but CPU lacks AVX-512 BF16 ({}); performance may be poor",
                    caps
                );
            }
            DType::BF16
        }
        "f32" => DType::F32,
        other => {
            log::error!("unsupported --dtype: {other}");
            std::process::exit(1);
        }
    }
}

// ── GGUF detection ──────────────────────────────────────────────────────────

/// Find a GGUF file in a directory. If the path itself is a .gguf file, returns it.
/// If it's a directory, looks for any .gguf file inside.
fn find_gguf_file(path: &std::path::Path) -> Option<PathBuf> {
    if path.is_file() && path.extension().is_some_and(|e| e == "gguf") {
        return Some(path.to_path_buf());
    }
    if path.is_dir() {
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.extension().is_some_and(|e| e == "gguf") {
                    return Some(p);
                }
            }
        }
    }
    None
}

/// Find tokenizer.json for a GGUF model. Checks:
/// 1. Same directory as the GGUF file
/// 2. model_dir (if different from GGUF file's parent)
fn find_tokenizer_for_gguf(
    gguf_path: &std::path::Path,
    model_dir: &std::path::Path,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    // Check same directory as GGUF file
    if let Some(parent) = gguf_path.parent() {
        let tok = parent.join("tokenizer.json");
        if tok.exists() {
            return Ok(tok);
        }
    }
    // Check model_dir
    let tok = model_dir.join("tokenizer.json");
    if tok.exists() {
        return Ok(tok);
    }
    Err(format!(
        "tokenizer.json not found near {} -- place it alongside the GGUF file",
        gguf_path.display()
    )
    .into())
}

// ── model resolution ─────────────────────────────────────────────────────────

/// Resolve a model path: HF repo ID (e.g. "Qwen/Qwen2.5-0.5B"), local directory, or file.
///
/// For HF repo IDs, downloads config.json, model.safetensors, and tokenizer.json
/// to the HF cache and returns the local snapshot directory.
fn resolve_model_path(model: &std::path::Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    // If it's an existing local path, use it directly.
    if model.exists() {
        return if model.is_dir() {
            Ok(model.to_path_buf())
        } else {
            Ok(model.parent().unwrap_or(model).to_path_buf())
        };
    }

    // Treat as HF repo ID if it looks like "org/model" or "org/model/revision".
    let model_str = model.to_str().unwrap_or("");
    let parts: Vec<&str> = model_str.split('/').collect();
    if parts.len() < 2 {
        return Err(format!("model path does not exist: {}", model.display()).into());
    }

    let repo_id = if parts.len() >= 3 {
        format!("{}/{}", parts[0], parts[1])
    } else {
        model_str.to_string()
    };
    let revision = if parts.len() >= 3 { parts[2] } else { "main" };

    log::info!(
        "fetching {} (rev: {}) from HuggingFace Hub",
        repo_id,
        revision
    );

    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_progress(true)
        .build()?;
    let repo = api.repo(hf_hub::Repo::with_revision(
        repo_id,
        hf_hub::RepoType::Model,
        revision.to_string(),
    ));

    // Download essential files. hf-hub caches them automatically.
    let mut snapshot_dir = None;
    let mut track = |path: &std::path::Path| {
        if snapshot_dir.is_none() {
            snapshot_dir = path.parent().map(|p| p.to_path_buf());
        }
    };

    // Required: config.json, tokenizer.json
    for file in &["config.json", "tokenizer.json"] {
        let path = repo
            .get(file)
            .map_err(|e| format!("failed to fetch {file}: {e}"))?;
        track(&path);
    }

    // Optional metadata
    for file in &["tokenizer_config.json", "generation_config.json"] {
        if let Ok(path) = repo.get(file) {
            track(&path);
        }
    }

    // Weights: try single model.safetensors first, fall back to sharded index.
    if let Ok(path) = repo.get("model.safetensors") {
        track(&path);
    } else {
        // Sharded: fetch the index, then each shard file.
        let index_path = repo
            .get("model.safetensors.index.json")
            .map_err(|e| format!("no model.safetensors or index: {e}"))?;
        track(&index_path);

        let index_text = std::fs::read_to_string(&index_path)?;
        let index: serde_json::Value = serde_json::from_str(&index_text)?;
        if let Some(map) = index.get("weight_map").and_then(|m| m.as_object()) {
            // Collect unique shard filenames.
            let mut shards: Vec<&str> = map.values().filter_map(|v| v.as_str()).collect();
            shards.sort();
            shards.dedup();
            for shard in &shards {
                log::info!("fetching shard {shard}");
                let path = repo
                    .get(shard)
                    .map_err(|e| format!("failed to fetch shard {shard}: {e}"))?;
                track(&path);
            }
        }
    }

    snapshot_dir.ok_or_else(|| "failed to resolve HF model directory".into())
}

// ── clean-cache ──────────────────────────────────────────────────────────────

fn cmd_clean_cache() -> Result<(), Box<dyn std::error::Error>> {
    let cache = homura::cache::CompilationCache::new();
    let dir = cache.cache_dir();
    if !dir.exists() {
        println!("Cache directory does not exist: {}", dir.display());
        return Ok(());
    }
    let mut count = 0u64;
    let mut bytes = 0u64;
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let meta = entry.metadata()?;
        if meta.is_file() {
            bytes += meta.len();
            std::fs::remove_file(entry.path())?;
            count += 1;
        }
    }
    println!(
        "Removed {count} cached files ({:.1} MB)",
        bytes as f64 / 1_048_576.0
    );
    Ok(())
}

// ── info ─────────────────────────────────────────────────────────────────────

fn cmd_info(path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let onnx = parser::parse_model(path)?;

    // Dynamic inputs
    println!("Inputs ({}):", onnx.dynamic_inputs.len());
    for inp in &onnx.dynamic_inputs {
        println!("  {} : {:?} {:?}", inp.name, inp.dtype, inp.dims);
    }

    // Outputs
    println!("Outputs ({}):", onnx.outputs.len());
    for name in &onnx.outputs {
        println!("  {name}");
    }

    // Initializers / weights
    let total_bytes: usize = onnx
        .initializers
        .iter()
        .map(|(_, buf)| buf.shape().num_elements() as usize * buf.dtype().size_bytes())
        .sum();
    println!(
        "Initializers: {} ({} bytes)",
        onnx.initializers.len(),
        total_bytes
    );

    // Ops
    println!("Nodes ({}):", onnx.nodes.len());
    let mut all_supported = true;
    for node in &onnx.nodes {
        let supported = SUPPORTED_OPS.contains(&node.op_type.as_str());
        if !supported {
            all_supported = false;
        }
        let flag = if supported { "" } else { "  [UNSUPPORTED]" };
        println!(
            "  {} ({} -> {}){flag}",
            node.op_type,
            node.inputs.join(", "),
            node.outputs.join(", ")
        );
    }

    if all_supported {
        println!("All ops supported.");
    } else {
        println!("Warning: model contains unsupported ops (marked above).");
    }

    Ok(())
}

// ── generate ─────────────────────────────────────────────────────────────────

fn cmd_generate(
    model_path: &std::path::Path,
    prompt: &str,
    max_tokens: usize,
    sampling: &homura::generate::SamplingConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    // Resolve model directory: if given a file, use its parent.
    let model_dir = if model_path.is_dir() {
        model_path.to_path_buf()
    } else {
        model_path
            .parent()
            .ok_or("cannot determine model directory from path")?
            .to_path_buf()
    };

    let model_dir_str = model_dir.to_str().ok_or("model path is not valid UTF-8")?;

    log::info!("loading generator from {}", model_dir.display());
    let t_load = Instant::now();

    // Prefer unified single-model KV cache, fall back to full-recompute.
    let generated = if has_unified_model(&model_dir) {
        log::info!("using unified KV cache generator");
        let mut generator = UnifiedKvGenerator::load(model_dir_str, 1024, 50256)?;
        log::info!("loaded in {:.2}s", t_load.elapsed().as_secs_f64());

        log::info!("generating (max_tokens={max_tokens})");
        generator.generate_with_sampling(prompt, max_tokens, sampling)
    } else {
        log::info!("using full-recompute generator — no unified model found");
        let generator = Generator::load(model_dir_str)?;
        log::info!("loaded in {:.2}s", t_load.elapsed().as_secs_f64());

        log::info!("generating (max_tokens={max_tokens})");
        generator.generate(prompt, max_tokens)
    };

    // If stdout is piped, print the full text at the end.
    if !atty::is(atty::Stream::Stdout) {
        println!("{}{}", prompt, generated);
    }

    Ok(())
}

// ── generate-hf ─────────────────────────────────────────────────────────────

fn cmd_generate_hf(
    model_dir: &std::path::Path,
    prompt: &str,
    max_tokens: usize,
    context_len: usize,
    weight_dtype: DType,
    sampling: &homura::generate::SamplingConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    use homura::hf::chat::{ChatMessage, ChatTemplate};

    log::info!("loading HF model from {}", model_dir.display());
    let t_load = Instant::now();
    let model = homura::hf::model::HfModel::load_with_dtype(model_dir, weight_dtype)?;
    let tokenizer =
        homura::hf::tokenizer::HfTokenizer::from_file(&model_dir.join("tokenizer.json"))?;
    log::info!("loaded in {:.2}s", t_load.elapsed().as_secs_f64());

    // For instruct models, wrap the prompt in the chat template so the model
    // sees properly formatted input instead of raw text.
    let formatted_prompt = if let Ok(template) = ChatTemplate::load(model_dir) {
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: prompt.to_string(),
        }];
        match template.render(&messages, true, None) {
            Ok(text) => {
                log::info!("applied chat template (instruct model)");
                text
            }
            Err(_) => prompt.to_string(),
        }
    } else {
        prompt.to_string()
    };

    let max_seq_len = std::cmp::min(model.config().max_position_embeddings, context_len);
    log::info!("generating (max_tokens={max_tokens}, max_seq_len={max_seq_len})");

    let generated = model.generate(
        &tokenizer,
        &formatted_prompt,
        max_tokens,
        max_seq_len,
        sampling,
    )?;

    // If stdout is piped, print the full text at the end.
    if !atty::is(atty::Stream::Stdout) {
        println!("{}{}", prompt, generated);
    }
    Ok(())
}

fn cmd_generate_gguf(
    gguf_path: &std::path::Path,
    model_dir: &std::path::Path,
    prompt: &str,
    max_tokens: usize,
    context_len: usize,
    sampling: &homura::generate::SamplingConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    use homura::hf::chat::{ChatMessage, ChatTemplate};

    log::info!("loading GGUF model from {}", gguf_path.display());
    let t_load = Instant::now();
    let model = homura::hf::model::HfModel::load_gguf(gguf_path)?;
    let tok_path = find_tokenizer_for_gguf(gguf_path, model_dir)?;
    let tokenizer = homura::hf::tokenizer::HfTokenizer::from_file(&tok_path)?;
    log::info!("loaded in {:.2}s", t_load.elapsed().as_secs_f64());

    let formatted_prompt = if let Ok(template) = ChatTemplate::load(model_dir) {
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: prompt.to_string(),
        }];
        match template.render(&messages, true, None) {
            Ok(text) => {
                log::info!("applied chat template (instruct model)");
                text
            }
            Err(_) => prompt.to_string(),
        }
    } else {
        prompt.to_string()
    };

    let max_seq_len = std::cmp::min(model.config().max_position_embeddings, context_len);
    log::info!("generating (max_tokens={max_tokens}, max_seq_len={max_seq_len})");

    let generated = model.generate(
        &tokenizer,
        &formatted_prompt,
        max_tokens,
        max_seq_len,
        sampling,
    )?;

    if !atty::is(atty::Stream::Stdout) {
        println!("{}{}", prompt, generated);
    }
    Ok(())
}

fn cmd_chat_gguf(
    gguf_path: &std::path::Path,
    model_dir: &std::path::Path,
    model_name: &str,
    system_prompt: &str,
    max_tokens_per_turn: usize,
    context_len: usize,
    enable_thinking: bool,
    sampling: &homura::generate::SamplingConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    use homura::generate::{GenerativeModel, generate_streaming_from_ids};
    use homura::hf::chat::{ChatMessage, ChatTemplate, find_chat_stop_token, find_think_tokens};
    use homura::hf::model::HfGenerationContext;
    use std::io::BufRead;

    log::info!("loading GGUF model from {}", gguf_path.display());
    let t_load = Instant::now();
    let model = homura::hf::model::HfModel::load_gguf(gguf_path)?;
    let tok_path = find_tokenizer_for_gguf(gguf_path, model_dir)?;
    let tokenizer = homura::hf::tokenizer::HfTokenizer::from_file(&tok_path)?;
    // Try loading chat template from model_dir (tokenizer_config.json)
    let chat_template = ChatTemplate::load(model_dir)?;
    log::info!("loaded in {:.2}s", t_load.elapsed().as_secs_f64());

    let max_seq_len = std::cmp::min(model.config().max_position_embeddings, context_len);
    let stop_token = find_chat_stop_token(model_dir, &tokenizer);
    let think_tokens = find_think_tokens(&tokenizer);
    log::info!(
        "max_seq_len={max_seq_len}, stop_token={:?}, think_tokens={:?}",
        stop_token,
        think_tokens
    );

    let mut ctx = HfGenerationContext::new_chat(&model, &tokenizer, max_seq_len, stop_token);

    let system_content = if !enable_thinking && think_tokens.is_some() {
        format!("{system_prompt} /no_think")
    } else {
        system_prompt.to_string()
    };
    let mut messages: Vec<ChatMessage> = vec![ChatMessage {
        role: "system".into(),
        content: system_content,
    }];

    // Prefill system prompt
    let system_text = chat_template.render(
        &messages,
        false,
        if enable_thinking { Some(true) } else { None },
    )?;
    let system_tokens = tokenizer.encode_with_special(&system_text);
    let system_token_ids: Vec<i64> = system_tokens.iter().map(|&id| id as i64).collect();

    log::info!(
        "prefilling system prompt ({} tokens)",
        system_token_ids.len()
    );
    ctx.prefill(&system_token_ids)?;

    let sampling_config = sampling.clone();

    // Interactive chat loop
    println!("Chat with {model_name} (type 'quit' to exit)");
    println!();

    let stdin = io::stdin();
    loop {
        print!("> ");
        io::stdout().flush()?;

        let mut user_input = String::new();
        stdin.lock().read_line(&mut user_input)?;
        let user_input = user_input.trim();

        if user_input.is_empty() {
            continue;
        }
        if user_input == "quit" || user_input == "exit" {
            break;
        }

        messages.push(ChatMessage {
            role: "user".into(),
            content: user_input.to_string(),
        });

        // Render the new turn
        let full_text = chat_template.render(
            &messages,
            true,
            if enable_thinking { Some(true) } else { None },
        )?;
        let full_tokens = tokenizer.encode_with_special(&full_text);
        let prev_len = model.kv_cache_len();
        let new_tokens: Vec<i64> = full_tokens[prev_len..]
            .iter()
            .map(|&id| id as i64)
            .collect();

        if new_tokens.is_empty() {
            continue;
        }

        let think_cfg = homura::generate::ThinkConfig {
            token_ids: think_tokens,
            style_content: enable_thinking,
        };
        let (clean_text, _gen_ids) = generate_streaming_from_ids(
            &mut ctx,
            &new_tokens,
            max_tokens_per_turn,
            &sampling_config,
            false,
            think_cfg,
        )?;

        let content = strip_think_block(&clean_text);
        if !content.ends_with('\n') {
            println!();
        }
        println!();

        messages.push(ChatMessage {
            role: "assistant".into(),
            content,
        });
    }

    Ok(())
}

/// Strip `<think>...</think>` blocks (and surrounding whitespace) from text.
///
/// If the content outside the think block is empty (model put its entire response
/// inside the think block), falls back to the content inside it.
/// Also handles unclosed `<think>` tags (strips the tag, keeps the rest).
fn strip_think_block(text: &str) -> String {
    if let Some(start) = text.find("<think>") {
        if let Some(end) = text.find("</think>") {
            let before = &text[..start];
            let after = &text[end + "</think>".len()..];
            let outer = format!("{}{}", before, after.trim_start_matches('\n'));
            if !outer.trim().is_empty() {
                return outer;
            }
            // No content outside think block — use content inside as fallback.
            let inside = text[start + "<think>".len()..end].trim();
            if !inside.is_empty() {
                return inside.to_string();
            }
        } else {
            // Unclosed <think> — strip the tag and return the rest.
            let rest = text[start + "<think>".len()..].trim_start();
            if !rest.is_empty() {
                return rest.to_string();
            }
        }
    }
    text.to_string()
}

// ── chat ─────────────────────────────────────────────────────────────────────

fn cmd_chat(
    model_dir: &std::path::Path,
    model_name: &str,
    system_prompt: &str,
    max_tokens_per_turn: usize,
    context_len: usize,
    enable_thinking: bool,
    weight_dtype: DType,
    sampling_args: &SamplingArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    use homura::generate::generate_streaming_from_ids;
    use homura::hf::chat::{ChatMessage, ChatTemplate, find_chat_stop_token, find_think_tokens};
    use homura::hf::model::HfGenerationContext;
    use std::io::BufRead;

    log::info!("loading HF model from {}", model_dir.display());
    let t_load = Instant::now();

    let model = homura::hf::model::HfModel::load_with_dtype(model_dir, weight_dtype)?;
    let tokenizer =
        homura::hf::tokenizer::HfTokenizer::from_file(&model_dir.join("tokenizer.json"))?;
    let chat_template = ChatTemplate::load(model_dir)?;
    log::info!("loaded in {:.2}s", t_load.elapsed().as_secs_f64());

    let max_seq_len = std::cmp::min(model.config().max_position_embeddings, context_len);
    let stop_token = find_chat_stop_token(model_dir, &tokenizer);
    // Always look up think tokens so we can hide them from output.
    // The --think flag controls whether content is styled, not whether tags are hidden.
    let think_tokens = find_think_tokens(&tokenizer);
    log::info!(
        "max_seq_len={max_seq_len}, stop_token={:?}, think_tokens={:?}",
        stop_token,
        think_tokens
    );

    let mut ctx = HfGenerationContext::new_chat(&model, &tokenizer, max_seq_len, stop_token);

    // Build conversation history starting with the system prompt.
    // Append /no_think for Qwen3-style models when thinking is disabled.
    let system_content = if !enable_thinking && think_tokens.is_some() {
        format!("{system_prompt} /no_think")
    } else {
        system_prompt.to_string()
    };
    let mut messages: Vec<ChatMessage> = vec![ChatMessage {
        role: "system".into(),
        content: system_content,
    }];

    // Prefill system prompt on startup so the KV cache is warm.
    let system_text = chat_template.render(
        &messages,
        false,
        if enable_thinking { Some(true) } else { None },
    )?;
    let system_tokens = tokenizer.encode_with_special(&system_text);
    let system_token_ids: Vec<i64> = system_tokens.iter().map(|&id| id as i64).collect();

    log::info!(
        "prefilling system prompt ({} tokens)",
        system_token_ids.len()
    );
    let prefill_start = Instant::now();
    // Use the model directly for the system-only prefill (no generation).
    let input_buf = homura::Buffer::from_slice::<i64>(
        &system_token_ids,
        &[1, system_token_ids.len() as u64],
        homura::DType::I64,
    );
    let outputs = model.run(&input_buf)?;
    let num_kv = model.config().num_hidden_layers * 2;
    let kv_cache: Vec<homura::Buffer> = outputs[1..1 + num_kv].to_vec();
    model.init_kv_cache(&kv_cache, max_seq_len)?;
    log::info!(
        "system prefill done in {:.2}s",
        prefill_start.elapsed().as_secs_f64()
    );

    // Number of tokens in the KV cache. Used to compute delta for each turn.
    let mut tokens_in_cache = system_token_ids.len();

    eprintln!("\x1b[1m{model_name}\x1b[0m -- type /help for commands.\n");

    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let sampling = sampling_args.to_config();

    loop {
        // Print prompt.
        eprint!("> ");
        let _ = io::stderr().flush();

        let mut line = String::new();
        let n = reader.read_line(&mut line)?;
        if n == 0 {
            // EOF
            break;
        }
        let input = line.trim();
        if input.is_empty() {
            continue;
        }

        match input {
            "/quit" | "/exit" => break,
            "/clear" => {
                model.reset_kv_cache();
                messages.truncate(1); // keep system prompt
                // Re-prefill system prompt.
                let sys_text = chat_template.render(
                    &messages,
                    false,
                    if enable_thinking { Some(true) } else { None },
                )?;
                let sys_toks = tokenizer.encode_with_special(&sys_text);
                let sys_ids: Vec<i64> = sys_toks.iter().map(|&id| id as i64).collect();
                let input_buf = homura::Buffer::from_slice::<i64>(
                    &sys_ids,
                    &[1, sys_ids.len() as u64],
                    homura::DType::I64,
                );
                let outputs = model.run(&input_buf)?;
                let kv_cache: Vec<homura::Buffer> = outputs[1..1 + num_kv].to_vec();
                model.init_kv_cache(&kv_cache, max_seq_len)?;
                tokens_in_cache = sys_ids.len();
                eprintln!("Conversation cleared.\n");
                continue;
            }
            "/help" => {
                eprintln!("  /clear  - Reset conversation");
                eprintln!("  /quit   - Exit chat");
                eprintln!("  /help   - Show this help\n");
                continue;
            }
            _ => {}
        }

        // Build the delta token sequence directly instead of re-rendering the
        // full conversation. The Qwen3 template normalizes think blocks differently
        // depending on context (last vs non-last assistant message), making
        // prefix-based or re-encode-based delta computation unreliable.
        //
        // The delta for a new user turn is always:
        //   <|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n
        //
        // On the first turn (after system prefill), we render the full template
        // to get the correct prefix including system prompt formatting.
        messages.push(ChatMessage {
            role: "user".into(),
            content: input.to_string(),
        });

        // Construct delta tokens directly instead of re-rendering the full
        // conversation. The Qwen3 template normalizes think blocks differently
        // depending on context, making re-encode-based deltas unreliable.
        //
        // After system prefill, the first turn delta is the user wrapper:
        //   <|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n
        //
        // After generation (which stops at EOS but doesn't include it in gen_ids),
        // subsequent turn deltas include the closing <|im_end|>:
        //   <|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n
        let is_first_turn = messages.len() == 2; // system + this user msg
        let delta_text = if is_first_turn {
            format!("<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n")
        } else {
            format!("<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n")
        };

        let delta_tokens = tokenizer.encode_with_special(&delta_text);

        // Context overflow check for subsequent turns.
        if tokens_in_cache + delta_tokens.len() + max_tokens_per_turn > max_seq_len {
            eprintln!(
                "Warning: context nearly full ({}/{} tokens). Clearing conversation.\n",
                tokens_in_cache + delta_tokens.len(),
                max_seq_len
            );
            model.reset_kv_cache();
            messages.truncate(1);
            messages.push(ChatMessage {
                role: "user".into(),
                content: input.to_string(),
            });
            tokens_in_cache = 0;
            // Re-render from scratch for this single turn.
            let fresh_text = chat_template.render(
                &messages,
                true,
                if enable_thinking { Some(true) } else { None },
            )?;
            let fresh_tokens = tokenizer.encode_with_special(&fresh_text);
            if fresh_tokens.len() + max_tokens_per_turn > max_seq_len {
                eprintln!(
                    "Error: message too long ({} tokens, max context {}).\n",
                    fresh_tokens.len(),
                    max_seq_len
                );
                messages.pop();
                continue;
            }
            // Use fresh render as delta (full re-prefill).
            let delta_ids: Vec<i64> = fresh_tokens.iter().map(|&id| id as i64).collect();
            let think_cfg = homura::generate::ThinkConfig {
                token_ids: think_tokens,
                style_content: enable_thinking,
            };
            let (_text, _ids) = generate_streaming_from_ids(
                &mut ctx,
                &delta_ids,
                max_tokens_per_turn,
                &sampling,
                false,
                think_cfg,
            )?;
            let content = strip_think_block(&_text);
            messages.push(ChatMessage {
                role: "assistant".into(),
                content,
            });
            tokens_in_cache = fresh_tokens.len() + _ids.len();
            eprintln!();
            continue;
        }

        let delta_token_ids: Vec<i64> = delta_tokens.iter().map(|&id| id as i64).collect();
        let think_cfg = homura::generate::ThinkConfig {
            token_ids: think_tokens,
            style_content: enable_thinking,
        };
        let (clean_text, gen_ids) = generate_streaming_from_ids(
            &mut ctx,
            &delta_token_ids,
            max_tokens_per_turn,
            &sampling,
            false,
            think_cfg,
        )?;

        // Store clean text (think blocks stripped) in message history.
        let content = strip_think_block(&clean_text);
        messages.push(ChatMessage {
            role: "assistant".into(),
            content,
        });

        // tokens_in_cache = what was there + delta we just prefilled + generated tokens.
        tokens_in_cache += delta_tokens.len() + gen_ids.len();

        eprintln!();
    }

    Ok(())
}

// ── run ──────────────────────────────────────────────────────────────────────

fn cmd_run(
    model_path: &std::path::Path,
    input_arg: Option<&str>,
    shape_arg: Option<&str>,
    output_path: Option<&std::path::Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Parse shape if given
    let shape: Option<Vec<u64>> = shape_arg
        .map(|s| {
            s.split(',')
                .map(|d| {
                    d.trim()
                        .parse::<u64>()
                        .map_err(|e| format!("invalid shape dim '{d}': {e}"))
                })
                .collect::<Result<Vec<u64>, String>>()
        })
        .transpose()?;

    // Validate: --input requires --shape
    if input_arg.is_some() && shape.is_none() {
        return Err("--shape is required when --input is provided".into());
    }

    // Load and compile model
    let model = Model::load(model_path)?;

    // Build the input buffer
    let input_buf = if let Some(src) = input_arg {
        let shape = shape.as_deref().unwrap(); // validated above
        let bytes = read_input_bytes(src)?;
        if bytes.len() % 4 != 0 {
            return Err(format!(
                "input byte count {} is not a multiple of 4 (f32)",
                bytes.len()
            )
            .into());
        }
        let expected_elems: u64 = shape.iter().product();
        let got_elems = bytes.len() as u64 / 4;
        if got_elems != expected_elems {
            return Err(format!(
                "input has {got_elems} f32 elements but shape {:?} requires {expected_elems}",
                shape
            )
            .into());
        }
        // Reinterpret bytes as f32 slice
        let floats: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Buffer::from_slice::<f32>(&floats, shape, DType::F32)
    } else {
        // Use the first dynamic input's shape from the model's parsed info.
        // We need to peek at the parsed model to know the shape.
        let onnx = parser::parse_model(model_path)?;
        if onnx.dynamic_inputs.is_empty() {
            return Err("model has no dynamic inputs".into());
        }
        let first = &onnx.dynamic_inputs[0];
        if onnx.dynamic_inputs.len() > 1 {
            log::info!(
                "model has {} dynamic inputs; using all-zeros for each",
                onnx.dynamic_inputs.len()
            );
        }
        let shape = first
            .concrete_shape()
            .ok_or("model has symbolic dims; provide --shape to specify input shape")?;
        Buffer::new(&shape.0, first.dtype)
    };

    // If the model has multiple dynamic inputs, we need all of them.
    // Re-parse to check count and build zero buffers for extras.
    let onnx = parser::parse_model(model_path)?;
    let mut buffers: Vec<Buffer> = Vec::new();
    if input_arg.is_some() {
        buffers.push(input_buf);
        // Fill remaining dynamic inputs with zeros
        for inp in onnx.dynamic_inputs.iter().skip(1) {
            let shape = inp
                .concrete_shape()
                .ok_or("model has symbolic dims; provide --shape to specify input shape")?;
            buffers.push(Buffer::new(&shape.0, inp.dtype));
        }
    } else {
        for inp in &onnx.dynamic_inputs {
            let shape = inp
                .concrete_shape()
                .ok_or("model has symbolic dims; provide --shape to specify input shape")?;
            buffers.push(Buffer::new(&shape.0, inp.dtype));
        }
    }

    let refs: Vec<&Buffer> = buffers.iter().collect();
    let outputs = model.run(&refs)?;
    let output = &outputs[0];

    // Write or print output
    if let Some(out_path) = output_path {
        let bytes: Vec<u8> = output
            .as_slice::<f32>()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        std::fs::write(out_path, &bytes)?;
        log::info!(
            "wrote {} f32 values to {}",
            output.as_slice::<f32>().len(),
            out_path.display()
        );
    } else {
        let stdout = io::stdout();
        let mut out = stdout.lock();
        for (i, v) in output.as_slice::<f32>().iter().enumerate() {
            writeln!(out, "[{i}] {v}")?;
        }
    }

    Ok(())
}

fn read_input_bytes(src: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    if src == "-" {
        let mut buf = Vec::new();
        io::stdin().read_to_end(&mut buf)?;
        Ok(buf)
    } else {
        Ok(std::fs::read(src)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn cli_run_prompt_parses() {
        let cli = Cli::try_parse_from([
            "homura",
            "run",
            "tests/fixtures",
            "--prompt",
            "The meaning of life is",
            "--max-tokens",
            "50",
        ])
        .expect("failed to parse CLI");

        match cli.command {
            Commands::Run {
                model,
                prompt,
                max_tokens,
                ..
            } => {
                assert_eq!(model, PathBuf::from("tests/fixtures"));
                assert_eq!(prompt.as_deref(), Some("The meaning of life is"));
                assert_eq!(max_tokens, 50);
            }
            _ => panic!("expected Run subcommand"),
        }
    }

    #[test]
    fn cli_run_default_max_tokens() {
        let cli = Cli::try_parse_from(["homura", "run", "model.onnx", "--prompt", "hello"])
            .expect("failed to parse CLI");

        match cli.command {
            Commands::Run { max_tokens, .. } => {
                assert_eq!(max_tokens, 100);
            }
            _ => panic!("expected Run subcommand"),
        }
    }

    #[test]
    fn cli_run_no_prompt_keeps_existing_behavior() {
        let cli =
            Cli::try_parse_from(["homura", "run", "model.onnx"]).expect("failed to parse CLI");

        match cli.command {
            Commands::Run { prompt, input, .. } => {
                assert!(prompt.is_none());
                assert!(input.is_none());
            }
            _ => panic!("expected Run subcommand"),
        }
    }
}
