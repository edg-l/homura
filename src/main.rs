use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, Subcommand};
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
    /// Show compilation progress (MLIR passes, kernel timing)
    #[arg(long, short, global = true)]
    verbose: bool,

    #[command(subcommand)]
    command: Commands,
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
    if cli.verbose {
        log::set_level(log::Level::Debug);
    }
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
            temperature,
            top_p,
            repetition_penalty,
            top_k,
            min_p,
            frequency_penalty,
            presence_penalty,
            seed,
            stop,
        } => {
            if let Some(prompt_text) = prompt {
                let stop_sequences = stop
                    .map(|s| s.split(',').map(|s| s.to_string()).collect())
                    .unwrap_or_default();
                let sampling = homura::generate::SamplingConfig {
                    temperature,
                    top_k,
                    top_p,
                    min_p,
                    repetition_penalty,
                    frequency_penalty,
                    presence_penalty,
                    stop_sequences,
                    seed,
                };
                // Resolve model path: HF repo ID, local directory, or file.
                let model_dir = resolve_model_path(&model)?;
                if model_dir.join("config.json").exists()
                    && model_dir.join("model.safetensors").exists()
                {
                    cmd_generate_hf(&model_dir, &prompt_text, max_tokens, &sampling)
                } else {
                    cmd_generate(&model, &prompt_text, max_tokens, &sampling)
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
    }
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

    // Download the essential files. hf-hub caches them automatically.
    let files = [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ];
    let mut snapshot_dir = None;
    for file in &files {
        match repo.get(file) {
            Ok(path) => {
                if snapshot_dir.is_none() {
                    snapshot_dir = path.parent().map(|p| p.to_path_buf());
                }
            }
            Err(e) => {
                // tokenizer_config.json is optional
                if *file != "tokenizer_config.json" {
                    return Err(format!("failed to fetch {file}: {e}").into());
                }
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
    sampling: &homura::generate::SamplingConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    log::info!("loading HF model from {}", model_dir.display());
    let t_load = Instant::now();

    let model = homura::hf::model::HfModel::load(model_dir)?;
    let tokenizer =
        homura::hf::tokenizer::HfTokenizer::from_file(&model_dir.join("tokenizer.json"))?;

    log::info!("loaded in {:.2}s", t_load.elapsed().as_secs_f64());

    let max_seq_len = std::cmp::min(model.config().max_position_embeddings, 2048);
    log::info!("generating (max_tokens={max_tokens}, max_seq_len={max_seq_len})");

    let generated = model.generate(&tokenizer, prompt, max_tokens, max_seq_len, sampling)?;

    // If stdout is piped, print the full text at the end.
    if !atty::is(atty::Stream::Stdout) {
        println!("{}{}", prompt, generated);
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
