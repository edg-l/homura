use std::io::{self, Read, Write};
use std::path::PathBuf;

use clap::{Parser, Subcommand};
use homura::onnx::parser;
use homura::{Buffer, DType, Model};

// Ops that homura currently supports.
const SUPPORTED_OPS: &[&str] = &[
    "Add", "Sub", "Mul", "Div", "Relu", "MatMul", "Gemm", "Conv", "MaxPool", "Reshape",
];

#[derive(Parser)]
#[command(name = "homura", about = "ONNX inference with homura")]
struct Cli {
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
    /// Run inference on an ONNX model
    Run {
        /// Path to the ONNX model file
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
    },
}

fn main() {
    let cli = Cli::parse();
    if let Err(e) = run(cli) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    match cli.command {
        Commands::Info { model } => cmd_info(&model),
        Commands::Run {
            model,
            input,
            shape,
            output,
        } => cmd_run(
            &model,
            input.as_deref(),
            shape.as_deref(),
            output.as_deref(),
        ),
    }
}

// ── info ─────────────────────────────────────────────────────────────────────

fn cmd_info(path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let onnx = parser::parse_model(path)?;

    // Dynamic inputs
    println!("Inputs ({}):", onnx.dynamic_inputs.len());
    for inp in &onnx.dynamic_inputs {
        println!("  {} : {:?} {:?}", inp.name, inp.dtype, inp.shape.0);
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
            eprintln!(
                "note: model has {} dynamic inputs; using all-zeros for each",
                onnx.dynamic_inputs.len()
            );
        }
        Buffer::new(&first.shape.0, first.dtype)
    };

    // If the model has multiple dynamic inputs, we need all of them.
    // Re-parse to check count and build zero buffers for extras.
    let onnx = parser::parse_model(model_path)?;
    let mut buffers: Vec<Buffer> = Vec::new();
    if input_arg.is_some() {
        buffers.push(input_buf);
        // Fill remaining dynamic inputs with zeros
        for inp in onnx.dynamic_inputs.iter().skip(1) {
            buffers.push(Buffer::new(&inp.shape.0, inp.dtype));
        }
    } else {
        for inp in &onnx.dynamic_inputs {
            buffers.push(Buffer::new(&inp.shape.0, inp.dtype));
        }
    }

    let refs: Vec<&Buffer> = buffers.iter().collect();
    let output = model.run(&refs)?;

    // Write or print output
    if let Some(out_path) = output_path {
        let bytes: Vec<u8> = output
            .as_slice::<f32>()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        std::fs::write(out_path, &bytes)?;
        eprintln!(
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
