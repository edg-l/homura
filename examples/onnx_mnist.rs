use std::time::Instant;

use homura::{Buffer, DType, Model};

fn main() {
    let t0 = Instant::now();
    let model = Model::load("tests/fixtures/mnist-12.onnx").expect("failed to load mnist-12.onnx");
    let compile_ms = t0.elapsed().as_millis();

    let pixels = match std::env::args().nth(1) {
        Some(path) => load_image(&path),
        None => {
            eprintln!("usage: cargo run --example onnx_mnist -- <image_path>");
            eprintln!("       (using blank image as default)\n");
            vec![0.0f32; 784]
        }
    };

    let input = Buffer::from_slice::<f32>(&pixels, &[1, 1, 28, 28], DType::F32);

    let t1 = Instant::now();
    let output = model.run(&[&input]).expect("inference failed");
    let run_ms = t1.elapsed().as_millis();

    let logits = output.as_slice::<f32>();

    println!("MNIST logits:");
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_logit = logits.iter().cloned().fold(f32::INFINITY, f32::min);
    let range = (max_logit - min_logit).max(1e-6);

    for (digit, &logit) in logits.iter().enumerate() {
        let bar_len = ((logit - min_logit) / range * 30.0) as usize;
        let bar = "█".repeat(bar_len);
        println!("  {digit}: {logit:>8.3}  {bar}");
    }

    let predicted = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    println!("\nPredicted digit: {predicted}");
    println!("compile: {compile_ms}ms | inference: {run_ms}ms");
}

fn load_image(path: &str) -> Vec<f32> {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("failed to open {path}: {e}"))
        .grayscale()
        .resize_exact(28, 28, image::imageops::FilterType::Lanczos3)
        .into_luma8();

    // MNIST expects white digit on black background, pixel values 0.0-1.0.
    // Standard images are typically black on white, so invert if the mean
    // brightness is high (>0.5 means mostly white background).
    let raw: Vec<f32> = img.pixels().map(|p| p.0[0] as f32 / 255.0).collect();
    let mean: f32 = raw.iter().sum::<f32>() / raw.len() as f32;
    if mean > 0.5 {
        // Invert: black-on-white → white-on-black
        raw.iter().map(|&v| 1.0 - v).collect()
    } else {
        raw
    }
}
