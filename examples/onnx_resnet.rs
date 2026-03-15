use std::time::Instant;

use homura::{Buffer, DType, Model};

// ImageNet mean/std (RGB order)
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

fn load_class_labels() -> Vec<String> {
    let text = std::fs::read_to_string("tests/fixtures/imagenet_classes.txt")
        .expect("failed to read imagenet_classes.txt");
    text.lines().map(|l| l.trim().to_string()).collect()
}

fn main() {
    let t0 = Instant::now();
    let model =
        Model::load("tests/fixtures/resnet18-v1-7.onnx").expect("failed to load resnet18");
    let compile_ms = t0.elapsed().as_millis();

    let classes = load_class_labels();

    let pixels = match std::env::args().nth(1) {
        Some(path) => load_and_preprocess(&path),
        None => {
            eprintln!("usage: cargo run --example onnx_resnet -- <image_path>");
            eprintln!("       (using random noise as default)\n");
            (0..3 * 224 * 224)
                .map(|i| (i as f32 * 0.0001).sin() * 0.5)
                .collect()
        }
    };

    let input = Buffer::from_slice::<f32>(&pixels, &[1, 3, 224, 224], DType::F32);

    let t1 = Instant::now();
    let outputs = model.run(&[&input]).expect("inference failed");
    let run_ms = t1.elapsed().as_millis();

    let logits = outputs[0].as_slice::<f32>();

    // Sort by logit value, print top 5
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("Top 5 predictions:");
    for (rank, &(class_idx, logit)) in indexed.iter().take(5).enumerate() {
        let name = classes.get(class_idx).map(|s| s.as_str()).unwrap_or("?");
        println!("  #{}: {name} (class {class_idx}, logit {logit:.3})", rank + 1);
    }
    println!("\ncompile: {compile_ms}ms | inference: {run_ms}ms");
}

fn load_and_preprocess(path: &str) -> Vec<f32> {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("failed to open {path}: {e}"))
        .resize_exact(224, 224, image::imageops::FilterType::Lanczos3)
        .into_rgb8();

    // NCHW layout: [1, 3, 224, 224], normalized with ImageNet mean/std
    let mut data = vec![0.0f32; 3 * 224 * 224];
    for y in 0..224 {
        for x in 0..224 {
            let pixel = img.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                let val = pixel[c] as f32 / 255.0;
                data[c * 224 * 224 + y * 224 + x] = (val - MEAN[c]) / STD[c];
            }
        }
    }
    data
}
