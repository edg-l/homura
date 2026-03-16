use std::collections::HashSet;

use homura::{Buffer, DType, GraphContext, Model};

fn load_image(path: &str) -> Vec<f32> {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("failed to open {path}: {e}"))
        .grayscale()
        .resize_exact(28, 28, image::imageops::FilterType::Lanczos3)
        .into_luma8();

    let raw: Vec<f32> = img.pixels().map(|p| p.0[0] as f32 / 255.0).collect();
    let mean: f32 = raw.iter().sum::<f32>() / raw.len() as f32;
    if mean > 0.5 {
        raw.iter().map(|&v| 1.0 - v).collect()
    } else {
        raw
    }
}

#[test]
fn mnist_predicts_7_from_image() {
    let model = Model::load("tests/fixtures/mnist-12.onnx").expect("load failed");
    let pixels = load_image("tests/fixtures/digit7.png");
    let input = Buffer::from_slice::<f32>(&pixels, &[1, 1, 28, 28], DType::F32);
    let outputs = model.run(&[&input]).expect("inference failed");
    let logits = outputs[0].as_slice::<f32>();

    assert_eq!(logits.len(), 10);
    assert!(logits.iter().all(|v| v.is_finite()));

    let predicted = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    assert_eq!(
        predicted, 7,
        "expected digit 7, got {predicted} (logits: {logits:?})"
    );
}

/// MNIST through the GraphBuilder emitter predicts digit 7.
#[test]
fn mnist_emitter_predicts_7_from_image() {
    let model =
        Model::load_with_dynamic_dims("tests/fixtures/mnist-12.onnx", HashSet::new())
            .expect("load failed");
    let pixels = load_image("tests/fixtures/digit7.png");
    let input = Buffer::from_slice::<f32>(&pixels, &[1, 1, 28, 28], DType::F32);
    let outputs = model.run(&[&input]).expect("inference failed");
    let logits = outputs[0].as_slice::<f32>();

    assert_eq!(logits.len(), 10);
    assert!(logits.iter().all(|v| v.is_finite()));

    let predicted = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    assert_eq!(
        predicted, 7,
        "expected digit 7, got {predicted} (logits: {logits:?})"
    );
}

/// Task 2.6: MNIST with split_threshold=50 forces splitting into multiple
/// sub-functions. Result must match the non-split emitter path.
#[test]
fn mnist_emitter_split_threshold_50() {
    let model_bytes = std::fs::read("tests/fixtures/mnist-12.onnx").expect("read model");
    let onnx_model = homura::onnx::parser::parse_bytes(&model_bytes).expect("parse model");

    let ctx = GraphContext::new();
    let mut builder = ctx.builder();
    let (output_tensors, weights) = homura::onnx::emitter::emit_graph_with_split(
        &onnx_model,
        &mut builder,
        &HashSet::new(),
        50, // force splitting
    )
    .expect("emit_graph_with_split");

    let output_refs: Vec<&homura::graph_builder::Tensor<'_>> = output_tensors.iter().collect();
    let graph = builder.compile(&output_refs).expect("compile with splitting");

    // Build input buffer.
    let pixels = load_image("tests/fixtures/digit7.png");
    let input = Buffer::from_slice::<f32>(&pixels, &[1, 1, 28, 28], DType::F32);

    // Build weight buffers (same order as emitter produced them).
    let mut run_inputs: Vec<&Buffer> = vec![&input];
    for w in &weights {
        run_inputs.push(w);
    }
    let outputs = graph.run(&run_inputs);
    let logits = outputs[0].as_slice::<f32>();

    assert_eq!(logits.len(), 10);
    assert!(logits.iter().all(|v| v.is_finite()), "logits contain NaN/Inf");

    let predicted = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    assert_eq!(
        predicted, 7,
        "expected digit 7 with splitting, got {predicted} (logits: {logits:?})"
    );
}
