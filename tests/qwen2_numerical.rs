//! Numerical debugging: compare our outputs against PyTorch reference values.

use std::path::Path;

use homura::graph_builder::GraphContext;
use homura::hf::config::TransformerConfig;
use homura::hf::model::HfModel;
use homura::hf::safetensors::load_safetensors;
use homura::hf::weights::load_transformer_weights;
use homura::{Buffer, DType};

const MODEL_DIR: &str = concat!(
    env!("HOME"),
    "/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/",
    "snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
);

fn model_available() -> bool {
    Path::new(MODEL_DIR).join("config.json").exists()
}

fn load_ref(name: &str) -> Vec<f32> {
    let path = format!("/tmp/qwen2_ref_{name}.bin");
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|_| panic!("run the Python reference script first: missing {path}"));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Test embedding lookup matches PyTorch.
#[test]
fn compare_embedding() {
    if !model_available() {
        eprintln!("skipping");
        return;
    }

    let config = TransformerConfig::load(&Path::new(MODEL_DIR).join("config.json")).unwrap();
    let tensors = load_safetensors(&Path::new(MODEL_DIR).join("model.safetensors")).unwrap();
    let weights = load_transformer_weights(&config, tensors).unwrap();

    let ref_embed = load_ref("embed"); // [5, 896]
    let hidden = config.hidden_size;

    // Token IDs for "The capital of France is": [785, 6722, 315, 9625, 374]
    let token_ids: Vec<i64> = vec![785, 6722, 315, 9625, 374];

    // Do embedding lookup via GraphBuilder
    let ctx = GraphContext::new();
    let mut gb = ctx.builder();
    let ids = gb.input(&[Some(1), Some(5)], DType::I64);
    let embed_w = gb.input(
        &[Some(config.vocab_size as u64), Some(hidden as u64)],
        DType::F32,
    );
    let out = gb.emit_embedding(&embed_w, &ids);
    let graph = gb.compile(&[&out]).unwrap();

    let ids_buf = Buffer::from_slice::<i64>(&token_ids, &[1, 5], DType::I64);
    let outputs = graph.run(&[&ids_buf, &weights.embed_tokens_weight]);
    let our_embed = outputs[0].as_slice::<f32>();

    let diff = max_diff(&ref_embed, our_embed);
    eprintln!("embedding max diff: {diff:.8}");
    eprintln!("  ref[0..5]: {:?}", &ref_embed[..5]);
    eprintln!("  our[0..5]: {:?}", &our_embed[..5]);
    assert!(diff < 1e-5, "embedding mismatch: max_diff={diff}");
    eprintln!("embedding OK");
}

/// Test RMSNorm matches PyTorch.
#[test]
fn compare_rmsnorm() {
    if !model_available() {
        eprintln!("skipping");
        return;
    }

    let config = TransformerConfig::load(&Path::new(MODEL_DIR).join("config.json")).unwrap();
    let tensors = load_safetensors(&Path::new(MODEL_DIR).join("model.safetensors")).unwrap();
    let weights = load_transformer_weights(&config, tensors).unwrap();

    let ref_embed = load_ref("embed"); // [5, 896]
    let ref_normed = load_ref("normed0"); // [5, 896]
    let hidden = config.hidden_size as u64;

    let ctx = GraphContext::new();
    let mut gb = ctx.builder();
    let input = gb.input(&[Some(1), Some(5), Some(hidden)], DType::F32);
    let ln_w = gb.input(&[Some(hidden)], DType::F32);
    let out = gb.emit_rms_norm(&input, &ln_w, config.rms_norm_eps as f32);
    let graph = gb.compile(&[&out]).unwrap();

    let embed_buf = Buffer::from_slice::<f32>(&ref_embed, &[1, 5, hidden], DType::F32);
    let ln_w_buf = &weights.layers[0].input_layernorm_weight;
    let outputs = graph.run(&[&embed_buf, ln_w_buf]);
    let our_normed = outputs[0].as_slice::<f32>();

    let diff = max_diff(&ref_normed, our_normed);
    eprintln!("RMSNorm max diff: {diff:.8}");
    eprintln!("  ref[0..5]: {:?}", &ref_normed[..5]);
    eprintln!("  our[0..5]: {:?}", &our_normed[..5]);
    assert!(diff < 1e-4, "RMSNorm mismatch: max_diff={diff}");
    eprintln!("RMSNorm OK");
}

/// Test Q projection matches PyTorch.
/// PyTorch does: Q = normed @ q_proj.weight.T + q_proj.bias
/// We do: Q = normed @ q_proj_weight (already transposed at load) + bias
#[test]
fn compare_q_projection() {
    if !model_available() {
        eprintln!("skipping");
        return;
    }

    let config = TransformerConfig::load(&Path::new(MODEL_DIR).join("config.json")).unwrap();
    let tensors = load_safetensors(&Path::new(MODEL_DIR).join("model.safetensors")).unwrap();
    let weights = load_transformer_weights(&config, tensors).unwrap();

    let ref_normed = load_ref("normed0"); // [5, 896]
    let ref_q = load_ref("q0"); // [5, 896]
    let hidden = config.hidden_size as u64;

    let ctx = GraphContext::new();
    let mut gb = ctx.builder();
    let input = gb.input(&[Some(1), Some(5), Some(hidden)], DType::F32);
    let q_w = gb.input(&[Some(hidden), Some(hidden)], DType::F32);
    let q_b = gb.input(&[Some(hidden)], DType::F32);
    let q = gb.emit_matmul(&input, &q_w);
    let out = gb.emit_add(&q, &q_b);
    let graph = gb.compile(&[&out]).unwrap();

    let normed_buf = Buffer::from_slice::<f32>(&ref_normed, &[1, 5, hidden], DType::F32);
    let outputs = graph.run(&[
        &normed_buf,
        &weights.layers[0].q_proj_weight,
        weights.layers[0].q_proj_bias.as_ref().unwrap(),
    ]);
    let our_q = outputs[0].as_slice::<f32>();

    let diff = max_diff(&ref_q, our_q);
    eprintln!("Q projection max diff: {diff:.8}");
    eprintln!("  ref[0..5]: {:?}", &ref_q[..5]);
    eprintln!("  our[0..5]: {:?}", &our_q[..5]);
    assert!(diff < 1e-3, "Q projection mismatch: max_diff={diff}");
    eprintln!("Q projection OK");
}

/// Test full forward pass logits match PyTorch.
#[test]
fn compare_full_logits() {
    if !model_available() {
        eprintln!("skipping");
        return;
    }

    let ref_logits = load_ref("logits"); // [151936]

    let model = HfModel::load(Path::new(MODEL_DIR)).expect("load failed");

    let token_ids: Vec<i64> = vec![785, 6722, 315, 9625, 374];
    let input_ids = Buffer::from_slice::<i64>(&token_ids, &[1, 5], DType::I64);
    let outputs = model.run(&input_ids).expect("forward failed");

    let logits = &outputs[0];
    let our_logits = logits.as_slice::<f32>();
    let vocab_size = 151936;

    // Compare last position logits
    let last_start = 4 * vocab_size;
    let our_last = &our_logits[last_start..last_start + vocab_size];

    let diff = max_diff(&ref_logits, our_last);

    // Top-5 from our model
    let mut indexed: Vec<(usize, f32)> = our_last.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    eprintln!("full logits max diff: {diff:.4}");
    eprintln!("our top-5:");
    for (i, (idx, val)) in indexed.iter().take(5).enumerate() {
        eprintln!("  {i}: token {idx} logit={val:.4}");
    }
    eprintln!("ref top-1: token 12095 logit=17.8433");
    eprintln!(
        "our top-1: token {} logit={:.4}",
        indexed[0].0, indexed[0].1
    );

    // Check our top-1 matches ref top-1
    if indexed[0].0 == 12095 {
        eprintln!("TOP-1 MATCHES: Paris!");
    } else {
        eprintln!(
            "TOP-1 MISMATCH: got token {} instead of 12095 (Paris)",
            indexed[0].0
        );
    }
}
