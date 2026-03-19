use std::path::Path;

use homura::hf::model::HfModel;
use homura::hf::tokenizer::HfTokenizer;
use homura::{Buffer, DType};

const MODEL_DIR: &str = concat!(
    env!("HOME"),
    "/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/",
    "snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
);

fn model_available() -> bool {
    Path::new(MODEL_DIR).join("config.json").exists()
        && Path::new(MODEL_DIR).join("model.safetensors").exists()
}

#[test]
fn qwen2_compile_and_forward() {
    if !model_available() {
        eprintln!("skipping: Qwen2.5-0.5B not found at {MODEL_DIR}");
        return;
    }

    let t0 = std::time::Instant::now();
    let model = HfModel::load(Path::new(MODEL_DIR)).expect("load failed");
    eprintln!("model loaded in {:.2}s", t0.elapsed().as_secs_f64());

    let config = model.config();
    assert_eq!(config.hidden_size, 896);
    assert_eq!(config.num_hidden_layers, 24);

    // Create dummy input: [1, 3] (3 tokens)
    let input_ids = Buffer::from_slice::<i64>(&[1, 2, 3], &[1, 3], DType::I64);

    let t1 = std::time::Instant::now();
    let outputs = model.run(&input_ids).expect("forward pass failed");
    eprintln!("forward pass in {:.2}s", t1.elapsed().as_secs_f64());

    // Should produce logits [1, 3, vocab_size] + KV cache outputs (2 per layer).
    let expected_outputs = 1 + config.num_hidden_layers * 2;
    assert_eq!(outputs.len(), expected_outputs);
    let logits = &outputs[0];
    assert_eq!(logits.shape().0, vec![1, 3, config.vocab_size as u64]);
}

#[test]
fn qwen2_generate_text() {
    if !model_available() {
        eprintln!("skipping: Qwen2.5-0.5B not found at {MODEL_DIR}");
        return;
    }

    let model = HfModel::load(Path::new(MODEL_DIR)).expect("load failed");
    let tokenizer = HfTokenizer::from_file(&Path::new(MODEL_DIR).join("tokenizer.json"))
        .expect("tokenizer load failed");

    let prompt = "The capital of France is";
    let input_ids_vec: Vec<i64> = tokenizer
        .encode(prompt)
        .iter()
        .map(|&id| id as i64)
        .collect();
    let seq_len = input_ids_vec.len();
    let input_ids = Buffer::from_slice::<i64>(&input_ids_vec, &[1, seq_len as u64], DType::I64);

    // Prefill
    let t0 = std::time::Instant::now();
    let outputs = model.run(&input_ids).expect("prefill failed");
    eprintln!(
        "prefill ({seq_len} tokens) in {:.2}s",
        t0.elapsed().as_secs_f64()
    );

    let logits = &outputs[0];
    let vocab_size = model.config().vocab_size;

    // Greedy decode from prefill logits (last token position)
    let logits_data = logits.as_slice::<f32>();
    let last_pos_start = (seq_len - 1) * vocab_size;
    let last_pos_logits = &logits_data[last_pos_start..last_pos_start + vocab_size];
    let next_token = last_pos_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;

    let decoded = tokenizer.decode(&[next_token as u32]);
    eprintln!("prompt: {prompt:?}");
    eprintln!("next token: {next_token} -> {decoded:?}");

    // The model should produce something reasonable after "The capital of France is"
    // (likely "Paris" or similar)
    assert!(!decoded.is_empty(), "expected non-empty decoded token");
}
