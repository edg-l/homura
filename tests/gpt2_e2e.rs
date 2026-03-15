use homura::{Buffer, DType, Model};

/// Task 6.1: GPT-2 decoder model loads with symbolic dims.
#[test]
fn gpt2_decoder_model_loads_with_symbolic_dims() {
    let model = Model::load("tests/fixtures/gpt2_decoder_model.onnx")
        .expect("failed to load GPT-2 decoder model");
    // Model should load without error (compilation deferred due to symbolic dims).
    let _ = model;
}

/// Task 6.2: Run decoder model, verify output shapes.
#[test]
fn gpt2_decoder_produces_logits_and_kv_cache() {
    let model = Model::load("tests/fixtures/gpt2_decoder_model.onnx")
        .expect("failed to load GPT-2 decoder model");

    // batch_size=1, sequence_length=4 (short prompt)
    let input_ids = Buffer::from_slice::<i64>(&[15496, 995, 318, 257], &[1, 4], DType::I64);
    let attention_mask = Buffer::from_slice::<i64>(&[1, 1, 1, 1], &[1, 4], DType::I64);

    let outputs = model
        .run(&[&input_ids, &attention_mask])
        .expect("inference failed");

    // Should produce 25 outputs: logits + 12 layers × (key + value)
    assert_eq!(outputs.len(), 25, "expected 25 outputs (logits + 24 KV)");

    // Logits: [batch=1, seq_len=4, vocab=50257]
    assert_eq!(
        outputs[0].shape().0,
        vec![1, 4, 50257],
        "logits shape mismatch"
    );
    assert_eq!(outputs[0].dtype(), DType::F32);

    // KV cache: each is [batch=1, heads=12, seq_len=4, head_dim=64]
    for i in 1..25 {
        assert_eq!(
            outputs[i].shape().0,
            vec![1, 12, 4, 64],
            "KV tensor {i} shape mismatch"
        );
        assert_eq!(outputs[i].dtype(), DType::F32);
    }

    // Logits should be finite
    let logits = outputs[0].as_slice::<f32>();
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "logits contain NaN or Inf"
    );
}
