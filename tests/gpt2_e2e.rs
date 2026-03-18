use std::collections::HashSet;

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

/// GPT-2 prefill through the emitter.
/// Verifies compilation succeeds and produces finite logits.
#[test]
#[ignore = "slow: compiles GPT-2 prefill model"]
fn gpt2_prefill_emitter_only() {
    let model =
        Model::load_with_dynamic_dims("tests/fixtures/gpt2_decoder_model.onnx", HashSet::new())
            .expect("failed to load GPT-2 decoder model (emitter)");

    let input_ids = Buffer::from_slice::<i64>(&[15496, 995, 318, 257], &[1, 4], DType::I64);
    let attention_mask = Buffer::from_slice::<i64>(&[1, 1, 1, 1], &[1, 4], DType::I64);

    let outputs = model
        .run(&[&input_ids, &attention_mask])
        .expect("emitter inference failed");

    // 25 outputs: logits + 24 KV tensors.
    assert_eq!(outputs.len(), 25, "expected 25 outputs");
    assert_eq!(outputs[0].shape().0, vec![1, 4, 50257], "logits shape");

    let logits = outputs[0].as_slice::<f32>();
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "logits contain NaN/Inf"
    );
}

/// Task 8.8: GPT-2 decode model with dynamic `past_sequence_length` compiles
/// once and produces sensible outputs for a single decode step.
///
/// The decode model takes: input_ids [1,1], 24 KV tensors [1,12,past_len,64],
/// attention_mask [1,past_len+1]. Output: logits [1,1,50257] + 24 new KV tensors
/// [1,12,past_len+1,64].
#[test]
#[ignore = "slow: compiles GPT-2 decode model (~90s first run)"]
fn gpt2_decode_emitter_dynamic_past_sequence_length() {
    let keep_dynamic: HashSet<String> = [
        "past_sequence_length".to_string(),
        "past_sequence_length + 1".to_string(),
    ]
    .into_iter()
    .collect();

    let model = Model::load_with_dynamic_dims(
        "tests/fixtures/gpt2_decoder_with_past_model.onnx",
        keep_dynamic,
    )
    .expect("failed to load GPT-2 decode model");

    const PAST_LEN: usize = 4;
    const NUM_KV: usize = 24;
    const NUM_HEADS: usize = 12;
    const HEAD_DIM: usize = 64;

    // 24 KV cache tensors, each [1, 12, PAST_LEN, 64] filled with zeros —
    // simulating prefill output that seeds the persistent KV cache.
    let kv_size = NUM_HEADS * PAST_LEN * HEAD_DIM;
    let kv_data: Vec<f32> = vec![0.0; kv_size];
    let kv_buffers: Vec<Buffer> = (0..NUM_KV)
        .map(|_| {
            Buffer::from_slice::<f32>(
                &kv_data,
                &[1, NUM_HEADS as u64, PAST_LEN as u64, HEAD_DIM as u64],
                DType::F32,
            )
        })
        .collect();

    // Initialize the persistent KV cache from the simulated prefill outputs.
    const MAX_SEQ_LEN: usize = 1024;
    model
        .init_kv_cache(&kv_buffers, MAX_SEQ_LEN)
        .expect("init_kv_cache failed");

    // Single input token (e.g., token 257 = "a").
    let input_ids = Buffer::from_slice::<i64>(&[257], &[1, 1], DType::I64);

    // Attention mask: 1s for the PAST_LEN real positions, 1 for the new token.
    let mask_len = PAST_LEN + 1;
    let mask_data: Vec<i64> = vec![1i64; mask_len];
    let attention_mask = Buffer::from_slice::<i64>(&mask_data, &[1, mask_len as u64], DType::I64);

    let outputs = model
        .run_kv(&[&input_ids, &attention_mask], MAX_SEQ_LEN)
        .expect("decode inference failed");

    // run_kv returns only non-KV outputs: [logits].
    assert_eq!(outputs.len(), 1, "expected 1 output (logits only)");

    // Logits: [1, 1, 50257].
    assert_eq!(
        outputs[0].shape().0,
        vec![1, 1, 50257],
        "logits shape mismatch"
    );
    assert_eq!(outputs[0].dtype(), DType::F32);

    // Logits must be finite.
    let logits = outputs[0].as_slice::<f32>();
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "decode logits contain NaN or Inf"
    );

    // Predicted token must be in valid vocab range.
    let predicted = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    assert!(
        predicted < 50257,
        "predicted token {predicted} out of vocab range"
    );
}
