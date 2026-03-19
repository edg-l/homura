/// Debug test: run kernels one at a time to find which one corrupts the heap.
use std::path::Path;

use homura::hf::config::TransformerConfig;
use homura::hf::model::HfModel;
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

/// Test just the embedding kernel in isolation using GraphBuilder directly.
#[test]
fn test_embed_kernel_standalone() {
    if !model_available() {
        eprintln!("skipping: model not found");
        return;
    }

    use homura::graph_builder::{GraphContext, TransformMode};

    let ctx = GraphContext::new();
    let mut gb = ctx.builder();
    // Use static shape to avoid dynamic dim issues
    let input_ids = gb.input(&[Some(1), Some(3)], DType::I64);
    let embed_w = gb.input(&[Some(10), Some(4)], DType::F32); // small: 10 vocab, 4 hidden
    let hidden = gb.emit_embedding(&embed_w, &input_ids);
    let graph = gb.compile(&[&hidden]).unwrap();

    // Embed weight: row i = [i, i, i, i]
    let mut embed_data = vec![0.0f32; 10 * 4];
    for i in 0..10 {
        for j in 0..4 {
            embed_data[i * 4 + j] = i as f32;
        }
    }
    let embed_buf = Buffer::from_slice::<f32>(&embed_data, &[10, 4], DType::F32);
    let ids = Buffer::from_slice::<i64>(&[2, 5, 7], &[1, 3], DType::I64);

    let outputs = graph.run(&[&ids, &embed_buf]);
    let out = &outputs[0];
    eprintln!("embed output shape: {:?}", out.shape().0);
    assert_eq!(out.shape().0, vec![1, 3, 4]);

    let data = out.as_slice::<f32>();
    // Token 2 -> row 2 -> [2, 2, 2, 2]
    assert!((data[0] - 2.0).abs() < 1e-5, "got {}", data[0]);
    // Token 5 -> row 5 -> [5, 5, 5, 5]
    assert!((data[4] - 5.0).abs() < 1e-5, "got {}", data[4]);
    // Token 7 -> row 7 -> [7, 7, 7, 7]
    assert!((data[8] - 7.0).abs() < 1e-5, "got {}", data[8]);
    eprintln!("embedding kernel OK");
}

/// Test QKV kernel in isolation with small dims.
#[test]
fn test_qkv_kernel_standalone() {
    use homura::graph_builder::{GraphContext, TransformMode};

    let hidden = 8u64;
    let kv_dim = 4u64; // 2 kv_heads * 2 head_dim
    let half_dim = 1u64; // head_dim/2

    let ctx = GraphContext::new();
    let mut gb = ctx.builder();

    let h = gb.input(&[Some(1), Some(2), Some(hidden)], DType::F32); // static seq=2
    let ln_w = gb.input(&[Some(hidden)], DType::F32);
    let q_w = gb.input(&[Some(hidden), Some(hidden)], DType::F32);
    let q_b = gb.input(&[Some(hidden)], DType::F32);
    let k_w = gb.input(&[Some(hidden), Some(kv_dim)], DType::F32);
    let k_b = gb.input(&[Some(kv_dim)], DType::F32);
    let v_w = gb.input(&[Some(hidden), Some(kv_dim)], DType::F32);
    let v_b = gb.input(&[Some(kv_dim)], DType::F32);
    let cos = gb.input(&[Some(2), Some(half_dim)], DType::F32);
    let sin = gb.input(&[Some(2), Some(half_dim)], DType::F32);

    // RMSNorm
    let normed = gb.emit_rms_norm(&h, &ln_w, 1e-6);
    // Q
    let mut q = gb.emit_matmul(&normed, &q_w);
    q = gb.emit_add(&q, &q_b);
    // K
    let mut k = gb.emit_matmul(&normed, &k_w);
    k = gb.emit_add(&k, &k_b);
    // V
    let mut v = gb.emit_matmul(&normed, &v_w);
    v = gb.emit_add(&v, &v_b);

    // Reshape Q: [1, 2, 8] -> [1, 2, 4, 2]  (4 heads, head_dim=2)
    let q_4d = gb.emit_reshape(&q, &[1, 2, 4, 2]);
    let k_4d = gb.emit_reshape(&k, &[1, 2, 2, 2]); // 2 kv_heads, head_dim=2
    let v_4d = gb.emit_reshape(&v, &[1, 2, 2, 2]);

    // RoPE
    let q_rope = gb.emit_rope(&q_4d, &cos, &sin);
    let k_rope = gb.emit_rope(&k_4d, &cos, &sin);

    // Transpose K/V to BHSD
    let k_bhsd = gb.emit_transpose(&k_rope, &[0, 2, 1, 3]);
    let v_bhsd = gb.emit_transpose(&v_4d, &[0, 2, 1, 3]);

    let graph = gb.compile(&[&q_rope, &k_bhsd, &v_bhsd]).unwrap();

    // Create small test data (all ones for simplicity)
    let h_buf = Buffer::from_slice::<f32>(&vec![1.0; 16], &[1, 2, 8], DType::F32);
    let ln_w_buf = Buffer::from_slice::<f32>(&vec![1.0; 8], &[8], DType::F32);
    let q_w_buf = Buffer::from_slice::<f32>(&vec![0.1; 64], &[8, 8], DType::F32);
    let q_b_buf = Buffer::from_slice::<f32>(&vec![0.0; 8], &[8], DType::F32);
    let k_w_buf = Buffer::from_slice::<f32>(&vec![0.1; 32], &[8, 4], DType::F32);
    let k_b_buf = Buffer::from_slice::<f32>(&vec![0.0; 4], &[4], DType::F32);
    let v_w_buf = Buffer::from_slice::<f32>(&vec![0.1; 32], &[8, 4], DType::F32);
    let v_b_buf = Buffer::from_slice::<f32>(&vec![0.0; 4], &[4], DType::F32);
    let cos_buf = Buffer::from_slice::<f32>(&vec![1.0; 2], &[2, 1], DType::F32);
    let sin_buf = Buffer::from_slice::<f32>(&vec![0.0; 2], &[2, 1], DType::F32);

    let outputs = graph.run(&[
        &h_buf, &ln_w_buf, &q_w_buf, &q_b_buf, &k_w_buf, &k_b_buf, &v_w_buf, &v_b_buf, &cos_buf,
        &sin_buf,
    ]);

    eprintln!("QKV outputs: {} tensors", outputs.len());
    for (i, out) in outputs.iter().enumerate() {
        eprintln!("  output[{i}] shape: {:?}", out.shape().0);
    }

    assert_eq!(outputs.len(), 3);
    // Q: [1, 2, 4, 2]
    assert_eq!(outputs[0].shape().0, vec![1, 2, 4, 2]);
    // K: [1, 2, 2, 2] (BHSD)
    assert_eq!(outputs[1].shape().0, vec![1, 2, 2, 2]);
    // V: [1, 2, 2, 2] (BHSD)
    assert_eq!(outputs[2].shape().0, vec![1, 2, 2, 2]);
    eprintln!("QKV kernel OK");
}

/// Test attention kernel in isolation with small static dims.
#[test]
fn test_attention_kernel_standalone() {
    use homura::graph_builder::GraphContext;

    let num_heads = 4u64;
    let kv_heads = 2u64;
    let head_dim = 2u64;
    let hidden = num_heads * head_dim; // 8
    let gqa_repeat = (num_heads / kv_heads) as usize; // 2

    let ctx = GraphContext::new();
    let mut gb = ctx.builder();

    // Q: [1, seq=2, num_heads=4, head_dim=2]
    let q = gb.input(
        &[Some(1), Some(2), Some(num_heads), Some(head_dim)],
        DType::F32,
    );
    // K: [1, kv_heads=2, total_seq=2, head_dim=2] (BHSD)
    let k = gb.input(
        &[Some(1), Some(kv_heads), Some(2), Some(head_dim)],
        DType::F32,
    );
    // V: [1, kv_heads=2, total_seq=2, head_dim=2]
    let v = gb.input(
        &[Some(1), Some(kv_heads), Some(2), Some(head_dim)],
        DType::F32,
    );
    // Mask: [1, 1, seq=2, total_seq=2]
    let mask = gb.input(&[Some(1), Some(1), Some(2), Some(2)], DType::F32);
    // O weight: [hidden=8, hidden=8]
    let o_w = gb.input(&[Some(hidden), Some(hidden)], DType::F32);

    // Same ops as emit_attention_kernel
    let q_bhsd = gb.emit_transpose(&q, &[0, 2, 1, 3]);
    let k_bshd = gb.emit_transpose(&k, &[0, 2, 1, 3]);
    let k_expanded = gb.emit_repeat_kv(&k_bshd, gqa_repeat);
    let k_bhsd = gb.emit_transpose(&k_expanded, &[0, 2, 1, 3]);
    let v_bshd = gb.emit_transpose(&v, &[0, 2, 1, 3]);
    let v_expanded = gb.emit_repeat_kv(&v_bshd, gqa_repeat);
    let v_bhsd = gb.emit_transpose(&v_expanded, &[0, 2, 1, 3]);
    let k_t = gb.emit_transpose(&k_bhsd, &[0, 1, 3, 2]);
    let scores = gb.emit_matmul(&q_bhsd, &k_t);
    let scale = gb.emit_arith_constant(1.0 / (head_dim as f64).sqrt(), DType::F32);
    let scaled = gb.emit_mul(&scores, &scale);
    let masked = gb.emit_add(&scaled, &mask);
    let attn_weights = gb.emit_softmax(&masked, -1);
    let attn_output = gb.emit_matmul(&attn_weights, &v_bhsd);
    let attn_bshd = gb.emit_transpose(&attn_output, &[0, 2, 1, 3]);
    let attn_flat = gb.emit_reshape(&attn_bshd, &[1, 2, hidden as i64]);
    let out = gb.emit_matmul(&attn_flat, &o_w);

    let graph = gb.compile(&[&out]).unwrap();

    let q_buf = Buffer::from_slice::<f32>(&vec![0.1; 16], &[1, 2, 4, 2], DType::F32);
    let k_buf = Buffer::from_slice::<f32>(&vec![0.1; 8], &[1, 2, 2, 2], DType::F32);
    let v_buf = Buffer::from_slice::<f32>(&vec![0.1; 8], &[1, 2, 2, 2], DType::F32);
    let mask_buf = Buffer::from_slice::<f32>(
        &[0.0, f32::NEG_INFINITY, 0.0, 0.0],
        &[1, 1, 2, 2],
        DType::F32,
    );
    let o_w_buf = Buffer::from_slice::<f32>(&vec![0.1; 64], &[8, 8], DType::F32);

    let outputs = graph.run(&[&q_buf, &k_buf, &v_buf, &mask_buf, &o_w_buf]);
    let out = &outputs[0];
    eprintln!("attention output shape: {:?}", out.shape().0);
    assert_eq!(out.shape().0, vec![1, 2, 8]);
    for &v in out.as_slice::<f32>() {
        assert!(v.is_finite(), "attention output contains non-finite: {v}");
    }
    eprintln!("attention kernel static OK");
}

/// Test attention kernel with DYNAMIC shapes matching real Qwen2 dims.
/// This is the exact kernel that crashes in the full plan.
#[test]
fn test_attention_kernel_dynamic_qwen2() {
    use homura::graph_builder::GraphContext;

    // Binary search: try larger dims to find where it breaks
    let num_heads = 4u64;
    let head_dim = 64u64;
    let hidden = num_heads * head_dim; // 256

    let ctx = GraphContext::new();
    let mut gb = ctx.builder();

    // Dynamic seq dims — same as emit_attention_kernel but without GQA
    let q = gb.input(
        &[Some(1), None, Some(num_heads), Some(head_dim)],
        DType::F32,
    );
    // K/V already at num_heads (no GQA expansion needed)
    let k = gb.input(
        &[Some(1), Some(num_heads), None, Some(head_dim)],
        DType::F32,
    );
    let v = gb.input(
        &[Some(1), Some(num_heads), None, Some(head_dim)],
        DType::F32,
    );
    let mask = gb.input(&[Some(1), Some(1), None, None], DType::F32);
    let o_w = gb.input(&[Some(hidden), Some(hidden)], DType::F32);

    let q_bhsd = gb.emit_transpose(&q, &[0, 2, 1, 3]);
    // No repeat_kv — K/V are already num_heads
    let k_t = gb.emit_transpose(&k, &[0, 1, 3, 2]); // [1, 14, 64, ?]
    let scores = gb.emit_matmul(&q_bhsd, &k_t);
    let scale = gb.emit_arith_constant(1.0 / (head_dim as f64).sqrt(), DType::F32);
    let scaled = gb.emit_mul(&scores, &scale);
    let masked = gb.emit_add(&scaled, &mask);
    let attn_weights = gb.emit_softmax(&masked, -1);
    let attn_output = gb.emit_matmul(&attn_weights, &v);
    let attn_bshd = gb.emit_transpose(&attn_output, &[0, 2, 1, 3]);
    let attn_flat = gb.emit_reshape(&attn_bshd, &[1, -1, hidden as i64]);
    let out = gb.emit_matmul(&attn_flat, &o_w);

    let graph = gb.compile(&[&out]).unwrap();

    // seq=3, total_seq=3 (prefill, no past), no GQA
    let seq = 3usize;
    let total_seq = 3usize;
    let q_buf = Buffer::from_slice::<f32>(
        &vec![0.1; 1 * seq * num_heads as usize * head_dim as usize],
        &[1, seq as u64, num_heads, head_dim],
        DType::F32,
    );
    let k_buf = Buffer::from_slice::<f32>(
        &vec![0.1; 1 * num_heads as usize * total_seq * head_dim as usize],
        &[1, num_heads, total_seq as u64, head_dim],
        DType::F32,
    );
    let v_buf = Buffer::from_slice::<f32>(
        &vec![0.1; 1 * num_heads as usize * total_seq * head_dim as usize],
        &[1, num_heads, total_seq as u64, head_dim],
        DType::F32,
    );
    // Causal mask [1, 1, 3, 3]
    let mut mask_data = vec![0.0f32; seq * total_seq];
    for i in 0..seq {
        for j in 0..total_seq {
            if j > i {
                mask_data[i * total_seq + j] = f32::NEG_INFINITY;
            }
        }
    }
    let mask_buf = Buffer::from_slice::<f32>(
        &mask_data,
        &[1, 1, seq as u64, total_seq as u64],
        DType::F32,
    );
    let o_w_buf = Buffer::from_slice::<f32>(
        &vec![0.01; hidden as usize * hidden as usize],
        &[hidden, hidden],
        DType::F32,
    );

    eprintln!(
        "running attention with dynamic shapes: seq={seq}, total_seq={total_seq}, heads={num_heads}, head_dim={head_dim}, hidden={hidden}"
    );
    let outputs = graph.run_dynamic(
        &[&q_buf, &k_buf, &v_buf, &mask_buf, &o_w_buf],
        &[homura::Shape(vec![1, seq as u64, hidden])],
    );

    let out = &outputs[0];
    eprintln!("attention output shape: {:?}", out.shape().0);
    assert_eq!(out.shape().0, vec![1, seq as u64, hidden]);
    for (i, &v) in out.as_slice::<f32>().iter().enumerate() {
        assert!(v.is_finite(), "attention output[{i}] is non-finite: {v}");
    }
    eprintln!("attention kernel dynamic OK");
}

/// Test MLP kernel in isolation.
#[test]
fn test_mlp_kernel_standalone() {
    use homura::graph_builder::{GraphContext, TransformMode};

    let hidden = 8u64;
    let intermediate = 16u64;

    let ctx = GraphContext::new();
    let mut gb = ctx.builder();

    let h = gb.input(&[Some(1), Some(2), Some(hidden)], DType::F32);
    let attn_out = gb.input(&[Some(1), Some(2), Some(hidden)], DType::F32);
    let ln_w = gb.input(&[Some(hidden)], DType::F32);
    let gate_w = gb.input(&[Some(hidden), Some(intermediate)], DType::F32);
    let up_w = gb.input(&[Some(hidden), Some(intermediate)], DType::F32);
    let down_w = gb.input(&[Some(intermediate), Some(hidden)], DType::F32);

    let residual = gb.emit_add(&h, &attn_out);
    let normed = gb.emit_rms_norm(&residual, &ln_w, 1e-6);
    let gate = gb.emit_matmul(&normed, &gate_w);
    let up = gb.emit_matmul(&normed, &up_w);
    let gate_act = gb.emit_silu(&gate);
    let activated = gb.emit_mul(&gate_act, &up);
    let down = gb.emit_matmul(&activated, &down_w);
    let out = gb.emit_add(&residual, &down);

    let graph = gb.compile(&[&out]).unwrap();

    let h_buf = Buffer::from_slice::<f32>(&vec![0.1; 16], &[1, 2, 8], DType::F32);
    let attn_buf = Buffer::from_slice::<f32>(&vec![0.1; 16], &[1, 2, 8], DType::F32);
    let ln_w_buf = Buffer::from_slice::<f32>(&vec![1.0; 8], &[8], DType::F32);
    let gate_w_buf = Buffer::from_slice::<f32>(&vec![0.01; 128], &[8, 16], DType::F32);
    let up_w_buf = Buffer::from_slice::<f32>(&vec![0.01; 128], &[8, 16], DType::F32);
    let down_w_buf = Buffer::from_slice::<f32>(&vec![0.01; 128], &[16, 8], DType::F32);

    let outputs = graph.run(&[
        &h_buf,
        &attn_buf,
        &ln_w_buf,
        &gate_w_buf,
        &up_w_buf,
        &down_w_buf,
    ]);
    let out = &outputs[0];
    eprintln!("MLP output shape: {:?}", out.shape().0);
    assert_eq!(out.shape().0, vec![1, 2, 8]);
    for &v in out.as_slice::<f32>() {
        assert!(v.is_finite(), "MLP output contains non-finite: {v}");
    }
    eprintln!("MLP kernel OK");
}
