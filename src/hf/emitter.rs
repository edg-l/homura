//! Native HF transformer emitter.
//!
//! Constructs per-kernel MLIR modules directly from `TransformerConfig` + weights,
//! bypassing ONNX entirely. Compiles in parallel and assembles an `ExecutionPlan`.

use crate::compiler::{CompileError, KernelEmitResult};
use crate::graph_builder::{GraphContext, TransformMode};
use crate::hf::config::TransformerConfig;
use crate::hf::weights::TransformerWeights;
use crate::runtime::{ExecutionPlan, KernelStep, KvPlanInfo, SlotDesc};
use crate::shape::SymDim;
use crate::{DType, Shape};

// ── Slot layout ──────────────────────────────────────────────────────────────

/// Deterministic buffer slot assignment for a decoder-only transformer.
struct SlotLayout {
    /// Total number of slots.
    num_slots: usize,
    /// Model runtime inputs: [input_ids, mask, cos, sin].
    input_slots: Vec<usize>,
    /// All weight buffer slots (in to_slot_buffers order).
    weight_slots: Vec<usize>,
    /// Model output: [logits].
    output_slots: Vec<usize>,
    /// Slot descriptors for every slot.
    slot_descs: Vec<SlotDesc>,
    /// Per-layer slot info needed to wire up kernels.
    layers: Vec<LayerSlots>,
    /// Embedding output slot.
    embed_out_slot: usize,
    /// LM head weight slot (may alias embed weight for tied embeddings).
    lm_head_weight_slot: usize,
    /// Final norm weight slot.
    final_norm_weight_slot: usize,
    /// Logits output slot.
    logits_slot: usize,
    /// KV plan info for run_kv.
    past_kv_input_slots: Vec<usize>,
    present_kv_output_slots: Vec<usize>,
}

struct LayerSlots {
    // Inputs from previous layer
    hidden_in_slot: usize,
    // Weights (indexes into the flat weight buffer order)
    input_ln_w_slot: usize,
    q_w_slot: usize,
    q_b_slot: Option<usize>,
    k_w_slot: usize,
    k_b_slot: Option<usize>,
    v_w_slot: usize,
    v_b_slot: Option<usize>,
    o_w_slot: usize,
    post_attn_ln_w_slot: usize,
    gate_w_slot: usize,
    up_w_slot: usize,
    down_w_slot: usize,
    // Intermediates
    q_slot: usize,
    k_slot: usize,
    v_slot: usize,
    past_k_slot: usize,
    past_v_slot: usize,
    present_k_slot: usize,
    present_v_slot: usize,
    attn_out_slot: usize,
    // Output
    hidden_out_slot: usize,
}

fn assign_transformer_slots(config: &TransformerConfig, has_bias: bool) -> SlotLayout {
    let mut next = 0usize;
    let mut alloc = |desc: SlotDesc| -> (usize, SlotDesc) {
        let slot = next;
        next += 1;
        (slot, desc)
    };

    let seq = || SymDim::Var("seq".into());
    let past_seq = || SymDim::Var("past_sequence_length".into());
    let total_seq = || past_seq().add(seq());
    let hidden = config.hidden_size as u64;
    let kv_dim = (config.kv_heads() * config.head_dim()) as u64;
    let intermediate = config.intermediate_size as u64;
    let vocab = config.vocab_size as u64;
    let num_heads = config.num_attention_heads as u64;
    let kv_heads = config.kv_heads() as u64;
    let head_dim = config.head_dim() as u64;

    let dyn_val = crate::shape::DIM_DYNAMIC;

    // --- Model inputs ---
    let (input_ids_slot, input_ids_desc) = alloc(SlotDesc {
        shape: Shape(vec![1, dyn_val]),
        dtype: DType::I64,
        sym_shape: Some(vec![SymDim::Concrete(1), seq()]),
    });
    let (mask_slot, mask_desc) = alloc(SlotDesc {
        shape: Shape(vec![1, 1, dyn_val, dyn_val]),
        dtype: DType::F32,
        sym_shape: Some(vec![
            SymDim::Concrete(1),
            SymDim::Concrete(1),
            seq(),
            total_seq(),
        ]),
    });
    let (cos_slot, cos_desc) = alloc(SlotDesc {
        shape: Shape(vec![dyn_val, head_dim]),
        dtype: DType::F32,
        sym_shape: Some(vec![seq(), SymDim::Concrete(head_dim)]),
    });
    let (sin_slot, sin_desc) = alloc(SlotDesc {
        shape: Shape(vec![dyn_val, head_dim]),
        dtype: DType::F32,
        sym_shape: Some(vec![seq(), SymDim::Concrete(head_dim)]),
    });

    let mut input_slots = vec![input_ids_slot, mask_slot, cos_slot, sin_slot];
    // past_kv slots will be appended per-layer below.

    // --- Weight slots ---
    // Must match TransformerWeights::to_slot_buffers() order exactly.
    let mut weight_slots = Vec::new();
    let mut slot_descs_vec: Vec<SlotDesc> = vec![input_ids_desc, mask_desc, cos_desc, sin_desc];

    // embed_tokens_weight [vocab, hidden]
    let (embed_w_slot, embed_w_desc) = alloc(SlotDesc {
        shape: Shape(vec![vocab, hidden]),
        dtype: DType::F32,
        sym_shape: None,
    });
    weight_slots.push(embed_w_slot);
    slot_descs_vec.push(embed_w_desc);

    // Per-layer weights + intermediates
    let mut layers: Vec<LayerSlots> = Vec::with_capacity(config.num_hidden_layers);
    let mut past_kv_input_slots = Vec::new();
    let mut present_kv_output_slots = Vec::new();

    // Embedding output
    let (embed_out_slot, embed_out_desc) = alloc(SlotDesc {
        shape: Shape(vec![1, dyn_val, hidden]),
        dtype: DType::F32,
        sym_shape: Some(vec![SymDim::Concrete(1), seq(), SymDim::Concrete(hidden)]),
    });
    slot_descs_vec.push(embed_out_desc);

    for _i in 0..config.num_hidden_layers {
        let hidden_in_slot = if _i == 0 {
            embed_out_slot
        } else {
            layers[_i - 1].hidden_out_slot
        };

        // input_layernorm_weight [hidden]
        let (input_ln_w_slot, d) = alloc(SlotDesc {
            shape: Shape(vec![hidden]),
            dtype: DType::F32,
            sym_shape: None,
        });
        weight_slots.push(input_ln_w_slot);
        slot_descs_vec.push(d);

        // q_proj_weight [hidden, hidden] (transposed)
        let (q_w_slot, d) = alloc(SlotDesc {
            shape: Shape(vec![hidden, hidden]),
            dtype: DType::F32,
            sym_shape: None,
        });
        weight_slots.push(q_w_slot);
        slot_descs_vec.push(d);

        let q_b_slot = if has_bias {
            let (s, d) = alloc(SlotDesc {
                shape: Shape(vec![hidden]),
                dtype: DType::F32,
                sym_shape: None,
            });
            weight_slots.push(s);
            slot_descs_vec.push(d);
            Some(s)
        } else {
            None
        };

        // k_proj_weight [hidden, kv_dim] (transposed)
        let (k_w_slot, d) = alloc(SlotDesc {
            shape: Shape(vec![hidden, kv_dim]),
            dtype: DType::F32,
            sym_shape: None,
        });
        weight_slots.push(k_w_slot);
        slot_descs_vec.push(d);

        let k_b_slot = if has_bias {
            let (s, d) = alloc(SlotDesc {
                shape: Shape(vec![kv_dim]),
                dtype: DType::F32,
                sym_shape: None,
            });
            weight_slots.push(s);
            slot_descs_vec.push(d);
            Some(s)
        } else {
            None
        };

        // v_proj_weight [hidden, kv_dim] (transposed)
        let (v_w_slot, d) = alloc(SlotDesc {
            shape: Shape(vec![hidden, kv_dim]),
            dtype: DType::F32,
            sym_shape: None,
        });
        weight_slots.push(v_w_slot);
        slot_descs_vec.push(d);

        let v_b_slot = if has_bias {
            let (s, d) = alloc(SlotDesc {
                shape: Shape(vec![kv_dim]),
                dtype: DType::F32,
                sym_shape: None,
            });
            weight_slots.push(s);
            slot_descs_vec.push(d);
            Some(s)
        } else {
            None
        };

        // o_proj_weight [hidden, hidden] (transposed)
        let (o_w_slot, d) = alloc(SlotDesc {
            shape: Shape(vec![hidden, hidden]),
            dtype: DType::F32,
            sym_shape: None,
        });
        weight_slots.push(o_w_slot);
        slot_descs_vec.push(d);

        // post_attention_layernorm_weight [hidden]
        let (post_attn_ln_w_slot, d) = alloc(SlotDesc {
            shape: Shape(vec![hidden]),
            dtype: DType::F32,
            sym_shape: None,
        });
        weight_slots.push(post_attn_ln_w_slot);
        slot_descs_vec.push(d);

        // gate_proj_weight [hidden, intermediate] (transposed)
        let (gate_w_slot, d) = alloc(SlotDesc {
            shape: Shape(vec![hidden, intermediate]),
            dtype: DType::F32,
            sym_shape: None,
        });
        weight_slots.push(gate_w_slot);
        slot_descs_vec.push(d);

        // up_proj_weight [hidden, intermediate] (transposed)
        let (up_w_slot, d) = alloc(SlotDesc {
            shape: Shape(vec![hidden, intermediate]),
            dtype: DType::F32,
            sym_shape: None,
        });
        weight_slots.push(up_w_slot);
        slot_descs_vec.push(d);

        // down_proj_weight [intermediate, hidden] (transposed)
        let (down_w_slot, d) = alloc(SlotDesc {
            shape: Shape(vec![intermediate, hidden]),
            dtype: DType::F32,
            sym_shape: None,
        });
        weight_slots.push(down_w_slot);
        slot_descs_vec.push(d);

        // --- Intermediate slots ---
        // Q: [1, seq, num_heads, head_dim] (BSHD — transposed to BHSD inside attn kernel)
        let (q_slot, d) = alloc(SlotDesc {
            shape: Shape(vec![1, dyn_val, num_heads, head_dim]),
            dtype: DType::F32,
            sym_shape: Some(vec![
                SymDim::Concrete(1),
                seq(),
                SymDim::Concrete(num_heads),
                SymDim::Concrete(head_dim),
            ]),
        });
        slot_descs_vec.push(d);

        // K: [1, kv_heads, seq, head_dim] (BHSD — QKV kernel transposes before output)
        let (k_slot, d) = alloc(SlotDesc {
            shape: Shape(vec![1, kv_heads, dyn_val, head_dim]),
            dtype: DType::F32,
            sym_shape: Some(vec![
                SymDim::Concrete(1),
                SymDim::Concrete(kv_heads),
                seq(),
                SymDim::Concrete(head_dim),
            ]),
        });
        slot_descs_vec.push(d);

        // V: [1, kv_heads, seq, head_dim] (BHSD)
        let (v_slot, d) = alloc(SlotDesc {
            shape: Shape(vec![1, kv_heads, dyn_val, head_dim]),
            dtype: DType::F32,
            sym_shape: Some(vec![
                SymDim::Concrete(1),
                SymDim::Concrete(kv_heads),
                seq(),
                SymDim::Concrete(head_dim),
            ]),
        });
        slot_descs_vec.push(d);

        // past_k, past_v: [1, kv_heads, past_seq, head_dim] (transposed for BHSD layout)
        let (past_k_slot, d) = alloc(SlotDesc {
            shape: Shape(vec![1, kv_heads, dyn_val, head_dim]),
            dtype: DType::F32,
            sym_shape: Some(vec![
                SymDim::Concrete(1),
                SymDim::Concrete(kv_heads),
                SymDim::Var("past_sequence_length".into()),
                SymDim::Concrete(head_dim),
            ]),
        });
        slot_descs_vec.push(d);
        past_kv_input_slots.push(past_k_slot);
        input_slots.push(past_k_slot);

        let (past_v_slot, d) = alloc(SlotDesc {
            shape: Shape(vec![1, kv_heads, dyn_val, head_dim]),
            dtype: DType::F32,
            sym_shape: Some(vec![
                SymDim::Concrete(1),
                SymDim::Concrete(kv_heads),
                SymDim::Var("past_sequence_length".into()),
                SymDim::Concrete(head_dim),
            ]),
        });
        slot_descs_vec.push(d);
        past_kv_input_slots.push(past_v_slot);
        input_slots.push(past_v_slot);

        // present_k, present_v: [1, kv_heads, total_seq, head_dim]
        let (present_k_slot, d) = alloc(SlotDesc {
            shape: Shape(vec![1, kv_heads, dyn_val, head_dim]),
            dtype: DType::F32,
            sym_shape: Some(vec![
                SymDim::Concrete(1),
                SymDim::Concrete(kv_heads),
                total_seq(),
                SymDim::Concrete(head_dim),
            ]),
        });
        slot_descs_vec.push(d);
        present_kv_output_slots.push(present_k_slot);

        let (present_v_slot, d) = alloc(SlotDesc {
            shape: Shape(vec![1, kv_heads, dyn_val, head_dim]),
            dtype: DType::F32,
            sym_shape: Some(vec![
                SymDim::Concrete(1),
                SymDim::Concrete(kv_heads),
                total_seq(),
                SymDim::Concrete(head_dim),
            ]),
        });
        slot_descs_vec.push(d);
        present_kv_output_slots.push(present_v_slot);

        // attn_out: [1, seq, hidden]
        let (attn_out_slot, d) = alloc(SlotDesc {
            shape: Shape(vec![1, dyn_val, hidden]),
            dtype: DType::F32,
            sym_shape: Some(vec![SymDim::Concrete(1), seq(), SymDim::Concrete(hidden)]),
        });
        slot_descs_vec.push(d);

        // hidden_out: [1, seq, hidden]
        let (hidden_out_slot, d) = alloc(SlotDesc {
            shape: Shape(vec![1, dyn_val, hidden]),
            dtype: DType::F32,
            sym_shape: Some(vec![SymDim::Concrete(1), seq(), SymDim::Concrete(hidden)]),
        });
        slot_descs_vec.push(d);

        layers.push(LayerSlots {
            hidden_in_slot,
            input_ln_w_slot,
            q_w_slot,
            q_b_slot,
            k_w_slot,
            k_b_slot,
            v_w_slot,
            v_b_slot,
            o_w_slot,
            post_attn_ln_w_slot,
            gate_w_slot,
            up_w_slot,
            down_w_slot,
            q_slot,
            k_slot,
            v_slot,
            past_k_slot,
            past_v_slot,
            present_k_slot,
            present_v_slot,
            attn_out_slot,
            hidden_out_slot,
        });
    }

    // final_norm_weight [hidden]
    let (final_norm_w_slot, d) = alloc(SlotDesc {
        shape: Shape(vec![hidden]),
        dtype: DType::F32,
        sym_shape: None,
    });
    weight_slots.push(final_norm_w_slot);
    slot_descs_vec.push(d);

    // lm_head_weight: always [hidden, vocab] (transposed at load time).
    // For tied embeddings, the caller provides a pre-transposed copy of embed_tokens.
    let (lm_head_weight_slot, d) = alloc(SlotDesc {
        shape: Shape(vec![hidden, vocab]),
        dtype: DType::F32,
        sym_shape: None,
    });
    weight_slots.push(lm_head_weight_slot);
    slot_descs_vec.push(d);

    // logits output: [1, seq, vocab]
    let (logits_slot, d) = alloc(SlotDesc {
        shape: Shape(vec![1, dyn_val, vocab]),
        dtype: DType::F32,
        sym_shape: Some(vec![SymDim::Concrete(1), seq(), SymDim::Concrete(vocab)]),
    });
    slot_descs_vec.push(d);

    // Output slots: logits + all present_kv (needed for prefill -> init_kv_cache)
    let mut output_slots = vec![logits_slot];
    output_slots.extend_from_slice(&present_kv_output_slots);

    // Pad slot_descs to cover all slots
    assert_eq!(slot_descs_vec.len(), next);

    SlotLayout {
        num_slots: next,
        input_slots,
        weight_slots,
        output_slots,
        slot_descs: slot_descs_vec,
        layers,
        embed_out_slot,
        lm_head_weight_slot,
        final_norm_weight_slot: final_norm_w_slot,
        logits_slot,
        past_kv_input_slots,
        present_kv_output_slots,
    }
}

// ── Kernel emission ──────────────────────────────────────────────────────────

/// Emit embedding lookup kernel.
///
/// Inputs:  input_ids [1, seq] I64, embed_weight [vocab, hidden] F32
/// Outputs: hidden_states [1, seq, hidden] F32
fn emit_embed_kernel(
    config: &TransformerConfig,
    kernel_idx: usize,
) -> Result<KernelEmitResult, CompileError> {
    let ctx = GraphContext::new();
    let mut gb = ctx.builder();

    let input_ids = gb.input(&[Some(1), None], DType::I64);
    let embed_w = gb.input(
        &[
            Some(config.vocab_size as u64),
            Some(config.hidden_size as u64),
        ],
        DType::F32,
    );

    let hidden = gb.emit_embedding(&embed_w, &input_ids);

    let func_name = format!("k{kernel_idx}");
    let (mlir_text, num_inputs, output_descs) =
        gb.finalize_to_mlir_named(&[&hidden], TransformMode::None, &func_name)?;

    Ok(KernelEmitResult {
        mlir_text,
        num_inputs,
        output_descs,
        group_idx: kernel_idx,
        num_in: 2,
        num_out: 1,
        ops_label: "Embed".into(),
    })
}

/// Emit QKV projection + RoPE kernel.
///
/// Inputs:  hidden [1, seq, hidden], input_ln_w [hidden],
///          q_w [hidden, hidden], [q_b [hidden]],
///          k_w [hidden, kv_dim], [k_b [kv_dim]],
///          v_w [hidden, kv_dim], [v_b [kv_dim]]
///          cos [seq, head_dim], sin [seq, head_dim]
/// Outputs: q [1, seq, num_heads, head_dim],
///          k [1, seq, kv_heads, head_dim],
///          v [1, seq, kv_heads, head_dim]
fn emit_qkv_kernel(
    config: &TransformerConfig,
    has_bias: bool,
    kernel_idx: usize,
) -> Result<KernelEmitResult, CompileError> {
    let hidden = config.hidden_size as u64;
    let kv_dim = (config.kv_heads() * config.head_dim()) as u64;
    let q_dim = (config.num_attention_heads * config.head_dim()) as u64;
    let num_heads = config.num_attention_heads as u64;
    let kv_heads = config.kv_heads() as u64;
    let head_dim = config.head_dim() as u64;

    let ctx = GraphContext::new();
    let mut gb = ctx.builder();

    // Inputs
    let h = gb.input(&[Some(1), None, Some(hidden)], DType::F32);
    let ln_w = gb.input(&[Some(hidden)], DType::F32);
    let q_w = gb.input(&[Some(hidden), Some(q_dim)], DType::F32);
    let q_b = if has_bias {
        Some(gb.input(&[Some(q_dim)], DType::F32))
    } else {
        None
    };
    let k_w = gb.input(&[Some(hidden), Some(kv_dim)], DType::F32);
    let k_b = if has_bias {
        Some(gb.input(&[Some(kv_dim)], DType::F32))
    } else {
        None
    };
    let v_w = gb.input(&[Some(hidden), Some(kv_dim)], DType::F32);
    let v_b = if has_bias {
        Some(gb.input(&[Some(kv_dim)], DType::F32))
    } else {
        None
    };
    let cos = gb.input(&[None, Some(head_dim)], DType::F32);
    let sin = gb.input(&[None, Some(head_dim)], DType::F32);

    // RMSNorm
    let normed = gb.emit_rms_norm(&h, &ln_w, config.rms_norm_eps as f32);

    // Q projection: [1, seq, hidden] @ [hidden, hidden] -> [1, seq, hidden]
    let mut q = gb.emit_matmul(&normed, &q_w);
    if let Some(bias) = &q_b {
        q = gb.emit_add(&q, bias);
    }

    // K projection: [1, seq, hidden] @ [hidden, kv_dim] -> [1, seq, kv_dim]
    let mut k = gb.emit_matmul(&normed, &k_w);
    if let Some(bias) = &k_b {
        k = gb.emit_add(&k, bias);
    }

    // V projection: [1, seq, hidden] @ [hidden, kv_dim] -> [1, seq, kv_dim]
    let mut v = gb.emit_matmul(&normed, &v_w);
    if let Some(bias) = &v_b {
        v = gb.emit_add(&v, bias);
    }

    // Reshape to head layout
    // Q: [1, seq, hidden] -> [1, seq, num_heads, head_dim]
    let q_4d = gb.emit_reshape(&q, &[1, -1, num_heads as i64, head_dim as i64]);
    // K: [1, seq, kv_dim] -> [1, seq, kv_heads, head_dim]
    let k_4d = gb.emit_reshape(&k, &[1, -1, kv_heads as i64, head_dim as i64]);
    // V: [1, seq, kv_dim] -> [1, seq, kv_heads, head_dim]
    let v_4d = gb.emit_reshape(&v, &[1, -1, kv_heads as i64, head_dim as i64]);

    // RoPE on Q and K (before KV concat so cached K is already rotated)
    let q_rope = gb.emit_rope_half(&q_4d, &cos, &sin);
    let k_rope = gb.emit_rope_half(&k_4d, &cos, &sin);

    // Transpose K and V from BSHD [1, seq, kv_heads, head_dim]
    // to BHSD [1, kv_heads, seq, head_dim] for KV cache concat on axis=2.
    let k_bhsd = gb.emit_transpose(&k_rope, &[0, 2, 1, 3]);
    let v_bhsd = gb.emit_transpose(&v_4d, &[0, 2, 1, 3]);

    let func_name = format!("k{kernel_idx}");
    let num_in = if has_bias { 11 } else { 8 }; // h, ln_w, q_w, [q_b], k_w, [k_b], v_w, [v_b], cos, sin
    let (mlir_text, num_inputs, output_descs) = gb.finalize_to_mlir_named(
        &[&q_rope, &k_bhsd, &v_bhsd],
        TransformMode::VectorizeOnly,
        &func_name,
    )?;

    Ok(KernelEmitResult {
        mlir_text,
        num_inputs,
        output_descs,
        group_idx: kernel_idx,
        num_in,
        num_out: 3,
        ops_label: "QKV+RoPE".into(),
    })
}

/// Emit attention kernel: scaled dot-product attention + O projection.
///
/// Inputs:  q [1, seq, num_heads, head_dim],
///          k [1, total_seq, kv_heads, head_dim],  (from KV cache)
///          v [1, total_seq, kv_heads, head_dim],
///          mask [1, 1, seq, total_seq],
///          o_w [hidden, hidden]
/// Outputs: attn_out [1, seq, hidden]
fn emit_attention_kernel(
    config: &TransformerConfig,
    kernel_idx: usize,
) -> Result<KernelEmitResult, CompileError> {
    let hidden = config.hidden_size as u64;
    let num_heads = config.num_attention_heads as u64;
    let kv_heads = config.kv_heads() as u64;
    let head_dim = config.head_dim() as u64;
    let q_dim = num_heads * head_dim; // may differ from hidden when head_dim is explicit
    let gqa_repeat = config.gqa_repeat();

    let ctx = GraphContext::new();
    let mut gb = ctx.builder();

    // Q: [1, seq, num_heads, head_dim]
    let q = gb.input(
        &[Some(1), None, Some(num_heads), Some(head_dim)],
        DType::F32,
    );
    // K from KV cache: [1, kv_heads, total_seq, head_dim] (BHSD layout)
    let k = gb.input(&[Some(1), Some(kv_heads), None, Some(head_dim)], DType::F32);
    // V from KV cache: [1, kv_heads, total_seq, head_dim]
    let v = gb.input(&[Some(1), Some(kv_heads), None, Some(head_dim)], DType::F32);
    // Mask: [1, 1, seq, total_seq]
    let mask = gb.input(&[Some(1), Some(1), None, None], DType::F32);
    // O projection weight: [q_dim, hidden]
    let o_w = gb.input(&[Some(q_dim), Some(hidden)], DType::F32);

    // Transpose Q from BSHD to BHSD: [1, seq, num_heads, head_dim] -> [1, num_heads, seq, head_dim]
    let q_bhsd = gb.emit_transpose(&q, &[0, 2, 1, 3]);

    // GQA: expand K and V heads if needed
    // K is [1, kv_heads, total_seq, head_dim], need [1, num_heads, total_seq, head_dim]
    // Transpose K to BSHD first for repeat_kv, then back to BHSD
    let k_bshd = gb.emit_transpose(&k, &[0, 2, 1, 3]); // [1, total_seq, kv_heads, head_dim]
    let k_expanded = gb.emit_repeat_kv(&k_bshd, gqa_repeat); // [1, total_seq, num_heads, head_dim]
    let k_bhsd = gb.emit_transpose(&k_expanded, &[0, 2, 1, 3]); // [1, num_heads, total_seq, head_dim]

    let v_bshd = gb.emit_transpose(&v, &[0, 2, 1, 3]);
    let v_expanded = gb.emit_repeat_kv(&v_bshd, gqa_repeat);
    let v_bhsd = gb.emit_transpose(&v_expanded, &[0, 2, 1, 3]);

    // QK^T: [1, num_heads, seq, head_dim] @ [1, num_heads, head_dim, total_seq]
    //     = [1, num_heads, seq, total_seq]
    let k_t = gb.emit_transpose(&k_bhsd, &[0, 1, 3, 2]);
    let scores = gb.emit_matmul(&q_bhsd, &k_t);

    // Scale by 1/sqrt(head_dim)
    let scale = gb.emit_arith_constant(1.0 / (head_dim as f64).sqrt(), DType::F32);
    let scaled = gb.emit_mul(&scores, &scale);

    // Apply causal mask (additive: 0 for attend, -inf for masked)
    let masked = gb.emit_add(&scaled, &mask);

    // Softmax along last axis
    let attn_weights = gb.emit_softmax(&masked, -1);

    // AV: [1, num_heads, seq, total_seq] @ [1, num_heads, total_seq, head_dim]
    //   = [1, num_heads, seq, head_dim]
    let attn_output = gb.emit_matmul(&attn_weights, &v_bhsd);

    // Transpose back: [1, num_heads, seq, head_dim] -> [1, seq, num_heads, head_dim]
    let attn_bshd = gb.emit_transpose(&attn_output, &[0, 2, 1, 3]);

    // Reshape to [1, seq, q_dim]
    let attn_flat = gb.emit_reshape(&attn_bshd, &[1, -1, q_dim as i64]);

    // O projection: [1, seq, q_dim] @ [q_dim, hidden] -> [1, seq, hidden]
    let out = gb.emit_matmul(&attn_flat, &o_w);

    let func_name = format!("k{kernel_idx}");
    let (mlir_text, num_inputs, output_descs) =
        gb.finalize_to_mlir_named(&[&out], TransformMode::VectorizeOnly, &func_name)?;

    Ok(KernelEmitResult {
        mlir_text,
        num_inputs,
        output_descs,
        group_idx: kernel_idx,
        num_in: 5,
        num_out: 1,
        ops_label: "Attn".into(),
    })
}

/// Emit residual + MLP kernel.
///
/// Inputs:  hidden [1, seq, hidden], attn_out [1, seq, hidden],
///          post_attn_ln_w [hidden],
///          gate_w [hidden, intermediate], up_w [hidden, intermediate],
///          down_w [intermediate, hidden]
/// Outputs: next_hidden [1, seq, hidden]
fn emit_mlp_kernel(
    config: &TransformerConfig,
    kernel_idx: usize,
) -> Result<KernelEmitResult, CompileError> {
    let hidden = config.hidden_size as u64;
    let intermediate = config.intermediate_size as u64;

    let ctx = GraphContext::new();
    let mut gb = ctx.builder();

    let h = gb.input(&[Some(1), None, Some(hidden)], DType::F32);
    let attn_out = gb.input(&[Some(1), None, Some(hidden)], DType::F32);
    let ln_w = gb.input(&[Some(hidden)], DType::F32);
    let gate_w = gb.input(&[Some(hidden), Some(intermediate)], DType::F32);
    let up_w = gb.input(&[Some(hidden), Some(intermediate)], DType::F32);
    let down_w = gb.input(&[Some(intermediate), Some(hidden)], DType::F32);

    // First residual: hidden + attn_out
    let residual = gb.emit_add(&h, &attn_out);

    // RMSNorm
    let normed = gb.emit_rms_norm(&residual, &ln_w, config.rms_norm_eps as f32);

    // SwiGLU MLP
    // gate = normed @ gate_w  -> [1, seq, intermediate]
    let gate = gb.emit_matmul(&normed, &gate_w);
    // up = normed @ up_w  -> [1, seq, intermediate]
    let up = gb.emit_matmul(&normed, &up_w);
    // activated = silu(gate) * up
    let gate_act = gb.emit_silu(&gate);
    let activated = gb.emit_mul(&gate_act, &up);
    // down = activated @ down_w  -> [1, seq, hidden]
    let down = gb.emit_matmul(&activated, &down_w);

    // Second residual: residual + down
    let out = gb.emit_add(&residual, &down);

    let func_name = format!("k{kernel_idx}");
    // Smallest matmul N in MLP: min(intermediate, hidden) for down projection.
    let min_n = hidden.min(intermediate) as usize;
    let mode = TransformMode::tile_parallel_adaptive(min_n);
    let (mlir_text, num_inputs, output_descs) =
        gb.finalize_to_mlir_named(&[&out], mode, &func_name)?;

    Ok(KernelEmitResult {
        mlir_text,
        num_inputs,
        output_descs,
        group_idx: kernel_idx,
        num_in: 6,
        num_out: 1,
        ops_label: "MLP".into(),
    })
}

/// Emit LM head kernel: final RMSNorm + logit projection.
///
/// Inputs:  hidden [1, seq, hidden], norm_w [hidden], lm_head_w [hidden, vocab]
/// Outputs: logits [1, seq, vocab]
fn emit_lm_head_kernel(
    config: &TransformerConfig,
    kernel_idx: usize,
) -> Result<KernelEmitResult, CompileError> {
    let hidden = config.hidden_size as u64;
    let vocab = config.vocab_size as u64;

    let ctx = GraphContext::new();
    let mut gb = ctx.builder();

    let h = gb.input(&[Some(1), None, Some(hidden)], DType::F32);
    let norm_w = gb.input(&[Some(hidden)], DType::F32);
    // lm_head_w is always [hidden, vocab] -- transposed at load time
    // (for tied embeddings, the caller pre-transposes embed_tokens)
    let lm_head_w = gb.input(&[Some(hidden), Some(vocab)], DType::F32);

    let normed = gb.emit_rms_norm(&h, &norm_w, config.rms_norm_eps as f32);
    let logits = gb.emit_matmul(&normed, &lm_head_w);

    let func_name = format!("k{kernel_idx}");
    let mode = TransformMode::tile_parallel_adaptive(vocab as usize);
    let (mlir_text, num_inputs, output_descs) =
        gb.finalize_to_mlir_named(&[&logits], mode, &func_name)?;

    Ok(KernelEmitResult {
        mlir_text,
        num_inputs,
        output_descs,
        group_idx: kernel_idx,
        num_in: 3,
        num_out: 1,
        ops_label: "LMHead".into(),
    })
}

// ── Top-level plan emission ──────────────────────────────────────────────────

/// Emit and compile all transformer kernels, producing an ExecutionPlan.
pub fn emit_transformer_plan(
    config: &TransformerConfig,
    weights: &TransformerWeights,
) -> Result<(ExecutionPlan, Vec<crate::runtime::Buffer>), CompileError> {
    let has_bias = weights
        .layers
        .first()
        .map(|l| l.q_proj_bias.is_some())
        .unwrap_or(false);
    let layout = assign_transformer_slots(config, has_bias);

    log_compile!(
        "hf",
        "emitting {} layers, {} total slots",
        config.num_hidden_layers,
        layout.num_slots
    );

    let mut emit_results: Vec<KernelEmitResult> = Vec::new();
    let mut steps: Vec<KernelStep> = Vec::new();
    let mut kernel_idx = 0usize;

    // --- Embedding kernel ---
    emit_results.push(emit_embed_kernel(config, kernel_idx)?);
    steps.push(KernelStep::kernel(
        kernel_idx,
        vec![layout.input_slots[0], layout.weight_slots[0]], // input_ids, embed_w
        vec![layout.embed_out_slot],
    ));
    kernel_idx += 1;

    // --- Per-layer kernels ---
    for i in 0..config.num_hidden_layers {
        let ls = &layout.layers[i];

        // QKV + RoPE kernel
        emit_results.push(emit_qkv_kernel(config, has_bias, kernel_idx)?);
        let mut qkv_inputs = vec![ls.hidden_in_slot, ls.input_ln_w_slot, ls.q_w_slot];
        if let Some(s) = ls.q_b_slot {
            qkv_inputs.push(s);
        }
        qkv_inputs.push(ls.k_w_slot);
        if let Some(s) = ls.k_b_slot {
            qkv_inputs.push(s);
        }
        qkv_inputs.push(ls.v_w_slot);
        if let Some(s) = ls.v_b_slot {
            qkv_inputs.push(s);
        }
        qkv_inputs.push(layout.input_slots[2]); // cos
        qkv_inputs.push(layout.input_slots[3]); // sin
        steps.push(KernelStep::kernel(
            kernel_idx,
            qkv_inputs,
            vec![ls.q_slot, ls.k_slot, ls.v_slot],
        ));
        kernel_idx += 1;

        // KV Concat native ops (K and V are in BHSD layout from QKV kernel)
        // KV concat K: [past_k (BHSD), new_k (BHSD)] -> present_k (BHSD) on axis=2
        steps.push(KernelStep {
            kernel_idx: usize::MAX,
            input_slots: vec![ls.past_k_slot, ls.k_slot],
            output_slots: vec![ls.present_k_slot],
            native_op: Some(crate::runtime::NativeOp::Concat { axis: 2 }),
        });

        // KV concat V
        steps.push(KernelStep {
            kernel_idx: usize::MAX,
            input_slots: vec![ls.past_v_slot, ls.v_slot],
            output_slots: vec![ls.present_v_slot],
            native_op: Some(crate::runtime::NativeOp::Concat { axis: 2 }),
        });

        // Attention kernel
        emit_results.push(emit_attention_kernel(config, kernel_idx)?);
        steps.push(KernelStep::kernel(
            kernel_idx,
            vec![
                ls.q_slot,
                ls.present_k_slot,
                ls.present_v_slot,
                layout.input_slots[1], // mask
                ls.o_w_slot,
            ],
            vec![ls.attn_out_slot],
        ));
        kernel_idx += 1;

        // MLP kernel
        emit_results.push(emit_mlp_kernel(config, kernel_idx)?);
        steps.push(KernelStep::kernel(
            kernel_idx,
            vec![
                ls.hidden_in_slot,
                ls.attn_out_slot,
                ls.post_attn_ln_w_slot,
                ls.gate_w_slot,
                ls.up_w_slot,
                ls.down_w_slot,
            ],
            vec![ls.hidden_out_slot],
        ));
        kernel_idx += 1;
    }

    // --- LM head kernel ---
    let last_hidden = layout.layers.last().unwrap().hidden_out_slot;
    emit_results.push(emit_lm_head_kernel(config, kernel_idx)?);
    steps.push(KernelStep::kernel(
        kernel_idx,
        vec![
            last_hidden,
            layout.final_norm_weight_slot,
            layout.lm_head_weight_slot,
        ],
        vec![layout.logits_slot],
    ));

    log_compile!(
        "hf",
        "{} compiled kernels + {} native concat steps",
        emit_results.len(),
        steps.iter().filter(|s| s.native_op.is_some()).count()
    );

    // Compile all kernels in parallel
    let (kernels, lib) = crate::compiler::compile_and_link_kernels(&emit_results)?;

    let mut plan = ExecutionPlan::new(
        kernels,
        steps,
        layout.num_slots,
        layout.input_slots,
        layout.weight_slots,
        layout.output_slots,
        layout.slot_descs,
    );
    plan.set_shared_lib(lib);

    // Attach KV cache info
    let kv_info = KvPlanInfo {
        num_layers: config.num_hidden_layers,
        num_heads: config.kv_heads(),
        head_dim: config.head_dim(),
        past_kv_input_slots: layout.past_kv_input_slots,
        present_kv_output_slots: layout.present_kv_output_slots,
    };
    plan.set_kv_info(kv_info);

    let weight_bufs = weights.to_slot_buffers();

    Ok((plan, weight_bufs))
}
