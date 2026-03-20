//! Quantized (Q8_0) transformer plan emitter.
//!
//! Emits an `ExecutionPlan` where each projection (Q/K/V/O/gate/up/down) is a
//! separate dequant-matmul kernel, and all lightweight ops (RMSNorm, residuals,
//! RoPE, SiLU, etc.) are grouped into tensor kernels with `TransformMode::None`.

use crate::compiler::{CompileError, KernelEmitResult};
use crate::graph_builder::{GraphContext, TransformMode};
use crate::hf::config::TransformerConfig;
use crate::hf::weights::TransformerWeights;
use crate::runtime::{ExecutionPlan, KernelStep, KvPlanInfo, SlotDesc};
use crate::shape::SymDim;
use crate::{DType, Shape};

// ── n_tile selection ─────────────────────────────────────────────────────────

/// Choose an n_tile for `emit_dequant_matmul_q8_0` that:
/// - gives roughly `num_cores * 1.5` tiles for good load balancing,
/// - is a divisor of `n`, and
/// - is clamped to [1, 256].
fn pick_n_tile(n: u64) -> usize {
    let num_cores = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(8);
    let target_tiles = num_cores + num_cores / 2;
    let ideal = (n as usize) / target_tiles;
    let mut tile = ideal.max(1).min(256);
    // Round down to a divisor of n.
    while tile > 1 && (n as usize) % tile != 0 {
        tile -= 1;
    }
    tile.max(1)
}

// ── Slot layout ───────────────────────────────────────────────────────────────

/// Buffer slot assignment for the quantized transformer execution plan.
struct QuantSlotLayout {
    num_slots: usize,
    /// Model runtime inputs: [input_ids, mask, cos, sin, past_k0, past_v0, ...].
    input_slots: Vec<usize>,
    /// All weight buffer slots (in to_slot_buffers_quant order).
    weight_slots: Vec<usize>,
    /// Model output slots: [logits, present_k0, present_v0, ...].
    output_slots: Vec<usize>,
    /// Shape+dtype for every slot.
    slot_descs: Vec<SlotDesc>,
    /// Per-layer slot indices needed to wire kernels.
    layers: Vec<QuantLayerSlots>,
    /// Embedding output slot.
    embed_out_slot: usize,
    /// Final norm weight slot.
    final_norm_weight_slot: usize,
    /// LM head weight slot.
    lm_head_weight_slot: usize,
    /// Logits output slot.
    logits_slot: usize,
    /// KV past-input slots (one pair per layer), in layer order.
    past_kv_input_slots: Vec<usize>,
    /// KV present-output slots (one pair per layer), in layer order.
    present_kv_output_slots: Vec<usize>,
}

/// Per-layer slot indices for the quantized plan.
struct QuantLayerSlots {
    hidden_in_slot: usize,
    // Weights
    input_ln_w_slot: usize,
    q_w_slot: usize,
    q_b_slot: Option<usize>,
    k_w_slot: usize,
    k_b_slot: Option<usize>,
    v_w_slot: usize,
    v_b_slot: Option<usize>,
    q_norm_w_slot: Option<usize>,
    k_norm_w_slot: Option<usize>,
    o_w_slot: usize,
    post_attn_ln_w_slot: usize,
    gate_w_slot: usize,
    up_w_slot: usize,
    down_w_slot: usize,
    // KV cache slots
    past_k_slot: usize,
    past_v_slot: usize,
    present_k_slot: usize,
    present_v_slot: usize,
    // Intermediates
    normed_slot: usize,
    q_proj_out_slot: usize,
    k_proj_out_slot: usize,
    v_proj_out_slot: usize,
    /// q [1, seq, num_heads, head_dim] BSHD (after RoPE, before transpose)
    q_slot: usize,
    /// k_bhsd [1, kv_heads, seq, head_dim] (BHSD, ready for KV concat)
    k_bhsd_slot: usize,
    /// v_bhsd [1, kv_heads, seq, head_dim]
    v_bhsd_slot: usize,
    attn_body_out_slot: usize,
    attn_out_slot: usize,
    residual_slot: usize,
    post_attn_normed_slot: usize,
    gate_out_slot: usize,
    up_out_slot: usize,
    silu_mul_out_slot: usize,
    down_out_slot: usize,
    hidden_out_slot: usize,
}

/// Compute total Q8_0 weight bytes for a matrix with `k` input features and `n` output features.
fn q8_0_weight_bytes(k: u64, n: u64) -> u64 {
    let num_blocks_per_row = k / 32;
    let total_blocks = n * num_blocks_per_row;
    total_blocks * 34
}

fn assign_transformer_slots_quant(
    config: &TransformerConfig,
    has_bias: bool,
    has_qk_norm: bool,
) -> QuantSlotLayout {
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
    let q_dim = (config.num_attention_heads * config.head_dim()) as u64;
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
    let mut weight_slots = Vec::new();
    let mut slot_descs_vec: Vec<SlotDesc> = vec![input_ids_desc, mask_desc, cos_desc, sin_desc];

    // embed_tokens_weight [vocab, hidden] f32
    let (embed_w_slot, d) = alloc(SlotDesc {
        shape: Shape(vec![vocab, hidden]),
        dtype: DType::F32,
        sym_shape: None,
    });
    weight_slots.push(embed_w_slot);
    slot_descs_vec.push(d);

    // embedding output
    let (embed_out_slot, d) = alloc(SlotDesc {
        shape: Shape(vec![1, dyn_val, hidden]),
        dtype: DType::F32,
        sym_shape: Some(vec![SymDim::Concrete(1), seq(), SymDim::Concrete(hidden)]),
    });
    slot_descs_vec.push(d);

    let mut layers: Vec<QuantLayerSlots> = Vec::with_capacity(config.num_hidden_layers);
    let mut past_kv_input_slots = Vec::new();
    let mut present_kv_output_slots = Vec::new();

    for layer_i in 0..config.num_hidden_layers {
        let hidden_in_slot = if layer_i == 0 {
            embed_out_slot
        } else {
            layers[layer_i - 1].hidden_out_slot
        };

        // input_layernorm_weight [hidden] f32
        let (input_ln_w_slot, d) = alloc(SlotDesc {
            shape: Shape(vec![hidden]),
            dtype: DType::F32,
            sym_shape: None,
        });
        weight_slots.push(input_ln_w_slot);
        slot_descs_vec.push(d);

        // q_proj weight: flat I8 bytes [total_weight_bytes]
        let q_bytes = q8_0_weight_bytes(hidden, q_dim);
        let (q_w_slot, d) = alloc(SlotDesc {
            shape: Shape(vec![q_bytes]),
            dtype: DType::I8,
            sym_shape: None,
        });
        weight_slots.push(q_w_slot);
        slot_descs_vec.push(d);

        let q_b_slot = if has_bias {
            let (s, d) = alloc(SlotDesc {
                shape: Shape(vec![q_dim]),
                dtype: DType::F32,
                sym_shape: None,
            });
            weight_slots.push(s);
            slot_descs_vec.push(d);
            Some(s)
        } else {
            None
        };

        // k_proj weight
        let k_bytes = q8_0_weight_bytes(hidden, kv_dim);
        let (k_w_slot, d) = alloc(SlotDesc {
            shape: Shape(vec![k_bytes]),
            dtype: DType::I8,
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

        // v_proj weight
        let v_bytes = q8_0_weight_bytes(hidden, kv_dim);
        let (v_w_slot, d) = alloc(SlotDesc {
            shape: Shape(vec![v_bytes]),
            dtype: DType::I8,
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

        // QK-norm weights (optional)
        let q_norm_w_slot = if has_qk_norm {
            let (s, d) = alloc(SlotDesc {
                shape: Shape(vec![head_dim]),
                dtype: DType::F32,
                sym_shape: None,
            });
            weight_slots.push(s);
            slot_descs_vec.push(d);
            Some(s)
        } else {
            None
        };
        let k_norm_w_slot = if has_qk_norm {
            let (s, d) = alloc(SlotDesc {
                shape: Shape(vec![head_dim]),
                dtype: DType::F32,
                sym_shape: None,
            });
            weight_slots.push(s);
            slot_descs_vec.push(d);
            Some(s)
        } else {
            None
        };

        // o_proj weight
        let o_bytes = q8_0_weight_bytes(q_dim, hidden);
        let (o_w_slot, d) = alloc(SlotDesc {
            shape: Shape(vec![o_bytes]),
            dtype: DType::I8,
            sym_shape: None,
        });
        weight_slots.push(o_w_slot);
        slot_descs_vec.push(d);

        // post_attention_layernorm_weight [hidden] f32
        let (post_attn_ln_w_slot, d) = alloc(SlotDesc {
            shape: Shape(vec![hidden]),
            dtype: DType::F32,
            sym_shape: None,
        });
        weight_slots.push(post_attn_ln_w_slot);
        slot_descs_vec.push(d);

        // gate_proj weight
        let gate_bytes = q8_0_weight_bytes(hidden, intermediate);
        let (gate_w_slot, d) = alloc(SlotDesc {
            shape: Shape(vec![gate_bytes]),
            dtype: DType::I8,
            sym_shape: None,
        });
        weight_slots.push(gate_w_slot);
        slot_descs_vec.push(d);

        // up_proj weight
        let up_bytes = q8_0_weight_bytes(hidden, intermediate);
        let (up_w_slot, d) = alloc(SlotDesc {
            shape: Shape(vec![up_bytes]),
            dtype: DType::I8,
            sym_shape: None,
        });
        weight_slots.push(up_w_slot);
        slot_descs_vec.push(d);

        // down_proj weight
        let down_bytes = q8_0_weight_bytes(intermediate, hidden);
        let (down_w_slot, d) = alloc(SlotDesc {
            shape: Shape(vec![down_bytes]),
            dtype: DType::I8,
            sym_shape: None,
        });
        weight_slots.push(down_w_slot);
        slot_descs_vec.push(d);

        // KV cache slots (BHSD layout: [1, kv_heads, seq, head_dim])
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

        // --- Intermediate slots ---

        // normed: RMSNorm output [1, seq, hidden]
        let (normed_slot, d) = alloc(SlotDesc {
            shape: Shape(vec![1, dyn_val, hidden]),
            dtype: DType::F32,
            sym_shape: Some(vec![SymDim::Concrete(1), seq(), SymDim::Concrete(hidden)]),
        });
        slot_descs_vec.push(d);

        // q_proj_out [1, seq, q_dim]
        let (q_proj_out_slot, d) = alloc(SlotDesc {
            shape: Shape(vec![1, dyn_val, q_dim]),
            dtype: DType::F32,
            sym_shape: Some(vec![SymDim::Concrete(1), seq(), SymDim::Concrete(q_dim)]),
        });
        slot_descs_vec.push(d);

        // k_proj_out [1, seq, kv_dim]
        let (k_proj_out_slot, d) = alloc(SlotDesc {
            shape: Shape(vec![1, dyn_val, kv_dim]),
            dtype: DType::F32,
            sym_shape: Some(vec![SymDim::Concrete(1), seq(), SymDim::Concrete(kv_dim)]),
        });
        slot_descs_vec.push(d);

        // v_proj_out [1, seq, kv_dim]
        let (v_proj_out_slot, d) = alloc(SlotDesc {
            shape: Shape(vec![1, dyn_val, kv_dim]),
            dtype: DType::F32,
            sym_shape: Some(vec![SymDim::Concrete(1), seq(), SymDim::Concrete(kv_dim)]),
        });
        slot_descs_vec.push(d);

        // q [1, seq, num_heads, head_dim] BSHD (post-RoPE, pre-transpose)
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

        // k_bhsd [1, kv_heads, seq, head_dim]
        let (k_bhsd_slot, d) = alloc(SlotDesc {
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

        // v_bhsd [1, kv_heads, seq, head_dim]
        let (v_bhsd_slot, d) = alloc(SlotDesc {
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

        // attn_body_out [1, seq, q_dim] (attention output before O_proj)
        let (attn_body_out_slot, d) = alloc(SlotDesc {
            shape: Shape(vec![1, dyn_val, q_dim]),
            dtype: DType::F32,
            sym_shape: Some(vec![SymDim::Concrete(1), seq(), SymDim::Concrete(q_dim)]),
        });
        slot_descs_vec.push(d);

        // attn_out [1, seq, hidden] (after O_proj dequant-matmul)
        let (attn_out_slot, d) = alloc(SlotDesc {
            shape: Shape(vec![1, dyn_val, hidden]),
            dtype: DType::F32,
            sym_shape: Some(vec![SymDim::Concrete(1), seq(), SymDim::Concrete(hidden)]),
        });
        slot_descs_vec.push(d);

        // residual [1, seq, hidden] (hidden_in + attn_out, before post-attn norm)
        let (residual_slot, d) = alloc(SlotDesc {
            shape: Shape(vec![1, dyn_val, hidden]),
            dtype: DType::F32,
            sym_shape: Some(vec![SymDim::Concrete(1), seq(), SymDim::Concrete(hidden)]),
        });
        slot_descs_vec.push(d);

        // post_attn_normed [1, seq, hidden]
        let (post_attn_normed_slot, d) = alloc(SlotDesc {
            shape: Shape(vec![1, dyn_val, hidden]),
            dtype: DType::F32,
            sym_shape: Some(vec![SymDim::Concrete(1), seq(), SymDim::Concrete(hidden)]),
        });
        slot_descs_vec.push(d);

        // gate_out [1, seq, intermediate]
        let (gate_out_slot, d) = alloc(SlotDesc {
            shape: Shape(vec![1, dyn_val, intermediate]),
            dtype: DType::F32,
            sym_shape: Some(vec![
                SymDim::Concrete(1),
                seq(),
                SymDim::Concrete(intermediate),
            ]),
        });
        slot_descs_vec.push(d);

        // up_out [1, seq, intermediate]
        let (up_out_slot, d) = alloc(SlotDesc {
            shape: Shape(vec![1, dyn_val, intermediate]),
            dtype: DType::F32,
            sym_shape: Some(vec![
                SymDim::Concrete(1),
                seq(),
                SymDim::Concrete(intermediate),
            ]),
        });
        slot_descs_vec.push(d);

        // silu_mul_out [1, seq, intermediate]
        let (silu_mul_out_slot, d) = alloc(SlotDesc {
            shape: Shape(vec![1, dyn_val, intermediate]),
            dtype: DType::F32,
            sym_shape: Some(vec![
                SymDim::Concrete(1),
                seq(),
                SymDim::Concrete(intermediate),
            ]),
        });
        slot_descs_vec.push(d);

        // down_out [1, seq, hidden]
        let (down_out_slot, d) = alloc(SlotDesc {
            shape: Shape(vec![1, dyn_val, hidden]),
            dtype: DType::F32,
            sym_shape: Some(vec![SymDim::Concrete(1), seq(), SymDim::Concrete(hidden)]),
        });
        slot_descs_vec.push(d);

        // hidden_out [1, seq, hidden]
        let (hidden_out_slot, d) = alloc(SlotDesc {
            shape: Shape(vec![1, dyn_val, hidden]),
            dtype: DType::F32,
            sym_shape: Some(vec![SymDim::Concrete(1), seq(), SymDim::Concrete(hidden)]),
        });
        slot_descs_vec.push(d);

        layers.push(QuantLayerSlots {
            hidden_in_slot,
            input_ln_w_slot,
            q_w_slot,
            q_b_slot,
            k_w_slot,
            k_b_slot,
            v_w_slot,
            v_b_slot,
            q_norm_w_slot,
            k_norm_w_slot,
            o_w_slot,
            post_attn_ln_w_slot,
            gate_w_slot,
            up_w_slot,
            down_w_slot,
            past_k_slot,
            past_v_slot,
            present_k_slot,
            present_v_slot,
            normed_slot,
            q_proj_out_slot,
            k_proj_out_slot,
            v_proj_out_slot,
            q_slot,
            k_bhsd_slot,
            v_bhsd_slot,
            attn_body_out_slot,
            attn_out_slot,
            residual_slot,
            post_attn_normed_slot,
            gate_out_slot,
            up_out_slot,
            silu_mul_out_slot,
            down_out_slot,
            hidden_out_slot,
        });
    }

    // final_norm_weight [hidden] f32
    let (final_norm_w_slot, d) = alloc(SlotDesc {
        shape: Shape(vec![hidden]),
        dtype: DType::F32,
        sym_shape: None,
    });
    weight_slots.push(final_norm_w_slot);
    slot_descs_vec.push(d);

    // lm_head_weight [hidden, vocab] f32 (always f32 in quant path)
    let (lm_head_weight_slot, d) = alloc(SlotDesc {
        shape: Shape(vec![hidden, vocab]),
        dtype: DType::F32,
        sym_shape: None,
    });
    weight_slots.push(lm_head_weight_slot);
    slot_descs_vec.push(d);

    // logits [1, seq, vocab]
    let (logits_slot, d) = alloc(SlotDesc {
        shape: Shape(vec![1, dyn_val, vocab]),
        dtype: DType::F32,
        sym_shape: Some(vec![SymDim::Concrete(1), seq(), SymDim::Concrete(vocab)]),
    });
    slot_descs_vec.push(d);

    let mut output_slots = vec![logits_slot];
    output_slots.extend_from_slice(&present_kv_output_slots);

    assert_eq!(slot_descs_vec.len(), next);

    QuantSlotLayout {
        num_slots: next,
        input_slots,
        weight_slots,
        output_slots,
        slot_descs: slot_descs_vec,
        layers,
        embed_out_slot,
        final_norm_weight_slot: final_norm_w_slot,
        lm_head_weight_slot,
        logits_slot,
        past_kv_input_slots,
        present_kv_output_slots,
    }
}

// ── Kernel emission helpers ───────────────────────────────────────────────────

/// Emit a Q8_0 dequant-matmul kernel.
///
/// Inputs: activation [1, seq, k], weight [total_weight_bytes] i8
/// Output: [1, seq, n] f32
fn emit_quant_matmul_kernel(k: u64, n: u64, kernel_idx: usize, label: &str) -> KernelEmitResult {
    let n_tile = pick_n_tile(n);
    let func_name = format!("k{kernel_idx}");
    let mlir_text = crate::graph_builder::emit_dequant_matmul_q8_0(k, n, &func_name, n_tile);

    let dyn_val = crate::shape::DIM_DYNAMIC;

    KernelEmitResult {
        mlir_text,
        // Q8_0 kernel takes 2 inputs: activation memref + weight memref.
        num_inputs: 2,
        output_descs: vec![crate::runtime::OutputDesc {
            shape: Shape(vec![1, dyn_val, n]),
            dtype: DType::F32,
        }],
        group_idx: kernel_idx,
        num_in: 2,
        num_out: 1,
        ops_label: format!("QuantMM[{label}]"),
    }
}

/// Emit RMSNorm-only kernel.
///
/// Inputs: hidden [1, seq, hidden], ln_w [hidden]
/// Output: normed [1, seq, hidden]
fn emit_rms_norm_kernel(
    hidden: u64,
    eps: f32,
    kernel_idx: usize,
) -> Result<KernelEmitResult, CompileError> {
    let ctx = GraphContext::new();
    let mut gb = ctx.builder();

    let h = gb.input(&[Some(1), None, Some(hidden)], DType::F32);
    let ln_w = gb.input(&[Some(hidden)], DType::F32);
    let normed = gb.emit_rms_norm(&h, &ln_w, eps);

    let func_name = format!("k{kernel_idx}");
    let (mlir_text, num_inputs, output_descs) =
        gb.finalize_to_mlir_named(&[&normed], TransformMode::None, &func_name)?;

    Ok(KernelEmitResult {
        mlir_text,
        num_inputs,
        output_descs,
        group_idx: kernel_idx,
        num_in: 2,
        num_out: 1,
        ops_label: "RMSNorm".into(),
    })
}

/// Emit post-QKV processing kernel: bias + QKNorm + reshape + RoPE + transpose.
///
/// Inputs: q_raw [1, seq, q_dim], k_raw [1, seq, kv_dim], v_raw [1, seq, kv_dim],
///         [q_b [q_dim]], [k_b [kv_dim]], [v_b [kv_dim]],
///         cos [seq, head_dim], sin [seq, head_dim],
///         [q_norm_w [head_dim]], [k_norm_w [head_dim]]
/// Outputs: q [1, seq, num_heads, head_dim],
///          k_bhsd [1, kv_heads, seq, head_dim],
///          v_bhsd [1, kv_heads, seq, head_dim]
fn emit_post_qkv_kernel(
    config: &TransformerConfig,
    has_bias: bool,
    has_qk_norm: bool,
    kernel_idx: usize,
) -> Result<KernelEmitResult, CompileError> {
    let kv_dim = (config.kv_heads() * config.head_dim()) as u64;
    let q_dim = (config.num_attention_heads * config.head_dim()) as u64;
    let num_heads = config.num_attention_heads as u64;
    let kv_heads = config.kv_heads() as u64;
    let head_dim = config.head_dim() as u64;

    let ctx = GraphContext::new();
    let mut gb = ctx.builder();

    let q_raw = gb.input(&[Some(1), None, Some(q_dim)], DType::F32);
    let k_raw = gb.input(&[Some(1), None, Some(kv_dim)], DType::F32);
    let v_raw = gb.input(&[Some(1), None, Some(kv_dim)], DType::F32);

    let q_b = if has_bias {
        Some(gb.input(&[Some(q_dim)], DType::F32))
    } else {
        None
    };
    let k_b = if has_bias {
        Some(gb.input(&[Some(kv_dim)], DType::F32))
    } else {
        None
    };
    let v_b = if has_bias {
        Some(gb.input(&[Some(kv_dim)], DType::F32))
    } else {
        None
    };
    let cos = gb.input(&[None, Some(head_dim)], DType::F32);
    let sin = gb.input(&[None, Some(head_dim)], DType::F32);
    let q_norm_w = if has_qk_norm {
        Some(gb.input(&[Some(head_dim)], DType::F32))
    } else {
        None
    };
    let k_norm_w = if has_qk_norm {
        Some(gb.input(&[Some(head_dim)], DType::F32))
    } else {
        None
    };

    // Apply biases if present
    let mut q = q_raw;
    if let Some(b) = &q_b {
        q = gb.emit_add(&q, b);
    }
    let mut k = k_raw;
    if let Some(b) = &k_b {
        k = gb.emit_add(&k, b);
    }
    let mut v = v_raw;
    if let Some(b) = &v_b {
        v = gb.emit_add(&v, b);
    }

    // Reshape to head layout
    let q_4d = gb.emit_reshape(&q, &[1, -1, num_heads as i64, head_dim as i64]);
    let k_4d = gb.emit_reshape(&k, &[1, -1, kv_heads as i64, head_dim as i64]);
    let v_4d = gb.emit_reshape(&v, &[1, -1, kv_heads as i64, head_dim as i64]);

    // QK-norm (Qwen3-style)
    let q_pre_rope = if let Some(w) = &q_norm_w {
        gb.emit_rms_norm(&q_4d, w, config.rms_norm_eps as f32)
    } else {
        q_4d
    };
    let k_pre_rope = if let Some(w) = &k_norm_w {
        gb.emit_rms_norm(&k_4d, w, config.rms_norm_eps as f32)
    } else {
        k_4d
    };

    // RoPE
    let q_rope = gb.emit_rope_half(&q_pre_rope, &cos, &sin);
    let k_rope = gb.emit_rope_half(&k_pre_rope, &cos, &sin);

    // Transpose K and V to BHSD [1, kv_heads, seq, head_dim]
    let k_bhsd = gb.emit_transpose(&k_rope, &[0, 2, 1, 3]);
    let v_bhsd = gb.emit_transpose(&v_4d, &[0, 2, 1, 3]);

    // q stays in BSHD [1, seq, num_heads, head_dim] as expected by attn_body kernel

    let func_name = format!("k{kernel_idx}");
    let (mlir_text, num_inputs, output_descs) = gb.finalize_to_mlir_named(
        &[&q_rope, &k_bhsd, &v_bhsd],
        TransformMode::None,
        &func_name,
    )?;

    let label = if has_qk_norm {
        "PostQKV+QKNorm+RoPE"
    } else {
        "PostQKV+RoPE"
    };

    Ok(KernelEmitResult {
        mlir_text,
        num_inputs,
        output_descs,
        group_idx: kernel_idx,
        num_in: num_inputs,
        num_out: 3,
        ops_label: label.into(),
    })
}

/// Emit attention body kernel: transpose + QK^T + softmax + AV + reshape.
/// Does NOT include O_proj (that is a separate quant-matmul step).
///
/// Inputs: q [1, seq, num_heads, head_dim] (BSHD),
///         k [1, kv_heads, total_seq, head_dim] (BHSD),
///         v [1, kv_heads, total_seq, head_dim] (BHSD),
///         mask [1, 1, seq, total_seq]
/// Output: attn_flat [1, seq, q_dim]
fn emit_attention_body_kernel(
    config: &TransformerConfig,
    kernel_idx: usize,
) -> Result<KernelEmitResult, CompileError> {
    let num_heads = config.num_attention_heads as u64;
    let kv_heads = config.kv_heads() as u64;
    let head_dim = config.head_dim() as u64;
    let q_dim = num_heads * head_dim;
    let gqa_repeat = config.gqa_repeat();

    let ctx = GraphContext::new();
    let mut gb = ctx.builder();

    let q = gb.input(
        &[Some(1), None, Some(num_heads), Some(head_dim)],
        DType::F32,
    );
    let k = gb.input(&[Some(1), Some(kv_heads), None, Some(head_dim)], DType::F32);
    let v = gb.input(&[Some(1), Some(kv_heads), None, Some(head_dim)], DType::F32);
    let mask = gb.input(&[Some(1), Some(1), None, None], DType::F32);

    // Transpose Q: BSHD -> BHSD
    let q_bhsd = gb.emit_transpose(&q, &[0, 2, 1, 3]);

    // GQA-aware QK^T: [1, num_heads, seq, head_dim] x [1, kv_heads, total_seq, head_dim]
    let scores = gb.emit_gqa_qk_transpose(&q_bhsd, &k, gqa_repeat);
    let scale = gb.emit_arith_constant(1.0 / (head_dim as f64).sqrt(), DType::F32);
    let scaled = gb.emit_mul(&scores, &scale);
    let masked = gb.emit_add(&scaled, &mask);
    let attn_weights = gb.emit_softmax(&masked, -1);

    // GQA-aware AV
    let attn_output = gb.emit_gqa_av(&attn_weights, &v, gqa_repeat);

    // Transpose back: BHSD -> BSHD
    let attn_bshd = gb.emit_transpose(&attn_output, &[0, 2, 1, 3]);

    // Reshape to [1, seq, q_dim]
    let attn_flat = gb.emit_reshape(&attn_bshd, &[1, -1, q_dim as i64]);

    let func_name = format!("k{kernel_idx}");
    let (mlir_text, num_inputs, output_descs) =
        gb.finalize_to_mlir_named(&[&attn_flat], TransformMode::None, &func_name)?;

    Ok(KernelEmitResult {
        mlir_text,
        num_inputs,
        output_descs,
        group_idx: kernel_idx,
        num_in: 4,
        num_out: 1,
        ops_label: "AttnBody".into(),
    })
}

/// Emit residual + post-attn RMSNorm kernel.
///
/// Inputs: hidden_in [1, seq, hidden], attn_out [1, seq, hidden], post_attn_ln_w [hidden]
/// Outputs: residual [1, seq, hidden], post_attn_normed [1, seq, hidden]
fn emit_residual_norm_kernel(
    hidden: u64,
    eps: f32,
    kernel_idx: usize,
) -> Result<KernelEmitResult, CompileError> {
    let ctx = GraphContext::new();
    let mut gb = ctx.builder();

    let h = gb.input(&[Some(1), None, Some(hidden)], DType::F32);
    let attn_out = gb.input(&[Some(1), None, Some(hidden)], DType::F32);
    let ln_w = gb.input(&[Some(hidden)], DType::F32);

    let residual = gb.emit_add(&h, &attn_out);
    let normed = gb.emit_rms_norm(&residual, &ln_w, eps);

    let func_name = format!("k{kernel_idx}");
    let (mlir_text, num_inputs, output_descs) =
        gb.finalize_to_mlir_named(&[&residual, &normed], TransformMode::None, &func_name)?;

    Ok(KernelEmitResult {
        mlir_text,
        num_inputs,
        output_descs,
        group_idx: kernel_idx,
        num_in: 3,
        num_out: 2,
        ops_label: "Residual+Norm".into(),
    })
}

/// Emit SiLU * up kernel.
///
/// Inputs: gate [1, seq, intermediate], up [1, seq, intermediate]
/// Output: silu(gate) * up [1, seq, intermediate]
fn emit_silu_mul_kernel(
    intermediate: u64,
    kernel_idx: usize,
) -> Result<KernelEmitResult, CompileError> {
    let ctx = GraphContext::new();
    let mut gb = ctx.builder();

    let gate = gb.input(&[Some(1), None, Some(intermediate)], DType::F32);
    let up = gb.input(&[Some(1), None, Some(intermediate)], DType::F32);

    let gate_act = gb.emit_silu(&gate);
    let out = gb.emit_mul(&gate_act, &up);

    let func_name = format!("k{kernel_idx}");
    let (mlir_text, num_inputs, output_descs) =
        gb.finalize_to_mlir_named(&[&out], TransformMode::None, &func_name)?;

    Ok(KernelEmitResult {
        mlir_text,
        num_inputs,
        output_descs,
        group_idx: kernel_idx,
        num_in: 2,
        num_out: 1,
        ops_label: "SiLU*Up".into(),
    })
}

/// Emit residual add kernel: residual + down.
///
/// Inputs: residual [1, seq, hidden], down [1, seq, hidden]
/// Output: hidden_out [1, seq, hidden]
fn emit_residual_add_kernel(
    hidden: u64,
    kernel_idx: usize,
) -> Result<KernelEmitResult, CompileError> {
    let ctx = GraphContext::new();
    let mut gb = ctx.builder();

    let residual = gb.input(&[Some(1), None, Some(hidden)], DType::F32);
    let down = gb.input(&[Some(1), None, Some(hidden)], DType::F32);

    let out = gb.emit_add(&residual, &down);

    let func_name = format!("k{kernel_idx}");
    let (mlir_text, num_inputs, output_descs) =
        gb.finalize_to_mlir_named(&[&out], TransformMode::None, &func_name)?;

    Ok(KernelEmitResult {
        mlir_text,
        num_inputs,
        output_descs,
        group_idx: kernel_idx,
        num_in: 2,
        num_out: 1,
        ops_label: "ResAdd".into(),
    })
}

// ── Top-level quantized plan emission ────────────────────────────────────────

/// Emit and compile all kernels for a quantized transformer, producing an ExecutionPlan.
///
/// Uses `emit_dequant_matmul_q8_0` for projection weights and lightweight tensor
/// kernels for all other ops (RMSNorm, RoPE, residuals, SiLU, attention body).
pub(crate) fn emit_transformer_plan_quant(
    config: &TransformerConfig,
    weights: &TransformerWeights,
) -> Result<(ExecutionPlan, Vec<crate::runtime::Buffer>), CompileError> {
    let has_bias = weights
        .layers
        .first()
        .map(|l| l.q_proj_bias.is_some())
        .unwrap_or(false);
    let has_qk_norm = weights.has_qk_norm();

    let layout = assign_transformer_slots_quant(config, has_bias, has_qk_norm);

    log_compile!(
        "hf",
        "quant emitting {} layers, {} total slots",
        config.num_hidden_layers,
        layout.num_slots
    );

    let hidden = config.hidden_size as u64;
    let kv_dim = (config.kv_heads() * config.head_dim()) as u64;
    let q_dim = (config.num_attention_heads * config.head_dim()) as u64;
    let intermediate = config.intermediate_size as u64;
    let vocab = config.vocab_size as u64;
    let eps = config.rms_norm_eps as f32;

    let mut emit_results: Vec<KernelEmitResult> = Vec::new();
    let mut steps: Vec<KernelStep> = Vec::new();
    let mut kernel_idx = 0usize;

    // --- Embedding kernel (same as non-quant) ---
    {
        use crate::graph_builder::GraphContext;
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
        let hidden_states = gb.emit_embedding(&embed_w, &input_ids);
        let func_name = format!("k{kernel_idx}");
        let (mlir_text, num_inputs, output_descs) =
            gb.finalize_to_mlir_named(&[&hidden_states], TransformMode::None, &func_name)?;
        emit_results.push(KernelEmitResult {
            mlir_text,
            num_inputs,
            output_descs,
            group_idx: kernel_idx,
            num_in: 2,
            num_out: 1,
            ops_label: "Embed".into(),
        });
        steps.push(KernelStep::kernel(
            kernel_idx,
            vec![layout.input_slots[0], layout.weight_slots[0]],
            vec![layout.embed_out_slot],
        ));
        kernel_idx += 1;
    }

    // --- Per-layer kernels ---
    for i in 0..config.num_hidden_layers {
        let ls = &layout.layers[i];

        // Step 1: pre-QKV RMSNorm
        emit_results.push(emit_rms_norm_kernel(hidden, eps, kernel_idx)?);
        steps.push(KernelStep::kernel(
            kernel_idx,
            vec![ls.hidden_in_slot, ls.input_ln_w_slot],
            vec![ls.normed_slot],
        ));
        kernel_idx += 1;

        // Step 2: Q_proj dequant-matmul
        emit_results.push(emit_quant_matmul_kernel(hidden, q_dim, kernel_idx, "Q"));
        steps.push(KernelStep::kernel(
            kernel_idx,
            vec![ls.normed_slot, ls.q_w_slot],
            vec![ls.q_proj_out_slot],
        ));
        kernel_idx += 1;

        // Step 3: K_proj dequant-matmul
        emit_results.push(emit_quant_matmul_kernel(hidden, kv_dim, kernel_idx, "K"));
        steps.push(KernelStep::kernel(
            kernel_idx,
            vec![ls.normed_slot, ls.k_w_slot],
            vec![ls.k_proj_out_slot],
        ));
        kernel_idx += 1;

        // Step 4: V_proj dequant-matmul
        emit_results.push(emit_quant_matmul_kernel(hidden, kv_dim, kernel_idx, "V"));
        steps.push(KernelStep::kernel(
            kernel_idx,
            vec![ls.normed_slot, ls.v_w_slot],
            vec![ls.v_proj_out_slot],
        ));
        kernel_idx += 1;

        // Step 5: post-QKV (bias + QKNorm + reshape + RoPE + transpose)
        emit_results.push(emit_post_qkv_kernel(
            config,
            has_bias,
            has_qk_norm,
            kernel_idx,
        )?);
        let mut post_qkv_inputs = vec![ls.q_proj_out_slot, ls.k_proj_out_slot, ls.v_proj_out_slot];
        if let Some(s) = ls.q_b_slot {
            post_qkv_inputs.push(s);
        }
        if let Some(s) = ls.k_b_slot {
            post_qkv_inputs.push(s);
        }
        if let Some(s) = ls.v_b_slot {
            post_qkv_inputs.push(s);
        }
        post_qkv_inputs.push(layout.input_slots[2]); // cos
        post_qkv_inputs.push(layout.input_slots[3]); // sin
        if let Some(s) = ls.q_norm_w_slot {
            post_qkv_inputs.push(s);
        }
        if let Some(s) = ls.k_norm_w_slot {
            post_qkv_inputs.push(s);
        }
        steps.push(KernelStep::kernel(
            kernel_idx,
            post_qkv_inputs,
            vec![ls.q_slot, ls.k_bhsd_slot, ls.v_bhsd_slot],
        ));
        kernel_idx += 1;

        // Step 6: KV concat K
        steps.push(KernelStep {
            kernel_idx: usize::MAX,
            input_slots: vec![ls.past_k_slot, ls.k_bhsd_slot],
            output_slots: vec![ls.present_k_slot],
            native_op: Some(crate::runtime::NativeOp::Concat { axis: 2 }),
        });

        // Step 7: KV concat V
        steps.push(KernelStep {
            kernel_idx: usize::MAX,
            input_slots: vec![ls.past_v_slot, ls.v_bhsd_slot],
            output_slots: vec![ls.present_v_slot],
            native_op: Some(crate::runtime::NativeOp::Concat { axis: 2 }),
        });

        // Step 8: Attention body (no O_proj)
        emit_results.push(emit_attention_body_kernel(config, kernel_idx)?);
        steps.push(KernelStep::kernel(
            kernel_idx,
            vec![
                ls.q_slot,
                ls.present_k_slot,
                ls.present_v_slot,
                layout.input_slots[1], // mask
            ],
            vec![ls.attn_body_out_slot],
        ));
        kernel_idx += 1;

        // Step 9: O_proj dequant-matmul
        emit_results.push(emit_quant_matmul_kernel(q_dim, hidden, kernel_idx, "O"));
        steps.push(KernelStep::kernel(
            kernel_idx,
            vec![ls.attn_body_out_slot, ls.o_w_slot],
            vec![ls.attn_out_slot],
        ));
        kernel_idx += 1;

        // Step 10: residual + post-attn RMSNorm (two outputs)
        emit_results.push(emit_residual_norm_kernel(hidden, eps, kernel_idx)?);
        steps.push(KernelStep::kernel(
            kernel_idx,
            vec![ls.hidden_in_slot, ls.attn_out_slot, ls.post_attn_ln_w_slot],
            vec![ls.residual_slot, ls.post_attn_normed_slot],
        ));
        kernel_idx += 1;

        // Step 11: gate dequant-matmul
        emit_results.push(emit_quant_matmul_kernel(
            hidden,
            intermediate,
            kernel_idx,
            "Gate",
        ));
        steps.push(KernelStep::kernel(
            kernel_idx,
            vec![ls.post_attn_normed_slot, ls.gate_w_slot],
            vec![ls.gate_out_slot],
        ));
        kernel_idx += 1;

        // Step 12: up dequant-matmul
        emit_results.push(emit_quant_matmul_kernel(
            hidden,
            intermediate,
            kernel_idx,
            "Up",
        ));
        steps.push(KernelStep::kernel(
            kernel_idx,
            vec![ls.post_attn_normed_slot, ls.up_w_slot],
            vec![ls.up_out_slot],
        ));
        kernel_idx += 1;

        // Step 13: SiLU * up
        emit_results.push(emit_silu_mul_kernel(intermediate, kernel_idx)?);
        steps.push(KernelStep::kernel(
            kernel_idx,
            vec![ls.gate_out_slot, ls.up_out_slot],
            vec![ls.silu_mul_out_slot],
        ));
        kernel_idx += 1;

        // Step 14: down dequant-matmul
        emit_results.push(emit_quant_matmul_kernel(
            intermediate,
            hidden,
            kernel_idx,
            "Down",
        ));
        steps.push(KernelStep::kernel(
            kernel_idx,
            vec![ls.silu_mul_out_slot, ls.down_w_slot],
            vec![ls.down_out_slot],
        ));
        kernel_idx += 1;

        // Step 15: residual add
        emit_results.push(emit_residual_add_kernel(hidden, kernel_idx)?);
        steps.push(KernelStep::kernel(
            kernel_idx,
            vec![ls.residual_slot, ls.down_out_slot],
            vec![ls.hidden_out_slot],
        ));
        kernel_idx += 1;
    }

    // --- LM head kernel: final RMSNorm + matmul (f32 weights) ---
    let last_hidden = layout
        .layers
        .last()
        .ok_or_else(|| CompileError::Shape("transformer has no layers".into()))?
        .hidden_out_slot;
    {
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();

        let h = gb.input(&[Some(1), None, Some(hidden)], DType::F32);
        let norm_w = gb.input(&[Some(hidden)], DType::F32);
        let lm_head_w = gb.input(&[Some(hidden), Some(vocab)], DType::F32);

        let normed = gb.emit_rms_norm(&h, &norm_w, eps);
        let logits = gb.emit_matmul(&normed, &lm_head_w);

        let func_name = format!("k{kernel_idx}");
        let mode = crate::graph_builder::TransformMode::tile_parallel_adaptive(vocab as usize);
        let (mlir_text, num_inputs, output_descs) =
            gb.finalize_to_mlir_named(&[&logits], mode, &func_name)?;

        emit_results.push(KernelEmitResult {
            mlir_text,
            num_inputs,
            output_descs,
            group_idx: kernel_idx,
            num_in: 3,
            num_out: 1,
            ops_label: "LMHead".into(),
        });
        steps.push(KernelStep::kernel(
            kernel_idx,
            vec![
                last_hidden,
                layout.final_norm_weight_slot,
                layout.lm_head_weight_slot,
            ],
            vec![layout.logits_slot],
        ));
        kernel_idx += 1;
    }

    log_compile!(
        "hf",
        "quant: {} compiled kernels + {} native concat steps",
        emit_results.len(),
        steps.iter().filter(|s| s.native_op.is_some()).count()
    );

    // Compile all kernels in parallel and link
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

    let weight_bufs = weights.to_slot_buffers_quant();

    let _ = kernel_idx; // suppress unused warning after last increment

    Ok((plan, weight_bufs))
}
