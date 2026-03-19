# Native HuggingFace Model Path — Implementation Plan

## Goal

Run Qwen2.5-0.5B (and Llama/Mistral/Phi family) directly from safetensors + config.json, bypassing ONNX entirely. Emit per-kernel MLIR via existing GraphBuilder primitives, compile via the same parallel pipeline, produce an ExecutionPlan compatible with run/run_kv.

## Target Model: Qwen2.5-0.5B

- 24 layers, hidden_size=896, intermediate_size=4864
- GQA: 14 query heads, 2 KV heads, head_dim=64
- RoPE (theta=1M), RMSNorm (eps=1e-6), SiLU activation
- SwiGLU MLP: gate_proj + up_proj -> silu(gate) * up -> down_proj
- Tied embeddings (lm_head = embed_tokens)
- Q/K/V projections have biases, O projection has no bias
- BF16 weights (~1.8GB as F32)

## Existing Infrastructure

| File | What it provides |
|------|------------------|
| `src/hf/safetensors.rs` | Loads safetensors, BF16->F32, mmap-based |
| `src/hf/config.rs` | `TransformerConfig` parses config.json |
| `src/hf/tokenizer.rs` | Wraps HF `tokenizers` crate |
| `src/graph_builder/emit_linalg.rs` | emit_rms_norm, emit_silu, emit_rope, emit_repeat_kv, emit_embedding |
| `src/graph_builder/emit_arithmetic.rs` | emit_add, emit_mul, emit_softmax |
| `src/graph_builder/emit_reshape.rs` | emit_reshape, emit_transpose, emit_concat, emit_slice |
| `src/runtime.rs` | ExecutionPlan, Buffer, KvCache, KvPlanInfo, CompiledGraph |
| `src/compiler.rs` | build_transform_schedule, emit_object_files, link_shared_lib |

## Kernel Partitioning

Per transformer layer: 5 compiled kernels + 2 native concat steps.

| Kernel | Ops | Transform Mode |
|--------|-----|----------------|
| `embed` | Embedding lookup | None |
| `layer{i}_qkv` | RMSNorm + Q/K/V matmuls + optional bias + reshape + RoPE(Q,K) | VectorizeOnly |
| `layer{i}_kv_concat_k` | NativeOp::Concat (KvCache managed) | N/A |
| `layer{i}_kv_concat_v` | NativeOp::Concat (KvCache managed) | N/A |
| `layer{i}_attn` | repeat_kv + QK^T + scale + mask + softmax + AV + O proj | VectorizeOnly |
| `layer{i}_residual_mlp` | residual add + RMSNorm + gate/up matmuls + SiLU*up + down matmul + residual add | TileParallel |
| `lm_head` | RMSNorm + matmul (final norm + logit projection) | TileParallel |

Qwen2.5-0.5B totals: 1 embed + 24*(4 compiled + 2 native) + 1 lm_head = **98 compiled kernels + 48 native concat steps**

## Buffer Slot Layout

```
Slot 0:       input_ids [1, seq]                      (input, I64)
Slot 1:       attention_mask [1, 1, seq, total_seq]    (input, F32)
Slot 2:       cos [seq, head_dim/2]                    (input, F32)
Slot 3:       sin [seq, head_dim/2]                    (input, F32)

Weights (borrowed, never copied at runtime):
Slot 4:       embed_tokens.weight [vocab, hidden]
Slot 5..N:    per-layer weights (ln, q/k/v/o weights, optional biases, mlp weights)
Slot N+1:     final_norm.weight
Slot N+2:     lm_head.weight (= slot 4 if tied)

KV slots (managed by run_kv):
past_k_layer{i}, past_v_layer{i}     (input)
present_k_layer{i}, present_v_layer{i} (output)

Intermediates (allocated/recycled per run):
hidden_states, q, k, v, attn_out, etc.

Output:
logits [1, seq, vocab]
```

All seq dims are symbolic (SymDim::Var("seq")), total_seq is Var("total_seq").

## Design Decisions

### Weight transposition at load time
HF stores linear weights as [out_features, in_features]. emit_matmul expects [in, out]. Transpose once at load time (one memcpy per weight) rather than emitting transpose ops in every kernel. Zero runtime cost after loading.

### RoPE in QKV kernel, before KV concat
Past K in KV cache is already rotated. New K must be rotated before entering the concat. So RoPE is applied in the QKV kernel to Q and new_K. The attention kernel receives already-rotated K from the cache.

### Fused MLP kernel
The MLP kernel includes both residual additions (pre-attn + post-mlp) plus RMSNorm + SwiGLU. Intermediates (gate [1, seq, 4864], up [1, seq, 4864]) stay within the kernel's register/cache hierarchy instead of being materialized as inter-kernel buffers.

### Fused attention kernel
Score matrix Q@K^T is not materialized between kernels. During decode (seq=1) it's tiny; during prefill it's bounded. Splitting would add unnecessary buffer management overhead.

### Causal mask computed in Rust
Passed as input buffer. Simple, flexible (different shape for prefill vs decode), avoids complex mask emission in MLIR.

### RoPE cos/sin precomputed in Rust
Full tables computed once at model load. Sliced per-call based on position. Avoids emitting trig ops in MLIR.

## Implementation Phases

### Phase 1: Precompute helpers
New file: `src/hf/precompute.rs`

- `precompute_rope_cos_sin(head_dim, max_seq_len, theta) -> (Buffer, Buffer)` -- full cos/sin tables [max_seq_len, head_dim/2] F32
- `build_causal_mask(seq_len, past_len) -> Buffer` -- [1, 1, seq_len, seq_len + past_len] with 0.0/NEG_INF
- `slice_rope_for_positions(full_cos, full_sin, positions) -> (Buffer, Buffer)` -- gather rows by position index
- Unit tests comparing against PyTorch reference values

### Phase 2: Weight loading and layout
New file: `src/hf/weights.rs`

- `TransformerWeights` struct: embed_weight, layers: Vec<LayerWeights>, final_norm_weight, lm_head_weight: Option<Buffer>
- `LayerWeights`: input_ln_weight, q/k/v/o proj weights, optional biases, gate/up/down proj weights, post_attn_ln_weight
- `load_transformer_weights(config, tensors) -> TransformerWeights` -- organize HashMap, detect biases, handle tied embeddings, **transpose linear weights [out, in] -> [in, out]**
- `to_flat_buffers() -> Vec<Buffer>` -- deterministic order matching slot assignment
- Unit test with real Qwen2.5-0.5B weights

### Phase 3: Transformer emission (core)
New file: `src/hf/emitter.rs`

- `emit_transformer_plan(config, weights) -> (ExecutionPlan, Vec<Buffer>)` -- top-level orchestrator
- `emit_embed_kernel` -- embedding lookup
- `emit_qkv_kernel` -- RMSNorm + Q/K/V projections + reshape + RoPE(Q, K)
- `emit_attention_kernel` -- repeat_kv + scaled dot-product attention + O projection
- `emit_mlp_kernel` -- residual add + RMSNorm + SwiGLU MLP + residual add
- `emit_lm_head_kernel` -- RMSNorm + logit projection
- `assign_transformer_slots(config) -> SlotLayout` -- deterministic slot assignment
- Compile all kernels in parallel, assemble ExecutionPlan with KvPlanInfo

### Phase 4: Extract shared compilation infrastructure
- Move `compile_and_link_kernels` and `KernelEmitResult` from `src/onnx/emitter.rs` to shared location
- Both ONNX and HF emitters call into it
- Verify existing e2e tests still pass

### Phase 5: HfModel top-level API
New file: `src/hf/model.rs`

- `HfModel::load(model_dir)` -- loads config.json, safetensors, organizes weights, precomputes RoPE tables
- `HfModel::run(input_ids) -> logits` -- prefill mode (computes mask, RoPE slice, calls plan.run)
- `HfModel::run_kv(input_ids, max_seq_len) -> logits` -- decode mode (calls plan.run_kv)
- Lazy compilation on first run

### Phase 6: Generator and e2e test
- Adapt or create generator for HfModel (prefill + decode loop)
- Integration test: `tests/qwen2_e2e.rs` -- load real model, generate text
- Example: `examples/hf_qwen2.rs`
- Download script: `scripts/download_qwen2.sh`

### Phase 7: Multi-file safetensors (stretch)
- Detect `model.safetensors.index.json`, load shards
- Merge tensors from multiple files

## Edge Cases

- **Tied embeddings**: lm_head and embed_tokens share the same buffer slot when `tie_word_embeddings=true`
- **GQA repeat factor**: must divide evenly (14 heads / 2 kv_heads = 7 repeats)
- **Dynamic seq dim**: all kernels compile with symbolic seq, resolve at runtime. Decode has seq=1, prefill has seq=N
- **Large vocab matmul**: lm_head [151936, 896] uses TileParallel with OpenMP
- **Causal mask shape**: prefill [1,1,seq,seq] (lower triangular), decode [1,1,1,past+1] (all ones)
- **Bias detection**: some architectures lack Q/K/V biases. Detect from weight keys at load time
