# KV Cache Design

The KV cache enables autoregressive decoding without reprocessing the full sequence each step. Homura supports KV caching for both ONNX and HF model backends.

## Overview

```
Prefill:  [token_0, token_1, ..., token_N] -> run() -> logits + KV tensors
                                                         |
                                               init_kv_cache(kv_tensors, max_seq_len)
                                                         |
Decode:   [token_N+1] -> run_kv() -> logits    (KV cache updated internally)
          [token_N+2] -> run_kv() -> logits
          ...
```

## KvCache struct

`KvCache` in `runtime.rs` pre-allocates contiguous buffers of shape `[1, heads, max_seq_len, head_dim]` for each KV layer. There are `num_layers * 2` buffers (one key and one value per layer).

Operations:
- `init_kv_cache(kv_buffers, max_seq_len)` -- copies prefill KV outputs into the pre-allocated buffers
- `append_and_view(data, layer, is_key)` -- appends new KV data and returns a view of the valid region
- `reset()` -- resets the sequence position to 0 (reuses the same allocation)
- `kv_cache_len()` -- returns the current sequence length stored in the cache

### Zero-copy views

KV cache views are implemented as `Buffer::View` -- a borrowed pointer into the pre-allocated buffer with custom strides. The strides use the max_seq_len-based layout so the view can be passed directly to compiled kernels without copying. The `Buffer::View` variant holds a raw pointer that must not outlive the `KvCache`.

## Multi-token append

`run_kv` accepts input sequences longer than 1 token. This enables:
- **Incremental prefill for chat**: each new turn feeds only the delta tokens (new user message + template markup) through `run_kv`, extending the existing KV cache without reprocessing the full conversation
- **Batch decode**: multiple tokens can be appended in a single step if needed

The append operation copies `[1, heads, new_tokens, head_dim]` into the pre-allocated buffer at the current position, then advances the position counter.

## KvPlanInfo

`KvPlanInfo` in the `ExecutionPlan` tracks which buffer slots correspond to KV cache operations:
- `kv_concat_steps` -- indices of `NativeOp::Concat` steps that are KV concats
- `past_kv_slots` -- input buffer slots for past key/value tensors
- `present_kv_slots` -- output buffer slots for present key/value tensors

During `run()` (prefill), KV concats execute as normal concat operations. During `run_kv()` (decode), the plan intercepts these steps and routes them through the `KvCache` instead.

## ONNX KV cache path

For ONNX models (e.g., GPT-2), `split_kv_concat_groups` in `onnx/emitter.rs` identifies KV Concat nodes (axis=-2, 2 inputs) and splits them out of kernel groups via intra-group dependency analysis. These become `NativeOp::Concat` steps with metadata in `KvPlanInfo`.

The `UnifiedKvGenerator` in `kv_generate.rs` wraps a single ONNX model and implements `GenerativeModel`:
- `prefill()` runs the model with empty past_kv inputs, then initializes the KV cache from outputs
- `decode_step()` runs `run_kv` with a single token, advancing the cache

## HF KV cache path

For HF models, the attention layers emit explicit KV concat operations in `hf/emitter.rs`. The same `KvPlanInfo` mechanism applies. `HfGenerationContext` implements `GenerativeModel`:
- `prefill()` detects whether a KV cache already exists (from a prior chat turn) and uses `run_kv` for incremental prefill or `run` for fresh prefill
- `decode_step()` always uses `run_kv`

## Memory layout

Each KV buffer is laid out as `[1, num_kv_heads, max_seq_len, head_dim]` in row-major order. Only the first `seq_len` positions along dim 2 contain valid data. Views expose `[1, num_kv_heads, seq_len, head_dim]` with the same underlying strides, so the memory is contiguous per-head but there is a gap between the valid region and the end of the pre-allocated buffer.

For Qwen2.5-0.5B with 2 KV heads, head_dim=64, and max_seq_len=2048:
- Per-layer KV memory: 2 * 2 * 2048 * 64 * 4 bytes = 2 MB
- Total KV memory (24 layers): ~48 MB
