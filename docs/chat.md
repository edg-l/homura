# Chat Mode

Interactive multi-turn chat with persistent KV cache. Launched via:

```sh
homura chat Qwen/Qwen2.5-0.5B
homura chat Qwen/Qwen3-0.6B --think --temperature 0.6
homura chat ./local-model --system "You are a pirate." --max-tokens 500
```

## Architecture

Chat mode connects several subsystems:

```
User input
  -> ChatTemplate.render() (minijinja Jinja2)
  -> delta token computation (skip tokens already in KV cache)
  -> generate_streaming_from_ids (incremental prefill + decode)
  -> HfGenerationContext (GenerativeModel impl)
  -> HfModel.run_kv() (multi-token KV cache append)
  -> streaming stdout with think-block styling
```

### Chat templates

`ChatTemplate` (in `src/hf/chat.rs`) loads the `chat_template` field from `tokenizer_config.json` and renders it via minijinja with `minijinja-contrib` for Python string method compatibility (needed by Qwen3's template which uses `startswith`, `strip`, etc.).

The template receives:
- `messages` -- the full conversation as `Vec<ChatMessage>` (role + content)
- `add_generation_prompt` -- whether to append the assistant turn prefix
- `enable_thinking` -- optional bool controlling `<think>` block emission

### Incremental prefill (delta tokens)

Each turn, the full conversation is rendered through the chat template and tokenized. The tokens that overlap with what is already in the KV cache are skipped -- only the delta (new user message + template markup) is fed through `run_kv`. This avoids reprocessing the entire conversation history.

```
Turn 1: [system + user1 + gen_prompt] -> full prefill -> decode
Turn 2: tokens_in_cache = len(system + user1 + assistant1)
         full_tokens = tokenize(system + user1 + assistant1 + user2 + gen_prompt)
         delta = full_tokens[tokens_in_cache:]  -> run_kv incremental prefill -> decode
```

The `generate_streaming_from_ids` function takes pre-tokenized IDs directly, bypassing the encode-then-decode-then-re-encode cycle that would lose special tokens like `<|im_start|>` and `<|im_end|>`.

### Context overflow

When the conversation approaches `max_seq_len`, the KV cache is reset and the conversation is truncated to just the system prompt + current user message. A warning is printed.

## Think blocks

For reasoning models like Qwen3, the `--think` flag enables thinking output.

### How it works

1. `find_think_tokens` looks up the token IDs for `<think>` and `</think>` in the tokenizer vocabulary
2. These IDs are passed to `ThinkConfig` which controls the streaming output
3. The `<think>` and `</think>` tag tokens are always hidden from output
4. With `--think`: content between tags is shown in gray italic (ANSI `\x1b[90;3m`)
5. Without `--think`: content between tags is hidden entirely, and trailing whitespace after `</think>` is swallowed

### /no_think system prompt

When `--think` is not set and the model has think tokens in its vocabulary, `/no_think` is appended to the system prompt. This is the Qwen3-recommended way to suppress chain-of-thought output. The alternative (`enable_thinking: Some(false)` in the template) inserts an empty `<think></think>` block which can confuse smaller models.

## REPL commands

- `/clear` -- reset conversation and KV cache, re-prefill system prompt
- `/quit` or `/exit` -- exit
- `/help` -- show commands

## Stop token detection

`find_chat_stop_token` reads `eos_token` from `tokenizer_config.json`. For chat models this is typically the turn-end marker (e.g., `<|im_end|>` for Qwen). It tries `<|im_end|>` first (even if `eos_token` is `<|endoftext|>`) since chat models use turn-end markers rather than sequence-end tokens.

## SamplingArgs

The CLI `SamplingArgs` struct is shared between `run` and `chat` commands via clap's `#[command(flatten)]`. It maps directly to `SamplingConfig` in `generate.rs`. Default values:
- temperature: 0.7
- top_p: 0.9
- top_k: 50
- repetition_penalty: 1.1
- min_p, frequency_penalty, presence_penalty: 0 (disabled)
