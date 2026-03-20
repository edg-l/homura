//! Chat template rendering via minijinja.
//!
//! Loads the Jinja2 `chat_template` from `tokenizer_config.json` and renders
//! conversation messages into the model's expected format.

use std::path::Path;

use minijinja::Environment;

/// A single message in the conversation.
#[derive(Clone, Debug, serde::Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Chat template loaded from tokenizer_config.json.
pub struct ChatTemplate {
    template_source: String,
    /// Special tokens passed as template variables (eos_token, bos_token, etc.)
    special_tokens: std::collections::HashMap<String, String>,
}

impl ChatTemplate {
    /// Load chat_template from tokenizer_config.json.
    pub fn load(model_dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let config_path = model_dir.join("tokenizer_config.json");
        let text = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("failed to read tokenizer_config.json: {e}"))?;
        let config: serde_json::Value = serde_json::from_str(&text)?;

        let template_source = config
            .get("chat_template")
            .and_then(|v| v.as_str())
            .ok_or("no chat_template field in tokenizer_config.json")?
            .to_string();

        // Collect special token strings that templates may reference.
        let token_keys = ["eos_token", "bos_token", "unk_token", "pad_token"];
        let mut special_tokens = std::collections::HashMap::new();
        for key in &token_keys {
            if let Some(val) = config.get(*key) {
                // Some configs store tokens as strings, others as {"content": "..."} objects.
                let token_str = val.as_str().map(|s| s.to_string()).or_else(|| {
                    val.get("content")
                        .and_then(|c| c.as_str())
                        .map(|s| s.to_string())
                });
                if let Some(s) = token_str {
                    special_tokens.insert(key.to_string(), s);
                }
            }
        }

        Ok(ChatTemplate {
            template_source,
            special_tokens,
        })
    }

    /// Render the full conversation to a string.
    ///
    /// If `add_generation_prompt` is true, appends the assistant turn prefix
    /// (e.g. `<|im_start|>assistant\n`) so the model can start generating.
    /// Render the full conversation to a string.
    ///
    /// If `add_generation_prompt` is true, appends the assistant turn prefix
    /// (e.g. `<|im_start|>assistant\n`) so the model can start generating.
    ///
    /// `enable_thinking` controls whether thinking/reasoning models (e.g. Qwen3)
    /// emit `<think>` blocks. Set to false to disable chain-of-thought output.
    /// Render the conversation.
    ///
    /// `enable_thinking`:
    /// - `Some(true)` — model thinks freely (Qwen3 `<think>` blocks)
    /// - `Some(false)` — template inserts empty `<think></think>` to suppress thinking
    /// - `None` — variable is undefined in the template (no think block inserted at all)
    ///
    /// For non-thinking mode, prefer `None` + `/no_think` in the system prompt
    /// over `Some(false)`, which can confuse small models.
    pub fn render(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        enable_thinking: Option<bool>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let mut env = Environment::new();
        // Python string method compatibility (startswith, endswith, strip, split, etc.)
        // needed by HF chat templates (especially Qwen3).
        minijinja_contrib::add_to_environment(&mut env);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_template("chat", &self.template_source)?;

        let tmpl = env.get_template("chat")?;

        // Build context map with messages, flags, and special tokens.
        let mut ctx = std::collections::BTreeMap::<String, minijinja::value::Value>::new();
        ctx.insert(
            "messages".into(),
            minijinja::value::Value::from_serialize(messages),
        );
        ctx.insert("add_generation_prompt".into(), add_generation_prompt.into());
        for (key, val) in &self.special_tokens {
            ctx.insert(key.clone(), val.as_str().into());
        }
        if let Some(val) = enable_thinking {
            ctx.insert("enable_thinking".into(), val.into());
        }

        let rendered = tmpl.render(ctx)?;

        Ok(rendered)
    }
}

/// Find the `<think>` and `</think>` token IDs for thinking models.
///
/// Returns `Some((open_id, close_id))` if both tokens exist in the vocabulary.
pub fn find_think_tokens(tokenizer: &super::tokenizer::HfTokenizer) -> Option<(u32, u32)> {
    let open_ids = tokenizer.encode_with_special("<think>");
    let close_ids = tokenizer.encode_with_special("</think>");
    if open_ids.len() == 1 && close_ids.len() == 1 {
        Some((open_ids[0], close_ids[0]))
    } else {
        None
    }
}

/// Find the token ID for the chat turn-end token.
///
/// Tries `eos_token` from tokenizer_config.json first (handles both string
/// and `{"content": "..."}` formats). Falls back to `<|im_end|>` for models
/// that use ChatML but set eos_token to something else (e.g. `<|endoftext|>`).
pub fn find_chat_stop_token(
    model_dir: &Path,
    tokenizer: &super::tokenizer::HfTokenizer,
) -> Option<u32> {
    let config_path = model_dir.join("tokenizer_config.json");
    let text = std::fs::read_to_string(&config_path).ok()?;
    let config: serde_json::Value = serde_json::from_str(&text).ok()?;

    let resolve = |token: &str| -> Option<u32> {
        let ids = tokenizer.encode_with_special(token);
        if ids.len() == 1 { Some(ids[0]) } else { None }
    };

    // Read eos_token (string or {"content": "..."} object).
    let eos = config.get("eos_token").and_then(|v| {
        v.as_str()
            .map(|s| s.to_string())
            .or_else(|| v.get("content").and_then(|c| c.as_str()).map(|s| s.to_string()))
    });

    // Use eos_token directly -- this covers </s>, <|im_end|>, <|endoftext|>, etc.
    if let Some(ref eos) = eos {
        if let Some(id) = resolve(eos) {
            return Some(id);
        }
    }

    // Fallback: try <|im_end|> for ChatML models where eos_token is different.
    resolve("<|im_end|>")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_chatml_basic() {
        // Minimal ChatML template for testing.
        let template = ChatTemplate {
            template_source: concat!(
                "{%- for message in messages %}",
                "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}",
                "{%- endfor %}",
                "{%- if add_generation_prompt %}",
                "{{- '<|im_start|>assistant\\n' }}",
                "{%- endif %}",
            )
            .to_string(),
            special_tokens: std::collections::HashMap::new(),
        };

        let messages = vec![
            ChatMessage {
                role: "system".into(),
                content: "You are helpful.".into(),
            },
            ChatMessage {
                role: "user".into(),
                content: "Hello".into(),
            },
        ];

        let rendered = template.render(&messages, true, None).unwrap();
        assert!(rendered.contains("<|im_start|>system\nYou are helpful.<|im_end|>"));
        assert!(rendered.contains("<|im_start|>user\nHello<|im_end|>"));
        assert!(rendered.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn render_without_generation_prompt() {
        let template = ChatTemplate {
            template_source: concat!(
                "{%- for message in messages %}",
                "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}",
                "{%- endfor %}",
                "{%- if add_generation_prompt %}",
                "{{- '<|im_start|>assistant\\n' }}",
                "{%- endif %}",
            )
            .to_string(),
            special_tokens: std::collections::HashMap::new(),
        };

        let messages = vec![ChatMessage {
            role: "user".into(),
            content: "Hi".into(),
        }];

        let rendered = template.render(&messages, false, None).unwrap();
        assert!(!rendered.contains("<|im_start|>assistant"));
    }
}
