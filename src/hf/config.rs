use std::path::Path;

use serde::Deserialize;

/// Generic decoder-only transformer configuration.
///
/// Parses any HuggingFace `config.json` — Qwen2, Llama, Mistral, Phi, etc.
/// all share the same field names. Optional fields have sensible defaults.
/// Model-specific quirks (e.g., QKV bias) are inferred from the weights
/// at load time, not from the config.
#[derive(Debug, Clone, Deserialize)]
pub struct TransformerConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    /// Number of KV heads for grouped-query attention.
    /// Defaults to `num_attention_heads` (standard MHA) if absent.
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    pub vocab_size: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub eos_token_id: Option<u32>,
    #[serde(default)]
    pub bos_token_id: Option<u32>,
}

fn default_rms_norm_eps() -> f64 {
    1e-6
}
fn default_rope_theta() -> f64 {
    10_000.0
}
fn default_max_position_embeddings() -> usize {
    2048
}

impl TransformerConfig {
    /// Load from a `config.json` file.
    pub fn load(config_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let text = std::fs::read_to_string(config_path)?;
        let config: TransformerConfig = serde_json::from_str(&text)?;
        Ok(config)
    }

    /// Effective number of KV heads (falls back to num_attention_heads for MHA).
    pub fn kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Per-head dimension: hidden_size / num_attention_heads.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// GQA repeat factor: num_attention_heads / kv_heads.
    pub fn gqa_repeat(&self) -> usize {
        self.num_attention_heads / self.kv_heads()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_qwen2_config() {
        let json = r#"{
            "architectures": ["Qwen2ForCausalLM"],
            "hidden_size": 896,
            "intermediate_size": 4864,
            "num_attention_heads": 14,
            "num_hidden_layers": 24,
            "num_key_value_heads": 2,
            "vocab_size": 151936,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "max_position_embeddings": 32768,
            "tie_word_embeddings": true,
            "eos_token_id": 151643,
            "bos_token_id": 151643
        }"#;

        let config: TransformerConfig = serde_json::from_str(json).expect("parse failed");
        assert_eq!(config.hidden_size, 896);
        assert_eq!(config.num_attention_heads, 14);
        assert_eq!(config.kv_heads(), 2);
        assert_eq!(config.head_dim(), 64);
        assert_eq!(config.gqa_repeat(), 7);
        assert!(config.tie_word_embeddings);
    }

    #[test]
    fn parse_llama_config_no_kv_heads() {
        // Llama-style config without num_key_value_heads → defaults to MHA
        let json = r#"{
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "rope_theta": 10000.0
        }"#;

        let config: TransformerConfig = serde_json::from_str(json).expect("parse failed");
        assert_eq!(config.kv_heads(), 32); // falls back to num_attention_heads
        assert_eq!(config.gqa_repeat(), 1); // no GQA
        assert!(!config.tie_word_embeddings); // default false
        assert_eq!(config.rope_theta, 10_000.0);
    }

    #[test]
    fn load_real_qwen2_config() {
        let path = std::path::Path::new(concat!(
            env!("HOME"),
            "/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/",
            "snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987/config.json"
        ));
        if !path.exists() {
            eprintln!("skipping: Qwen2.5-0.5B config not found");
            return;
        }

        let config = TransformerConfig::load(path).expect("load failed");
        assert_eq!(config.hidden_size, 896);
        assert_eq!(config.num_hidden_layers, 24);
        assert_eq!(config.kv_heads(), 2);
        assert_eq!(config.head_dim(), 64);
        assert_eq!(config.gqa_repeat(), 7);
    }
}
