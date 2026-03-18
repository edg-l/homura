use std::path::Path;

use serde::Deserialize;

/// Detected HuggingFace model architecture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HfModelType {
    Qwen2,
    Unknown(String),
}

/// Qwen2 model configuration (from `config.json`).
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen2Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_eos_token_id")]
    pub eos_token_id: u32,
    #[serde(default = "default_bos_token_id")]
    pub bos_token_id: u32,
}

fn default_rms_norm_eps() -> f64 { 1e-6 }
fn default_rope_theta() -> f64 { 1_000_000.0 }
fn default_max_position_embeddings() -> usize { 32768 }
fn default_eos_token_id() -> u32 { 151643 }
fn default_bos_token_id() -> u32 { 151643 }

impl Qwen2Config {
    /// Per-head dimension: hidden_size / num_attention_heads.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// GQA repeat factor: num_attention_heads / num_key_value_heads.
    pub fn gqa_repeat(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}

/// Detect the model architecture from `config.json`.
pub fn detect_architecture(config_path: &Path) -> Result<HfModelType, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(config_path)?;
    let json: serde_json::Value = serde_json::from_str(&text)?;

    let arch = json
        .get("architectures")
        .and_then(|a| a.as_array())
        .and_then(|a| a.first())
        .and_then(|v| v.as_str())
        .unwrap_or("");

    Ok(match arch {
        "Qwen2ForCausalLM" => HfModelType::Qwen2,
        other => HfModelType::Unknown(other.to_string()),
    })
}

/// Load a Qwen2 config from `config.json`.
pub fn load_qwen2_config(config_path: &Path) -> Result<Qwen2Config, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(config_path)?;
    let config: Qwen2Config = serde_json::from_str(&text)?;
    Ok(config)
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

        let config: Qwen2Config = serde_json::from_str(json).expect("parse failed");
        assert_eq!(config.hidden_size, 896);
        assert_eq!(config.num_attention_heads, 14);
        assert_eq!(config.num_key_value_heads, 2);
        assert_eq!(config.head_dim(), 64);
        assert_eq!(config.gqa_repeat(), 7);
        assert!(config.tie_word_embeddings);
    }

    #[test]
    fn detect_qwen2_architecture() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.json");
        std::fs::write(
            &config_path,
            r#"{"architectures": ["Qwen2ForCausalLM"], "hidden_size": 896,
                "intermediate_size": 4864, "num_attention_heads": 14,
                "num_hidden_layers": 24, "num_key_value_heads": 2, "vocab_size": 151936}"#,
        )
        .unwrap();

        let arch = detect_architecture(&config_path).unwrap();
        assert_eq!(arch, HfModelType::Qwen2);
    }

    #[test]
    fn detect_unknown_architecture() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.json");
        std::fs::write(
            &config_path,
            r#"{"architectures": ["SomeNewModel"]}"#,
        )
        .unwrap();

        let arch = detect_architecture(&config_path).unwrap();
        assert!(matches!(arch, HfModelType::Unknown(_)));
    }
}
