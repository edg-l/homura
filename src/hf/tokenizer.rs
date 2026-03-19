use std::path::Path;

/// Wrapper around the HuggingFace `tokenizers` crate.
pub struct HfTokenizer {
    inner: tokenizers::Tokenizer,
}

impl HfTokenizer {
    /// Load a tokenizer from a `tokenizer.json` file.
    pub fn from_file(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| format!("failed to load tokenizer from {}: {e}", path.display()))?;
        Ok(HfTokenizer { inner })
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let encoding = self
            .inner
            .encode(text, false)
            .expect("tokenizer encode failed");
        encoding.get_ids().to_vec()
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, ids: &[u32]) -> String {
        self.inner
            .decode(ids, true)
            .expect("tokenizer decode failed")
    }

    /// Encode text with special tokens recognized (e.g. `<|im_end|>`).
    pub fn encode_with_special(&self, text: &str) -> Vec<u32> {
        let encoding = self
            .inner
            .encode(text, true)
            .expect("tokenizer encode failed");
        encoding.get_ids().to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_qwen2_tokenizer() {
        let path = Path::new(concat!(
            env!("HOME"),
            "/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/",
            "snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987/tokenizer.json"
        ));
        if !path.exists() {
            eprintln!("skipping: Qwen2 tokenizer not found at {}", path.display());
            return;
        }

        let tok = HfTokenizer::from_file(path).expect("load failed");
        let ids = tok.encode("Hello world");
        assert!(!ids.is_empty());
        let text = tok.decode(&ids);
        assert_eq!(text, "Hello world");
    }
}
