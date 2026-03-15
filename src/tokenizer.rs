use std::collections::HashMap;
use std::io;

use fancy_regex::Regex;

pub struct Tokenizer {
    encoder: HashMap<String, u32>,
    decoder: HashMap<u32, String>,
    bpe_ranks: HashMap<(String, String), usize>,
    byte_encoder: HashMap<u8, char>,
    byte_decoder: HashMap<char, u8>,
    pattern: Regex,
}

/// Replicates the GPT-2 bytes_to_unicode() mapping.
///
/// Printable Latin bytes (!, …, ~), (¡, …, ¬), (®, …, ÿ) map to themselves.
/// The remaining 256 - 188 = 68 bytes map to U+0100 onwards.
fn build_byte_encoder() -> HashMap<u8, char> {
    // These byte values map to themselves (all printable).
    let mut bs: Vec<u8> = (b'!'..=b'~').chain(0xA1u8..=0xACu8).chain(0xAEu8..=0xFFu8).collect();

    // Start counter for unmapped bytes at U+0100.
    let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
    let mut n: u32 = 0;
    for b in 0u8..=255 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }

    bs.into_iter()
        .zip(cs)
        .map(|(b, c)| (b, char::from_u32(c).unwrap()))
        .collect()
}

impl Tokenizer {
    pub fn from_files(vocab_path: &str, merges_path: &str) -> Result<Self, io::Error> {
        // Load vocab.json: {"token": id, ...}
        let vocab_str = std::fs::read_to_string(vocab_path)?;
        let encoder: HashMap<String, u32> = serde_json::from_str(&vocab_str)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let decoder: HashMap<u32, String> =
            encoder.iter().map(|(k, &v)| (v, k.clone())).collect();

        // Load merges.txt: first line is "#version: 0.2", then "tok1 tok2" per line.
        let merges_str = std::fs::read_to_string(merges_path)?;
        let bpe_ranks: HashMap<(String, String), usize> = merges_str
            .lines()
            .filter(|l| !l.starts_with('#') && !l.is_empty())
            .enumerate()
            .map(|(rank, line)| {
                let mut parts = line.splitn(2, ' ');
                let a = parts.next().unwrap_or("").to_string();
                let b = parts.next().unwrap_or("").to_string();
                ((a, b), rank)
            })
            .collect();

        let byte_encoder = build_byte_encoder();
        let byte_decoder: HashMap<char, u8> =
            byte_encoder.iter().map(|(&b, &c)| (c, b)).collect();

        // GPT-2 pre-tokenization regex (uses lookahead, so requires fancy-regex).
        let pattern = Regex::new(
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        )
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

        Ok(Tokenizer {
            encoder,
            decoder,
            bpe_ranks,
            byte_encoder,
            byte_decoder,
            pattern,
        })
    }

    /// Apply BPE merges to a single pre-tokenized word (already byte-encoded).
    fn bpe(&self, token: &str) -> Vec<String> {
        let mut word: Vec<String> = token.chars().map(|c| c.to_string()).collect();
        if word.len() <= 1 {
            return word;
        }

        loop {
            let mut best_pair: Option<usize> = None;
            let mut best_rank = usize::MAX;
            for i in 0..word.len() - 1 {
                let pair = (word[i].clone(), word[i + 1].clone());
                if let Some(&rank) = self.bpe_ranks.get(&pair) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_pair = Some(i);
                    }
                }
            }

            let Some(i) = best_pair else { break };

            let merged = format!("{}{}", word[i], word[i + 1]);
            word[i] = merged;
            word.remove(i + 1);

            if word.len() == 1 {
                break;
            }
        }

        word
    }

    /// Encode a text string into a sequence of GPT-2 token IDs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        let mut ids = Vec::new();

        for mat in self.pattern.find_iter(text) {
            let word_str = match mat {
                Ok(m) => m.as_str(),
                Err(_) => continue,
            };

            // Convert UTF-8 bytes to byte-unicode representation.
            let byte_encoded: String = word_str
                .bytes()
                .map(|b| self.byte_encoder[&b])
                .collect();

            for piece in self.bpe(&byte_encoded) {
                if let Some(&id) = self.encoder.get(&piece) {
                    ids.push(id);
                }
            }
        }

        ids
    }

    /// Decode a sequence of token IDs back to a UTF-8 string.
    pub fn decode(&self, ids: &[u32]) -> String {
        let text: String = ids
            .iter()
            .filter_map(|id| self.decoder.get(id).map(|s| s.as_str()))
            .collect();

        let bytes: Vec<u8> = text
            .chars()
            .filter_map(|c| self.byte_decoder.get(&c).copied())
            .collect();

        String::from_utf8_lossy(&bytes).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load_tokenizer() -> Tokenizer {
        Tokenizer::from_files(
            "tests/fixtures/vocab.json",
            "tests/fixtures/merges.txt",
        )
        .unwrap()
    }

    #[test]
    fn encode_hello_world() {
        let tok = load_tokenizer();
        assert_eq!(tok.encode("Hello world"), vec![15496, 995]);
    }

    #[test]
    fn encode_hello_comma_world() {
        let tok = load_tokenizer();
        assert_eq!(tok.encode("Hello, world!"), vec![15496, 11, 995, 0]);
    }

    #[test]
    fn encode_empty_string() {
        let tok = load_tokenizer();
        assert_eq!(tok.encode(""), Vec::<u32>::new());
    }

    #[test]
    fn decode_round_trip() {
        let tok = load_tokenizer();
        let text = "The quick brown fox jumps over the lazy dog.";
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn decode_single_token() {
        let tok = load_tokenizer();
        assert_eq!(tok.decode(&[15496]), "Hello");
    }

    #[test]
    fn encode_numbers() {
        let tok = load_tokenizer();
        // GPT-2 splits numbers into 1-3 digit chunks
        let ids = tok.encode("12345");
        assert!(!ids.is_empty());
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, "12345");
    }

    #[test]
    fn encode_special_characters() {
        let tok = load_tokenizer();
        let text = "café résumé naïve";
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
    }
}
