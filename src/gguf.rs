//! GGUF v3 file parser.
//!
//! Parses the GGUF binary format (used by llama.cpp / GGML) via memory-mapping.
//! Provides access to metadata key-value pairs and raw tensor data bytes.
//!
//! Reference: <https://github.com/ggerganov/ggml/blob/master/docs/gguf.md>

use std::collections::HashMap;
use std::path::Path;

use memmap2::Mmap;

use crate::DType;

// ── GGUF constants ───────────────────────────────────────────────────────────

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" as little-endian u32 (bytes: 47 47 55 46)
const GGUF_DEFAULT_ALIGNMENT: usize = 32;

// ── GGML dtype codes ─────────────────────────────────────────────────────────

/// GGML tensor type codes (from ggml.h).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    IQ2XXS = 16,
    IQ2XS = 17,
    IQ3XXS = 18,
    IQ1S = 19,
    IQ4NL = 20,
    IQ3S = 21,
    IQ2S = 22,
    IQ4XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1M = 29,
    BF16 = 30,
}

impl GgmlType {
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2K),
            11 => Some(Self::Q3K),
            12 => Some(Self::Q4K),
            13 => Some(Self::Q5K),
            14 => Some(Self::Q6K),
            15 => Some(Self::Q8K),
            16 => Some(Self::IQ2XXS),
            17 => Some(Self::IQ2XS),
            18 => Some(Self::IQ3XXS),
            19 => Some(Self::IQ1S),
            20 => Some(Self::IQ4NL),
            21 => Some(Self::IQ3S),
            22 => Some(Self::IQ2S),
            23 => Some(Self::IQ4XS),
            24 => Some(Self::I8),
            25 => Some(Self::I16),
            26 => Some(Self::I32),
            27 => Some(Self::I64),
            28 => Some(Self::F64),
            29 => Some(Self::IQ1M),
            30 => Some(Self::BF16),
            _ => None,
        }
    }

    /// Convert to Homura DType. Returns None for unsupported quant types.
    pub fn to_dtype(self) -> Option<DType> {
        match self {
            Self::F32 => Some(DType::F32),
            Self::F16 => Some(DType::F16),
            Self::BF16 => Some(DType::BF16),
            Self::F64 => Some(DType::F64),
            Self::I8 => Some(DType::I8),
            Self::I16 => Some(DType::I16),
            Self::I32 => Some(DType::I32),
            Self::I64 => Some(DType::I64),
            Self::Q8_0 => Some(DType::Q8_0),
            Self::Q4K => Some(DType::Q4_K),
            Self::Q6K => Some(DType::Q6_K),
            _ => None,
        }
    }

    /// Number of logical elements per block.
    fn block_size(self) -> usize {
        match self {
            Self::F32 | Self::F64 | Self::I8 | Self::I16 | Self::I32 | Self::I64 => 1,
            Self::F16 | Self::BF16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 => 32,
            Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => 256,
            Self::IQ2XXS
            | Self::IQ2XS
            | Self::IQ3XXS
            | Self::IQ1S
            | Self::IQ4NL
            | Self::IQ3S
            | Self::IQ2S
            | Self::IQ4XS
            | Self::IQ1M => 256,
        }
    }

    /// Byte size per block.
    fn type_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F64 => 8,
            Self::F16 | Self::BF16 => 2,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::Q4_0 => 18, // 2 (f16 scale) + 16 (32 nibbles)
            Self::Q4_1 => 20, // 2 (f16 scale) + 2 (f16 min) + 16
            Self::Q5_0 => 22, // 2 + 4 (high bits) + 16
            Self::Q5_1 => 24, // 2 + 2 + 4 + 16
            Self::Q8_0 => 34, // 2 (f16 scale) + 32 (i8 quants)
            Self::Q8_1 => 40, // 2 + 2 + 32 + 4 (sum)
            Self::Q2K => 256 / 16 + 256 / 4 + 2 + 2, // 84
            Self::Q3K => 256 / 8 + 256 / 4 + 12 + 2, // 110
            Self::Q4K => 2 + 2 + 12 + 256 / 2, // 144
            Self::Q5K => 2 + 2 + 12 + 256 / 8 + 256 / 2, // 176
            Self::Q6K => 256 / 2 + 256 / 4 + 256 / 16 + 2, // 210
            Self::Q8K => 4 + 256 + 16 * 2, // 292
            // IQ types: use reference values from ggml
            Self::IQ2XXS => 66,
            Self::IQ2XS => 74,
            Self::IQ3XXS => 98,
            Self::IQ1S => 50,
            Self::IQ4NL => 18, // same as Q4_0
            Self::IQ3S => 110,
            Self::IQ2S => 82,
            Self::IQ4XS => 36,
            Self::IQ1M => 56,
        }
    }

    /// Total bytes for `num_elements` values.
    pub fn byte_size_for_elements(self, num_elements: usize) -> usize {
        let bs = self.block_size();
        assert!(
            num_elements % bs == 0,
            "element count {} not a multiple of block_size {} for {:?}",
            num_elements,
            bs,
            self
        );
        (num_elements / bs) * self.type_size()
    }
}

// ── Metadata value types ─────────────────────────────────────────────────────

/// GGUF metadata value.
#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            GgufValue::U32(v) => Some(*v),
            GgufValue::I32(v) => Some(*v as u32),
            GgufValue::U64(v) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            GgufValue::U64(v) => Some(*v),
            GgufValue::U32(v) => Some(*v as u64),
            GgufValue::I32(v) => Some(*v as u64),
            GgufValue::I64(v) => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            GgufValue::F32(v) => Some(*v),
            GgufValue::F64(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_string_array(&self) -> Option<Vec<&str>> {
        match self {
            GgufValue::Array(arr) => {
                let mut out = Vec::with_capacity(arr.len());
                for v in arr {
                    out.push(v.as_str()?);
                }
                Some(out)
            }
            _ => None,
        }
    }

    pub fn as_f32_array(&self) -> Option<Vec<f32>> {
        match self {
            GgufValue::Array(arr) => {
                let mut out = Vec::with_capacity(arr.len());
                for v in arr {
                    out.push(v.as_f32()?);
                }
                Some(out)
            }
            _ => None,
        }
    }
}

// ── GGUF metadata value type codes ───────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
#[repr(u32)]
enum GgufMetaType {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    F32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
}

impl GgufMetaType {
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::U8),
            1 => Some(Self::I8),
            2 => Some(Self::U16),
            3 => Some(Self::I16),
            4 => Some(Self::U32),
            5 => Some(Self::I32),
            6 => Some(Self::F32),
            7 => Some(Self::Bool),
            8 => Some(Self::String),
            9 => Some(Self::Array),
            10 => Some(Self::U64),
            11 => Some(Self::I64),
            12 => Some(Self::F64),
            _ => None,
        }
    }
}

// ── Tensor info ──────────────────────────────────────────────────────────────

/// Metadata for a single tensor in the GGUF file.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    /// Logical shape (outermost-first, same convention as NumPy/PyTorch).
    /// GGUF stores dims innermost-first, so we reverse during parsing.
    pub shape: Vec<u64>,
    pub ggml_type: GgmlType,
    /// Byte offset within the tensor data region (after alignment).
    pub offset: u64,
}

impl GgufTensorInfo {
    /// Total number of logical elements.
    pub fn num_elements(&self) -> u64 {
        self.shape.iter().product()
    }

    /// Total byte size of the tensor's raw data.
    pub fn byte_size(&self) -> usize {
        self.ggml_type
            .byte_size_for_elements(self.num_elements() as usize)
    }
}

// ── GgufFile ─────────────────────────────────────────────────────────────────

/// A parsed GGUF file backed by a memory-mapped region.
pub struct GgufFile {
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<GgufTensorInfo>,
    tensor_by_name: HashMap<String, usize>,
    mmap: Mmap,
    /// Byte offset where tensor data starts (aligned).
    data_offset: usize,
    pub version: u32,
}

impl GgufFile {
    /// Parse a GGUF file from disk.
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Hint the OS to use huge pages for the mmap'd weight data.
        // Reduces TLB misses when threads stride across large weight buffers.
        #[cfg(target_os = "linux")]
        unsafe {
            libc::madvise(
                mmap.as_ptr() as *mut libc::c_void,
                mmap.len(),
                libc::MADV_HUGEPAGE,
            );
        }

        if mmap.len() < 16 {
            return Err("GGUF file too small".into());
        }

        let mut cursor = Cursor::new(&mmap);

        // Header
        let magic = cursor.read_u32()?;
        if magic != GGUF_MAGIC {
            return Err(format!(
                "invalid GGUF magic: 0x{:08X} (expected 0x{:08X})",
                magic, GGUF_MAGIC
            )
            .into());
        }

        let version = cursor.read_u32()?;
        if version < 2 || version > 3 {
            return Err(format!("unsupported GGUF version: {version} (expected 2 or 3)").into());
        }

        // v2 uses u32, v3 uses u64 for counts
        let tensor_count = if version >= 3 {
            cursor.read_u64()?
        } else {
            cursor.read_u32()? as u64
        };
        let metadata_kv_count = if version >= 3 {
            cursor.read_u64()?
        } else {
            cursor.read_u32()? as u64
        };

        // Parse metadata
        let mut metadata = HashMap::with_capacity(metadata_kv_count as usize);
        for _ in 0..metadata_kv_count {
            let key = cursor.read_gguf_string()?;
            let value = cursor.read_gguf_value()?;
            metadata.insert(key, value);
        }

        // Determine alignment
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(GGUF_DEFAULT_ALIGNMENT as u64) as usize;

        // Parse tensor infos
        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = cursor.read_gguf_string()?;
            let n_dims = cursor.read_u32()? as usize;
            // GGUF stores dims innermost-first; reverse to outermost-first.
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(cursor.read_u64()?);
            }
            dims.reverse();

            let type_code = cursor.read_u32()?;
            let ggml_type = GgmlType::from_u32(type_code)
                .ok_or_else(|| format!("unknown GGML type code {type_code} for tensor '{name}'"))?;
            let offset = cursor.read_u64()?;

            tensors.push(GgufTensorInfo {
                name,
                shape: dims,
                ggml_type,
                offset,
            });
        }

        // Tensor data starts at the next aligned boundary after all header/metadata/tensor-infos.
        let data_offset = align_up(cursor.pos, alignment);

        // Build name -> index map
        let tensor_by_name: HashMap<String, usize> = tensors
            .iter()
            .enumerate()
            .map(|(i, t)| (t.name.clone(), i))
            .collect();

        Ok(Self {
            metadata,
            tensors,
            tensor_by_name,
            mmap,
            data_offset,
            version,
        })
    }

    /// Get tensor info by name.
    pub fn tensor_info(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensor_by_name.get(name).map(|&i| &self.tensors[i])
    }

    /// Get raw bytes for a tensor (no dequantization).
    pub fn tensor_data(&self, info: &GgufTensorInfo) -> &[u8] {
        let start = self.data_offset + info.offset as usize;
        let end = start + info.byte_size();
        &self.mmap[start..end]
    }

    /// Get a metadata value.
    pub fn meta(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }

    /// Get the model architecture string (e.g. "llama", "qwen2").
    pub fn architecture(&self) -> Option<&str> {
        self.meta("general.architecture")?.as_str()
    }

    /// Print a summary of the file.
    pub fn summary(&self) {
        println!("GGUF v{}", self.version);
        if let Some(arch) = self.architecture() {
            println!("Architecture: {arch}");
        }
        if let Some(name) = self.meta("general.name").and_then(|v| v.as_str()) {
            println!("Model name: {name}");
        }
        println!("Tensors: {}", self.tensors.len());
        println!("Metadata keys: {}", self.metadata.len());

        // Count tensors by type
        let mut type_counts: HashMap<GgmlType, usize> = HashMap::new();
        for t in &self.tensors {
            *type_counts.entry(t.ggml_type).or_default() += 1;
        }
        for (ty, count) in &type_counts {
            println!("  {:?}: {} tensors", ty, count);
        }
    }
}

// ── Config extraction ────────────────────────────────────────────────────────

use crate::hf::config::TransformerConfig;
use crate::hf::weights::{LayerWeights, TransformerWeights};
use crate::runtime::Buffer;

impl GgufFile {
    /// Build a `TransformerConfig` from GGUF metadata.
    ///
    /// GGUF encodes model config under `<arch>.*` metadata keys
    /// (e.g. `qwen2.embedding_length`, `llama.block_count`).
    pub fn transformer_config(&self) -> Result<TransformerConfig, Box<dyn std::error::Error>> {
        let arch = self
            .architecture()
            .ok_or("missing general.architecture metadata")?;

        let get_u64 = |key: &str| -> Result<u64, Box<dyn std::error::Error>> {
            self.meta(key)
                .and_then(|v| v.as_u64())
                .ok_or_else(|| format!("missing or invalid metadata: {key}").into())
        };
        let get_f64 = |key: &str| -> Result<f64, Box<dyn std::error::Error>> {
            self.meta(key)
                .and_then(|v| v.as_f32())
                .map(|v| v as f64)
                .ok_or_else(|| format!("missing or invalid metadata: {key}").into())
        };

        let hidden_size = get_u64(&format!("{arch}.embedding_length"))? as usize;
        let intermediate_size = get_u64(&format!("{arch}.feed_forward_length"))? as usize;
        let num_attention_heads = get_u64(&format!("{arch}.attention.head_count"))? as usize;
        let num_hidden_layers = get_u64(&format!("{arch}.block_count"))? as usize;
        let num_key_value_heads = get_u64(&format!("{arch}.attention.head_count_kv"))
            .ok()
            .map(|v| v as usize);

        // vocab_size: try metadata first, fall back to tokenizer token count
        let vocab_size = get_u64(&format!("{arch}.vocab_size")).or_else(|_| {
            self.meta("tokenizer.ggml.tokens")
                .and_then(|v| match v {
                    GgufValue::Array(arr) => Some(arr.len() as u64),
                    _ => None,
                })
                .ok_or_else(|| -> Box<dyn std::error::Error> {
                    "cannot determine vocab_size".into()
                })
        })? as usize;

        let rms_norm_eps =
            get_f64(&format!("{arch}.attention.layer_norm_rms_epsilon")).unwrap_or(1e-6);
        let rope_theta = get_f64(&format!("{arch}.rope.freq_base")).unwrap_or(10_000.0);
        let max_position_embeddings =
            get_u64(&format!("{arch}.context_length")).unwrap_or(2048) as usize;

        // Detect tied embeddings: if output.weight tensor is absent, embeddings are tied.
        let tie_word_embeddings = self.tensor_info("output.weight").is_none();

        // Explicit head_dim (Qwen3 uses this)
        let head_dim = get_u64(&format!("{arch}.attention.key_length"))
            .ok()
            .map(|v| v as usize);

        // EOS/BOS token IDs
        let eos_token_id = self
            .meta("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u32());
        let bos_token_id = self
            .meta("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_u32());

        Ok(TransformerConfig {
            hidden_size,
            intermediate_size,
            num_attention_heads,
            num_hidden_layers,
            num_key_value_heads,
            vocab_size,
            rms_norm_eps,
            rope_theta,
            max_position_embeddings,
            tie_word_embeddings,
            head_dim,
            eos_token_id,
            bos_token_id,
        })
    }

    /// Load transformer weights from GGUF tensors.
    ///
    /// Quantized projection weights are loaded as raw packed bytes (no transpose,
    /// no dequantization). Non-projection weights (norms, embeddings) are
    /// dequantized to f32 since they are used in elementwise ops.
    pub fn load_transformer_weights(
        &self,
        config: &TransformerConfig,
    ) -> Result<TransformerWeights, Box<dyn std::error::Error>> {
        let take = |name: &str| -> Result<Buffer, Box<dyn std::error::Error>> {
            let info = self
                .tensor_info(name)
                .ok_or_else(|| format!("missing GGUF tensor: {name}"))?;
            let data = self.tensor_data(info);
            if let Some(dtype) = info.ggml_type.to_dtype() {
                if dtype.is_quantized() {
                    // Quantized: keep raw bytes, store with logical shape
                    Ok(Buffer::from_raw_bytes(data.to_vec(), &info.shape, dtype))
                } else {
                    // Non-quantized: load as typed buffer
                    self.load_typed_buffer(info)
                }
            } else {
                Err(format!(
                    "unsupported GGML type {:?} for tensor '{name}'",
                    info.ggml_type
                )
                .into())
            }
        };

        let take_f32 = |name: &str| -> Result<Buffer, Box<dyn std::error::Error>> {
            let info = self
                .tensor_info(name)
                .ok_or_else(|| format!("missing GGUF tensor: {name}"))?;
            self.load_f32_buffer(info)
        };

        // Embedding: always f32 (used in gather, not matmul)
        let embed_tokens_weight = take_f32("token_embd.weight")?;

        // Detect features from first layer
        let has_bias = self.tensor_info("blk.0.attn_q.bias").is_some();
        let has_qk_norm = self.tensor_info("blk.0.attn_q_norm.weight").is_some();

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let pb = crate::progress::load_progress(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let input_layernorm_weight = take_f32(&format!("blk.{i}.attn_norm.weight"))?;

            // Projection weights: keep quantized (no transpose for quantized)
            let q_proj_weight = take(&format!("blk.{i}.attn_q.weight"))?;
            let q_proj_bias = if has_bias {
                Some(take_f32(&format!("blk.{i}.attn_q.bias"))?)
            } else {
                None
            };
            let k_proj_weight = take(&format!("blk.{i}.attn_k.weight"))?;
            let k_proj_bias = if has_bias {
                Some(take_f32(&format!("blk.{i}.attn_k.bias"))?)
            } else {
                None
            };
            let v_proj_weight = take(&format!("blk.{i}.attn_v.weight"))?;
            let v_proj_bias = if has_bias {
                Some(take_f32(&format!("blk.{i}.attn_v.bias"))?)
            } else {
                None
            };

            let q_norm_weight = if has_qk_norm {
                Some(take_f32(&format!("blk.{i}.attn_q_norm.weight"))?)
            } else {
                None
            };
            let k_norm_weight = if has_qk_norm {
                Some(take_f32(&format!("blk.{i}.attn_k_norm.weight"))?)
            } else {
                None
            };

            let o_proj_weight = take(&format!("blk.{i}.attn_output.weight"))?;

            let post_attention_layernorm_weight = take_f32(&format!("blk.{i}.ffn_norm.weight"))?;

            let gate_proj_weight = take(&format!("blk.{i}.ffn_gate.weight"))?;
            let up_proj_weight = take(&format!("blk.{i}.ffn_up.weight"))?;
            let down_proj_weight = take(&format!("blk.{i}.ffn_down.weight"))?;

            crate::progress::update_load(&pb, i + 1, &format!("layer {i}"));
            layers.push(LayerWeights {
                input_layernorm_weight,
                q_proj_weight,
                q_proj_bias,
                k_proj_weight,
                k_proj_bias,
                v_proj_weight,
                v_proj_bias,
                q_norm_weight,
                k_norm_weight,
                o_proj_weight,
                post_attention_layernorm_weight,
                gate_proj_weight,
                up_proj_weight,
                down_proj_weight,
            });
        }
        crate::progress::finish_load(&pb);

        let final_norm_weight = take_f32("output_norm.weight")?;

        let lm_head_weight = if config.tie_word_embeddings {
            None
        } else {
            // Always dequant to f32 — the LM head uses a standard matmul, not a quant kernel
            Some(take_f32("output.weight")?)
        };

        Ok(TransformerWeights {
            embed_tokens_weight,
            layers,
            final_norm_weight,
            lm_head_weight,
        })
    }

    /// Load a tensor as a typed (non-quantized) Buffer.
    fn load_typed_buffer(
        &self,
        info: &GgufTensorInfo,
    ) -> Result<Buffer, Box<dyn std::error::Error>> {
        let data = self.tensor_data(info);
        let dtype = info
            .ggml_type
            .to_dtype()
            .ok_or_else(|| format!("unsupported type {:?}", info.ggml_type))?;
        Ok(Buffer::from_raw_bytes(data.to_vec(), &info.shape, dtype))
    }

    /// Load a tensor and convert to f32 (for norms, embeddings, biases).
    ///
    /// Handles dequantization of Q8_0 tensors (e.g. embeddings that are
    /// quantized in the GGUF but need f32 for gather ops).
    fn load_f32_buffer(&self, info: &GgufTensorInfo) -> Result<Buffer, Box<dyn std::error::Error>> {
        let data = self.tensor_data(info);
        match info.ggml_type {
            GgmlType::F32 => Ok(Buffer::from_raw_bytes(
                data.to_vec(),
                &info.shape,
                DType::F32,
            )),
            GgmlType::F16 => {
                let num_elements = info.num_elements() as usize;
                let mut f32_data = Vec::with_capacity(num_elements);
                for i in 0..num_elements {
                    let bits = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
                    f32_data.push(f16_to_f32(bits));
                }
                Ok(Buffer::from_slice(&f32_data, &info.shape, DType::F32))
            }
            GgmlType::BF16 => {
                let num_elements = info.num_elements() as usize;
                let mut f32_data = Vec::with_capacity(num_elements);
                for i in 0..num_elements {
                    let bits = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
                    f32_data.push(f32::from_bits((bits as u32) << 16));
                }
                Ok(Buffer::from_slice(&f32_data, &info.shape, DType::F32))
            }
            GgmlType::Q8_0 => {
                let num_elements = info.num_elements() as usize;
                let mut f32_data = vec![0.0f32; num_elements];
                dequant_q8_0(data, &mut f32_data);
                Ok(Buffer::from_slice(&f32_data, &info.shape, DType::F32))
            }
            GgmlType::Q4K => {
                let num_elements = info.num_elements() as usize;
                let mut f32_data = vec![0.0f32; num_elements];
                dequant_q4_k(data, &mut f32_data);
                Ok(Buffer::from_slice(&f32_data, &info.shape, DType::F32))
            }
            GgmlType::Q6K => {
                let num_elements = info.num_elements() as usize;
                let mut f32_data = vec![0.0f32; num_elements];
                dequant_q6_k(data, &mut f32_data);
                Ok(Buffer::from_slice(&f32_data, &info.shape, DType::F32))
            }
            _ => Err(format!(
                "load_f32_buffer: unsupported type {:?} for tensor '{}'",
                info.ggml_type, info.name
            )
            .into()),
        }
    }
}

/// Convert IEEE 754 half-precision (f16) bits to f32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;

    if exp == 0 {
        if frac == 0 {
            // Zero
            f32::from_bits(sign << 31)
        } else {
            // Subnormal f16 -> normal f32
            let mut e = 0i32;
            let mut f = frac;
            while (f & 0x400) == 0 {
                f <<= 1;
                e -= 1;
            }
            f &= 0x3FF;
            let exp32 = (127 - 15 + 1 + e) as u32;
            f32::from_bits((sign << 31) | (exp32 << 23) | (f << 13))
        }
    } else if exp == 31 {
        // Inf or NaN
        f32::from_bits((sign << 31) | (0xFF << 23) | (frac << 13))
    } else {
        // Normal
        let exp32 = (exp as i32 - 15 + 127) as u32;
        f32::from_bits((sign << 31) | (exp32 << 23) | (frac << 13))
    }
}

// ── Dequantization (CPU, for non-matmul tensors like embeddings) ─────────────

/// Dequantize Q8_0 blocks to f32.
/// Q8_0: 32-element blocks, each = 2-byte f16 scale + 32 i8 quants.
/// dequant: x[i] = qs[i] * d
fn dequant_q8_0(src: &[u8], dst: &mut [f32]) {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 34;
    let num_blocks = src.len() / BLOCK_BYTES;
    assert_eq!(dst.len(), num_blocks * BLOCK_SIZE);

    for b in 0..num_blocks {
        let block = &src[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for j in 0..BLOCK_SIZE {
            let q = block[2 + j] as i8;
            dst[b * BLOCK_SIZE + j] = q as f32 * scale;
        }
    }
}

/// Dequantize Q4_K blocks to f32.
/// Q4_K: 256-element super-blocks (144 bytes). 8 sub-blocks of 32 elements.
pub(crate) fn dequant_q4_k(src: &[u8], dst: &mut [f32]) {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 144;
    let num_blocks = src.len() / BLOCK_BYTES;
    assert_eq!(dst.len(), num_blocks * BLOCK_SIZE);

    for b in 0..num_blocks {
        let block = &src[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales_raw = &block[4..16]; // 12 bytes of packed scales
        let qs = &block[16..144]; // 128 bytes of 4-bit quants

        // Unpack 6-bit scales and mins for 8 sub-blocks
        let mut sc = [0u8; 8];
        let mut mn = [0u8; 8];
        for i in 0..4 {
            sc[i] = scales_raw[i] & 0x3F;
            mn[i] = scales_raw[4 + i] & 0x3F;
        }
        for i in 0..4 {
            // Match llama.cpp get_scale_min_k4: for j>=4,
            //   scale = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)
            //   min   = (q[j+4] >> 4)  | ((q[j]   >> 6) << 4)
            // q[j+4] = scales_raw[8+i], low nibble=scale, high nibble=min
            sc[4 + i] = (scales_raw[8 + i] & 0x0F) | ((scales_raw[i] >> 6) << 4);
            mn[4 + i] = (scales_raw[8 + i] >> 4) | ((scales_raw[4 + i] >> 6) << 4);
        }

        // Match llama.cpp dequantize_row_q4_K: sub-blocks come in pairs
        // sharing 32 qs bytes. Even sub-block = low nibbles, odd = high nibbles.
        for pair in 0..4 {
            let scale0 = d * sc[2 * pair] as f32;
            let min0 = dmin * mn[2 * pair] as f32;
            let scale1 = d * sc[2 * pair + 1] as f32;
            let min1 = dmin * mn[2 * pair + 1] as f32;
            let q = &qs[pair * 32..(pair + 1) * 32];
            let base = pair * 64;
            for l in 0..32 {
                dst[b * BLOCK_SIZE + base + l] = (q[l] & 0x0F) as f32 * scale0 - min0;
            }
            for l in 0..32 {
                dst[b * BLOCK_SIZE + base + 32 + l] = (q[l] >> 4) as f32 * scale1 - min1;
            }
        }
    }
}

/// Dequantize Q6_K blocks to f32.
/// Q6_K: 256-element super-blocks (210 bytes). 16 sub-blocks of 16 elements.
/// Matches llama.cpp dequantize_row_q6_K layout.
pub(crate) fn dequant_q6_k(src: &[u8], dst: &mut [f32]) {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 210;
    let num_blocks = src.len() / BLOCK_BYTES;
    assert_eq!(dst.len(), num_blocks * BLOCK_SIZE);

    for b in 0..num_blocks {
        let block = &src[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
        let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));

        // Two chunks of 128 elements: ql advances by 64, qh by 32, sc by 8.
        for chunk in 0..2usize {
            let ql = &block[chunk * 64..];
            let qh = &block[128 + chunk * 32..];
            let sc = &block[192 + chunk * 8..];
            let base = b * BLOCK_SIZE + chunk * 128;

            for l in 0..32usize {
                let is = l / 16; // 0 for l=0..15, 1 for l=16..31
                let q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) as i8 - 32;
                let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;

                dst[base + l] = d * (sc[is] as i8) as f32 * q1 as f32;
                dst[base + l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                dst[base + l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                dst[base + l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
            }
        }
    }
}

// ── Cursor helper ────────────────────────────────────────────────────────────

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], Box<dyn std::error::Error>> {
        if self.remaining() < n {
            return Err(format!(
                "GGUF: unexpected EOF at offset {} (need {} bytes, {} remaining)",
                self.pos,
                n,
                self.remaining()
            )
            .into());
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8, Box<dyn std::error::Error>> {
        Ok(self.read_bytes(1)?[0])
    }

    fn read_i8(&mut self) -> Result<i8, Box<dyn std::error::Error>> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16, Box<dyn std::error::Error>> {
        Ok(u16::from_le_bytes(self.read_bytes(2)?.try_into().unwrap()))
    }

    fn read_i16(&mut self) -> Result<i16, Box<dyn std::error::Error>> {
        Ok(i16::from_le_bytes(self.read_bytes(2)?.try_into().unwrap()))
    }

    fn read_u32(&mut self) -> Result<u32, Box<dyn std::error::Error>> {
        Ok(u32::from_le_bytes(self.read_bytes(4)?.try_into().unwrap()))
    }

    fn read_i32(&mut self) -> Result<i32, Box<dyn std::error::Error>> {
        Ok(i32::from_le_bytes(self.read_bytes(4)?.try_into().unwrap()))
    }

    fn read_u64(&mut self) -> Result<u64, Box<dyn std::error::Error>> {
        Ok(u64::from_le_bytes(self.read_bytes(8)?.try_into().unwrap()))
    }

    fn read_i64(&mut self) -> Result<i64, Box<dyn std::error::Error>> {
        Ok(i64::from_le_bytes(self.read_bytes(8)?.try_into().unwrap()))
    }

    fn read_f32(&mut self) -> Result<f32, Box<dyn std::error::Error>> {
        Ok(f32::from_le_bytes(self.read_bytes(4)?.try_into().unwrap()))
    }

    fn read_f64(&mut self) -> Result<f64, Box<dyn std::error::Error>> {
        Ok(f64::from_le_bytes(self.read_bytes(8)?.try_into().unwrap()))
    }

    fn read_bool(&mut self) -> Result<bool, Box<dyn std::error::Error>> {
        Ok(self.read_u8()? != 0)
    }

    /// Read a GGUF string: u64 length + raw UTF-8 bytes (no null terminator).
    fn read_gguf_string(&mut self) -> Result<String, Box<dyn std::error::Error>> {
        let len = self.read_u64()? as usize;
        let bytes = self.read_bytes(len)?;
        Ok(String::from_utf8(bytes.to_vec())?)
    }

    /// Read a single GGUF metadata value (type code + payload).
    fn read_gguf_value(&mut self) -> Result<GgufValue, Box<dyn std::error::Error>> {
        let type_code = self.read_u32()?;
        let meta_type = GgufMetaType::from_u32(type_code)
            .ok_or_else(|| format!("unknown GGUF metadata type code: {type_code}"))?;
        self.read_gguf_value_of_type(meta_type)
    }

    fn read_gguf_value_of_type(
        &mut self,
        meta_type: GgufMetaType,
    ) -> Result<GgufValue, Box<dyn std::error::Error>> {
        match meta_type {
            GgufMetaType::U8 => Ok(GgufValue::U8(self.read_u8()?)),
            GgufMetaType::I8 => Ok(GgufValue::I8(self.read_i8()?)),
            GgufMetaType::U16 => Ok(GgufValue::U16(self.read_u16()?)),
            GgufMetaType::I16 => Ok(GgufValue::I16(self.read_i16()?)),
            GgufMetaType::U32 => Ok(GgufValue::U32(self.read_u32()?)),
            GgufMetaType::I32 => Ok(GgufValue::I32(self.read_i32()?)),
            GgufMetaType::U64 => Ok(GgufValue::U64(self.read_u64()?)),
            GgufMetaType::I64 => Ok(GgufValue::I64(self.read_i64()?)),
            GgufMetaType::F32 => Ok(GgufValue::F32(self.read_f32()?)),
            GgufMetaType::F64 => Ok(GgufValue::F64(self.read_f64()?)),
            GgufMetaType::Bool => Ok(GgufValue::Bool(self.read_bool()?)),
            GgufMetaType::String => Ok(GgufValue::String(self.read_gguf_string()?)),
            GgufMetaType::Array => {
                let elem_type_code = self.read_u32()?;
                let elem_type = GgufMetaType::from_u32(elem_type_code).ok_or_else(|| {
                    format!("unknown GGUF array element type code: {elem_type_code}")
                })?;
                let len = self.read_u64()? as usize;
                let mut arr = Vec::with_capacity(len);
                for _ in 0..len {
                    arr.push(self.read_gguf_value_of_type(elem_type)?);
                }
                Ok(GgufValue::Array(arr))
            }
        }
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn align_up(offset: usize, alignment: usize) -> usize {
    (offset + alignment - 1) & !(alignment - 1)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal GGUF v3 file in memory with one F32 tensor.
    fn build_minimal_gguf() -> Vec<u8> {
        let mut buf = Vec::new();
        let alignment = GGUF_DEFAULT_ALIGNMENT;

        // Magic
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        // Version 3
        buf.extend_from_slice(&3u32.to_le_bytes());
        // Tensor count: 1
        buf.extend_from_slice(&1u64.to_le_bytes());
        // Metadata KV count: 1
        buf.extend_from_slice(&1u64.to_le_bytes());

        // One metadata KV: "general.architecture" = "test"
        write_gguf_string(&mut buf, "general.architecture");
        buf.extend_from_slice(&(GgufMetaType::String as u32).to_le_bytes());
        write_gguf_string(&mut buf, "test");

        // One tensor info: "weight" shape [2, 3] F32
        write_gguf_string(&mut buf, "weight");
        buf.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        // GGUF dims are innermost-first: [3, 2] for logical [2, 3]
        buf.extend_from_slice(&3u64.to_le_bytes());
        buf.extend_from_slice(&2u64.to_le_bytes());
        buf.extend_from_slice(&(GgmlType::F32 as u32).to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset within data region

        // Pad to alignment
        let data_start = align_up(buf.len(), alignment);
        buf.resize(data_start, 0);

        // Tensor data: 6 f32 values
        for i in 0..6u32 {
            buf.extend_from_slice(&(i as f32).to_le_bytes());
        }

        buf
    }

    fn write_gguf_string(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    #[test]
    fn parse_minimal_gguf() {
        let data = build_minimal_gguf();
        let tmp = std::env::temp_dir().join("homura_test_minimal.gguf");
        std::fs::write(&tmp, &data).unwrap();

        let gguf = GgufFile::load(&tmp).unwrap();
        assert_eq!(gguf.version, 3);
        assert_eq!(gguf.architecture(), Some("test"));
        assert_eq!(gguf.tensors.len(), 1);

        let t = &gguf.tensors[0];
        assert_eq!(t.name, "weight");
        assert_eq!(t.shape, vec![2, 3]); // reversed from GGUF [3, 2]
        assert_eq!(t.ggml_type, GgmlType::F32);
        assert_eq!(t.num_elements(), 6);
        assert_eq!(t.byte_size(), 24);

        let raw = gguf.tensor_data(t);
        assert_eq!(raw.len(), 24);
        // Verify first value is 0.0f32
        let first = f32::from_le_bytes(raw[0..4].try_into().unwrap());
        assert_eq!(first, 0.0);
        // Last value is 5.0f32
        let last = f32::from_le_bytes(raw[20..24].try_into().unwrap());
        assert_eq!(last, 5.0);

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn parse_with_q8_0_tensor() {
        let mut buf = Vec::new();
        let alignment = GGUF_DEFAULT_ALIGNMENT;

        // Header
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
        buf.extend_from_slice(&0u64.to_le_bytes()); // 0 metadata

        // Tensor: "blk.0.attn_q.weight" shape [64, 32] Q8_0
        write_gguf_string(&mut buf, "blk.0.attn_q.weight");
        buf.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        // GGUF innermost-first: [32, 64]
        buf.extend_from_slice(&32u64.to_le_bytes());
        buf.extend_from_slice(&64u64.to_le_bytes());
        buf.extend_from_slice(&(GgmlType::Q8_0 as u32).to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset

        // Align
        let data_start = align_up(buf.len(), alignment);
        buf.resize(data_start, 0);

        // Q8_0 data: 64*32 = 2048 elements, 2048/32 = 64 blocks, 64*34 = 2176 bytes
        let tensor_bytes = 64 * 34;
        buf.resize(buf.len() + tensor_bytes, 0xAB);

        let tmp = std::env::temp_dir().join("homura_test_q8_0.gguf");
        std::fs::write(&tmp, &buf).unwrap();

        let gguf = GgufFile::load(&tmp).unwrap();
        let t = &gguf.tensors[0];
        assert_eq!(t.name, "blk.0.attn_q.weight");
        assert_eq!(t.shape, vec![64, 32]); // reversed
        assert_eq!(t.ggml_type, GgmlType::Q8_0);
        assert_eq!(t.num_elements(), 2048);
        assert_eq!(t.byte_size(), 2176);

        let raw = gguf.tensor_data(t);
        assert_eq!(raw.len(), 2176);

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn ggml_type_byte_sizes() {
        assert_eq!(GgmlType::Q8_0.type_size(), 34);
        assert_eq!(GgmlType::Q8_0.block_size(), 32);
        assert_eq!(GgmlType::Q4K.type_size(), 144);
        assert_eq!(GgmlType::Q4K.block_size(), 256);
        assert_eq!(GgmlType::Q6K.type_size(), 210);
        assert_eq!(GgmlType::Q6K.block_size(), 256);
    }

    #[test]
    fn align_up_works() {
        assert_eq!(align_up(0, 32), 0);
        assert_eq!(align_up(1, 32), 32);
        assert_eq!(align_up(31, 32), 32);
        assert_eq!(align_up(32, 32), 32);
        assert_eq!(align_up(33, 32), 64);
    }

    #[test]
    fn load_real_qwen25_3b_gguf() {
        let path =
            std::path::Path::new("models/Qwen2.5-3B-Instruct-GGUF/Qwen2.5-3B-Instruct-Q8_0.gguf");
        if !path.exists() {
            eprintln!("skipping: {path:?} not found");
            return;
        }

        let gguf = GgufFile::load(path).unwrap();
        gguf.summary();

        // Architecture should be qwen2
        assert_eq!(gguf.architecture(), Some("qwen2"));

        // Check key config metadata
        let arch = gguf.architecture().unwrap();
        let hidden = gguf
            .meta(&format!("{arch}.embedding_length"))
            .unwrap()
            .as_u64()
            .unwrap();
        let layers = gguf
            .meta(&format!("{arch}.block_count"))
            .unwrap()
            .as_u64()
            .unwrap();
        let heads = gguf
            .meta(&format!("{arch}.attention.head_count"))
            .unwrap()
            .as_u64()
            .unwrap();
        let kv_heads = gguf
            .meta(&format!("{arch}.attention.head_count_kv"))
            .unwrap()
            .as_u64()
            .unwrap();
        println!("hidden={hidden} layers={layers} heads={heads} kv_heads={kv_heads}");

        // Qwen2.5-3B: 2048 hidden, 36 layers, 16 heads, 2 kv_heads
        assert_eq!(hidden, 2048);
        assert_eq!(layers, 36);
        assert_eq!(heads, 16);
        assert_eq!(kv_heads, 2);

        // Should have tensors -- at minimum embed + layers + output
        assert!(
            gguf.tensors.len() > 100,
            "expected >100 tensors, got {}",
            gguf.tensors.len()
        );

        // Check a specific tensor exists with expected type
        let q_proj = gguf
            .tensor_info("blk.0.attn_q.weight")
            .expect("blk.0.attn_q.weight missing");
        assert_eq!(q_proj.ggml_type, GgmlType::Q8_0);
        println!(
            "blk.0.attn_q.weight: shape={:?} type={:?} bytes={}",
            q_proj.shape,
            q_proj.ggml_type,
            q_proj.byte_size()
        );

        // Verify we can access tensor data without panicking
        let data = gguf.tensor_data(q_proj);
        assert_eq!(data.len(), q_proj.byte_size());

        // Check tokenizer metadata exists
        assert!(
            gguf.meta("tokenizer.ggml.tokens").is_some(),
            "tokenizer tokens missing"
        );
        assert!(
            gguf.meta("tokenizer.ggml.eos_token_id").is_some(),
            "eos_token_id missing"
        );
    }

    #[test]
    fn extract_transformer_config_from_gguf() {
        let path =
            std::path::Path::new("models/Qwen2.5-3B-Instruct-GGUF/Qwen2.5-3B-Instruct-Q8_0.gguf");
        if !path.exists() {
            eprintln!("skipping: {path:?} not found");
            return;
        }

        let gguf = GgufFile::load(path).unwrap();
        let config = gguf.transformer_config().unwrap();

        println!("{config:#?}");

        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.intermediate_size, 11008);
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.num_hidden_layers, 36);
        assert_eq!(config.kv_heads(), 2);
        assert_eq!(config.head_dim(), 128);
        assert_eq!(config.gqa_repeat(), 8);
        assert!(
            config.vocab_size > 100000,
            "vocab_size should be large, got {}",
            config.vocab_size
        );
        // Qwen2.5-3B-Instruct GGUF has tied embeddings (no output.weight tensor)
        assert!(config.tie_word_embeddings);
    }

    #[test]
    fn load_transformer_weights_from_gguf() {
        let path =
            std::path::Path::new("models/Qwen2.5-3B-Instruct-GGUF/Qwen2.5-3B-Instruct-Q8_0.gguf");
        if !path.exists() {
            eprintln!("skipping: {path:?} not found");
            return;
        }

        let gguf = GgufFile::load(path).unwrap();
        let config = gguf.transformer_config().unwrap();
        let weights = gguf.load_transformer_weights(&config).unwrap();

        assert_eq!(weights.layers.len(), 36);

        // Embedding: f32, [vocab, hidden]
        assert_eq!(weights.embed_tokens_weight.dtype(), DType::F32);
        let embed_shape = &weights.embed_tokens_weight.shape().0;
        assert_eq!(embed_shape[1], 2048);
        println!(
            "embed_tokens: shape={:?} dtype={:?}",
            embed_shape,
            weights.embed_tokens_weight.dtype()
        );

        // Projection weights: Q8_0, NOT transposed (GGUF layout [out, in])
        let l0 = &weights.layers[0];
        assert_eq!(l0.q_proj_weight.dtype(), DType::Q8_0);
        println!(
            "q_proj: shape={:?} dtype={:?} bytes={}",
            l0.q_proj_weight.shape().0,
            l0.q_proj_weight.dtype(),
            l0.q_proj_weight.byte_len()
        );

        // Norms: f32
        assert_eq!(l0.input_layernorm_weight.dtype(), DType::F32);
        assert_eq!(l0.post_attention_layernorm_weight.dtype(), DType::F32);

        // Qwen2.5 has biases
        assert!(l0.q_proj_bias.is_some());
        assert_eq!(l0.q_proj_bias.as_ref().unwrap().dtype(), DType::F32);

        // Final norm
        assert_eq!(weights.final_norm_weight.dtype(), DType::F32);

        // LM head: tied (None), embed_tokens used instead
        assert!(
            weights.lm_head_weight.is_none(),
            "Qwen2.5-3B-Instruct has tied embeddings"
        );
    }

}
