use std::collections::HashMap;
use std::path::Path;

use memmap2::Mmap;

use crate::DType;
use crate::runtime::Buffer;

/// Metadata for a single tensor in a safetensors file.
#[derive(Debug, serde::Deserialize)]
struct TensorInfo {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [usize; 2],
}

/// Load all tensors from a safetensors file, converting to f32.
///
/// Weights are memory-mapped to avoid doubling memory usage during parsing.
/// bf16 and f16 tensors are converted to f32; f32 tensors are copied as-is.
pub fn load_safetensors(
    path: &Path,
) -> Result<HashMap<String, Buffer>, Box<dyn std::error::Error>> {
    load_safetensors_impl(path, false)
}

/// Load tensors, keeping bf16 weights in native bf16 format.
///
/// bf16 tensors stay as `DType::BF16` (2 bytes/element). f16 is converted to f32.
/// f32/f64/i32/i64 are loaded as usual.
pub fn load_safetensors_bf16(
    path: &Path,
) -> Result<HashMap<String, Buffer>, Box<dyn std::error::Error>> {
    load_safetensors_impl(path, true)
}

fn load_safetensors_impl(
    path: &Path,
    keep_bf16: bool,
) -> Result<HashMap<String, Buffer>, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };

    if mmap.len() < 8 {
        return Err("safetensors file too small (< 8 bytes)".into());
    }

    // First 8 bytes: little-endian u64 header length.
    let header_len = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
    let header_end = 8 + header_len;
    if header_end > mmap.len() {
        return Err(format!(
            "safetensors header length {header_len} exceeds file size {}",
            mmap.len()
        )
        .into());
    }

    let header_bytes = &mmap[8..header_end];
    let header: HashMap<String, serde_json::Value> = serde_json::from_slice(header_bytes)?;

    let data_base = header_end;
    let mut tensors = HashMap::new();

    for (name, value) in &header {
        if name == "__metadata__" {
            continue;
        }

        let info: TensorInfo = serde_json::from_value(value.clone())?;
        let [start, end] = info.data_offsets;
        let raw = &mmap[data_base + start..data_base + end];

        let buf = match info.dtype.as_str() {
            "BF16" => {
                if keep_bf16 {
                    bf16_native_buffer(raw, &info.shape)
                } else {
                    bf16_to_f32_buffer(raw, &info.shape)
                }
            }
            "F16" => f16_to_f32_buffer(raw, &info.shape),
            "F32" => f32_buffer(raw, &info.shape),
            "F64" => f64_to_f32_buffer(raw, &info.shape),
            "I32" => i32_buffer(raw, &info.shape),
            "I64" => i64_buffer(raw, &info.shape),
            other => {
                return Err(format!("unsupported safetensors dtype: {other}").into());
            }
        };

        tensors.insert(name.clone(), buf);
    }

    Ok(tensors)
}

/// Keep bf16 raw bytes as a native bf16 buffer (no conversion).
fn bf16_native_buffer(raw: &[u8], shape: &[u64]) -> Buffer {
    let mut buf = Buffer::new(shape, DType::BF16);
    buf.data_mut().copy_from_slice(raw);
    buf
}

/// Convert bf16 raw bytes to f32 buffer.
/// bf16 is the top 16 bits of f32: `f32::from_bits((u16 as u32) << 16)`.
fn bf16_to_f32_buffer(raw: &[u8], shape: &[u64]) -> Buffer {
    let num_elems = raw.len() / 2;
    let mut f32_data = Vec::with_capacity(num_elems);
    for chunk in raw.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        f32_data.push(f32::from_bits((bits as u32) << 16));
    }
    Buffer::from_slice::<f32>(&f32_data, shape, DType::F32)
}

/// Convert f16 (IEEE 754 half) raw bytes to f32 buffer.
fn f16_to_f32_buffer(raw: &[u8], shape: &[u64]) -> Buffer {
    let num_elems = raw.len() / 2;
    let mut f32_data = Vec::with_capacity(num_elems);
    for chunk in raw.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        f32_data.push(f16_to_f32(bits));
    }
    Buffer::from_slice::<f32>(&f32_data, shape, DType::F32)
}

/// Convert a single f16 (IEEE 754) to f32.
fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1F;
    let mant = h & 0x3FF;

    if exp == 0 {
        if mant == 0 {
            // Zero
            f32::from_bits((sign as u32) << 31)
        } else {
            // Subnormal f16 → normal f32
            let mut m = mant as u32;
            let mut e: i32 = -14; // f16 subnormal exponent
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF; // remove implicit bit
            let f32_exp = (e + 127) as u32;
            f32::from_bits(((sign as u32) << 31) | (f32_exp << 23) | (m << 13))
        }
    } else if exp == 31 {
        // Inf or NaN
        let f32_mant = (mant as u32) << 13;
        f32::from_bits(((sign as u32) << 31) | (0xFF << 23) | f32_mant)
    } else {
        // Normal
        let f32_exp = (exp as u32) - 15 + 127;
        let f32_mant = (mant as u32) << 13;
        f32::from_bits(((sign as u32) << 31) | (f32_exp << 23) | f32_mant)
    }
}

fn f32_buffer(raw: &[u8], shape: &[u64]) -> Buffer {
    let mut buf = Buffer::new(shape, DType::F32);
    buf.data_mut().copy_from_slice(raw);
    buf
}

fn f64_to_f32_buffer(raw: &[u8], shape: &[u64]) -> Buffer {
    let num_elems = raw.len() / 8;
    let mut f32_data = Vec::with_capacity(num_elems);
    for chunk in raw.chunks_exact(8) {
        let val = f64::from_le_bytes(chunk.try_into().unwrap());
        f32_data.push(val as f32);
    }
    Buffer::from_slice::<f32>(&f32_data, shape, DType::F32)
}

fn i32_buffer(raw: &[u8], shape: &[u64]) -> Buffer {
    let mut buf = Buffer::new(shape, DType::I32);
    buf.data_mut().copy_from_slice(raw);
    buf
}

fn i64_buffer(raw: &[u8], shape: &[u64]) -> Buffer {
    let mut buf = Buffer::new(shape, DType::I64);
    buf.data_mut().copy_from_slice(raw);
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_conversion_basic() {
        // bf16 1.0 = 0x3F80, f32 1.0 = 0x3F800000
        let bf16_one: u16 = 0x3F80;
        let f32_val = f32::from_bits((bf16_one as u32) << 16);
        assert_eq!(f32_val, 1.0);

        // bf16 -2.0 = 0xC000
        let bf16_neg2: u16 = 0xC000;
        let f32_val = f32::from_bits((bf16_neg2 as u32) << 16);
        assert_eq!(f32_val, -2.0);
    }

    #[test]
    fn bf16_to_f32_buffer_roundtrip() {
        // Encode [1.0, -0.5, 0.0] as bf16 bytes
        let values: Vec<f32> = vec![1.0, -0.5, 0.0];
        let raw: Vec<u8> = values
            .iter()
            .flat_map(|v| {
                let bits = v.to_bits();
                let bf16 = (bits >> 16) as u16;
                bf16.to_le_bytes()
            })
            .collect();

        let buf = bf16_to_f32_buffer(&raw, &[3]);
        let result = buf.as_slice::<f32>();
        assert_eq!(result, &[1.0, -0.5, 0.0]);
    }

    #[test]
    fn f16_conversion_basic() {
        // f16 1.0 = 0x3C00
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        // f16 -1.0 = 0xBC00
        assert_eq!(f16_to_f32(0xBC00), -1.0);
        // f16 0.0 = 0x0000
        assert_eq!(f16_to_f32(0x0000), 0.0);
        // f16 +inf = 0x7C00
        assert!(f16_to_f32(0x7C00).is_infinite());
        // f16 NaN = 0x7E00
        assert!(f16_to_f32(0x7E00).is_nan());
    }

    #[test]
    fn load_safetensors_file() {
        // Create a minimal safetensors file in a temp dir.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.safetensors");

        // Header: one tensor "x" of shape [2] F32
        let header = r#"{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let header_bytes = header.as_bytes();
        let header_len = header_bytes.len() as u64;

        // Data: [1.0f32, 2.0f32]
        let data: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();

        let mut file_bytes = Vec::new();
        file_bytes.extend_from_slice(&header_len.to_le_bytes());
        file_bytes.extend_from_slice(header_bytes);
        file_bytes.extend_from_slice(&data);
        std::fs::write(&path, &file_bytes).unwrap();

        let tensors = load_safetensors(&path).unwrap();
        assert_eq!(tensors.len(), 1);
        let x = &tensors["x"];
        assert_eq!(x.shape().0, vec![2]);
        assert_eq!(x.as_slice::<f32>(), &[1.0, 2.0]);
    }

    #[test]
    fn load_real_qwen2_safetensors() {
        let path = std::path::Path::new(concat!(
            env!("HOME"),
            "/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/",
            "snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987/model.safetensors"
        ));
        if !path.exists() {
            eprintln!("skipping: Qwen2.5-0.5B not found");
            return;
        }

        let t = std::time::Instant::now();
        let tensors = load_safetensors(path).unwrap();
        let elapsed = t.elapsed();
        eprintln!(
            "loaded {} tensors in {:.2}s",
            tensors.len(),
            elapsed.as_secs_f64()
        );

        assert_eq!(tensors.len(), 290);

        // Check embed_tokens.weight shape
        let embed = &tensors["model.embed_tokens.weight"];
        assert_eq!(embed.shape().0, vec![151936, 896]);
        assert_eq!(embed.dtype(), crate::DType::F32);

        // Check a layer weight
        let q = &tensors["model.layers.0.self_attn.q_proj.weight"];
        assert_eq!(q.shape().0, vec![896, 896]);

        // Check KV proj is smaller (GQA: 2 kv_heads * 64 = 128)
        let k = &tensors["model.layers.0.self_attn.k_proj.weight"];
        assert_eq!(k.shape().0, vec![128, 896]);

        // Check bias exists
        assert!(tensors.contains_key("model.layers.0.self_attn.q_proj.bias"));

        let total_bytes: usize = tensors
            .values()
            .map(|b| b.shape().num_elements() as usize * 4)
            .sum();
        eprintln!(
            "total f32 weight memory: {:.1} MB",
            total_bytes as f64 / 1_048_576.0
        );
    }

    #[test]
    fn load_safetensors_bf16() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_bf16.safetensors");

        let header = r#"{"w":{"dtype":"BF16","shape":[3],"data_offsets":[0,6]}}"#;
        let header_bytes = header.as_bytes();
        let header_len = header_bytes.len() as u64;

        // bf16 values: [1.0, -0.5, 2.0]
        let bf16_vals: Vec<u16> = [1.0f32, -0.5, 2.0]
            .iter()
            .map(|v| (v.to_bits() >> 16) as u16)
            .collect();
        let data: Vec<u8> = bf16_vals.iter().flat_map(|v| v.to_le_bytes()).collect();

        let mut file_bytes = Vec::new();
        file_bytes.extend_from_slice(&header_len.to_le_bytes());
        file_bytes.extend_from_slice(header_bytes);
        file_bytes.extend_from_slice(&data);
        std::fs::write(&path, &file_bytes).unwrap();

        let tensors = load_safetensors(&path).unwrap();
        let w = &tensors["w"];
        assert_eq!(w.dtype(), DType::F32); // converted from bf16
        assert_eq!(w.shape().0, vec![3]);
        assert_eq!(w.as_slice::<f32>(), &[1.0, -0.5, 2.0]);
    }
}
