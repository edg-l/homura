use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

use crate::{DType, Shape, runtime::OutputDesc};

// ── CompilationCache ──────────────────────────────────────────────────────────

/// Disk-based compilation cache that stores native shared libraries (.so).
///
/// Cache directory is `~/.cache/homura/` by default, or `HOMURA_CACHE_DIR` if set.
/// Each cache entry consists of two files:
/// - `{key}.so`  — the native compiled shared library
/// - `{key}.meta` — JSON sidecar with `num_inputs` and output shapes/dtypes
///
/// On a cache hit, the compiler skips compilation entirely and loads
/// the .so directly via dlopen, which is near-instant.
pub struct CompilationCache {
    cache_dir: PathBuf,
}

/// Sidecar metadata stored alongside a cached .so.
pub(crate) struct CacheMeta {
    pub(crate) num_inputs: usize,
    pub(crate) outputs: Vec<OutputDesc>,
}

impl CompilationCache {
    /// Create a cache, creating the cache directory if it does not exist.
    pub fn new() -> Self {
        let cache_dir = std::env::var("HOMURA_CACHE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                dirs_cache_dir()
                    .unwrap_or_else(|| PathBuf::from("/tmp"))
                    .join("homura")
            });
        std::fs::create_dir_all(&cache_dir).ok();
        Self { cache_dir }
    }

    /// Return the cache directory path.
    pub fn cache_dir(&self) -> &std::path::Path {
        &self.cache_dir
    }

    /// Compute a cache key from model bytes, input shapes, and compiler fingerprint.
    ///
    /// The fingerprint includes LLVM version, homura crate version, and host CPU
    /// features, so cached `.so` files are automatically invalidated when any of
    /// these change.
    ///
    /// Uses a 64-bit hash via `DefaultHasher`. Collisions are benign (cache miss,
    /// not corruption) and vanishingly rare in practice.
    pub fn cache_key(model_bytes: &[u8], input_shapes: &[&[u64]]) -> String {
        let mut hasher = DefaultHasher::new();

        // Compiler fingerprint: invalidates cache on LLVM/homura/CPU changes.
        compiler_fingerprint().hash(&mut hasher);

        model_bytes.hash(&mut hasher);
        for shape in input_shapes {
            shape.hash(&mut hasher);
            // Separator between shapes so [2,3] + [4] ≠ [2] + [3,4].
            0xffu8.hash(&mut hasher);
        }
        format!("{:016x}", hasher.finish())
    }

    /// Return the paths to the cached .so and .meta files if both exist.
    pub fn get(&self, key: &str) -> Option<(PathBuf, PathBuf)> {
        let so_path = self.cache_dir.join(format!("{key}.so"));
        let meta_path = self.cache_dir.join(format!("{key}.meta"));
        if so_path.exists() && meta_path.exists() {
            Some((so_path, meta_path))
        } else {
            None
        }
    }

    /// Write the native .so (from an already-linked file) and metadata sidecar.
    ///
    /// `so_src` is the path to the .so file to store (it is copied into the
    /// cache directory under `{key}.so`).
    pub(crate) fn store(
        &self,
        key: &str,
        so_src: &std::path::Path,
        meta: &CacheMeta,
    ) -> std::io::Result<()> {
        let so_dst = self.cache_dir.join(format!("{key}.so"));
        std::fs::copy(so_src, &so_dst)?;

        let meta_path = self.cache_dir.join(format!("{key}.meta"));
        let json = encode_meta(meta);
        std::fs::write(meta_path, json)?;

        Ok(())
    }

    /// Parse the metadata sidecar for a cache entry.
    pub(crate) fn load_meta(meta_path: &std::path::Path) -> Option<CacheMeta> {
        let text = std::fs::read_to_string(meta_path).ok()?;
        decode_meta(&text)
    }
}

// ── Simple JSON meta encoding (no external dep) ───────────────────────────────

fn dtype_to_str(d: DType) -> &'static str {
    match d {
        DType::F32 => "f32",
        DType::F64 => "f64",
        DType::I32 => "i32",
        DType::I64 => "i64",
        DType::BF16 => panic!("BF16 not supported in computation; convert to F32 first"),
    }
}

fn dtype_from_str(s: &str) -> Option<DType> {
    match s {
        "f32" => Some(DType::F32),
        "f64" => Some(DType::F64),
        "i32" => Some(DType::I32),
        "i64" => Some(DType::I64),
        _ => None,
    }
}

/// Encode metadata as a minimal JSON string.
///
/// Format: `{"num_inputs":N,"outputs":[{"shape":[...],"dtype":"f32"},...]}`
fn encode_meta(meta: &CacheMeta) -> String {
    let mut s = String::new();
    s.push_str("{\"num_inputs\":");
    s.push_str(&meta.num_inputs.to_string());
    s.push_str(",\"outputs\":[");
    for (i, out) in meta.outputs.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str("{\"shape\":[");
        for (j, &dim) in out.shape.0.iter().enumerate() {
            if j > 0 {
                s.push(',');
            }
            s.push_str(&dim.to_string());
        }
        s.push_str("],\"dtype\":\"");
        s.push_str(dtype_to_str(out.dtype));
        s.push_str("\"}");
    }
    s.push_str("]}");
    s
}

/// Decode metadata from the minimal JSON produced by `encode_meta`.
///
/// Returns `None` on any parse error (treated as a cache miss, not a fatal error).
fn decode_meta(s: &str) -> Option<CacheMeta> {
    // Extract num_inputs value.
    let ni_key = "\"num_inputs\":";
    let ni_pos = s.find(ni_key)? + ni_key.len();
    let ni_end = s[ni_pos..].find([',', '}']).map(|p| ni_pos + p)?;
    let num_inputs: usize = s[ni_pos..ni_end].trim().parse().ok()?;

    // Extract outputs array content.
    let out_start = s.find("\"outputs\":[")?;
    let array_start = out_start + "\"outputs\":[".len();
    // Find matching closing bracket.
    let array_content = s.get(array_start..)?;
    let mut depth = 0i32;
    let mut array_end = array_content.len();
    for (i, c) in array_content.char_indices() {
        match c {
            '[' | '{' => depth += 1,
            ']' | '}' => {
                if depth == 0 {
                    array_end = i;
                    break;
                }
                depth -= 1;
            }
            _ => {}
        }
    }
    let array_content = &array_content[..array_end];

    let outputs = parse_output_array(array_content)?;

    Some(CacheMeta {
        num_inputs,
        outputs,
    })
}

/// Parse the JSON array of output descriptors.
fn parse_output_array(s: &str) -> Option<Vec<OutputDesc>> {
    let mut outputs = Vec::new();
    let mut remaining = s.trim();

    while !remaining.is_empty() {
        // Find opening {
        let obj_start = remaining.find('{')?;
        remaining = &remaining[obj_start + 1..];

        // Parse shape array.
        let shape_key = "\"shape\":[";
        let shape_pos = remaining.find(shape_key)? + shape_key.len();
        let after_shape = &remaining[shape_pos..];
        let shape_end = after_shape.find(']')?;
        let shape_str = &after_shape[..shape_end];
        let shape: Vec<u64> = if shape_str.trim().is_empty() {
            vec![]
        } else {
            shape_str
                .split(',')
                .map(|x| x.trim().parse::<u64>().ok())
                .collect::<Option<Vec<_>>>()?
        };

        // Parse dtype string.
        let dtype_key = "\"dtype\":\"";
        let dtype_pos = remaining.find(dtype_key)? + dtype_key.len();
        let after_dtype = &remaining[dtype_pos..];
        let dtype_end = after_dtype.find('"')?;
        let dtype = dtype_from_str(&after_dtype[..dtype_end])?;

        outputs.push(OutputDesc {
            shape: Shape(shape),
            dtype,
        });

        // Advance past the closing }
        let obj_end = remaining.find('}')?;
        remaining = &remaining[obj_end + 1..];
        // Skip comma separator if present.
        remaining = remaining.trim_start_matches([',', ' ']);
    }

    Some(outputs)
}

/// Portable `~/.cache` resolution (mirrors the `dirs` crate without a new dep).
fn dirs_cache_dir() -> Option<PathBuf> {
    // On Linux/macOS: $XDG_CACHE_HOME or ~/.cache
    if let Ok(dir) = std::env::var("XDG_CACHE_HOME") {
        return Some(PathBuf::from(dir));
    }
    std::env::var("HOME")
        .ok()
        .map(|h| PathBuf::from(h).join(".cache"))
}

// ── Bucket padding ────────────────────────────────────────────────────────────

/// Round `len` up to the nearest power-of-2 bucket in {32, 64, 128, 256, 512, 1024}.
///
/// Used by the generation loop to avoid a unique compilation per sequence
/// length: a prompt of 5 tokens compiles for seq_len=32, a prompt of 100
/// tokens compiles for seq_len=128, giving at most 6 unique compilations.
/// Returns 1024 for lengths > 1024.
pub fn bucket_pad(len: usize) -> usize {
    const BUCKETS: [usize; 6] = [32, 64, 128, 256, 512, 1024];
    *BUCKETS.iter().find(|&&b| b >= len).unwrap_or(&1024)
}

// ── Compiler fingerprint ──────────────────────────────────────────────────────

/// Returns a string that changes whenever the compilation environment changes,
/// invalidating cached `.so` files. Includes:
/// - Homura crate version (codegen changes)
/// - LLVM version (backend changes)
/// - Host CPU name + features (AVX2, SSE4.2, etc.)
/// Bump this when the pass pipeline changes in a way that invalidates cached
/// .so files (e.g. adding OpenMP threading support).
const PIPELINE_VERSION: u32 = 2;

fn compiler_fingerprint() -> String {
    use std::sync::OnceLock;
    static FINGERPRINT: OnceLock<String> = OnceLock::new();
    FINGERPRINT
        .get_or_init(|| {
            let homura_version = env!("CARGO_PKG_VERSION");
            let (cpu, features, llvm_version) = unsafe {
                use llvm_sys::core::LLVMDisposeMessage;
                use llvm_sys::target_machine::{LLVMGetHostCPUFeatures, LLVMGetHostCPUName};
                use std::ffi::CStr;

                let cpu_ptr = LLVMGetHostCPUName();
                let feat_ptr = LLVMGetHostCPUFeatures();
                let cpu = CStr::from_ptr(cpu_ptr).to_string_lossy().into_owned();
                let feat = CStr::from_ptr(feat_ptr).to_string_lossy().into_owned();
                LLVMDisposeMessage(cpu_ptr);
                LLVMDisposeMessage(feat_ptr);

                // LLVM version from the linked library
                let ver = llvm_sys::core::LLVMGetVersion;
                let mut major = 0u32;
                let mut minor = 0u32;
                let mut patch = 0u32;
                ver(&mut major, &mut minor, &mut patch);
                let llvm_ver = format!("{major}.{minor}.{patch}");

                (cpu, feat, llvm_ver)
            };
            format!("homura={homura_version};pipeline={PIPELINE_VERSION};llvm={llvm_version};cpu={cpu};features={features}")
        })
        .clone()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_key_is_deterministic() {
        let bytes = b"model_data";
        let shapes: &[&[u64]] = &[&[1, 3, 224, 224], &[1, 1000]];
        let k1 = CompilationCache::cache_key(bytes, shapes);
        let k2 = CompilationCache::cache_key(bytes, shapes);
        assert_eq!(k1, k2);
        assert_eq!(k1.len(), 16, "key should be 16 hex chars (64-bit hash)");
    }

    #[test]
    fn cache_key_differs_for_different_shapes() {
        let bytes = b"model_data";
        let k1 = CompilationCache::cache_key(bytes, &[&[1, 32]]);
        let k2 = CompilationCache::cache_key(bytes, &[&[1, 64]]);
        assert_ne!(k1, k2);
    }

    #[test]
    fn cache_key_differs_for_different_model_bytes() {
        let shapes: &[&[u64]] = &[&[1, 32]];
        let k1 = CompilationCache::cache_key(b"model_a", shapes);
        let k2 = CompilationCache::cache_key(b"model_b", shapes);
        assert_ne!(k1, k2);
    }

    #[test]
    fn bucket_pad_rounds_up() {
        assert_eq!(bucket_pad(1), 32);
        assert_eq!(bucket_pad(32), 32);
        assert_eq!(bucket_pad(33), 64);
        assert_eq!(bucket_pad(64), 64);
        assert_eq!(bucket_pad(100), 128);
        assert_eq!(bucket_pad(256), 256);
        assert_eq!(bucket_pad(513), 1024);
        assert_eq!(bucket_pad(1024), 1024);
        assert_eq!(bucket_pad(2000), 1024);
    }

    #[test]
    fn meta_roundtrip_f32() {
        let meta = CacheMeta {
            num_inputs: 2,
            outputs: vec![
                OutputDesc {
                    shape: Shape(vec![1, 3, 224, 224]),
                    dtype: DType::F32,
                },
                OutputDesc {
                    shape: Shape(vec![1000]),
                    dtype: DType::F32,
                },
            ],
        };
        let json = encode_meta(&meta);
        let decoded = decode_meta(&json).expect("decode failed");
        assert_eq!(decoded.num_inputs, 2);
        assert_eq!(decoded.outputs.len(), 2);
        assert_eq!(decoded.outputs[0].shape.0, vec![1, 3, 224, 224]);
        assert_eq!(decoded.outputs[0].dtype, DType::F32);
        assert_eq!(decoded.outputs[1].shape.0, vec![1000]);
    }

    #[test]
    fn meta_roundtrip_scalar_output() {
        let meta = CacheMeta {
            num_inputs: 1,
            outputs: vec![OutputDesc {
                shape: Shape(vec![]),
                dtype: DType::I64,
            }],
        };
        let json = encode_meta(&meta);
        let decoded = decode_meta(&json).expect("decode failed");
        assert_eq!(decoded.num_inputs, 1);
        assert_eq!(decoded.outputs[0].shape.0, Vec::<u64>::new());
        assert_eq!(decoded.outputs[0].dtype, DType::I64);
    }

    #[test]
    fn meta_roundtrip_mixed_dtypes() {
        let meta = CacheMeta {
            num_inputs: 3,
            outputs: vec![
                OutputDesc {
                    shape: Shape(vec![4, 4]),
                    dtype: DType::F64,
                },
                OutputDesc {
                    shape: Shape(vec![2]),
                    dtype: DType::I32,
                },
            ],
        };
        let json = encode_meta(&meta);
        let decoded = decode_meta(&json).expect("decode failed");
        assert_eq!(decoded.num_inputs, 3);
        assert_eq!(decoded.outputs[0].dtype, DType::F64);
        assert_eq!(decoded.outputs[1].dtype, DType::I32);
    }
}
