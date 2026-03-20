/// Runtime CPU feature detection, cached via OnceLock.
/// Used to gate bf16 mixed-precision paths and other ISA-specific optimizations.
use std::sync::OnceLock;

/// Detected CPU capabilities.
pub struct CpuCaps {
    pub avx2: bool,
    pub fma: bool,
    pub avx512f: bool,
    pub avx512_bf16: bool,
    pub avx512_vnni: bool,
}

impl CpuCaps {
    /// Detect features using `is_x86_feature_detected!`.
    #[cfg(target_arch = "x86_64")]
    fn detect() -> Self {
        CpuCaps {
            avx2: std::arch::is_x86_feature_detected!("avx2"),
            fma: std::arch::is_x86_feature_detected!("fma"),
            avx512f: std::arch::is_x86_feature_detected!("avx512f"),
            avx512_bf16: std::arch::is_x86_feature_detected!("avx512bf16"),
            avx512_vnni: std::arch::is_x86_feature_detected!("avx512vnni"),
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn detect() -> Self {
        CpuCaps {
            avx2: false,
            fma: false,
            avx512f: false,
            avx512_bf16: false,
            avx512_vnni: false,
        }
    }

    /// Cached singleton.
    pub fn get() -> &'static Self {
        static CAPS: OnceLock<CpuCaps> = OnceLock::new();
        CAPS.get_or_init(Self::detect)
    }

    /// Whether bf16 mixed-precision matmul is beneficial on this CPU.
    pub fn supports_bf16_compute(&self) -> bool {
        self.avx512_bf16
    }
}

impl std::fmt::Display for CpuCaps {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut features = Vec::new();
        if self.avx2 {
            features.push("AVX2");
        }
        if self.fma {
            features.push("FMA");
        }
        if self.avx512f {
            features.push("AVX-512F");
        }
        if self.avx512_bf16 {
            features.push("AVX-512-BF16");
        }
        if self.avx512_vnni {
            features.push("AVX-512-VNNI");
        }
        if features.is_empty() {
            write!(f, "(none)")
        } else {
            write!(f, "{}", features.join(", "))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_and_display() {
        let caps = CpuCaps::get();
        eprintln!("CPU features: {caps}");
        eprintln!("  avx2={}, fma={}", caps.avx2, caps.fma);
        eprintln!(
            "  avx512f={}, avx512_bf16={}, avx512_vnni={}",
            caps.avx512f, caps.avx512_bf16, caps.avx512_vnni
        );
        eprintln!("  supports_bf16_compute={}", caps.supports_bf16_compute());
        // On x86_64, at minimum AVX2 should be present on any modern CPU.
        #[cfg(target_arch = "x86_64")]
        assert!(caps.avx2, "expected AVX2 on x86_64");
    }
}
