use melior::{
    Context,
    ir::{Type, r#type::IntegerType},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum DType {
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    /// BFloat16 — used for mixed-precision matmul: bf16 weight inputs with
    /// f32 accumulation. Also used for native bf16 weight storage.
    BF16,
    F16,
    /// GGML Q8_0: 32-element blocks, each block = 2-byte f16 scale + 32 i8 quants (34 bytes).
    Q8_0,
    /// GGML Q4_K: 256-element super-blocks (144 bytes). 8 sub-blocks of 32 elements,
    /// 6-bit packed scales/mins, 4-bit packed quants.
    Q4_K,
    /// GGML Q6_K: 256-element super-blocks (210 bytes). 16 sub-blocks of 16 elements,
    /// 6-bit quants (4-bit low + 2-bit high), i8 sub-block scales + f16 super-block scale.
    Q6_K,
}

impl DType {
    /// Size in bytes of one element. Panics on quantized types (use `bytes_for_elements`).
    pub fn size_bytes(self) -> usize {
        match self {
            DType::I8 => 1,
            DType::BF16 | DType::F16 | DType::I16 => 2,
            DType::F32 | DType::I32 => 4,
            DType::F64 | DType::I64 => 8,
            _ => panic!(
                "size_bytes() not valid for quantized dtype {:?}; use bytes_for_elements()",
                self
            ),
        }
    }

    /// Whether this is a block-quantized type.
    pub fn is_quantized(self) -> bool {
        matches!(self, DType::Q8_0 | DType::Q4_K | DType::Q6_K)
    }

    /// Number of logical elements per quantization block.
    ///
    /// # Panics
    /// Panics on non-quantized types.
    pub fn block_size(self) -> usize {
        match self {
            DType::Q8_0 => 32,
            DType::Q4_K | DType::Q6_K => 256,
            _ => panic!("block_size() called on non-quantized dtype {:?}", self),
        }
    }

    /// Byte size of one quantization block.
    ///
    /// # Panics
    /// Panics on non-quantized types.
    pub fn block_bytes(self) -> usize {
        match self {
            DType::Q8_0 => 34,
            DType::Q4_K => 144,
            DType::Q6_K => 210,
            _ => panic!("block_bytes() called on non-quantized dtype {:?}", self),
        }
    }

    /// Total bytes needed for `num_elements` values of this dtype.
    ///
    /// For non-quantized types, this is `num_elements * size_bytes()`.
    /// For quantized types, `num_elements` must be a multiple of `block_size()`.
    pub fn bytes_for_elements(self, num_elements: usize) -> usize {
        if self.is_quantized() {
            let bs = self.block_size();
            assert!(
                num_elements % bs == 0,
                "element count {} must be a multiple of block_size {} for {:?}",
                num_elements,
                bs,
                self
            );
            (num_elements / bs) * self.block_bytes()
        } else {
            num_elements * self.size_bytes()
        }
    }

    /// Convert to an MLIR Type.
    pub(crate) fn to_mlir_type<'c>(self, context: &'c Context) -> Type<'c> {
        match self {
            DType::BF16 => Type::bfloat16(context),
            DType::F16 => Type::float16(context),
            DType::F32 => Type::float32(context),
            DType::F64 => Type::float64(context),
            DType::I8 => IntegerType::new(context, 8).into(),
            DType::I16 => IntegerType::new(context, 16).into(),
            DType::I32 => IntegerType::new(context, 32).into(),
            DType::I64 => IntegerType::new(context, 64).into(),
            // Quantized types: the memref element type is i8 (raw bytes).
            DType::Q8_0 | DType::Q4_K | DType::Q6_K => IntegerType::new(context, 8).into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_bytes() {
        assert_eq!(DType::F32.size_bytes(), 4);
        assert_eq!(DType::F64.size_bytes(), 8);
        assert_eq!(DType::I8.size_bytes(), 1);
        assert_eq!(DType::I16.size_bytes(), 2);
        assert_eq!(DType::I32.size_bytes(), 4);
        assert_eq!(DType::I64.size_bytes(), 8);
        assert_eq!(DType::F16.size_bytes(), 2);
        assert_eq!(DType::BF16.size_bytes(), 2);
    }

    #[test]
    fn quantized_block_info() {
        assert!(DType::Q8_0.is_quantized());
        assert_eq!(DType::Q8_0.block_size(), 32);
        assert_eq!(DType::Q8_0.block_bytes(), 34);

        assert!(DType::Q4_K.is_quantized());
        assert_eq!(DType::Q4_K.block_size(), 256);
        assert_eq!(DType::Q4_K.block_bytes(), 144);

        assert!(DType::Q6_K.is_quantized());
        assert_eq!(DType::Q6_K.block_size(), 256);
        assert_eq!(DType::Q6_K.block_bytes(), 210);
    }

    #[test]
    fn bytes_for_elements() {
        // Q8_0: 32 elements -> 34 bytes, 64 elements -> 68 bytes
        assert_eq!(DType::Q8_0.bytes_for_elements(32), 34);
        assert_eq!(DType::Q8_0.bytes_for_elements(64), 68);
        assert_eq!(
            DType::Q8_0.bytes_for_elements(4096 * 4096),
            4096 * 4096 / 32 * 34
        );

        // Non-quantized
        assert_eq!(DType::F32.bytes_for_elements(100), 400);
        assert_eq!(DType::BF16.bytes_for_elements(100), 200);
    }

    #[test]
    #[should_panic(expected = "must be a multiple of block_size")]
    fn bytes_for_elements_bad_alignment() {
        DType::Q8_0.bytes_for_elements(33);
    }

    #[test]
    #[should_panic(expected = "not valid for quantized")]
    fn size_bytes_panics_on_quantized() {
        DType::Q8_0.size_bytes();
    }
}
