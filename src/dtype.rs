use melior::{
    Context,
    ir::{Type, r#type::IntegerType},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F64,
    I32,
    I64,
}

impl DType {
    /// Size in bytes of one element.
    pub fn size_bytes(self) -> usize {
        match self {
            DType::F32 | DType::I32 => 4,
            DType::F64 | DType::I64 => 8,
        }
    }

    /// Convert to an MLIR Type.
    pub(crate) fn to_mlir_type<'c>(self, context: &'c Context) -> Type<'c> {
        match self {
            DType::F32 => Type::float32(context),
            DType::F64 => Type::float64(context),
            DType::I32 => IntegerType::new(context, 32).into(),
            DType::I64 => IntegerType::new(context, 64).into(),
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
        assert_eq!(DType::I32.size_bytes(), 4);
        assert_eq!(DType::I64.size_bytes(), 8);
    }
}
