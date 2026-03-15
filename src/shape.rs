/// Sentinel value representing a dynamic (unknown at compile time) dimension.
///
/// Chosen as `u64::MAX` for distinctness. Note that `u64::MAX as i64 == -1`,
/// which is NOT MLIR's `ShapedType::kDynamic` (= `i64::MIN`). The compiler
/// helper `dim_to_mlir` / `dim_to_mlir_i64` in `compiler.rs` translates this
/// sentinel to `i64::MIN` (kDynamic) before passing to MLIR C API calls such
/// as `RankedTensorType::new` and `MemRefType::new`.
pub const DIM_DYNAMIC: u64 = u64::MAX;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape(pub Vec<u64>);

impl Shape {
    /// Returns the total number of elements (product of all dims).
    ///
    /// # Panics
    ///
    /// Panics if any dimension is `DIM_DYNAMIC`. Use `concrete_num_elements()`
    /// when the shape may contain dynamic dims.
    pub fn num_elements(&self) -> u64 {
        assert!(
            !self.has_dynamic_dims(),
            "num_elements() called on a shape with dynamic dims: {:?}",
            self.0
        );
        self.0.iter().product()
    }

    /// Returns `Some(product)` if all dims are concrete, `None` if any is `DIM_DYNAMIC`.
    pub fn concrete_num_elements(&self) -> Option<u64> {
        if self.has_dynamic_dims() {
            None
        } else {
            Some(self.0.iter().product())
        }
    }

    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` if the dimension at `idx` is `DIM_DYNAMIC`.
    pub fn is_dynamic_dim(&self, idx: usize) -> bool {
        self.0[idx] == DIM_DYNAMIC
    }

    /// Returns `true` if any dimension in this shape is `DIM_DYNAMIC`.
    pub fn has_dynamic_dims(&self) -> bool {
        self.0.iter().any(|&d| d == DIM_DYNAMIC)
    }

    /// Compute the numpy-style broadcast shape of `self` and `other`.
    ///
    /// Rules:
    /// 1. Right-align dimensions.
    /// 2. Each position: dims are compatible if equal or one is 1.
    ///    If either dim is `DIM_DYNAMIC`, output dim is `DIM_DYNAMIC`.
    /// 3. Missing leading dims are treated as 1.
    /// 4. Output dim = max(a_dim, b_dim) at each position.
    pub fn broadcast(&self, other: &Shape) -> Result<Shape, String> {
        let a = &self.0;
        let b = &other.0;
        let max_rank = a.len().max(b.len());
        let mut result = Vec::with_capacity(max_rank);
        for i in 0..max_rank {
            let a_dim = if i < max_rank - a.len() {
                1
            } else {
                a[i - (max_rank - a.len())]
            };
            let b_dim = if i < max_rank - b.len() {
                1
            } else {
                b[i - (max_rank - b.len())]
            };
            // If either dim is dynamic, the output dim is also dynamic.
            if a_dim == DIM_DYNAMIC || b_dim == DIM_DYNAMIC {
                result.push(DIM_DYNAMIC);
            } else if a_dim == b_dim {
                result.push(a_dim);
            } else if a_dim == 1 {
                result.push(b_dim);
            } else if b_dim == 1 {
                result.push(a_dim);
            } else {
                return Err(format!(
                    "shapes {:?} and {:?} are not broadcast-compatible",
                    self.0, other.0
                ));
            }
        }
        Ok(Shape(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn num_elements() {
        assert_eq!(Shape(vec![4]).num_elements(), 4);
        assert_eq!(Shape(vec![2, 3]).num_elements(), 6);
        assert_eq!(Shape(vec![]).num_elements(), 1); // empty product
    }

    #[test]
    #[should_panic(expected = "num_elements() called on a shape with dynamic dims")]
    fn num_elements_panics_on_dynamic() {
        Shape(vec![4, DIM_DYNAMIC]).num_elements();
    }

    #[test]
    fn concrete_num_elements() {
        assert_eq!(Shape(vec![4]).concrete_num_elements(), Some(4));
        assert_eq!(Shape(vec![2, 3]).concrete_num_elements(), Some(6));
        assert_eq!(Shape(vec![4, DIM_DYNAMIC]).concrete_num_elements(), None);
        assert_eq!(Shape(vec![DIM_DYNAMIC]).concrete_num_elements(), None);
    }

    #[test]
    fn is_dynamic_dim() {
        let s = Shape(vec![4, DIM_DYNAMIC, 3]);
        assert!(!s.is_dynamic_dim(0));
        assert!(s.is_dynamic_dim(1));
        assert!(!s.is_dynamic_dim(2));
    }

    #[test]
    fn has_dynamic_dims() {
        assert!(!Shape(vec![4, 3]).has_dynamic_dims());
        assert!(Shape(vec![4, DIM_DYNAMIC]).has_dynamic_dims());
        assert!(!Shape(vec![]).has_dynamic_dims());
    }

    #[test]
    fn broadcast_with_dynamic() {
        // DIM_DYNAMIC on one side produces DIM_DYNAMIC output.
        let a = Shape(vec![4, DIM_DYNAMIC]);
        let b = Shape(vec![4, 3]);
        assert_eq!(a.broadcast(&b).unwrap(), Shape(vec![4, DIM_DYNAMIC]));

        // Both dynamic.
        let c = Shape(vec![DIM_DYNAMIC]);
        let d = Shape(vec![DIM_DYNAMIC]);
        assert_eq!(c.broadcast(&d).unwrap(), Shape(vec![DIM_DYNAMIC]));

        // Dynamic broadcast with size-1.
        let e = Shape(vec![1, DIM_DYNAMIC]);
        let f = Shape(vec![4, 64]);
        assert_eq!(e.broadcast(&f).unwrap(), Shape(vec![4, DIM_DYNAMIC]));
    }

    #[test]
    fn rank() {
        assert_eq!(Shape(vec![4]).rank(), 1);
        assert_eq!(Shape(vec![2, 3]).rank(), 2);
    }

    #[test]
    fn broadcast_same_shape() {
        let a = Shape(vec![4, 128]);
        let b = Shape(vec![4, 128]);
        assert_eq!(a.broadcast(&b).unwrap(), Shape(vec![4, 128]));
    }

    #[test]
    fn broadcast_trailing_dim() {
        // [4, 128] + [128] -> [4, 128]
        let a = Shape(vec![4, 128]);
        let b = Shape(vec![128]);
        assert_eq!(a.broadcast(&b).unwrap(), Shape(vec![4, 128]));
    }

    #[test]
    fn broadcast_mutual() {
        // [4, 1] + [1, 128] -> [4, 128]
        let a = Shape(vec![4, 1]);
        let b = Shape(vec![1, 128]);
        assert_eq!(a.broadcast(&b).unwrap(), Shape(vec![4, 128]));
    }

    #[test]
    fn broadcast_scalar_like() {
        // [4, 128] + [1] -> [4, 128]
        let a = Shape(vec![4, 128]);
        let b = Shape(vec![1]);
        assert_eq!(a.broadcast(&b).unwrap(), Shape(vec![4, 128]));
    }

    #[test]
    fn broadcast_incompatible() {
        let a = Shape(vec![4, 128]);
        let b = Shape(vec![3]);
        assert!(a.broadcast(&b).is_err());
    }

    #[test]
    fn broadcast_same_rank_size1_middle() {
        // [3, 1, 4] + [3, 5, 4] -> [3, 5, 4]
        let a = Shape(vec![3, 1, 4]);
        let b = Shape(vec![3, 5, 4]);
        assert_eq!(a.broadcast(&b).unwrap(), Shape(vec![3, 5, 4]));
    }
}
