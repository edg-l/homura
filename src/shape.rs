#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape(pub Vec<u64>);

impl Shape {
    pub fn num_elements(&self) -> u64 {
        self.0.iter().product()
    }

    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// Compute the numpy-style broadcast shape of `self` and `other`.
    ///
    /// Rules:
    /// 1. Right-align dimensions.
    /// 2. Each position: dims are compatible if equal or one is 1.
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
            if a_dim == b_dim {
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
