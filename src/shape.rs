/// Sentinel value representing a dynamic (unknown at compile time) dimension.
///
/// Chosen as `u64::MAX` for distinctness. Note that `u64::MAX as i64 == -1`,
/// which is NOT MLIR's `ShapedType::kDynamic` (= `i64::MIN`). The compiler
/// helper `dim_to_mlir` / `dim_to_mlir_i64` in `compiler.rs` translates this
/// sentinel to `i64::MIN` (kDynamic) before passing to MLIR C API calls such
/// as `RankedTensorType::new` and `MemRefType::new`.
pub const DIM_DYNAMIC: u64 = u64::MAX;

/// Convert a u64 dim to i64, preserving DIM_DYNAMIC as i64::MIN (MLIR's kDynamic).
/// This avoids collision with ONNX's -1 "infer this dim" convention.
pub fn dim_to_i64(d: u64) -> i64 {
    if d == DIM_DYNAMIC { i64::MIN } else { d as i64 }
}

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
        self.0.contains(&DIM_DYNAMIC)
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

/// Symbolic dimension expression for tracking dynamic shapes through compilation.
///
/// Used alongside MLIR's concrete/dynamic type system to propagate symbolic
/// relationships (e.g., `past_seq + 1`) through the op graph. At runtime,
/// variable bindings from actual input shapes are substituted to get concrete values.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SymDim {
    /// Known concrete value.
    Concrete(u64),
    /// Named variable from ONNX model input (e.g., "past_sequence_length").
    Var(String),
    /// Sum of two expressions.
    Add(Box<SymDim>, Box<SymDim>),
    /// Product of two expressions.
    Mul(Box<SymDim>, Box<SymDim>),
    /// Integer division (floor).
    Div(Box<SymDim>, Box<SymDim>),
}

/// Symbolic shape — one SymDim per tensor axis.
pub type SymShape = Vec<SymDim>;

impl SymDim {
    /// Convenience: create a variable.
    pub fn var(name: &str) -> Self {
        SymDim::Var(name.to_string())
    }

    /// Convenience: create a concrete value.
    pub fn concrete(v: u64) -> Self {
        SymDim::Concrete(v)
    }

    /// Build Add expression.
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, other: SymDim) -> SymDim {
        SymDim::Add(Box::new(self), Box::new(other))
    }

    /// Build Mul expression.
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, other: SymDim) -> SymDim {
        SymDim::Mul(Box::new(self), Box::new(other))
    }

    /// Build Div expression.
    #[allow(clippy::should_implement_trait)]
    pub fn div(self, other: SymDim) -> SymDim {
        SymDim::Div(Box::new(self), Box::new(other))
    }

    /// Returns true if this is a concrete value with no variables.
    pub fn is_concrete(&self) -> bool {
        match self {
            SymDim::Concrete(_) => true,
            SymDim::Var(_) => false,
            SymDim::Add(a, b) | SymDim::Mul(a, b) | SymDim::Div(a, b) => {
                a.is_concrete() && b.is_concrete()
            }
        }
    }

    /// Try to extract a concrete value (only if fully concrete).
    pub fn as_concrete(&self) -> Option<u64> {
        match self {
            SymDim::Concrete(v) => Some(*v),
            _ if self.is_concrete() => self.simplify().as_concrete(),
            _ => None,
        }
    }

    /// Evaluate with variable bindings. Returns None if any variable is unbound.
    pub fn eval(&self, bindings: &std::collections::HashMap<String, u64>) -> Option<u64> {
        match self {
            SymDim::Concrete(v) => Some(*v),
            SymDim::Var(name) => bindings.get(name).copied(),
            SymDim::Add(a, b) => {
                let av = a.eval(bindings)?;
                let bv = b.eval(bindings)?;
                Some(av + bv)
            }
            SymDim::Mul(a, b) => {
                let av = a.eval(bindings)?;
                let bv = b.eval(bindings)?;
                Some(av * bv)
            }
            SymDim::Div(a, b) => {
                let av = a.eval(bindings)?;
                let bv = b.eval(bindings)?;
                if bv == 0 { None } else { Some(av / bv) }
            }
        }
    }

    /// Simplify: fold concrete arithmetic, cancel common factors.
    /// Not a full CAS — handles patterns GPT-2 actually produces.
    pub fn simplify(&self) -> SymDim {
        match self {
            SymDim::Concrete(_) | SymDim::Var(_) => self.clone(),
            SymDim::Add(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (&a, &b) {
                    (SymDim::Concrete(x), SymDim::Concrete(y)) => SymDim::Concrete(x + y),
                    (SymDim::Concrete(0), other) | (other, SymDim::Concrete(0)) => other.clone(),
                    _ => SymDim::Add(Box::new(a), Box::new(b)),
                }
            }
            SymDim::Mul(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (&a, &b) {
                    (SymDim::Concrete(x), SymDim::Concrete(y)) => SymDim::Concrete(x * y),
                    (SymDim::Concrete(1), other) | (other, SymDim::Concrete(1)) => other.clone(),
                    (SymDim::Concrete(0), _) | (_, SymDim::Concrete(0)) => SymDim::Concrete(0),
                    _ => SymDim::Mul(Box::new(a), Box::new(b)),
                }
            }
            SymDim::Div(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (&a, &b) {
                    (SymDim::Concrete(x), SymDim::Concrete(y)) if *y != 0 => {
                        SymDim::Concrete(x / y)
                    }
                    (SymDim::Concrete(0), _) => SymDim::Concrete(0),
                    // Div(Mul(k, x), k) -> x  and  Div(Mul(x, k), k) -> x
                    (SymDim::Mul(ma, mb), divisor) => {
                        if **ma == *divisor {
                            mb.simplify()
                        } else if **mb == *divisor {
                            ma.simplify()
                        } else {
                            SymDim::Div(Box::new(a), Box::new(b))
                        }
                    }
                    _ => SymDim::Div(Box::new(a), Box::new(b)),
                }
            }
        }
    }
}

impl std::fmt::Display for SymDim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SymDim::Concrete(v) => write!(f, "{v}"),
            SymDim::Var(name) => write!(f, "{name}"),
            SymDim::Add(a, b) => write!(f, "({a} + {b})"),
            SymDim::Mul(a, b) => write!(f, "({a} * {b})"),
            SymDim::Div(a, b) => write!(f, "({a} / {b})"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

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

    #[test]
    fn sym_dim_eval_concrete() {
        let d = SymDim::Concrete(42);
        assert_eq!(d.eval(&HashMap::new()), Some(42));
    }

    #[test]
    fn sym_dim_eval_var() {
        let d = SymDim::var("past_seq");
        let mut bindings = HashMap::new();
        bindings.insert("past_seq".to_string(), 10);
        assert_eq!(d.eval(&bindings), Some(10));
        assert_eq!(d.eval(&HashMap::new()), None);
    }

    #[test]
    fn sym_dim_eval_add() {
        let d = SymDim::var("past_seq").add(SymDim::Concrete(1));
        let mut bindings = HashMap::new();
        bindings.insert("past_seq".to_string(), 5);
        assert_eq!(d.eval(&bindings), Some(6));
    }

    #[test]
    fn sym_dim_eval_complex() {
        // past_seq + seq_len (two variables)
        let d = SymDim::var("past_seq").add(SymDim::var("seq_len"));
        let mut bindings = HashMap::new();
        bindings.insert("past_seq".to_string(), 5);
        bindings.insert("seq_len".to_string(), 8);
        assert_eq!(d.eval(&bindings), Some(13));
    }

    #[test]
    fn sym_dim_simplify_concrete_fold() {
        let d = SymDim::Concrete(3).add(SymDim::Concrete(4));
        assert_eq!(d.simplify(), SymDim::Concrete(7));
    }

    #[test]
    fn sym_dim_simplify_add_zero() {
        let d = SymDim::var("x").add(SymDim::Concrete(0));
        assert_eq!(d.simplify(), SymDim::var("x"));
    }

    #[test]
    fn sym_dim_simplify_mul_one() {
        let d = SymDim::Concrete(1).mul(SymDim::var("x"));
        assert_eq!(d.simplify(), SymDim::var("x"));
    }

    #[test]
    fn sym_dim_simplify_div_cancel() {
        // Div(Mul(768, x), 768) -> x
        let d = SymDim::Concrete(768)
            .mul(SymDim::var("x"))
            .div(SymDim::Concrete(768));
        assert_eq!(d.simplify(), SymDim::var("x"));
    }

    #[test]
    fn sym_dim_display() {
        let d = SymDim::var("past_seq").add(SymDim::Concrete(1));
        assert_eq!(format!("{d}"), "(past_seq + 1)");
    }

    #[test]
    fn sym_dim_is_concrete() {
        assert!(SymDim::Concrete(5).is_concrete());
        assert!(!SymDim::var("x").is_concrete());
        assert!(SymDim::Concrete(3).add(SymDim::Concrete(4)).is_concrete());
        assert!(!SymDim::Concrete(3).add(SymDim::var("x")).is_concrete());
    }
}
