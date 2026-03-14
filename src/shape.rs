#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape(pub Vec<u64>);

impl Shape {
    pub fn num_elements(&self) -> u64 {
        self.0.iter().product()
    }

    pub fn rank(&self) -> usize {
        self.0.len()
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
}
