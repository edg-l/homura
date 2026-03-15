use std::slice;

use melior::ExecutionEngine;

use crate::{DType, Shape};

// ── Buffer ────────────────────────────────────────────────────────────────────

/// A type-erased, owned tensor buffer. Stores raw bytes with an associated
/// shape, dtype, and row-major strides.
#[derive(Debug, Clone)]
pub struct Buffer {
    pub(crate) data: Vec<u8>,
    shape: Shape,
    strides: Vec<i64>,
    dtype: DType,
}

impl Buffer {
    /// Allocate a zero-initialised buffer for the given shape and dtype.
    ///
    /// Strides are row-major: for shape [a, b, c] → strides = [b*c, c, 1].
    pub fn new(shape: &[u64], dtype: DType) -> Self {
        let s = Shape(shape.to_vec());
        let strides = row_major_strides(shape);
        let num_bytes = s.num_elements() as usize * dtype.size_bytes();
        Self {
            data: vec![0u8; num_bytes],
            shape: s,
            strides,
            dtype,
        }
    }

    /// Build a Buffer by copying typed data into raw bytes.
    ///
    /// # Panics
    ///
    /// Panics if `data.len()` does not equal the product of `shape`.
    pub fn from_slice<T: Copy + 'static>(data: &[T], shape: &[u64], dtype: DType) -> Self {
        assert_eq!(
            std::mem::size_of::<T>(),
            dtype.size_bytes(),
            "type size {} does not match dtype {:?} (size {})",
            std::mem::size_of::<T>(),
            dtype,
            dtype.size_bytes(),
        );
        let num_elems: u64 = shape.iter().product();
        assert_eq!(
            data.len(),
            num_elems as usize,
            "data length {} does not match shape product {}",
            data.len(),
            num_elems,
        );
        let num_bytes = std::mem::size_of_val(data);
        let raw = unsafe { slice::from_raw_parts(data.as_ptr() as *const u8, num_bytes) };
        Self {
            data: raw.to_vec(),
            shape: Shape(shape.to_vec()),
            strides: row_major_strides(shape),
            dtype,
        }
    }

    /// Reinterpret the raw bytes as a typed slice.
    ///
    /// # Panics
    ///
    /// Panics if the buffer byte length is not a multiple of `size_of::<T>()`.
    pub fn as_slice<T: Copy + 'static>(&self) -> &[T] {
        let elem_size = std::mem::size_of::<T>();
        assert_eq!(
            elem_size,
            self.dtype.size_bytes(),
            "type size {} does not match dtype {:?} (size {})",
            elem_size,
            self.dtype,
            self.dtype.size_bytes(),
        );
        unsafe {
            slice::from_raw_parts(self.data.as_ptr() as *const T, self.data.len() / elem_size)
        }
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn strides(&self) -> &[i64] {
        &self.strides
    }

    /// Mutable access to the raw byte storage.
    pub(crate) fn data_mut(&mut self) -> &mut Vec<u8> {
        &mut self.data
    }
}

/// Compute row-major strides for `shape`. For a scalar (rank 0) returns `[]`.
fn row_major_strides(shape: &[u64]) -> Vec<i64> {
    let n = shape.len();
    if n == 0 {
        return vec![];
    }
    let mut strides = vec![1i64; n];
    for i in (0..n - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1] as i64;
    }
    strides
}

// ── Memref descriptor ─────────────────────────────────────────────────────────

/// Build a rank-N MLIR memref descriptor as a raw byte blob.
///
/// Layout:
/// - bytes  0– 7: allocated_ptr (i64-sized pointer)
/// - bytes  8–15: aligned_ptr   (same as allocated)
/// - bytes 16–23: offset        (i64, always 0)
/// - bytes 24 .. 24+8N: sizes[0..N]
/// - bytes 24+8N .. 24+16N: strides[0..N]
///
/// Total: 24 + 16*N bytes.
pub(crate) fn build_memref_descriptor(
    data_ptr: *mut u8,
    shape: &[i64],
    strides: &[i64],
) -> Vec<u8> {
    assert_eq!(shape.len(), strides.len());
    let n = shape.len();
    let total = 24 + 16 * n;
    let mut buf = vec![0u8; total];

    let ptr_val = data_ptr as u64;
    buf[0..8].copy_from_slice(&ptr_val.to_ne_bytes()); // allocated_ptr
    buf[8..16].copy_from_slice(&ptr_val.to_ne_bytes()); // aligned_ptr
    buf[16..24].copy_from_slice(&0i64.to_ne_bytes()); // offset = 0

    for (i, &s) in shape.iter().enumerate() {
        let off = 24 + i * 8;
        buf[off..off + 8].copy_from_slice(&s.to_ne_bytes());
    }
    for (i, &s) in strides.iter().enumerate() {
        let off = 24 + n * 8 + i * 8;
        buf[off..off + 8].copy_from_slice(&s.to_ne_bytes());
    }

    buf
}

// ── CompiledGraph ─────────────────────────────────────────────────────────────

/// A compiled computation graph, ready to execute.
pub struct CompiledGraph {
    engine: ExecutionEngine,
    num_inputs: usize,
    output_shape: Shape,
    output_dtype: DType,
}

impl CompiledGraph {
    pub(crate) fn new(
        engine: ExecutionEngine,
        num_inputs: usize,
        output_shape: Shape,
        output_dtype: DType,
    ) -> Self {
        Self {
            engine,
            num_inputs,
            output_shape,
            output_dtype,
        }
    }

    /// Execute the graph with the given input `Buffer`s. Returns the output as
    /// an owned `Buffer`.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - The number of inputs doesn't match the compiled graph.
    /// - Any input dtype doesn't match the graph's output dtype.
    /// - The JIT invocation fails (mismatched ABI).
    pub fn run(&self, inputs: &[&Buffer]) -> Buffer {
        assert_eq!(
            inputs.len(),
            self.num_inputs,
            "expected {} inputs, got {}",
            self.num_inputs,
            inputs.len()
        );

        // Allocate output buffer (must exist before descriptor is built).
        let mut output = Buffer::new(self.output_shape.0.as_slice(), self.output_dtype);

        // Build memref descriptors for inputs. Inputs are immutable references,
        // but the JIT only reads them, so the const→mut cast is safe.
        let input_shapes: Vec<Vec<i64>> = inputs
            .iter()
            .map(|b| b.shape().0.iter().map(|&d| d as i64).collect())
            .collect();
        let input_strides: Vec<&[i64]> = inputs.iter().map(|b| b.strides()).collect();

        let mut input_descs: Vec<Vec<u8>> = inputs
            .iter()
            .zip(input_shapes.iter())
            .zip(input_strides.iter())
            .map(|((buf, shape), strides)| {
                build_memref_descriptor(buf.data.as_ptr() as *mut u8, shape.as_slice(), strides)
            })
            .collect();

        let output_shape_i64: Vec<i64> = self.output_shape.0.iter().map(|&d| d as i64).collect();
        let output_strides: Vec<i64> = output.strides().to_vec();
        let mut output_desc =
            build_memref_descriptor(output.data.as_mut_ptr(), &output_shape_i64, &output_strides);

        // Build args: each args[i] = &mut (ptr_to_descriptor) cast to *mut ().
        // invoke_packed dereferences each args[i] as a void* to get the
        // pointer passed to the MLIR C-interface wrapper, which then
        // dereferences that pointer to get the MemRefDescriptor struct.
        let mut desc_ptrs: Vec<*mut u8> = input_descs.iter_mut().map(|d| d.as_mut_ptr()).collect();
        let mut output_desc_ptr = output_desc.as_mut_ptr();

        let mut args: Vec<*mut ()> = desc_ptrs
            .iter_mut()
            .map(|p| p as *mut *mut u8 as *mut ())
            .collect();
        args.push(&mut output_desc_ptr as *mut *mut u8 as *mut ());

        unsafe {
            self.engine
                .invoke_packed("compute", &mut args)
                .expect("JIT invocation failed");
        }

        output
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::{
        Compiler, DType,
        runtime::Buffer,
        tensor::Tensor,
        trace::{begin_trace, take_trace},
    };

    // ── Buffer unit tests (task 1.3) ──────────────────────────────────────────

    #[test]
    fn buffer_new_f32() {
        let b = Buffer::new(&[4], DType::F32);
        assert_eq!(b.dtype(), DType::F32);
        assert_eq!(b.shape().0, vec![4]);
        assert_eq!(b.strides(), &[1i64]);
        assert_eq!(b.as_slice::<f32>(), &[0.0f32; 4]);
    }

    #[test]
    fn buffer_new_f64() {
        let b = Buffer::new(&[3], DType::F64);
        assert_eq!(b.dtype(), DType::F64);
        assert_eq!(b.as_slice::<f64>(), &[0.0f64; 3]);
    }

    #[test]
    fn buffer_new_i32() {
        let b = Buffer::new(&[2], DType::I32);
        assert_eq!(b.dtype(), DType::I32);
        assert_eq!(b.as_slice::<i32>(), &[0i32; 2]);
    }

    #[test]
    fn buffer_new_i64() {
        let b = Buffer::new(&[5], DType::I64);
        assert_eq!(b.dtype(), DType::I64);
        assert_eq!(b.as_slice::<i64>(), &[0i64; 5]);
    }

    #[test]
    fn buffer_from_slice_f32() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = Buffer::from_slice::<f32>(&data, &[4], DType::F32);
        assert_eq!(b.as_slice::<f32>(), data.as_slice());
    }

    #[test]
    fn buffer_from_slice_f64() {
        let data = vec![1.0f64, -2.5, 3.14];
        let b = Buffer::from_slice::<f64>(&data, &[3], DType::F64);
        assert_eq!(b.as_slice::<f64>(), data.as_slice());
    }

    #[test]
    fn buffer_from_slice_i32() {
        let data = vec![10i32, 20, 30];
        let b = Buffer::from_slice::<i32>(&data, &[3], DType::I32);
        assert_eq!(b.as_slice::<i32>(), data.as_slice());
    }

    #[test]
    fn buffer_from_slice_i64() {
        let data = vec![100i64, 200];
        let b = Buffer::from_slice::<i64>(&data, &[2], DType::I64);
        assert_eq!(b.as_slice::<i64>(), data.as_slice());
    }

    #[test]
    fn buffer_strides_2d() {
        // shape [2, 3] → strides [3, 1]
        let b = Buffer::new(&[2, 3], DType::F32);
        assert_eq!(b.strides(), &[3i64, 1i64]);
    }

    #[test]
    fn buffer_strides_3d() {
        // shape [2, 3, 4] → strides [12, 4, 1]
        let b = Buffer::new(&[2, 3, 4], DType::F32);
        assert_eq!(b.strides(), &[12i64, 4i64, 1i64]);
    }

    // ── Integration tests using Buffer API (task 1.6) ─────────────────────────

    #[test]
    fn run_add() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F32);
        let b = Tensor::new(&[4], DType::F32);
        let c = &a + &b;
        let trace = take_trace();

        let compiled = Compiler::compile(&trace, &[c.id]).expect("compile failed");
        let a_buf = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[4], DType::F32);
        let b_buf = Buffer::from_slice::<f32>(&[5.0, 6.0, 7.0, 8.0], &[4], DType::F32);
        let result = compiled.run(&[&a_buf, &b_buf]);
        assert_eq!(result.as_slice::<f32>(), &[6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn run_chained_add() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F32);
        let b = Tensor::new(&[4], DType::F32);
        let c = Tensor::new(&[4], DType::F32);
        let d = &(&a + &b) + &c;
        let trace = take_trace();

        let compiled = Compiler::compile(&trace, &[d.id]).expect("compile failed");
        let a_buf = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[4], DType::F32);
        let b_buf = Buffer::from_slice::<f32>(&[10.0, 20.0, 30.0, 40.0], &[4], DType::F32);
        let c_buf = Buffer::from_slice::<f32>(&[100.0, 200.0, 300.0, 400.0], &[4], DType::F32);
        let result = compiled.run(&[&a_buf, &b_buf, &c_buf]);
        assert_eq!(result.as_slice::<f32>(), &[111.0, 222.0, 333.0, 444.0]);
    }

    #[test]
    fn run_larger_vector() {
        begin_trace();
        let a = Tensor::new(&[8], DType::F32);
        let b = Tensor::new(&[8], DType::F32);
        let c = &a + &b;
        let trace = take_trace();

        let compiled = Compiler::compile(&trace, &[c.id]).expect("compile failed");
        let a_data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..8).map(|i| (i * 2) as f32).collect();
        let a_buf = Buffer::from_slice::<f32>(&a_data, &[8], DType::F32);
        let b_buf = Buffer::from_slice::<f32>(&b_data, &[8], DType::F32);
        let result = compiled.run(&[&a_buf, &b_buf]);

        let expected: Vec<f32> = (0..8).map(|i| (i + i * 2) as f32).collect();
        assert_eq!(result.as_slice::<f32>(), expected.as_slice());
    }

    #[test]
    #[should_panic(expected = "expected 2 inputs")]
    fn wrong_input_count_panics() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F32);
        let b = Tensor::new(&[4], DType::F32);
        let c = &a + &b;
        let trace = take_trace();

        let compiled = Compiler::compile(&trace, &[c.id]).expect("compile failed");
        let a_buf = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[4], DType::F32);
        compiled.run(&[&a_buf]); // missing second input
    }

    #[test]
    fn run_sub() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F32);
        let b = Tensor::new(&[4], DType::F32);
        let c = &a - &b;
        let trace = take_trace();

        let compiled = Compiler::compile(&trace, &[c.id]).expect("compile failed");
        let a_buf = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[4], DType::F32);
        let b_buf = Buffer::from_slice::<f32>(&[4.0, 3.0, 2.0, 1.0], &[4], DType::F32);
        let result = compiled.run(&[&a_buf, &b_buf]);
        assert_eq!(result.as_slice::<f32>(), &[-3.0, -1.0, 1.0, 3.0]);
    }

    #[test]
    fn run_mul() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F32);
        let b = Tensor::new(&[4], DType::F32);
        let c = &a * &b;
        let trace = take_trace();

        let compiled = Compiler::compile(&trace, &[c.id]).expect("compile failed");
        let a_buf = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[4], DType::F32);
        let b_buf = Buffer::from_slice::<f32>(&[5.0, 6.0, 7.0, 8.0], &[4], DType::F32);
        let result = compiled.run(&[&a_buf, &b_buf]);
        assert_eq!(result.as_slice::<f32>(), &[5.0, 12.0, 21.0, 32.0]);
    }

    #[test]
    fn run_div() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F32);
        let b = Tensor::new(&[4], DType::F32);
        let c = &a / &b;
        let trace = take_trace();

        let compiled = Compiler::compile(&trace, &[c.id]).expect("compile failed");
        let a_buf = Buffer::from_slice::<f32>(&[10.0, 20.0, 30.0, 40.0], &[4], DType::F32);
        let b_buf = Buffer::from_slice::<f32>(&[2.0, 4.0, 5.0, 8.0], &[4], DType::F32);
        let result = compiled.run(&[&a_buf, &b_buf]);
        assert_eq!(result.as_slice::<f32>(), &[5.0, 5.0, 6.0, 5.0]);
    }

    #[test]
    fn run_neg() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F32);
        let b = -&a;
        let trace = take_trace();

        let compiled = Compiler::compile(&trace, &[b.id]).expect("compile failed");
        let a_buf = Buffer::from_slice::<f32>(&[1.0, -2.0, 3.0, -4.0], &[4], DType::F32);
        let result = compiled.run(&[&a_buf]);
        assert_eq!(result.as_slice::<f32>(), &[-1.0, 2.0, -3.0, 4.0]);
    }

    #[test]
    fn run_relu() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F32);
        let b = a.relu();
        let trace = take_trace();

        let compiled = Compiler::compile(&trace, &[b.id]).expect("compile failed");
        let a_buf = Buffer::from_slice::<f32>(&[-1.0, 2.0, -3.0, 4.0], &[4], DType::F32);
        let result = compiled.run(&[&a_buf]);
        assert_eq!(result.as_slice::<f32>(), &[0.0, 2.0, 0.0, 4.0]);
    }

    #[test]
    fn run_chained_relu() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F32);
        let b = Tensor::new(&[4], DType::F32);
        let c = (&a + &b).relu();
        let trace = take_trace();

        let compiled = Compiler::compile(&trace, &[c.id]).expect("compile failed");
        // [1,-5,3,-7] + [2,3,-4,5] = [3,-2,-1,-2] -> relu -> [3,0,0,0]
        let a_buf = Buffer::from_slice::<f32>(&[1.0, -5.0, 3.0, -7.0], &[4], DType::F32);
        let b_buf = Buffer::from_slice::<f32>(&[2.0, 3.0, -4.0, 5.0], &[4], DType::F32);
        let result = compiled.run(&[&a_buf, &b_buf]);
        assert_eq!(result.as_slice::<f32>(), &[3.0, 0.0, 0.0, 0.0]);
    }

    // ── Rank-2 and rank-3 integration tests ──────────────────────────────────

    #[test]
    fn run_add_rank2() {
        begin_trace();
        let a = Tensor::new(&[2, 3], DType::F32);
        let b = Tensor::new(&[2, 3], DType::F32);
        let c = &a + &b;
        let trace = take_trace();

        let compiled = Compiler::compile(&trace, &[c.id]).expect("compile failed");
        let a_buf = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
        let b_buf =
            Buffer::from_slice::<f32>(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[2, 3], DType::F32);
        let result = compiled.run(&[&a_buf, &b_buf]);
        assert_eq!(
            result.as_slice::<f32>(),
            &[11.0, 22.0, 33.0, 44.0, 55.0, 66.0]
        );
    }

    #[test]
    fn run_add_rank3() {
        begin_trace();
        let a = Tensor::new(&[2, 2, 2], DType::F32);
        let b = Tensor::new(&[2, 2, 2], DType::F32);
        let c = &a + &b;
        let trace = take_trace();

        let compiled = Compiler::compile(&trace, &[c.id]).expect("compile failed");
        let a_buf = Buffer::from_slice::<f32>(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 2, 2],
            DType::F32,
        );
        let b_buf = Buffer::from_slice::<f32>(
            &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            &[2, 2, 2],
            DType::F32,
        );
        let result = compiled.run(&[&a_buf, &b_buf]);
        assert_eq!(
            result.as_slice::<f32>(),
            &[11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0]
        );
    }

    // ── Matmul integration tests (tasks 3.4, 3.5, 3.6) ───────────────────────

    #[test]
    fn run_matmul_f32() {
        begin_trace();
        let a = Tensor::new(&[2, 3], DType::F32);
        let b = Tensor::new(&[3, 4], DType::F32);
        let c = a.matmul(&b);
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[c.id]).expect("compile failed");

        // A = [[1, 2, 3], [4, 5, 6]] (2x3)
        // B = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]] (3x4)
        // C = A @ B = [[38, 44, 50, 56], [83, 98, 113, 128]] (2x4)
        let a_buf = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
        let b_buf = Buffer::from_slice::<f32>(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[3, 4],
            DType::F32,
        );
        let result = compiled.run(&[&a_buf, &b_buf]);
        assert_eq!(
            result.as_slice::<f32>(),
            &[38.0, 44.0, 50.0, 56.0, 83.0, 98.0, 113.0, 128.0]
        );
    }

    #[test]
    fn run_matmul_f64() {
        begin_trace();
        let a = Tensor::new(&[2, 3], DType::F64);
        let b = Tensor::new(&[3, 4], DType::F64);
        let c = a.matmul(&b);
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[c.id]).expect("compile failed");

        // Same values as f32 test
        let a_buf = Buffer::from_slice::<f64>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F64);
        let b_buf = Buffer::from_slice::<f64>(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[3, 4],
            DType::F64,
        );
        let result = compiled.run(&[&a_buf, &b_buf]);
        assert_eq!(
            result.as_slice::<f64>(),
            &[38.0, 44.0, 50.0, 56.0, 83.0, 98.0, 113.0, 128.0]
        );
    }

    // Task 3.6: i32 matmul works because our codegen provides an explicit body
    // using arith.muli + arith.addi rather than relying on the linalg.matmul
    // named op's implicit body (which is float-only). Integer matmul compiles
    // and produces correct results.
    #[test]
    fn run_matmul_i32() {
        begin_trace();
        let a = Tensor::new(&[2, 3], DType::I32);
        let b = Tensor::new(&[3, 4], DType::I32);
        let c = a.matmul(&b);
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[c.id]).expect("compile failed");

        // A = [[1, 2, 3], [4, 5, 6]] (2x3)
        // B = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]] (3x4)
        // C = A @ B = [[38, 44, 50, 56], [83, 98, 113, 128]] (2x4)
        let a_buf = Buffer::from_slice::<i32>(&[1, 2, 3, 4, 5, 6], &[2, 3], DType::I32);
        let b_buf = Buffer::from_slice::<i32>(
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            &[3, 4],
            DType::I32,
        );
        let result = compiled.run(&[&a_buf, &b_buf]);
        assert_eq!(
            result.as_slice::<i32>(),
            &[38, 44, 50, 56, 83, 98, 113, 128]
        );
    }

    #[test]
    fn run_exp() {
        begin_trace();
        let a = Tensor::new(&[3], DType::F32);
        let b = a.exp();
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[b.id]).expect("compile failed");
        let a_buf = Buffer::from_slice::<f32>(&[0.0, 1.0, 2.0], &[3], DType::F32);
        let result = compiled.run(&[&a_buf]);
        let out = result.as_slice::<f32>();
        // exp(0) = 1.0, exp(1) ≈ 2.718, exp(2) ≈ 7.389
        assert!((out[0] - 1.0).abs() < 1e-5);
        assert!((out[1] - std::f32::consts::E).abs() < 1e-5);
        assert!((out[2] - std::f32::consts::E * std::f32::consts::E).abs() < 1e-4);
    }

    #[test]
    fn run_tanh() {
        begin_trace();
        let a = Tensor::new(&[3], DType::F32);
        let b = a.tanh();
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[b.id]).expect("compile failed");
        let a_buf = Buffer::from_slice::<f32>(&[-1.0, 0.0, 1.0], &[3], DType::F32);
        let result = compiled.run(&[&a_buf]);
        let out = result.as_slice::<f32>();
        // tanh(-1) ≈ -0.7616, tanh(0) = 0.0, tanh(1) ≈ 0.7616
        assert!((out[0] - (-0.7615942)).abs() < 1e-5);
        assert!((out[1] - 0.0).abs() < 1e-5);
        assert!((out[2] - 0.7615942).abs() < 1e-5);
    }

    #[test]
    fn run_neg_rank2() {
        begin_trace();
        let a = Tensor::new(&[2, 3], DType::F32);
        let b = -&a;
        let trace = take_trace();

        let compiled = Compiler::compile(&trace, &[b.id]).expect("compile failed");
        let a_buf =
            Buffer::from_slice::<f32>(&[1.0, -2.0, 3.0, -4.0, 5.0, -6.0], &[2, 3], DType::F32);
        let result = compiled.run(&[&a_buf]);
        assert_eq!(result.as_slice::<f32>(), &[-1.0, 2.0, -3.0, 4.0, -5.0, 6.0]);
    }

    // ── Broadcast integration tests (task 4.6) ───────────────────────────────

    #[test]
    fn run_add_broadcast_bias() {
        // [2, 3] + [3] -> [2, 3]
        begin_trace();
        let a = Tensor::new(&[2, 3], DType::F32);
        let b = Tensor::new(&[3], DType::F32);
        let c = &a + &b;
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[c.id]).expect("compile failed");
        let a_buf = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
        let b_buf = Buffer::from_slice::<f32>(&[10.0, 20.0, 30.0], &[3], DType::F32);
        let result = compiled.run(&[&a_buf, &b_buf]);
        assert_eq!(
            result.as_slice::<f32>(),
            &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]
        );
    }

    #[test]
    fn run_add_broadcast_mutual() {
        // [2, 1] + [1, 3] -> [2, 3]
        begin_trace();
        let a = Tensor::new(&[2, 1], DType::F32);
        let b = Tensor::new(&[1, 3], DType::F32);
        let c = &a + &b;
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[c.id]).expect("compile failed");
        let a_buf = Buffer::from_slice::<f32>(&[1.0, 2.0], &[2, 1], DType::F32);
        let b_buf = Buffer::from_slice::<f32>(&[10.0, 20.0, 30.0], &[1, 3], DType::F32);
        let result = compiled.run(&[&a_buf, &b_buf]);
        assert_eq!(
            result.as_slice::<f32>(),
            &[11.0, 21.0, 31.0, 12.0, 22.0, 32.0]
        );
    }

    #[test]
    fn run_add_broadcast_rank3() {
        // [3, 1, 4] + [3, 5, 4] -> [3, 5, 4]
        begin_trace();
        let a = Tensor::new(&[3, 1, 4], DType::F32);
        let b = Tensor::new(&[3, 5, 4], DType::F32);
        let c = &a + &b;
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[c.id]).expect("compile failed");
        // a[i, 0, j] is broadcast across dim-1; values are 1..12 row-major
        let a_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let a_buf = Buffer::from_slice::<f32>(&a_data, &[3, 1, 4], DType::F32);
        // b: all 100.0
        let b_data: Vec<f32> = vec![100.0; 60];
        let b_buf = Buffer::from_slice::<f32>(&b_data, &[3, 5, 4], DType::F32);
        let result = compiled.run(&[&a_buf, &b_buf]);
        // Expected: for outer in 0..3, repeat a_row 5 times each plus 100.0
        let expected: Vec<f32> = (0..3_usize)
            .flat_map(|outer| {
                let base = (outer * 4) as f32;
                let a_row = [base + 1.0, base + 2.0, base + 3.0, base + 4.0];
                (0..5)
                    .flat_map(move |_| a_row.iter().map(|&v| v + 100.0).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(result.as_slice::<f32>(), expected.as_slice());
    }

    // ── Reduction integration tests (tasks 6.4 – 6.7) ────────────────────────

    #[test]
    fn run_reduce_sum_dim0() {
        begin_trace();
        let a = Tensor::new(&[2, 3], DType::F32);
        let b = a.reduce_sum(0, false);
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[b.id]).expect("compile failed");
        // [[1,2,3],[4,5,6]] -> sum along dim 0 -> [5, 7, 9]
        let a_buf = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
        let result = compiled.run(&[&a_buf]);
        assert_eq!(result.as_slice::<f32>(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn run_reduce_sum_last_dim() {
        begin_trace();
        let a = Tensor::new(&[2, 3], DType::F32);
        let b = a.reduce_sum(-1, false);
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[b.id]).expect("compile failed");
        // [[1,2,3],[4,5,6]] -> sum along dim -1 (=1) -> [6, 15]
        let a_buf = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
        let result = compiled.run(&[&a_buf]);
        assert_eq!(result.as_slice::<f32>(), &[6.0, 15.0]);
    }

    #[test]
    fn run_reduce_sum_keepdim() {
        use crate::Shape;
        begin_trace();
        let a = Tensor::new(&[2, 3], DType::F32);
        let b = a.reduce_sum(-1, true);
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[b.id]).expect("compile failed");
        // [[1,2,3],[4,5,6]] -> sum along dim -1 keepdim -> [[6], [15]]
        let a_buf = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
        let result = compiled.run(&[&a_buf]);
        assert_eq!(result.shape(), &Shape(vec![2, 1]));
        assert_eq!(result.as_slice::<f32>(), &[6.0, 15.0]);
    }

    #[test]
    fn run_reduce_max() {
        begin_trace();
        let a = Tensor::new(&[2, 3], DType::F32);
        let b = a.reduce_max(-1, false);
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[b.id]).expect("compile failed");
        // [[1,3,2],[6,4,5]] -> max along dim -1 -> [3, 6]
        let a_buf = Buffer::from_slice::<f32>(&[1.0, 3.0, 2.0, 6.0, 4.0, 5.0], &[2, 3], DType::F32);
        let result = compiled.run(&[&a_buf]);
        assert_eq!(result.as_slice::<f32>(), &[3.0, 6.0]);
    }

    #[test]
    fn run_reduce_max_all_negative() {
        begin_trace();
        let a = Tensor::new(&[2, 3], DType::F32);
        let b = a.reduce_max(-1, false);
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[b.id]).expect("compile failed");
        // [[-5,-3,-1],[-9,-7,-4]] -> max along dim -1 -> [-1, -4]  (proves init is -inf, not 0)
        let a_buf =
            Buffer::from_slice::<f32>(&[-5.0, -3.0, -1.0, -9.0, -7.0, -4.0], &[2, 3], DType::F32);
        let result = compiled.run(&[&a_buf]);
        assert_eq!(result.as_slice::<f32>(), &[-1.0, -4.0]);
    }

    #[test]
    fn run_reduce_max_i32() {
        begin_trace();
        let a = Tensor::new(&[2, 3], DType::I32);
        let b = a.reduce_max(-1, false);
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[b.id]).expect("compile failed");
        // [[-5, -3, -1], [-10, -7, -4]] -> max along last dim -> [-1, -4]
        let a_buf = Buffer::from_slice::<i32>(&[-5, -3, -1, -10, -7, -4], &[2, 3], DType::I32);
        let result = compiled.run(&[&a_buf]);
        assert_eq!(result.as_slice::<i32>(), &[-1, -4]);
    }

    // ── F64 dtype coverage ───────────────────────────────────────────────────

    #[test]
    fn run_sub_f64() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F64);
        let b = Tensor::new(&[4], DType::F64);
        let c = &a - &b;
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[c.id]).expect("compile failed");
        let a_buf = Buffer::from_slice::<f64>(&[5.0, 6.0, 7.0, 8.0], &[4], DType::F64);
        let b_buf = Buffer::from_slice::<f64>(&[1.0, 2.0, 3.0, 4.0], &[4], DType::F64);
        let result = compiled.run(&[&a_buf, &b_buf]);
        assert_eq!(result.as_slice::<f64>(), &[4.0, 4.0, 4.0, 4.0]);
    }

    #[test]
    fn run_mul_f64() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F64);
        let b = Tensor::new(&[4], DType::F64);
        let c = &a * &b;
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[c.id]).expect("compile failed");
        let a_buf = Buffer::from_slice::<f64>(&[2.0, 3.0, 4.0, 5.0], &[4], DType::F64);
        let b_buf = Buffer::from_slice::<f64>(&[1.5, 2.0, 0.5, 3.0], &[4], DType::F64);
        let result = compiled.run(&[&a_buf, &b_buf]);
        assert_eq!(result.as_slice::<f64>(), &[3.0, 6.0, 2.0, 15.0]);
    }

    #[test]
    fn run_div_f64() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F64);
        let b = Tensor::new(&[4], DType::F64);
        let c = &a / &b;
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[c.id]).expect("compile failed");
        let a_buf = Buffer::from_slice::<f64>(&[10.0, 20.0, 30.0, 40.0], &[4], DType::F64);
        let b_buf = Buffer::from_slice::<f64>(&[2.0, 4.0, 5.0, 8.0], &[4], DType::F64);
        let result = compiled.run(&[&a_buf, &b_buf]);
        assert_eq!(result.as_slice::<f64>(), &[5.0, 5.0, 6.0, 5.0]);
    }

    #[test]
    fn run_neg_f64() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F64);
        let b = -&a;
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[b.id]).expect("compile failed");
        let a_buf = Buffer::from_slice::<f64>(&[1.0, -2.0, 3.0, -4.0], &[4], DType::F64);
        let result = compiled.run(&[&a_buf]);
        assert_eq!(result.as_slice::<f64>(), &[-1.0, 2.0, -3.0, 4.0]);
    }

    #[test]
    fn run_relu_f64() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F64);
        let b = a.relu();
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[b.id]).expect("compile failed");
        let a_buf = Buffer::from_slice::<f64>(&[-1.0, 0.0, 3.0, -4.0], &[4], DType::F64);
        let result = compiled.run(&[&a_buf]);
        assert_eq!(result.as_slice::<f64>(), &[0.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn run_exp_f64() {
        begin_trace();
        let a = Tensor::new(&[3], DType::F64);
        let b = a.exp();
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[b.id]).expect("compile failed");
        let a_buf = Buffer::from_slice::<f64>(&[0.0, 1.0, -1.0], &[3], DType::F64);
        let result = compiled.run(&[&a_buf]);
        let out = result.as_slice::<f64>();
        assert!((out[0] - 1.0).abs() < 1e-10);
        assert!((out[1] - std::f64::consts::E).abs() < 1e-10);
        assert!((out[2] - 1.0 / std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn run_tanh_f64() {
        begin_trace();
        let a = Tensor::new(&[3], DType::F64);
        let b = a.tanh();
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[b.id]).expect("compile failed");
        let a_buf = Buffer::from_slice::<f64>(&[0.0, 1.0, -1.0], &[3], DType::F64);
        let result = compiled.run(&[&a_buf]);
        let out = result.as_slice::<f64>();
        assert!(out[0].abs() < 1e-10);
        assert!((out[1] - 1.0_f64.tanh()).abs() < 1e-10);
        assert!((out[2] - (-1.0_f64).tanh()).abs() < 1e-10);
    }

    #[test]
    fn run_reduce_sum_f64() {
        begin_trace();
        let a = Tensor::new(&[2, 3], DType::F64);
        let b = a.reduce_sum(-1, false);
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[b.id]).expect("compile failed");
        let a_buf = Buffer::from_slice::<f64>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F64);
        let result = compiled.run(&[&a_buf]);
        assert_eq!(result.as_slice::<f64>(), &[6.0, 15.0]);
    }

    #[test]
    fn run_reduce_max_f64() {
        begin_trace();
        let a = Tensor::new(&[2, 3], DType::F64);
        let b = a.reduce_max(-1, false);
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[b.id]).expect("compile failed");
        let a_buf = Buffer::from_slice::<f64>(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3], DType::F64);
        let result = compiled.run(&[&a_buf]);
        assert_eq!(result.as_slice::<f64>(), &[5.0, 6.0]);
    }

    #[test]
    fn run_reshape_f64() {
        begin_trace();
        let a = Tensor::new(&[2, 3], DType::F64);
        let b = a.reshape(&[6]);
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[b.id]).expect("compile failed");
        let a_buf = Buffer::from_slice::<f64>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F64);
        let result = compiled.run(&[&a_buf]);
        assert_eq!(result.as_slice::<f64>(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // ── Broadcast tests for Sub, Mul, Div ────────────────────────────────────

    #[test]
    fn run_sub_broadcast() {
        begin_trace();
        let a = Tensor::new(&[2, 3], DType::F32);
        let b = Tensor::new(&[3], DType::F32);
        let c = &a - &b;
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[c.id]).expect("compile failed");
        let a_buf =
            Buffer::from_slice::<f32>(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[2, 3], DType::F32);
        let b_buf = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3], DType::F32);
        let result = compiled.run(&[&a_buf, &b_buf]);
        assert_eq!(
            result.as_slice::<f32>(),
            &[9.0, 18.0, 27.0, 39.0, 48.0, 57.0]
        );
    }

    #[test]
    fn run_mul_broadcast() {
        begin_trace();
        let a = Tensor::new(&[2, 3], DType::F32);
        let b = Tensor::new(&[3], DType::F32);
        let c = &a * &b;
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[c.id]).expect("compile failed");
        let a_buf = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
        let b_buf = Buffer::from_slice::<f32>(&[10.0, 20.0, 30.0], &[3], DType::F32);
        let result = compiled.run(&[&a_buf, &b_buf]);
        assert_eq!(
            result.as_slice::<f32>(),
            &[10.0, 40.0, 90.0, 40.0, 100.0, 180.0]
        );
    }

    #[test]
    fn run_div_broadcast() {
        begin_trace();
        let a = Tensor::new(&[2, 3], DType::F32);
        let b = Tensor::new(&[3], DType::F32);
        let c = &a / &b;
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[c.id]).expect("compile failed");
        let a_buf =
            Buffer::from_slice::<f32>(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[2, 3], DType::F32);
        let b_buf = Buffer::from_slice::<f32>(&[2.0, 5.0, 10.0], &[3], DType::F32);
        let result = compiled.run(&[&a_buf, &b_buf]);
        assert_eq!(result.as_slice::<f32>(), &[5.0, 4.0, 3.0, 20.0, 10.0, 6.0]);
    }

    // ── Edge value tests ─────────────────────────────────────────────────────

    #[test]
    fn run_div_by_zero_produces_inf() {
        begin_trace();
        let a = Tensor::new(&[3], DType::F32);
        let b = Tensor::new(&[3], DType::F32);
        let c = &a / &b;
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[c.id]).expect("compile failed");
        let a_buf = Buffer::from_slice::<f32>(&[1.0, -1.0, 0.0], &[3], DType::F32);
        let b_buf = Buffer::from_slice::<f32>(&[0.0, 0.0, 0.0], &[3], DType::F32);
        let result = compiled.run(&[&a_buf, &b_buf]);
        let out = result.as_slice::<f32>();
        assert!(out[0].is_infinite() && out[0] > 0.0, "1/0 should be +inf");
        assert!(out[1].is_infinite() && out[1] < 0.0, "-1/0 should be -inf");
        assert!(out[2].is_nan(), "0/0 should be NaN");
    }

    #[test]
    fn run_exp_large_input() {
        begin_trace();
        let a = Tensor::new(&[3], DType::F32);
        let b = a.exp();
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[b.id]).expect("compile failed");
        let a_buf = Buffer::from_slice::<f32>(&[100.0, -100.0, 0.0], &[3], DType::F32);
        let result = compiled.run(&[&a_buf]);
        let out = result.as_slice::<f32>();
        assert!(out[0].is_infinite(), "exp(100) should overflow to inf");
        assert!((out[1]).abs() < 1e-30, "exp(-100) should underflow to ~0");
        assert!((out[2] - 1.0).abs() < 1e-6, "exp(0) = 1");
    }

    #[test]
    fn run_tanh_extreme_values() {
        begin_trace();
        let a = Tensor::new(&[3], DType::F32);
        let b = a.tanh();
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[b.id]).expect("compile failed");
        let a_buf = Buffer::from_slice::<f32>(&[100.0, -100.0, 0.0], &[3], DType::F32);
        let result = compiled.run(&[&a_buf]);
        let out = result.as_slice::<f32>();
        assert!((out[0] - 1.0).abs() < 1e-6, "tanh(100) ≈ 1");
        assert!((out[1] + 1.0).abs() < 1e-6, "tanh(-100) ≈ -1");
        assert!(out[2].abs() < 1e-6, "tanh(0) = 0");
    }

    #[test]
    fn run_add_nan_propagation() {
        begin_trace();
        let a = Tensor::new(&[3], DType::F32);
        let b = Tensor::new(&[3], DType::F32);
        let c = &a + &b;
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[c.id]).expect("compile failed");
        let a_buf = Buffer::from_slice::<f32>(&[f32::NAN, 1.0, f32::INFINITY], &[3], DType::F32);
        let b_buf =
            Buffer::from_slice::<f32>(&[1.0, f32::NAN, f32::NEG_INFINITY], &[3], DType::F32);
        let result = compiled.run(&[&a_buf, &b_buf]);
        let out = result.as_slice::<f32>();
        assert!(out[0].is_nan(), "NaN + 1 = NaN");
        assert!(out[1].is_nan(), "1 + NaN = NaN");
        assert!(out[2].is_nan(), "inf + -inf = NaN");
    }

    // ── Single-element and identity tests ────────────────────────────────────

    #[test]
    fn run_matmul_1x1() {
        begin_trace();
        let a = Tensor::new(&[1, 1], DType::F32);
        let b = Tensor::new(&[1, 1], DType::F32);
        let c = a.matmul(&b);
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[c.id]).expect("compile failed");
        let a_buf = Buffer::from_slice::<f32>(&[3.0], &[1, 1], DType::F32);
        let b_buf = Buffer::from_slice::<f32>(&[5.0], &[1, 1], DType::F32);
        let result = compiled.run(&[&a_buf, &b_buf]);
        assert_eq!(result.as_slice::<f32>(), &[15.0]);
    }

    #[test]
    fn run_add_single_element() {
        begin_trace();
        let a = Tensor::new(&[1], DType::F32);
        let b = Tensor::new(&[1], DType::F32);
        let c = &a + &b;
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[c.id]).expect("compile failed");
        let a_buf = Buffer::from_slice::<f32>(&[3.0], &[1], DType::F32);
        let b_buf = Buffer::from_slice::<f32>(&[7.0], &[1], DType::F32);
        let result = compiled.run(&[&a_buf, &b_buf]);
        assert_eq!(result.as_slice::<f32>(), &[10.0]);
    }

    // ── Complex graph test ───────────────────────────────────────────────────

    #[test]
    fn run_matmul_bias_relu_reduce() {
        // x @ w + b → relu → reduce_sum: a multi-op chain
        begin_trace();
        let x = Tensor::new(&[2, 3], DType::F32);
        let w = Tensor::new(&[3, 4], DType::F32);
        let b = Tensor::new(&[4], DType::F32);
        let y = x.matmul(&w);
        let y = &y + &b;
        let y = y.relu();
        let y = y.reduce_sum(-1, false);
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[y.id]).expect("compile failed");

        // x = [[1,0,0],[0,1,0]], w = identity-like 3x4 (pad zeros), b = [0,0,0,-10]
        let x_buf = Buffer::from_slice::<f32>(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3], DType::F32);
        let w_buf = Buffer::from_slice::<f32>(
            &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            &[3, 4],
            DType::F32,
        );
        let b_buf = Buffer::from_slice::<f32>(&[0.0, 0.0, 0.0, -10.0], &[4], DType::F32);
        let result = compiled.run(&[&x_buf, &w_buf, &b_buf]);
        // x@w = [[1,0,0,0],[0,1,0,0]], + b = [[1,0,0,-10],[0,1,0,-10]]
        // relu = [[1,0,0,0],[0,1,0,0]], sum = [1, 1]
        assert_eq!(result.as_slice::<f32>(), &[1.0, 1.0]);
    }

    // ── I64 dtype tests ──────────────────────────────────────────────────────

    #[test]
    fn run_add_i64() {
        begin_trace();
        let a = Tensor::new(&[3], DType::I64);
        let b = Tensor::new(&[3], DType::I64);
        let c = &a + &b;
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[c.id]).expect("compile failed");
        let a_buf = Buffer::from_slice::<i64>(&[1_000_000_000, -500, 0], &[3], DType::I64);
        let b_buf = Buffer::from_slice::<i64>(&[1_000_000_000, 500, 0], &[3], DType::I64);
        let result = compiled.run(&[&a_buf, &b_buf]);
        assert_eq!(result.as_slice::<i64>(), &[2_000_000_000, 0, 0]);
    }

    #[test]
    fn run_neg_i64() {
        begin_trace();
        let a = Tensor::new(&[3], DType::I64);
        let b = -&a;
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[b.id]).expect("compile failed");
        let a_buf = Buffer::from_slice::<i64>(&[100, -200, 0], &[3], DType::I64);
        let result = compiled.run(&[&a_buf]);
        assert_eq!(result.as_slice::<i64>(), &[-100, 200, 0]);
    }

    #[test]
    fn run_matmul_i64() {
        begin_trace();
        let a = Tensor::new(&[2, 2], DType::I64);
        let b = Tensor::new(&[2, 2], DType::I64);
        let c = a.matmul(&b);
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[c.id]).expect("compile failed");
        let a_buf = Buffer::from_slice::<i64>(&[1, 2, 3, 4], &[2, 2], DType::I64);
        let b_buf = Buffer::from_slice::<i64>(&[5, 6, 7, 8], &[2, 2], DType::I64);
        let result = compiled.run(&[&a_buf, &b_buf]);
        assert_eq!(result.as_slice::<i64>(), &[19, 22, 43, 50]);
    }

    // ── I32 reduce tests ─────────────────────────────────────────────────────

    #[test]
    fn run_reduce_sum_i32() {
        begin_trace();
        let a = Tensor::new(&[2, 3], DType::I32);
        let b = a.reduce_sum(-1, false);
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[b.id]).expect("compile failed");
        let a_buf = Buffer::from_slice::<i32>(&[1, 2, 3, 4, 5, 6], &[2, 3], DType::I32);
        let result = compiled.run(&[&a_buf]);
        assert_eq!(result.as_slice::<i32>(), &[6, 15]);
    }

    // ── 3D reduce tests ──────────────────────────────────────────────────────

    #[test]
    fn run_reduce_sum_3d_middle_dim() {
        begin_trace();
        let a = Tensor::new(&[2, 3, 4], DType::F32);
        let b = a.reduce_sum(1, false);
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[b.id]).expect("compile failed");
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let a_buf = Buffer::from_slice::<f32>(&data, &[2, 3, 4], DType::F32);
        let result = compiled.run(&[&a_buf]);
        let out = result.as_slice::<f32>();
        assert_eq!(out.len(), 8);
        // batch 0: sum rows [[0,1,2,3],[4,5,6,7],[8,9,10,11]] → [12,15,18,21]
        assert!((out[0] - 12.0).abs() < 1e-5);
        assert!((out[3] - 21.0).abs() < 1e-5);
        // batch 1: → [48,51,54,57]
        assert!((out[4] - 48.0).abs() < 1e-5);
    }

    #[test]
    fn run_reduce_max_3d_dim0() {
        begin_trace();
        let a = Tensor::new(&[2, 3, 4], DType::F32);
        let b = a.reduce_max(0, false);
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[b.id]).expect("compile failed");
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let a_buf = Buffer::from_slice::<f32>(&data, &[2, 3, 4], DType::F32);
        let result = compiled.run(&[&a_buf]);
        let out = result.as_slice::<f32>();
        assert_eq!(out.len(), 12);
        // max(a[0,i,j], a[1,i,j]) — second batch always larger
        assert!((out[0] - 12.0).abs() < 1e-5);
    }

    // ── Chained reshape test ─────────────────────────────────────────────────

    #[test]
    fn run_chained_reshapes() {
        begin_trace();
        let a = Tensor::new(&[24], DType::F32);
        let b = a.reshape(&[2, 3, 4]);
        let c = b.reshape(&[6, 4]);
        let d = c.reshape(&[24]);
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[d.id]).expect("compile failed");
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let a_buf = Buffer::from_slice::<f32>(&data, &[24], DType::F32);
        let result = compiled.run(&[&a_buf]);
        assert_eq!(result.as_slice::<f32>(), data.as_slice());
    }

    // ── Gemm with bias + alpha + beta combined ───────────────────────────────

    #[test]
    fn run_gemm_bias_alpha_beta() {
        begin_trace();
        let a = Tensor::new(&[2, 3], DType::F32);
        let b = Tensor::new(&[3, 2], DType::F32);
        let bias = Tensor::new(&[2], DType::F32);
        let c = a.gemm(&b, Some(&bias), 2.0, 0.5, false, false);
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[c.id]).expect("compile failed");
        // A = I-like, B = I-like → A@B = [[1,0],[0,1]]
        // 2*[[1,0],[0,1]] + 0.5*[10,20] = [[2,0],[0,2]] + [5,10] = [[7,10],[5,12]]
        let a_buf = Buffer::from_slice::<f32>(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3], DType::F32);
        let b_buf = Buffer::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], &[3, 2], DType::F32);
        let bias_buf = Buffer::from_slice::<f32>(&[10.0, 20.0], &[2], DType::F32);
        let result = compiled.run(&[&a_buf, &b_buf, &bias_buf]);
        let out = result.as_slice::<f32>();
        assert!((out[0] - 7.0).abs() < 1e-4);
        assert!((out[1] - 10.0).abs() < 1e-4);
        assert!((out[2] - 5.0).abs() < 1e-4);
        assert!((out[3] - 12.0).abs() < 1e-4);
    }

    // ── Broadcast scalar ─────────────────────────────────────────────────────

    #[test]
    fn run_mul_broadcast_scalar() {
        begin_trace();
        let a = Tensor::new(&[3, 4], DType::F32);
        let b = Tensor::new(&[1], DType::F32);
        let c = &a * &b;
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[c.id]).expect("compile failed");
        let a_data: Vec<f32> = (1..=12).map(|i| i as f32).collect();
        let a_buf = Buffer::from_slice::<f32>(&a_data, &[3, 4], DType::F32);
        let b_buf = Buffer::from_slice::<f32>(&[2.0], &[1], DType::F32);
        let result = compiled.run(&[&a_buf, &b_buf]);
        let expected: Vec<f32> = a_data.iter().map(|x| x * 2.0).collect();
        assert_eq!(result.as_slice::<f32>(), expected.as_slice());
    }

    // ── 3D softmax ───────────────────────────────────────────────────────────

    #[test]
    fn run_softmax_3d() {
        begin_trace();
        let a = Tensor::new(&[2, 2, 3], DType::F32);
        let b = a.softmax(-1);
        let trace = take_trace();
        let compiled = Compiler::compile(&trace, &[b.id]).expect("compile failed");
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let a_buf = Buffer::from_slice::<f32>(&data, &[2, 2, 3], DType::F32);
        let result = compiled.run(&[&a_buf]);
        let out = result.as_slice::<f32>();
        // Each group of 3 should sum to 1.0
        for i in 0..4 {
            let sum: f32 = out[i * 3..i * 3 + 3].iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "group {i} sum={sum}");
        }
    }
}
