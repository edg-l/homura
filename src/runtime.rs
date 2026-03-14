use melior::ExecutionEngine;

use crate::{DType, Shape};

/// The memref descriptor struct matching MLIR's C interface ABI for a
/// rank-1 `memref<NxT>`. Must be `#[repr(C)]` and match MLIR's
/// `StridedMemRefType<T, 1>` layout exactly.
#[repr(C)]
pub struct MemRefDescriptor1D<T> {
    pub allocated: *mut T,
    pub aligned: *mut T,
    pub offset: i64,
    pub sizes: [i64; 1],
    pub strides: [i64; 1],
}

impl<T> MemRefDescriptor1D<T> {
    /// Build a descriptor that views a mutable slice.
    pub fn from_slice(data: &mut [T]) -> Self {
        Self {
            allocated: data.as_mut_ptr(),
            aligned: data.as_mut_ptr(),
            offset: 0,
            sizes: [data.len() as i64],
            strides: [1],
        }
    }
}

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

    /// Execute the graph with the given f32 input slices. Returns the output
    /// as a `Vec<f32>`.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - The number of inputs doesn't match the compiled graph.
    /// - The output dtype is not F32 (use `run_f64` for F64).
    /// - The JIT invocation fails (mismatched ABI).
    pub fn run(&self, inputs: &[&[f32]]) -> Vec<f32> {
        assert_eq!(
            inputs.len(),
            self.num_inputs,
            "expected {} inputs, got {}",
            self.num_inputs,
            inputs.len()
        );
        assert_eq!(
            self.output_dtype,
            DType::F32,
            "output dtype is {:?}, not F32; use the appropriate run method",
            self.output_dtype
        );

        let num_elems = self.output_shape.num_elements() as usize;
        let mut output = vec![0.0f32; num_elems];

        // Build descriptors. We need to hold mutable copies of input data
        // because MemRefDescriptor1D requires *mut T. The JIT function only
        // reads the inputs, so this is safe.
        let mut input_data: Vec<Vec<f32>> = inputs.iter().map(|s| s.to_vec()).collect();
        let mut input_descs: Vec<MemRefDescriptor1D<f32>> = input_data
            .iter_mut()
            .map(|v| MemRefDescriptor1D::from_slice(v))
            .collect();
        let mut output_desc = MemRefDescriptor1D::from_slice(&mut output);

        // The invoke_packed machinery works through two levels of indirection:
        //
        //   invoke_packed("compute", args) calls _mlir__mlir_ciface_compute(args.as_ptr())
        //
        // That packed wrapper, for each argument of _mlir_ciface_compute of type T:
        //   arg_ptr = args[i]                      (load a void* from args[i])
        //   value   = *(T*)arg_ptr                 (load T from arg_ptr)
        //   pass value to _mlir_ciface_compute
        //
        // For _mlir_ciface_compute(%arg: !llvm.ptr), T = ptr.
        // So args[i] must be a **MemRefDescriptor (pointer to a pointer to the struct).
        //
        // Step 1: build *mut MemRefDescriptor pointers.
        let mut input_ptrs: Vec<*mut MemRefDescriptor1D<f32>> = input_descs
            .iter_mut()
            .map(|d| d as *mut MemRefDescriptor1D<f32>)
            .collect();
        let mut output_ptr = &mut output_desc as *mut MemRefDescriptor1D<f32>;

        // Step 2: args[i] = &mut ptr[i] cast to *mut ()
        let mut args: Vec<*mut ()> = input_ptrs
            .iter_mut()
            .map(|p| p as *mut *mut MemRefDescriptor1D<f32> as *mut ())
            .collect();
        args.push(&mut output_ptr as *mut *mut MemRefDescriptor1D<f32> as *mut ());

        unsafe {
            self.engine
                .invoke_packed("compute", &mut args)
                .expect("JIT invocation failed");
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use crate::{Compiler, DType, trace::{begin_trace, take_trace}, tensor::Tensor};

    #[test]
    fn run_add() {
        begin_trace();
        let a = Tensor::new(&[4], DType::F32);
        let b = Tensor::new(&[4], DType::F32);
        let c = &a + &b;
        let trace = take_trace();

        let compiled = Compiler::compile(&trace, &[c.id]).expect("compile failed");
        let result = compiled.run(&[&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]]);
        assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
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
        let result = compiled.run(&[
            &[1.0, 2.0, 3.0, 4.0],
            &[10.0, 20.0, 30.0, 40.0],
            &[100.0, 200.0, 300.0, 400.0],
        ]);
        assert_eq!(result, vec![111.0, 222.0, 333.0, 444.0]);
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
        let result = compiled.run(&[&a_data, &b_data]);

        let expected: Vec<f32> = (0..8).map(|i| (i + i * 2) as f32).collect();
        assert_eq!(result, expected);
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
        compiled.run(&[&[1.0, 2.0, 3.0, 4.0]]); // missing second input
    }
}
