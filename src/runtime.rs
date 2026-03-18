use std::path::Path;
use std::slice;

use crate::{DType, Shape};
use crate::shape::DIM_DYNAMIC;

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
    ///
    /// # Panics
    ///
    /// Panics if any dim is `DIM_DYNAMIC` (use `run_dynamic` on `CompiledGraph`
    /// to provide concrete output shapes for dynamic models).
    pub fn new(shape: &[u64], dtype: DType) -> Self {
        assert!(
            !shape.iter().any(|&d| d == DIM_DYNAMIC),
            "Buffer::new called with DIM_DYNAMIC in shape {:?}; use CompiledGraph::run_dynamic instead",
            shape
        );
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
        if self.data.is_empty() {
            return &[];
        }
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

    /// Reinterpret the raw bytes as a mutable typed slice.
    ///
    /// # Panics
    ///
    /// Panics if `size_of::<T>()` does not match the buffer's dtype element size.
    pub fn as_slice_mut<T: Copy + 'static>(&mut self) -> &mut [T] {
        let elem_size = std::mem::size_of::<T>();
        assert_eq!(
            elem_size,
            self.dtype.size_bytes(),
            "type size {} does not match dtype {:?} (size {})",
            elem_size,
            self.dtype,
            self.dtype.size_bytes(),
        );
        if self.data.is_empty() {
            return &mut [];
        }
        unsafe {
            slice::from_raw_parts_mut(
                self.data.as_mut_ptr() as *mut T,
                self.data.len() / elem_size,
            )
        }
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

    // For zero-element memrefs (e.g., memref<0xi64> for scalar shape),
    // the data pointer from an empty Vec is dangling. Use a valid aligned
    // address — the pointer is never dereferenced for 0-element tensors,
    // but MLIR's generated code may still read it into a descriptor.
    static EMPTY_BUF: [u64; 1] = [0];
    let ptr_val = if data_ptr.is_null() || shape.iter().any(|&d| d == 0) {
        EMPTY_BUF.as_ptr() as u64
    } else {
        data_ptr as u64
    };
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

/// Metadata for a single output tensor of a compiled graph.
#[derive(Clone)]
pub struct OutputDesc {
    pub shape: Shape,
    pub dtype: DType,
}

/// A compiled computation graph, loaded from a native `.so` via dlopen.
pub struct CompiledGraph {
    /// dlopen handle — kept alive so the library is not unloaded while the
    /// graph is live. Linux keeps the inode alive even if the file is removed.
    _lib: *mut libc::c_void,
    /// Pointer to `_mlir__mlir_ciface_compute`: takes a single `*mut *mut ()`
    /// (array of void-pointers each pointing to a MemRefDescriptor).
    func: unsafe extern "C" fn(*mut *mut ()),
    num_inputs: usize,
    outputs: Vec<OutputDesc>,
}

// SAFETY: the dlopen handle is process-global and valid for the lifetime of
// the `CompiledGraph`. We never move the raw pointer across threads
// concurrently with mutation.
unsafe impl Send for CompiledGraph {}
unsafe impl Sync for CompiledGraph {}

impl Drop for CompiledGraph {
    fn drop(&mut self) {
        if !self._lib.is_null() {
            unsafe { libc::dlclose(self._lib); }
        }
    }
}

impl CompiledGraph {
    /// Load a pre-compiled `.so` and wrap it as a `CompiledGraph`.
    ///
    /// The .so must export the symbol `_mlir__mlir_ciface_compute` which is
    /// the packed-argument wrapper generated by MLIR's `llvm.emit_c_interface`
    /// mechanism. It takes a single `void**` argument whose entries are
    /// `void*` pointers each pointing to a MemRefDescriptor struct.
    pub(crate) fn load(
        so_path: &Path,
        num_inputs: usize,
        outputs: Vec<OutputDesc>,
    ) -> Result<Self, String> {
        use std::ffi::CString;

        let path_str = so_path
            .to_str()
            .ok_or_else(|| format!("non-UTF-8 path: {}", so_path.display()))?;
        let path_cstr =
            CString::new(path_str).map_err(|e| format!("path contains NUL: {e}"))?;

        let lib = unsafe { libc::dlopen(path_cstr.as_ptr(), libc::RTLD_NOW) };
        if lib.is_null() {
            let err = unsafe {
                let msg = libc::dlerror();
                if msg.is_null() {
                    "unknown dlopen error".to_string()
                } else {
                    std::ffi::CStr::from_ptr(msg).to_string_lossy().into_owned()
                }
            };
            return Err(format!("dlopen({}) failed: {err}", so_path.display()));
        }

        // `_mlir__mlir_ciface_compute` is the packed-convention wrapper:
        // takes void** where each void* points to a MemRefDescriptor.
        let sym_name =
            CString::new("_mlir__mlir_ciface_compute").expect("static name has no NUL");
        let sym = unsafe { libc::dlsym(lib, sym_name.as_ptr()) };
        if sym.is_null() {
            unsafe {
                libc::dlclose(lib);
            }
            return Err(format!(
                "_mlir__mlir_ciface_compute not found in {}",
                so_path.display()
            ));
        }

        let func: unsafe extern "C" fn(*mut *mut ()) = unsafe { std::mem::transmute(sym) };

        Ok(Self { _lib: lib, func, num_inputs, outputs })
    }

    /// Return the output descriptors (shape + dtype) for all outputs.
    pub fn output_descs(&self) -> &[OutputDesc] {
        &self.outputs
    }

    /// Execute the graph with caller-provided concrete output shapes.
    ///
    /// Use this when the compiled graph has dynamic dimensions in its outputs:
    /// the compiled code only reads/writes through the provided memref
    /// descriptors, so the shapes in `output_shapes` govern buffer allocation.
    ///
    /// `output_shapes` must have exactly `self.outputs.len()` entries, one per
    /// output. All dims must be concrete (no `DIM_DYNAMIC`).
    ///
    /// # Panics
    ///
    /// Panics if the number of inputs or output shapes doesn't match.
    pub fn run_dynamic(&self, inputs: &[&Buffer], output_shapes: &[Shape]) -> Vec<Buffer> {
        assert_eq!(
            inputs.len(),
            self.num_inputs,
            "run_dynamic: expected {} inputs, got {}",
            self.num_inputs,
            inputs.len()
        );
        assert_eq!(
            output_shapes.len(),
            self.outputs.len(),
            "run_dynamic: expected {} output shapes, got {}",
            self.outputs.len(),
            output_shapes.len()
        );

        // Allocate output buffers using the caller-provided concrete shapes.
        let mut output_bufs: Vec<Buffer> = output_shapes
            .iter()
            .zip(self.outputs.iter())
            .map(|(shape, desc)| Buffer::new(shape.0.as_slice(), desc.dtype))
            .collect();

        // Build and call — reuse the shared JIT call logic.
        self.run_with_output_bufs(inputs, &mut output_bufs);
        output_bufs
    }

    /// Execute the graph with the given input `Buffer`s. Returns all outputs as
    /// owned `Buffer`s in the same order as the `outputs` slice passed to
    /// `Compiler::compile`.
    ///
    /// # Panics
    ///
    /// Panics if the number of inputs doesn't match the compiled graph, or if
    /// any output shape contains `DIM_DYNAMIC` (use `run_dynamic` instead).
    pub fn run(&self, inputs: &[&Buffer]) -> Vec<Buffer> {
        assert_eq!(
            inputs.len(),
            self.num_inputs,
            "expected {} inputs, got {}",
            self.num_inputs,
            inputs.len()
        );

        // Allocate output buffers (must exist before descriptors are built).
        let mut output_bufs: Vec<Buffer> = self
            .outputs
            .iter()
            .map(|desc| Buffer::new(desc.shape.0.as_slice(), desc.dtype))
            .collect();

        self.run_with_output_bufs(inputs, &mut output_bufs);
        output_bufs
    }

    /// Shared JIT-call implementation for `run` and `run_dynamic`.
    fn run_with_output_bufs(&self, inputs: &[&Buffer], output_bufs: &mut Vec<Buffer>) {

        // Build memref descriptors for inputs. The function only reads inputs,
        // so the const→mut cast is safe.
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

        let mut output_descs: Vec<Vec<u8>> = output_bufs
            .iter_mut()
            .map(|buf| {
                let shape_i64: Vec<i64> = buf.shape().0.iter().map(|&d| d as i64).collect();
                let strides = buf.strides().to_vec();
                build_memref_descriptor(buf.data.as_mut_ptr(), &shape_i64, &strides)
            })
            .collect();

        // args[i] = pointer-to-descriptor-pointer (double indirection).
        // The MLIR C-interface wrapper dereferences each entry to get the
        // MemRefDescriptor struct.
        let mut desc_ptrs: Vec<*mut u8> = input_descs.iter_mut().map(|d| d.as_mut_ptr()).collect();
        let mut output_desc_ptrs: Vec<*mut u8> =
            output_descs.iter_mut().map(|d| d.as_mut_ptr()).collect();

        let mut args: Vec<*mut ()> = desc_ptrs
            .iter_mut()
            .map(|p| p as *mut *mut u8 as *mut ())
            .collect();
        for p in output_desc_ptrs.iter_mut() {
            args.push(p as *mut *mut u8 as *mut ());
        }

        {
            tracing::debug!(count = inputs.len(), "memref inputs");
            for (i, buf) in inputs.iter().enumerate() {
                let data_ptr = buf.data.as_ptr();
                match buf.dtype() {
                    DType::F32 => {
                        let n = buf.data.len() / 4;
                        let show = n.min(4);
                        let elems: Vec<f32> = (0..show)
                            .map(|k| {
                                let bytes = &buf.data[k * 4..(k + 1) * 4];
                                f32::from_ne_bytes(bytes.try_into().unwrap())
                            })
                            .collect();
                        tracing::debug!(
                            i, shape = ?buf.shape().0, dtype = ?buf.dtype(),
                            ptr = ?data_ptr, first_elems = ?elems, "input memref"
                        );
                    }
                    DType::I64 => {
                        let n = buf.data.len() / 8;
                        let show = n.min(4);
                        let elems: Vec<i64> = (0..show)
                            .map(|k| {
                                let bytes = &buf.data[k * 8..(k + 1) * 8];
                                i64::from_ne_bytes(bytes.try_into().unwrap())
                            })
                            .collect();
                        tracing::debug!(
                            i, shape = ?buf.shape().0, dtype = ?buf.dtype(),
                            ptr = ?data_ptr, first_elems = ?elems, "input memref"
                        );
                    }
                    _ => {
                        tracing::debug!(
                            i, shape = ?buf.shape().0, dtype = ?buf.dtype(),
                            ptr = ?data_ptr, "input memref"
                        );
                    }
                }
            }
            tracing::debug!(count = output_bufs.len(), "memref outputs");
            for (i, buf) in output_bufs.iter().enumerate() {
                let data_ptr = buf.data.as_ptr();
                tracing::debug!(
                    i, shape = ?buf.shape().0, dtype = ?buf.dtype(),
                    ptr = ?data_ptr, size_bytes = buf.data.len(), "output memref"
                );
            }
        }

        unsafe {
            (self.func)(args.as_mut_ptr());
        }
    }
}

// ── ExecutionPlan ─────────────────────────────────────────────────────────────

/// Describes one buffer slot in the execution plan's buffer pool.
#[derive(Clone, Debug)]
pub struct SlotDesc {
    pub shape: Shape,
    pub dtype: DType,
}

/// One step in the execution plan: invoke a kernel with specific buffer routing.
#[derive(Clone, Debug)]
pub struct KernelStep {
    /// Index into `ExecutionPlan::kernels`.
    pub kernel_idx: usize,
    /// Buffer pool slot indices to pass as inputs to this kernel.
    /// Order matches the kernel's compiled input arguments.
    pub input_slots: Vec<usize>,
    /// Buffer pool slot indices where this kernel writes its outputs.
    /// Order matches the kernel's compiled output arguments.
    pub output_slots: Vec<usize>,
}

/// A compiled model consisting of multiple independently-compiled kernels
/// executed in sequence with Rust-side buffer routing.
///
/// Each kernel is a `CompiledGraph` (its own `.so`). The execution plan
/// specifies the order of kernel invocations and how buffer slots map to
/// each kernel's inputs and outputs.
pub struct ExecutionPlan {
    /// Compiled kernels, indexed by `KernelStep::kernel_idx`.
    pub(crate) kernels: Vec<CompiledGraph>,
    /// Execution steps in order.
    pub(crate) steps: Vec<KernelStep>,
    /// Total number of buffer slots in the pool.
    pub(crate) num_slots: usize,
    /// Buffer pool slot indices for model inputs (in order).
    pub(crate) input_slots: Vec<usize>,
    /// Buffer pool slot indices for weight buffers (in order).
    pub(crate) weight_slots: Vec<usize>,
    /// Buffer pool slot indices that are model outputs (extracted at the end).
    pub(crate) output_slots: Vec<usize>,
    /// Shape + dtype for every slot (used to allocate intermediate buffers).
    pub(crate) slot_descs: Vec<SlotDesc>,
}

/// A buffer pool entry: either borrowed (inputs/weights) or owned (intermediates/outputs).
enum PoolEntry<'a> {
    Borrowed(&'a Buffer),
    Owned(Buffer),
}

impl<'a> PoolEntry<'a> {
    fn as_ref(&self) -> &Buffer {
        match self {
            PoolEntry::Borrowed(b) => b,
            PoolEntry::Owned(b) => b,
        }
    }

    fn into_owned(self) -> Buffer {
        match self {
            PoolEntry::Borrowed(b) => b.clone(),
            PoolEntry::Owned(b) => b,
        }
    }
}

impl ExecutionPlan {
    /// Return the `SlotDesc` for each model output slot.
    pub fn output_slot_descs(&self) -> Vec<&SlotDesc> {
        self.output_slots.iter().map(|&s| &self.slot_descs[s]).collect()
    }

    /// Execute the plan with the given model inputs and weight buffers.
    ///
    /// # Panics
    ///
    /// Panics if the number of inputs or weights doesn't match the plan.
    pub fn run(&self, inputs: &[&Buffer], weights: &[Buffer]) -> Vec<Buffer> {
        assert_eq!(
            inputs.len(),
            self.input_slots.len(),
            "ExecutionPlan::run: expected {} inputs, got {}",
            self.input_slots.len(),
            inputs.len()
        );
        assert_eq!(
            weights.len(),
            self.weight_slots.len(),
            "ExecutionPlan::run: expected {} weights, got {}",
            self.weight_slots.len(),
            weights.len()
        );

        let mut pool: Vec<Option<PoolEntry>> = (0..self.num_slots).map(|_| None).collect();

        // Place inputs (borrowed).
        for (i, &slot) in self.input_slots.iter().enumerate() {
            pool[slot] = Some(PoolEntry::Borrowed(inputs[i]));
        }

        // Place weights (borrowed).
        for (i, &slot) in self.weight_slots.iter().enumerate() {
            pool[slot] = Some(PoolEntry::Borrowed(&weights[i]));
        }

        // Execute steps.
        for step in &self.steps {
            let kernel = &self.kernels[step.kernel_idx];

            // Gather input refs for this kernel.
            let step_inputs: Vec<&Buffer> = step
                .input_slots
                .iter()
                .map(|&s| {
                    pool[s]
                        .as_ref()
                        .unwrap_or_else(|| panic!("buffer slot {s} not populated at kernel step"))
                        .as_ref()
                })
                .collect();

            // Check if any output has dynamic dims — if so, resolve from slot_descs.
            let has_dynamic = kernel
                .output_descs()
                .iter()
                .any(|d| d.shape.has_dynamic_dims());

            let outputs: Vec<Buffer> = if has_dynamic {
                let concrete_shapes: Vec<Shape> = step
                    .output_slots
                    .iter()
                    .map(|&s| self.slot_descs[s].shape.clone())
                    .collect();
                kernel.run_dynamic(&step_inputs, &concrete_shapes)
            } else {
                kernel.run(&step_inputs)
            };

            // Place outputs into pool.
            assert_eq!(
                outputs.len(),
                step.output_slots.len(),
                "kernel {} produced {} outputs but step expects {}",
                step.kernel_idx,
                outputs.len(),
                step.output_slots.len()
            );
            for (buf, &slot) in outputs.into_iter().zip(step.output_slots.iter()) {
                pool[slot] = Some(PoolEntry::Owned(buf));
            }
        }

        // Extract model outputs.
        self.output_slots
            .iter()
            .map(|&s| {
                pool[s]
                    .take()
                    .unwrap_or_else(|| panic!("output slot {s} not populated"))
                    .into_owned()
            })
            .collect()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::{DType, runtime::Buffer};

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

    // ── ExecutionPlan tests ──────────────────────────────────────────────────

    #[test]
    fn execution_plan_two_kernels() {
        // Build a plan: kernel0 = add(a, b) → c, kernel1 = mul(c, b) → d
        // Equivalent to: d = (a + b) * b
        use crate::graph_builder::GraphContext;
        use super::{ExecutionPlan, KernelStep, SlotDesc};
        use crate::Shape;

        // Kernel 0: add(x, y) → z
        let ctx0 = GraphContext::new();
        let mut gb0 = ctx0.builder();
        let x0 = gb0.input(&[Some(4)], DType::F32);
        let y0 = gb0.input(&[Some(4)], DType::F32);
        let z0 = gb0.emit_add(&x0, &y0);
        let k0 = gb0.compile(&[&z0]).expect("compile kernel0");

        // Kernel 1: mul(x, y) → z
        let ctx1 = GraphContext::new();
        let mut gb1 = ctx1.builder();
        let x1 = gb1.input(&[Some(4)], DType::F32);
        let y1 = gb1.input(&[Some(4)], DType::F32);
        let z1 = gb1.emit_mul(&x1, &y1);
        let k1 = gb1.compile(&[&z1]).expect("compile kernel1");

        // Buffer slots:
        //   0 = input a
        //   1 = input b
        //   2 = intermediate c (output of kernel0)
        //   3 = output d (output of kernel1)
        let plan = ExecutionPlan {
            kernels: vec![k0, k1],
            steps: vec![
                KernelStep {
                    kernel_idx: 0,
                    input_slots: vec![0, 1],  // a, b
                    output_slots: vec![2],     // c
                },
                KernelStep {
                    kernel_idx: 1,
                    input_slots: vec![2, 1],  // c, b
                    output_slots: vec![3],     // d
                },
            ],
            num_slots: 4,
            input_slots: vec![0, 1],
            weight_slots: vec![],
            output_slots: vec![3],
            slot_descs: vec![
                SlotDesc { shape: Shape(vec![4]), dtype: DType::F32 },
                SlotDesc { shape: Shape(vec![4]), dtype: DType::F32 },
                SlotDesc { shape: Shape(vec![4]), dtype: DType::F32 },
                SlotDesc { shape: Shape(vec![4]), dtype: DType::F32 },
            ],
        };

        let a = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[4], DType::F32);
        let b = Buffer::from_slice::<f32>(&[10.0, 20.0, 30.0, 40.0], &[4], DType::F32);

        let result = plan.run(&[&a, &b], &[]);
        // d = (a + b) * b = (11, 22, 33, 44) * (10, 20, 30, 40) = (110, 440, 990, 1760)
        assert_eq!(result.len(), 1);
        assert_eq!(
            result[0].as_slice::<f32>(),
            &[110.0, 440.0, 990.0, 1760.0]
        );
    }

    #[test]
    fn execution_plan_with_weights() {
        // kernel0 = add(input, weight) → output
        use crate::graph_builder::GraphContext;
        use super::{ExecutionPlan, KernelStep, SlotDesc};
        use crate::Shape;

        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let x = gb.input(&[Some(3)], DType::F32);
        let w = gb.input(&[Some(3)], DType::F32);
        let y = gb.emit_add(&x, &w);
        let k = gb.compile(&[&y]).expect("compile");

        // Slots: 0=input, 1=weight, 2=output
        let plan = ExecutionPlan {
            kernels: vec![k],
            steps: vec![KernelStep {
                kernel_idx: 0,
                input_slots: vec![0, 1],
                output_slots: vec![2],
            }],
            num_slots: 3,
            input_slots: vec![0],
            weight_slots: vec![1],
            output_slots: vec![2],
            slot_descs: vec![
                SlotDesc { shape: Shape(vec![3]), dtype: DType::F32 },
                SlotDesc { shape: Shape(vec![3]), dtype: DType::F32 },
                SlotDesc { shape: Shape(vec![3]), dtype: DType::F32 },
            ],
        };

        let input = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3], DType::F32);
        let weight = Buffer::from_slice::<f32>(&[0.5, 0.5, 0.5], &[3], DType::F32);

        let result = plan.run(&[&input], &[weight]);
        assert_eq!(result[0].as_slice::<f32>(), &[1.5, 2.5, 3.5]);
    }
}
