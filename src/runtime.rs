use std::path::Path;
use std::slice;

use crate::shape::DIM_DYNAMIC;
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

    /// Reconfigure this buffer for a new shape/dtype, reusing the existing
    /// allocation when it has sufficient capacity.  Does **not** zero the data
    /// — the caller (compiled kernel) is expected to write every output byte.
    ///
    /// # Safety contract
    ///
    /// The returned buffer's `data` contents are **uninitialized** (or stale).
    /// Only pass it to a compiled kernel that overwrites the entire output.
    pub(crate) fn reconfigure(&mut self, shape: &[u64], dtype: DType) {
        let s = Shape(shape.to_vec());
        let num_bytes = s.num_elements() as usize * dtype.size_bytes();
        self.data.clear();
        self.data.resize(num_bytes, 0u8);
        self.shape = s;
        self.strides = row_major_strides(shape);
        self.dtype = dtype;
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
            unsafe {
                libc::dlclose(self._lib);
            }
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
        Self::load_named(so_path, num_inputs, outputs, "compute")
    }

    /// Load a compiled graph from a .so, resolving a specific function name.
    pub(crate) fn load_named(
        so_path: &Path,
        num_inputs: usize,
        outputs: Vec<OutputDesc>,
        func_name: &str,
    ) -> Result<Self, String> {
        use std::ffi::CString;

        let path_str = so_path
            .to_str()
            .ok_or_else(|| format!("non-UTF-8 path: {}", so_path.display()))?;
        let path_cstr = CString::new(path_str).map_err(|e| format!("path contains NUL: {e}"))?;

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

        let sym_str = format!("_mlir__mlir_ciface_{func_name}");
        let sym_name = CString::new(sym_str.clone()).expect("symbol name has no NUL");
        let sym = unsafe { libc::dlsym(lib, sym_name.as_ptr()) };
        if sym.is_null() {
            unsafe {
                libc::dlclose(lib);
            }
            return Err(format!("{sym_str} not found in {}", so_path.display()));
        }

        let func: unsafe extern "C" fn(*mut *mut ()) = unsafe { std::mem::transmute(sym) };

        Ok(Self {
            _lib: lib,
            func,
            num_inputs,
            outputs,
        })
    }

    /// Load a kernel function from an already-opened shared library handle.
    ///
    /// The handle is NOT owned — the caller must ensure it outlives this
    /// `CompiledGraph`. We store a null `_lib` to skip dlclose on drop.
    pub(crate) fn load_from_handle(
        lib: *mut std::ffi::c_void,
        num_inputs: usize,
        outputs: Vec<OutputDesc>,
        func_name: &str,
    ) -> Result<Self, String> {
        use std::ffi::CString;

        let sym_str = format!("_mlir__mlir_ciface_{func_name}");
        let sym_name = CString::new(sym_str.clone()).expect("symbol name has no NUL");
        let sym = unsafe { libc::dlsym(lib, sym_name.as_ptr()) };
        if sym.is_null() {
            return Err(format!("{sym_str} not found in shared library"));
        }
        let func: unsafe extern "C" fn(*mut *mut ()) = unsafe { std::mem::transmute(sym) };
        Ok(Self {
            _lib: std::ptr::null_mut(), // not owned — caller manages lifetime
            func,
            num_inputs,
            outputs,
        })
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

    /// Execute the graph, writing results into pre-allocated `output_bufs`.
    ///
    /// Each output buffer must already have the correct shape, strides, dtype,
    /// and sufficient allocation.  The kernel overwrites all output bytes.
    pub fn run_into(&self, inputs: &[&Buffer], output_bufs: &mut Vec<Buffer>) {
        assert_eq!(inputs.len(), self.num_inputs);
        assert_eq!(output_bufs.len(), self.outputs.len());
        self.run_with_output_bufs(inputs, output_bufs);
    }

    /// Like `run_into` but reconfigures each output buffer to `output_shapes`
    /// first (reusing the allocation when possible).
    pub fn run_dynamic_into(
        &self,
        inputs: &[&Buffer],
        output_shapes: &[Shape],
        output_bufs: &mut Vec<Buffer>,
    ) {
        assert_eq!(inputs.len(), self.num_inputs);
        assert_eq!(output_shapes.len(), self.outputs.len());
        assert_eq!(output_bufs.len(), self.outputs.len());
        for (buf, (shape, desc)) in output_bufs
            .iter_mut()
            .zip(output_shapes.iter().zip(self.outputs.iter()))
        {
            buf.reconfigure(&shape.0, desc.dtype);
        }
        self.run_with_output_bufs(inputs, output_bufs);
    }

    /// Shared JIT-call implementation for `run`, `run_dynamic`, and `*_into`.
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
    /// Symbolic shape tracking (e.g., Var("past_sequence_length")).
    /// `None` if symbolic propagation did not reach this slot.
    pub sym_shape: Option<crate::shape::SymShape>,
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
    /// Unified shared library handle (if all kernels are in one .so).
    /// When set, individual `CompiledGraph._lib` entries are null (non-owning).
    _shared_lib: *mut std::ffi::c_void,
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
    /// For each slot, the last step index that reads it as an input.
    /// `None` means the slot is never read (model output only) or is
    /// an input/weight that lives for the entire run.
    slot_last_read: Vec<Option<usize>>,
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

// SAFETY: the dlopen handle and function pointers are safe to send across threads.
unsafe impl Send for ExecutionPlan {}
unsafe impl Sync for ExecutionPlan {}

impl Drop for ExecutionPlan {
    fn drop(&mut self) {
        if !self._shared_lib.is_null() {
            unsafe {
                libc::dlclose(self._shared_lib);
            }
        }
    }
}

impl ExecutionPlan {
    /// Build an execution plan, precomputing buffer lifetime metadata.
    pub fn new(
        kernels: Vec<CompiledGraph>,
        steps: Vec<KernelStep>,
        num_slots: usize,
        input_slots: Vec<usize>,
        weight_slots: Vec<usize>,
        output_slots: Vec<usize>,
        slot_descs: Vec<SlotDesc>,
    ) -> Self {
        // Compute last-read step index for each slot.
        let mut last_read: Vec<Option<usize>> = vec![None; num_slots];
        for (step_idx, step) in steps.iter().enumerate() {
            for &slot in &step.input_slots {
                last_read[slot] = Some(step_idx);
            }
        }
        // Output slots must survive the entire run — mark as None.
        for &slot in &output_slots {
            last_read[slot] = None;
        }
        // Input/weight slots are borrowed — never recyclable.
        for &slot in input_slots.iter().chain(weight_slots.iter()) {
            last_read[slot] = None;
        }
        Self {
            _shared_lib: std::ptr::null_mut(),
            kernels,
            steps,
            num_slots,
            input_slots,
            weight_slots,
            output_slots,
            slot_descs,
            slot_last_read: last_read,
        }
    }

    /// Set the unified shared library handle (transfers ownership).
    pub(crate) fn set_shared_lib(&mut self, lib: *mut std::ffi::c_void) {
        self._shared_lib = lib;
    }

    /// Return the `SlotDesc` for each model output slot.
    pub fn output_slot_descs(&self) -> Vec<&SlotDesc> {
        self.output_slots
            .iter()
            .map(|&s| &self.slot_descs[s])
            .collect()
    }

    /// Build concrete shapes for all slots by evaluating symbolic dim expressions
    /// with variable bindings extracted from actual input shapes.
    fn resolve_slot_shapes(&self, inputs: &[&Buffer]) -> Vec<Shape> {
        use std::collections::HashMap;

        let mut bindings: HashMap<String, u64> = HashMap::new();

        // Extract variable bindings from input slots.
        for (input_idx, &slot) in self.input_slots.iter().enumerate() {
            if let Some(sym_shape) = &self.slot_descs[slot].sym_shape {
                let actual_shape = &inputs[input_idx].shape().0;
                for (dim, sym) in sym_shape.iter().enumerate() {
                    if let crate::shape::SymDim::Var(name) = sym {
                        let actual = actual_shape[dim];
                        if let Some(&existing) = bindings.get(name) {
                            if existing != actual {
                                tracing::warn!(
                                    name,
                                    existing,
                                    actual,
                                    "conflicting sym dim binding"
                                );
                            }
                        }
                        bindings.insert(name.clone(), actual);
                    }
                }
            }
        }

        // Evaluate every slot's sym_shape to get concrete dims.
        self.slot_descs
            .iter()
            .map(|desc| {
                match &desc.sym_shape {
                    Some(sym_shape) => {
                        let dims: Vec<u64> = sym_shape
                            .iter()
                            .enumerate()
                            .map(|(d, sym)| {
                                sym.eval(&bindings).unwrap_or_else(|| {
                                    panic!(
                                        "unresolvable sym dim [{d}] = {sym} \
                                         for slot with shape {:?}, bindings: {bindings:?}",
                                        desc.shape
                                    )
                                })
                            })
                            .collect();
                        Shape(dims)
                    }
                    // No sym_shape — use the static shape (all concrete).
                    None => desc.shape.clone(),
                }
            })
            .collect()
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

        let resolved_shapes = self.resolve_slot_shapes(inputs);

        let mut pool: Vec<Option<PoolEntry>> = (0..self.num_slots).map(|_| None).collect();

        // Place inputs (borrowed).
        for (i, &slot) in self.input_slots.iter().enumerate() {
            pool[slot] = Some(PoolEntry::Borrowed(inputs[i]));
        }

        // Place weights (borrowed).
        for (i, &slot) in self.weight_slots.iter().enumerate() {
            pool[slot] = Some(PoolEntry::Borrowed(&weights[i]));
        }

        // Per-step profiling (enabled by HOMURA_PROFILE=1).
        let profile = std::env::var("HOMURA_PROFILE").is_ok_and(|v| v == "1");
        let mut step_times: Vec<(usize, std::time::Duration, Vec<Vec<u64>>)> = Vec::new();

        // Free-list for buffer reuse: recycle dead intermediate buffers.
        let mut free_list: Vec<Buffer> = Vec::new();

        // Execute steps.
        for (step_idx, step) in self.steps.iter().enumerate() {
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

            // Skip kernel if any output has a zero dimension — the output is
            // trivially zero-filled.  Check outputs (not inputs) so that ops
            // like Concat with a zero-dim input but non-zero output still run.
            let has_zero_output = step
                .output_slots
                .iter()
                .any(|&slot| resolved_shapes[slot].0.iter().any(|&d| d == 0));
            if has_zero_output {
                tracing::debug!(
                    kernel = step.kernel_idx,
                    "skipping kernel: output has zero dimension"
                );
                for &slot in &step.output_slots {
                    let shape = &resolved_shapes[slot];
                    let dtype = self.slot_descs[slot].dtype;
                    pool[slot] = Some(PoolEntry::Owned(Buffer::new(&shape.0, dtype)));
                }
                continue;
            }

            // Check if any output has dynamic dims — if so, resolve from slot_descs.
            let has_dynamic = kernel
                .output_descs()
                .iter()
                .any(|d| d.shape.has_dynamic_dims());

            let t0 = if profile {
                Some(std::time::Instant::now())
            } else {
                None
            };

            // Try to grab recycled buffers from the free-list for outputs.
            // For non-dynamic kernels, use the kernel's compiled output shape
            // (matches original `kernel.run()` behavior). For dynamic kernels,
            // use the resolved slot shapes.
            let out_shapes: Vec<(&[u64], DType)> = if has_dynamic {
                step.output_slots
                    .iter()
                    .map(|&slot| {
                        (
                            resolved_shapes[slot].0.as_slice(),
                            self.slot_descs[slot].dtype,
                        )
                    })
                    .collect()
            } else {
                kernel
                    .output_descs()
                    .iter()
                    .map(|desc| (desc.shape.0.as_slice(), desc.dtype))
                    .collect()
            };

            let mut out_bufs: Vec<Buffer> = out_shapes
                .iter()
                .map(|&(shape, dtype)| {
                    let need_bytes = shape.iter().product::<u64>() as usize * dtype.size_bytes();
                    // Find a free buffer with enough capacity.
                    let reuse_idx = free_list
                        .iter()
                        .position(|b| b.data.capacity() >= need_bytes);
                    let mut buf = if let Some(idx) = reuse_idx {
                        free_list.swap_remove(idx)
                    } else {
                        Buffer::new(shape, dtype)
                    };
                    buf.reconfigure(shape, dtype);
                    buf
                })
                .collect();

            kernel.run_into(&step_inputs, &mut out_bufs);

            if let Some(t0) = t0 {
                let shapes: Vec<Vec<u64>> =
                    step_inputs.iter().map(|b| b.shape().0.clone()).collect();
                step_times.push((step.kernel_idx, t0.elapsed(), shapes));
            }

            // Place outputs into pool.
            for (buf, &slot) in out_bufs.into_iter().zip(step.output_slots.iter()) {
                pool[slot] = Some(PoolEntry::Owned(buf));
            }

            // Recycle dead intermediate buffers into the free-list.
            for &slot in &step.input_slots {
                if let Some(last) = self.slot_last_read[slot] {
                    if last == step_idx {
                        if let Some(PoolEntry::Owned(buf)) = pool[slot].take() {
                            free_list.push(buf);
                        }
                    }
                }
            }
        }

        // Print per-step profiling summary.
        if profile && !step_times.is_empty() {
            let total: std::time::Duration = step_times.iter().map(|(_, d, _)| *d).sum();
            eprintln!(
                "  ┌─ kernel profile ({} steps, {:.1}ms total)",
                step_times.len(),
                total.as_secs_f64() * 1000.0
            );
            for (kid, dur, shapes) in &step_times {
                let ms = dur.as_secs_f64() * 1000.0;
                if ms >= 0.5 {
                    let pct = dur.as_secs_f64() / total.as_secs_f64() * 100.0;
                    let shape_str: Vec<String> =
                        shapes.iter().map(|s| format!("{:?}", s)).collect();
                    eprintln!(
                        "  │ k{:<4} {:>8.2}ms  ({:>5.1}%)  {}",
                        kid,
                        ms,
                        pct,
                        shape_str.join(" × ")
                    );
                }
            }
            eprintln!("  └─");
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
        use super::{ExecutionPlan, KernelStep, SlotDesc};
        use crate::Shape;
        use crate::graph_builder::GraphContext;

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
        let plan = ExecutionPlan::new(
            vec![k0, k1],
            vec![
                KernelStep {
                    kernel_idx: 0,
                    input_slots: vec![0, 1], // a, b
                    output_slots: vec![2],   // c
                },
                KernelStep {
                    kernel_idx: 1,
                    input_slots: vec![2, 1], // c, b
                    output_slots: vec![3],   // d
                },
            ],
            4,
            vec![0, 1],
            vec![],
            vec![3],
            vec![
                SlotDesc {
                    shape: Shape(vec![4]),
                    dtype: DType::F32,
                    sym_shape: None,
                },
                SlotDesc {
                    shape: Shape(vec![4]),
                    dtype: DType::F32,
                    sym_shape: None,
                },
                SlotDesc {
                    shape: Shape(vec![4]),
                    dtype: DType::F32,
                    sym_shape: None,
                },
                SlotDesc {
                    shape: Shape(vec![4]),
                    dtype: DType::F32,
                    sym_shape: None,
                },
            ],
        );

        let a = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[4], DType::F32);
        let b = Buffer::from_slice::<f32>(&[10.0, 20.0, 30.0, 40.0], &[4], DType::F32);

        let result = plan.run(&[&a, &b], &[]);
        // d = (a + b) * b = (11, 22, 33, 44) * (10, 20, 30, 40) = (110, 440, 990, 1760)
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].as_slice::<f32>(), &[110.0, 440.0, 990.0, 1760.0]);
    }

    #[test]
    fn execution_plan_zero_dim_skips_kernel() {
        // When an input has a zero dimension, the kernel should be skipped
        // and the output should be a zero-filled buffer with the resolved shape.
        use super::{ExecutionPlan, KernelStep, SlotDesc};
        use crate::Shape;
        use crate::graph_builder::GraphContext;

        // Compile a real kernel (add) — it won't actually be called.
        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let x = gb.input(&[Some(4)], DType::F32);
        let y = gb.input(&[Some(4)], DType::F32);
        let z = gb.emit_add(&x, &y);
        let k = gb.compile(&[&z]).expect("compile");

        let plan = ExecutionPlan::new(
            vec![k],
            vec![KernelStep {
                kernel_idx: 0,
                input_slots: vec![0, 1],
                output_slots: vec![2],
            }],
            3,
            vec![0, 1],
            vec![],
            vec![2],
            vec![
                SlotDesc {
                    shape: Shape(vec![0]),
                    dtype: DType::F32,
                    sym_shape: None,
                },
                SlotDesc {
                    shape: Shape(vec![0]),
                    dtype: DType::F32,
                    sym_shape: None,
                },
                SlotDesc {
                    shape: Shape(vec![0]),
                    dtype: DType::F32,
                    sym_shape: None,
                },
            ],
        );

        let a = Buffer::new(&[0], DType::F32);
        let b = Buffer::new(&[0], DType::F32);

        let result = plan.run(&[&a, &b], &[]);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].shape().0, vec![0]);
        assert_eq!(result[0].as_slice::<f32>(), &[] as &[f32]);
    }

    #[test]
    fn execution_plan_with_weights() {
        // kernel0 = add(input, weight) → output
        use super::{ExecutionPlan, KernelStep, SlotDesc};
        use crate::Shape;
        use crate::graph_builder::GraphContext;

        let ctx = GraphContext::new();
        let mut gb = ctx.builder();
        let x = gb.input(&[Some(3)], DType::F32);
        let w = gb.input(&[Some(3)], DType::F32);
        let y = gb.emit_add(&x, &w);
        let k = gb.compile(&[&y]).expect("compile");

        // Slots: 0=input, 1=weight, 2=output
        let plan = ExecutionPlan::new(
            vec![k],
            vec![KernelStep {
                kernel_idx: 0,
                input_slots: vec![0, 1],
                output_slots: vec![2],
            }],
            3,
            vec![0],
            vec![1],
            vec![2],
            vec![
                SlotDesc {
                    shape: Shape(vec![3]),
                    dtype: DType::F32,
                    sym_shape: None,
                },
                SlotDesc {
                    shape: Shape(vec![3]),
                    dtype: DType::F32,
                    sym_shape: None,
                },
                SlotDesc {
                    shape: Shape(vec![3]),
                    dtype: DType::F32,
                    sym_shape: None,
                },
            ],
        );

        let input = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3], DType::F32);
        let weight = Buffer::from_slice::<f32>(&[0.5, 0.5, 0.5], &[3], DType::F32);

        let result = plan.run(&[&input], &[weight]);
        assert_eq!(result[0].as_slice::<f32>(), &[1.5, 2.5, 3.5]);
    }
}
