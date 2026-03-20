use std::path::Path;
use std::slice;

use crate::shape::DIM_DYNAMIC;
use crate::{DType, Shape};

// ── Buffer ────────────────────────────────────────────────────────────────────

/// Internal storage for a Buffer: either an owned allocation or a borrowed
/// view into externally-owned memory (used for zero-copy KV cache views).
///
/// # Safety
///
/// The `View` variant holds a raw pointer that must outlive the `Buffer`.
/// This is only used internally by `KvCache::view_single`, which returns a
/// view into the KvCache's own pre-allocated buffers. Callers must not let
/// a `View` buffer escape the scope in which the KvCache is alive.
enum BufferData {
    Owned(Vec<u8>),
    /// Borrowed view into external memory. Not owned — must not outlive source.
    View {
        ptr: *const u8,
        len: usize,
    },
}

// SAFETY: the raw pointer in View points to KvCache-owned data, which is
// Send+Sync. We only ever create View buffers from &[u8] references.
unsafe impl Send for BufferData {}
unsafe impl Sync for BufferData {}

/// A type-erased tensor buffer. Stores raw bytes with an associated
/// shape, dtype, and strides. Strides may be non-row-major for zero-copy
/// views (e.g., KV cache views with max_len-based strides).
pub struct Buffer {
    data: BufferData,
    shape: Shape,
    strides: Vec<i64>,
    dtype: DType,
}

impl Clone for Buffer {
    fn clone(&self) -> Self {
        let data = match &self.data {
            BufferData::Owned(v) => BufferData::Owned(v.clone()),
            // View: copy the pointer — both point to the same underlying data.
            BufferData::View { ptr, len } => BufferData::View {
                ptr: *ptr,
                len: *len,
            },
        };
        Self {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            dtype: self.dtype,
        }
    }
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match &self.data {
            BufferData::Owned(_) => "Owned",
            BufferData::View { .. } => "View",
        };
        f.debug_struct("Buffer")
            .field("kind", &kind)
            .field("shape", &self.shape)
            .field("strides", &self.strides)
            .field("dtype", &self.dtype)
            .field("byte_len", &self.byte_len())
            .finish()
    }
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
            !shape.contains(&DIM_DYNAMIC),
            "Buffer::new called with DIM_DYNAMIC in shape {:?}; use CompiledGraph::run_dynamic instead",
            shape
        );
        let s = Shape(shape.to_vec());
        let strides = row_major_strides(shape);
        let num_bytes = s.num_elements() as usize * dtype.size_bytes();
        Self {
            data: BufferData::Owned(vec![0u8; num_bytes]),
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
            data: BufferData::Owned(raw.to_vec()),
            shape: Shape(shape.to_vec()),
            strides: row_major_strides(shape),
            dtype,
        }
    }

    /// Raw pointer to the buffer data.
    pub(crate) fn as_ptr(&self) -> *const u8 {
        match &self.data {
            BufferData::Owned(v) => v.as_ptr(),
            BufferData::View { ptr, .. } => *ptr,
        }
    }

    /// Mutable raw pointer. Only valid for Owned buffers.
    ///
    /// # Panics
    ///
    /// Panics on View buffers (views are read-only).
    pub(crate) fn as_mut_ptr(&mut self) -> *mut u8 {
        match &mut self.data {
            BufferData::Owned(v) => v.as_mut_ptr(),
            BufferData::View { .. } => panic!("as_mut_ptr called on a View buffer (read-only)"),
        }
    }

    /// Byte length of the data.
    pub(crate) fn byte_len(&self) -> usize {
        match &self.data {
            BufferData::Owned(v) => v.len(),
            BufferData::View { len, .. } => *len,
        }
    }

    /// Byte capacity of the underlying allocation. Returns 0 for View buffers.
    pub(crate) fn capacity(&self) -> usize {
        match &self.data {
            BufferData::Owned(v) => v.capacity(),
            BufferData::View { .. } => 0,
        }
    }

    /// Raw byte slice view of the data.
    pub(crate) fn as_bytes(&self) -> &[u8] {
        match &self.data {
            BufferData::Owned(v) => v.as_slice(),
            BufferData::View { ptr, len } => unsafe { slice::from_raw_parts(*ptr, *len) },
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
        if self.byte_len() == 0 {
            return &[];
        }
        unsafe { slice::from_raw_parts(self.as_ptr() as *const T, self.byte_len() / elem_size) }
    }

    /// Create a buffer from pre-packed raw bytes (for quantized weights).
    ///
    /// `shape` is the **logical** shape (dequantized dimensions, e.g. [K, N]).
    /// The byte length is validated against `dtype.bytes_for_elements()`.
    pub fn from_raw_bytes(data: Vec<u8>, shape: &[u64], dtype: DType) -> Self {
        let num_elements: u64 = shape.iter().product();
        let expected_bytes = dtype.bytes_for_elements(num_elements as usize);
        assert_eq!(
            data.len(),
            expected_bytes,
            "from_raw_bytes: data length {} does not match expected {} for shape {:?} dtype {:?}",
            data.len(),
            expected_bytes,
            shape,
            dtype,
        );
        let s = Shape(shape.to_vec());
        let strides = row_major_strides(shape);
        Self {
            data: BufferData::Owned(data),
            shape: s,
            strides,
            dtype,
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
    /// Panics if `size_of::<T>()` does not match the buffer's dtype element
    /// size, or if this is a View buffer (views are read-only).
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
        if self.byte_len() == 0 {
            return &mut [];
        }
        unsafe {
            slice::from_raw_parts_mut(self.as_mut_ptr() as *mut T, self.byte_len() / elem_size)
        }
    }

    /// Mutable access to the raw byte Vec. Only valid for Owned buffers.
    ///
    /// # Panics
    ///
    /// Panics on View buffers.
    pub(crate) fn data_mut(&mut self) -> &mut Vec<u8> {
        match &mut self.data {
            BufferData::Owned(v) => v,
            BufferData::View { .. } => panic!("data_mut called on a View buffer (read-only)"),
        }
    }

    /// Reconfigure this buffer for a new shape/dtype, reusing the existing
    /// allocation when it has sufficient capacity.  Does **not** zero the data
    /// — the caller (compiled kernel) is expected to write every output byte.
    ///
    /// # Safety contract
    ///
    /// The returned buffer's `data` contents are **uninitialized** (or stale).
    /// Only pass it to a compiled kernel that overwrites the entire output.
    ///
    /// # Panics
    ///
    /// Panics on View buffers.
    pub(crate) fn reconfigure(&mut self, shape: &[u64], dtype: DType) {
        let v = match &mut self.data {
            BufferData::Owned(v) => v,
            BufferData::View { .. } => panic!("reconfigure called on a View buffer"),
        };
        let s = Shape(shape.to_vec());
        let num_bytes = s.num_elements() as usize * dtype.size_bytes();
        v.clear();
        v.resize(num_bytes, 0);
        self.shape = s;
        self.strides = row_major_strides(shape);
        self.dtype = dtype;
    }
}

/// Native concat: copy data from multiple input buffers into an output buffer
/// along `axis`. All inputs must have the same shape except along `axis`.
fn native_concat(inputs: &[&Buffer], axis: usize, out_shape: &[u64], dtype: DType) -> Buffer {
    let elem_size = dtype.size_bytes();
    let rank = out_shape.len();
    let total_bytes = out_shape.iter().product::<u64>() as usize * elem_size;
    let mut out_data = vec![0u8; total_bytes];

    // Compute the size of one "slice" along the concat axis.
    // outer_size = product of dims before axis
    // inner_size = product of dims after axis (in bytes)
    let outer_size: usize = out_shape[..axis].iter().product::<u64>() as usize;
    let inner_size: usize = if axis + 1 < rank {
        out_shape[axis + 1..].iter().product::<u64>() as usize * elem_size
    } else {
        elem_size
    };

    let mut write_offset = 0;
    for outer in 0..outer_size {
        for input in inputs {
            let in_axis_len = input.shape().0[axis] as usize;
            let chunk = in_axis_len * inner_size;
            let read_offset = outer * in_axis_len * inner_size;
            out_data[write_offset..write_offset + chunk]
                .copy_from_slice(&input.as_bytes()[read_offset..read_offset + chunk]);
            write_offset += chunk;
        }
    }

    Buffer {
        data: BufferData::Owned(out_data),
        shape: Shape(out_shape.to_vec()),
        strides: row_major_strides(out_shape),
        dtype,
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

// ── KV Cache ──────────────────────────────────────────────────────────────────

/// Per-layer KV cache: pre-allocated K and V buffers for max_seq_len.
struct KvLayerCache {
    /// Pre-allocated key buffer: [1, num_heads, max_seq_len, head_dim].
    key: Vec<u8>,
    /// Pre-allocated value buffer: same shape as key.
    value: Vec<u8>,
}

/// Persistent KV cache for autoregressive decoding.
///
/// Pre-allocates buffers for `max_seq_len` tokens per layer. Each decode step
/// appends one new K/V entry via `append()`, and `view()` / `view_key()` /
/// `view_value()` return Buffers covering `[0..current_len]`.
///
/// `view_key` and `view_value` return zero-copy views into the pre-allocated
/// buffers using non-row-major strides (stride along seq dim = max_len *
/// head_dim instead of current_len * head_dim). The compiled kernel's memref
/// descriptor uses these strides to correctly skip over unused max_len gaps.
///
/// `view()` (used for prefill shape resolution) still copies the valid region
/// into a contiguous buffer, because callers may call `as_slice()` on it.
pub struct KvCache {
    layers: Vec<KvLayerCache>,
    current_len: usize,
    /// Number of new tokens being appended in the current step (set by append_key).
    /// Used by view_key/view_value to include all new entries before advance_by.
    new_tokens: usize,
    max_len: usize,
    num_heads: usize,
    head_dim: usize,
    dtype: DType,
}

impl KvCache {
    /// Create a new KV cache pre-allocated for the given dimensions.
    ///
    /// Each layer gets two buffers (K, V) of shape
    /// `[1, num_heads, max_len, head_dim]`, zero-initialized.
    pub fn new(
        num_layers: usize,
        num_heads: usize,
        max_len: usize,
        head_dim: usize,
        dtype: DType,
    ) -> Self {
        let elem_size = dtype.size_bytes();
        let buf_bytes = num_heads * max_len * head_dim * elem_size;
        let layers = (0..num_layers)
            .map(|_| KvLayerCache {
                key: vec![0u8; buf_bytes],
                value: vec![0u8; buf_bytes],
            })
            .collect();
        Self {
            layers,
            current_len: 0,
            new_tokens: 0,
            max_len,
            num_heads,
            head_dim,
            dtype,
        }
    }

    /// Append new K/V entries at position `current_len` for the given layer.
    ///
    /// `new_k` and `new_v` must have shape `[1, num_heads, 1, head_dim]`.
    /// Copies `num_heads * head_dim * elem_size` bytes into the pre-allocated
    /// buffer at the correct offset.
    #[cfg(test)]
    pub fn append(&mut self, layer: usize, new_k: &Buffer, new_v: &Buffer) {
        assert!(
            self.current_len < self.max_len,
            "KV cache full: current_len={} >= max_len={}",
            self.current_len,
            self.max_len
        );
        let elem_size = self.dtype.size_bytes();
        let row_bytes = self.head_dim * elem_size;
        let stride_seq = self.head_dim * elem_size; // stride along seq dimension

        // Copy each head's new entry into the correct position.
        // Layout: [1, num_heads, max_len, head_dim] row-major.
        // Offset for head h, seq s = (h * max_len + s) * head_dim * elem_size.
        for h in 0..self.num_heads {
            let dst_offset = (h * self.max_len + self.current_len) * stride_seq;
            let src_offset = h * row_bytes; // new_k is [1, heads, 1, head_dim]
            self.layers[layer].key[dst_offset..dst_offset + row_bytes]
                .copy_from_slice(&new_k.as_bytes()[src_offset..src_offset + row_bytes]);
            self.layers[layer].value[dst_offset..dst_offset + row_bytes]
                .copy_from_slice(&new_v.as_bytes()[src_offset..src_offset + row_bytes]);
        }
    }

    /// Return Buffer views of the KV cache for the given layer,
    /// covering `[1, num_heads, 0..current_len, head_dim]`.
    ///
    /// Return a pair of zero-byte Buffers with the correct KV shape for
    /// shape resolution, without copying any data. Used in `run_kv` to
    /// resolve symbolic output shapes before execution.
    pub fn view_shape(&self) -> (Buffer, Buffer) {
        let shape = &[
            1u64,
            self.num_heads as u64,
            self.current_len as u64,
            self.head_dim as u64,
        ];
        let k = Buffer {
            data: BufferData::Owned(Vec::new()),
            shape: Shape(shape.to_vec()),
            strides: row_major_strides(shape),
            dtype: self.dtype,
        };
        let v = Buffer {
            data: BufferData::Owned(Vec::new()),
            shape: Shape(shape.to_vec()),
            strides: row_major_strides(shape),
            dtype: self.dtype,
        };
        (k, v)
    }

    /// Copies the valid region into new contiguous Buffers (safe for `as_slice`
    /// callers). Used in tests.
    #[cfg(test)]
    pub fn view(&self, layer: usize) -> (Buffer, Buffer) {
        let elem_size = self.dtype.size_bytes();
        let view_shape = [
            1u64,
            self.num_heads as u64,
            self.current_len as u64,
            self.head_dim as u64,
        ];
        let row_bytes = self.head_dim * elem_size;
        let view_bytes = self.num_heads * self.current_len * self.head_dim * elem_size;

        let mut k_data = vec![0u8; view_bytes];
        let mut v_data = vec![0u8; view_bytes];

        // Copy each head's [0..current_len] rows from the max_len-strided buffer.
        for h in 0..self.num_heads {
            for s in 0..self.current_len {
                let src_off = (h * self.max_len + s) * row_bytes;
                let dst_off = (h * self.current_len + s) * row_bytes;
                k_data[dst_off..dst_off + row_bytes]
                    .copy_from_slice(&self.layers[layer].key[src_off..src_off + row_bytes]);
                v_data[dst_off..dst_off + row_bytes]
                    .copy_from_slice(&self.layers[layer].value[src_off..src_off + row_bytes]);
            }
        }

        let k_buf = Buffer {
            data: BufferData::Owned(k_data),
            shape: Shape(view_shape.to_vec()),
            strides: row_major_strides(&view_shape),
            dtype: self.dtype,
        };
        let v_buf = Buffer {
            data: BufferData::Owned(v_data),
            shape: Shape(view_shape.to_vec()),
            strides: row_major_strides(&view_shape),
            dtype: self.dtype,
        };
        (k_buf, v_buf)
    }

    /// Append a new K entry at position `current_len` for the given layer.
    pub fn append_key(&mut self, layer: usize, new_k: &Buffer) {
        let new_seq = new_k.shape().0[2] as usize;
        assert!(
            self.current_len + new_seq <= self.max_len,
            "KV cache full ({} + {} > {})",
            self.current_len,
            new_seq,
            self.max_len
        );
        self.new_tokens = new_seq;
        let row_bytes = self.head_dim * self.dtype.size_bytes();
        for h in 0..self.num_heads {
            let dst = (h * self.max_len + self.current_len) * row_bytes;
            let src = h * new_seq * row_bytes;
            let chunk = new_seq * row_bytes;
            self.layers[layer].key[dst..dst + chunk]
                .copy_from_slice(&new_k.as_bytes()[src..src + chunk]);
        }
    }

    /// Append new V entries at position `current_len` for the given layer.
    ///
    /// Handles both single-token (shape `[1, heads, 1, dim]`) and multi-token
    /// (shape `[1, heads, N, dim]`) inputs.
    pub fn append_value(&mut self, layer: usize, new_v: &Buffer) {
        let new_seq = new_v.shape().0[2] as usize;
        assert!(
            self.current_len + new_seq <= self.max_len,
            "KV cache full ({} + {} > {})",
            self.current_len,
            new_seq,
            self.max_len
        );
        debug_assert!(
            self.new_tokens == 0 || self.new_tokens == new_seq,
            "KV cache: append_value new_seq={new_seq} != append_key new_tokens={}",
            self.new_tokens
        );
        self.new_tokens = new_seq;
        let row_bytes = self.head_dim * self.dtype.size_bytes();
        for h in 0..self.num_heads {
            let dst = (h * self.max_len + self.current_len) * row_bytes;
            let src = h * new_seq * row_bytes;
            let chunk = new_seq * row_bytes;
            self.layers[layer].value[dst..dst + chunk]
                .copy_from_slice(&new_v.as_bytes()[src..src + chunk]);
        }
    }

    /// Return a zero-copy K view covering `[1, heads, 0..current_len+1, head_dim]`.
    ///
    /// Includes the entry just appended at `current_len` (before `advance()`).
    /// The returned Buffer uses max_len-based strides so the kernel's memref
    /// descriptor correctly skips over the unused max_len gap between heads.
    pub fn view_key(&self, layer: usize) -> Buffer {
        self.view_single(&self.layers[layer].key, self.current_len + self.new_tokens)
    }

    /// Return a zero-copy V view covering `[1, heads, 0..current_len+new_tokens, head_dim]`.
    pub fn view_value(&self, layer: usize) -> Buffer {
        self.view_single(
            &self.layers[layer].value,
            self.current_len + self.new_tokens,
        )
    }

    /// Return a zero-copy view of `[1, heads, 0..seq_len, head_dim]` into
    /// a max_len-strided pre-allocated buffer.
    ///
    /// The shape is `[1, num_heads, seq_len, head_dim]` but strides are based
    /// on max_len, not seq_len. The compiled kernel uses these strides via its
    /// memref descriptor to correctly access each head's data with the max_len
    /// gap between heads.
    ///
    /// # Safety
    ///
    /// The returned Buffer holds a raw pointer into `src`. It must not outlive
    /// the KvCache (and thus `src`) — this is guaranteed by the call sites in
    /// `view_key` and `view_value`, which pass slices from `self.layers`.
    fn view_single(&self, src: &[u8], seq_len: usize) -> Buffer {
        let view_shape = [
            1u64,
            self.num_heads as u64,
            seq_len as u64,
            self.head_dim as u64,
        ];
        // Strides in element counts, based on max_len layout.
        // The seq dimension strides by head_dim (row size), not seq_len * head_dim.
        // The head dimension strides by max_len * head_dim (full row in buffer).
        let strides = vec![
            (self.num_heads * self.max_len * self.head_dim) as i64,
            (self.max_len * self.head_dim) as i64,
            self.head_dim as i64,
            1i64,
        ];
        Buffer {
            data: BufferData::View {
                ptr: src.as_ptr(),
                len: src.len(),
            },
            shape: Shape(view_shape.to_vec()),
            strides,
            dtype: self.dtype,
        }
    }

    /// Advance the sequence position after all layers have appended.
    #[cfg(test)]
    pub fn advance(&mut self) {
        self.current_len += 1;
        self.new_tokens = 0;
    }

    /// Advance by N positions (for multi-token KV cache append).
    pub fn advance_by(&mut self, n: usize) {
        self.current_len += n;
        self.new_tokens = 0;
    }

    /// Reset for a new sequence. If `initial_len > 0`, the caller must
    /// populate the cache with prefill data via `append` calls.
    pub fn reset(&mut self, initial_len: usize) {
        self.current_len = initial_len;
    }

    /// Current sequence length stored in the cache.
    pub fn current_len(&self) -> usize {
        self.current_len
    }
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
    let ptr_val = if data_ptr.is_null() || shape.contains(&0) {
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
    pub fn run_into(&self, inputs: &[&Buffer], output_bufs: &mut [Buffer]) {
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
        output_bufs: &mut [Buffer],
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
    fn run_with_output_bufs(&self, inputs: &[&Buffer], output_bufs: &mut [Buffer]) {
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
                build_memref_descriptor(buf.as_ptr() as *mut u8, shape.as_slice(), strides)
            })
            .collect();

        let mut output_descs: Vec<Vec<u8>> = output_bufs
            .iter_mut()
            .map(|buf| {
                let shape_i64: Vec<i64> = buf.shape().0.iter().map(|&d| d as i64).collect();
                let strides = buf.strides().to_vec();
                build_memref_descriptor(buf.as_mut_ptr(), &shape_i64, &strides)
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
            log_trace!("memref inputs: count={}", inputs.len());
            for (i, buf) in inputs.iter().enumerate() {
                let data_ptr = buf.as_ptr();
                match buf.dtype() {
                    DType::F32 => {
                        let n = buf.byte_len() / 4;
                        let show = n.min(4);
                        let elems: Vec<f32> = (0..show)
                            .map(|k| {
                                let bytes = &buf.as_bytes()[k * 4..(k + 1) * 4];
                                f32::from_ne_bytes(bytes.try_into().unwrap())
                            })
                            .collect();
                        log_debug!(
                            "input memref[{i}]: shape={:?} dtype={:?} ptr={:?} first_elems={:?}",
                            buf.shape().0,
                            buf.dtype(),
                            data_ptr,
                            elems
                        );
                    }
                    DType::I64 => {
                        let n = buf.byte_len() / 8;
                        let show = n.min(4);
                        let elems: Vec<i64> = (0..show)
                            .map(|k| {
                                let bytes = &buf.as_bytes()[k * 8..(k + 1) * 8];
                                i64::from_ne_bytes(bytes.try_into().unwrap())
                            })
                            .collect();
                        log_debug!(
                            "input memref[{i}]: shape={:?} dtype={:?} ptr={:?} first_elems={:?}",
                            buf.shape().0,
                            buf.dtype(),
                            data_ptr,
                            elems
                        );
                    }
                    _ => {
                        log_debug!(
                            "input memref[{i}]: shape={:?} dtype={:?} ptr={:?}",
                            buf.shape().0,
                            buf.dtype(),
                            data_ptr
                        );
                    }
                }
            }
            log_trace!("memref outputs: count={}", output_bufs.len());
            for (i, buf) in output_bufs.iter().enumerate() {
                let data_ptr = buf.as_ptr();
                log_debug!(
                    "output memref[{i}]: shape={:?} dtype={:?} ptr={:?} size_bytes={}",
                    buf.shape().0,
                    buf.dtype(),
                    data_ptr,
                    buf.byte_len()
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
    /// If set, execute this native operation instead of calling a compiled kernel.
    #[allow(clippy::struct_field_names)]
    pub native_op: Option<NativeOp>,
}

impl KernelStep {
    /// Create a kernel step (compiled kernel).
    pub fn kernel(kernel_idx: usize, input_slots: Vec<usize>, output_slots: Vec<usize>) -> Self {
        Self {
            kernel_idx,
            input_slots,
            output_slots,
            native_op: None,
        }
    }
}

/// A native operation executed in Rust instead of a compiled kernel.
/// Avoids kernel launch overhead for simple data-movement ops.
#[derive(Clone, Debug)]
pub enum NativeOp {
    /// Concatenate inputs along `axis` into the output buffer.
    /// Inputs: [past, new], output: past ++ new along axis.
    Concat { axis: usize },
}

/// Metadata for KV cache management within an ExecutionPlan.
/// Present only when the model has KV Concat ops that should use cache in run_kv().
#[derive(Clone, Debug)]
pub struct KvPlanInfo {
    /// Number of transformer layers with KV caching.
    pub num_layers: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Buffer slot indices for past_kv model inputs (to be excluded from external inputs).
    /// Ordered: [layer0_k, layer0_v, layer1_k, layer1_v, ...].
    pub past_kv_input_slots: Vec<usize>,
    /// Buffer slot indices for present_kv model outputs (to be excluded from external outputs).
    /// Same ordering as past_kv_input_slots.
    pub present_kv_output_slots: Vec<usize>,
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
    /// KV cache metadata. Present when the model has KV Concat ops.
    pub(crate) kv_info: Option<KvPlanInfo>,
    /// Persistent KV cache, created on first `run_kv` call.
    kv_cache: Option<KvCache>,
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
            kv_info: None,
            kv_cache: None,
        }
    }

    /// Attach KV cache metadata to this plan.
    pub(crate) fn set_kv_info(&mut self, info: KvPlanInfo) {
        self.kv_info = Some(info);
    }

    /// Whether this plan has KV cache support.
    pub fn has_kv_cache(&self) -> bool {
        self.kv_info.is_some()
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
                        if let Some(&existing) = bindings.get(name)
                            && existing != actual
                        {
                            log_warn!(
                                "conflicting sym dim binding: name={} existing={} actual={}",
                                name,
                                existing,
                                actual
                            );
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
        let debug_steps = std::env::var("HOMURA_DEBUG_STEPS").is_ok();
        for (step_idx, step) in self.steps.iter().enumerate() {
            if debug_steps {
                let in_shapes: Vec<_> = step
                    .input_slots
                    .iter()
                    .map(|&s| {
                        pool[s]
                            .as_ref()
                            .map(|e| e.as_ref().shape().0.clone())
                            .unwrap_or_default()
                    })
                    .collect();
                let out_shapes: Vec<_> = step
                    .output_slots
                    .iter()
                    .map(|&s| resolved_shapes[s].0.clone())
                    .collect();
                eprintln!(
                    "[step {step_idx}] k={} native={:?} in_slots={:?} in_shapes={:?} out_slots={:?} out_shapes={:?}",
                    step.kernel_idx,
                    step.native_op.is_some(),
                    step.input_slots,
                    in_shapes,
                    step.output_slots,
                    out_shapes
                );
            }
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
                .any(|&slot| resolved_shapes[slot].0.contains(&0));
            if has_zero_output {
                log_debug!(
                    "skipping kernel {}: output has zero dimension",
                    step.kernel_idx
                );
                for &slot in &step.output_slots {
                    let shape = &resolved_shapes[slot];
                    let dtype = self.slot_descs[slot].dtype;
                    pool[slot] = Some(PoolEntry::Owned(Buffer::new(&shape.0, dtype)));
                }
                continue;
            }

            // Native ops: execute in Rust instead of calling a compiled kernel.
            if let Some(ref native_op) = step.native_op {
                let t0 = if profile {
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                let (buf, out_slot) = match native_op {
                    NativeOp::Concat { axis } => {
                        assert_eq!(step.output_slots.len(), 1);
                        let out_slot = step.output_slots[0];
                        let out_shape = &resolved_shapes[out_slot];
                        let dtype = self.slot_descs[out_slot].dtype;
                        let buf = native_concat(&step_inputs, *axis, &out_shape.0, dtype);
                        (buf, out_slot)
                    }
                };
                if let Some(t0) = t0 {
                    let shapes: Vec<Vec<u64>> =
                        step_inputs.iter().map(|b| b.shape().0.clone()).collect();
                    step_times.push((step.kernel_idx, t0.elapsed(), shapes));
                }
                drop(step_inputs); // Release pool borrows before mutating.
                pool[out_slot] = Some(PoolEntry::Owned(buf));
                // Recycle dead inputs.
                for &slot in &step.input_slots {
                    if let Some(last) = self.slot_last_read[slot]
                        && last == step_idx
                        && let Some(PoolEntry::Owned(buf)) = pool[slot].take()
                    {
                        free_list.push(buf);
                    }
                }
                continue;
            }

            // Compiled kernel — safe to index now (native ops continued above).
            let kernel = &self.kernels[step.kernel_idx];

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
                    let reuse_idx = free_list.iter().position(|b| b.capacity() >= need_bytes);
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
                if let Some(last) = self.slot_last_read[slot]
                    && last == step_idx
                    && let Some(PoolEntry::Owned(buf)) = pool[slot].take()
                {
                    free_list.push(buf);
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

    /// Run the plan with internal KV cache management.
    ///
    /// `inputs` should contain only the non-KV model inputs (e.g., input_ids +
    /// attention_mask). The past_kv inputs are populated from the internal
    /// KvCache, and present_kv outputs are fed back to the cache automatically.
    ///
    /// On first call, initializes the KV cache with `max_seq_len` capacity.
    /// Returns only the non-KV model outputs (e.g., logits).
    ///
    /// After each call, the cache advances by 1 position.
    pub fn run_kv(
        &mut self,
        inputs: &[&Buffer],
        weights: &[Buffer],
        max_seq_len: usize,
    ) -> Vec<Buffer> {
        let kv_info = self
            .kv_info
            .as_ref()
            .expect("run_kv called but plan has no KV cache info")
            .clone();

        // Initialize KV cache on first call.
        if self.kv_cache.is_none() {
            self.kv_cache = Some(KvCache::new(
                kv_info.num_layers,
                kv_info.num_heads,
                max_seq_len,
                kv_info.head_dim,
                DType::F32,
            ));
        }

        // The external inputs exclude past_kv slots. Map them to the
        // non-KV input slots (those not in past_kv_input_slots).
        let past_kv_set: std::collections::HashSet<usize> =
            kv_info.past_kv_input_slots.iter().copied().collect();
        let non_kv_input_slots: Vec<usize> = self
            .input_slots
            .iter()
            .copied()
            .filter(|s| !past_kv_set.contains(s))
            .collect();
        assert_eq!(
            inputs.len(),
            non_kv_input_slots.len(),
            "run_kv: expected {} non-KV inputs, got {}",
            non_kv_input_slots.len(),
            inputs.len()
        );

        // Resolve shapes using the KV cache shape (no data copy needed).
        let cache = self.kv_cache.as_ref().unwrap();
        let (shape_k, shape_v) = cache.view_shape();
        let kv_views: Vec<&Buffer> = (0..kv_info.num_layers)
            .flat_map(|_| vec![&shape_k, &shape_v])
            .collect();

        // Build the full input list matching input_slots order.
        let mut full_inputs: Vec<&Buffer> = Vec::with_capacity(self.input_slots.len());
        let mut ext_idx = 0;
        let mut kv_idx = 0;
        for &slot in &self.input_slots {
            if past_kv_set.contains(&slot) {
                full_inputs.push(kv_views[kv_idx]);
                kv_idx += 1;
            } else {
                full_inputs.push(inputs[ext_idx]);
                ext_idx += 1;
            }
        }

        let resolved_shapes = self.resolve_slot_shapes(&full_inputs);

        let profile = std::env::var("HOMURA_PROFILE").is_ok_and(|v| v == "1");
        let mut step_times: Vec<(usize, std::time::Duration, Vec<Vec<u64>>)> = Vec::new();

        // Initialize pool.
        let mut pool: Vec<Option<PoolEntry<'_>>> = (0..self.num_slots).map(|_| None).collect();
        for (&slot, input) in self.input_slots.iter().zip(full_inputs.iter()) {
            pool[slot] = Some(PoolEntry::Borrowed(input));
        }
        for (&slot, weight) in self.weight_slots.iter().zip(weights.iter()) {
            pool[slot] = Some(PoolEntry::Borrowed(weight));
        }

        let mut free_list: Vec<Buffer> = Vec::new();

        // Build KV concat lookup: present_kv output slot → (layer, is_value).
        let mut kv_output_meta: std::collections::HashMap<usize, (usize, bool)> =
            std::collections::HashMap::new();
        if let Some(ref info) = self.kv_info {
            for (i, &slot) in info.present_kv_output_slots.iter().enumerate() {
                kv_output_meta.insert(slot, (i / 2, (i % 2) == 1));
            }
        }

        // Execute steps — same as run(), but KV Concat steps use cache.append+view.
        for (step_idx, step) in self.steps.iter().enumerate() {
            let step_inputs: Vec<&Buffer> = step
                .input_slots
                .iter()
                .map(|&s| {
                    pool[s]
                        .as_ref()
                        .unwrap_or_else(|| {
                            panic!("buffer slot {s} not populated at step {step_idx}")
                        })
                        .as_ref()
                })
                .collect();

            let has_zero_output = step
                .output_slots
                .iter()
                .any(|&slot| resolved_shapes[slot].0.contains(&0));
            if has_zero_output {
                for &slot in &step.output_slots {
                    let shape = &resolved_shapes[slot];
                    let dtype = self.slot_descs[slot].dtype;
                    pool[slot] = Some(PoolEntry::Owned(Buffer::new(&shape.0, dtype)));
                }
                continue;
            }

            if let Some(ref native_op) = step.native_op {
                let t0 = if profile {
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                let (buf, out_slot) = match native_op {
                    NativeOp::Concat { axis } => {
                        let out_slot = step.output_slots[0];
                        // Check if this Concat is a KV concat.
                        if let Some(&(layer, is_value)) = kv_output_meta.get(&out_slot) {
                            let cache = self.kv_cache.as_mut().expect("KV concat but no KvCache");
                            let new_data = step
                                .input_slots
                                .iter()
                                .zip(step_inputs.iter())
                                .find(|(slot, _)| !past_kv_set.contains(slot))
                                .expect("KV concat has no non-past input")
                                .1;
                            if is_value {
                                cache.append_value(layer, new_data);
                            } else {
                                cache.append_key(layer, new_data);
                            }
                            let view = if is_value {
                                cache.view_value(layer)
                            } else {
                                cache.view_key(layer)
                            };
                            (view, out_slot)
                        } else {
                            let out_shape = &resolved_shapes[out_slot];
                            let dtype = self.slot_descs[out_slot].dtype;
                            let buf = native_concat(&step_inputs, *axis, &out_shape.0, dtype);
                            (buf, out_slot)
                        }
                    }
                };
                if let Some(t0) = t0 {
                    let shapes: Vec<Vec<u64>> =
                        step_inputs.iter().map(|b| b.shape().0.clone()).collect();
                    step_times.push((step.kernel_idx, t0.elapsed(), shapes));
                }
                drop(step_inputs);
                pool[out_slot] = Some(PoolEntry::Owned(buf));
                for &slot in &step.input_slots {
                    if let Some(last) = self.slot_last_read[slot]
                        && last == step_idx
                        && let Some(PoolEntry::Owned(buf)) = pool[slot].take()
                    {
                        free_list.push(buf);
                    }
                }
                continue;
            }

            let kernel = &self.kernels[step.kernel_idx];
            let has_dynamic = kernel
                .output_descs()
                .iter()
                .any(|d| d.shape.has_dynamic_dims());

            let t0 = if profile {
                Some(std::time::Instant::now())
            } else {
                None
            };

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
                    let reuse_idx = free_list.iter().position(|b| b.capacity() >= need_bytes);
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

            for (buf, &slot) in out_bufs.into_iter().zip(step.output_slots.iter()) {
                pool[slot] = Some(PoolEntry::Owned(buf));
            }

            for &slot in &step.input_slots {
                if let Some(last) = self.slot_last_read[slot]
                    && last == step_idx
                    && let Some(PoolEntry::Owned(buf)) = pool[slot].take()
                {
                    free_list.push(buf);
                }
            }
        }

        // Print per-step profiling summary.
        if profile && !step_times.is_empty() {
            let total: std::time::Duration = step_times.iter().map(|(_, d, _)| *d).sum();
            eprintln!(
                "  ┌─ decode kernel profile ({} steps, {:.1}ms total)",
                step_times.len(),
                total.as_secs_f64() * 1000.0
            );
            for (kid, dur, shapes) in &step_times {
                let ms = dur.as_secs_f64() * 1000.0;
                if ms >= 0.01 {
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

        // Advance KV cache by the number of tokens appended during this step.
        if let Some(ref mut cache) = self.kv_cache {
            if cache.new_tokens > 0 {
                cache.advance_by(cache.new_tokens);
            }
        }

        // Extract non-KV outputs only.
        let present_kv_set: std::collections::HashSet<usize> =
            kv_info.present_kv_output_slots.iter().copied().collect();
        self.output_slots
            .iter()
            .filter(|s| !present_kv_set.contains(s))
            .map(|&s| {
                pool[s]
                    .take()
                    .unwrap_or_else(|| panic!("output slot {s} not populated"))
                    .into_owned()
            })
            .collect()
    }

    /// Initialize the KV cache from prefill output buffers.
    ///
    /// `kv_buffers` should contain `num_layers * 2` buffers (K and V alternating),
    /// each shaped `[1, heads, seq_len, head_dim]`.
    pub fn init_kv_cache(&mut self, kv_buffers: &[Buffer], max_seq_len: usize) {
        let kv_info = self
            .kv_info
            .as_ref()
            .expect("init_kv_cache called but plan has no KV cache info");
        let num_layers = kv_info.num_layers;
        let num_heads = kv_info.num_heads;
        let head_dim = kv_info.head_dim;
        assert_eq!(
            kv_buffers.len(),
            num_layers * 2,
            "expected {} KV buffers, got {}",
            num_layers * 2,
            kv_buffers.len()
        );

        let mut cache = KvCache::new(num_layers, num_heads, max_seq_len, head_dim, DType::F32);

        // Bulk-load prefill KV data.
        let seq_len = kv_buffers[0].shape().0[2] as usize;
        let elem_size = DType::F32.size_bytes();
        let row_bytes = head_dim * elem_size;

        for layer in 0..num_layers {
            let k_buf = &kv_buffers[layer * 2];
            let v_buf = &kv_buffers[layer * 2 + 1];
            // Copy each head's [0..seq_len] into the max_len-strided buffer.
            for h in 0..num_heads {
                let src_off = h * seq_len * row_bytes;
                let dst_off = h * max_seq_len * row_bytes;
                let chunk = seq_len * row_bytes;
                cache.layers[layer].key[dst_off..dst_off + chunk]
                    .copy_from_slice(&k_buf.as_bytes()[src_off..src_off + chunk]);
                cache.layers[layer].value[dst_off..dst_off + chunk]
                    .copy_from_slice(&v_buf.as_bytes()[src_off..src_off + chunk]);
            }
        }
        cache.current_len = seq_len;

        self.kv_cache = Some(cache);
    }

    /// Reset the KV cache for a new sequence.
    pub fn reset_kv_cache(&mut self) {
        if let Some(ref mut cache) = self.kv_cache {
            cache.reset(0);
        }
    }

    /// Current sequence length in the KV cache.
    pub fn kv_cache_len(&self) -> usize {
        self.kv_cache.as_ref().map(|c| c.current_len()).unwrap_or(0)
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
                    native_op: None,
                },
                KernelStep {
                    kernel_idx: 1,
                    input_slots: vec![2, 1], // c, b
                    output_slots: vec![3],   // d
                    native_op: None,
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
                native_op: None,
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
                native_op: None,
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

    // ── KvCache unit tests ──────────────────────────────────────────────────

    #[test]
    fn kv_cache_append_and_view() {
        use super::KvCache;

        // 1 layer, 2 heads, max 4 tokens, head_dim=3, f32
        let mut cache = KvCache::new(1, 2, 4, 3, DType::F32);
        assert_eq!(cache.current_len(), 0);

        // Append first token: new_k shape [1, 2, 1, 3]
        let k0 =
            Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 1, 3], DType::F32);
        let v0 = Buffer::from_slice::<f32>(
            &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            &[1, 2, 1, 3],
            DType::F32,
        );
        cache.append(0, &k0, &v0);
        cache.advance();
        assert_eq!(cache.current_len(), 1);

        // View should be [1, 2, 1, 3]
        let (kv, vv) = cache.view(0);
        assert_eq!(kv.shape().0, vec![1, 2, 1, 3]);
        assert_eq!(kv.as_slice::<f32>(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(vv.as_slice::<f32>(), &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);

        // Append second token.
        let k1 =
            Buffer::from_slice::<f32>(&[7.0, 8.0, 9.0, 0.1, 0.2, 0.3], &[1, 2, 1, 3], DType::F32);
        let v1 = Buffer::from_slice::<f32>(
            &[70.0, 80.0, 90.0, 0.4, 0.5, 0.6],
            &[1, 2, 1, 3],
            DType::F32,
        );
        cache.append(0, &k1, &v1);
        cache.advance();
        assert_eq!(cache.current_len(), 2);

        // View should be [1, 2, 2, 3] with both tokens.
        let (kv, vv) = cache.view(0);
        assert_eq!(kv.shape().0, vec![1, 2, 2, 3]);
        // Head 0: [tok0, tok1], Head 1: [tok0, tok1]
        assert_eq!(
            kv.as_slice::<f32>(),
            &[1.0, 2.0, 3.0, 7.0, 8.0, 9.0, 4.0, 5.0, 6.0, 0.1, 0.2, 0.3]
        );
        assert_eq!(
            vv.as_slice::<f32>(),
            &[
                10.0, 20.0, 30.0, 70.0, 80.0, 90.0, 40.0, 50.0, 60.0, 0.4, 0.5, 0.6
            ]
        );
    }

    #[test]
    fn kv_cache_multi_token_append() {
        use super::KvCache;

        // 1 layer, 2 heads, max 8 tokens, head_dim=3, f32
        let mut cache = KvCache::new(1, 2, 8, 3, DType::F32);

        // Append 3 tokens at once: shape [1, 2, 3, 3]
        // Head 0: [[1,2,3], [4,5,6], [7,8,9]], Head 1: [[10,11,12], [13,14,15], [16,17,18]]
        let k = Buffer::from_slice::<f32>(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, // head 0
                10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, // head 1
            ],
            &[1, 2, 3, 3],
            DType::F32,
        );
        let v = Buffer::from_slice::<f32>(
            &[
                91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, // head 0
                81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, // head 1
            ],
            &[1, 2, 3, 3],
            DType::F32,
        );
        cache.append_key(0, &k);
        cache.append_value(0, &v);
        cache.advance_by(3);
        assert_eq!(cache.current_len(), 3);

        // view (contiguous copy) should show all 3 tokens.
        let (kv, vv) = cache.view(0);
        assert_eq!(kv.shape().0, vec![1, 2, 3, 3]);
        assert_eq!(
            kv.as_slice::<f32>(),
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0
            ]
        );
        assert_eq!(vv.shape().0, vec![1, 2, 3, 3]);

        // Now append 1 more token (single-token path still works).
        let k1 = Buffer::from_slice::<f32>(
            &[20.0, 21.0, 22.0, 30.0, 31.0, 32.0],
            &[1, 2, 1, 3],
            DType::F32,
        );
        let v1 = Buffer::from_slice::<f32>(
            &[40.0, 41.0, 42.0, 50.0, 51.0, 52.0],
            &[1, 2, 1, 3],
            DType::F32,
        );
        cache.append_key(0, &k1);
        cache.append_value(0, &v1);
        cache.advance_by(1);
        assert_eq!(cache.current_len(), 4);

        let (kv, _) = cache.view(0);
        assert_eq!(kv.shape().0, vec![1, 2, 4, 3]);
        // Head 0: 3 multi-token entries + 1 single = [1..9, 20..22]
        // Head 1: 3 multi-token entries + 1 single = [10..18, 30..32]
        assert_eq!(
            kv.as_slice::<f32>(),
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 20.0, 21.0, 22.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 30.0, 31.0, 32.0
            ]
        );
    }

    #[test]
    fn kv_cache_reset() {
        use super::KvCache;

        let mut cache = KvCache::new(2, 1, 8, 4, DType::F32);
        let k = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 1, 4], DType::F32);
        let v = Buffer::from_slice::<f32>(&[5.0, 6.0, 7.0, 8.0], &[1, 1, 1, 4], DType::F32);
        cache.append(0, &k, &v);
        cache.advance();
        assert_eq!(cache.current_len(), 1);

        cache.reset(0);
        assert_eq!(cache.current_len(), 0);
    }
}
