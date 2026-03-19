use crate::DType;
use crate::runtime::Buffer;

/// Precompute RoPE cos/sin tables (HuggingFace half-rotation convention).
///
/// Returns (cos, sin) buffers of shape `[max_seq_len, head_dim]`.
/// The first `head_dim/2` elements and second `head_dim/2` elements are
/// identical (matching PyTorch's `rotate_half` convention where
/// `q_embed = q * cos + rotate_half(q) * sin`).
pub fn precompute_rope_cos_sin(
    head_dim: usize,
    max_seq_len: usize,
    theta: f64,
) -> (Buffer, Buffer) {
    let half_dim = head_dim / 2;
    let num_elems = max_seq_len * head_dim;
    let mut cos_data = vec![0.0f32; num_elems];
    let mut sin_data = vec![0.0f32; num_elems];

    for pos in 0..max_seq_len {
        for i in 0..half_dim {
            let freq = 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64);
            let angle = pos as f64 * freq;
            let c = angle.cos() as f32;
            let s = angle.sin() as f32;
            // Duplicate: first half and second half are identical
            cos_data[pos * head_dim + i] = c;
            cos_data[pos * head_dim + half_dim + i] = c;
            sin_data[pos * head_dim + i] = s;
            sin_data[pos * head_dim + half_dim + i] = s;
        }
    }

    let shape = &[max_seq_len as u64, head_dim as u64];
    let cos_buf = Buffer::from_slice::<f32>(&cos_data, shape, DType::F32);
    let sin_buf = Buffer::from_slice::<f32>(&sin_data, shape, DType::F32);
    (cos_buf, sin_buf)
}

/// Build a causal attention mask.
///
/// Returns a buffer of shape [1, 1, seq_len, seq_len + past_len].
/// mask[0, 0, i, j] = 0.0 if j <= i + past_len, else -infinity (f32::NEG_INFINITY).
///
/// For decode (seq_len=1): all positions are attended (all 0.0).
/// For prefill (past_len=0): lower-triangular mask.
pub fn build_causal_mask(seq_len: usize, past_len: usize) -> Buffer {
    let total_len = seq_len + past_len;
    let num_elems = seq_len * total_len;
    let mut data = vec![0.0f32; num_elems];

    for i in 0..seq_len {
        for j in 0..total_len {
            if j > i + past_len {
                data[i * total_len + j] = f32::NEG_INFINITY;
            }
        }
    }

    let shape = &[1u64, 1, seq_len as u64, total_len as u64];
    Buffer::from_slice::<f32>(&data, shape, DType::F32)
}

/// Slice RoPE tables for specific positions.
///
/// Given full cos/sin tables of shape [max_seq_len, dim], extract rows
/// at the given positions. Returns (cos, sin) of shape [positions.len(), dim].
///
/// For prefill: positions = [0, 1, ..., seq_len-1]
/// For decode:  positions = [past_len]
pub fn slice_rope_for_positions(
    full_cos: &Buffer,
    full_sin: &Buffer,
    positions: &[usize],
) -> (Buffer, Buffer) {
    let shape = full_cos.shape();
    let row_dim = shape.0[1] as usize;
    let n = positions.len();

    let full_cos_data = full_cos.as_slice::<f32>();
    let full_sin_data = full_sin.as_slice::<f32>();

    let mut cos_data = vec![0.0f32; n * row_dim];
    let mut sin_data = vec![0.0f32; n * row_dim];

    for (out_row, &pos) in positions.iter().enumerate() {
        let src_start = pos * row_dim;
        let dst_start = out_row * row_dim;
        cos_data[dst_start..dst_start + row_dim]
            .copy_from_slice(&full_cos_data[src_start..src_start + row_dim]);
        sin_data[dst_start..dst_start + row_dim]
            .copy_from_slice(&full_sin_data[src_start..src_start + row_dim]);
    }

    let out_shape = &[n as u64, row_dim as u64];
    let cos_buf = Buffer::from_slice::<f32>(&cos_data, out_shape, DType::F32);
    let sin_buf = Buffer::from_slice::<f32>(&sin_data, out_shape, DType::F32);
    (cos_buf, sin_buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_basic_values() {
        // theta=10000, head_dim=4, max_seq_len=3
        // freq[0] = 1/10000^(0/4) = 1.0
        // freq[1] = 1/10000^(2/4) = 1/100 = 0.01
        // cos[pos=0] = [cos(0), cos(0)] = [1.0, 1.0]
        // cos[pos=1] = [cos(1.0), cos(0.01)] = [0.5403, 0.99995]
        let (cos, sin) = precompute_rope_cos_sin(4, 3, 10_000.0);
        assert_eq!(cos.shape().0, vec![3, 2]);
        let cos_data = cos.as_slice::<f32>();
        assert!((cos_data[0] - 1.0).abs() < 1e-5); // pos=0, i=0
        assert!((cos_data[1] - 1.0).abs() < 1e-5); // pos=0, i=1
        assert!((cos_data[2] - 0.5403023).abs() < 1e-4); // pos=1, i=0
        assert!((cos_data[3] - 0.99995).abs() < 1e-4); // pos=1, i=1
        let sin_data = sin.as_slice::<f32>();
        assert!((sin_data[0]).abs() < 1e-5); // pos=0 => sin(0)
        assert!((sin_data[2] - 0.8414709).abs() < 1e-4); // pos=1, i=0 => sin(1.0)
    }

    #[test]
    fn causal_mask_prefill() {
        let mask = build_causal_mask(3, 0);
        assert_eq!(mask.shape().0, vec![1, 1, 3, 3]);
        let d = mask.as_slice::<f32>();
        // Row 0: [0, -inf, -inf]
        assert_eq!(d[0], 0.0);
        assert!(d[1].is_infinite() && d[1] < 0.0);
        assert!(d[2].is_infinite() && d[2] < 0.0);
        // Row 1: [0, 0, -inf]
        assert_eq!(d[3], 0.0);
        assert_eq!(d[4], 0.0);
        assert!(d[5].is_infinite() && d[5] < 0.0);
        // Row 2: [0, 0, 0]
        assert_eq!(d[6], 0.0);
        assert_eq!(d[7], 0.0);
        assert_eq!(d[8], 0.0);
    }

    #[test]
    fn causal_mask_decode() {
        // seq_len=1, past_len=5 => shape [1,1,1,6], all zeros
        let mask = build_causal_mask(1, 5);
        assert_eq!(mask.shape().0, vec![1, 1, 1, 6]);
        let d = mask.as_slice::<f32>();
        for &v in d {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn slice_rope() {
        let (full_cos, full_sin) = precompute_rope_cos_sin(4, 10, 10_000.0);
        // Slice positions [3, 7]
        let (cos, _sin) = slice_rope_for_positions(&full_cos, &full_sin, &[3, 7]);
        assert_eq!(cos.shape().0, vec![2, 2]);
        let full_cos_data = full_cos.as_slice::<f32>();
        let cos_data = cos.as_slice::<f32>();
        // Row 0 should match position 3 in full table
        assert_eq!(cos_data[0], full_cos_data[3 * 2]);
        assert_eq!(cos_data[1], full_cos_data[3 * 2 + 1]);
        // Row 1 should match position 7
        assert_eq!(cos_data[2], full_cos_data[7 * 2]);
        assert_eq!(cos_data[3], full_cos_data[7 * 2 + 1]);
    }
}
