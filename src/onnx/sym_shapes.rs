//! Symbolic shape propagation for ONNX nodes.
//!
//! Each function computes output symbolic shapes from input symbolic shapes for a given
//! ONNX op type. These are called during the emission loop in `emit_and_compile_plan`
//! to maintain a parallel `sym_shape_info` map alongside the existing MLIR-derived
//! `shape_info` map.
//!
//! The goal is to propagate symbolic dimensions (e.g., `past_sequence_length`) through
//! Shape → Gather → Concat → Reshape chains so that downstream buffer slots have
//! symbolic shapes instead of all-dynamic (DIM_DYNAMIC) shapes.

use std::collections::HashMap;

use super::parser::{OnnxAttribute, OnnxNode};
use crate::shape::{SymDim, SymShape};

// ── Broadcasting ──────────────────────────────────────────────────────────────

/// Broadcast two symbolic dims at a single position.
/// Rule: if one is Concrete(1), take the other. If both equal, keep one. Otherwise prefer
/// the non-Concrete(1) one. If both are Concrete(1) or the same, return the first.
fn broadcast_sym_dim(a: &SymDim, b: &SymDim) -> SymDim {
    match (a, b) {
        (SymDim::Concrete(1), other) => other.clone(),
        (other, SymDim::Concrete(1)) => other.clone(),
        (x, y) if x == y => x.clone(),
        // Both non-1 and different — can't resolve, use first (best-effort)
        (x, _) => x.clone(),
    }
}

/// Compute numpy-style broadcast shape for two symbolic shapes.
/// Right-aligns, treats missing leading dims as Concrete(1).
fn broadcast_sym_shapes(a: &SymShape, b: &SymShape) -> SymShape {
    let max_rank = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_rank);
    for i in 0..max_rank {
        let a_dim = if i < max_rank - a.len() {
            &SymDim::Concrete(1)
        } else {
            &a[i - (max_rank - a.len())]
        };
        let b_dim = if i < max_rank - b.len() {
            &SymDim::Concrete(1)
        } else {
            &b[i - (max_rank - b.len())]
        };
        result.push(broadcast_sym_dim(a_dim, b_dim));
    }
    result
}

// ── Axis normalization ────────────────────────────────────────────────────────

fn normalize_axis(axis: i64, rank: usize) -> usize {
    let r = rank as i64;
    let a = if axis < 0 { axis + r } else { axis };
    a.max(0).min(r - 1) as usize
}

// ── Attribute helpers (mirrors emitter.rs helpers but without error) ──────────

fn get_int_attr(attrs: &HashMap<String, OnnxAttribute>, name: &str, default: i64) -> i64 {
    attrs
        .get(name)
        .and_then(|a| {
            if let OnnxAttribute::Int(v) = a {
                Some(*v)
            } else {
                None
            }
        })
        .unwrap_or(default)
}

fn get_bool_attr(attrs: &HashMap<String, OnnxAttribute>, name: &str) -> bool {
    attrs
        .get(name)
        .and_then(|a| {
            if let OnnxAttribute::Int(v) = a {
                Some(*v != 0)
            } else {
                None
            }
        })
        .unwrap_or(false)
}

fn get_keepdims_attr(attrs: &HashMap<String, OnnxAttribute>, default: bool) -> bool {
    attrs
        .get("keepdims")
        .and_then(|a| {
            if let OnnxAttribute::Int(v) = a {
                Some(*v != 0)
            } else {
                None
            }
        })
        .unwrap_or(default)
}

// ── Main dispatch ─────────────────────────────────────────────────────────────

/// Compute symbolic output shapes for an ONNX node.
///
/// `input_sym_shapes`: symbolic shapes of each node input (in order); may be empty
///   if an input has no entry in `sym_shape_info` (e.g., optional inputs).
/// `sym_const_i64`: symbolic integer tensor values (for Shape/Gather/Reshape chains).
/// `const_i64`: concrete integer tensor values (for static Reshape targets, etc.).
/// `node`: the ONNX node being processed.
///
/// Returns `(output_sym_shapes, sym_const_i64_updates)` where:
/// - `output_sym_shapes[i]` is the symbolic shape for `node.outputs[i]`
/// - `sym_const_i64_updates` is a map of new symbolic constant entries to merge
pub fn propagate_sym_shapes(
    node: &OnnxNode,
    input_sym_shapes: &[&SymShape],
    sym_const_i64: &HashMap<String, Vec<SymDim>>,
    const_i64: &HashMap<String, Vec<i64>>,
) -> (Vec<SymShape>, HashMap<String, Vec<SymDim>>) {
    let mut sym_updates: HashMap<String, Vec<SymDim>> = HashMap::new();
    let outputs = propagate_inner(
        node,
        input_sym_shapes,
        sym_const_i64,
        const_i64,
        &mut sym_updates,
    );
    (outputs, sym_updates)
}

fn propagate_inner(
    node: &OnnxNode,
    input_sym_shapes: &[&SymShape],
    sym_const_i64: &HashMap<String, Vec<SymDim>>,
    const_i64: &HashMap<String, Vec<i64>>,
    sym_updates: &mut HashMap<String, Vec<SymDim>>,
) -> Vec<SymShape> {
    let get = |idx: usize| -> Option<&SymShape> { input_sym_shapes.get(idx).copied() };
    let attrs = &node.attributes;

    match node.op_type.as_str() {
        // ── Elementwise unary ──────────────────────────────────────────────────
        "Relu" | "Exp" | "Tanh" | "Sigmoid" | "Sqrt" | "Neg" | "Erf" => {
            if let Some(s) = get(0) {
                vec![s.clone()]
            } else {
                vec![]
            }
        }

        "Cast" => {
            if let Some(s) = get(0) {
                let out_sym = s.clone();
                // Propagate sym_const_i64 through Cast.
                if let Some(vals) = sym_const_i64.get(&node.inputs[0]) {
                    sym_updates.insert(node.outputs[0].clone(), vals.clone());
                }
                vec![out_sym]
            } else {
                vec![]
            }
        }

        // ── Elementwise binary/ternary ─────────────────────────────────────────
        "Add" | "Sub" | "Mul" | "Div" | "Pow" => {
            if let (Some(a), Some(b)) = (get(0), get(1)) {
                // Propagate sym_const_i64 through arithmetic on integer tensors.
                if node.op_type == "Add" {
                    if let (Some(va), Some(vb)) = (
                        sym_const_i64.get(&node.inputs[0]),
                        sym_const_i64.get(&node.inputs[1]),
                    ) {
                        let result = sym_const_i64_elementwise(va, vb, |x, y| x.add(y));
                        if let Some(vals) = result {
                            sym_updates.insert(node.outputs[0].clone(), vals);
                        }
                    }
                } else if node.op_type == "Sub" {
                    if let (Some(va), Some(vb)) = (
                        sym_const_i64.get(&node.inputs[0]),
                        sym_const_i64.get(&node.inputs[1]),
                    ) {
                        let result = sym_const_i64_elementwise(va, vb, |x, y| {
                            x.add(
                                SymDim::Concrete(0).add(y), // placeholder: x - y not a native SymDim op
                            )
                        });
                        // For Sub, just skip (no Sub in SymDim).
                        let _ = result;
                    }
                } else if node.op_type == "Mul" {
                    if let (Some(va), Some(vb)) = (
                        sym_const_i64.get(&node.inputs[0]),
                        sym_const_i64.get(&node.inputs[1]),
                    ) {
                        let result = sym_const_i64_elementwise(va, vb, |x, y| x.mul(y));
                        if let Some(vals) = result {
                            sym_updates.insert(node.outputs[0].clone(), vals);
                        }
                    }
                } else if node.op_type == "Div" {
                    if let (Some(va), Some(vb)) = (
                        sym_const_i64.get(&node.inputs[0]),
                        sym_const_i64.get(&node.inputs[1]),
                    ) {
                        if let Some(vals) = sym_const_i64_elementwise(va, vb, |x, y| x.div(y)) {
                            sym_updates.insert(node.outputs[0].clone(), vals);
                        }
                    }
                }
                vec![broadcast_sym_shapes(a, b)]
            } else if let Some(s) = get(0) {
                vec![s.clone()]
            } else {
                vec![]
            }
        }

        "Where" => {
            // Output shape = broadcast of x and y (inputs[1] and inputs[2])
            match (get(1), get(2)) {
                (Some(x), Some(y)) => vec![broadcast_sym_shapes(x, y)],
                (Some(x), None) => vec![x.clone()],
                (None, Some(y)) => vec![y.clone()],
                (None, None) => vec![],
            }
        }

        // ── MatMul ────────────────────────────────────────────────────────────
        "MatMul" => {
            if let (Some(a), Some(b)) = (get(0), get(1)) {
                let out = matmul_sym_shape(a, b);
                vec![out]
            } else {
                vec![]
            }
        }

        // ── Gemm ──────────────────────────────────────────────────────────────
        "Gemm" => {
            if let (Some(a), Some(b)) = (get(0), get(1)) {
                if a.len() != 2 || b.len() != 2 {
                    return vec![];
                }
                let trans_a = get_bool_attr(attrs, "transA");
                let trans_b = get_bool_attr(attrs, "transB");
                let m = if trans_a { a[1].clone() } else { a[0].clone() };
                let n = if trans_b { b[0].clone() } else { b[1].clone() };
                vec![vec![m, n]]
            } else {
                vec![]
            }
        }

        // ── Softmax ───────────────────────────────────────────────────────────
        "Softmax" => {
            if let Some(s) = get(0) {
                vec![s.clone()]
            } else {
                vec![]
            }
        }

        // ── LayerNorm / GroupNorm (if present) ────────────────────────────────
        "LayerNormalization" => {
            if let Some(s) = get(0) {
                vec![s.clone()]
            } else {
                vec![]
            }
        }

        // ── Transpose ─────────────────────────────────────────────────────────
        "Transpose" => {
            if let Some(s) = get(0) {
                let rank = s.len();
                let perm: Vec<i64> = if let Some(OnnxAttribute::Ints(v)) = attrs.get("perm") {
                    v.clone()
                } else {
                    (0..rank as i64).rev().collect()
                };
                let out: SymShape = perm
                    .iter()
                    .map(|&p| {
                        let idx = if p < 0 {
                            (rank as i64 + p) as usize
                        } else {
                            p as usize
                        };
                        s.get(idx).cloned().unwrap_or(SymDim::Concrete(0))
                    })
                    .collect();
                vec![out]
            } else {
                vec![]
            }
        }

        // ── Reshape ───────────────────────────────────────────────────────────
        "Reshape" => {
            let input_sym = get(0);
            let shape_name = node.inputs.get(1).map(|s| s.as_str()).unwrap_or("");
            let allowzero = get_int_attr(attrs, "allowzero", 0);

            // Propagate sym_const_i64 through Reshape (values unchanged).
            if let Some(vals) = sym_const_i64.get(&node.inputs[0]) {
                sym_updates.insert(node.outputs[0].clone(), vals.clone());
            }

            // Case 1: static const_i64 target (all concrete, no CONST_I64_UNKNOWN sentinel).
            if let Some(target_vals) = const_i64.get(shape_name) {
                // Check for CONST_I64_UNKNOWN sentinel (i64::MIN).
                if target_vals.iter().all(|&v| v != i64::MIN) {
                    let out_sym =
                        reshape_sym_with_concrete_target(input_sym, target_vals, allowzero != 0);
                    return vec![out_sym];
                }
            }

            // Case 2: symbolic const target from sym_const_i64.
            if let Some(sym_target) = sym_const_i64.get(shape_name) {
                let out_sym = reshape_sym_with_sym_target(input_sym, sym_target, allowzero != 0);
                return vec![out_sym];
            }

            // Case 3: fallback — return empty (no info).
            vec![]
        }

        // ── Concat ────────────────────────────────────────────────────────────
        "Concat" => {
            let axis = get_int_attr(attrs, "axis", 0);

            // Determine rank from first input.
            let rank = match get(0) {
                Some(s) => s.len(),
                None => return vec![],
            };
            let axis_usize = normalize_axis(axis, rank);

            // Build output shape: copy dims from first input, sum the concat axis.
            let first = get(0).unwrap();
            let mut out = first.clone();

            // Sum all inputs along concat axis.
            let mut concat_dim: SymDim = first
                .get(axis_usize)
                .cloned()
                .unwrap_or(SymDim::Concrete(0));
            for i in 1..node.inputs.len() {
                if let Some(s) = get(i) {
                    let d = s.get(axis_usize).cloned().unwrap_or(SymDim::Concrete(0));
                    concat_dim = concat_dim.add(d).simplify();
                }
            }
            if axis_usize < out.len() {
                out[axis_usize] = concat_dim;
            }

            // Propagate sym_const_i64 through concatenation.
            {
                let mut concat_vals: Vec<SymDim> = Vec::new();
                let mut has_any = false;
                for name in &node.inputs {
                    if let Some(vals) = sym_const_i64.get(name) {
                        concat_vals.extend(vals.iter().cloned());
                        has_any = true;
                    } else if let Some(vals) = const_i64.get(name) {
                        // Fall back to concrete values. Negative values like -1
                        // (Reshape infer) are stored as-is via i64→u64 cast; the
                        // Reshape handler recognizes them.
                        // Skip CONST_I64_UNKNOWN sentinel (i64::MIN) entries.
                        if vals.iter().any(|&v| v == i64::MIN) {
                            // Has unknown entries — skip this input entirely.
                        } else {
                            concat_vals.extend(vals.iter().map(|&v| SymDim::Concrete(v as u64)));
                            has_any = true;
                        }
                    }
                }
                if has_any {
                    sym_updates.insert(node.outputs[0].clone(), concat_vals);
                }
            }

            vec![out]
        }

        // ── Gather ────────────────────────────────────────────────────────────
        "Gather" => {
            let data_sym = match get(0) {
                Some(s) => s,
                None => return vec![],
            };
            let indices_sym = get(1);
            let axis = get_int_attr(attrs, "axis", 0);
            let axis_usize = normalize_axis(axis, data_sym.len());

            // Output = data.shape[:axis] + indices.shape + data.shape[axis+1:]
            let mut out: SymShape = data_sym[..axis_usize].to_vec();
            if let Some(idx_sym) = indices_sym {
                out.extend(idx_sym.iter().cloned());
            }
            if axis_usize + 1 < data_sym.len() {
                out.extend(data_sym[axis_usize + 1..].iter().cloned());
            }

            // Propagate sym_const_i64: if data is a sym_const_i64 entry and indices
            // are concrete scalars, extract those elements.
            let data_name = &node.inputs[0];
            let indices_name = node.inputs.get(1).map(|s| s.as_str()).unwrap_or("");
            if axis_usize == 0 {
                let data_vals = sym_const_i64.get(data_name);
                let idx_concrete = const_i64.get(indices_name);
                if let (Some(dv), Some(iv)) = (data_vals, idx_concrete) {
                    if !dv.is_empty() {
                        let gathered: Vec<SymDim> = iv
                            .iter()
                            .filter_map(|&idx| {
                                let i = if idx < 0 {
                                    (dv.len() as i64 + idx).max(0) as usize
                                } else {
                                    idx as usize
                                };
                                dv.get(i).cloned()
                            })
                            .collect();
                        if !gathered.is_empty() {
                            sym_updates.insert(node.outputs[0].clone(), gathered);
                        }
                    }
                }
                // Also try sym indices.
                let idx_sym_vals = sym_const_i64.get(indices_name);
                if let (Some(dv), Some(iv)) = (data_vals, idx_sym_vals) {
                    // Only extract if ALL indices are concrete.
                    let all_concrete: Option<Vec<usize>> = iv
                        .iter()
                        .map(|d| {
                            if let SymDim::Concrete(v) = d {
                                Some(*v as usize)
                            } else {
                                None
                            }
                        })
                        .collect();
                    if let Some(idxs) = all_concrete {
                        let gathered: Vec<SymDim> =
                            idxs.iter().filter_map(|&i| dv.get(i).cloned()).collect();
                        if !gathered.is_empty() {
                            sym_updates.insert(node.outputs[0].clone(), gathered);
                        }
                    }
                }
            }

            vec![out]
        }

        // ── Shape op ──────────────────────────────────────────────────────────
        "Shape" => {
            // Output is a 1-D tensor with rank(input) elements.
            let rank = match get(0) {
                Some(s) => s.len(),
                None => return vec![],
            };
            let out: SymShape = vec![SymDim::Concrete(rank as u64)];

            // KEY: seed sym_const_i64 with the input's symbolic dims as values.
            // E.g., Shape([1, 12, Var("past"), 64]) → sym_const_i64 entry = [Concrete(1), Concrete(12), Var("past"), Concrete(64)]
            let input_sym = get(0).unwrap();
            sym_updates.insert(node.outputs[0].clone(), input_sym.clone());

            vec![out]
        }

        // ── ConstantOfShape ───────────────────────────────────────────────────
        "ConstantOfShape" => {
            let shape_name = node.inputs.first().map(|s| s.as_str()).unwrap_or("");

            // Try sym_const_i64 first (symbolic shape).
            if let Some(sym_vals) = sym_const_i64.get(shape_name) {
                let out: SymShape = sym_vals
                    .iter()
                    .map(|d| match d {
                        SymDim::Concrete(v) => SymDim::Concrete(*v),
                        other => other.clone(),
                    })
                    .collect();
                return vec![out];
            }

            // Fall back to const_i64 (concrete shape).
            if let Some(vals) = const_i64.get(shape_name) {
                let out: SymShape = vals.iter().map(|&v| SymDim::Concrete(v as u64)).collect();
                return vec![out];
            }

            vec![]
        }

        // ── ReduceMean / ReduceSum / ReduceMax ────────────────────────────────
        "ReduceMean" | "ReduceSum" | "ReduceMax" => {
            let input_sym = match get(0) {
                Some(s) => s,
                None => return vec![],
            };
            let keepdim = get_keepdims_attr(attrs, true);
            let rank = input_sym.len();

            // Get axes.
            let axes: Vec<usize> = if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
                // axes from second input.
                if let Some(vals) = const_i64.get(&node.inputs[1]) {
                    vals.iter()
                        .map(|&a| {
                            let a = if a < 0 {
                                (rank as i64 + a) as usize
                            } else {
                                a as usize
                            };
                            a.min(rank.saturating_sub(1))
                        })
                        .collect()
                } else {
                    return vec![];
                }
            } else if let Some(OnnxAttribute::Ints(v)) = attrs.get("axes") {
                v.iter()
                    .map(|&a| {
                        let a = if a < 0 {
                            (rank as i64 + a) as usize
                        } else {
                            a as usize
                        };
                        a.min(rank.saturating_sub(1))
                    })
                    .collect()
            } else {
                // No axes = reduce all.
                (0..rank).collect()
            };

            let out: SymShape = if keepdim {
                input_sym
                    .iter()
                    .enumerate()
                    .map(|(i, d)| {
                        if axes.contains(&i) {
                            SymDim::Concrete(1)
                        } else {
                            d.clone()
                        }
                    })
                    .collect()
            } else {
                input_sym
                    .iter()
                    .enumerate()
                    .filter_map(|(i, d)| {
                        if axes.contains(&i) {
                            None
                        } else {
                            Some(d.clone())
                        }
                    })
                    .collect()
            };
            vec![out]
        }

        // ── Unsqueeze ─────────────────────────────────────────────────────────
        "Unsqueeze" => {
            let input_sym = match get(0) {
                Some(s) => s,
                None => return vec![],
            };

            // Get axes.
            let axes: Vec<i64> = if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
                if let Some(vals) = const_i64.get(&node.inputs[1]) {
                    vals.clone()
                } else {
                    return vec![];
                }
            } else if let Some(OnnxAttribute::Ints(v)) = attrs.get("axes") {
                v.clone()
            } else {
                return vec![];
            };

            let out_rank = input_sym.len() + axes.len();
            let mut norm_axes: Vec<usize> = axes
                .iter()
                .map(|&a| {
                    if a < 0 {
                        (out_rank as i64 + a) as usize
                    } else {
                        a as usize
                    }
                })
                .collect();
            norm_axes.sort_unstable();

            let mut out: SymShape = Vec::with_capacity(out_rank);
            let mut src_idx = 0usize;
            for i in 0..out_rank {
                if norm_axes.contains(&i) {
                    out.push(SymDim::Concrete(1));
                } else {
                    out.push(
                        input_sym
                            .get(src_idx)
                            .cloned()
                            .unwrap_or(SymDim::Concrete(1)),
                    );
                    src_idx += 1;
                }
            }

            // Propagate sym_const_i64 through Unsqueeze (values unchanged).
            if let Some(vals) = sym_const_i64.get(&node.inputs[0]) {
                sym_updates.insert(node.outputs[0].clone(), vals.clone());
            }

            vec![out]
        }

        // ── Squeeze ───────────────────────────────────────────────────────────
        "Squeeze" => {
            let input_sym = match get(0) {
                Some(s) => s,
                None => return vec![],
            };

            let axes: Vec<i64> = if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
                if let Some(vals) = const_i64.get(&node.inputs[1]) {
                    vals.clone()
                } else {
                    return vec![];
                }
            } else if let Some(OnnxAttribute::Ints(v)) = attrs.get("axes") {
                v.clone()
            } else {
                // Squeeze all size-1 dims.
                return vec![
                    input_sym
                        .iter()
                        .filter(|d| !matches!(d, SymDim::Concrete(1)))
                        .cloned()
                        .collect(),
                ];
            };

            let rank = input_sym.len();
            let norm_axes: Vec<usize> = axes
                .iter()
                .map(|&a| {
                    if a < 0 {
                        (rank as i64 + a) as usize
                    } else {
                        a as usize
                    }
                })
                .collect();

            let out: SymShape = input_sym
                .iter()
                .enumerate()
                .filter_map(|(i, d)| {
                    if norm_axes.contains(&i) {
                        None
                    } else {
                        Some(d.clone())
                    }
                })
                .collect();

            // Propagate sym_const_i64 through Squeeze (values unchanged).
            if let Some(vals) = sym_const_i64.get(&node.inputs[0]) {
                sym_updates.insert(node.outputs[0].clone(), vals.clone());
            }

            vec![out]
        }

        // ── Slice ─────────────────────────────────────────────────────────────
        "Slice" => {
            let data_sym = match get(0) {
                Some(s) => s,
                None => return vec![],
            };
            let rank = data_sym.len();

            let starts_static = const_i64.get(node.inputs.get(1).map(|s| s.as_str()).unwrap_or(""));
            let ends_static = const_i64.get(node.inputs.get(2).map(|s| s.as_str()).unwrap_or(""));

            let result: SymShape = if let (Some(starts), Some(ends)) = (starts_static, ends_static)
            {
                let axes: Vec<usize> = if node.inputs.len() > 3 && !node.inputs[3].is_empty() {
                    if let Some(vals) = const_i64.get(&node.inputs[3]) {
                        vals.iter()
                            .map(|&a| {
                                if a < 0 {
                                    (rank as i64 + a) as usize
                                } else {
                                    a as usize
                                }
                            })
                            .collect()
                    } else {
                        (0..starts.len()).collect()
                    }
                } else {
                    (0..starts.len()).collect()
                };

                let steps: Vec<i64> = if node.inputs.len() > 4 && !node.inputs[4].is_empty() {
                    if let Some(vals) = const_i64.get(&node.inputs[4]) {
                        vals.clone()
                    } else {
                        vec![1i64; starts.len()]
                    }
                } else {
                    vec![1i64; starts.len()]
                };

                let mut out = data_sym.clone();
                for ((&start, &end), (&axis, &step)) in starts
                    .iter()
                    .zip(ends.iter())
                    .zip(axes.iter().zip(steps.iter()))
                {
                    if axis >= rank {
                        continue;
                    }
                    // Try to compute concrete output dim for this axis.
                    if let SymDim::Concrete(dim_val) = &data_sym[axis] {
                        let n = *dim_val as i64;
                        let s = if start < 0 {
                            (n + start).max(0)
                        } else {
                            start.min(n)
                        };
                        let e = if end < 0 {
                            (n + end).max(0)
                        } else {
                            end.min(n)
                        };
                        let size = ((e - s) as f64 / step.abs() as f64).ceil().max(0.0) as u64;
                        out[axis] = SymDim::Concrete(size);
                    } else {
                        // Dynamic dim sliced: keep as-is (can't determine output size symbolically).
                        // out[axis] unchanged.
                    }
                }
                out
            } else {
                // Dynamic slice — keep input shape as-is (best-effort).
                data_sym.clone()
            };

            // Propagate sym_const_i64 through Slice on 1-D data.
            if let Some(data_vals) = sym_const_i64.get(&node.inputs[0]) {
                if let (Some(starts), Some(ends)) = (
                    const_i64.get(node.inputs.get(1).map(|s| s.as_str()).unwrap_or("")),
                    const_i64.get(node.inputs.get(2).map(|s| s.as_str()).unwrap_or("")),
                ) {
                    if starts.len() == 1 && ends.len() == 1 {
                        let n = data_vals.len() as i64;
                        let s = if starts[0] < 0 {
                            (n + starts[0]).max(0) as usize
                        } else {
                            (starts[0] as usize).min(data_vals.len())
                        };
                        let e = if ends[0] < 0 {
                            (n + ends[0]).max(0) as usize
                        } else {
                            (ends[0] as usize).min(data_vals.len())
                        };
                        let step = const_i64
                            .get(node.inputs.get(4).map(|s| s.as_str()).unwrap_or(""))
                            .and_then(|v| v.first().copied())
                            .unwrap_or(1);
                        if step == 1 && s <= e {
                            sym_updates.insert(node.outputs[0].clone(), data_vals[s..e].to_vec());
                        }
                    }
                }
            }

            vec![result]
        }

        // ── Split ─────────────────────────────────────────────────────────────
        "Split" => {
            let input_sym = match get(0) {
                Some(s) => s,
                None => return vec![],
            };
            let axis = get_int_attr(attrs, "axis", 0);
            let axis_usize = normalize_axis(axis, input_sym.len());
            let num_outputs = node.outputs.len();

            let split_sizes: Vec<SymDim> = if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
                if let Some(vals) = const_i64.get(&node.inputs[1]) {
                    vals.iter().map(|&v| SymDim::Concrete(v as u64)).collect()
                } else if let Some(sym_vals) = sym_const_i64.get(&node.inputs[1]) {
                    sym_vals.clone()
                } else {
                    return vec![];
                }
            } else if let Some(OnnxAttribute::Ints(v)) = attrs.get("split") {
                v.iter().map(|&v| SymDim::Concrete(v as u64)).collect()
            } else {
                // Equal split.
                let ax_dim = &input_sym[axis_usize];
                match ax_dim {
                    SymDim::Concrete(d) => {
                        let each = d / num_outputs as u64;
                        vec![SymDim::Concrete(each); num_outputs]
                    }
                    _ => return vec![],
                }
            };

            split_sizes
                .into_iter()
                .map(|size| {
                    let mut out = input_sym.clone();
                    if axis_usize < out.len() {
                        out[axis_usize] = size;
                    }
                    out
                })
                .collect()
        }

        // ── Flatten ───────────────────────────────────────────────────────────
        "Flatten" => {
            let input_sym = match get(0) {
                Some(s) => s,
                None => return vec![],
            };
            let axis = get_int_attr(attrs, "axis", 1) as usize;
            let rank = input_sym.len();
            let axis = axis.min(rank);

            // Outer product.
            let outer: SymDim = if axis == 0 {
                SymDim::Concrete(1)
            } else {
                input_sym[..axis]
                    .iter()
                    .fold(SymDim::Concrete(1), |acc, d| acc.mul(d.clone()).simplify())
            };
            // Inner product.
            let inner: SymDim = if axis >= rank {
                SymDim::Concrete(1)
            } else {
                input_sym[axis..]
                    .iter()
                    .fold(SymDim::Concrete(1), |acc, d| acc.mul(d.clone()).simplify())
            };

            vec![vec![outer, inner]]
        }

        // ── Constant ──────────────────────────────────────────────────────────
        "Constant" => {
            if let Some(OnnxAttribute::Tensor(buf)) = attrs.get("value") {
                let out: SymShape = buf.shape().0.iter().map(|&d| SymDim::Concrete(d)).collect();

                // Seed sym_const_i64 for small non-negative integer constants.
                // Negative values are ONNX sentinels (-1 = infer, etc.); casting to u64 overflows.
                if buf.shape().num_elements() <= 64 {
                    if let Ok(vals) = read_i64_buf(buf) {
                        if vals.iter().all(|&v| v >= 0) {
                            sym_updates.insert(
                                node.outputs[0].clone(),
                                vals.iter().map(|&v| SymDim::Concrete(v as u64)).collect(),
                            );
                        }
                    }
                }
                vec![out]
            } else {
                vec![]
            }
        }

        // ── Range ─────────────────────────────────────────────────────────────
        "Range" => {
            // Try to compute output size statically.
            let s = const_i64.get(node.inputs.first().map(|s| s.as_str()).unwrap_or(""));
            let l = const_i64.get(node.inputs.get(1).map(|s| s.as_str()).unwrap_or(""));
            let d = const_i64.get(node.inputs.get(2).map(|s| s.as_str()).unwrap_or(""));
            match (s, l, d) {
                (Some(sv), Some(lv), Some(dv))
                    if !sv.is_empty() && !lv.is_empty() && !dv.is_empty() && dv[0] != 0 =>
                {
                    let size = ((lv[0] - sv[0] + dv[0] - 1) / dv[0]).max(0) as u64;
                    vec![vec![SymDim::Concrete(size)]]
                }
                _ => vec![vec![]], // unknown size
            }
        }

        // ── Conv (output shape is concrete — computed by MLIR) ─────────────────
        "Conv" | "MaxPool" | "GlobalAveragePool" | "AveragePool" => {
            // These ops produce shapes MLIR computes. Return empty to fall back to MLIR.
            vec![]
        }

        "BatchNormalization" => {
            // Output shape = input shape.
            if let Some(s) = get(0) {
                vec![s.clone()]
            } else {
                vec![]
            }
        }

        "Clip" => {
            if let Some(s) = get(0) {
                vec![s.clone()]
            } else {
                vec![]
            }
        }

        // ── Fallback ──────────────────────────────────────────────────────────
        _ => vec![],
    }
}

// ── MatMul shape helpers ──────────────────────────────────────────────────────

/// Compute symbolic output shape for MatMul.
/// - 2D: [M, K] x [K, N] → [M, N]
/// - 3D: [B, M, K] x [B, K, N] → [B, M, N]
/// - Mixed: [B, M, K] x [K, N] → [B, M, N]
fn matmul_sym_shape(a: &SymShape, b: &SymShape) -> SymShape {
    match (a.len(), b.len()) {
        (2, 2) => vec![a[0].clone(), b[1].clone()],
        (3, 3) => vec![a[0].clone(), a[1].clone(), b[2].clone()],
        (3, 2) => vec![a[0].clone(), a[1].clone(), b[1].clone()],
        (4, 4) => vec![a[0].clone(), a[1].clone(), a[2].clone(), b[3].clone()],
        (4, 2) => vec![a[0].clone(), a[1].clone(), a[2].clone(), b[1].clone()],
        // General: copy all batch dims from a, take last dim of b.
        _ if a.len() >= 2 && !b.is_empty() => {
            let mut out = a[..a.len() - 1].to_vec();
            out.push(b[b.len() - 1].clone());
            out
        }
        _ => vec![],
    }
}

// ── sym_const_i64 elementwise helper ─────────────────────────────────────────

/// Apply an element-wise function to two sym_const_i64 vectors.
/// Supports scalar broadcast. Returns None if sizes are incompatible.
fn sym_const_i64_elementwise(
    a: &[SymDim],
    b: &[SymDim],
    op: impl Fn(SymDim, SymDim) -> SymDim,
) -> Option<Vec<SymDim>> {
    let apply = |x: SymDim, y: SymDim| op(x, y).simplify();
    if a.len() == b.len() {
        Some(
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| apply(x.clone(), y.clone()))
                .collect(),
        )
    } else if a.len() == 1 {
        Some(b.iter().map(|y| apply(a[0].clone(), y.clone())).collect())
    } else if b.len() == 1 {
        Some(a.iter().map(|x| apply(x.clone(), b[0].clone())).collect())
    } else {
        None
    }
}

// ── Reshape helpers ───────────────────────────────────────────────────────────

/// Compute symbolic output shape for Reshape with a concrete target shape (all-concrete values).
fn reshape_sym_with_concrete_target(
    input_sym: Option<&SymShape>,
    target: &[i64],
    _allowzero: bool,
) -> SymShape {
    // Compute total input elements (symbolic).
    let total_sym: Option<SymDim> = input_sym.map(|s| {
        s.iter()
            .fold(SymDim::Concrete(1), |acc, d| acc.mul(d.clone()).simplify())
    });

    // Compute known product of non-(-1) target dims.
    let known_product: Option<u64> = target.iter().try_fold(1u64, |acc, &d| {
        if d == -1 {
            Some(acc)
        }
        // skip -1 in product
        else if d > 0 {
            Some(acc * d as u64)
        } else {
            None
        }
    });

    target
        .iter()
        .map(|&d| {
            if d > 0 {
                SymDim::Concrete(d as u64)
            } else if d == -1 {
                // Infer: total / known_product
                match (total_sym.clone(), known_product) {
                    (Some(total), Some(kp)) if kp > 0 => total.div(SymDim::Concrete(kp)).simplify(),
                    _ => SymDim::Concrete(0), // can't infer
                }
            } else {
                // d == 0: copy from input (allowzero=0 case)
                SymDim::Concrete(0)
            }
        })
        .collect()
}

/// Compute symbolic output shape for Reshape with a symbolic target shape.
fn reshape_sym_with_sym_target(
    input_sym: Option<&SymShape>,
    sym_target: &[SymDim],
    _allowzero: bool,
) -> SymShape {
    // Compute total input elements (symbolic).
    let total_sym: Option<SymDim> = input_sym.map(|s| {
        s.iter()
            .fold(SymDim::Concrete(1), |acc, d| acc.mul(d.clone()).simplify())
    });

    // Compute known product of non-(-1) target dims.
    // In sym_const_i64, -1 means "infer" — stored as u64::MAX mapped through the chain.
    // Actually sym_const_i64 stores SymDim::Concrete(u64::MAX as u64) or Var for -1.
    // We look for concrete values that aren't u64::MAX.
    let known_product: SymDim = sym_target.iter().fold(SymDim::Concrete(1), |acc, d| {
        match d {
            // Skip the -1 inferred dim (check for values that look like -1 in i64 → u64::MAX).
            SymDim::Concrete(v) if *v == u64::MAX => acc, // skip inferred dim
            other => acc.mul(other.clone()).simplify(),
        }
    });

    sym_target
        .iter()
        .map(|d| {
            match d {
                SymDim::Concrete(v) if *v == u64::MAX => {
                    // This is the -1 (infer) dimension.
                    if let Some(total) = total_sym.clone() {
                        total.div(known_product.clone()).simplify()
                    } else {
                        SymDim::Concrete(0)
                    }
                }
                other => other.clone(),
            }
        })
        .collect()
}

// ── Buffer reading helper ─────────────────────────────────────────────────────

fn read_i64_buf(buf: &crate::runtime::Buffer) -> Result<Vec<i64>, ()> {
    use crate::DType;
    match buf.dtype() {
        DType::I64 => Ok(buf.as_slice::<i64>().to_vec()),
        DType::I32 => Ok(buf.as_slice::<i32>().iter().map(|&v| v as i64).collect()),
        _ => Err(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn concrete(v: u64) -> SymDim {
        SymDim::Concrete(v)
    }
    fn var(s: &str) -> SymDim {
        SymDim::var(s)
    }

    #[test]
    fn broadcast_sym_shapes_test() {
        // [B, 1, seq] broadcast with [1, 12, seq] = [B, 12, seq]
        let a: SymShape = vec![var("B"), concrete(1), var("seq")];
        let b: SymShape = vec![concrete(1), concrete(12), var("seq")];
        let out = broadcast_sym_shapes(&a, &b);
        assert_eq!(out, vec![var("B"), concrete(12), var("seq")]);
    }

    #[test]
    fn matmul_2d() {
        let a: SymShape = vec![var("M"), concrete(64)];
        let b: SymShape = vec![concrete(64), concrete(768)];
        let out = matmul_sym_shape(&a, &b);
        assert_eq!(out, vec![var("M"), concrete(768)]);
    }

    #[test]
    fn matmul_3d() {
        let a: SymShape = vec![concrete(1), var("seq"), concrete(64)];
        let b: SymShape = vec![concrete(1), concrete(64), var("past")];
        let out = matmul_sym_shape(&a, &b);
        assert_eq!(out, vec![concrete(1), var("seq"), var("past")]);
    }

    #[test]
    fn transpose_sym() {
        let mut attrs = HashMap::new();
        attrs.insert("perm".to_string(), OnnxAttribute::Ints(vec![0, 2, 1, 3]));
        let node = OnnxNode {
            op_type: "Transpose".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            attributes: attrs,
        };
        let shape: SymShape = vec![concrete(1), concrete(12), var("seq"), concrete(64)];
        let (outs, _) = propagate_sym_shapes(&node, &[&shape], &HashMap::new(), &HashMap::new());
        assert_eq!(
            outs[0],
            vec![concrete(1), var("seq"), concrete(12), concrete(64)]
        );
    }

    #[test]
    fn shape_op_seeds_sym_const() {
        let node = OnnxNode {
            op_type: "Shape".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["shape_out".to_string()],
            attributes: HashMap::new(),
        };
        let shape: SymShape = vec![concrete(1), concrete(12), var("past"), concrete(64)];
        let (outs, updates) =
            propagate_sym_shapes(&node, &[&shape], &HashMap::new(), &HashMap::new());
        assert_eq!(outs[0], vec![concrete(4)]);
        assert_eq!(updates.get("shape_out").unwrap(), &shape);
    }

    #[test]
    fn concat_sums_axis_dim() {
        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), OnnxAttribute::Int(2));
        let node = OnnxNode {
            op_type: "Concat".to_string(),
            inputs: vec!["a".to_string(), "b".to_string()],
            outputs: vec!["c".to_string()],
            attributes: attrs,
        };
        let a: SymShape = vec![concrete(1), concrete(12), var("past"), concrete(64)];
        let b: SymShape = vec![concrete(1), concrete(12), concrete(1), concrete(64)];
        let (outs, _) = propagate_sym_shapes(&node, &[&a, &b], &HashMap::new(), &HashMap::new());
        // axis=2: past + 1
        let expected_dim = var("past").add(concrete(1)).simplify();
        assert_eq!(outs[0][2], expected_dim);
    }

    #[test]
    fn reshape_sym_concrete_target() {
        let mut attrs = HashMap::new();
        attrs.insert("allowzero".to_string(), OnnxAttribute::Int(0));
        let node = OnnxNode {
            op_type: "Reshape".to_string(),
            inputs: vec!["x".to_string(), "shape".to_string()],
            outputs: vec!["y".to_string()],
            attributes: attrs,
        };
        let input_sym: SymShape = vec![concrete(1), concrete(12), var("seq"), concrete(64)];
        let mut const_i64 = HashMap::new();
        const_i64.insert("shape".to_string(), vec![1i64, -1, 768]);
        let (outs, _) = propagate_sym_shapes(&node, &[&input_sym], &HashMap::new(), &const_i64);
        // -1 dim = total / known_product = (1*12*seq*64) / (1*768) = Div(Mul(Mul(12, seq), 64), 768)
        // simplify() doesn't fold 12*64/768=1, but the expression is semantically correct.
        assert_eq!(outs[0].len(), 3, "output should have 3 dims");
        assert_eq!(outs[0][0], concrete(1), "first dim should be 1");
        assert_eq!(outs[0][2], concrete(768), "last dim should be 768");
        // dim[1] should be some expression involving seq — evaluate it to check.
        let dim1 = &outs[0][1];
        let mut bindings = HashMap::new();
        bindings.insert("seq".to_string(), 10u64);
        assert_eq!(
            dim1.eval(&bindings),
            Some(10),
            "dim[1] should eval to seq=10, got {dim1:?}"
        );
    }
}
