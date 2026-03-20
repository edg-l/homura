/// Emit MLIR text for quantized (Q8_0) dequant-matmul kernels.
///
/// These kernels operate directly on memref arguments (no tensor semantics),
/// so they bypass the full linalg/bufferize pipeline and use a dedicated
/// quant pipeline instead.

/// Emit a complete MLIR module for a Q8_0 dequant-matmul kernel.
///
/// Computes: `out[0, m, n] = sum_k(act[0, m, k] * dequant(weight_block[n, k/32]))`
///
/// # Arguments
/// - `k`: input features (K dimension, must be a multiple of 32)
/// - `n`: output features (N dimension)
/// - `func_name`: MLIR function name (e.g. `"k42"`)
/// - `n_tile`: forall tile size for N-dimension parallelism
///
/// # Weight layout
/// Flat `memref<?xi8>` with `total_blocks * 34` bytes.
/// Each Q8_0 block is 34 bytes: 2-byte little-endian f16 scale + 32 x i8 quants.
/// Blocks are laid out row-major `[N, K/32]`: block for row `n`, chunk `kb` is at
/// byte offset `(n * num_blocks_per_row + kb) * 34`.
///
/// # Activation / output
/// - Activation: `memref<1x?x{k}xf32>` (batch=1, seq=dynamic, features=K)
/// - Output:     `memref<1x?x{n}xf32>`
///
/// # Parallelism
/// Uses `scf.forall` over N tiles → `convert-scf-to-openmp` → OpenMP threads.
/// The `n_tile` parameter controls how many N columns each thread processes.
pub fn emit_dequant_matmul_q8_0(k: u64, n: u64, func_name: &str, n_tile: usize) -> String {
    assert!(k % 32 == 0, "k must be a multiple of 32, got {k}");
    assert!(
        n % n_tile as u64 == 0,
        "n ({n}) must be divisible by n_tile ({n_tile})"
    );

    let num_blocks_per_row: u64 = k / 32;
    let total_blocks: u64 = n * num_blocks_per_row;
    let total_weight_bytes: u64 = total_blocks * 34;
    let n_tiles: u64 = n / n_tile as u64;

    // We generate one fixed SSA name per loop variable and reuse them.
    // All arithmetic uses separate SSA values to keep the IR valid.
    format!(
        r#"// Q8_0 dequant-matmul: K={k} N={n} n_tile={n_tile}
// Weight layout: {total_weight_bytes} bytes ({total_blocks} blocks x 34 bytes/block)
module attributes {{"homura.quant_kernel"}} {{
  func.func @{func_name}(
      %act    : memref<1x?x{k}xf32>,
      %weight : memref<{total_weight_bytes}xi8>,
      %out    : memref<1x?x{n}xf32>)
      attributes {{llvm.emit_c_interface}} {{

    %c0        = arith.constant 0 : index
    %c1        = arith.constant 1 : index
    %c2        = arith.constant 2 : index
    %c8_i16    = arith.constant 8 : i16
    %c32       = arith.constant 32 : index
    %c34       = arith.constant 34 : index
    %c0_f32    = arith.constant 0.000000e+00 : f32
    %nbpr      = arith.constant {num_blocks_per_row} : index
    %n_tiles_c = arith.constant {n_tiles} : index
    %tile_sz   = arith.constant {n_tile} : index

    // Dynamic sequence length (M dimension).
    %seq = memref.dim %act, %c1 : memref<1x?x{k}xf32>

    // Parallel over N tiles.
    scf.forall (%tile_idx) in (%n_tiles_c) {{
      // Base N index for this tile.
      %n_base = arith.muli %tile_idx, %tile_sz : index

      // Sequential over M (sequence positions).
      scf.for %m = %c0 to %seq step %c1 {{

        // Sequential over N within this tile.
        scf.for %n_local = %c0 to %tile_sz step %c1 {{
          %n_idx = arith.addi %n_base, %n_local : index

          // Accumulate dot-product over K blocks.
          %acc = scf.for %kb = %c0 to %nbpr step %c1
                     iter_args(%acc_in = %c0_f32) -> (f32) {{

            // Flat block index: n_idx * num_blocks_per_row + kb
            %blk_n   = arith.muli %n_idx, %nbpr : index
            %blk_idx = arith.addi %blk_n, %kb   : index

            // Byte offset of this block in the flat weight buffer: blk_idx * 34
            %blk_off = arith.muli %blk_idx, %c34 : index

            // Decode little-endian f16 scale from bytes [0] and [1] of the block.
            %s0_i8  = memref.load %weight[%blk_off] : memref<{total_weight_bytes}xi8>
            %s1_raw = arith.addi %blk_off, %c1 : index
            %s1_i8  = memref.load %weight[%s1_raw] : memref<{total_weight_bytes}xi8>
            %s0_i16 = arith.extui %s0_i8 : i8 to i16
            %s1_i16 = arith.extui %s1_i8 : i8 to i16
            %s1_sh  = arith.shli  %s1_i16, %c8_i16 : i16
            %sc_i16 = arith.ori   %s0_i16, %s1_sh  : i16
            %sc_f16 = arith.bitcast %sc_i16 : i16 to f16
            %sc_f32 = arith.extf  %sc_f16 : f16 to f32
            %sc_vec = vector.splat %sc_f32 : vector<32xf32>

            // Vector-load 32 quant bytes starting at offset 2 within the block.
            %q_off = arith.addi %blk_off, %c2 : index
            %q_i8  = vector.load %weight[%q_off]
                         : memref<{total_weight_bytes}xi8>, vector<32xi8>

            // Dequantize: i8 -> i32 -> f32, then multiply by scale.
            %q_i32 = arith.extsi  %q_i8  : vector<32xi8>  to vector<32xi32>
            %q_f32 = arith.sitofp %q_i32 : vector<32xi32> to vector<32xf32>
            %dq    = arith.mulf   %q_f32, %sc_vec : vector<32xf32>

            // Load 32 activation elements for the current (m, kb) position.
            %k_base = arith.muli %kb, %c32 : index
            %av     = vector.load %act[%c0, %m, %k_base]
                          : memref<1x?x{k}xf32>, vector<32xf32>

            // Dot product via element-wise mul + horizontal add.
            %prod    = arith.mulf %dq, %av : vector<32xf32>
            %dot     = vector.reduction <add>, %prod : vector<32xf32> into f32
            %new_acc = arith.addf %acc_in, %dot : f32
            scf.yield %new_acc : f32
          }}

          // Store accumulated result.
          memref.store %acc, %out[%c0, %m, %n_idx] : memref<1x?x{n}xf32>
        }}
      }}
    }}
    return
  }}
}}
"#,
        k = k,
        n = n,
        func_name = func_name,
        n_tile = n_tile,
        num_blocks_per_row = num_blocks_per_row,
        total_weight_bytes = total_weight_bytes,
        n_tiles = n_tiles,
    )
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DType,
        compiler::link_shared_lib_pub,
        graph_builder::{compile_to_objects, create_context_pub, tempfile_dir},
        runtime::{Buffer, CompiledGraph, OutputDesc},
    };

    /// Parse the emitted MLIR and verify it compiles through the quant pipeline.
    #[test]
    fn emit_q8_0_parse_and_compile() {
        let mlir = emit_dequant_matmul_q8_0(64, 64, "k0", 16);

        // 1. Verify the MLIR text is parseable.
        let context = create_context_pub();
        let _module =
            melior::ir::Module::parse(&context, &mlir).expect("Q8_0 MLIR failed to parse");

        // 2. Compile through the quant pipeline to .o files.
        let tmp = tempfile_dir().unwrap();
        let obj_paths = compile_to_objects(&mlir, "test_q8_0", "k0", &tmp)
            .expect("Q8_0 compile_to_objects failed");
        assert!(
            !obj_paths.is_empty(),
            "expected at least one .o file from Q8_0 kernel"
        );
        for p in &obj_paths {
            assert!(p.exists(), "object file not found: {}", p.display());
        }
    }

    /// Correctness check: build a tiny Q8_0 kernel, run it, verify the output.
    ///
    /// Weight layout: scale = 1.0f16, quants[i] = (i+1) for i in 0..32.
    /// Activation: all 1.0f32.
    /// Expected output per neuron: dot = 1+2+…+32 = 528.
    #[test]
    fn emit_q8_0_correctness() {
        const K: u64 = 32;
        const N: u64 = 4;
        const N_TILE: usize = 2;
        const SEQ: u64 = 1;

        let mlir = emit_dequant_matmul_q8_0(K, N, "kq_corr", N_TILE);

        let tmp = tempfile_dir().unwrap();
        let obj_paths =
            compile_to_objects(&mlir, "corr_q8_0", "kq_corr", &tmp).expect("compile failed");

        let so_path = tmp.join("homura_test_q8_0_corr.so");
        link_shared_lib_pub(&obj_paths, &so_path).expect("link failed");

        // Output descriptor: memref<1 x SEQ x N xf32>
        let out_desc = OutputDesc {
            shape: crate::shape::Shape(vec![1, SEQ, N]),
            dtype: DType::F32,
        };
        let graph = CompiledGraph::load_named(&so_path, 2, vec![out_desc], "kq_corr")
            .expect("dlopen failed");

        // Build the flat weight buffer.
        // total_weight_bytes = N * (K/32) * 34 = 4 * 1 * 34 = 136 bytes
        let num_blocks_per_row = (K / 32) as usize;
        let total_blocks = N as usize * num_blocks_per_row;
        let total_bytes = total_blocks * 34;
        let mut weight_bytes = vec![0u8; total_bytes];

        // 1.0 in f16 = 0x3C00 (little-endian: lo=0x00, hi=0x3C)
        let scale_bits: u16 = 0x3C00;
        let scale_lo = (scale_bits & 0xFF) as u8;
        let scale_hi = (scale_bits >> 8) as u8;
        for b in 0..total_blocks {
            let off = b * 34;
            weight_bytes[off] = scale_lo;
            weight_bytes[off + 1] = scale_hi;
            for qi in 0..32usize {
                // quant value (qi+1): fits in i8, store as two's-complement u8
                weight_bytes[off + 2 + qi] = (qi + 1) as u8;
            }
        }

        // Activation: 1 x SEQ x K, all 1.0
        let act_data = vec![1.0f32; K as usize * SEQ as usize];
        let act_buf = Buffer::from_slice(&act_data, &[1, SEQ, K], DType::F32);
        let weight_buf = Buffer::from_slice(&weight_bytes, &[total_bytes as u64], DType::I8);

        let outputs = graph.run(&[&act_buf, &weight_buf]);
        assert_eq!(outputs.len(), 1);

        let out_slice = outputs[0].as_slice::<f32>();
        assert_eq!(out_slice.len(), N as usize);

        // Each neuron accumulates: sum_{i=1}^{32} i = 528
        let expected: f32 = (1u32..=32).sum::<u32>() as f32;
        for (i, &v) in out_slice.iter().enumerate() {
            let diff = (v - expected).abs();
            assert!(
                diff < 1.0,
                "out[{i}] = {v}, expected ~{expected} (diff = {diff})"
            );
        }
    }
}
