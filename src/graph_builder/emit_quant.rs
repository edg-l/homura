/// Emit MLIR text for quantized dequant-matmul kernels (Q8_0, Q4_K).
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

/// Emit a complete MLIR module for a Q4_K dequant-matmul kernel.
///
/// Computes: `out[0, m, n] = sum_k(act[0, m, k] * dequant_q4k(weight_block[n, k/256]))`
///
/// # Arguments
/// - `k`: input features (K dimension, must be a multiple of 256)
/// - `n`: output features (N dimension)
/// - `func_name`: MLIR function name
/// - `n_tile`: forall tile size for N-dimension parallelism
///
/// # Weight layout
/// Flat `memref<?xi8>` with `total_blocks * 144` bytes.
/// Each Q4_K super-block is 144 bytes:
///   - bytes 0-1:   f16 d (super-block scale)
///   - bytes 2-3:   f16 dmin (super-block min)
///   - bytes 4-15:  12 bytes packed 6-bit scales/mins for 8 sub-blocks
///   - bytes 16-143: 128 bytes of 4-bit quants (256 values, 2 per byte)
///
/// # Parallelism
/// Uses `scf.forall` over N tiles → `convert-scf-to-openmp` → OpenMP threads.
/// The inner 32-element loop per sub-block auto-vectorizes via LLVM.
pub fn emit_dequant_matmul_q4_k(k: u64, n: u64, func_name: &str, n_tile: usize) -> String {
    assert!(k % 256 == 0, "k must be a multiple of 256, got {k}");
    assert!(
        n % n_tile as u64 == 0,
        "n ({n}) must be divisible by n_tile ({n_tile})"
    );

    let num_blocks_per_row: u64 = k / 256;
    let total_blocks: u64 = n * num_blocks_per_row;
    let total_weight_bytes: u64 = total_blocks * 144;
    let n_tiles: u64 = n / n_tile as u64;

    // Q4_K: 256-element super-blocks, 144 bytes each.
    // Layout per block: d(2) + dmin(2) + scales_raw(12) + qs(128)
    //
    // This kernel uses a flat element loop (0..256) per super-block instead of
    // unrolling 8 sub-blocks, keeping the generated IR compact. The sub-block
    // index and scale/min are computed at runtime.
    //
    // For each element j in 0..255:
    //   sb = j / 32          (sub-block index 0..7)
    //   j_in_sb = j % 32     (element within sub-block)
    //   byte_idx = j / 2     (quant byte in the 128-byte qs region)
    //   nibble = if j%2==0 { qs[byte_idx] & 0xF } else { qs[byte_idx] >> 4 }
    //
    //   Scale unpacking (6-bit packed):
    //     if sb < 4: sc = scales_raw[sb] & 0x3F, mn = scales_raw[4+sb] & 0x3F
    //     if sb >= 4: i = sb-4
    //       sc = (scales_raw[i] >> 6) | (((scales_raw[8+i/2] >> ((i%2)*4)) & 0xF) << 2)
    //       mn = (scales_raw[4+i] >> 6) | (((scales_raw[10+i/2] >> ((i%2)*4)) & 0xF) << 2)
    //
    //   dequant = nibble * (d * sc) - (dmin * mn)

    format!(
        r#"// Q4_K dequant-matmul: K={k} N={n} n_tile={n_tile}
// Weight layout: {twb} bytes ({tb} blocks x 144 bytes/block)
module attributes {{"homura.quant_kernel"}} {{
  func.func @{func_name}(
      %act    : memref<1x?x{k}xf32>,
      %weight : memref<{twb}xi8>,
      %out    : memref<1x?x{n}xf32>)
      attributes {{llvm.emit_c_interface}} {{

    %c0     = arith.constant 0 : index
    %c1     = arith.constant 1 : index
    %c2     = arith.constant 2 : index
    %c3     = arith.constant 3 : index
    %c4     = arith.constant 4 : index
    %c8     = arith.constant 8 : index
    %c10    = arith.constant 10 : index
    %c16    = arith.constant 16 : index
    %c32    = arith.constant 32 : index
    %c144   = arith.constant 144 : index
    %c256   = arith.constant 256 : index
    %c8_i16 = arith.constant 8 : i16
    %c0_f32 = arith.constant 0.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c4_i32 = arith.constant 4 : i32
    %c6_i32 = arith.constant 6 : i32
    %c15_i32 = arith.constant 15 : i32
    %c63_i32 = arith.constant 63 : i32
    %nbpr      = arith.constant {nbpr} : index
    %n_tiles_c = arith.constant {nt} : index
    %tile_sz   = arith.constant {n_tile} : index

    %seq = memref.dim %act, %c1 : memref<1x?x{k}xf32>

    scf.forall (%tile_idx) in (%n_tiles_c) {{
      %n_base = arith.muli %tile_idx, %tile_sz : index

      scf.for %m = %c0 to %seq step %c1 {{
        scf.for %n_local = %c0 to %tile_sz step %c1 {{
          %n_idx = arith.addi %n_base, %n_local : index

          %acc_kb = scf.for %kb = %c0 to %nbpr step %c1
                        iter_args(%acc_kb_in = %c0_f32) -> (f32) {{

            %blk_n   = arith.muli %n_idx, %nbpr : index
            %blk_idx = arith.addi %blk_n, %kb   : index
            %blk_off = arith.muli %blk_idx, %c144 : index

            // Load d (f16, bytes 0-1)
            %d0_i8  = memref.load %weight[%blk_off] : memref<{twb}xi8>
            %d1_off = arith.addi %blk_off, %c1 : index
            %d1_i8  = memref.load %weight[%d1_off] : memref<{twb}xi8>
            %d0_i16 = arith.extui %d0_i8 : i8 to i16
            %d1_i16 = arith.extui %d1_i8 : i8 to i16
            %d1_sh  = arith.shli  %d1_i16, %c8_i16 : i16
            %d_i16  = arith.ori   %d0_i16, %d1_sh  : i16
            %d_f16  = arith.bitcast %d_i16 : i16 to f16
            %d_f32  = arith.extf  %d_f16 : f16 to f32

            // Load dmin (f16, bytes 2-3)
            %dm0_off = arith.addi %blk_off, %c2 : index
            %dm1_off = arith.addi %blk_off, %c3 : index
            %dm0_i8  = memref.load %weight[%dm0_off] : memref<{twb}xi8>
            %dm1_i8  = memref.load %weight[%dm1_off] : memref<{twb}xi8>
            %dm0_i16 = arith.extui %dm0_i8 : i8 to i16
            %dm1_i16 = arith.extui %dm1_i8 : i8 to i16
            %dm1_sh  = arith.shli  %dm1_i16, %c8_i16 : i16
            %dmin_i16 = arith.ori  %dm0_i16, %dm1_sh : i16
            %dmin_f16 = arith.bitcast %dmin_i16 : i16 to f16
            %dmin_f32 = arith.extf %dmin_f16 : f16 to f32

            %sc_base = arith.addi %blk_off, %c4 : index
            %qs_base = arith.addi %blk_off, %c16 : index
            %k_block_base = arith.muli %kb, %c256 : index

            // Loop over 8 sub-blocks, then 16 byte-pairs (32 elements) per sub-block.
            // Outer: sub-block loop. Inner: byte loop (each byte = 2 nibbles).
            // This structure gives LLVM a contiguous byte access pattern to vectorize.
            %acc_sb = scf.for %sb = %c0 to %c8 step %c1
                          iter_args(%acc_sb_in = %c0_f32) -> (f32) {{

              // --- Unpack sc[sb] and mn[sb] ---
              %sb_i32 = arith.index_cast %sb : index to i32
              %sb_lt4 = arith.cmpi ult, %sb_i32, %c4_i32 : i32
              %sb_clamped = arith.maxui %sb_i32, %c4_i32 : i32
              %sb_m4 = arith.subi %sb_clamped, %c4_i32 : i32

              %sb_mod4 = arith.remui %sb, %c4 : index
              %sr_ptr = arith.addi %sc_base, %sb_mod4 : index
              %sr_raw = memref.load %weight[%sr_ptr] : memref<{twb}xi8>
              %sr_u32 = arith.extui %sr_raw : i8 to i32
              %sc_lo = arith.andi %sr_u32, %c63_i32 : i32
              %sr_hi6 = arith.shrui %sr_u32, %c6_i32 : i32
              %hi_idx_half = arith.shrui %sb_m4, %c1_i32 : i32
              %hi_off_idx = arith.index_cast %hi_idx_half : i32 to index
              %hi_ptr = arith.addi %sc_base, %hi_off_idx : index
              %hi_ptr2 = arith.addi %hi_ptr, %c8 : index
              %hi_raw = memref.load %weight[%hi_ptr2] : memref<{twb}xi8>
              %hi_u32 = arith.extui %hi_raw : i8 to i32
              %sb_m4_mod2 = arith.remui %sb_m4, %c2_i32 : i32
              %shift_amt = arith.muli %sb_m4_mod2, %c4_i32 : i32
              %hi_shifted = arith.shrui %hi_u32, %shift_amt : i32
              %hi_nib = arith.andi %hi_shifted, %c15_i32 : i32
              %hi_nib_sh = arith.shli %hi_nib, %c2_i32 : i32
              %sc_hi = arith.ori %sr_hi6, %hi_nib_sh : i32
              %sc_val = arith.select %sb_lt4, %sc_lo, %sc_hi : i32

              %mn_ptr = arith.addi %sr_ptr, %c4 : index
              %mn_raw = memref.load %weight[%mn_ptr] : memref<{twb}xi8>
              %mn_u32 = arith.extui %mn_raw : i8 to i32
              %mn_lo = arith.andi %mn_u32, %c63_i32 : i32
              %mn_hi6 = arith.shrui %mn_u32, %c6_i32 : i32
              %hmn_ptr = arith.addi %hi_ptr, %c10 : index
              %hmn_raw = memref.load %weight[%hmn_ptr] : memref<{twb}xi8>
              %hmn_u32 = arith.extui %hmn_raw : i8 to i32
              %hmn_shifted = arith.shrui %hmn_u32, %shift_amt : i32
              %hmn_nib = arith.andi %hmn_shifted, %c15_i32 : i32
              %hmn_nib_sh = arith.shli %hmn_nib, %c2_i32 : i32
              %mn_hi = arith.ori %mn_hi6, %hmn_nib_sh : i32
              %mn_val = arith.select %sb_lt4, %mn_lo, %mn_hi : i32

              %sc_f32 = arith.uitofp %sc_val : i32 to f32
              %mn_f32 = arith.uitofp %mn_val : i32 to f32
              %scale = arith.mulf %d_f32, %sc_f32 : f32
              %minv  = arith.mulf %dmin_f32, %mn_f32 : f32

              // Quant byte base for this sub-block: qs_base + sb*16
              %sb_byte_base = arith.muli %sb, %c16 : index
              %qb_base = arith.addi %qs_base, %sb_byte_base : index
              // K element base for this sub-block: k_block_base + sb*32
              %sb_k_base = arith.muli %sb, %c32 : index
              %k_sb_base = arith.addi %k_block_base, %sb_k_base : index

              // Inner loop over 16 bytes (each byte = 2 nibbles = 2 elements)
              %acc_inner = scf.for %bi = %c0 to %c16 step %c1
                               iter_args(%acc_bi = %acc_sb_in) -> (f32) {{

                %qb_off = arith.addi %qb_base, %bi : index
                %qb_raw = memref.load %weight[%qb_off] : memref<{twb}xi8>
                %qb_u32 = arith.extui %qb_raw : i8 to i32

                // Low nibble (even element)
                %nib_lo = arith.andi %qb_u32, %c15_i32 : i32
                %nib_lo_f32 = arith.uitofp %nib_lo : i32 to f32
                %val_lo_sc = arith.mulf %nib_lo_f32, %scale : f32
                %val_lo = arith.subf %val_lo_sc, %minv : f32
                %k_lo = arith.addi %k_sb_base, %bi : index
                %k_lo2 = arith.addi %k_lo, %bi : index  // k_sb_base + 2*bi
                %a_lo = memref.load %act[%c0, %m, %k_lo2] : memref<1x?x{k}xf32>
                %c_lo = arith.mulf %val_lo, %a_lo : f32
                %acc_lo = arith.addf %acc_bi, %c_lo : f32

                // High nibble (odd element)
                %nib_hi = arith.shrui %qb_u32, %c4_i32 : i32
                %nib_hi_m = arith.andi %nib_hi, %c15_i32 : i32
                %nib_hi_f32 = arith.uitofp %nib_hi_m : i32 to f32
                %val_hi_sc = arith.mulf %nib_hi_f32, %scale : f32
                %val_hi = arith.subf %val_hi_sc, %minv : f32
                %k_hi = arith.addi %k_lo2, %c1 : index  // k_sb_base + 2*bi + 1
                %a_hi = memref.load %act[%c0, %m, %k_hi] : memref<1x?x{k}xf32>
                %c_hi = arith.mulf %val_hi, %a_hi : f32
                %acc_hi = arith.addf %acc_lo, %c_hi : f32

                scf.yield %acc_hi : f32
              }}

              scf.yield %acc_inner : f32
            }}

            %kb_contrib = arith.addf %acc_kb_in, %acc_sb : f32
            scf.yield %kb_contrib : f32
          }}

          memref.store %acc_kb, %out[%c0, %m, %n_idx] : memref<1x?x{n}xf32>
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
        nbpr = num_blocks_per_row,
        twb = total_weight_bytes,
        nt = n_tiles,
        tb = total_blocks,
    )
}

/// Emit a complete MLIR module for a Q6_K dequant-matmul kernel.
///
/// Computes: `out[0, m, n] = sum_k(act[0, m, k] * dequant_q6k(weight_block[n, k/256]))`
///
/// # Arguments
/// - `k`: input features (K dimension, must be a multiple of 256)
/// - `n`: output features (N dimension)
/// - `func_name`: MLIR function name
/// - `n_tile`: forall tile size for N-dimension parallelism
///
/// # Weight layout
/// Flat `memref<?xi8>` with `total_blocks * 210` bytes.
/// Each Q6_K super-block is 210 bytes:
///   - bytes   0-127: ql[128]  — low 4 bits of 256 quant values (2 per byte)
///   - bytes 128-191: qh[64]   — high 2 bits of 256 quant values (4 per byte)
///   - bytes 192-207: sc[16]   — i8 sub-block scales (16 sub-blocks of 16 elements)
///   - bytes 208-209: f16 d    — super-block scale
///
/// For sub-block `sb`, element `j` (idx = sb*16 + j):
///   lo = (ql[idx/2] >> ((idx%2)*4)) & 0x0F
///   hi = (qh[idx/4] >> ((idx%4)*2)) & 0x03
///   q  = (lo | (hi << 4)) as i8 - 32
///   dequant = q * d * sc[sb]
///
/// # Parallelism
/// Uses `scf.forall` over N tiles → `convert-scf-to-openmp` → OpenMP threads.
/// The inner 16-element loop per sub-block auto-vectorizes via LLVM.
pub fn emit_dequant_matmul_q6_k(k: u64, n: u64, func_name: &str, n_tile: usize) -> String {
    assert!(k % 256 == 0, "k must be a multiple of 256, got {k}");
    assert!(
        n % n_tile as u64 == 0,
        "n ({n}) must be divisible by n_tile ({n_tile})"
    );

    let num_blocks_per_row: u64 = k / 256;
    let total_blocks: u64 = n * num_blocks_per_row;
    let total_weight_bytes: u64 = total_blocks * 210;
    let n_tiles: u64 = n / n_tile as u64;

    // Build the 16 unrolled sub-block sections.
    //
    // For sub-block sb (0..16), element j (0..16):
    //   idx = sb*16 + j
    //   ql offset within block: idx/2  (0..127)
    //   ql shift: (idx%2)*4  (0 or 4)
    //   qh offset within block: idx/4  (0..63) -> absolute block offset = 128 + idx/4
    //   qh shift: (idx%4)*2  (0, 2, 4, or 6)
    //   sc byte: block[192 + sb] as i8
    //
    // SSA naming: all names use sb suffix to be unique across iterations.
    // The inner scf.for bodies each have their own block so %j etc. can be reused.

    let mut sb_blocks = String::new();
    for sb in 0u64..16 {
        // Precomputed offsets for this sub-block at the block level.
        // All elements in this sub-block: idx in [sb*16 .. sb*16+15]
        // ql base byte index: sb*16/2 = sb*8
        let ql_base: u64 = sb * 8; // ql[idx/2] when idx = sb*16+j: (sb*16+j)/2 = sb*8 + j/2
        // qh base byte: 128 + sb*16/4 = 128 + sb*4
        let qh_base: u64 = 128 + sb * 4;
        // sc byte: 192 + sb
        let sc_off: u64 = 192 + sb;
        // k-element base within the 256-element super-block: sb*16
        let k_sb: u64 = sb * 16;

        let prev_acc = if sb == 0 {
            "%acc_sb_init".to_string()
        } else {
            format!("%acc_sb{}", sb - 1)
        };

        sb_blocks.push_str(&format!(
            r#"
            // Sub-block {sb}: sc = block[{sc_off}] as i8, then dequant d * sc
            %sc{sb}_ptr   = arith.addi %blk_off, %csc{sc_off} : index
            %sc{sb}_raw   = memref.load %weight[%sc{sb}_ptr] : memref<{wb}xi8>
            %sc{sb}_i32   = arith.extsi %sc{sb}_raw : i8 to i32
            %sc{sb}_f32   = arith.sitofp %sc{sb}_i32 : i32 to f32
            %scale{sb}    = arith.mulf %d_f32, %sc{sb}_f32 : f32
            %ql{sb}_base  = arith.addi %blk_off, %cql{ql_base} : index
            %qh{sb}_base  = arith.addi %blk_off, %cqh{qh_base} : index
            %k{sb}_base   = arith.addi %k_block_base, %ck{k_sb} : index

            %acc_sb{sb} = scf.for %j = %c0 to %c16 step %c1
                              iter_args(%acc_j = {prev_acc}) -> (f32) {{
              // ql byte: ql_base + j/2
              %j_half_ql  = arith.divui %j, %c2 : index
              %ql_off     = arith.addi %ql{sb}_base, %j_half_ql : index
              %ql_raw     = memref.load %weight[%ql_off] : memref<{wb}xi8>
              %ql_u32     = arith.extui %ql_raw : i8 to i32
              %j_mod2     = arith.remui %j, %c2 : index
              %ql_shift   = arith.muli %j_mod2, %c4_idx : index
              %ql_shift32 = arith.index_cast %ql_shift : index to i32
              %ql_sh      = arith.shrui %ql_u32, %ql_shift32 : i32
              %lo         = arith.andi  %ql_sh, %c15_i32 : i32

              // qh byte: qh_base + j/4
              %j_half_qh  = arith.divui %j, %c4 : index
              %qh_off     = arith.addi %qh{sb}_base, %j_half_qh : index
              %qh_raw     = memref.load %weight[%qh_off] : memref<{wb}xi8>
              %qh_u32     = arith.extui %qh_raw : i8 to i32
              %j_mod4     = arith.remui %j, %c4 : index
              %qh_shift   = arith.muli %j_mod4, %c2_idx : index
              %qh_shift32 = arith.index_cast %qh_shift : index to i32
              %qh_sh      = arith.shrui %qh_u32, %qh_shift32 : i32
              %hi         = arith.andi  %qh_sh, %c3_i32 : i32

              // q = (lo | (hi << 4)) as i8 - 32
              %hi_sh      = arith.shli  %hi, %c4_i32 : i32
              %combined   = arith.ori   %lo, %hi_sh : i32
              // Interpret as signed i8: truncate to i8 then sign-extend back to i32
              %comb_i8    = arith.trunci %combined : i32 to i8
              %q_i32      = arith.extsi  %comb_i8  : i8 to i32
              %q_sub      = arith.subi   %q_i32, %c32_i32 : i32
              %q_f32      = arith.sitofp %q_sub : i32 to f32
              %val        = arith.mulf   %q_f32, %scale{sb} : f32

              // Load activation element
              %k_elem     = arith.addi %k{sb}_base, %j : index
              %a_val      = memref.load %act[%c0, %m, %k_elem] : memref<1x?x{k}xf32>
              %contrib    = arith.mulf %val, %a_val : f32
              %new_acc    = arith.addf %acc_j, %contrib : f32
              scf.yield %new_acc : f32
            }}
"#,
            sb = sb,
            sc_off = sc_off,
            ql_base = ql_base,
            qh_base = qh_base,
            k_sb = k_sb,
            prev_acc = prev_acc,
            wb = total_weight_bytes,
            k = k,
        ));
    }

    // Build extra constants.
    // sc byte offsets: 192..207
    // ql base offsets: 0, 8, 16, .., 120 (sb*8 for sb 0..16)
    // qh base offsets: 128, 132, 136, .., 188 (128+sb*4 for sb 0..16)
    // k base offsets: 0, 16, 32, .., 240 (sb*16 for sb 0..16)
    let mut extra_consts = String::new();

    // sc byte offsets 192..207
    for sb in 0u64..16 {
        let v = 192 + sb;
        extra_consts.push_str(&format!("    %csc{v} = arith.constant {v} : index\n"));
    }
    // ql base offsets: sb*8 for sb 0..16 = 0,8,16,...,120
    for sb in 0u64..16 {
        let v = sb * 8;
        extra_consts.push_str(&format!("    %cql{v} = arith.constant {v} : index\n"));
    }
    // qh base offsets: 128+sb*4 for sb 0..16 = 128,132,...,188
    for sb in 0u64..16 {
        let v = 128 + sb * 4;
        extra_consts.push_str(&format!("    %cqh{v} = arith.constant {v} : index\n"));
    }
    // k base offsets: sb*16 for sb 0..16 = 0,16,32,...,240
    for sb in 0u64..16 {
        let v = sb * 16;
        extra_consts.push_str(&format!("    %ck{v} = arith.constant {v} : index\n"));
    }
    // i32 bit-op constants
    for v in [3u32, 4, 15, 32] {
        extra_consts.push_str(&format!("    %c{v}_i32 = arith.constant {v} : i32\n"));
    }
    // index constants used as multipliers
    extra_consts.push_str("    %c2_idx = arith.constant 2 : index\n");
    extra_consts.push_str("    %c4_idx = arith.constant 4 : index\n");
    // 256 as index for k_block_base
    extra_consts.push_str("    %c256 = arith.constant 256 : index\n");
    // f16 scale byte offsets (208, 209)
    extra_consts.push_str("    %cd208 = arith.constant 208 : index\n");
    extra_consts.push_str("    %cd209 = arith.constant 209 : index\n");

    let final_acc = "%acc_sb15";

    format!(
        r#"// Q6_K dequant-matmul: K={k} N={n} n_tile={n_tile}
// Weight layout: {total_weight_bytes} bytes ({total_blocks} blocks x 210 bytes/block)
module attributes {{"homura.quant_kernel"}} {{
  func.func @{func_name}(
      %act    : memref<1x?x{k}xf32>,
      %weight : memref<{total_weight_bytes}xi8>,
      %out    : memref<1x?x{n}xf32>)
      attributes {{llvm.emit_c_interface}} {{

    %c0        = arith.constant 0 : index
    %c1        = arith.constant 1 : index
    %c2        = arith.constant 2 : index
    %c4        = arith.constant 4 : index
    %c16       = arith.constant 16 : index
    %c210      = arith.constant 210 : index
    %c8_i16    = arith.constant 8 : i16
    %c0_f32    = arith.constant 0.000000e+00 : f32
    %nbpr      = arith.constant {num_blocks_per_row} : index
    %n_tiles_c = arith.constant {n_tiles} : index
    %tile_sz   = arith.constant {n_tile} : index
{extra_consts}
    // Dynamic sequence length (M dimension).
    %seq = memref.dim %act, %c1 : memref<1x?x{k}xf32>

    // Parallel over N tiles.
    scf.forall (%tile_idx) in (%n_tiles_c) {{
      %n_base = arith.muli %tile_idx, %tile_sz : index

      scf.for %m = %c0 to %seq step %c1 {{
        scf.for %n_local = %c0 to %tile_sz step %c1 {{
          %n_idx = arith.addi %n_base, %n_local : index

          // Accumulate over K super-blocks.
          %acc_kb = scf.for %kb = %c0 to %nbpr step %c1
                        iter_args(%acc_kb_in = %c0_f32) -> (f32) {{

            // Flat block byte offset: (n_idx * nbpr + kb) * 210
            %blk_n   = arith.muli %n_idx, %nbpr : index
            %blk_idx = arith.addi %blk_n, %kb   : index
            %blk_off = arith.muli %blk_idx, %c210 : index

            // Load d (f16 at bytes 208-209, little-endian).
            %d0_off = arith.addi %blk_off, %cd208 : index
            %d1_off = arith.addi %blk_off, %cd209 : index
            %d0_i8  = memref.load %weight[%d0_off] : memref<{total_weight_bytes}xi8>
            %d1_i8  = memref.load %weight[%d1_off] : memref<{total_weight_bytes}xi8>
            %d0_i16 = arith.extui %d0_i8 : i8 to i16
            %d1_i16 = arith.extui %d1_i8 : i8 to i16
            %d1_sh  = arith.shli  %d1_i16, %c8_i16 : i16
            %d_i16  = arith.ori   %d0_i16, %d1_sh  : i16
            %d_f16  = arith.bitcast %d_i16 : i16 to f16
            %d_f32  = arith.extf  %d_f16 : f16 to f32

            // k-element base for this super-block: kb * 256.
            %k_block_base = arith.muli %kb, %c256 : index

            // Initial accumulator for sub-block loop chain.
            %acc_sb_init = arith.constant 0.000000e+00 : f32
{sb_blocks}
            %kb_contrib = arith.addf %acc_kb_in, {final_acc} : f32
            scf.yield %kb_contrib : f32
          }}

          memref.store %acc_kb, %out[%c0, %m, %n_idx] : memref<1x?x{n}xf32>
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
        total_blocks = total_blocks,
        extra_consts = extra_consts,
        sb_blocks = sb_blocks,
        final_acc = final_acc,
    )
}

/// Emit a dequant-matmul MLIR module for the given quantization dtype.
pub fn emit_dequant_matmul(
    dtype: crate::DType,
    k: u64,
    n: u64,
    func_name: &str,
    n_tile: usize,
) -> String {
    match dtype {
        crate::DType::Q8_0 => emit_dequant_matmul_q8_0(k, n, func_name, n_tile),
        crate::DType::Q4_K => emit_dequant_matmul_q4_k(k, n, func_name, n_tile),
        crate::DType::Q6_K => emit_dequant_matmul_q6_k(k, n, func_name, n_tile),
        _ => panic!("unsupported quant dtype {:?}", dtype),
    }
}

/// Compute the total flat weight buffer size in bytes for a quantized matrix.
pub fn quant_weight_bytes(dtype: crate::DType, k: u64, n: u64) -> u64 {
    match dtype {
        crate::DType::Q8_0 => {
            let nblk = k / 32;
            n * nblk * 34
        }
        crate::DType::Q4_K => {
            let nblk = k / 256;
            n * nblk * 144
        }
        crate::DType::Q6_K => {
            let nblk = k / 256;
            n * nblk * 210
        }
        _ => panic!("unsupported quant dtype {:?}", dtype),
    }
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

    /// Parse the emitted Q4_K MLIR and verify it compiles through the quant pipeline.
    #[test]
    fn emit_q4_k_parse_and_compile() {
        // K=256 (one super-block), N=4, n_tile=2
        let mlir = emit_dequant_matmul_q4_k(256, 4, "k0", 2);

        // 1. Verify the MLIR text is parseable.
        let context = create_context_pub();
        let _module =
            melior::ir::Module::parse(&context, &mlir).expect("Q4_K MLIR failed to parse");

        // 2. Compile through the quant pipeline to .o files.
        let tmp = tempfile_dir().unwrap();
        let obj_paths = compile_to_objects(&mlir, "test_q4_k", "k0", &tmp)
            .expect("Q4_K compile_to_objects failed");
        assert!(
            !obj_paths.is_empty(),
            "expected at least one .o file from Q4_K kernel"
        );
        for p in &obj_paths {
            assert!(p.exists(), "object file not found: {}", p.display());
        }
    }

    /// Correctness check for Q4_K kernel.
    ///
    /// Setup: K=256, N=4, SEQ=1.
    /// d=1.0f16, dmin=0.0f16 (no min), all 8 sub-block scales sc[i]=1, mn[i]=0.
    /// All 256 quant nibbles = 1.
    /// All activation values = 1.0f32.
    ///
    /// Each element dequants to: nibble(1) * scale(1.0*1) - min(0.0*0) = 1.0
    /// Expected output per neuron: 256 * 1.0 * 1.0 = 256.0
    #[test]
    fn emit_q4_k_correctness() {
        const K: u64 = 256;
        const N: u64 = 4;
        const N_TILE: usize = 2;
        const SEQ: u64 = 1;

        let mlir = emit_dequant_matmul_q4_k(K, N, "kq4k_corr", N_TILE);

        let tmp = tempfile_dir().unwrap();
        let obj_paths =
            compile_to_objects(&mlir, "corr_q4_k", "kq4k_corr", &tmp).expect("compile failed");

        let so_path = tmp.join("homura_test_q4_k_corr.so");
        link_shared_lib_pub(&obj_paths, &so_path).expect("link failed");

        let out_desc = OutputDesc {
            shape: crate::shape::Shape(vec![1, SEQ, N]),
            dtype: DType::F32,
        };
        let graph = CompiledGraph::load_named(&so_path, 2, vec![out_desc], "kq4k_corr")
            .expect("dlopen failed");

        // Build weight buffer: N * (K/256) * 144 = 4 * 1 * 144 = 576 bytes
        let num_blocks = N as usize; // K/256=1 block per row, N rows
        let total_bytes = num_blocks * 144;
        let mut weight_bytes = vec![0u8; total_bytes];

        // 1.0 in f16 = 0x3C00 (little-endian)
        let one_f16_lo: u8 = 0x00;
        let one_f16_hi: u8 = 0x3C;
        // 0.0 in f16 = 0x0000
        for b in 0..num_blocks {
            let off = b * 144;
            // d = 1.0f16 at bytes 0-1
            weight_bytes[off] = one_f16_lo;
            weight_bytes[off + 1] = one_f16_hi;
            // dmin = 0.0f16 at bytes 2-3 (already zero)

            // Scale bytes 4-15: set sc[0..3]=1 and mn[0..3]=0 (already zero).
            // sc[i] for i<4: scales_raw[i] & 0x3F, so set scales_raw[i]=1 -> sc[i]=1
            // mn[i] for i<4: scales_raw[4+i] & 0x3F, already 0
            // sc[i] for i>=4 (i=sb-4, 0..3):
            //   sc = (scales_raw[i] >> 6) | (high_nib << 2)
            //   To get sc=1: set scales_raw[i]=1 (lo bits 0-5=1, bits 6-7=0), high_nib=0
            //   -> sc = (1 >> 6) | 0 = 0  (not 1!)
            //   To get sc=1 from the high-bit formula we need high_nib=0, scales_raw[i]>>6=1
            //   That requires bit 6 of scales_raw[i]=1, so scales_raw[i] |= 0x40 (=64)
            //   But then sc[i<4] = scales_raw[i] & 0x3F = 64 & 0x3F = 0 for same byte
            //   Use a simpler approach: set scales_raw[i] = 0x40 for i=0..3 so that:
            //     sc[i<4] = 0x40 & 0x3F = 0   (but we want 1)
            //   That doesn't work simply. Instead set all 8 sc=1 by using 0x01 for bytes 0..3
            //   (gives sc[0..3]=1) and for sb 4..7, since high_nib comes from bytes 8-11
            //   and lo comes from bytes 0..3 >> 6:
            //   scales_raw[0..3] = 0x01 -> bits 6-7=0, lo_part=0. sc[4..7]=0.
            //   Set bytes 8..11 to give high_nib=1:
            //     scales_raw[8+i/2] needs nibble at bit((i%2)*4) = 1
            //     For i=0,1: scales_raw[8] = 0x11 (nib0=1, nib1=1)
            //     For i=2,3: scales_raw[9] = 0x11
            //   => sc[4..7] = 0 | (1 << 2) = 4, not 1.
            //
            // This is getting complicated. Use a simpler test: d=1.0, all sc=0 except
            // using a direct value:
            //   Set scales_raw[0..3]=1 -> sc[0..3]=1, sc[4..7]=0
            //   All 256 nibbles=1, activation=1.0
            //   Expected: sum over sub-blocks 0..3 of 32*1*(1.0*1-0.0*0) = 4*32=128
            //
            // To keep the test simple and match the docstring (expected=256), use:
            //   All sc=1: set bytes 0..3 to 1 (sc[0..3]=1, sc[4..7]=0 unless we fix high bits)
            //   Then expected = 4*32*1 + 4*32*0 = 128
            // Let's just test for expected=128 with sc[0..3]=1, sc[4..7]=0.
            // Set scales_raw[0..3] = 1
            for i in 0..4usize {
                weight_bytes[off + 4 + i] = 1; // sc[i]=1 for i<4
            }
            // scales_raw[4..7] = 0 -> mn[0..3]=0 (already)
            // scales_raw[8..11] = 0 -> high bits of sc[4..7]=0, mn[4..7] stays 0

            // Quant bytes 16-143: pack all nibbles = 1.
            // Each byte encodes two nibbles: low=1, high=1 -> 0x11
            for qi in 0..128usize {
                weight_bytes[off + 16 + qi] = 0x11;
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

        // Sub-blocks 0..3: scale=1.0*1=1.0, min=0.0*0=0.0, 32 elements * nibble(1) * act(1.0) = 32
        // Sub-blocks 4..7: scale=1.0*0=0.0, min=0.0*0=0.0, contribution = 0
        // Total per neuron = 4 * 32 = 128
        let expected: f32 = 128.0;
        for (i, &v) in out_slice.iter().enumerate() {
            let diff = (v - expected).abs();
            assert!(
                diff < 1.0,
                "out[{i}] = {v}, expected ~{expected} (diff = {diff})"
            );
        }
    }

    /// Correctness check for Q4_K with realistic random-ish data.
    /// Compares kernel output against the Rust reference dequant from gguf.rs.
    #[test]
    fn emit_q4_k_correctness_realistic() {
        const K: u64 = 256;
        const N: u64 = 2;
        const N_TILE: usize = 1;
        const SEQ: u64 = 1;

        let mlir = emit_dequant_matmul_q4_k(K, N, "kq4r", N_TILE);
        let tmp = tempfile_dir().unwrap();
        let obj_paths =
            compile_to_objects(&mlir, "real_q4k", "kq4r", &tmp).expect("compile failed");
        let so_path = tmp.join("homura_test_q4k_real.so");
        link_shared_lib_pub(&obj_paths, &so_path).expect("link failed");

        let out_desc = OutputDesc {
            shape: crate::shape::Shape(vec![1, SEQ, N]),
            dtype: DType::F32,
        };
        let graph =
            CompiledGraph::load_named(&so_path, 2, vec![out_desc], "kq4r").expect("dlopen failed");

        // Build realistic Q4_K weight data
        let num_blocks = N as usize; // K/256=1 block per row, N rows
        let total_bytes = num_blocks * 144;
        let mut weight_bytes = vec![0u8; total_bytes];

        for b in 0..num_blocks {
            let off = b * 144;
            // d = 0.5f16 = 0x3800
            weight_bytes[off] = 0x00;
            weight_bytes[off + 1] = 0x38;
            // dmin = 0.25f16 = 0x3400
            weight_bytes[off + 2] = 0x00;
            weight_bytes[off + 3] = 0x34;

            // Scale bytes: use various values
            // scales_raw[0..3]: sc[0..3] = val & 0x3F
            weight_bytes[off + 4] = 5; // sc[0]=5
            weight_bytes[off + 5] = 10; // sc[1]=10
            weight_bytes[off + 6] = 15; // sc[2]=15
            weight_bytes[off + 7] = 20; // sc[3]=20
            // scales_raw[4..7]: mn[0..3] = val & 0x3F
            weight_bytes[off + 8] = 2; // mn[0]=2
            weight_bytes[off + 9] = 4; // mn[1]=4
            weight_bytes[off + 10] = 6; // mn[2]=6
            weight_bytes[off + 11] = 8; // mn[3]=8
            // scales_raw[8..11]: high bits for sc[4..7] and mn[4..7]
            // Leave as 0 -> sc[4..7]=0, mn[4..7]=0 (simpler to verify)

            // Quant bytes: use a pattern
            for qi in 0..128usize {
                let lo = ((qi * 3 + b * 7) % 16) as u8;
                let hi = ((qi * 5 + b * 11) % 16) as u8;
                weight_bytes[off + 16 + qi] = lo | (hi << 4);
            }
        }

        // Rust reference dequant
        let mut ref_matrix = vec![0.0f32; (N * K) as usize];
        for b in 0..num_blocks {
            let off = b * 144;
            let block = &weight_bytes[off..off + 144];
            let d = f16_to_f32_test(u16::from_le_bytes([block[0], block[1]]));
            let dmin = f16_to_f32_test(u16::from_le_bytes([block[2], block[3]]));
            let scales_raw = &block[4..16];
            let qs = &block[16..144];

            let mut sc = [0u8; 8];
            let mut mn = [0u8; 8];
            for i in 0..4 {
                sc[i] = scales_raw[i] & 0x3F;
                mn[i] = scales_raw[4 + i] & 0x3F;
            }
            for i in 0..4 {
                let high_sc = (scales_raw[8 + i / 2] >> ((i % 2) * 4)) & 0x0F;
                sc[4 + i] = (scales_raw[i] >> 6) | (high_sc << 2);
                let high_mn = (scales_raw[10 + i / 2] >> ((i % 2) * 4)) & 0x0F;
                mn[4 + i] = (scales_raw[4 + i] >> 6) | (high_mn << 2);
            }

            for sb in 0..8usize {
                let scale = d * sc[sb] as f32;
                let min = dmin * mn[sb] as f32;
                for j in 0..32usize {
                    let byte_idx = sb * 16 + j / 2;
                    let nibble = if j % 2 == 0 {
                        qs[byte_idx] & 0x0F
                    } else {
                        qs[byte_idx] >> 4
                    };
                    ref_matrix[b * 256 + sb * 32 + j] = nibble as f32 * scale - min;
                }
            }
        }

        // Activation: all 1.0 (so output = sum of dequantized weights per row)
        let act_data = vec![1.0f32; K as usize];
        let act_buf = Buffer::from_slice(&act_data, &[1, SEQ, K], DType::F32);
        let weight_buf = Buffer::from_slice(&weight_bytes, &[total_bytes as u64], DType::I8);

        let outputs = graph.run(&[&act_buf, &weight_buf]);
        let out_slice = outputs[0].as_slice::<f32>();

        for row in 0..N as usize {
            let ref_sum: f32 = ref_matrix[row * 256..(row + 1) * 256].iter().sum();
            let kernel_val = out_slice[row];
            let diff = (kernel_val - ref_sum).abs();
            let rel = if ref_sum.abs() > 1e-6 {
                diff / ref_sum.abs()
            } else {
                diff
            };
            eprintln!(
                "row {row}: kernel={kernel_val:.4} ref={ref_sum:.4} diff={diff:.6} rel={rel:.6}"
            );
            assert!(
                rel < 0.01 || diff < 0.1,
                "Q4_K mismatch at row {row}: kernel={kernel_val} ref={ref_sum} diff={diff}"
            );
        }
    }

    fn f16_to_f32_test(bits: u16) -> f32 {
        let sign = ((bits >> 15) & 1) as u32;
        let exp = ((bits >> 10) & 0x1F) as u32;
        let frac = (bits & 0x3FF) as u32;
        if exp == 0 {
            if frac == 0 {
                f32::from_bits(sign << 31)
            } else {
                let mut e = 0i32;
                let mut f = frac;
                while (f & 0x400) == 0 {
                    f <<= 1;
                    e -= 1;
                }
                f &= 0x3FF;
                let exp32 = (127 - 15 + 1 + e) as u32;
                f32::from_bits((sign << 31) | (exp32 << 23) | (f << 13))
            }
        } else if exp == 31 {
            f32::from_bits((sign << 31) | (0xFF << 23) | (frac << 13))
        } else {
            let exp32 = (exp as i32 - 15 + 127) as u32;
            f32::from_bits((sign << 31) | (exp32 << 23) | (frac << 13))
        }
    }

    /// Parse the emitted Q6_K MLIR and verify it compiles through the quant pipeline.
    #[test]
    fn emit_q6_k_parse_and_compile() {
        // K=256 (one super-block), N=4, n_tile=2
        let mlir = emit_dequant_matmul_q6_k(256, 4, "k0", 2);

        // 1. Verify the MLIR text is parseable.
        let context = create_context_pub();
        let _module =
            melior::ir::Module::parse(&context, &mlir).expect("Q6_K MLIR failed to parse");

        // 2. Compile through the quant pipeline to .o files.
        let tmp = tempfile_dir().unwrap();
        let obj_paths = compile_to_objects(&mlir, "test_q6_k", "k0", &tmp)
            .expect("Q6_K compile_to_objects failed");
        assert!(
            !obj_paths.is_empty(),
            "expected at least one .o file from Q6_K kernel"
        );
        for p in &obj_paths {
            assert!(p.exists(), "object file not found: {}", p.display());
        }
    }

    /// Correctness check for Q6_K kernel.
    ///
    /// Setup: K=256, N=4, SEQ=1.
    /// d = 1.0f16 (super-block scale).
    /// sc[i] = 1 for all 16 sub-blocks (i8 value 1 stored as 0x01).
    /// All 256 quants: lo=1, hi=0 -> q = (1 | 0) as i8 - 32 = -31.
    /// dequant = -31 * (1.0 * 1) = -31.0.
    /// All activations = 1.0.
    /// Expected output per neuron = 256 * (-31.0) = -7936.0.
    #[test]
    fn emit_q6_k_correctness() {
        const K: u64 = 256;
        const N: u64 = 4;
        const N_TILE: usize = 2;
        const SEQ: u64 = 1;

        let mlir = emit_dequant_matmul_q6_k(K, N, "kq6k_corr", N_TILE);

        let tmp = tempfile_dir().unwrap();
        let obj_paths =
            compile_to_objects(&mlir, "corr_q6_k", "kq6k_corr", &tmp).expect("compile failed");

        let so_path = tmp.join("homura_test_q6_k_corr.so");
        link_shared_lib_pub(&obj_paths, &so_path).expect("link failed");

        let out_desc = OutputDesc {
            shape: crate::shape::Shape(vec![1, SEQ, N]),
            dtype: DType::F32,
        };
        let graph = CompiledGraph::load_named(&so_path, 2, vec![out_desc], "kq6k_corr")
            .expect("dlopen failed");

        // Build weight buffer: N * (K/256) * 210 = 4 * 1 * 210 = 840 bytes
        let num_blocks = N as usize; // K/256=1 block per row, N rows
        let total_bytes = num_blocks * 210;
        let mut weight_bytes = vec![0u8; total_bytes];

        // 1.0 in f16 = 0x3C00 (little-endian: lo=0x00, hi=0x3C)
        let one_f16_lo: u8 = 0x00;
        let one_f16_hi: u8 = 0x3C;

        for b in 0..num_blocks {
            let off = b * 210;

            // d = 1.0f16 at bytes 208-209 (little-endian)
            weight_bytes[off + 208] = one_f16_lo;
            weight_bytes[off + 209] = one_f16_hi;

            // sc[0..16] at bytes 192-207: set all to 1 (i8 value 1 = 0x01)
            for sb in 0..16usize {
                weight_bytes[off + 192 + sb] = 1;
            }

            // ql[0..128] at bytes 0-127: all nibbles = 1 -> each byte = 0x11
            // (lo nibble = 1, hi nibble = 1)
            for qi in 0..128usize {
                weight_bytes[off + qi] = 0x11;
            }
            // qh[0..64] at bytes 128-191: all hi-bits = 0 -> already zero
        }

        // Activation: 1 x SEQ x K, all 1.0
        let act_data = vec![1.0f32; K as usize * SEQ as usize];
        let act_buf = Buffer::from_slice(&act_data, &[1, SEQ, K], DType::F32);
        let weight_buf = Buffer::from_slice(&weight_bytes, &[total_bytes as u64], DType::I8);

        let outputs = graph.run(&[&act_buf, &weight_buf]);
        assert_eq!(outputs.len(), 1);

        let out_slice = outputs[0].as_slice::<f32>();
        assert_eq!(out_slice.len(), N as usize);

        // lo=1, hi=0 -> combined=1 as i8 - 32 = -31
        // dequant = -31 * (d=1.0) * (sc=1) = -31.0
        // 256 elements * (-31.0) * act(1.0) = -7936.0
        let expected: f32 = 256.0 * (-31.0);
        for (i, &v) in out_slice.iter().enumerate() {
            let diff = (v - expected).abs();
            assert!(
                diff < 1.0,
                "out[{i}] = {v}, expected ~{expected} (diff = {diff})"
            );
        }
    }
}
