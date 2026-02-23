//! VMP (vector-matrix product) operations for the NTT120 backend.
//!
//! This module provides the NTT-domain VMP primitives used by
//! `poulpy-cpu-ref`. The workflow is:
//!
//! 1. **Prepare** — encode a `MatZnx` (i64 coefficients) into the
//!    [`VmpPMat`] prepared format (q120c, NTT domain) via
//!    [`ntt120_vmp_prepare`].
//! 2. **Apply** — multiply a [`VecZnxDft`] (q120b) by a prepared
//!    [`VmpPMat`] (q120c) to obtain a new [`VecZnxDft`] (q120b) via
//!    [`ntt120_vmp_apply_dft_to_dft`] or [`ntt120_vmp_apply_dft_to_dft_add`].
//!
//! # Layout
//!
//! The `VmpPMat` buffer stores q120c data in a **block-interleaved**
//! cache-friendly layout designed for the `x2` NTT operations.
//!
//! Let `n_blks = n/2`, `offset = nrows * ncols * 16` (in u32 units).
//! For row `r`, output-column `c`, NTT x2-block `b`:
//!
//! | Column type | Buffer position (u32 index) |
//! |-------------|----------------------------|
//! | Even column `c` (paired) | `(c/2)*(nrows*32) + r*32 + (c%2)*16 + b*offset` |
//! | Last odd column `c`      | `c*nrows*16 + r*16 + b*offset` |
//!
//! Each x2-block slot stores 16 u32 = two consecutive q120c coefficients
//! (8 u32 each).

use bytemuck::{cast_slice, cast_slice_mut};

use crate::{
    layouts::{
        Backend, DataViewMut, MatZnx, MatZnxToRef, VecZnxDft, VecZnxDftToMut, VecZnxDftToRef, VmpPMat, VmpPMatToMut,
        VmpPMatToRef, ZnxInfos, ZnxView, ZnxViewMut,
    },
    reference::ntt120::{
        arithmetic::{b_from_znx64_ref, c_from_b_ref},
        mat_vec::{
            BbcMeta, extract_1blk_from_contiguous_q120b_ref, vec_mat1col_product_x2_bbc_ref, vec_mat2cols_product_x2_bbc_ref,
        },
        ntt::ntt_ref,
        primes::{PrimeSet, Primes30},
        types::Q120bScalar,
        vec_znx_dft::NttModuleHandle,
    },
};

// Lazy-reduction bound for q120b addition.
const Q_SHIFTED: [u64; 4] = [
    (Primes30::Q[0] as u64) << 33,
    (Primes30::Q[1] as u64) << 33,
    (Primes30::Q[2] as u64) << 33,
    (Primes30::Q[3] as u64) << 33,
];

// ──────────────────────────────────────────────────────────────────────────────
// Prepare
// ──────────────────────────────────────────────────────────────────────────────

/// Scratch space (in bytes) required by [`ntt120_vmp_prepare`].
///
/// Returns `4 * n * 8` bytes (one q120b NTT buffer of `4*n` u64).
pub fn ntt120_vmp_prepare_tmp_bytes(n: usize) -> usize {
    4 * n * size_of::<u64>()
}

/// Encode a polynomial matrix into the q120c NTT-domain prepared format.
///
/// For each `(row, col)` entry of the matrix:
/// 1. Map i64 coefficients to q120b ([`b_from_znx64_ref`]).
/// 2. Apply forward NTT ([`ntt_ref`]).
/// 3. Convert q120b → q120c ([`c_from_b_ref`]).
/// 4. Store in `res` in the block-interleaved layout (see module doc).
///
/// `tmp` must be at least [`ntt120_vmp_prepare_tmp_bytes`]`(n)` bytes.
pub fn ntt120_vmp_prepare<R, A, BE>(module: &impl NttModuleHandle, res: &mut R, a: &A, tmp: &mut [u8])
where
    BE: Backend<ScalarPrep = Q120bScalar>,
    R: VmpPMatToMut<BE>,
    A: MatZnxToRef,
{
    let mut res: VmpPMat<&mut [u8], BE> = res.to_mut();
    let a: MatZnx<&[u8]> = a.to_ref();
    let n = res.n();

    debug_assert_eq!(a.n(), n);
    debug_assert_eq!(res.cols_in(), a.cols_in());
    debug_assert_eq!(res.rows(), a.rows());
    debug_assert_eq!(res.cols_out(), a.cols_out());
    debug_assert_eq!(res.size(), a.size());
    debug_assert!(tmp.len() >= ntt120_vmp_prepare_tmp_bytes(n));

    let nrows: usize = a.cols_in() * a.rows();
    let ncols: usize = a.cols_out() * a.size();
    let n_blks: usize = n / 2;
    let offset: usize = nrows * ncols * 16; // u32 stride between blocks

    let tmp_u64: &mut [u64] = cast_slice_mut(tmp);
    let mat_i64: &[i64] = a.raw();
    let pmat_u32: &mut [u32] = cast_slice_mut(res.data_mut().as_mut());

    for row_i in 0..nrows {
        for col_i in 0..ncols {
            let pos = n * (row_i * ncols + col_i);

            // Step 1 & 2: i64 → q120b → NTT (in-place in tmp_u64)
            b_from_znx64_ref::<Primes30>(n, tmp_u64, &mat_i64[pos..pos + n]);
            ntt_ref(module.get_ntt_table(), tmp_u64);

            // Step 3: q120b → q120c (write into tmp_u32, same backing buffer)
            // We use a local Vec to avoid aliasing between tmp_u64 and tmp_u32.
            let tmp_q120c: Vec<u32> = {
                let mut v = vec![0u32; 8 * n];
                c_from_b_ref::<Primes30>(n, &mut v, tmp_u64);
                v
            };

            // Step 4: scatter into block-interleaved layout
            let dst_base: usize = if col_i == ncols - 1 && !ncols.is_multiple_of(2) {
                // Last odd column: uses the "single" slot layout
                col_i * nrows * 16 + row_i * 16
            } else {
                // Paired column
                (col_i / 2) * (nrows * 32) + row_i * 32 + (col_i % 2) * 16
            };

            for blk_j in 0..n_blks {
                let pmat_off = dst_base + blk_j * offset;
                pmat_u32[pmat_off..pmat_off + 16].copy_from_slice(&tmp_q120c[16 * blk_j..16 * blk_j + 16]);
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Apply helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Scratch space (in bytes) required by `ntt120_vmp_apply_dft_to_dft*`.
///
/// Allocates space for:
/// - `mat2cols_output`: 16 u64 (two paired-column x2-block results)
/// - `extracted_blk`:  8 × `row_max` u64 (one x2-block from each input row)
///
/// where `row_max = a_size.min(b_rows) * b_cols_in`.
pub fn ntt120_vmp_apply_dft_to_dft_tmp_bytes(a_size: usize, b_rows: usize, b_cols_in: usize) -> usize {
    let row_max = a_size.min(b_rows) * b_cols_in;
    (16 + 8 * row_max) * size_of::<u64>()
}

/// Save an x2-block (8 u64) into a q120b vector (overwrite mode).
#[inline(always)]
fn save_blk_overwrite(n: usize, blk: usize, dst: &mut [u64], src: &[u64]) {
    debug_assert!(src.len() >= 8);
    debug_assert!(dst.len() >= 4 * n);
    for i in 0..8 {
        dst[8 * blk + i] = src[i];
    }
}

/// Save an x2-block (8 u64) into a q120b vector with lazy accumulation.
#[inline(always)]
fn save_blk_add(n: usize, blk: usize, dst: &mut [u64], src: &[u64]) {
    debug_assert!(src.len() >= 8);
    debug_assert!(dst.len() >= 4 * n);
    for i in 0..8 {
        let k = i % 4;
        dst[8 * blk + i] = dst[8 * blk + i] % Q_SHIFTED[k] + src[i] % Q_SHIFTED[k];
    }
}

/// Zero an x2-block slot in a q120b vector.
#[inline(always)]
fn zero_blk(n: usize, blk: usize, dst: &mut [u64]) {
    debug_assert!(dst.len() >= 4 * n);
    for i in 0..8 {
        dst[8 * blk + i] = 0;
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Apply core
// ──────────────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn vmp_apply_dft_to_dft_core<const OVERWRITE: bool>(
    n: usize,
    res_u64: &mut [u64],
    a_u64: &[u64],
    pmat_u32: &[u32],
    limb_offset: usize,
    nrows: usize,
    ncols: usize,
    meta: &BbcMeta<Primes30>,
    tmp: &mut [u64],
) {
    debug_assert!(n >= 2);
    debug_assert!(n.is_power_of_two());

    let n_blks = n / 2;
    let a_size = a_u64.len() / (4 * n); // number of input polynomials
    let res_size = res_u64.len() / (4 * n); // number of output polynomials

    let row_max = nrows.min(a_size);
    let col_max = ncols.min(res_size + limb_offset);

    if limb_offset >= col_max {
        if OVERWRITE {
            res_u64.fill(0);
        }
        return;
    }

    // Split scratch: mat2cols_output (16 u64) | extracted_blk (8 * row_max u64)
    let (mat2cols_output, extracted_blk) = tmp.split_at_mut(16);

    let offset = nrows * ncols * 16; // u32 stride between blocks

    for blk_j in 0..n_blks {
        let mat_blk_u32 = &pmat_u32[blk_j * offset..];

        // Extract one x2-block from each input row into extracted_blk.
        extract_1blk_from_contiguous_q120b_ref(n, row_max, blk_j, extracted_blk, a_u64);
        let extracted_u32: &[u32] = cast_slice(extracted_blk);

        if limb_offset.is_multiple_of(2) {
            // Process paired columns: limb_offset, limb_offset+2, limb_offset+4, ...
            for (col_res, col_pmat) in (0..).step_by(2).zip((limb_offset..col_max - 1).step_by(2)) {
                let col_offset = col_pmat * (nrows * 16); // u32
                vec_mat2cols_product_x2_bbc_ref::<Primes30>(
                    meta,
                    row_max,
                    mat2cols_output,
                    extracted_u32,
                    &mat_blk_u32[col_offset..],
                );

                let (res_col0, res_col1) = (col_res, col_res + 1);
                let base0 = res_col0 * 4 * n;
                let base1 = res_col1 * 4 * n;
                if OVERWRITE {
                    save_blk_overwrite(n, blk_j, &mut res_u64[base0..], &mat2cols_output[0..8]);
                    save_blk_overwrite(n, blk_j, &mut res_u64[base1..], &mat2cols_output[8..16]);
                } else {
                    save_blk_add(n, blk_j, &mut res_u64[base0..], &mat2cols_output[0..8]);
                    save_blk_add(n, blk_j, &mut res_u64[base1..], &mat2cols_output[8..16]);
                }
            }
        } else {
            // Odd limb_offset: the first output col is the 2nd col of pair (limb_offset-1, limb_offset).
            let col_offset = (limb_offset - 1) * (nrows * 16);
            vec_mat2cols_product_x2_bbc_ref::<Primes30>(
                meta,
                row_max,
                mat2cols_output,
                extracted_u32,
                &mat_blk_u32[col_offset..],
            );

            // Only save the 2nd column result (mat2cols_output[8..16]) → col_res = 0
            if OVERWRITE {
                save_blk_overwrite(n, blk_j, &mut res_u64[0..], &mat2cols_output[8..16]);
            } else {
                save_blk_add(n, blk_j, &mut res_u64[0..], &mat2cols_output[8..16]);
            }

            // Process remaining paired columns.
            for (col_res, col_pmat) in (1..).step_by(2).zip((limb_offset + 1..col_max - 1).step_by(2)) {
                let col_offset = col_pmat * (nrows * 16);
                vec_mat2cols_product_x2_bbc_ref::<Primes30>(
                    meta,
                    row_max,
                    mat2cols_output,
                    extracted_u32,
                    &mat_blk_u32[col_offset..],
                );

                let base0 = col_res * 4 * n;
                let base1 = (col_res + 1) * 4 * n;
                if OVERWRITE {
                    save_blk_overwrite(n, blk_j, &mut res_u64[base0..], &mat2cols_output[0..8]);
                    save_blk_overwrite(n, blk_j, &mut res_u64[base1..], &mat2cols_output[8..16]);
                } else {
                    save_blk_add(n, blk_j, &mut res_u64[base0..], &mat2cols_output[0..8]);
                    save_blk_add(n, blk_j, &mut res_u64[base1..], &mat2cols_output[8..16]);
                }
            }
        }

        // Handle last odd output column (col_max is odd).
        if !col_max.is_multiple_of(2) {
            let last_col = col_max - 1;
            if last_col >= limb_offset {
                let col_offset = last_col * (nrows * 16);
                vec_mat1col_product_x2_bbc_ref::<Primes30>(
                    meta,
                    row_max,
                    &mut mat2cols_output[0..8],
                    extracted_u32,
                    &mat_blk_u32[col_offset..],
                );

                let col_res = last_col - limb_offset;
                let base = col_res * 4 * n;
                if OVERWRITE {
                    save_blk_overwrite(n, blk_j, &mut res_u64[base..], &mat2cols_output[0..8]);
                } else {
                    save_blk_add(n, blk_j, &mut res_u64[base..], &mat2cols_output[0..8]);
                }
            }
        }
    }

    // Zero output columns beyond col_max (overwrite mode only).
    if OVERWRITE {
        let active_cols = col_max - limb_offset;
        for col in active_cols..res_size {
            res_u64[col * 4 * n..(col + 1) * 4 * n].fill(0);
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Public apply functions
// ──────────────────────────────────────────────────────────────────────────────

/// NTT-domain vector-matrix product (overwrite): `res = a · pmat`.
///
/// For each NTT coefficient and each output polynomial, computes the
/// inner product of the input vector `a` with the corresponding column
/// of `pmat` using lazy q120b × q120c accumulation.
///
/// `tmp` must be at least [`ntt120_vmp_apply_dft_to_dft_tmp_bytes`]`(...)` bytes.
pub fn ntt120_vmp_apply_dft_to_dft<R, A, M, BE>(module: &impl NttModuleHandle, res: &mut R, a: &A, pmat: &M, tmp: &mut [u8])
where
    BE: Backend<ScalarPrep = Q120bScalar>,
    R: VecZnxDftToMut<BE>,
    A: VecZnxDftToRef<BE>,
    M: VmpPMatToRef<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: VecZnxDft<&[u8], BE> = a.to_ref();
    let pmat: VmpPMat<&[u8], BE> = pmat.to_ref();

    debug_assert_eq!(res.n(), pmat.n());
    debug_assert_eq!(a.n(), pmat.n());

    let n = res.n();
    let nrows = pmat.cols_in() * pmat.rows();
    let ncols = pmat.cols_out() * pmat.size();
    let meta = module.get_bbc_meta();

    let res_u64: &mut [u64] = cast_slice_mut(res.raw_mut());
    let a_u64: &[u64] = cast_slice(a.raw());
    let pmat_u32: &[u32] = cast_slice(pmat.raw());
    let tmp_u64: &mut [u64] = cast_slice_mut(tmp);

    vmp_apply_dft_to_dft_core::<true>(n, res_u64, a_u64, pmat_u32, 0, nrows, ncols, meta, tmp_u64);
}

/// NTT-domain vector-matrix product (accumulate): `res += a · pmat[limb_offset..]`.
///
/// Like [`ntt120_vmp_apply_dft_to_dft`] but accumulates into `res` with
/// lazy q120b addition, starting from column `limb_offset` of `pmat`.
///
/// `tmp` must be at least [`ntt120_vmp_apply_dft_to_dft_tmp_bytes`]`(...)` bytes.
pub fn ntt120_vmp_apply_dft_to_dft_add<R, A, M, BE>(
    module: &impl NttModuleHandle,
    res: &mut R,
    a: &A,
    pmat: &M,
    limb_offset: usize,
    tmp: &mut [u8],
) where
    BE: Backend<ScalarPrep = Q120bScalar>,
    R: VecZnxDftToMut<BE>,
    A: VecZnxDftToRef<BE>,
    M: VmpPMatToRef<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: VecZnxDft<&[u8], BE> = a.to_ref();
    let pmat: VmpPMat<&[u8], BE> = pmat.to_ref();

    debug_assert_eq!(res.n(), pmat.n());
    debug_assert_eq!(a.n(), pmat.n());

    let n = res.n();
    let nrows = pmat.cols_in() * pmat.rows();
    let ncols = pmat.cols_out() * pmat.size();
    let meta = module.get_bbc_meta();

    let res_u64: &mut [u64] = cast_slice_mut(res.raw_mut());
    let a_u64: &[u64] = cast_slice(a.raw());
    let pmat_u32: &[u32] = cast_slice(pmat.raw());
    let tmp_u64: &mut [u64] = cast_slice_mut(tmp);

    vmp_apply_dft_to_dft_core::<false>(n, res_u64, a_u64, pmat_u32, limb_offset, nrows, ncols, meta, tmp_u64);
}

// ──────────────────────────────────────────────────────────────────────────────
// Utility
// ──────────────────────────────────────────────────────────────────────────────

/// Zero all entries of a prepared polynomial matrix.
pub fn ntt120_vmp_zero<R, BE: Backend>(res: &mut R)
where
    R: VmpPMatToMut<BE>,
{
    res.to_mut().data_mut().as_mut().fill(0);
}
