//! Vector-matrix product AVX512 kernels for [`NTTIfma`](crate::NTTIfma).
//!
//! This module contains AVX512-IFMA SIMD kernels for vector-matrix product
//! (VMP) operations in the IFMA NTT layout. These kernels can be used to
//! override the default reference implementations for improved performance.

use bytemuck::{cast_slice, cast_slice_mut};
use core::arch::x86_64::{__m256i, _mm256_add_epi64, _mm256_loadu_si256, _mm256_storeu_si256};

use poulpy_cpu_ref::reference::ntt_ifma::{
    NttIfmaExtract1BlkContiguous, NttIfmaMulBbc1ColX2, NttIfmaMulBbc2ColsX2, mat_vec::BbcIfmaMeta, primes::Primes40,
    types::Q_SHIFTED_IFMA,
};

use super::ntt_ifma_avx512::cond_sub_2q_si256;
use crate::NTTIfma;

// ──────────────────────────────────────────────────────────────────────────────
// SIMD save_blk_add: replaces scalar % with conditional-subtract
// ──────────────────────────────────────────────────────────────────────────────

/// Q_SHIFTED_IFMA as a `__m256i`: `[2*Q[0], 2*Q[1], 2*Q[2], 0]`.
fn q2_shifted_vec() -> __m256i {
    unsafe { _mm256_loadu_si256(Q_SHIFTED_IFMA.as_ptr() as *const __m256i) }
}

#[inline(always)]
fn save_blk_overwrite(_n: usize, blk: usize, dst: &mut [u64], src: &[u64]) {
    dst[8 * blk..8 * blk + 8].copy_from_slice(&src[..8]);
}

/// SIMD accumulation: reduce `dst` from `[0, 4q)` to `[0, 2q)` via `cond_sub(2q)`,
/// then add `src` (in `[0, q)`). Result in `[0, 3q)`.
///
/// Replaces the scalar `%` (hardware division) path.
#[target_feature(enable = "avx512vl")]
unsafe fn save_blk_add_simd(_n: usize, blk: usize, dst: &mut [u64], src: &[u64]) {
    unsafe {
        let q2 = q2_shifted_vec();
        let off = 8 * blk;
        let dst_ptr = dst.as_mut_ptr().add(off) as *mut __m256i;
        let src_ptr = src.as_ptr() as *const __m256i;

        // First pair (lanes 0-3)
        let d0 = _mm256_loadu_si256(dst_ptr as *const __m256i);
        let s0 = _mm256_loadu_si256(src_ptr);
        let d0_red = cond_sub_2q_si256(d0, q2);
        _mm256_storeu_si256(dst_ptr, _mm256_add_epi64(d0_red, s0));

        // Second pair (lanes 4-7)
        let d1 = _mm256_loadu_si256((dst_ptr as *const __m256i).add(1));
        let s1 = _mm256_loadu_si256(src_ptr.add(1));
        let d1_red = cond_sub_2q_si256(d1, q2);
        _mm256_storeu_si256(dst_ptr.add(1), _mm256_add_epi64(d1_red, s1));
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Local VMP apply core with SIMD save_blk_add
// ──────────────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512vl")]
unsafe fn vmp_apply_core_simd<const OVERWRITE: bool>(
    n: usize,
    res_u64: &mut [u64],
    a_u64: &[u64],
    pmat_u32: &[u32],
    limb_offset: usize,
    nrows: usize,
    ncols: usize,
    meta: &BbcIfmaMeta<Primes40>,
    tmp: &mut [u64],
) {
    if n < 2 {
        return;
    }

    let n_blks = n / 2;
    let a_size = a_u64.len() / (4 * n);
    let res_size = res_u64.len() / (4 * n);
    let row_max = nrows.min(a_size);
    let col_max = ncols.min(res_size + limb_offset);

    if limb_offset >= col_max {
        if OVERWRITE {
            res_u64.fill(0);
        }
        return;
    }

    let (mat2cols_output, extracted_blk) = tmp.split_at_mut(16);
    let offset = nrows * ncols * 16;

    for blk_j in 0..n_blks {
        let mat_blk_u32 = &pmat_u32[blk_j * offset..];
        NTTIfma::ntt_ifma_extract_1blk_contiguous(n, row_max, blk_j, extracted_blk, a_u64);
        let extracted_u32: &[u32] = cast_slice(extracted_blk);

        let start_col = limb_offset;
        let mut col_pmat = start_col;
        let mut col_res = 0usize;

        // Handle odd start
        if !start_col.is_multiple_of(2) && col_pmat < col_max {
            let col_offset = (col_pmat - 1) * (nrows * 16);
            NTTIfma::ntt_ifma_mul_bbc_2cols_x2(meta, row_max, mat2cols_output, extracted_u32, &mat_blk_u32[col_offset..]);
            let base = col_res * 4 * n;
            if OVERWRITE {
                save_blk_overwrite(n, blk_j, &mut res_u64[base..], &mat2cols_output[8..16]);
            } else {
                unsafe { save_blk_add_simd(n, blk_j, &mut res_u64[base..], &mat2cols_output[8..16]) };
            }
            col_pmat += 1;
            col_res += 1;
        }

        // Process paired columns
        while col_pmat + 1 < col_max {
            let col_offset = col_pmat * (nrows * 16);
            NTTIfma::ntt_ifma_mul_bbc_2cols_x2(meta, row_max, mat2cols_output, extracted_u32, &mat_blk_u32[col_offset..]);
            let base0 = col_res * 4 * n;
            let base1 = (col_res + 1) * 4 * n;
            if OVERWRITE {
                save_blk_overwrite(n, blk_j, &mut res_u64[base0..], &mat2cols_output[0..8]);
                save_blk_overwrite(n, blk_j, &mut res_u64[base1..], &mat2cols_output[8..16]);
            } else {
                unsafe {
                    save_blk_add_simd(n, blk_j, &mut res_u64[base0..], &mat2cols_output[0..8]);
                    save_blk_add_simd(n, blk_j, &mut res_u64[base1..], &mat2cols_output[8..16]);
                }
            }
            col_pmat += 2;
            col_res += 2;
        }

        // Handle last odd column
        if col_pmat < col_max {
            let col_offset = col_pmat * (nrows * 16);
            NTTIfma::ntt_ifma_mul_bbc_1col_x2(
                meta,
                row_max,
                &mut mat2cols_output[0..8],
                extracted_u32,
                &mat_blk_u32[col_offset..],
            );
            let base = col_res * 4 * n;
            if OVERWRITE {
                save_blk_overwrite(n, blk_j, &mut res_u64[base..], &mat2cols_output[0..8]);
            } else {
                unsafe { save_blk_add_simd(n, blk_j, &mut res_u64[base..], &mat2cols_output[0..8]) };
            }
        }
    }

    // Zero remaining output columns in overwrite mode
    if OVERWRITE {
        let active_cols = col_max.saturating_sub(limb_offset);
        for col in active_cols..res_size {
            res_u64[col * 4 * n..(col + 1) * 4 * n].fill(0);
        }
    }
}

use poulpy_cpu_ref::reference::ntt_ifma::vec_znx_dft::NttIfmaModuleHandle;
use poulpy_hal::layouts::{
    Module, VecZnxDft, VecZnxDftToMut, VecZnxDftToRef, VmpPMat, VmpPMatToRef, ZnxInfos, ZnxView, ZnxViewMut,
};

/// AVX512-accelerated VMP apply with SIMD save_blk_add (conditional-subtract
/// instead of scalar `%`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn vmp_apply_dft_to_dft_ifma<R, A, C>(
    module: &Module<crate::NTTIfma>,
    res: &mut R,
    a: &A,
    pmat: &C,
    limb_offset: usize,
    tmp: &mut [u64],
) where
    R: VecZnxDftToMut<crate::NTTIfma>,
    A: VecZnxDftToRef<crate::NTTIfma>,
    C: VmpPMatToRef<crate::NTTIfma>,
{
    let mut res_ref: VecZnxDft<&mut [u8], crate::NTTIfma> = res.to_mut();
    let a_ref: VecZnxDft<&[u8], crate::NTTIfma> = a.to_ref();
    let pmat_ref: VmpPMat<&[u8], crate::NTTIfma> = pmat.to_ref();

    let n = res_ref.n();
    let nrows = pmat_ref.rows() * pmat_ref.cols_in();
    let ncols = pmat_ref.cols_out() * pmat_ref.size();

    let res_u64: &mut [u64] = cast_slice_mut(res_ref.raw_mut());
    let a_u64: &[u64] = cast_slice(a_ref.raw());
    let pmat_u32: &[u32] = cast_slice(pmat_ref.raw());

    let meta = module.get_bbc_ifma_meta();

    unsafe {
        vmp_apply_core_simd::<true>(
            n,
            res_u64,
            a_u64,
            pmat_u32,
            limb_offset * pmat_ref.cols_out(),
            nrows,
            ncols,
            meta,
            tmp,
        );
    }
}
