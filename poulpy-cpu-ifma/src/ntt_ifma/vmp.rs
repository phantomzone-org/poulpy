//! Vector-matrix product for [`NTTIfma`](crate::NTTIfma).
//!
//! This module implements the VMP OEP traits for the IFMA NTT layout.
//! Preparation converts integer-domain matrices into IFMA prepared form, and the
//! DFT-to-DFT apply path uses IFMA BBC kernels together with SIMD accumulation.

use bytemuck::{cast_slice, cast_slice_mut};
use core::arch::x86_64::{__m256i, _mm256_add_epi64, _mm256_loadu_si256, _mm256_storeu_si256};

use poulpy_hal::{
    api::TakeSlice,
    layouts::{
        MatZnxToRef, Module, Scratch, VecZnxDft, VecZnxDftToMut, VecZnxDftToRef, VmpPMat, VmpPMatToMut, VmpPMatToRef, ZnxInfos,
        ZnxView, ZnxViewMut,
    },
    oep::{VmpApplyDftToDftImpl, VmpApplyDftToDftTmpBytesImpl, VmpPrepareImpl, VmpPrepareTmpBytesImpl, VmpZeroImpl},
    reference::ntt_ifma::{
        NttIfmaExtract1BlkContiguous, NttIfmaMulBbc1ColX2, NttIfmaMulBbc2ColsX2, mat_vec::BbcIfmaMeta, primes::Primes40,
        types::Q_SHIFTED_IFMA, vec_znx_dft::NttIfmaModuleHandle, vmp::ntt_ifma_vmp_zero,
    },
};

use std::mem::size_of;

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

// ──────────────────────────────────────────────────────────────────────────────
// Trait implementations
// ──────────────────────────────────────────────────────────────────────────────

unsafe impl VmpPrepareTmpBytesImpl<Self> for NTTIfma {
    fn vmp_prepare_tmp_bytes_impl(module: &Module<Self>, _rows: usize, _cols_in: usize, _cols_out: usize, _size: usize) -> usize {
        // Need 4*n u64 for NTT tmp + 4*n u64 for c-format tmp (eliminates per-element heap alloc)
        8 * module.n() * size_of::<u64>()
    }
}

unsafe impl VmpPrepareImpl<Self> for NTTIfma {
    fn vmp_prepare_impl<R, A>(module: &Module<Self>, res: &mut R, a: &A, scratch: &mut Scratch<Self>)
    where
        R: VmpPMatToMut<Self>,
        A: MatZnxToRef,
    {
        use poulpy_hal::layouts::{DataViewMut, MatZnx};
        use poulpy_hal::reference::ntt_ifma::{NttIfmaCFromB, NttIfmaDFTExecute, NttIfmaFromZnx64};

        let mut res_view: VmpPMat<&mut [u8], Self> = res.to_mut();
        let a_view: MatZnx<&[u8]> = a.to_ref();
        let n = res_view.n();

        let nrows = a_view.cols_in() * a_view.rows();
        let ncols = a_view.cols_out() * a_view.size();
        let n_blks = n / 2;
        let offset = nrows * ncols * 16;

        let mat_i64: &[i64] = a_view.raw();
        let pmat_u32: &mut [u32] = cast_slice_mut(res_view.data_mut());

        let bytes = 8 * n * size_of::<u64>();
        let (scratch_u64, _) = scratch.take_slice::<u64>(bytes / size_of::<u64>());
        let (tmp_ntt, tmp_c) = scratch_u64.split_at_mut(4 * n);
        // Reinterpret tmp_c (4*n u64 = 8*n u32) as u32 slice for c_from_b
        let tmp_c_u32: &mut [u32] = unsafe { std::slice::from_raw_parts_mut(tmp_c.as_mut_ptr() as *mut u32, 8 * n) };

        let table = module.get_ntt_ifma_table();

        for row_i in 0..nrows {
            for col_i in 0..ncols {
                let pos = n * (row_i * ncols + col_i);

                NTTIfma::ntt_ifma_from_znx64(tmp_ntt, &mat_i64[pos..pos + n]);
                NTTIfma::ntt_ifma_dft_execute(table, tmp_ntt);
                NTTIfma::ntt_ifma_c_from_b(n, tmp_c_u32, tmp_ntt);

                let dst_base = if col_i == ncols - 1 && !ncols.is_multiple_of(2) {
                    col_i * nrows * 16 + row_i * 16
                } else {
                    (col_i / 2) * (nrows * 32) + row_i * 32 + (col_i % 2) * 16
                };

                for blk_j in 0..n_blks {
                    let pmat_off = dst_base + blk_j * offset;
                    pmat_u32[pmat_off..pmat_off + 16].copy_from_slice(&tmp_c_u32[16 * blk_j..16 * blk_j + 16]);
                }
            }
        }
    }
}

unsafe impl VmpApplyDftToDftTmpBytesImpl<Self> for NTTIfma {
    fn vmp_apply_dft_to_dft_tmp_bytes_impl(
        module: &Module<Self>,
        _res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        _b_cols_out: usize,
        _b_size: usize,
    ) -> usize {
        let n = module.n();
        let row_max = a_size.min(b_rows) * b_cols_in;
        // 16 u64 for mat2cols_output + 8 * row_max u64 for one extracted x2-block
        let _ = n;
        (16 + row_max * 8) * size_of::<u64>()
    }
}

unsafe impl VmpApplyDftToDftImpl<Self> for NTTIfma
where
    Scratch<Self>: TakeSlice,
    NTTIfma: VmpApplyDftToDftTmpBytesImpl<Self>,
{
    fn vmp_apply_dft_to_dft_impl<R, A, C>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        pmat: &C,
        limb_offset: usize,
        scratch: &mut Scratch<Self>,
    ) where
        R: VecZnxDftToMut<Self>,
        A: VecZnxDftToRef<Self>,
        C: VmpPMatToRef<Self>,
    {
        let mut res_ref: VecZnxDft<&mut [u8], Self> = res.to_mut();
        let a_ref: VecZnxDft<&[u8], Self> = a.to_ref();
        let pmat_ref: VmpPMat<&[u8], Self> = pmat.to_ref();

        let n = res_ref.n();
        let nrows = pmat_ref.rows() * pmat_ref.cols_in();
        let ncols = pmat_ref.cols_out() * pmat_ref.size();

        let bytes = Self::vmp_apply_dft_to_dft_tmp_bytes_impl(
            module,
            res_ref.size(),
            a_ref.size(),
            pmat_ref.rows(),
            pmat_ref.cols_in(),
            pmat_ref.cols_out(),
            pmat_ref.size(),
        );
        let (tmp, _) = scratch.take_slice::<u64>(bytes / size_of::<u64>());

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
}

unsafe impl VmpZeroImpl<Self> for NTTIfma {
    fn vmp_zero_impl<R>(_module: &Module<Self>, res: &mut R)
    where
        R: VmpPMatToMut<Self>,
    {
        ntt_ifma_vmp_zero::<R, Self>(res);
    }
}
