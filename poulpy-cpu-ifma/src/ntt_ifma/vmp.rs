//! Vector-matrix product AVX512 kernels for [`NTTIfma`](crate::NTTIfma).
//!
//! This module contains AVX512-IFMA SIMD kernels for vector-matrix product
//! (VMP) operations in the IFMA NTT layout. These kernels override the generic
//! IFMA reference path with a backend-local 4-column tiled layout and a direct
//! row-strided apply kernel.

use bytemuck::{cast_slice, cast_slice_mut};
use core::arch::x86_64::{
    __m256i, __m512i, _mm_sfence, _mm512_add_epi64, _mm512_loadu_si512, _mm512_madd52hi_epu64, _mm512_madd52lo_epu64,
    _mm512_setzero_si512, _mm512_storeu_si512, _mm512_stream_si512,
};
use std::mem::size_of;

use poulpy_cpu_ref::reference::ntt_ifma::{
    NttIfmaCFromB, NttIfmaDFTExecute, NttIfmaFromZnx64, mat_vec::BbcIfmaMeta, primes::Primes40, types::Q_SHIFTED_IFMA,
    vec_znx_dft::NttIfmaModuleHandle,
};
use poulpy_hal::layouts::{
    DataView, DataViewMut, MatZnx, MatZnxToRef, Module, VecZnxDft, VecZnxDftToMut, VecZnxDftToRef, VmpPMat, VmpPMatToMut,
    VmpPMatToRef, ZnxInfos, ZnxView, ZnxViewMut,
};

use super::{
    mat_vec_ifma::{PrimeConsts512, reduce_bbc_single_prime_512},
    ntt_ifma_avx512::cond_sub_2q_si512,
};

// ──────────────────────────────────────────────────────────────────────────────
// SIMD save helpers
// ──────────────────────────────────────────────────────────────────────────────

fn q2_shifted_vec_512() -> __m512i {
    let q = Q_SHIFTED_IFMA;
    let q2_512 = [q[0], q[1], q[2], q[3], q[0], q[1], q[2], q[3]];
    unsafe { _mm512_loadu_si512(q2_512.as_ptr() as *const __m512i) }
}

/// Non-temporal write: bypass the cache so the result lines do not evict
/// matrix data the kernel still needs. `dst.as_mut_ptr().add(8 * blk)` is
/// 64-byte aligned because `VecZnxDft` storage is `DEFAULTALIGN`-aligned and
/// `8 * blk * size_of::<u64>() = 64 * blk` keeps that alignment. The
/// `_mm_sfence` to make these stores globally ordered is issued once at the
/// end of the apply loop, not per call.
#[target_feature(enable = "avx512f")]
unsafe fn save_blk_overwrite(_n: usize, blk: usize, dst: &mut [u64], src: &[u64]) {
    let off = 8 * blk;
    let dst_ptr = unsafe { dst.as_mut_ptr().add(off) as *mut __m512i };
    let src_ptr = src.as_ptr() as *const __m512i;
    unsafe {
        _mm512_stream_si512(dst_ptr, _mm512_loadu_si512(src_ptr));
    }
}

#[target_feature(enable = "avx512f,avx512vl")]
unsafe fn save_blk_add_simd(_n: usize, blk: usize, dst: &mut [u64], src: &[u64]) {
    unsafe {
        let q2_512 = q2_shifted_vec_512();
        let off = 8 * blk;
        let dst_ptr = dst.as_mut_ptr().add(off) as *mut __m256i;
        let src_ptr = src.as_ptr() as *const __m256i;

        let dst_ptr_512 = dst_ptr as *mut __m512i;
        let src_ptr_512 = src_ptr as *const __m512i;
        let d = _mm512_loadu_si512(dst_ptr_512 as *const __m512i);
        let s = _mm512_loadu_si512(src_ptr_512);
        let d_red = cond_sub_2q_si512(d, q2_512);
        _mm512_storeu_si512(dst_ptr_512, _mm512_add_epi64(d_red, s));
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// IFMA-local VMP prepare
// ──────────────────────────────────────────────────────────────────────────────

pub(crate) fn vmp_prepare_tmp_bytes_ifma(n: usize) -> usize {
    8 * n * size_of::<u64>()
}

pub(crate) fn vmp_apply_tmp_bytes_ifma(a_size: usize, b_rows: usize, b_cols_in: usize) -> usize {
    let row_max = a_size.min(b_rows) * b_cols_in;
    // 32 u64 for kernel output (4 blocks × 8 u64)
    // + 3 * 8 * row_max u64 for prime-major x extract (3 primes × 8 u64 × nrows)
    (32 + 3 * 8 * row_max) * size_of::<u64>()
}

/// Prime-major VMP prepare.
///
/// Layout: 3 planes (one per prime), each containing `n_blk_quads × nrows × ncols`
/// groups of 8 u64 (= 4 x2-blocks × 2 coefficients). Within each plane, columns are
/// stored column-major for streaming access during apply.
///
/// Plane `p` byte offset = `p * plane_bytes`.
/// Within a plane, element `(blk_quad, row, col)` offset in u64:
///   `blk_quad * nrows * ncols * 8  +  col * nrows * 8  +  row * 8`.
pub(crate) fn vmp_prepare_ifma<R, A>(module: &Module<crate::NTTIfma>, res: &mut R, a: &A, tmp: &mut [u64])
where
    R: VmpPMatToMut<crate::NTTIfma>,
    A: MatZnxToRef,
{
    let mut res: VmpPMat<&mut [u8], crate::NTTIfma> = res.to_mut();
    let a: MatZnx<&[u8]> = a.to_ref();
    let n = res.n();
    let nrows = a.cols_in() * a.rows();
    let ncols = a.cols_out() * a.size();
    let n_blks = n / 2;
    let n_blk_quads = n_blks / 4;

    let (tmp_b, tmp_c_u64) = tmp.split_at_mut(4 * n);
    let tmp_c: &mut [u32] = cast_slice_mut(tmp_c_u64);
    let mat_i64: &[i64] = a.raw();
    let pmat_u64: &mut [u64] = cast_slice_mut(res.data_mut());

    let plane_stride = n_blk_quads * nrows * ncols * 8; // u64 per prime plane
    let bq_stride = nrows * ncols * 8; // u64 per block-quad within a plane
    let col_stride = nrows * 8; // u64 per column within a block-quad

    // tmp_c is in interleaved q120c format: per coefficient 4 u64 (as 8 u32),
    // layout [p0, p1, p2, pad] × n coefficients. We scatter each coefficient
    // into the 3 prime planes at the right block-quad slot.
    for row_i in 0..nrows {
        for col_i in 0..ncols {
            let pos = n * (row_i * ncols + col_i);
            crate::NTTIfma::ntt_ifma_from_znx64(tmp_b, &mat_i64[pos..pos + n]);
            crate::NTTIfma::ntt_ifma_dft_execute(module.get_ntt_ifma_table(), tmp_b);
            crate::NTTIfma::ntt_ifma_c_from_b(n, tmp_c, tmp_b);

            let tmp_c_u64: &[u64] = bytemuck::cast_slice(tmp_c);

            for blk_j in 0..n_blks {
                let bq = blk_j / 4;
                let slot = blk_j % 4; // 0..3, maps to lanes 2*slot, 2*slot+1
                let coeff0_base = blk_j * 8; // in tmp_c_u64: 4 u64 per coeff, 2 coeffs per x2-block = 8 u64
                let dst_base = bq * bq_stride + col_i * col_stride + row_i * 8 + 2 * slot;

                for p in 0..3usize {
                    pmat_u64[p * plane_stride + dst_base] = tmp_c_u64[coeff0_base + p]; // coeff 0
                    pmat_u64[p * plane_stride + dst_base + 1] = tmp_c_u64[coeff0_base + 4 + p]; // coeff 1
                }
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// IFMA-local VMP apply
// ──────────────────────────────────────────────────────────────────────────────

/// SIMD index vectors for `permutex2var` de-interleave of two blocks'
/// `[p0,p1,p2,pad,p0,p1,p2,pad]` data. Used twice per row (blocks 0,1 then 2,3)
/// to fill the low and high halves of each prime-major `__m512i`.
const IDX_PM_P0: [i64; 8] = [0, 4, 8, 12, 0, 0, 0, 0];
const IDX_PM_P1: [i64; 8] = [1, 5, 9, 13, 0, 0, 0, 0];
const IDX_PM_P2: [i64; 8] = [2, 6, 10, 14, 0, 0, 0, 0];

/// Extract a block-quad from interleaved q120b into 3 prime-major planes.
#[target_feature(enable = "avx512f")]
unsafe fn extract_blk_quad_prime_major(n: usize, row_max: usize, bq: usize, a_u64: &[u64], x_pm: &mut [u64]) {
    use core::arch::x86_64::{_mm512_castsi512_si256, _mm512_inserti64x4, _mm512_permutex2var_epi64};

    let plane_stride = 8 * row_max;
    let blk_base = bq * 4 * 2;

    unsafe {
        let idx_p0 = _mm512_loadu_si512(IDX_PM_P0.as_ptr() as *const __m512i);
        let idx_p1 = _mm512_loadu_si512(IDX_PM_P1.as_ptr() as *const __m512i);
        let idx_p2 = _mm512_loadu_si512(IDX_PM_P2.as_ptr() as *const __m512i);

        for row in 0..row_max {
            let src = a_u64.as_ptr().add(row * 4 * n + 4 * blk_base) as *const __m512i;
            let s0 = _mm512_loadu_si512(src);
            let s1 = _mm512_loadu_si512(src.add(1));
            let s2 = _mm512_loadu_si512(src.add(2));
            let s3 = _mm512_loadu_si512(src.add(3));

            let lo_p0 = _mm512_permutex2var_epi64(s0, idx_p0, s1);
            let hi_p0 = _mm512_permutex2var_epi64(s2, idx_p0, s3);
            let p0 = _mm512_inserti64x4::<1>(lo_p0, _mm512_castsi512_si256(hi_p0));

            let lo_p1 = _mm512_permutex2var_epi64(s0, idx_p1, s1);
            let hi_p1 = _mm512_permutex2var_epi64(s2, idx_p1, s3);
            let p1 = _mm512_inserti64x4::<1>(lo_p1, _mm512_castsi512_si256(hi_p1));

            let lo_p2 = _mm512_permutex2var_epi64(s0, idx_p2, s1);
            let hi_p2 = _mm512_permutex2var_epi64(s2, idx_p2, s3);
            let p2 = _mm512_inserti64x4::<1>(lo_p2, _mm512_castsi512_si256(hi_p2));

            let dst = x_pm.as_mut_ptr().add(row * 8);
            _mm512_storeu_si512(dst as *mut __m512i, p0);
            _mm512_storeu_si512(dst.add(plane_stride) as *mut __m512i, p1);
            _mm512_storeu_si512(dst.add(2 * plane_stride) as *mut __m512i, p2);
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512ifma,avx512vl")]
unsafe fn vmp_apply_core_pm<const OVERWRITE: bool>(
    n: usize,
    res_u64: &mut [u64],
    a_u64: &[u64],
    pmat_u64: &[u64],
    limb_offset: usize,
    nrows: usize,
    ncols: usize,
    _meta: &BbcIfmaMeta<Primes40>,
    tmp: &mut [u64],
) {
    if n < 2 {
        return;
    }

    let n_blks = n / 2;
    let n_blk_quads = n_blks / 4;
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

    let pc = unsafe { [PrimeConsts512::new(0), PrimeConsts512::new(1), PrimeConsts512::new(2)] };

    // Scratch: 32 u64 for kernel output (4 blocks × 8 u64)
    //        + 3 * 8 * row_max u64 for prime-major x extract
    let (kernel_output, x_pm) = tmp.split_at_mut(32);
    let x_pm = &mut x_pm[..3 * 8 * row_max];

    // Matrix layout constants
    let plane_stride = n_blk_quads * nrows * ncols * 8; // u64 per prime plane
    let bq_stride = nrows * ncols * 8; // u64 per block-quad
    let col_stride_y = nrows * 8; // u64 per column within a block-quad

    for bq in 0..n_blk_quads {
        unsafe { extract_blk_quad_prime_major(n, row_max, bq, a_u64, x_pm) };

        for col_pmat in limb_offset..col_max {
            let col_res = col_pmat - limb_offset;
            let y_off = bq * bq_stride + col_pmat * col_stride_y;
            let x_plane_sz = 8 * row_max;

            unsafe {
                let mut red = [_mm512_setzero_si512(); 3];

                for p in 0..3 {
                    let mut acc_lo = _mm512_setzero_si512();
                    let mut acc_hi = _mm512_setzero_si512();
                    let x_base = x_pm.as_ptr().add(p * x_plane_sz) as *const __m512i;
                    let y_base = pmat_u64.as_ptr().add(p * plane_stride + y_off) as *const __m512i;

                    for r in 0..row_max {
                        let xv = _mm512_loadu_si512(x_base.add(r));
                        let yv = _mm512_loadu_si512(y_base.add(r));
                        acc_lo = _mm512_madd52lo_epu64(acc_lo, xv, yv);
                        acc_hi = _mm512_madd52hi_epu64(acc_hi, xv, yv);
                    }

                    red[p] = reduce_bbc_single_prime_512(
                        acc_lo,
                        acc_hi,
                        pc[p].q,
                        pc[p].q2,
                        pc[p].pow40,
                        pc[p].pow52,
                        pc[p].pow52_quot,
                    );
                }

                // SoA → AoS: interleave 3 prime results into 4 q120b blocks.
                let mut p0v = [0u64; 8];
                let mut p1v = [0u64; 8];
                let mut p2v = [0u64; 8];
                _mm512_storeu_si512(p0v.as_mut_ptr() as *mut __m512i, red[0]);
                _mm512_storeu_si512(p1v.as_mut_ptr() as *mut __m512i, red[1]);
                _mm512_storeu_si512(p2v.as_mut_ptr() as *mut __m512i, red[2]);

                for i in 0..4usize {
                    let blk = bq * 4 + i;
                    let buf = [
                        p0v[2 * i],
                        p1v[2 * i],
                        p2v[2 * i],
                        0,
                        p0v[2 * i + 1],
                        p1v[2 * i + 1],
                        p2v[2 * i + 1],
                        0,
                    ];
                    let base = col_res * 4 * n;
                    if OVERWRITE {
                        save_blk_overwrite(n, blk, &mut res_u64[base..], &buf);
                    } else {
                        kernel_output[..8].copy_from_slice(&buf);
                        save_blk_add_simd(n, blk, &mut res_u64[base..], &kernel_output[..8]);
                    }
                }
            }
        }
    }

    if OVERWRITE {
        let active_cols = col_max.saturating_sub(limb_offset);
        for col in active_cols..res_size {
            res_u64[col * 4 * n..(col + 1) * 4 * n].fill(0);
        }
        _mm_sfence();
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Public IFMA hooks
// ──────────────────────────────────────────────────────────────────────────────

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
    let res_size = res_ref.size();
    let nrows = pmat_ref.rows() * pmat_ref.cols_in();
    let ncols = pmat_ref.cols_out() * pmat_ref.size();
    let limb_offset = limb_offset * pmat_ref.cols_out();
    let _ = res_size;

    let res_u64: &mut [u64] = cast_slice_mut(res_ref.raw_mut());
    let a_u64: &[u64] = cast_slice(a_ref.raw());
    let pmat_u64: &[u64] = cast_slice(pmat_ref.data());

    unsafe {
        vmp_apply_core_pm::<true>(
            n,
            res_u64,
            a_u64,
            pmat_u64,
            limb_offset,
            nrows,
            ncols,
            module.get_bbc_ifma_meta(),
            tmp,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn vmp_apply_dft_to_dft_accumulate_ifma<R, A, C>(
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
    let res_size = res_ref.size();
    let nrows = pmat_ref.rows() * pmat_ref.cols_in();
    let ncols = pmat_ref.cols_out() * pmat_ref.size();
    let limb_offset = limb_offset * pmat_ref.cols_out();
    let _ = res_size;

    let res_u64: &mut [u64] = cast_slice_mut(res_ref.raw_mut());
    let a_u64: &[u64] = cast_slice(a_ref.raw());
    let pmat_u64: &[u64] = cast_slice(pmat_ref.data());

    unsafe {
        vmp_apply_core_pm::<false>(
            n,
            res_u64,
            a_u64,
            pmat_u64,
            limb_offset,
            nrows,
            ncols,
            module.get_bbc_ifma_meta(),
            tmp,
        );
    }
}
