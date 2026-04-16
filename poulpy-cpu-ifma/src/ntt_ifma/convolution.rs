//! Polynomial convolution AVX512 kernels for [`NTTIfma`](crate::NTTIfma).
//!
//! Mirrors the block-outer pack-then-multiply structure used by the AVX
//! [`NTT120Avx`](poulpy_cpu_avx::NTT120Avx) convolution: for each x2 NTT
//! block, the left and right operand rows are first gathered into contiguous
//! scratch buffers (with the right operand in reversed row order), then each
//! output limb consumes a contiguous window of those buffers via the
//! [`vec_mat1col_product_x2_bbc_ifma`] kernel.
//!
//! Moving the block loop outermost turns the inner j-sum into a straight-line
//! accumulation over adjacent rows, lets the prefetcher stream ahead, and
//! keeps the hot BBC kernel's 4-way unrolling saturated.

use bytemuck::{cast_slice, cast_slice_mut};

use poulpy_cpu_ref::reference::ntt120::types::Q120bScalar;
use poulpy_hal::layouts::{CnvPVecLToRef, CnvPVecRToRef, VecZnxDftToMut, ZnxInfos, ZnxView, ZnxViewMut};

use super::mat_vec_ifma::{PrimeConsts512, reduce_bbc_single_prime_512};

use crate::NTTIfma;
use core::arch::x86_64::{
    __m512i, _mm_sfence, _mm512_add_epi64, _mm512_castsi512_si256, _mm512_inserti64x4, _mm512_loadu_si512, _mm512_madd52hi_epu64,
    _mm512_madd52lo_epu64, _mm512_permutex2var_epi64, _mm512_setzero_si512, _mm512_storeu_si512, _mm512_stream_si512,
};

// ─────────────────────────────────────────────────────────────────────────────
// Block-quad prime-major pack kernels
// ─────────────────────────────────────────────────────────────────────────────

const IDX_PM_P0: [i64; 8] = [0, 4, 8, 12, 0, 0, 0, 0];
const IDX_PM_P1: [i64; 8] = [1, 5, 9, 13, 0, 0, 0, 0];
const IDX_PM_P2: [i64; 8] = [2, 6, 10, 14, 0, 0, 0, 0];

/// De-interleave 4 consecutive `__m512i` (4 x2-blocks in `[p0,p1,p2,pad]`
/// format) into 3 prime-major `__m512i` and write them to `dst` at prime
/// offsets `0`, `plane_stride`, `2*plane_stride`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn deinterleave_quad_pm(
    s0: __m512i,
    s1: __m512i,
    s2: __m512i,
    s3: __m512i,
    dst: *mut u64,
    plane_stride: usize,
    idx_p0: __m512i,
    idx_p1: __m512i,
    idx_p2: __m512i,
) {
    unsafe {
        let lo = _mm512_permutex2var_epi64(s0, idx_p0, s1);
        let hi = _mm512_permutex2var_epi64(s2, idx_p0, s3);
        _mm512_storeu_si512(dst as *mut __m512i, _mm512_inserti64x4::<1>(lo, _mm512_castsi512_si256(hi)));

        let lo = _mm512_permutex2var_epi64(s0, idx_p1, s1);
        let hi = _mm512_permutex2var_epi64(s2, idx_p1, s3);
        _mm512_storeu_si512(
            dst.add(plane_stride) as *mut __m512i,
            _mm512_inserti64x4::<1>(lo, _mm512_castsi512_si256(hi)),
        );

        let lo = _mm512_permutex2var_epi64(s0, idx_p2, s1);
        let hi = _mm512_permutex2var_epi64(s2, idx_p2, s3);
        _mm512_storeu_si512(
            dst.add(2 * plane_stride) as *mut __m512i,
            _mm512_inserti64x4::<1>(lo, _mm512_castsi512_si256(hi)),
        );
    }
}

/// Pack left operand for a block-quad into prime-major scratch (forward order).
#[target_feature(enable = "avx512f")]
unsafe fn pack_left_quad_pm(dst: &mut [u64], a: &[u64], row_count: usize, row_stride: usize, blk_quad: usize) {
    let plane_stride = 8 * row_count;
    unsafe {
        let idx_p0 = _mm512_loadu_si512(IDX_PM_P0.as_ptr() as *const __m512i);
        let idx_p1 = _mm512_loadu_si512(IDX_PM_P1.as_ptr() as *const __m512i);
        let idx_p2 = _mm512_loadu_si512(IDX_PM_P2.as_ptr() as *const __m512i);
        let blk_base = blk_quad * 4;
        for row in 0..row_count {
            let src = a.as_ptr().add(row * row_stride + 8 * blk_base) as *const __m512i;
            deinterleave_quad_pm(
                _mm512_loadu_si512(src),
                _mm512_loadu_si512(src.add(1)),
                _mm512_loadu_si512(src.add(2)),
                _mm512_loadu_si512(src.add(3)),
                dst.as_mut_ptr().add(row * 8),
                plane_stride,
                idx_p0,
                idx_p1,
                idx_p2,
            );
        }
    }
}

/// Pack right operand for a block-quad into prime-major scratch (reversed row order).
#[target_feature(enable = "avx512f")]
unsafe fn pack_right_quad_pm(dst: &mut [u64], a: &[u64], row_count: usize, row_stride: usize, blk_quad: usize) {
    let plane_stride = 8 * row_count;
    unsafe {
        let idx_p0 = _mm512_loadu_si512(IDX_PM_P0.as_ptr() as *const __m512i);
        let idx_p1 = _mm512_loadu_si512(IDX_PM_P1.as_ptr() as *const __m512i);
        let idx_p2 = _mm512_loadu_si512(IDX_PM_P2.as_ptr() as *const __m512i);
        let blk_base = blk_quad * 4;
        for row in 0..row_count {
            let src_row = row_count - 1 - row;
            let src = a.as_ptr().add(src_row * row_stride + 8 * blk_base) as *const __m512i;
            deinterleave_quad_pm(
                _mm512_loadu_si512(src),
                _mm512_loadu_si512(src.add(1)),
                _mm512_loadu_si512(src.add(2)),
                _mm512_loadu_si512(src.add(3)),
                dst.as_mut_ptr().add(row * 8),
                plane_stride,
                idx_p0,
                idx_p1,
                idx_p2,
            );
        }
    }
}

/// Pairwise pack left for a block-quad (lane-add two columns, forward order).
#[target_feature(enable = "avx512f")]
unsafe fn pairwise_pack_left_quad_pm(
    dst: &mut [u64],
    a: &[u64],
    b: &[u64],
    row_count: usize,
    row_stride: usize,
    blk_quad: usize,
) {
    let plane_stride = 8 * row_count;
    unsafe {
        let idx_p0 = _mm512_loadu_si512(IDX_PM_P0.as_ptr() as *const __m512i);
        let idx_p1 = _mm512_loadu_si512(IDX_PM_P1.as_ptr() as *const __m512i);
        let idx_p2 = _mm512_loadu_si512(IDX_PM_P2.as_ptr() as *const __m512i);
        let blk_base = blk_quad * 4;
        for row in 0..row_count {
            let off = row * row_stride + 8 * blk_base;
            let sa = a.as_ptr().add(off) as *const __m512i;
            let sb = b.as_ptr().add(off) as *const __m512i;
            deinterleave_quad_pm(
                _mm512_add_epi64(_mm512_loadu_si512(sa), _mm512_loadu_si512(sb)),
                _mm512_add_epi64(_mm512_loadu_si512(sa.add(1)), _mm512_loadu_si512(sb.add(1))),
                _mm512_add_epi64(_mm512_loadu_si512(sa.add(2)), _mm512_loadu_si512(sb.add(2))),
                _mm512_add_epi64(_mm512_loadu_si512(sa.add(3)), _mm512_loadu_si512(sb.add(3))),
                dst.as_mut_ptr().add(row * 8),
                plane_stride,
                idx_p0,
                idx_p1,
                idx_p2,
            );
        }
    }
}

/// Pairwise pack right for a block-quad (lane-add two columns, reversed).
#[target_feature(enable = "avx512f")]
unsafe fn pairwise_pack_right_quad_pm(
    dst: &mut [u64],
    a: &[u64],
    b: &[u64],
    row_count: usize,
    row_stride: usize,
    blk_quad: usize,
) {
    let plane_stride = 8 * row_count;
    unsafe {
        let idx_p0 = _mm512_loadu_si512(IDX_PM_P0.as_ptr() as *const __m512i);
        let idx_p1 = _mm512_loadu_si512(IDX_PM_P1.as_ptr() as *const __m512i);
        let idx_p2 = _mm512_loadu_si512(IDX_PM_P2.as_ptr() as *const __m512i);
        let blk_base = blk_quad * 4;
        for row in 0..row_count {
            let src_row = row_count - 1 - row;
            let off = src_row * row_stride + 8 * blk_base;
            let sa = a.as_ptr().add(off) as *const __m512i;
            let sb = b.as_ptr().add(off) as *const __m512i;
            deinterleave_quad_pm(
                _mm512_add_epi64(_mm512_loadu_si512(sa), _mm512_loadu_si512(sb)),
                _mm512_add_epi64(_mm512_loadu_si512(sa.add(1)), _mm512_loadu_si512(sb.add(1))),
                _mm512_add_epi64(_mm512_loadu_si512(sa.add(2)), _mm512_loadu_si512(sb.add(2))),
                _mm512_add_epi64(_mm512_loadu_si512(sa.add(3)), _mm512_loadu_si512(sb.add(3))),
                dst.as_mut_ptr().add(row * 8),
                plane_stride,
                idx_p0,
                idx_p1,
                idx_p2,
            );
        }
    }
}

/// Prime-major inner product + interleaved write for one output limb of a block-quad.
#[target_feature(enable = "avx512ifma,avx512vl")]
#[allow(clippy::too_many_arguments)]
unsafe fn pm_inner_product_and_write(
    pc: &[PrimeConsts512; 3],
    ell: usize,
    a_pm: &[u64],
    b_pm: &[u64],
    a_start: usize,
    b_start: usize,
    res_u64: &mut [u64],
    blk_quad: usize,
) {
    let plane_stride = a_pm.len() / 3;
    unsafe {
        let mut red = [_mm512_setzero_si512(); 3];
        for p in 0..3 {
            let mut acc_lo = _mm512_setzero_si512();
            let mut acc_hi = _mm512_setzero_si512();
            let x_base = a_pm.as_ptr().add(p * plane_stride + 8 * a_start) as *const __m512i;
            let y_base = b_pm.as_ptr().add(p * plane_stride + 8 * b_start) as *const __m512i;
            for r in 0..ell {
                let xv = _mm512_loadu_si512(x_base.add(r));
                let yv = _mm512_loadu_si512(y_base.add(r));
                acc_lo = _mm512_madd52lo_epu64(acc_lo, xv, yv);
                acc_hi = _mm512_madd52hi_epu64(acc_hi, xv, yv);
            }
            red[p] = reduce_bbc_single_prime_512(acc_lo, acc_hi, pc[p].q, pc[p].q2, pc[p].pow40, pc[p].pow52, pc[p].pow52_quot);
        }

        let mut p0v = [0u64; 8];
        let mut p1v = [0u64; 8];
        let mut p2v = [0u64; 8];
        _mm512_storeu_si512(p0v.as_mut_ptr() as *mut __m512i, red[0]);
        _mm512_storeu_si512(p1v.as_mut_ptr() as *mut __m512i, red[1]);
        _mm512_storeu_si512(p2v.as_mut_ptr() as *mut __m512i, red[2]);

        for i in 0..4usize {
            let blk = blk_quad * 4 + i;
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
            _mm512_stream_si512(
                res_u64.as_mut_ptr().add(8 * blk) as *mut __m512i,
                _mm512_loadu_si512(buf.as_ptr() as *const __m512i),
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Scratch accounting
// ─────────────────────────────────────────────────────────────────────────────

/// Scratch bytes required by [`cnv_apply_dft_ifma`].
pub(crate) fn cnv_apply_dft_ifma_tmp_bytes(a_size: usize, b_size: usize) -> usize {
    // 3 prime planes × 8 u64 per row for each operand
    3 * 8 * (a_size + b_size) * size_of::<u64>()
}

/// Scratch bytes required by [`cnv_pairwise_apply_dft_ifma`].
///
/// Same requirement as the non-pairwise variant since the pairwise path
/// delegates to it when `col_0 == col_1`.
pub(crate) fn cnv_pairwise_apply_dft_ifma_tmp_bytes(res_size: usize, a_size: usize, b_size: usize) -> usize {
    if a_size == 0 || b_size == 0 || res_size == 0 {
        0
    } else {
        cnv_apply_dft_ifma_tmp_bytes(a_size, b_size)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// cnv_apply_dft
// ─────────────────────────────────────────────────────────────────────────────

/// DFT-domain bivariate convolution `res[k] = Σ a[j] ⊙ b[k−j]` for the IFMA
/// backend.
///
/// Iterates over x2 NTT blocks outermost: for each block the left and right
/// rows are packed once into contiguous scratch buffers, then each output
/// limb `k` consumes a contiguous `ell`-row window to compute the BBC
/// accumulation via [`vec_mat1col_product_x2_bbc_ifma`].
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512ifma,avx512vl")]
pub(crate) unsafe fn cnv_apply_dft_ifma<R, A, B>(
    res: &mut R,
    cnv_offset: usize,
    res_col: usize,
    a: &A,
    a_col: usize,
    b: &B,
    b_col: usize,
    tmp: &mut [u8],
) where
    R: VecZnxDftToMut<NTTIfma>,
    A: CnvPVecLToRef<NTTIfma>,
    B: CnvPVecRToRef<NTTIfma>,
{
    let mut res = res.to_mut();
    let a = a.to_ref();
    let b = b.to_ref();

    let n = res.n();
    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.size();
    if res_size == 0 || a_size == 0 || b_size == 0 {
        for j in 0..res_size {
            cast_slice_mut::<_, u64>(res.at_mut(res_col, j)).fill(0);
        }
        return;
    }

    let bound = a_size + b_size - 1;
    let offset = cnv_offset.min(bound);
    let min_size = res_size.min((bound + 1).saturating_sub(offset));

    let a_cols = a.cols();
    let b_cols = b.cols();
    let n_blks = n / 2;
    let row_stride_a = 4 * n * a_cols;
    let row_stride_b = 4 * n * b_cols;
    let a_col_offset = 4 * n * a_col;
    let b_col_offset = 4 * n * b_col;
    let a_raw_u64: &[u64] = cast_slice(a.raw());
    let b_raw_u64: &[u64] = cast_slice(b.raw());

    let pc = unsafe { [PrimeConsts512::new(0), PrimeConsts512::new(1), PrimeConsts512::new(2)] };

    let (prefix, tmp_u64, suffix) = unsafe { tmp.align_to_mut::<u64>() };
    debug_assert!(prefix.is_empty());
    debug_assert!(suffix.is_empty());
    debug_assert!(tmp_u64.len() >= 3 * 8 * (a_size + b_size));
    let (a_tmp, b_tmp) = tmp_u64.split_at_mut(3 * 8 * a_size);

    let n_blk_quads = n_blks / 4;

    for bq in 0..n_blk_quads {
        unsafe {
            pack_left_quad_pm(a_tmp, &a_raw_u64[a_col_offset..], a_size, row_stride_a, bq);
            pack_right_quad_pm(b_tmp, &b_raw_u64[b_col_offset..], b_size, row_stride_b, bq);
        }

        for k in 0..min_size {
            let k_abs = k + offset;
            let j_min = k_abs.saturating_sub(a_size - 1);
            let j_max = (k_abs + 1).min(b_size);
            let ell = j_max - j_min;
            let a_start = k_abs + 1 - j_max;
            let b_start = b_size - j_max;

            let res_u64: &mut [u64] = cast_slice_mut(res.at_mut(res_col, k));
            unsafe {
                pm_inner_product_and_write(&pc, ell, a_tmp, b_tmp, a_start, b_start, res_u64, bq);
            }
        }
    }

    for j in min_size..res_size {
        res.at_mut(res_col, j).fill(Q120bScalar([0; 4]));
    }
    _mm_sfence();
}

// ─────────────────────────────────────────────────────────────────────────────
// cnv_pairwise_apply_dft
// ─────────────────────────────────────────────────────────────────────────────

/// Pairwise DFT-domain convolution:
///
/// ```text
/// res[k] = Σ_{j=j_min..j_max}
///             (a[col_0, k−j] + a[col_1, k−j])
///           ⊙ (b[col_0, j]   + b[col_1, j])
/// ```
///
/// When `col_0 == col_1`, delegates to [`cnv_apply_dft_ifma`].
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512ifma,avx512vl")]
pub(crate) unsafe fn cnv_pairwise_apply_dft_ifma<R, A, B>(
    res: &mut R,
    cnv_offset: usize,
    res_col: usize,
    a: &A,
    b: &B,
    col_0: usize,
    col_1: usize,
    tmp: &mut [u8],
) where
    R: VecZnxDftToMut<NTTIfma>,
    A: CnvPVecLToRef<NTTIfma>,
    B: CnvPVecRToRef<NTTIfma>,
{
    if col_0 == col_1 {
        unsafe { cnv_apply_dft_ifma(res, cnv_offset, res_col, a, col_0, b, col_1, tmp) };
        return;
    }

    let mut res = res.to_mut();
    let a = a.to_ref();
    let b = b.to_ref();

    let n = res.n();
    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.size();
    if res_size == 0 || a_size == 0 || b_size == 0 {
        for j in 0..res_size {
            cast_slice_mut::<_, u64>(res.at_mut(res_col, j)).fill(0);
        }
        return;
    }

    let bound = a_size + b_size - 1;
    let offset = cnv_offset.min(bound);
    let min_size = res_size.min((bound + 1).saturating_sub(offset));

    let a_cols = a.cols();
    let b_cols = b.cols();
    let n_blks = n / 2;
    let row_stride_a = 4 * n * a_cols;
    let row_stride_b = 4 * n * b_cols;
    let a_col_offset_0 = 4 * n * col_0;
    let a_col_offset_1 = 4 * n * col_1;
    let b_col_offset_0 = 4 * n * col_0;
    let b_col_offset_1 = 4 * n * col_1;
    let a_raw_u64: &[u64] = cast_slice(a.raw());
    let b_raw_u64: &[u64] = cast_slice(b.raw());

    let pc = unsafe { [PrimeConsts512::new(0), PrimeConsts512::new(1), PrimeConsts512::new(2)] };

    let (prefix, tmp_u64, suffix) = unsafe { tmp.align_to_mut::<u64>() };
    debug_assert!(prefix.is_empty());
    debug_assert!(suffix.is_empty());
    debug_assert!(tmp_u64.len() >= 3 * 8 * (a_size + b_size));
    let (a_tmp, b_tmp) = tmp_u64.split_at_mut(3 * 8 * a_size);

    let n_blk_quads = n_blks / 4;

    for bq in 0..n_blk_quads {
        unsafe {
            pairwise_pack_left_quad_pm(
                a_tmp,
                &a_raw_u64[a_col_offset_0..],
                &a_raw_u64[a_col_offset_1..],
                a_size,
                row_stride_a,
                bq,
            );
            pairwise_pack_right_quad_pm(
                b_tmp,
                &b_raw_u64[b_col_offset_0..],
                &b_raw_u64[b_col_offset_1..],
                b_size,
                row_stride_b,
                bq,
            );
        }

        for k in 0..min_size {
            let k_abs = k + offset;
            let j_min = k_abs.saturating_sub(a_size - 1);
            let j_max = (k_abs + 1).min(b_size);
            let ell = j_max - j_min;
            let a_start = k_abs + 1 - j_max;
            let b_start = b_size - j_max;

            let res_u64: &mut [u64] = cast_slice_mut(res.at_mut(res_col, k));
            unsafe {
                pm_inner_product_and_write(&pc, ell, a_tmp, b_tmp, a_start, b_start, res_u64, bq);
            }
        }
    }

    for j in min_size..res_size {
        res.at_mut(res_col, j).fill(Q120bScalar([0; 4]));
    }
    _mm_sfence();
}
