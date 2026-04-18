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

use poulpy_cpu_ref::reference::ntt_ifma::{mat_vec::BbcIfmaMeta, primes::Primes40};
use poulpy_cpu_ref::reference::ntt120::types::Q120bScalar;
use poulpy_hal::layouts::{CnvPVecLToRef, CnvPVecRToRef, VecZnxDftToMut, ZnxInfos, ZnxView, ZnxViewMut};

use super::mat_vec_ifma::vec_mat1col_product_x2_bbc_ifma;

use crate::NTTIfma;
use core::arch::x86_64::{__m512i, _mm_sfence, _mm512_add_epi64, _mm512_loadu_si512, _mm512_storeu_si512};

// ─────────────────────────────────────────────────────────────────────────────
// Pack kernels
// ─────────────────────────────────────────────────────────────────────────────
//
// In IFMA layout each NTT coefficient is 4 × u64 (3 active primes + 1 padding
// lane), so one x2-block (two consecutive coefficients) is 8 u64 = one
// `__m512i`. Packing therefore reduces to copying one `__m512i` per row, with
// optional row reversal or pairwise summation.

/// Gather a row range of q120b x2-blocks into a contiguous buffer.
///
/// `a` is a column-start q120b slice with row stride `row_stride` (in `u64`
/// units). For each row, block `blk` (8 u64 values) is copied to `dst`.
/// `dst` must hold at least `8 * row_count` u64.
#[target_feature(enable = "avx512f")]
unsafe fn pack_left_1blk_x2_ifma(dst: &mut [u64], a: &[u64], row_count: usize, row_stride: usize, blk: usize) {
    debug_assert!(dst.len() >= 8 * row_count);
    debug_assert!(a.len() >= row_stride.saturating_mul(row_count.saturating_sub(1)) + 8 * blk + 8);
    unsafe {
        let mut dst_ptr = dst.as_mut_ptr() as *mut __m512i;
        let mut a_ptr = a.as_ptr().add(8 * blk) as *const __m512i;
        for _ in 0..row_count {
            _mm512_storeu_si512(dst_ptr, _mm512_loadu_si512(a_ptr));
            a_ptr = (a_ptr as *const u64).add(row_stride) as *const __m512i;
            dst_ptr = dst_ptr.add(1);
        }
    }
}

/// Gather a row range of q120b x2-blocks in reversed row order.
///
/// Same layout as [`pack_left_1blk_x2_ifma`] but row 0 in `dst` receives the
/// source's last row. This lets each output limb consume a contiguous window
/// `[b_size - j_max ..]` inside the packed buffer.
#[target_feature(enable = "avx512f")]
unsafe fn pack_right_1blk_x2_ifma(dst: &mut [u64], a: &[u64], row_count: usize, row_stride: usize, blk: usize) {
    debug_assert!(dst.len() >= 8 * row_count);
    debug_assert!(a.len() >= row_stride.saturating_mul(row_count.saturating_sub(1)) + 8 * blk + 8);
    unsafe {
        let mut dst_ptr = dst.as_mut_ptr() as *mut __m512i;
        let mut a_ptr = a.as_ptr().add(row_stride * row_count.saturating_sub(1) + 8 * blk) as *const __m512i;
        for _ in 0..row_count {
            _mm512_storeu_si512(dst_ptr, _mm512_loadu_si512(a_ptr));
            a_ptr = (a_ptr as *const u64).sub(row_stride) as *const __m512i;
            dst_ptr = dst_ptr.add(1);
        }
    }
}

/// Pairwise pack: gather and lane-add the x2-blocks of two columns.
///
/// Inputs are in `[0, 2Q)` (left side), so the sum is in `[0, 4Q) < 2^42`,
/// which stays inside the 52-bit VPMADD52 input window.
#[target_feature(enable = "avx512f")]
unsafe fn pairwise_pack_left_1blk_x2_ifma(
    dst: &mut [u64],
    a: &[u64],
    b: &[u64],
    row_count: usize,
    row_stride: usize,
    blk: usize,
) {
    debug_assert!(dst.len() >= 8 * row_count);
    debug_assert!(a.len() >= row_stride.saturating_mul(row_count.saturating_sub(1)) + 8 * blk + 8);
    debug_assert!(b.len() >= row_stride.saturating_mul(row_count.saturating_sub(1)) + 8 * blk + 8);
    unsafe {
        let mut dst_ptr = dst.as_mut_ptr() as *mut __m512i;
        let mut a_ptr = a.as_ptr().add(8 * blk) as *const __m512i;
        let mut b_ptr = b.as_ptr().add(8 * blk) as *const __m512i;
        for _ in 0..row_count {
            _mm512_storeu_si512(
                dst_ptr,
                _mm512_add_epi64(_mm512_loadu_si512(a_ptr), _mm512_loadu_si512(b_ptr)),
            );
            a_ptr = (a_ptr as *const u64).add(row_stride) as *const __m512i;
            b_ptr = (b_ptr as *const u64).add(row_stride) as *const __m512i;
            dst_ptr = dst_ptr.add(1);
        }
    }
}

/// Pairwise pack in reversed row order. Right-side inputs are in `[0, Q)`,
/// so the sum is in `[0, 2Q) < 2^41`, well within the madd52 window.
#[target_feature(enable = "avx512f")]
unsafe fn pairwise_pack_right_1blk_x2_ifma(
    dst: &mut [u64],
    a: &[u64],
    b: &[u64],
    row_count: usize,
    row_stride: usize,
    blk: usize,
) {
    debug_assert!(dst.len() >= 8 * row_count);
    debug_assert!(a.len() >= row_stride.saturating_mul(row_count.saturating_sub(1)) + 8 * blk + 8);
    debug_assert!(b.len() >= row_stride.saturating_mul(row_count.saturating_sub(1)) + 8 * blk + 8);
    unsafe {
        let mut dst_ptr = dst.as_mut_ptr() as *mut __m512i;
        let mut a_ptr = a.as_ptr().add(row_stride * row_count.saturating_sub(1) + 8 * blk) as *const __m512i;
        let mut b_ptr = b.as_ptr().add(row_stride * row_count.saturating_sub(1) + 8 * blk) as *const __m512i;
        for _ in 0..row_count {
            _mm512_storeu_si512(
                dst_ptr,
                _mm512_add_epi64(_mm512_loadu_si512(a_ptr), _mm512_loadu_si512(b_ptr)),
            );
            a_ptr = (a_ptr as *const u64).sub(row_stride) as *const __m512i;
            b_ptr = (b_ptr as *const u64).sub(row_stride) as *const __m512i;
            dst_ptr = dst_ptr.add(1);
        }
    }
}

/// Lane-wise u64 add on two already-packed x2-block buffers (one `__m512i`
/// per row).  Used by the fused rank-1 tensor kernel to form the pairwise
/// sum `a_tmp_sum = a_tmp_0 + a_tmp_1` in L1 without re-reading `a_prep`.
#[target_feature(enable = "avx512f")]
unsafe fn add_tmp_u64_avx512(dst: &mut [u64], lhs: &[u64], rhs: &[u64]) {
    debug_assert_eq!(dst.len(), lhs.len());
    debug_assert_eq!(dst.len(), rhs.len());
    let n = dst.len() / 8;
    let dst_ptr = dst.as_mut_ptr() as *mut __m512i;
    let lhs_ptr = lhs.as_ptr() as *const __m512i;
    let rhs_ptr = rhs.as_ptr() as *const __m512i;
    unsafe {
        for i in 0..n {
            let a = _mm512_loadu_si512(lhs_ptr.add(i));
            let b = _mm512_loadu_si512(rhs_ptr.add(i));
            _mm512_storeu_si512(dst_ptr.add(i), _mm512_add_epi64(a, b));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Scratch accounting
// ─────────────────────────────────────────────────────────────────────────────

/// Scratch bytes required by [`cnv_apply_dft_ifma`].
///
/// Stores packed x2-block rows for both operands: 8 u64 per row.
pub(crate) fn cnv_apply_dft_ifma_tmp_bytes(a_size: usize, b_size: usize) -> usize {
    8 * (a_size + b_size) * size_of::<u64>()
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

    let meta = BbcIfmaMeta::<Primes40>::new();
    let a_cols = a.cols();
    let b_cols = b.cols();
    let n_blks = n / 2;
    let row_stride_a = 4 * n * a_cols;
    let row_stride_b = 4 * n * b_cols;
    let a_col_offset = 4 * n * a_col;
    let b_col_offset = 4 * n * b_col;
    let a_raw_u64: &[u64] = cast_slice(a.raw());
    let b_raw_u64: &[u64] = cast_slice(b.raw());

    let (prefix, tmp_u64, suffix) = unsafe { tmp.align_to_mut::<u64>() };
    debug_assert!(prefix.is_empty());
    debug_assert!(suffix.is_empty());
    debug_assert!(tmp_u64.len() >= 8 * (a_size + b_size));
    let (a_tmp, b_tmp) = tmp_u64.split_at_mut(8 * a_size);

    for blk in 0..n_blks {
        unsafe {
            pack_left_1blk_x2_ifma(a_tmp, &a_raw_u64[a_col_offset..], a_size, row_stride_a, blk);
            pack_right_1blk_x2_ifma(b_tmp, &b_raw_u64[b_col_offset..], b_size, row_stride_b, blk);
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
                vec_mat1col_product_x2_bbc_ifma::<true>(
                    &meta,
                    ell,
                    &mut res_u64[8 * blk..8 * blk + 8],
                    cast_slice(&a_tmp[8 * a_start..]),
                    cast_slice(&b_tmp[8 * b_start..]),
                );
            }
        }
    }

    for j in min_size..res_size {
        res.at_mut(res_col, j).fill(Q120bScalar([0; 4]));
    }
    // Order the non-temporal stores from the kernel against any subsequent
    // load of `res` (e.g. by the next stage of the FHE pipeline).
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

    let meta = BbcIfmaMeta::<Primes40>::new();
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

    let (prefix, tmp_u64, suffix) = unsafe { tmp.align_to_mut::<u64>() };
    debug_assert!(prefix.is_empty());
    debug_assert!(suffix.is_empty());
    debug_assert!(tmp_u64.len() >= 8 * (a_size + b_size));
    let (a_tmp, b_tmp) = tmp_u64.split_at_mut(8 * a_size);

    for blk in 0..n_blks {
        unsafe {
            pairwise_pack_left_1blk_x2_ifma(
                a_tmp,
                &a_raw_u64[a_col_offset_0..],
                &a_raw_u64[a_col_offset_1..],
                a_size,
                row_stride_a,
                blk,
            );
            pairwise_pack_right_1blk_x2_ifma(
                b_tmp,
                &b_raw_u64[b_col_offset_0..],
                &b_raw_u64[b_col_offset_1..],
                b_size,
                row_stride_b,
                blk,
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
                vec_mat1col_product_x2_bbc_ifma::<true>(
                    &meta,
                    ell,
                    &mut res_u64[8 * blk..8 * blk + 8],
                    cast_slice(&a_tmp[8 * a_start..]),
                    cast_slice(&b_tmp[8 * b_start..]),
                );
            }
        }
    }

    for j in min_size..res_size {
        res.at_mut(res_col, j).fill(Q120bScalar([0; 4]));
    }
    _mm_sfence();
}

// ─────────────────────────────────────────────────────────────────────────────
// Fused rank-1 tensor convolution (3 outputs, 1 X-block sweep)
// ─────────────────────────────────────────────────────────────────────────────

/// Scratch bytes for [`cnv_apply_dft_rank1_fused_ifma`].
///
/// Holds six packed buffers per X-block (left col-0, left col-1, left sum,
/// right col-0, right col-1, right sum).  All fit in L1.
pub(crate) fn cnv_apply_dft_rank1_fused_ifma_tmp_bytes(a_size: usize, b_size: usize) -> usize {
    use std::mem::size_of;
    8 * (3 * a_size + 3 * b_size) * size_of::<u64>()
}

/// DFT-domain rank-1 tensor convolution — emits **three** output polynomials
/// (the two diagonals and the pairwise cross-term) in a **single** sweep
/// over `a_prep` / `b_prep`.
///
/// Classical sequential tensor_apply reads `a_prep` (80 MiB for N=2^16, L=20)
/// and `b_prep` (40 MiB) once per convolution (three times total) — ~240 MiB
/// of prep-data reads per tensor.  This kernel folds the three convs into one
/// X-block-major pass and reads each prep entry exactly once → ~120 MiB, about
/// half the DRAM-read pressure of the baseline.
///
/// # Semantics
///
/// For each X-block `blk` and each output limb `k ∈ [0, min_size)`:
///
/// ```text
///   res_diag_0[k][blk] = Σ_j a[0][j][blk] * b[0][k-j][blk]
///   res_diag_1[k][blk] = Σ_j a[1][j][blk] * b[1][k-j][blk]
///   res_pair[k][blk]   = Σ_j (a[0][j][blk] + a[1][j][blk])
///                          · (b[0][k-j][blk] + b[1][k-j][blk])
/// ```
///
/// Callers should subtract `res_diag_0 + res_diag_1` from `res_pair` at the
/// core level (the same pre-subtract `glwe_tensor_apply` already does in the
/// sequential path).
///
/// # Safety
///
/// - `a.cols() >= 2`, `b.cols() >= 2` (rank-1 tensor input).
/// - `tmp.len() >= cnv_apply_dft_rank1_fused_ifma_tmp_bytes(a_size, b_size)`.
/// - Caller must issue `_mm_sfence` **after** this call — the kernel stores
///   to all three result buffers via `_mm512_stream_si512`.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512ifma,avx512vl")]
pub(crate) unsafe fn cnv_apply_dft_rank1_fused_ifma<R0, R1, RP, A, B>(
    res_diag_0: &mut R0,
    res_diag_1: &mut R1,
    res_pair: &mut RP,
    cnv_offset: usize,
    a: &A,
    b: &B,
    tmp: &mut [u8],
) where
    R0: VecZnxDftToMut<NTTIfma>,
    R1: VecZnxDftToMut<NTTIfma>,
    RP: VecZnxDftToMut<NTTIfma>,
    A: CnvPVecLToRef<NTTIfma>,
    B: CnvPVecRToRef<NTTIfma>,
{
    let mut r0 = res_diag_0.to_mut();
    let mut r1 = res_diag_1.to_mut();
    let mut rp = res_pair.to_mut();
    let a = a.to_ref();
    let b = b.to_ref();

    let n = r0.n();
    debug_assert_eq!(r1.n(), n);
    debug_assert_eq!(rp.n(), n);
    debug_assert!(a.cols() >= 2);
    debug_assert!(b.cols() >= 2);

    let res_size = r0.size().min(r1.size()).min(rp.size());
    let a_size = a.size();
    let b_size = b.size();

    if res_size == 0 || a_size == 0 || b_size == 0 {
        for j in 0..r0.size() {
            cast_slice_mut::<_, u64>(r0.at_mut(0, j)).fill(0);
        }
        for j in 0..r1.size() {
            cast_slice_mut::<_, u64>(r1.at_mut(0, j)).fill(0);
        }
        for j in 0..rp.size() {
            cast_slice_mut::<_, u64>(rp.at_mut(0, j)).fill(0);
        }
        return;
    }

    let bound = a_size + b_size - 1;
    let offset = cnv_offset.min(bound);
    let min_size = res_size.min((bound + 1).saturating_sub(offset));

    let meta = BbcIfmaMeta::<Primes40>::new();
    let a_cols = a.cols();
    let b_cols = b.cols();
    let n_blks = n / 2;
    let row_stride_a = 4 * n * a_cols;
    let row_stride_b = 4 * n * b_cols;
    let a_col_offset_0 = 0;
    let a_col_offset_1 = 4 * n;
    let b_col_offset_0 = 0;
    let b_col_offset_1 = 4 * n;
    let a_raw_u64: &[u64] = cast_slice(a.raw());
    let b_raw_u64: &[u64] = cast_slice(b.raw());

    let (prefix, tmp_u64, suffix) = unsafe { tmp.align_to_mut::<u64>() };
    debug_assert!(prefix.is_empty());
    debug_assert!(suffix.is_empty());
    debug_assert!(tmp_u64.len() >= 8 * (3 * a_size + 3 * b_size));
    // Layout: [a_tmp_0 | a_tmp_1 | a_tmp_sum | b_tmp_0 | b_tmp_1 | b_tmp_sum]
    let (a_section, b_section) = tmp_u64.split_at_mut(8 * 3 * a_size);
    let (a_tmp_0, a_rest) = a_section.split_at_mut(8 * a_size);
    let (a_tmp_1, a_tmp_sum) = a_rest.split_at_mut(8 * a_size);
    let (b_tmp_0, b_rest) = b_section.split_at_mut(8 * b_size);
    let (b_tmp_1, b_tmp_sum) = b_rest.split_at_mut(8 * b_size);

    for blk in 0..n_blks {
        unsafe {
            // Left side: two strided packs (col 0, col 1) from `a_raw` —
            // touching each input cache line exactly once — then compute
            // `a_tmp_sum = a_tmp_0 + a_tmp_1` in L1 (not re-reading prep).
            pack_left_1blk_x2_ifma(a_tmp_0, &a_raw_u64[a_col_offset_0..], a_size, row_stride_a, blk);
            pack_left_1blk_x2_ifma(a_tmp_1, &a_raw_u64[a_col_offset_1..], a_size, row_stride_a, blk);
            add_tmp_u64_avx512(a_tmp_sum, a_tmp_0, a_tmp_1);

            // Right side: same pattern with row-reversed packs.
            pack_right_1blk_x2_ifma(b_tmp_0, &b_raw_u64[b_col_offset_0..], b_size, row_stride_b, blk);
            pack_right_1blk_x2_ifma(b_tmp_1, &b_raw_u64[b_col_offset_1..], b_size, row_stride_b, blk);
            add_tmp_u64_avx512(b_tmp_sum, b_tmp_0, b_tmp_1);
        }

        for k in 0..min_size {
            let k_abs = k + offset;
            let j_min = k_abs.saturating_sub(a_size - 1);
            let j_max = (k_abs + 1).min(b_size);
            let ell = j_max - j_min;
            let a_start = k_abs + 1 - j_max;
            let b_start = b_size - j_max;

            let res0: &mut [u64] = cast_slice_mut(r0.at_mut(0, k));
            let res1: &mut [u64] = cast_slice_mut(r1.at_mut(0, k));
            let resp: &mut [u64] = cast_slice_mut(rp.at_mut(0, k));
            unsafe {
                vec_mat1col_product_x2_bbc_ifma::<true>(
                    &meta,
                    ell,
                    &mut res0[8 * blk..8 * blk + 8],
                    cast_slice(&a_tmp_0[8 * a_start..]),
                    cast_slice(&b_tmp_0[8 * b_start..]),
                );
                vec_mat1col_product_x2_bbc_ifma::<true>(
                    &meta,
                    ell,
                    &mut res1[8 * blk..8 * blk + 8],
                    cast_slice(&a_tmp_1[8 * a_start..]),
                    cast_slice(&b_tmp_1[8 * b_start..]),
                );
                vec_mat1col_product_x2_bbc_ifma::<true>(
                    &meta,
                    ell,
                    &mut resp[8 * blk..8 * blk + 8],
                    cast_slice(&a_tmp_sum[8 * a_start..]),
                    cast_slice(&b_tmp_sum[8 * b_start..]),
                );
            }
        }
    }

    // Zero the tail limbs of each output past min_size.
    for j in min_size..r0.size() {
        r0.at_mut(0, j).fill(Q120bScalar([0; 4]));
    }
    for j in min_size..r1.size() {
        r1.at_mut(0, j).fill(Q120bScalar([0; 4]));
    }
    for j in min_size..rp.size() {
        rp.at_mut(0, j).fill(Q120bScalar([0; 4]));
    }
    _mm_sfence();
}
