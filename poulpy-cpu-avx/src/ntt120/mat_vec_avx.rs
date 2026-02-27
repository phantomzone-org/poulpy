// ----------------------------------------------------------------------
// DISCLAIMER
//
// This module contains code that has been directly ported from the
// spqlios-arithmetic library
// (https://github.com/tfhe/spqlios-arithmetic), which is licensed
// under the Apache License, Version 2.0.
//
// The porting process from C to Rust was done with minimal changes
// in order to preserve the semantics and performance characteristics
// of the original implementation.
//
// Both Poulpy and spqlios-arithmetic are distributed under the terms
// of the Apache License, Version 2.0. See the LICENSE file for details.
//
// ----------------------------------------------------------------------

//! AVX2-accelerated Q120 matrix-vector dot products.
//!
//! Rust ports of the BBC (q120b × q120c → q120b) inner-product functions
//! from `q120_arithmetic_avx2.c` in spqlios-arithmetic.
//!
//! # Layout conventions
//!
//! | Format | Bytes/element | AVX2 view |
//! |--------|--------------|-----------|
//! | q120b  | 32 (4 × u64) | one `__m256i` |
//! | q120c  | 32 (8 × u32) | one `__m256i` |
//! | x2-block | 64 (2 × q120b/c) | two `__m256i`s |
//!
//! All three exported functions share the same two-accumulator strategy:
//! accumulate low-32-bit and high-32-bit partial products separately,
//! then collapse with the precomputed `s2l_pow_red` / `s2h_pow_red`
//! constants from [`BbcMeta`].
//!
//! # Functions
//!
//! | AVX2 function | spqlios C equivalent | Trait |
//! |---|---|---|
//! | [`vec_mat1col_product_bbc_avx2`] | `q120_vec_mat1col_product_bbc_avx2` | [`NttMulBbc`] |
//! | [`vec_mat1col_product_x2_bbc_avx2`] | `q120x2_vec_mat1col_product_bbc_avx2` | [`NttMulBbc1ColX2`] |
//! | [`vec_mat2cols_product_x2_bbc_avx2`] | `q120x2_vec_mat2cols_product_bbc_avx2` | [`NttMulBbc2ColsX2`] |

use core::arch::x86_64::{
    __m256i, _mm_cvtsi64_si128, _mm256_add_epi64, _mm256_and_si256, _mm256_loadu_si256, _mm256_mul_epu32, _mm256_set1_epi64x,
    _mm256_setzero_si256, _mm256_srl_epi64, _mm256_srli_epi64, _mm256_storeu_si256,
};

use poulpy_hal::reference::ntt120::{mat_vec::BbcMeta, primes::Primes30};

// ─────────────────────────────────────────────────────────────────────────────
// Shared final-reduction helper
// ─────────────────────────────────────────────────────────────────────────────

/// Collapse two accumulator vectors `(s_lo, s_hi)` into a q120b result.
///
/// Computes `res = s_lo + (s_hi & mask_h2) * S2L + (s_hi >> H2) * S2H`,
/// matching the final-reduction step shared by all three BBC functions.
#[inline(always)]
unsafe fn reduce_bbc(s_lo: __m256i, s_hi: __m256i, mask_h2: __m256i, h2: u64, s2l: __m256i, s2h: __m256i) -> __m256i {
    unsafe {
        let h2_count = _mm_cvtsi64_si128(h2 as i64);
        let hi_lo = _mm256_and_si256(s_hi, mask_h2);
        let hi_hi = _mm256_srl_epi64(s_hi, h2_count);
        let t = _mm256_add_epi64(s_lo, _mm256_mul_epu32(hi_lo, s2l));
        _mm256_add_epi64(t, _mm256_mul_epu32(hi_hi, s2h))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Single-column: q120b × q120c → q120b
// ─────────────────────────────────────────────────────────────────────────────

/// AVX2 inner product: `res = Σᵢ x[i] · y[i]` in q120b format.
///
/// Port of `q120_vec_mat1col_product_bbc_avx2`.
///
/// - `x`: q120b in u32 view — `ell` elements × 8 u32 (one `__m256i` each).
/// - `y`: q120c — `ell` elements × 8 u32 (one `__m256i` each).
/// - `res`: q120b output — at least 4 u64 (one `__m256i`).
///
/// # Safety
///
/// Caller must ensure AVX2 support. Slice lengths must satisfy
/// `x.len() >= 8 * ell`, `y.len() >= 8 * ell`, `res.len() >= 4`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn vec_mat1col_product_bbc_avx2(meta: &BbcMeta<Primes30>, ell: usize, res: &mut [u64], x: &[u32], y: &[u32]) {
    unsafe {
        let mask32 = _mm256_set1_epi64x(u32::MAX as i64);
        let mut s1 = _mm256_setzero_si256(); // accumulator for low 32-bit partial products
        let mut s2 = _mm256_setzero_si256(); // accumulator for high 32-bit partial products

        let mut x_ptr = x.as_ptr() as *const __m256i;
        let mut y_ptr = y.as_ptr() as *const __m256i;

        for _ in 0..ell {
            let xv = _mm256_loadu_si256(x_ptr);
            let xl = _mm256_and_si256(xv, mask32); // low 32 bits of each u64 lane
            let xh = _mm256_srli_epi64::<32>(xv); // high 32 bits shifted down

            let yv = _mm256_loadu_si256(y_ptr);
            let y0 = _mm256_and_si256(yv, mask32); // y_i mod 2^32 (= r_i mod Q)
            let y1 = _mm256_srli_epi64::<32>(yv); // y_i >> 32 (= (r_i * 2^32) mod Q)

            // a = xl * y0  (xl and y0 are both 32-bit, result ≤ 64 bits)
            let a = _mm256_mul_epu32(xl, y0);
            // b = xh * y1
            let b = _mm256_mul_epu32(xh, y1);

            s1 = _mm256_add_epi64(s1, _mm256_and_si256(a, mask32));
            s1 = _mm256_add_epi64(s1, _mm256_and_si256(b, mask32));
            s2 = _mm256_add_epi64(s2, _mm256_srli_epi64::<32>(a));
            s2 = _mm256_add_epi64(s2, _mm256_srli_epi64::<32>(b));

            x_ptr = x_ptr.add(1);
            y_ptr = y_ptr.add(1);
        }

        let mask_h2 = _mm256_set1_epi64x(((1u64 << meta.h) - 1) as i64);
        let s2l_pow_red = _mm256_loadu_si256(meta.s2l_pow_red.as_ptr() as *const __m256i);
        let s2h_pow_red = _mm256_loadu_si256(meta.s2h_pow_red.as_ptr() as *const __m256i);

        let t = reduce_bbc(s1, s2, mask_h2, meta.h, s2l_pow_red, s2h_pow_red);
        _mm256_storeu_si256(res.as_mut_ptr() as *mut __m256i, t);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// x2-block, single column: two q120b × q120c pairs → two q120b results
// ─────────────────────────────────────────────────────────────────────────────

/// AVX2 x2-block inner product: one column, two paired rows.
///
/// Port of `q120x2_vec_mat1col_product_bbc_avx2`.
///
/// Computes two q120b inner products simultaneously using two interleaved
/// q120b/q120c pairs per step:
/// - `res[0..4]` ← `Σᵢ x_a[i] · y_a[i]`
/// - `res[4..8]` ← `Σᵢ x_b[i] · y_b[i]`
///
/// - `x`: x2-block in u32 view — `ell` elements × 16 u32 (two `__m256i`s each).
/// - `y`: x2-block q120c — `ell` elements × 16 u32 (two `__m256i`s each).
/// - `res`: two q120b outputs — at least 8 u64.
///
/// # Safety
///
/// Caller must ensure AVX2 support. Slice lengths must satisfy
/// `x.len() >= 16 * ell`, `y.len() >= 16 * ell`, `res.len() >= 8`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn vec_mat1col_product_x2_bbc_avx2(
    meta: &BbcMeta<Primes30>,
    ell: usize,
    res: &mut [u64],
    x: &[u32],
    y: &[u32],
) {
    unsafe {
        let mask32 = _mm256_set1_epi64x(u32::MAX as i64);
        // Pair A accumulators
        let mut s0 = _mm256_setzero_si256(); // low sums  for x_a * y_a
        let mut s1 = _mm256_setzero_si256(); // high sums for x_a * y_a
        // Pair B accumulators
        let mut s2 = _mm256_setzero_si256(); // low sums  for x_b * y_b
        let mut s3 = _mm256_setzero_si256(); // high sums for x_b * y_b

        let mut x_ptr = x.as_ptr() as *const __m256i;
        let mut y_ptr = y.as_ptr() as *const __m256i;

        for _ in 0..ell {
            // Pair A: x[2i] × y[2i]
            let xa = _mm256_loadu_si256(x_ptr);
            let xa_hi = _mm256_srli_epi64::<32>(xa);
            let ya = _mm256_loadu_si256(y_ptr);
            let ya_hi = _mm256_srli_epi64::<32>(ya);

            let prod_a_lo = _mm256_mul_epu32(xa, ya);
            let prod_a_hi = _mm256_mul_epu32(xa_hi, ya_hi);

            s0 = _mm256_add_epi64(s0, _mm256_and_si256(prod_a_lo, mask32));
            s0 = _mm256_add_epi64(s0, _mm256_and_si256(prod_a_hi, mask32));
            s1 = _mm256_add_epi64(s1, _mm256_srli_epi64::<32>(prod_a_lo));
            s1 = _mm256_add_epi64(s1, _mm256_srli_epi64::<32>(prod_a_hi));

            // Pair B: x[2i+1] × y[2i+1]
            let xb = _mm256_loadu_si256(x_ptr.add(1));
            let xb_hi = _mm256_srli_epi64::<32>(xb);
            let yb = _mm256_loadu_si256(y_ptr.add(1));
            let yb_hi = _mm256_srli_epi64::<32>(yb);

            let prod_b_lo = _mm256_mul_epu32(xb, yb);
            let prod_b_hi = _mm256_mul_epu32(xb_hi, yb_hi);

            s2 = _mm256_add_epi64(s2, _mm256_and_si256(prod_b_lo, mask32));
            s2 = _mm256_add_epi64(s2, _mm256_and_si256(prod_b_hi, mask32));
            s3 = _mm256_add_epi64(s3, _mm256_srli_epi64::<32>(prod_b_lo));
            s3 = _mm256_add_epi64(s3, _mm256_srli_epi64::<32>(prod_b_hi));

            x_ptr = x_ptr.add(2);
            y_ptr = y_ptr.add(2);
        }

        let mask_h2 = _mm256_set1_epi64x(((1u64 << meta.h) - 1) as i64);
        let s2l_pow_red = _mm256_loadu_si256(meta.s2l_pow_red.as_ptr() as *const __m256i);
        let s2h_pow_red = _mm256_loadu_si256(meta.s2h_pow_red.as_ptr() as *const __m256i);

        let res_ptr = res.as_mut_ptr() as *mut __m256i;
        _mm256_storeu_si256(res_ptr, reduce_bbc(s0, s1, mask_h2, meta.h, s2l_pow_red, s2h_pow_red));
        _mm256_storeu_si256(res_ptr.add(1), reduce_bbc(s2, s3, mask_h2, meta.h, s2l_pow_red, s2h_pow_red));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// x2-block, two columns: two q120b × four q120c pairs → four q120b results
// ─────────────────────────────────────────────────────────────────────────────

/// AVX2 x2-block inner product: two columns simultaneously.
///
/// Port of `q120x2_vec_mat2cols_product_bbc_avx2`.
///
/// Computes four q120b inner products (two x2-block rows × two matrix columns):
/// - `res[0..4]`   ← `Σᵢ x_a[i] · y_col0_a[i]`
/// - `res[4..8]`   ← `Σᵢ x_b[i] · y_col0_b[i]`
/// - `res[8..12]`  ← `Σᵢ x_a[i] · y_col1_a[i]`
/// - `res[12..16]` ← `Σᵢ x_b[i] · y_col1_b[i]`
///
/// - `x`: x2-block in u32 view — `ell` × 16 u32 (two `__m256i`s per step).
/// - `y`: two paired x2-block q120c columns — `ell` × 32 u32 (four `__m256i`s per step):
///   `[col0_a, col0_b, col1_a, col1_b]` per element.
/// - `res`: four q120b outputs — at least 16 u64.
///
/// # Safety
///
/// Caller must ensure AVX2 support. Slice lengths must satisfy
/// `x.len() >= 16 * ell`, `y.len() >= 32 * ell`, `res.len() >= 16`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn vec_mat2cols_product_x2_bbc_avx2(
    meta: &BbcMeta<Primes30>,
    ell: usize,
    res: &mut [u64],
    x: &[u32],
    y: &[u32],
) {
    unsafe {
        let mask32 = _mm256_set1_epi64x(u32::MAX as i64);

        // col 0, pair A
        let mut s0 = _mm256_setzero_si256();
        let mut s1 = _mm256_setzero_si256();
        // col 0, pair B
        let mut s2 = _mm256_setzero_si256();
        let mut s3 = _mm256_setzero_si256();
        // col 1, pair A
        let mut s4 = _mm256_setzero_si256();
        let mut s5 = _mm256_setzero_si256();
        // col 1, pair B
        let mut s6 = _mm256_setzero_si256();
        let mut s7 = _mm256_setzero_si256();

        let mut x_ptr = x.as_ptr() as *const __m256i;
        let mut y_ptr = y.as_ptr() as *const __m256i;

        for _ in 0..ell {
            // Load x pair
            let xa = _mm256_loadu_si256(x_ptr);
            let xa_hi = _mm256_srli_epi64::<32>(xa);
            let xb = _mm256_loadu_si256(x_ptr.add(1));
            let xb_hi = _mm256_srli_epi64::<32>(xb);

            // ── Column 0 ──
            // pair A: xa × y_col0_a
            let yc0a = _mm256_loadu_si256(y_ptr);
            let yc0a_hi = _mm256_srli_epi64::<32>(yc0a);
            let p0a_lo = _mm256_mul_epu32(xa, yc0a);
            let p0a_hi = _mm256_mul_epu32(xa_hi, yc0a_hi);
            s0 = _mm256_add_epi64(s0, _mm256_and_si256(p0a_lo, mask32));
            s0 = _mm256_add_epi64(s0, _mm256_and_si256(p0a_hi, mask32));
            s1 = _mm256_add_epi64(s1, _mm256_srli_epi64::<32>(p0a_lo));
            s1 = _mm256_add_epi64(s1, _mm256_srli_epi64::<32>(p0a_hi));

            // pair B: xb × y_col0_b
            let yc0b = _mm256_loadu_si256(y_ptr.add(1));
            let yc0b_hi = _mm256_srli_epi64::<32>(yc0b);
            let p0b_lo = _mm256_mul_epu32(xb, yc0b);
            let p0b_hi = _mm256_mul_epu32(xb_hi, yc0b_hi);
            s2 = _mm256_add_epi64(s2, _mm256_and_si256(p0b_lo, mask32));
            s2 = _mm256_add_epi64(s2, _mm256_and_si256(p0b_hi, mask32));
            s3 = _mm256_add_epi64(s3, _mm256_srli_epi64::<32>(p0b_lo));
            s3 = _mm256_add_epi64(s3, _mm256_srli_epi64::<32>(p0b_hi));

            // ── Column 1 ──
            // pair A: xa × y_col1_a
            let yc1a = _mm256_loadu_si256(y_ptr.add(2));
            let yc1a_hi = _mm256_srli_epi64::<32>(yc1a);
            let p1a_lo = _mm256_mul_epu32(xa, yc1a);
            let p1a_hi = _mm256_mul_epu32(xa_hi, yc1a_hi);
            s4 = _mm256_add_epi64(s4, _mm256_and_si256(p1a_lo, mask32));
            s4 = _mm256_add_epi64(s4, _mm256_and_si256(p1a_hi, mask32));
            s5 = _mm256_add_epi64(s5, _mm256_srli_epi64::<32>(p1a_lo));
            s5 = _mm256_add_epi64(s5, _mm256_srli_epi64::<32>(p1a_hi));

            // pair B: xb × y_col1_b
            let yc1b = _mm256_loadu_si256(y_ptr.add(3));
            let yc1b_hi = _mm256_srli_epi64::<32>(yc1b);
            let p1b_lo = _mm256_mul_epu32(xb, yc1b);
            let p1b_hi = _mm256_mul_epu32(xb_hi, yc1b_hi);
            s6 = _mm256_add_epi64(s6, _mm256_and_si256(p1b_lo, mask32));
            s6 = _mm256_add_epi64(s6, _mm256_and_si256(p1b_hi, mask32));
            s7 = _mm256_add_epi64(s7, _mm256_srli_epi64::<32>(p1b_lo));
            s7 = _mm256_add_epi64(s7, _mm256_srli_epi64::<32>(p1b_hi));

            x_ptr = x_ptr.add(2);
            y_ptr = y_ptr.add(4);
        }

        let mask_h2 = _mm256_set1_epi64x(((1u64 << meta.h) - 1) as i64);
        let s2l_pow_red = _mm256_loadu_si256(meta.s2l_pow_red.as_ptr() as *const __m256i);
        let s2h_pow_red = _mm256_loadu_si256(meta.s2h_pow_red.as_ptr() as *const __m256i);

        let res_ptr = res.as_mut_ptr() as *mut __m256i;
        _mm256_storeu_si256(res_ptr, reduce_bbc(s0, s1, mask_h2, meta.h, s2l_pow_red, s2h_pow_red));
        _mm256_storeu_si256(res_ptr.add(1), reduce_bbc(s2, s3, mask_h2, meta.h, s2l_pow_red, s2h_pow_red));
        _mm256_storeu_si256(res_ptr.add(2), reduce_bbc(s4, s5, mask_h2, meta.h, s2l_pow_red, s2h_pow_red));
        _mm256_storeu_si256(res_ptr.add(3), reduce_bbc(s6, s7, mask_h2, meta.h, s2l_pow_red, s2h_pow_red));
    }
}
