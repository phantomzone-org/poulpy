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

//! AVX2-accelerated coefficient-domain Q120 arithmetic.
//!
//! Provides four kernels used by [`super::prim`]:
//!
//! | Function | Trait |
//! |---|---|
//! | [`b_from_znx64_avx2`] | `NttFromZnx64` |
//! | [`c_from_b_avx2`] | `NttCFromB` |
//! | [`vec_mat1col_product_bbb_avx2`] | `NttMulBbb` |
//! | [`b_to_znx128_avx2`] | `NttToZnx128` |
//!
//! All functions are gated on AVX2 (`#[target_feature(enable = "avx2")]`)
//! and marked `pub(crate) unsafe fn`; the caller (trait impls in `prim.rs`)
//! must have verified CPU support at module construction time.

use core::arch::x86_64::{
    __m256i, _mm_cvtsi64_si128, _mm256_add_epi64, _mm256_and_si256, _mm256_andnot_si256, _mm256_cmpgt_epi64, _mm256_loadu_si256,
    _mm256_mul_epu32, _mm256_or_si256, _mm256_set1_epi64x, _mm256_setzero_si256, _mm256_slli_epi64, _mm256_srl_epi64,
    _mm256_srli_epi64, _mm256_storeu_si256, _mm256_sub_epi64,
};

use poulpy_hal::reference::ntt120::{
    mat_vec::BbbMeta,
    primes::{PrimeSet, Primes30},
};

// ─────────────────────────────────────────────────────────────────────────────
// Primes30-specific compile-time constants
// ─────────────────────────────────────────────────────────────────────────────

/// `Q[k]` as `u64`, one per prime, for use in AVX2 lanes.
const Q_VEC: [u64; 4] = [
    Primes30::Q[0] as u64,
    Primes30::Q[1] as u64,
    Primes30::Q[2] as u64,
    Primes30::Q[3] as u64,
];

/// `oq[k] = Q[k] - (2^63 mod Q[k])`.
///
/// Used by `b_from_znx64_avx2`: for a negative input `x`, each prime lane
/// receives `(x as u64 & i64::MAX) + oq[k]`, which equals `x mod Q[k]` as u64.
const OQ: [u64; 4] = {
    let mut oq = [0u64; 4];
    let mut k = 0usize;
    while k < 4 {
        let q = Q_VEC[k];
        oq[k] = q - (i64::MIN as u64 % q); // i64::MIN as u64 = 2^63
        k += 1;
    }
    oq
};

/// Barrett multiplier: `mu[k] = floor(2^61 / Q[k])`.
///
/// Used for Barrett reduction of values `x < 2^60` mod `Q[k]`.
/// Since `Q[k] > 2^29` for Primes30, `mu[k] < 2^32` (fits in u32 / lower 32 bits of u64).
const BARRETT_MU: [u64; 4] = {
    let mut mu = [0u64; 4];
    let mut k = 0usize;
    while k < 4 {
        mu[k] = (1u64 << 61) / Q_VEC[k];
        k += 1;
    }
    mu
};

/// `pow32[k] = 2^32 mod Q[k]`.
///
/// Used in `c_from_b_avx2` and `b_to_znx128_avx2`:
/// - Combines `x_hi_r * pow32 + x_lo` to reduce a 63-bit q120b value.
/// - Computes `r_shift = r * pow32 mod Q[k]` (i.e., `r * 2^32 mod Q[k]`).
const POW32: [u64; 4] = {
    let mut p = [0u64; 4];
    let mut k = 0usize;
    while k < 4 {
        p[k] = ((1u128 << 32) % Q_VEC[k] as u128) as u64;
        k += 1;
    }
    p
};

/// `CRT_CST[k]` as u64, for `b_to_znx128_avx2`.
const CRT_VEC: [u64; 4] = [
    Primes30::CRT_CST[0] as u64,
    Primes30::CRT_CST[1] as u64,
    Primes30::CRT_CST[2] as u64,
    Primes30::CRT_CST[3] as u64,
];

// ─────────────────────────────────────────────────────────────────────────────
// AVX2 helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Single conditional subtract: `x = if x >= q { x - q } else { x }`.
///
/// Valid when both `x < 2^63` and `q < 2^63` (signed cmpgt gives correct unsigned order).
#[inline(always)]
unsafe fn cond_sub(x: __m256i, q: __m256i) -> __m256i {
    unsafe {
        // lt = all-ones in lanes where q > x (i.e., x < q — no subtract needed)
        let lt = _mm256_cmpgt_epi64(q, x);
        _mm256_sub_epi64(x, _mm256_andnot_si256(lt, q))
    }
}

/// Barrett reduction: reduce `tmp < 2^61` to `[0, Q[k])` for all four Primes30 lanes.
///
/// Uses precomputed `mu[k] = floor(2^61 / Q[k])` (stored in lower 32 bits of each u64 lane).
/// The quotient approximation may underestimate the true quotient by up to 2,
/// so two conditional subtracts bring the remainder into `[0, Q)`.
#[inline(always)]
unsafe fn barrett_reduce(tmp: __m256i, q: __m256i, mu: __m256i) -> __m256i {
    unsafe {
        let mask32 = _mm256_set1_epi64x(u32::MAX as i64);
        // Split tmp at bit 32: tmp_hi < 2^29, tmp_lo < 2^32
        let tmp_hi = _mm256_srli_epi64::<32>(tmp);
        let tmp_lo = _mm256_and_si256(tmp, mask32);
        // q_approx_hi = floor(tmp_hi * mu / 2^29)
        //   tmp_hi * mu < 2^29 * 2^32 = 2^61, fits in u64
        let q_hi = _mm256_srli_epi64::<29>(_mm256_mul_epu32(tmp_hi, mu));
        // q_approx_lo = floor(tmp_lo * mu / 2^61)
        //   tmp_lo * mu < 2^32 * 2^32 = 2^64, may overflow — clamp contribution to 0..7
        let q_lo = _mm256_srli_epi64::<61>(_mm256_mul_epu32(tmp_lo, mu));
        let q_approx = _mm256_add_epi64(q_hi, q_lo);
        // r = tmp - q_approx * Q  (q_approx < 2^31, Q < 2^30, product < 2^61)
        let r = _mm256_sub_epi64(tmp, _mm256_mul_epu32(q_approx, q));
        // r < 3*Q after the approximation; two subtracts bring it into [0, Q)
        let r = cond_sub(r, q);
        cond_sub(r, q)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// b_from_znx64_avx2
// ─────────────────────────────────────────────────────────────────────────────

/// AVX2 port of `b_from_znx64_ref`: convert `i64` coefficients to q120b.
///
/// For each coefficient `x[j]`:
/// - Strips the sign bit to get `xl = x[j] as u64 & i64::MAX`.
/// - For negative inputs, adds `oq[k]` per prime so the result is congruent to `x[j]` mod `Q[k]`.
///
/// Processes one coefficient per loop iteration, writing one `__m256i` (4 × u64) to `res`.
///
/// # Safety
///
/// Caller must ensure AVX2 support. `res.len() >= 4 * nn`, `x.len() >= nn`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn b_from_znx64_avx2(nn: usize, res: &mut [u64], x: &[i64]) {
    assert!(
        res.len() >= 4 * nn,
        "b_from_znx64_avx2: res.len()={} < 4*nn={}",
        res.len(),
        4 * nn
    );
    assert!(x.len() >= nn, "b_from_znx64_avx2: x.len()={} < nn={}", x.len(), nn);
    unsafe {
        let oq_vec = _mm256_loadu_si256(OQ.as_ptr() as *const __m256i);
        let i64_max = _mm256_set1_epi64x(i64::MAX);
        let zero = _mm256_setzero_si256();
        let mut r_ptr = res.as_mut_ptr() as *mut __m256i;

        for &xval in &x[..nn] {
            // Broadcast xval into all 4 prime lanes
            let xv = _mm256_set1_epi64x(xval);
            // Strip sign bit: xl = xval as u64 & 0x7FFF_FFFF_FFFF_FFFF
            let xl = _mm256_and_si256(xv, i64_max);
            // sign = all-ones in lanes where xval < 0
            let sign = _mm256_cmpgt_epi64(zero, xv);
            // add oq[k] only for negative inputs
            let add = _mm256_and_si256(sign, oq_vec);
            _mm256_storeu_si256(r_ptr, _mm256_add_epi64(xl, add));
            r_ptr = r_ptr.add(1);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// c_from_b_avx2
// ─────────────────────────────────────────────────────────────────────────────

/// Reduce a single q120b `__m256i` to the canonical residue in `[0, Q[k])` for each prime.
///
/// Input `x` holds values in `[0, Q[k] << 33)`, so `x < 2^63`.
/// Returns the residue in the lower 32 bits of each 64-bit lane.
#[inline(always)]
unsafe fn reduce_b_to_canonical(x: __m256i, q: __m256i, mu: __m256i, pow32: __m256i) -> __m256i {
    unsafe {
        let mask32 = _mm256_set1_epi64x(u32::MAX as i64);
        // x_hi = x >> 32 < 2 * Q[k] (since x < Q << 33)
        let x_hi = _mm256_srli_epi64::<32>(x);
        let x_lo = _mm256_and_si256(x, mask32);
        // Reduce x_hi to [0, Q) with one conditional subtract
        let x_hi_r = cond_sub(x_hi, q);
        // tmp = x_hi_r * pow32 + x_lo  (<  Q * Q + 2^32 < 2^60 + 2^32 < 2^61)
        let tmp = _mm256_add_epi64(_mm256_mul_epu32(x_hi_r, pow32), x_lo);
        // Barrett-reduce tmp to [0, Q)
        barrett_reduce(tmp, q, mu)
    }
}

/// AVX2 port of `c_from_b_ref`: convert q120b to q120c.
///
/// For each of `nn` ring elements, reads one `__m256i` (4 × u64, q120b layout) and writes
/// one `__m256i` (8 × u32, q120c layout `[r[0], r_shift[0], ..., r[3], r_shift[3]]`).
///
/// # Safety
///
/// Caller must ensure AVX2 support. `res.len() >= 8 * nn`, `a.len() >= 4 * nn`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn c_from_b_avx2(nn: usize, res: &mut [u32], a: &[u64]) {
    assert!(
        res.len() >= 8 * nn,
        "c_from_b_avx2: res.len()={} < 8*nn={}",
        res.len(),
        8 * nn
    );
    assert!(a.len() >= 4 * nn, "c_from_b_avx2: a.len()={} < 4*nn={}", a.len(), 4 * nn);
    unsafe {
        let q = _mm256_loadu_si256(Q_VEC.as_ptr() as *const __m256i);
        let mu = _mm256_loadu_si256(BARRETT_MU.as_ptr() as *const __m256i);
        let pow32 = _mm256_loadu_si256(POW32.as_ptr() as *const __m256i);

        let mut a_ptr = a.as_ptr() as *const __m256i;
        let mut r_ptr = res.as_mut_ptr() as *mut __m256i;

        for _ in 0..nn {
            let xv = _mm256_loadu_si256(a_ptr);
            // r[k] = xv[k] mod Q[k], in lower 32 bits of each lane
            let r = reduce_b_to_canonical(xv, q, mu, pow32);
            // r_shift[k] = r[k] * 2^32 mod Q[k] = (r * pow32) mod Q
            let r_shift = barrett_reduce(_mm256_mul_epu32(r, pow32), q, mu);
            // Pack into q120c: [r[0], r_shift[0], r[1], r_shift[1], r[2], r_shift[2], r[3], r_shift[3]]
            // r in lower 32 bits, r_shift goes into upper 32 bits via shift then OR
            let packed = _mm256_or_si256(r, _mm256_slli_epi64::<32>(r_shift));
            _mm256_storeu_si256(r_ptr, packed);

            a_ptr = a_ptr.add(1);
            r_ptr = r_ptr.add(1);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// vec_mat1col_product_bbb_avx2
// ─────────────────────────────────────────────────────────────────────────────

/// AVX2 port of `vec_mat1col_product_bbb_ref`: q120b × q120b → q120b dot product.
///
/// Computes `res = Σᵢ x[i] · y[i]` in q120b format for `i ∈ 0..ell`.
/// Each element is one `__m256i` (4 × u64, one u64 per prime).
///
/// Uses a four-bin accumulation scheme (`s1`–`s4`) matching the scalar reference,
/// with the same `BbbMeta` reduction constants for the final collapse.
///
/// # Safety
///
/// Caller must ensure AVX2 support. `res.len() >= 4`, `x.len() >= 4 * ell`, `y.len() >= 4 * ell`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn vec_mat1col_product_bbb_avx2(meta: &BbbMeta<Primes30>, ell: usize, res: &mut [u64], x: &[u64], y: &[u64]) {
    assert!(res.len() >= 4, "vec_mat1col_product_bbb_avx2: res.len()={} < 4", res.len());
    assert!(
        x.len() >= 4 * ell,
        "vec_mat1col_product_bbb_avx2: x.len()={} < 4*ell={}",
        x.len(),
        4 * ell
    );
    assert!(
        y.len() >= 4 * ell,
        "vec_mat1col_product_bbb_avx2: y.len()={} < 4*ell={}",
        y.len(),
        4 * ell
    );
    unsafe {
        let mask32 = _mm256_set1_epi64x(u32::MAX as i64);
        let mut s1 = _mm256_setzero_si256();
        let mut s2 = _mm256_setzero_si256();
        let mut s3 = _mm256_setzero_si256();
        let mut s4 = _mm256_setzero_si256();

        let mut x_ptr = x.as_ptr() as *const __m256i;
        let mut y_ptr = y.as_ptr() as *const __m256i;

        for _ in 0..ell {
            let xv = _mm256_loadu_si256(x_ptr);
            let xl = _mm256_and_si256(xv, mask32);
            let xh = _mm256_srli_epi64::<32>(xv);

            let yv = _mm256_loadu_si256(y_ptr);
            let yl = _mm256_and_si256(yv, mask32);
            let yh = _mm256_srli_epi64::<32>(yv);

            // Four 32×32→64 cross-products (one per term in the scalar)
            let a = _mm256_mul_epu32(xl, yl); // xl * yl
            let b = _mm256_mul_epu32(xl, yh); // xl * yh
            let c = _mm256_mul_epu32(xh, yl); // xh * yl
            let d = _mm256_mul_epu32(xh, yh); // xh * yh

            // Accumulate into four bins, matching scalar binning:
            //   s1 += a_lo
            //   s2 += a_hi + b_lo + c_lo
            //   s3 += b_hi + c_hi + d_lo
            //   s4 += d_hi
            s1 = _mm256_add_epi64(s1, _mm256_and_si256(a, mask32));
            s2 = _mm256_add_epi64(s2, _mm256_srli_epi64::<32>(a));
            s2 = _mm256_add_epi64(s2, _mm256_and_si256(b, mask32));
            s2 = _mm256_add_epi64(s2, _mm256_and_si256(c, mask32));
            s3 = _mm256_add_epi64(s3, _mm256_srli_epi64::<32>(b));
            s3 = _mm256_add_epi64(s3, _mm256_srli_epi64::<32>(c));
            s3 = _mm256_add_epi64(s3, _mm256_and_si256(d, mask32));
            s4 = _mm256_add_epi64(s4, _mm256_srli_epi64::<32>(d));

            x_ptr = x_ptr.add(1);
            y_ptr = y_ptr.add(1);
        }

        // Final reduction using BbbMeta constants
        let h2 = meta.h;
        let mask_h2 = _mm256_set1_epi64x(((1u64 << h2) - 1) as i64);
        let h2_cnt = _mm_cvtsi64_si128(h2 as i64);
        let s1h_pow = _mm256_set1_epi64x(meta.s1h_pow_red as i64); // prime-independent
        let s2l_pow = _mm256_loadu_si256(meta.s2l_pow_red.as_ptr() as *const __m256i);
        let s2h_pow = _mm256_loadu_si256(meta.s2h_pow_red.as_ptr() as *const __m256i);
        let s3l_pow = _mm256_loadu_si256(meta.s3l_pow_red.as_ptr() as *const __m256i);
        let s3h_pow = _mm256_loadu_si256(meta.s3h_pow_red.as_ptr() as *const __m256i);
        let s4l_pow = _mm256_loadu_si256(meta.s4l_pow_red.as_ptr() as *const __m256i);
        let s4h_pow = _mm256_loadu_si256(meta.s4h_pow_red.as_ptr() as *const __m256i);

        // Split each sX into low-h2 and high-h2 bits
        let s1l = _mm256_and_si256(s1, mask_h2);
        let s1h = _mm256_srl_epi64(s1, h2_cnt);
        let s2l = _mm256_and_si256(s2, mask_h2);
        let s2h = _mm256_srl_epi64(s2, h2_cnt);
        let s3l = _mm256_and_si256(s3, mask_h2);
        let s3h = _mm256_srl_epi64(s3, h2_cnt);
        let s4l = _mm256_and_si256(s4, mask_h2);
        let s4h = _mm256_srl_epi64(s4, h2_cnt);

        // t = s1l + s1h*2^h + s2l*s2l_pow + s2h*s2h_pow + s3l*s3l_pow + s3h*s3h_pow
        //       + s4l*s4l_pow + s4h*s4h_pow
        let mut t = s1l;
        t = _mm256_add_epi64(t, _mm256_mul_epu32(s1h, s1h_pow));
        t = _mm256_add_epi64(t, _mm256_mul_epu32(s2l, s2l_pow));
        t = _mm256_add_epi64(t, _mm256_mul_epu32(s2h, s2h_pow));
        t = _mm256_add_epi64(t, _mm256_mul_epu32(s3l, s3l_pow));
        t = _mm256_add_epi64(t, _mm256_mul_epu32(s3h, s3h_pow));
        t = _mm256_add_epi64(t, _mm256_mul_epu32(s4l, s4l_pow));
        t = _mm256_add_epi64(t, _mm256_mul_epu32(s4h, s4h_pow));

        _mm256_storeu_si256(res.as_mut_ptr() as *mut __m256i, t);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// b_to_znx128_avx2
// ─────────────────────────────────────────────────────────────────────────────

/// Hybrid AVX2 / scalar CRT reconstruction: q120b → i128 coefficients.
///
/// For each of `nn` ring elements:
/// - **AVX2**: Computes `t[k] = (x[4*j+k] % Q[k] * CRT_CST[k]) % Q[k]` for k=0..3.
/// - **Scalar**: Accumulates `tmp = Σ_k t[k] * (Q/Q[k])` in i128, reduces mod `total_Q`,
///   and applies a symmetric lift to `(-total_Q/2, total_Q/2]`.
///
/// # Safety
///
/// Caller must ensure AVX2 support. `res.len() >= nn`, `a.len() >= 4 * nn`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn b_to_znx128_avx2(nn: usize, res: &mut [i128], a: &[u64]) {
    assert!(res.len() >= nn, "b_to_znx128_avx2: res.len()={} < nn={}", res.len(), nn);
    assert!(a.len() >= 4 * nn, "b_to_znx128_avx2: a.len()={} < 4*nn={}", a.len(), 4 * nn);
    // Scalar CRT reconstruction constants (same as b_to_znx128_ref)
    let q: [i128; 4] = Primes30::Q.map(|qi| qi as i128);
    let total_q: i128 = q[0] * q[1] * q[2] * q[3];
    let qm: [i128; 4] = [q[1] * q[2] * q[3], q[0] * q[2] * q[3], q[0] * q[1] * q[3], q[0] * q[1] * q[2]];
    let half = (total_q + 1) / 2;

    unsafe {
        let q_vec = _mm256_loadu_si256(Q_VEC.as_ptr() as *const __m256i);
        let mu_vec = _mm256_loadu_si256(BARRETT_MU.as_ptr() as *const __m256i);
        let pow32_vec = _mm256_loadu_si256(POW32.as_ptr() as *const __m256i);
        let crt_vec = _mm256_loadu_si256(CRT_VEC.as_ptr() as *const __m256i);

        let mut a_ptr = a.as_ptr() as *const __m256i;

        for r in &mut res[..nn] {
            let xv = _mm256_loadu_si256(a_ptr);

            // Step 1+2 (AVX2): xk = x % Q,  t = (xk * CRT_CST) % Q
            let xk = reduce_b_to_canonical(xv, q_vec, mu_vec, pow32_vec);
            let crt_prod = _mm256_mul_epu32(xk, crt_vec); // xk * CRT_CST < Q^2 < 2^60
            let t = barrett_reduce(crt_prod, q_vec, mu_vec);

            // Extract t[0..4] to stack for scalar accumulation
            let mut t_arr = [0u64; 4];
            _mm256_storeu_si256(t_arr.as_mut_ptr() as *mut __m256i, t);

            // Step 3 (scalar): CRT accumulation in i128
            let mut tmp: i128 = 0;
            for k in 0..4 {
                tmp += t_arr[k] as i128 * qm[k];
            }
            tmp %= total_q;
            *r = if tmp >= half { tmp - total_q } else { tmp };

            a_ptr = a_ptr.add(1);
        }
    }
}
