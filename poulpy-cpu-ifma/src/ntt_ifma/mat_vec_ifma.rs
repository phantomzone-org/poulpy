//! AVX512-IFMA BBC inner-product kernels for the IFMA backend.
//!
//! This module replaces the scalar reference q120b x q120c inner-product routines
//! with `VPMADD52*`-based SIMD kernels and SIMD-only final reduction.
//!
//! # Layout conventions
//!
//! | Format | Bytes/element | AVX view |
//! |--------|--------------|----------|
//! | q120b  | 32 (4 × u64) | one `__m256i` |
//! | q120c  | 32 (4 × u64, reduced residues) | one `__m256i` |
//! | x2-block | 64 (2 × q120b/c) | two `__m256i`s |
//!
//! The IFMA "c" format stores reduced u64 residues (same layout as b, just
//! reduced mod Q[k]). This differs from the AVX/NTT120 c format which uses
//! split lo32/hi32 pairs. The `&[u32]` slice types in the function signatures
//! are for trait compatibility — the data is actually u64 values (each u32
//! pair forms one u64).
//!
//! # Accumulation strategy
//!
//! Uses VPMADD52LUQ / VPMADD52HUQ to split each 104-bit product at bit 52:
//! - `acc_lo += (x[51:0] * y[51:0])[51:0]` — low 52 bits
//! - `acc_hi += (x[51:0] * y[51:0])[103:52]` — high 52 bits
//!
//! Since x < 2^41 (values in [0, 2q)) and y < 2^40 (reduced mod Q), both
//! fit within the 52-bit input window. After `ell` iterations, `acc_lo < ell × 2^52`
//! which fits in u64 for ell < 4096.

use core::arch::x86_64::{
    __m256i, __m512i, _mm256_add_epi64, _mm256_and_si256, _mm256_loadu_si256, _mm256_madd52hi_epu64, _mm256_madd52lo_epu64,
    _mm256_mul_epu32, _mm256_set1_epi64x, _mm256_setzero_si256, _mm256_srli_epi64, _mm256_storeu_si256, _mm512_add_epi64,
    _mm512_and_si512, _mm512_loadu_si512, _mm512_madd52hi_epu64, _mm512_madd52lo_epu64, _mm512_mul_epu32, _mm512_set1_epi64,
    _mm512_setzero_si512, _mm512_srli_epi64, _mm512_storeu_si512, _mm512_stream_si512,
};

use super::ntt_ifma_avx512::{cond_sub_2q_si256, cond_sub_2q_si512, harvey_modmul_si256, harvey_modmul_si512};

use poulpy_cpu_ref::reference::ntt_ifma::{
    mat_vec::BbcIfmaMeta,
    primes::{PrimeSetIfma, Primes40},
};

// ─────────────────────────────────────────────────────────────────────────────
// Constants for SIMD reduction
// ─────────────────────────────────────────────────────────────────────────────

const Q_IFMA: [u64; 3] = <Primes40 as PrimeSetIfma>::Q;

/// Q vector: `[Q[0], Q[1], Q[2], 0]`.
const Q_VEC: [u64; 4] = [Q_IFMA[0], Q_IFMA[1], Q_IFMA[2], 0];

/// 2Q vector: `[2*Q[0], 2*Q[1], 2*Q[2], 0]`.
const Q2_VEC: [u64; 4] = [2 * Q_IFMA[0], 2 * Q_IFMA[1], 2 * Q_IFMA[2], 0];

/// `2^40 mod Q[k]` — for two-pass modular reduction of wide values.
/// Since Q[k] < 2^40, this equals `2^40 - Q[k]` (small, < 2^22).
const POW40_MOD_Q: [u64; 4] = {
    let pow40 = 1u64 << 40;
    [pow40 - Q_IFMA[0], pow40 - Q_IFMA[1], pow40 - Q_IFMA[2], 0]
};

/// `2^52 mod Q[k]` — value of the 52-bit accumulator boundary mod Q.
const POW52_MOD_Q_VEC: [u64; 4] = {
    let mut r = [0u64; 4];
    let mut k = 0;
    while k < 3 {
        r[k] = (1u64 << 52) % Q_IFMA[k];
        k += 1;
    }
    r
};

/// Harvey quotient for POW52_MOD_Q: `floor(POW52_MOD_Q[k] * 2^52 / Q[k])`.
const POW52_MOD_Q_QUOT: [u64; 4] = {
    let mut r = [0u64; 4];
    let mut k = 0;
    while k < 3 {
        r[k] = ((POW52_MOD_Q_VEC[k] as u128 * (1u128 << 52)) / Q_IFMA[k] as u128) as u64;
        k += 1;
    }
    r
};

// ─────────────────────────────────────────────────────────────────────────────
// 512-bit (2-coefficient) duplicated constants
// ─────────────────────────────────────────────────────────────────────────────

/// Q vector duplicated for 512-bit: `[Q[0], Q[1], Q[2], 0, Q[0], Q[1], Q[2], 0]`.
const Q_VEC_512: [u64; 8] = [Q_IFMA[0], Q_IFMA[1], Q_IFMA[2], 0, Q_IFMA[0], Q_IFMA[1], Q_IFMA[2], 0];

/// 2Q vector duplicated for 512-bit.
const Q2_VEC_512: [u64; 8] = [
    2 * Q_IFMA[0],
    2 * Q_IFMA[1],
    2 * Q_IFMA[2],
    0,
    2 * Q_IFMA[0],
    2 * Q_IFMA[1],
    2 * Q_IFMA[2],
    0,
];

/// `2^40 mod Q[k]` duplicated for 512-bit.
const POW40_MOD_Q_512: [u64; 8] = {
    let pow40 = 1u64 << 40;
    let a = pow40 - Q_IFMA[0];
    let b = pow40 - Q_IFMA[1];
    let c = pow40 - Q_IFMA[2];
    [a, b, c, 0, a, b, c, 0]
};

/// `2^52 mod Q[k]` duplicated for 512-bit.
const POW52_MOD_Q_VEC_512: [u64; 8] = {
    let mut r = [0u64; 8];
    let mut k = 0;
    while k < 3 {
        r[k] = (1u64 << 52) % Q_IFMA[k];
        r[k + 4] = r[k];
        k += 1;
    }
    r
};

/// Harvey quotient for POW52_MOD_Q duplicated for 512-bit.
const POW52_MOD_Q_QUOT_512: [u64; 8] = {
    let mut r = [0u64; 8];
    let mut k = 0;
    while k < 3 {
        r[k] = ((POW52_MOD_Q_VEC[k] as u128 * (1u128 << 52)) / Q_IFMA[k] as u128) as u64;
        r[k + 4] = r[k];
        k += 1;
    }
    r
};

// ─────────────────────────────────────────────────────────────────────────────
// SIMD reduction helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Reduce a wide u64 value (< ell × 2^52) to [0, Q) per lane, fully in SIMD.
///
/// Uses two-pass split at bit 40: since Q ≈ 2^40, `2^40 mod Q` is small (< 2^22),
/// so `hi * POW40_MOD_Q + lo` rapidly converges to a value < 2Q.
///
/// Valid for values up to ~2^64 (ell < 4096).
#[inline]
#[target_feature(enable = "avx512vl")]
unsafe fn reduce_wide_mod_q(x: __m256i) -> __m256i {
    unsafe {
        let mask40 = _mm256_set1_epi64x((1i64 << 40) - 1);
        let pow40 = _mm256_loadu_si256(POW40_MOD_Q.as_ptr() as *const __m256i);
        let q = _mm256_loadu_si256(Q_VEC.as_ptr() as *const __m256i);

        // Pass 1: split at bit 40
        let hi = _mm256_srli_epi64::<40>(x); // < 2^24 (for x < 2^64)
        let lo = _mm256_and_si256(x, mask40); // < 2^40
        // y = hi * POW40_MOD_Q + lo < 2^24 * 2^22 + 2^40 < 2^47
        let y = _mm256_add_epi64(_mm256_mul_epu32(hi, pow40), lo);

        // Pass 2: split at bit 40 again
        let hi2 = _mm256_srli_epi64::<40>(y); // < 2^7
        let lo2 = _mm256_and_si256(y, mask40);
        // z = hi2 * POW40_MOD_Q + lo2 < 2^7 * 2^22 + 2^40 < 2^41 < 2Q
        let z = _mm256_add_epi64(_mm256_mul_epu32(hi2, pow40), lo2);

        // Final cond_sub: [0, 2Q) → [0, Q)
        cond_sub_2q_si256(z, q)
    }
}

/// Collapse MADD52 accumulators `(acc_lo, acc_hi)` into a q120b `__m256i`, fully in SIMD.
///
/// Computes `(acc_lo + acc_hi × 2^52) mod Q` per lane using:
/// 1. Two-pass reduction of `acc_lo` via POW40 → `lo_red ∈ [0, Q)`
/// 2. Harvey modular multiply of `acc_hi × POW52_MOD_Q` → `hi_red ∈ [0, 2Q)`
/// 3. Add + two conditional subtracts → `[0, Q)`
///
/// No stack spills. All intermediate values stay in SIMD registers.
///
/// # Overflow constraints
///
/// Valid for `ell < 4096`:
/// - `acc_lo < ell × 2^52 < 2^64`
/// - `acc_hi < ell × 2^29 < 2^41 < 2Q` (required by Harvey modmul)
#[inline]
#[target_feature(enable = "avx512ifma,avx512vl")]
pub(crate) unsafe fn reduce_bbc_ifma_simd(acc_lo: __m256i, acc_hi: __m256i) -> __m256i {
    unsafe {
        let q = _mm256_loadu_si256(Q_VEC.as_ptr() as *const __m256i);
        let q2 = _mm256_loadu_si256(Q2_VEC.as_ptr() as *const __m256i);
        let pow52 = _mm256_loadu_si256(POW52_MOD_Q_VEC.as_ptr() as *const __m256i);
        let pow52_quot = _mm256_loadu_si256(POW52_MOD_Q_QUOT.as_ptr() as *const __m256i);

        // Step 1: reduce acc_lo from [0, ell×2^52) to [0, Q)
        let lo_red = reduce_wide_mod_q(acc_lo);

        // Step 2: acc_hi * (2^52 mod Q) mod Q via Harvey modular multiply
        // acc_hi ∈ [0, 2Q) for ell < 4096, POW52_MOD_Q ∈ [0, Q)
        let hi_red = harvey_modmul_si256(acc_hi, pow52, pow52_quot, q);
        // hi_red ∈ [0, 2Q)

        // Step 3: combine and reduce: lo_red + hi_red ∈ [0, 3Q)
        let sum = _mm256_add_epi64(lo_red, hi_red);
        // Two conditional subtracts: [0, 3Q) → [0, Q)
        let r = cond_sub_2q_si256(sum, q2); // subtract 2Q if >= 2Q → [0, Q) or [0, 2Q)
        cond_sub_2q_si256(r, q) // subtract Q if >= Q → [0, Q)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 512-bit (2-coefficient) reduction
// ─────────────────────────────────────────────────────────────────────────────

/// 512-bit two-pass modular reduction: wide u64 → [0, Q) per lane.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn reduce_wide_mod_q_512(x: __m512i) -> __m512i {
    unsafe {
        let mask40 = _mm512_set1_epi64((1i64 << 40) - 1);
        let pow40 = _mm512_loadu_si512(POW40_MOD_Q_512.as_ptr() as *const __m512i);
        let q = _mm512_loadu_si512(Q_VEC_512.as_ptr() as *const __m512i);

        let hi = _mm512_srli_epi64::<40>(x);
        let lo = _mm512_and_si512(x, mask40);
        let y = _mm512_add_epi64(_mm512_mul_epu32(hi, pow40), lo);

        let hi2 = _mm512_srli_epi64::<40>(y);
        let lo2 = _mm512_and_si512(y, mask40);
        let z = _mm512_add_epi64(_mm512_mul_epu32(hi2, pow40), lo2);

        cond_sub_2q_si512(z, q)
    }
}

/// Collapse MADD52 accumulators into q120b — 512-bit (2 coefficients at once).
#[inline]
#[target_feature(enable = "avx512ifma")]
pub(crate) unsafe fn reduce_bbc_ifma_simd_512(acc_lo: __m512i, acc_hi: __m512i) -> __m512i {
    unsafe {
        let q = _mm512_loadu_si512(Q_VEC_512.as_ptr() as *const __m512i);
        let q2 = _mm512_loadu_si512(Q2_VEC_512.as_ptr() as *const __m512i);
        let pow52 = _mm512_loadu_si512(POW52_MOD_Q_VEC_512.as_ptr() as *const __m512i);
        let pow52_quot = _mm512_loadu_si512(POW52_MOD_Q_QUOT_512.as_ptr() as *const __m512i);

        let lo_red = reduce_wide_mod_q_512(acc_lo);
        let hi_red = harvey_modmul_si512(acc_hi, pow52, pow52_quot, q);
        let sum = _mm512_add_epi64(lo_red, hi_red);
        let r = cond_sub_2q_si512(sum, q2);
        cond_sub_2q_si512(r, q)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-prime reduction (uniform lanes, for prime-major kernels)
// ─────────────────────────────────────────────────────────────────────────────

/// Reduce MADD52 accumulators where all 8 lanes belong to the SAME prime.
///
/// Same math as [`reduce_bbc_ifma_simd_512`] but uses broadcast (uniform)
/// constants instead of interleaved per-lane constants. This is what the
/// prime-major VMP kernel calls after accumulating one prime's inner products
/// across a block-quad (4 x2-blocks × 2 coeffs = 8 lanes, all same prime).
#[inline]
#[target_feature(enable = "avx512ifma,avx512vl")]
pub(crate) unsafe fn reduce_bbc_single_prime_512(
    acc_lo: __m512i,
    acc_hi: __m512i,
    q: __m512i,
    q2: __m512i,
    pow40: __m512i,
    pow52: __m512i,
    pow52_quot: __m512i,
) -> __m512i {
    unsafe {
        let mask40 = _mm512_set1_epi64((1i64 << 40) - 1);

        // reduce_wide: two-pass POW40 fold
        let hi1 = _mm512_srli_epi64::<40>(acc_lo);
        let lo1 = _mm512_and_si512(acc_lo, mask40);
        let y = _mm512_add_epi64(_mm512_mul_epu32(hi1, pow40), lo1);
        let hi2 = _mm512_srli_epi64::<40>(y);
        let lo2 = _mm512_and_si512(y, mask40);
        let z = _mm512_add_epi64(_mm512_mul_epu32(hi2, pow40), lo2);
        let lo_red = cond_sub_2q_si512(z, q);

        // Harvey modmul: acc_hi * POW52_MOD_Q mod Q
        let hi_red = harvey_modmul_si512(acc_hi, pow52, pow52_quot, q);

        let sum = _mm512_add_epi64(lo_red, hi_red);
        cond_sub_2q_si512(cond_sub_2q_si512(sum, q2), q)
    }
}

/// Precomputed broadcast constants for one CRT prime, used by [`reduce_bbc_single_prime_512`].
pub(crate) struct PrimeConsts512 {
    pub q: __m512i,
    pub q2: __m512i,
    pub pow40: __m512i,
    pub pow52: __m512i,
    pub pow52_quot: __m512i,
}

impl PrimeConsts512 {
    #[target_feature(enable = "avx512f")]
    pub(crate) unsafe fn new(prime_idx: usize) -> Self {
        let q_val = Q_IFMA[prime_idx];
        Self {
            q: _mm512_set1_epi64(q_val as i64),
            q2: _mm512_set1_epi64((2 * q_val) as i64),
            pow40: _mm512_set1_epi64(((1u64 << 40) - q_val) as i64),
            pow52: _mm512_set1_epi64(((1u64 << 52) % q_val) as i64),
            pow52_quot: _mm512_set1_epi64((((1u64 << 52) % q_val) as u128 * (1u128 << 52) / q_val as u128) as i64),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Single-column: q120b × q120c → q120b
// ─────────────────────────────────────────────────────────────────────────────

/// AVX512-IFMA inner product: `res = Σᵢ x[i] · y[i]` in q120b format.
///
/// - `x`: q120b in u32 view — `ell` elements × 8 u32 (one `__m256i` each).
/// - `y`: q120c in u32 view — `ell` elements × 8 u32 (one `__m256i` each).
/// - `res`: q120b output — at least 4 u64 (one `__m256i`).
///
/// # Safety
///
/// Caller must ensure AVX512-IFMA and AVX512-VL support. Slice lengths must
/// satisfy `x.len() >= 8 * ell`, `y.len() >= 8 * ell`, `res.len() >= 4`.
#[target_feature(enable = "avx512ifma,avx512vl")]
pub(crate) unsafe fn vec_mat1col_product_bbc_ifma(
    _meta: &BbcIfmaMeta<Primes40>,
    ell: usize,
    res: &mut [u64],
    x: &[u32],
    y: &[u32],
) {
    unsafe {
        // 4-way unrolling: 8 independent accumulator chains to better saturate
        // the VPMADD52 pipeline (6-cycle latency, 2/cycle throughput on Zen 5).
        let mut acc_lo0 = _mm256_setzero_si256();
        let mut acc_hi0 = _mm256_setzero_si256();
        let mut acc_lo1 = _mm256_setzero_si256();
        let mut acc_hi1 = _mm256_setzero_si256();
        let mut acc_lo2 = _mm256_setzero_si256();
        let mut acc_hi2 = _mm256_setzero_si256();
        let mut acc_lo3 = _mm256_setzero_si256();
        let mut acc_hi3 = _mm256_setzero_si256();

        let mut x_ptr = x.as_ptr() as *const __m256i;
        let mut y_ptr = y.as_ptr() as *const __m256i;
        let quads = ell / 4;

        for _ in 0..quads {
            let xv0 = _mm256_loadu_si256(x_ptr);
            let yv0 = _mm256_loadu_si256(y_ptr);
            let xv1 = _mm256_loadu_si256(x_ptr.add(1));
            let yv1 = _mm256_loadu_si256(y_ptr.add(1));
            let xv2 = _mm256_loadu_si256(x_ptr.add(2));
            let yv2 = _mm256_loadu_si256(y_ptr.add(2));
            let xv3 = _mm256_loadu_si256(x_ptr.add(3));
            let yv3 = _mm256_loadu_si256(y_ptr.add(3));

            acc_lo0 = _mm256_madd52lo_epu64(acc_lo0, xv0, yv0);
            acc_hi0 = _mm256_madd52hi_epu64(acc_hi0, xv0, yv0);
            acc_lo1 = _mm256_madd52lo_epu64(acc_lo1, xv1, yv1);
            acc_hi1 = _mm256_madd52hi_epu64(acc_hi1, xv1, yv1);
            acc_lo2 = _mm256_madd52lo_epu64(acc_lo2, xv2, yv2);
            acc_hi2 = _mm256_madd52hi_epu64(acc_hi2, xv2, yv2);
            acc_lo3 = _mm256_madd52lo_epu64(acc_lo3, xv3, yv3);
            acc_hi3 = _mm256_madd52hi_epu64(acc_hi3, xv3, yv3);

            x_ptr = x_ptr.add(4);
            y_ptr = y_ptr.add(4);
        }

        // Handle remainder (0-3 elements)
        let rem = ell % 4;
        if rem >= 2 {
            let xv0 = _mm256_loadu_si256(x_ptr);
            let yv0 = _mm256_loadu_si256(y_ptr);
            let xv1 = _mm256_loadu_si256(x_ptr.add(1));
            let yv1 = _mm256_loadu_si256(y_ptr.add(1));
            acc_lo0 = _mm256_madd52lo_epu64(acc_lo0, xv0, yv0);
            acc_hi0 = _mm256_madd52hi_epu64(acc_hi0, xv0, yv0);
            acc_lo1 = _mm256_madd52lo_epu64(acc_lo1, xv1, yv1);
            acc_hi1 = _mm256_madd52hi_epu64(acc_hi1, xv1, yv1);
            x_ptr = x_ptr.add(2);
            y_ptr = y_ptr.add(2);
        }
        if rem % 2 == 1 {
            let xv = _mm256_loadu_si256(x_ptr);
            let yv = _mm256_loadu_si256(y_ptr);
            acc_lo0 = _mm256_madd52lo_epu64(acc_lo0, xv, yv);
            acc_hi0 = _mm256_madd52hi_epu64(acc_hi0, xv, yv);
        }

        let acc_lo = _mm256_add_epi64(_mm256_add_epi64(acc_lo0, acc_lo1), _mm256_add_epi64(acc_lo2, acc_lo3));
        let acc_hi = _mm256_add_epi64(_mm256_add_epi64(acc_hi0, acc_hi1), _mm256_add_epi64(acc_hi2, acc_hi3));
        let r = reduce_bbc_ifma_simd(acc_lo, acc_hi);
        _mm256_storeu_si256(res.as_mut_ptr() as *mut __m256i, r);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// x2-block, single column: two q120b × q120c pairs → two q120b results
// ─────────────────────────────────────────────────────────────────────────────

/// AVX512-IFMA x2-block inner product: one column, two paired rows.
///
/// Computes two q120b inner products simultaneously:
/// - `res[0..4]` ← `Σᵢ x_a[i] · y_a[i]`
/// - `res[4..8]` ← `Σᵢ x_b[i] · y_b[i]`
///
/// - `x`: x2-block in u32 view — `ell` elements × 16 u32 (two `__m256i`s each).
/// - `y`: x2-block q120c — `ell` elements × 16 u32 (two `__m256i`s each).
/// - `res`: two q120b outputs — at least 8 u64.
///
/// # Safety
///
/// Caller must ensure AVX512-IFMA and AVX512-VL support. Slice lengths must
/// satisfy `x.len() >= 16 * ell`, `y.len() >= 16 * ell`, `res.len() >= 8`.
///
/// `NT_STORE`: when `true`, the final result is committed via
/// `_mm512_stream_si512` (cache-bypassing). The caller MUST then issue an
/// `_mm_sfence` before any subsequent load of the destination, and `res` must
/// be 64-byte aligned. Use this when the kernel writes a hot output buffer
/// that won't be re-read in the current loop and would otherwise evict
/// matrix cache lines.
#[target_feature(enable = "avx512ifma,avx512vl")]
pub(crate) unsafe fn vec_mat1col_product_x2_bbc_ifma<const NT_STORE: bool>(
    _meta: &BbcIfmaMeta<Primes40>,
    ell: usize,
    res: &mut [u64],
    x: &[u32],
    y: &[u32],
) {
    unsafe {
        // Both paired rows fit in a single __m512i (2 × 4 u64 lanes).
        // Lanes [0..4] = pair A (4 limbs), lanes [4..8] = pair B (4 limbs).
        let mut acc_lo0 = _mm512_setzero_si512();
        let mut acc_hi0 = _mm512_setzero_si512();
        let mut acc_lo1 = _mm512_setzero_si512();
        let mut acc_hi1 = _mm512_setzero_si512();

        let mut x_ptr = x.as_ptr() as *const __m512i;
        let mut y_ptr = y.as_ptr() as *const __m512i;
        let pairs = ell / 2;

        for _ in 0..pairs {
            let xv = _mm512_loadu_si512(x_ptr);
            let yv = _mm512_loadu_si512(y_ptr);
            let xv_next = _mm512_loadu_si512(x_ptr.add(1));
            let yv_next = _mm512_loadu_si512(y_ptr.add(1));
            acc_lo0 = _mm512_madd52lo_epu64(acc_lo0, xv, yv);
            acc_hi0 = _mm512_madd52hi_epu64(acc_hi0, xv, yv);
            acc_lo1 = _mm512_madd52lo_epu64(acc_lo1, xv_next, yv_next);
            acc_hi1 = _mm512_madd52hi_epu64(acc_hi1, xv_next, yv_next);

            x_ptr = x_ptr.add(2);
            y_ptr = y_ptr.add(2);
        }

        if !ell.is_multiple_of(2) {
            let xv = _mm512_loadu_si512(x_ptr);
            let yv = _mm512_loadu_si512(y_ptr);
            acc_lo0 = _mm512_madd52lo_epu64(acc_lo0, xv, yv);
            acc_hi0 = _mm512_madd52hi_epu64(acc_hi0, xv, yv);
        }

        // Reduce both pairs in one call.
        let result = reduce_bbc_ifma_simd_512(_mm512_add_epi64(acc_lo0, acc_lo1), _mm512_add_epi64(acc_hi0, acc_hi1));
        let res_ptr = res.as_mut_ptr() as *mut __m512i;
        if NT_STORE {
            _mm512_stream_si512(res_ptr, result);
        } else {
            _mm512_storeu_si512(res_ptr, result);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// x2-block, two columns: two q120b × four q120c pairs → four q120b results
// ─────────────────────────────────────────────────────────────────────────────

/// AVX512-IFMA x2-block inner product: two columns simultaneously.
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
/// Caller must ensure AVX512-IFMA and AVX512-VL support. Slice lengths must
/// satisfy `x.len() >= 16 * ell`, `y.len() >= 32 * ell`, `res.len() >= 16`.
#[target_feature(enable = "avx512ifma,avx512vl")]
pub(crate) unsafe fn vec_mat2cols_product_x2_bbc_ifma(
    _meta: &BbcIfmaMeta<Primes40>,
    ell: usize,
    res: &mut [u64],
    x: &[u32],
    y: &[u32],
) {
    unsafe {
        // Pack the two pairs (A and B) of each column into a single __m512i.
        // Lanes [0..4] = pair A, lanes [4..8] = pair B.
        let mut acc_lo_c0_0 = _mm512_setzero_si512();
        let mut acc_hi_c0_0 = _mm512_setzero_si512();
        let mut acc_lo_c1_0 = _mm512_setzero_si512();
        let mut acc_hi_c1_0 = _mm512_setzero_si512();
        let mut acc_lo_c0_1 = _mm512_setzero_si512();
        let mut acc_hi_c0_1 = _mm512_setzero_si512();
        let mut acc_lo_c1_1 = _mm512_setzero_si512();
        let mut acc_hi_c1_1 = _mm512_setzero_si512();

        let mut x_ptr = x.as_ptr() as *const __m512i;
        let mut y_ptr = y.as_ptr() as *const __m512i;
        let pairs = ell / 2;

        for _ in 0..pairs {
            // Load x pair: [xa | xb] in one __m512i.
            let xv = _mm512_loadu_si512(x_ptr);
            let xv_next = _mm512_loadu_si512(x_ptr.add(1));

            // Column 0: [yc0a | yc0b] in one __m512i.
            let yc0 = _mm512_loadu_si512(y_ptr);
            // Column 1: [yc1a | yc1b] in one __m512i (next 64 bytes after col0).
            let yc1 = _mm512_loadu_si512(y_ptr.add(1));
            let yc0_next = _mm512_loadu_si512(y_ptr.add(2));
            let yc1_next = _mm512_loadu_si512(y_ptr.add(3));

            acc_lo_c0_0 = _mm512_madd52lo_epu64(acc_lo_c0_0, xv, yc0);
            acc_hi_c0_0 = _mm512_madd52hi_epu64(acc_hi_c0_0, xv, yc0);
            acc_lo_c1_0 = _mm512_madd52lo_epu64(acc_lo_c1_0, xv, yc1);
            acc_hi_c1_0 = _mm512_madd52hi_epu64(acc_hi_c1_0, xv, yc1);
            acc_lo_c0_1 = _mm512_madd52lo_epu64(acc_lo_c0_1, xv_next, yc0_next);
            acc_hi_c0_1 = _mm512_madd52hi_epu64(acc_hi_c0_1, xv_next, yc0_next);
            acc_lo_c1_1 = _mm512_madd52lo_epu64(acc_lo_c1_1, xv_next, yc1_next);
            acc_hi_c1_1 = _mm512_madd52hi_epu64(acc_hi_c1_1, xv_next, yc1_next);

            x_ptr = x_ptr.add(2);
            y_ptr = y_ptr.add(4);
        }

        if !ell.is_multiple_of(2) {
            let xv = _mm512_loadu_si512(x_ptr);
            let yc0 = _mm512_loadu_si512(y_ptr);
            let yc1 = _mm512_loadu_si512(y_ptr.add(1));
            acc_lo_c0_0 = _mm512_madd52lo_epu64(acc_lo_c0_0, xv, yc0);
            acc_hi_c0_0 = _mm512_madd52hi_epu64(acc_hi_c0_0, xv, yc0);
            acc_lo_c1_0 = _mm512_madd52lo_epu64(acc_lo_c1_0, xv, yc1);
            acc_hi_c1_0 = _mm512_madd52hi_epu64(acc_hi_c1_0, xv, yc1);
        }

        let res_ptr = res.as_mut_ptr() as *mut __m512i;
        _mm512_storeu_si512(
            res_ptr,
            reduce_bbc_ifma_simd_512(
                _mm512_add_epi64(acc_lo_c0_0, acc_lo_c0_1),
                _mm512_add_epi64(acc_hi_c0_0, acc_hi_c0_1),
            ),
        );
        _mm512_storeu_si512(
            res_ptr.add(1),
            reduce_bbc_ifma_simd_512(
                _mm512_add_epi64(acc_lo_c1_0, acc_lo_c1_1),
                _mm512_add_epi64(acc_hi_c1_0, acc_hi_c1_1),
            ),
        );
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(all(test, target_feature = "avx512ifma", target_feature = "avx512vl"))]
mod tests {
    use super::*;
    use poulpy_cpu_ref::reference::ntt_ifma::{
        arithmetic::{b_ifma_from_znx64_ref, c_ifma_from_b_ref},
        mat_vec::{
            BbcIfmaMeta, vec_mat1col_product_bbc_ifma_ref, vec_mat1col_product_x2_bbc_ifma_ref,
            vec_mat2cols_product_x2_bbc_ifma_ref,
        },
        primes::Primes40,
    };

    /// Build q120b slice (as u32 view) from small i64 coefficients.
    fn make_q120b_u32(count: usize, seed: i64) -> Vec<u32> {
        let coeffs: Vec<i64> = (0..count).map(|i| (i as i64 * seed + 1) % 50 + 1).collect();
        let mut b = vec![0u64; 4 * count];
        b_ifma_from_znx64_ref(count, &mut b, &coeffs);
        // Reinterpret u64 as u32 pairs
        b.iter().flat_map(|&v| [v as u32, (v >> 32) as u32]).collect()
    }

    /// Build q120c slice (as u32 view) from small i64 coefficients.
    fn make_q120c_u32(count: usize, seed: i64) -> Vec<u32> {
        let coeffs: Vec<i64> = (0..count).map(|i| (i as i64 * seed + 2) % 50 + 1).collect();
        let mut b = vec![0u64; 4 * count];
        b_ifma_from_znx64_ref(count, &mut b, &coeffs);
        let mut c = vec![0u32; 8 * count];
        c_ifma_from_b_ref(count, &mut c, &b);
        c
    }

    /// IFMA `vec_mat1col_product_bbc` matches reference (single column, single output).
    #[test]
    fn vec_mat1col_product_bbc_ifma_vs_ref() {
        let ell = 8usize;
        let meta = BbcIfmaMeta::<Primes40>::new();

        let x = make_q120b_u32(ell, 7);
        let y = make_q120c_u32(ell, 13);

        let mut res_ifma = vec![0u64; 4];
        let mut res_ref = vec![0u64; 4];

        unsafe { vec_mat1col_product_bbc_ifma(&meta, ell, &mut res_ifma, &x, &y) };
        vec_mat1col_product_bbc_ifma_ref(&meta, ell, &mut res_ref, &x, &y);

        assert_eq!(res_ifma, res_ref, "vec_mat1col_product_bbc: IFMA vs ref mismatch");
    }

    /// IFMA `vec_mat1col_product_bbc` matches for larger ell values.
    #[test]
    fn vec_mat1col_product_bbc_ifma_vs_ref_large_ell() {
        let ell = 64usize;
        let meta = BbcIfmaMeta::<Primes40>::new();

        let x = make_q120b_u32(ell, 3);
        let y = make_q120c_u32(ell, 17);

        let mut res_ifma = vec![0u64; 4];
        let mut res_ref = vec![0u64; 4];

        unsafe { vec_mat1col_product_bbc_ifma(&meta, ell, &mut res_ifma, &x, &y) };
        vec_mat1col_product_bbc_ifma_ref(&meta, ell, &mut res_ref, &x, &y);

        assert_eq!(res_ifma, res_ref, "vec_mat1col_product_bbc (large ell): IFMA vs ref mismatch");
    }

    /// IFMA `vec_mat1col_product_x2_bbc` matches reference.
    #[test]
    fn vec_mat1col_product_x2_bbc_ifma_vs_ref() {
        let ell = 8usize;
        let meta = BbcIfmaMeta::<Primes40>::new();

        // x: 2 interleaved q120b (16 u32 per row)
        let x: Vec<u32> = {
            let a = make_q120b_u32(ell, 5);
            let b = make_q120b_u32(ell, 11);
            (0..ell)
                .flat_map(|i| a[8 * i..8 * i + 8].iter().chain(b[8 * i..8 * i + 8].iter()).copied())
                .collect()
        };
        // y: 2 interleaved q120c (16 u32 per row)
        let y: Vec<u32> = {
            let a = make_q120c_u32(ell, 3);
            let b = make_q120c_u32(ell, 17);
            (0..ell)
                .flat_map(|i| a[8 * i..8 * i + 8].iter().chain(b[8 * i..8 * i + 8].iter()).copied())
                .collect()
        };

        let mut res_ifma = vec![0u64; 8];
        let mut res_ref = vec![0u64; 8];

        unsafe { vec_mat1col_product_x2_bbc_ifma::<false>(&meta, ell, &mut res_ifma, &x, &y) };
        vec_mat1col_product_x2_bbc_ifma_ref(&meta, ell, &mut res_ref, &x, &y);

        assert_eq!(res_ifma, res_ref, "vec_mat1col_product_x2_bbc: IFMA vs ref mismatch");
    }

    /// IFMA `vec_mat2cols_product_x2_bbc` matches reference.
    #[test]
    fn vec_mat2cols_product_x2_bbc_ifma_vs_ref() {
        let ell = 8usize;
        let meta = BbcIfmaMeta::<Primes40>::new();

        // x: 2 interleaved q120b (16 u32 per row)
        let x: Vec<u32> = {
            let a = make_q120b_u32(ell, 7);
            let b = make_q120b_u32(ell, 19);
            (0..ell)
                .flat_map(|i| a[8 * i..8 * i + 8].iter().chain(b[8 * i..8 * i + 8].iter()).copied())
                .collect()
        };
        // y: 4 interleaved q120c (32 u32 per row: col0_a, col0_b, col1_a, col1_b)
        let y: Vec<u32> = {
            let c0a = make_q120c_u32(ell, 2);
            let c0b = make_q120c_u32(ell, 9);
            let c1a = make_q120c_u32(ell, 23);
            let c1b = make_q120c_u32(ell, 31);
            (0..ell)
                .flat_map(|i| {
                    c0a[8 * i..8 * i + 8]
                        .iter()
                        .chain(c0b[8 * i..8 * i + 8].iter())
                        .chain(c1a[8 * i..8 * i + 8].iter())
                        .chain(c1b[8 * i..8 * i + 8].iter())
                        .copied()
                })
                .collect()
        };

        let mut res_ifma = vec![0u64; 16];
        let mut res_ref = vec![0u64; 16];

        unsafe { vec_mat2cols_product_x2_bbc_ifma(&meta, ell, &mut res_ifma, &x, &y) };
        vec_mat2cols_product_x2_bbc_ifma_ref(&meta, ell, &mut res_ref, &x, &y);

        assert_eq!(res_ifma, res_ref, "vec_mat2cols_product_x2_bbc: IFMA vs ref mismatch");
    }
}
