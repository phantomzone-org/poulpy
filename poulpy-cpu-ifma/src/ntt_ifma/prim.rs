//! Primitive NTT-domain trait implementations for [`NTTIfma`](crate::NTTIfma).
//!
//! This module connects the IFMA backend type to the low-level reference IFMA traits:
//! NTT execution, b/c domain conversion, BBC multiply-accumulate, and basic
//! transform-domain arithmetic on q120b values.

use poulpy_hal::reference::ntt_ifma::{
    NttIfmaAdd, NttIfmaAddInplace, NttIfmaCFromB, NttIfmaCopy, NttIfmaDFTExecute, NttIfmaExtract1BlkContiguous, NttIfmaFromZnx64,
    NttIfmaMulBbc, NttIfmaMulBbc1ColX2, NttIfmaMulBbc2ColsX2, NttIfmaNegate, NttIfmaNegateInplace, NttIfmaSub, NttIfmaSubInplace,
    NttIfmaSubNegateInplace, NttIfmaToZnx128, NttIfmaZero,
    mat_vec::BbcIfmaMeta,
    ntt::{NttIfmaTable, NttIfmaTableInv},
    primes::Primes40,
    types::Q_SHIFTED_IFMA,
};

use super::mat_vec_ifma::{vec_mat1col_product_bbc_ifma, vec_mat1col_product_x2_bbc_ifma, vec_mat2cols_product_x2_bbc_ifma};

use super::ntt_ifma_avx512::{cond_sub_2q_si256, cond_sub_2q_si512, intt_ifma_avx512, ntt_ifma_avx512};

use core::arch::x86_64::{
    __m256i, __m512i, _mm256_add_epi64, _mm256_and_si256, _mm256_cmpgt_epi64, _mm256_loadu_si256, _mm256_mul_epu32,
    _mm256_set_epi64x, _mm256_set1_epi64x, _mm256_setzero_si256, _mm256_srli_epi64, _mm256_storeu_si256, _mm256_sub_epi64,
    _mm512_add_epi64, _mm512_loadu_si512, _mm512_storeu_si512, _mm512_sub_epi64,
};

use poulpy_hal::reference::ntt120::{
    NttAdd, NttAddInplace, NttCopy, NttNegate, NttNegateInplace, NttSub, NttSubInplace, NttSubNegateInplace, NttZero,
};

use crate::NTTIfma;

/// Q_SHIFTED_IFMA as 256-bit: `[2*Q[0], 2*Q[1], 2*Q[2], 0]`.
fn q2_vec() -> __m256i {
    unsafe { _mm256_loadu_si256(Q_SHIFTED_IFMA.as_ptr() as *const __m256i) }
}

/// Q_SHIFTED_IFMA duplicated for 512-bit: `[2Q0,2Q1,2Q2,0, 2Q0,2Q1,2Q2,0]`.
const Q2_512: [u64; 8] = {
    let q = <Primes40 as poulpy_hal::reference::ntt_ifma::primes::PrimeSetIfma>::Q;
    [2 * q[0], 2 * q[1], 2 * q[2], 0, 2 * q[0], 2 * q[1], 2 * q[2], 0]
};

/// Generic pattern for DFT-domain operations: 512-bit main loop + 256-bit tail.
/// This macro avoids repeating the 512/256 dispatch for each operation variant.
macro_rules! dft_op_512 {
    // 3-operand: res = f(a, b)
    (res=$res:ident, a=$a:ident, b=$b:ident, op512=$op512:expr, op256=$op256:expr) => {{
        let n = $res.len() / 4;
        let pairs = n / 2;
        let res_512 = $res.as_mut_ptr() as *mut __m512i;
        let a_512 = $a.as_ptr() as *const __m512i;
        let b_512 = $b.as_ptr() as *const __m512i;
        let q2_512v = _mm512_loadu_si512(Q2_512.as_ptr() as *const __m512i);
        for i in 0..pairs {
            let av = _mm512_loadu_si512(a_512.add(i));
            let bv = _mm512_loadu_si512(b_512.add(i));
            _mm512_storeu_si512(res_512.add(i), $op512(av, bv, q2_512v));
        }
        if n % 2 != 0 {
            let idx = n - 1;
            let res_256 = $res.as_mut_ptr() as *mut __m256i;
            let a_256 = $a.as_ptr() as *const __m256i;
            let b_256 = $b.as_ptr() as *const __m256i;
            let q2 = q2_vec();
            let av = _mm256_loadu_si256(a_256.add(idx));
            let bv = _mm256_loadu_si256(b_256.add(idx));
            _mm256_storeu_si256(res_256.add(idx), $op256(av, bv, q2));
        }
    }};
    // 2-operand inplace: res = f(res, a)
    (res_inplace=$res:ident, a=$a:ident, op512=$op512:expr, op256=$op256:expr) => {{
        let n = $res.len() / 4;
        let pairs = n / 2;
        let res_512 = $res.as_mut_ptr() as *mut __m512i;
        let a_512 = $a.as_ptr() as *const __m512i;
        let q2_512v = _mm512_loadu_si512(Q2_512.as_ptr() as *const __m512i);
        for i in 0..pairs {
            let rv = _mm512_loadu_si512(res_512.add(i) as *const __m512i);
            let av = _mm512_loadu_si512(a_512.add(i));
            _mm512_storeu_si512(res_512.add(i), $op512(rv, av, q2_512v));
        }
        if n % 2 != 0 {
            let idx = n - 1;
            let res_256 = $res.as_mut_ptr() as *mut __m256i;
            let a_256 = $a.as_ptr() as *const __m256i;
            let q2 = q2_vec();
            let rv = _mm256_loadu_si256(res_256.add(idx) as *const __m256i);
            let av = _mm256_loadu_si256(a_256.add(idx));
            _mm256_storeu_si256(res_256.add(idx), $op256(rv, av, q2));
        }
    }};
    // 1-operand inplace negate: res = f(res)
    (negate_inplace=$res:ident, op512=$op512:expr, op256=$op256:expr) => {{
        let n = $res.len() / 4;
        let pairs = n / 2;
        let res_512 = $res.as_mut_ptr() as *mut __m512i;
        let q2_512v = _mm512_loadu_si512(Q2_512.as_ptr() as *const __m512i);
        for i in 0..pairs {
            let rv = _mm512_loadu_si512(res_512.add(i) as *const __m512i);
            _mm512_storeu_si512(res_512.add(i), $op512(rv, q2_512v));
        }
        if n % 2 != 0 {
            let idx = n - 1;
            let res_256 = $res.as_mut_ptr() as *mut __m256i;
            let q2 = q2_vec();
            let rv = _mm256_loadu_si256(res_256.add(idx) as *const __m256i);
            _mm256_storeu_si256(res_256.add(idx), $op256(rv, q2));
        }
    }};
}

#[target_feature(enable = "avx512f")]
unsafe fn simd_add(res: &mut [u64], a: &[u64], b: &[u64]) {
    unsafe {
        dft_op_512!(
            res = res,
            a = a,
            b = b,
            op512 = |av: __m512i, bv: __m512i, q2: __m512i| cond_sub_2q_si512(_mm512_add_epi64(av, bv), q2),
            op256 = |av: __m256i, bv: __m256i, q2: __m256i| cond_sub_2q_si256(_mm256_add_epi64(av, bv), q2)
        );
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn simd_add_inplace(res: &mut [u64], a: &[u64]) {
    unsafe {
        dft_op_512!(
            res_inplace = res,
            a = a,
            op512 = |rv: __m512i, av: __m512i, q2: __m512i| cond_sub_2q_si512(_mm512_add_epi64(rv, av), q2),
            op256 = |rv: __m256i, av: __m256i, q2: __m256i| cond_sub_2q_si256(_mm256_add_epi64(rv, av), q2)
        );
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn simd_sub(res: &mut [u64], a: &[u64], b: &[u64]) {
    unsafe {
        dft_op_512!(
            res = res,
            a = a,
            b = b,
            op512 = |av: __m512i, bv: __m512i, q2: __m512i| cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(av, q2), bv), q2),
            op256 = |av: __m256i, bv: __m256i, q2: __m256i| cond_sub_2q_si256(_mm256_sub_epi64(_mm256_add_epi64(av, q2), bv), q2)
        );
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn simd_sub_inplace(res: &mut [u64], a: &[u64]) {
    unsafe {
        dft_op_512!(
            res_inplace = res,
            a = a,
            op512 = |rv: __m512i, av: __m512i, q2: __m512i| cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(rv, q2), av), q2),
            op256 = |rv: __m256i, av: __m256i, q2: __m256i| cond_sub_2q_si256(_mm256_sub_epi64(_mm256_add_epi64(rv, q2), av), q2)
        );
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn simd_sub_negate_inplace(res: &mut [u64], a: &[u64]) {
    unsafe {
        dft_op_512!(
            res_inplace = res,
            a = a,
            op512 = |rv: __m512i, av: __m512i, q2: __m512i| cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(av, q2), rv), q2),
            op256 = |rv: __m256i, av: __m256i, q2: __m256i| cond_sub_2q_si256(_mm256_sub_epi64(_mm256_add_epi64(av, q2), rv), q2)
        );
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn simd_negate(res: &mut [u64], a: &[u64]) {
    unsafe {
        dft_op_512!(
            res = res,
            a = a,
            b = a,
            op512 = |av: __m512i, _bv: __m512i, q2: __m512i| cond_sub_2q_si512(_mm512_sub_epi64(q2, av), q2),
            op256 = |av: __m256i, _bv: __m256i, q2: __m256i| cond_sub_2q_si256(_mm256_sub_epi64(q2, av), q2)
        );
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn simd_negate_inplace(res: &mut [u64]) {
    unsafe {
        dft_op_512!(
            negate_inplace = res,
            op512 = |rv: __m512i, q2: __m512i| cond_sub_2q_si512(_mm512_sub_epi64(q2, rv), q2),
            op256 = |rv: __m256i, q2: __m256i| cond_sub_2q_si256(_mm256_sub_epi64(q2, rv), q2)
        );
    }
}

/// Q vector (not 2Q) for c_from_b reduction: `[Q[0], Q[1], Q[2], 0]`.
fn q_vec() -> __m256i {
    use poulpy_hal::reference::ntt_ifma::primes::{PrimeSetIfma, Primes40};
    let q = <Primes40 as PrimeSetIfma>::Q;
    unsafe { _mm256_set_epi64x(0, q[2] as i64, q[1] as i64, q[0] as i64) }
}

/// `oq[k] = Q[k] - (2^63 mod Q[k])` for negative i64 handling.
const OQ_IFMA: [u64; 4] = {
    let q = <Primes40 as poulpy_hal::reference::ntt_ifma::primes::PrimeSetIfma>::Q;
    let mut oq = [0u64; 4];
    let mut k = 0;
    while k < 3 {
        oq[k] = q[k] - (i64::MIN as u64 % q[k]);
        k += 1;
    }
    oq
};

/// `2^40 mod Q[k]` — used for two-pass modular reduction of values up to 2^63.
///
/// Since Q[k] ≈ 2^40, `2^40 mod Q[k] = 2^40 - Q[k]` which is small (< 2^22).
const POW40_MOD_Q_IFMA: [u64; 4] = {
    let q = <Primes40 as poulpy_hal::reference::ntt_ifma::primes::PrimeSetIfma>::Q;
    let pow40 = 1u64 << 40;
    // All three primes are < 2^40, so pow40 mod Q[k] = pow40 - Q[k]
    [pow40 - q[0], pow40 - q[1], pow40 - q[2], 0]
};

/// SIMD: convert n i64 coefficients to 3-prime CRT b format.
///
/// For each i64 x:
/// 1. Strip sign bit, conditionally add oq[k] for negative inputs
/// 2. Two-pass reduction: split at bit 40, multiply high part by (2^40 mod Q),
///    add to low part, repeat. Final conditional subtract gives [0, Q).
///
/// Result: `res[4*i+k] = a[i] mod Q[k]` for k in {0,1,2}, `res[4*i+3] = 0`.
#[target_feature(enable = "avx512vl")]
unsafe fn simd_b_from_znx64(n: usize, res: &mut [u64], a: &[i64]) {
    unsafe {
        let oq_vec = _mm256_loadu_si256(OQ_IFMA.as_ptr() as *const __m256i);
        let i64_max = _mm256_set1_epi64x(i64::MAX);
        let zero = _mm256_setzero_si256();
        let mask40 = _mm256_set1_epi64x((1i64 << 40) - 1);
        let pow40 = _mm256_loadu_si256(POW40_MOD_Q_IFMA.as_ptr() as *const __m256i);
        let q = q_vec();
        // Mask to zero lane 3: [all-ones, all-ones, all-ones, 0]
        let lane3_zero = _mm256_set_epi64x(0, -1, -1, -1);
        let mut r_ptr = res.as_mut_ptr() as *mut __m256i;

        for &xval in &a[..n] {
            // Broadcast xval into all 4 lanes
            let xv = _mm256_set1_epi64x(xval);
            // Strip sign bit: xl = xval as u64 & 0x7FFF_FFFF_FFFF_FFFF
            let xl = _mm256_and_si256(xv, i64_max);
            // sign = all-ones in lanes where xval < 0
            let sign = _mm256_cmpgt_epi64(zero, xv);
            // add oq[k] only for negative inputs
            let add = _mm256_and_si256(sign, oq_vec);
            let val = _mm256_add_epi64(xl, add);

            // Two-pass modular reduction: val (< 2^63) → [0, Q)
            // Pass 1: split at bit 40
            let hi = _mm256_srli_epi64::<40>(val); // < 2^23, fits in 32 bits
            let lo = _mm256_and_si256(val, mask40);
            // y = hi * (2^40 mod Q) + lo < 2^23 * 2^22 + 2^40 < 2^46
            let y = _mm256_add_epi64(_mm256_mul_epu32(hi, pow40), lo);

            // Pass 2: split at bit 40 again
            let hi2 = _mm256_srli_epi64::<40>(y); // < 2^6
            let lo2 = _mm256_and_si256(y, mask40);
            // z = hi2 * (2^40 mod Q) + lo2 < 2^6 * 2^22 + 2^40 < 2^41 < 2*Q
            let z = _mm256_add_epi64(_mm256_mul_epu32(hi2, pow40), lo2);

            // Final: conditional subtract to get [0, Q)
            let result = cond_sub_2q_si256(z, q);

            // Zero lane 3 (padding)
            let result = _mm256_and_si256(result, lane3_zero);

            _mm256_storeu_si256(r_ptr, result);
            r_ptr = r_ptr.add(1);
        }
    }
}

/// AVX-512: reduce b-format values in [0, 2q) to c-format values in [0, q).
/// Uses 512-bit main loop (2 coefficients per iteration).
#[target_feature(enable = "avx512f")]
unsafe fn simd_c_from_b(n: usize, res: &mut [u64], a: &[u64]) {
    unsafe {
        let q_512 = {
            let q = <Primes40 as poulpy_hal::reference::ntt_ifma::primes::PrimeSetIfma>::Q;
            let arr: [u64; 8] = [q[0], q[1], q[2], 0, q[0], q[1], q[2], 0];
            _mm512_loadu_si512(arr.as_ptr() as *const __m512i)
        };
        let res_512 = res.as_mut_ptr() as *mut __m512i;
        let a_512 = a.as_ptr() as *const __m512i;
        let pairs = n / 2;
        for i in 0..pairs {
            let av = _mm512_loadu_si512(a_512.add(i));
            _mm512_storeu_si512(res_512.add(i), cond_sub_2q_si512(av, q_512));
        }
        if !n.is_multiple_of(2) {
            let idx = n - 1;
            let res_256 = res.as_mut_ptr() as *mut __m256i;
            let a_256 = a.as_ptr() as *const __m256i;
            let q = q_vec();
            let av = _mm256_loadu_si256(a_256.add(idx));
            _mm256_storeu_si256(res_256.add(idx), cond_sub_2q_si256(av, q));
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// IFMA NTT execution
// ──────────────────────────────────────────────────────────────────────────────

impl NttIfmaDFTExecute<NttIfmaTable<Primes40>> for NTTIfma {
    #[inline(always)]
    fn ntt_ifma_dft_execute(table: &NttIfmaTable<Primes40>, data: &mut [u64]) {
        unsafe { ntt_ifma_avx512::<Primes40>(table, data) }
    }
}

impl NttIfmaDFTExecute<NttIfmaTableInv<Primes40>> for NTTIfma {
    #[inline(always)]
    fn ntt_ifma_dft_execute(table: &NttIfmaTableInv<Primes40>, data: &mut [u64]) {
        unsafe { intt_ifma_avx512::<Primes40>(table, data) }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Domain conversion
// ──────────────────────────────────────────────────────────────────────────────

impl NttIfmaFromZnx64 for NTTIfma {
    #[inline(always)]
    fn ntt_ifma_from_znx64(res: &mut [u64], a: &[i64]) {
        unsafe { simd_b_from_znx64(a.len(), res, a) };
    }
}

impl NttIfmaToZnx128 for NTTIfma {
    #[inline(always)]
    fn ntt_ifma_to_znx128(res: &mut [i128], divisor_is_n: usize, a: &[u64]) {
        unsafe { super::vec_znx_dft::simd_b_ifma_to_znx128(divisor_is_n, res, a) };
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// IFMA-specific addition / subtraction / negation / copy / zero
// ──────────────────────────────────────────────────────────────────────────────

impl NttIfmaAdd for NTTIfma {
    #[inline(always)]
    fn ntt_ifma_add(res: &mut [u64], a: &[u64], b: &[u64]) {
        unsafe { simd_add(res, a, b) };
    }
}

impl NttIfmaAddInplace for NTTIfma {
    #[inline(always)]
    fn ntt_ifma_add_inplace(res: &mut [u64], a: &[u64]) {
        unsafe { simd_add_inplace(res, a) };
    }
}

impl NttIfmaSub for NTTIfma {
    #[inline(always)]
    fn ntt_ifma_sub(res: &mut [u64], a: &[u64], b: &[u64]) {
        unsafe { simd_sub(res, a, b) };
    }
}

impl NttIfmaSubInplace for NTTIfma {
    #[inline(always)]
    fn ntt_ifma_sub_inplace(res: &mut [u64], a: &[u64]) {
        unsafe { simd_sub_inplace(res, a) };
    }
}

impl NttIfmaSubNegateInplace for NTTIfma {
    #[inline(always)]
    fn ntt_ifma_sub_negate_inplace(res: &mut [u64], a: &[u64]) {
        unsafe { simd_sub_negate_inplace(res, a) };
    }
}

impl NttIfmaNegate for NTTIfma {
    #[inline(always)]
    fn ntt_ifma_negate(res: &mut [u64], a: &[u64]) {
        unsafe { simd_negate(res, a) };
    }
}

impl NttIfmaNegateInplace for NTTIfma {
    #[inline(always)]
    fn ntt_ifma_negate_inplace(res: &mut [u64]) {
        unsafe { simd_negate_inplace(res) };
    }
}

impl NttIfmaZero for NTTIfma {
    #[inline(always)]
    fn ntt_ifma_zero(res: &mut [u64]) {
        res.fill(0);
    }
}

impl NttIfmaCopy for NTTIfma {
    #[inline(always)]
    fn ntt_ifma_copy(res: &mut [u64], a: &[u64]) {
        res.copy_from_slice(a);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// IFMA multiply-accumulate
// ──────────────────────────────────────────────────────────────────────────────

impl NttIfmaMulBbc for NTTIfma {
    #[inline(always)]
    fn ntt_ifma_mul_bbc(meta: &BbcIfmaMeta<Primes40>, ell: usize, res: &mut [u64], ntt_coeff: &[u32], prepared: &[u32]) {
        unsafe { vec_mat1col_product_bbc_ifma(meta, ell, res, ntt_coeff, prepared) };
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// b -> c conversion
// ──────────────────────────────────────────────────────────────────────────────

impl NttIfmaCFromB for NTTIfma {
    #[inline(always)]
    fn ntt_ifma_c_from_b(n: usize, res: &mut [u32], a: &[u64]) {
        // c format for IFMA = reduced residues: a[k] mod Q[k].
        // Values in [0, 2q) → cond_sub with q → [0, q).
        // res is typed as &mut [u32] for trait compatibility but is actually u64 data.
        let res_u64: &mut [u64] = unsafe { std::slice::from_raw_parts_mut(res.as_mut_ptr() as *mut u64, res.len() / 2) };
        unsafe { simd_c_from_b(n, res_u64, a) };
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// VMP x2-block kernels
// ──────────────────────────────────────────────────────────────────────────────

impl NttIfmaMulBbc1ColX2 for NTTIfma {
    #[inline(always)]
    fn ntt_ifma_mul_bbc_1col_x2(meta: &BbcIfmaMeta<Primes40>, ell: usize, res: &mut [u64], a: &[u32], b: &[u32]) {
        unsafe { vec_mat1col_product_x2_bbc_ifma(meta, ell, res, a, b) };
    }
}

impl NttIfmaMulBbc2ColsX2 for NTTIfma {
    #[inline(always)]
    fn ntt_ifma_mul_bbc_2cols_x2(meta: &BbcIfmaMeta<Primes40>, ell: usize, res: &mut [u64], a: &[u32], b: &[u32]) {
        unsafe { vec_mat2cols_product_x2_bbc_ifma(meta, ell, res, a, b) };
    }
}

impl NttIfmaExtract1BlkContiguous for NTTIfma {
    #[inline(always)]
    fn ntt_ifma_extract_1blk_contiguous(n: usize, row_max: usize, blk: usize, dst: &mut [u64], src: &[u64]) {
        // Each x2-block = 2 consecutive coefficients = 8 u64 = 1 __m512i
        let coeff_idx = blk * 2;
        let dst_ptr = dst.as_mut_ptr();
        let src_ptr = src.as_ptr();
        for row in 0..row_max {
            let src_base = row * 4 * n + 4 * coeff_idx;
            let dst_base = row * 8;
            // Copy 8 u64 (64 bytes = 1 __m512i) via SIMD store
            unsafe {
                let data = _mm512_loadu_si512(src_ptr.add(src_base) as *const __m512i);
                _mm512_storeu_si512(dst_ptr.add(dst_base) as *mut __m512i, data);
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// NTT120 Ntt* traits (DFT-domain arithmetic reuse)
//
// The ntt120 generic functions (ntt120_vec_znx_dft_add, etc.) require these
// traits. Since the 3-prime IFMA layout uses the same 4 x u64 per coefficient
// representation (with lane 3 as padding), the implementation is identical
// to the NTT120 version but uses Q_SHIFTED_IFMA for the 3 active lanes.
// ──────────────────────────────────────────────────────────────────────────────

impl NttAdd for NTTIfma {
    #[inline(always)]
    fn ntt_add(res: &mut [u64], a: &[u64], b: &[u64]) {
        unsafe { simd_add(res, a, b) };
    }
}

impl NttAddInplace for NTTIfma {
    #[inline(always)]
    fn ntt_add_inplace(res: &mut [u64], a: &[u64]) {
        unsafe { simd_add_inplace(res, a) };
    }
}

impl NttSub for NTTIfma {
    #[inline(always)]
    fn ntt_sub(res: &mut [u64], a: &[u64], b: &[u64]) {
        unsafe { simd_sub(res, a, b) };
    }
}

impl NttSubInplace for NTTIfma {
    #[inline(always)]
    fn ntt_sub_inplace(res: &mut [u64], a: &[u64]) {
        unsafe { simd_sub_inplace(res, a) };
    }
}

impl NttSubNegateInplace for NTTIfma {
    #[inline(always)]
    fn ntt_sub_negate_inplace(res: &mut [u64], a: &[u64]) {
        unsafe { simd_sub_negate_inplace(res, a) };
    }
}

impl NttNegate for NTTIfma {
    #[inline(always)]
    fn ntt_negate(res: &mut [u64], a: &[u64]) {
        unsafe { simd_negate(res, a) };
    }
}

impl NttNegateInplace for NTTIfma {
    #[inline(always)]
    fn ntt_negate_inplace(res: &mut [u64]) {
        unsafe { simd_negate_inplace(res) };
    }
}

impl NttZero for NTTIfma {
    #[inline(always)]
    fn ntt_zero(res: &mut [u64]) {
        res.fill(0);
    }
}

impl NttCopy for NTTIfma {
    #[inline(always)]
    fn ntt_copy(res: &mut [u64], a: &[u64]) {
        res.copy_from_slice(a);
    }
}
