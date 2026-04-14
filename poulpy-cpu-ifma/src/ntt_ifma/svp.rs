//! Scalar-vector product AVX512 kernels for [`NTTIfma`](crate::NTTIfma).
//!
//! This module contains AVX512-IFMA SIMD kernels for scalar-vector product
//! (SVP) operations in the IFMA NTT domain. These kernels can be used to
//! override the default reference implementations for improved performance.

#![allow(dead_code)]

use core::arch::x86_64::{
    __m256i, __m512i, _mm256_loadu_si256, _mm256_madd52hi_epu64, _mm256_madd52lo_epu64, _mm256_setzero_si256,
    _mm256_storeu_si256, _mm512_loadu_si512, _mm512_madd52hi_epu64, _mm512_madd52lo_epu64, _mm512_setzero_si512,
    _mm512_storeu_si512,
};

use super::mat_vec_ifma::{reduce_bbc_ifma_simd, reduce_bbc_ifma_simd_512};

/// AVX-512 pointwise multiply: processes 2 coefficients per iteration using `__m512i`.
///
/// Falls back to 256-bit for odd tail.
#[target_feature(enable = "avx512ifma")]
pub(crate) unsafe fn svp_pointwise_mul_ifma(n: usize, res: *mut __m256i, b: *const __m256i, a: *const __m256i) {
    unsafe {
        let res_512 = res as *mut __m512i;
        let b_512 = b as *const __m512i;
        let a_512 = a as *const __m512i;
        let zero_512 = _mm512_setzero_si512();

        // Main loop: 2 coefficients per iteration via __m512i
        let pairs = n / 2;
        for i in 0..pairs {
            let bv = _mm512_loadu_si512(b_512.add(i));
            let av = _mm512_loadu_si512(a_512.add(i));
            let acc_lo = _mm512_madd52lo_epu64(zero_512, bv, av);
            let acc_hi = _mm512_madd52hi_epu64(zero_512, bv, av);
            _mm512_storeu_si512(res_512.add(i), reduce_bbc_ifma_simd_512(acc_lo, acc_hi));
        }

        // Odd tail: 1 remaining coefficient via __m256i
        if !n.is_multiple_of(2) {
            let i = n - 1;
            let zero = _mm256_setzero_si256();
            let bv = _mm256_loadu_si256(b.add(i));
            let av = _mm256_loadu_si256(a.add(i));
            let acc_lo = _mm256_madd52lo_epu64(zero, bv, av);
            let acc_hi = _mm256_madd52hi_epu64(zero, bv, av);
            _mm256_storeu_si256(res.add(i), reduce_bbc_ifma_simd(acc_lo, acc_hi));
        }
    }
}
