//! Polynomial convolution AVX512 kernels for [`NTTIfma`](crate::NTTIfma).
//!
//! This module contains AVX512-IFMA SIMD kernels for polynomial convolution
//! operations in the IFMA NTT domain. These kernels can be used to override
//! the default reference implementations for improved performance.

#![allow(dead_code)]

use poulpy_cpu_ref::reference::ntt120::types::Q120bScalar;
use poulpy_hal::layouts::{CnvPVecLToRef, CnvPVecRToRef, VecZnxDftToMut, ZnxInfos, ZnxView, ZnxViewMut};

use super::mat_vec_ifma::{reduce_bbc_ifma_simd, reduce_bbc_ifma_simd_512};
use super::ntt_ifma_avx512::{cond_sub_2q_si256, ntt_ifma_avx512};
use crate::NTTIfma;
use core::arch::x86_64::{
    __m256i, __m512i, _mm256_add_epi64, _mm256_loadu_si256, _mm256_madd52hi_epu64, _mm256_madd52lo_epu64, _mm256_set_epi64x,
    _mm256_setzero_si256, _mm256_storeu_si256, _mm512_add_epi64, _mm512_loadu_si512, _mm512_madd52hi_epu64,
    _mm512_madd52lo_epu64, _mm512_setzero_si512, _mm512_storeu_si512,
};

use poulpy_cpu_ref::reference::ntt_ifma::primes::{PrimeSetIfma, Primes40};

#[inline(always)]
unsafe fn bbc_accumulate_ifma(acc_lo: &mut __m256i, acc_hi: &mut __m256i, x: __m256i, y: __m256i) {
    unsafe {
        *acc_lo = _mm256_madd52lo_epu64(*acc_lo, x, y);
        *acc_hi = _mm256_madd52hi_epu64(*acc_hi, x, y);
    }
}

/// 512-bit BBC accumulate: processes 2 coefficients at once.
#[inline(always)]
unsafe fn bbc_accumulate_ifma_512(acc_lo: &mut __m512i, acc_hi: &mut __m512i, x: __m512i, y: __m512i) {
    unsafe {
        *acc_lo = _mm512_madd52lo_epu64(*acc_lo, x, y);
        *acc_hi = _mm512_madd52hi_epu64(*acc_hi, x, y);
    }
}

#[target_feature(enable = "avx512vl")]
unsafe fn reduce_b_to_c_inplace_ifma(buf: &mut [u64]) {
    let q = <Primes40 as PrimeSetIfma>::Q;
    let q_vec = _mm256_set_epi64x(0, q[2] as i64, q[1] as i64, q[0] as i64);
    let ptr = buf.as_mut_ptr() as *mut __m256i;
    let n = buf.len() / 4;
    for i in 0..n {
        let x = unsafe { _mm256_loadu_si256(ptr.add(i)) };
        unsafe { _mm256_storeu_si256(ptr.add(i), cond_sub_2q_si256(x, q_vec)) };
    }
}

/// Maximum number of precomputed pointer pairs for convolution summation.
/// Convolution j-range is typically small (< 64 for practical CKKS parameters).
const MAX_CNV_J_RANGE: usize = 128;

#[target_feature(enable = "avx512ifma,avx512vl")]
unsafe fn cnv_apply_dft_ifma<R, A, B>(res: &mut R, res_offset: usize, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
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

    let bound = a_size + b_size - 1;
    let min_size = res_size.min(bound);
    let offset = res_offset.min(bound);

    for k in 0..min_size {
        let k_abs = k + offset;
        let j_min = k_abs.saturating_sub(a_size - 1);
        let j_max = (k_abs + 1).min(b_size);
        let j_count = j_max.saturating_sub(j_min);
        debug_assert!(
            j_count <= MAX_CNV_J_RANGE,
            "convolution j-range {} exceeds MAX_CNV_J_RANGE ({})",
            j_count,
            MAX_CNV_J_RANGE
        );

        let res_ptr = res.at_mut_ptr(res_col, k) as *mut __m256i;

        // Hoist at_ptr: precompute base pointers for all j values
        let mut a_bases = [core::ptr::null::<__m256i>(); MAX_CNV_J_RANGE];
        let mut b_bases = [core::ptr::null::<__m256i>(); MAX_CNV_J_RANGE];
        for (idx, j) in (j_min..j_max).enumerate() {
            a_bases[idx] = a.at_ptr(a_col, k_abs - j) as *const __m256i;
            b_bases[idx] = b.at_ptr(b_col, j) as *const __m256i;
        }

        // Process n_i in pairs via __m512i (2 coefficients per accumulator)
        let mut n_i = 0usize;
        while n_i + 2 <= n {
            let mut acc_lo = _mm512_setzero_si512();
            let mut acc_hi = _mm512_setzero_si512();
            for idx in 0..j_count {
                unsafe {
                    // Load 2 consecutive coefficients as one __m512i
                    let x = _mm512_loadu_si512(a_bases[idx].add(n_i) as *const __m512i);
                    let y = _mm512_loadu_si512(b_bases[idx].add(n_i) as *const __m512i);
                    bbc_accumulate_ifma_512(&mut acc_lo, &mut acc_hi, x, y);
                }
            }
            unsafe {
                _mm512_storeu_si512(res_ptr.add(n_i) as *mut __m512i, reduce_bbc_ifma_simd_512(acc_lo, acc_hi));
            }
            n_i += 2;
        }

        // Handle odd tail
        if n_i < n {
            let mut acc_lo = _mm256_setzero_si256();
            let mut acc_hi = _mm256_setzero_si256();
            for idx in 0..j_count {
                unsafe {
                    let x = _mm256_loadu_si256(a_bases[idx].add(n_i));
                    let y = _mm256_loadu_si256(b_bases[idx].add(n_i));
                    bbc_accumulate_ifma(&mut acc_lo, &mut acc_hi, x, y);
                }
            }
            unsafe { _mm256_storeu_si256(res_ptr.add(n_i), reduce_bbc_ifma_simd(acc_lo, acc_hi)) };
        }
    }

    for j in min_size..res_size {
        res.at_mut(res_col, j).fill(Q120bScalar([0; 4]));
    }
}

#[target_feature(enable = "avx512ifma,avx512vl")]
unsafe fn cnv_pairwise_apply_dft_ifma<R, A, B>(
    res: &mut R,
    res_offset: usize,
    res_col: usize,
    a: &A,
    b: &B,
    col_0: usize,
    col_1: usize,
) where
    R: VecZnxDftToMut<NTTIfma>,
    A: CnvPVecLToRef<NTTIfma>,
    B: CnvPVecRToRef<NTTIfma>,
{
    if col_0 == col_1 {
        unsafe { cnv_apply_dft_ifma(res, res_offset, res_col, a, col_0, b, col_1) };
        return;
    }

    let mut res = res.to_mut();
    let a = a.to_ref();
    let b = b.to_ref();

    let n = res.n();
    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.size();

    let bound = a_size + b_size - 1;
    let min_size = res_size.min(bound);
    let offset = res_offset.min(bound);
    // Values are in [0, 2q) per lane. After adding two: [0, 4q) < 2^42.
    // This fits within MADD52's 52-bit input window, so no intermediate
    // reduction is needed — the accumulation is still mathematically exact.
    for k in 0..min_size {
        let k_abs = k + offset;
        let j_min = k_abs.saturating_sub(a_size - 1);
        let j_max = (k_abs + 1).min(b_size);
        let j_count = j_max.saturating_sub(j_min);
        debug_assert!(
            j_count <= MAX_CNV_J_RANGE,
            "convolution j-range {} exceeds MAX_CNV_J_RANGE ({})",
            j_count,
            MAX_CNV_J_RANGE
        );

        let res_ptr = res.at_mut_ptr(res_col, k) as *mut __m256i;

        // Hoist at_ptr: precompute base pointers for all j values (4 pointers per j)
        let mut a0_bases = [core::ptr::null::<__m256i>(); MAX_CNV_J_RANGE];
        let mut a1_bases = [core::ptr::null::<__m256i>(); MAX_CNV_J_RANGE];
        let mut b0_bases = [core::ptr::null::<__m256i>(); MAX_CNV_J_RANGE];
        let mut b1_bases = [core::ptr::null::<__m256i>(); MAX_CNV_J_RANGE];
        for (idx, j) in (j_min..j_max).enumerate() {
            a0_bases[idx] = a.at_ptr(col_0, k_abs - j) as *const __m256i;
            a1_bases[idx] = a.at_ptr(col_1, k_abs - j) as *const __m256i;
            b0_bases[idx] = b.at_ptr(col_0, j) as *const __m256i;
            b1_bases[idx] = b.at_ptr(col_1, j) as *const __m256i;
        }

        // Process n_i in pairs via __m512i (2 coefficients per accumulator)
        let mut n_i = 0usize;
        while n_i + 2 <= n {
            let mut acc_lo = _mm512_setzero_si512();
            let mut acc_hi = _mm512_setzero_si512();
            for idx in 0..j_count {
                unsafe {
                    let a_sum = _mm512_add_epi64(
                        _mm512_loadu_si512(a0_bases[idx].add(n_i) as *const __m512i),
                        _mm512_loadu_si512(a1_bases[idx].add(n_i) as *const __m512i),
                    );
                    let b_sum = _mm512_add_epi64(
                        _mm512_loadu_si512(b0_bases[idx].add(n_i) as *const __m512i),
                        _mm512_loadu_si512(b1_bases[idx].add(n_i) as *const __m512i),
                    );
                    bbc_accumulate_ifma_512(&mut acc_lo, &mut acc_hi, a_sum, b_sum);
                }
            }
            unsafe {
                _mm512_storeu_si512(res_ptr.add(n_i) as *mut __m512i, reduce_bbc_ifma_simd_512(acc_lo, acc_hi));
            }
            n_i += 2;
        }

        // Handle odd tail
        if n_i < n {
            let mut acc_lo = _mm256_setzero_si256();
            let mut acc_hi = _mm256_setzero_si256();
            for idx in 0..j_count {
                unsafe {
                    let a_sum = _mm256_add_epi64(
                        _mm256_loadu_si256(a0_bases[idx].add(n_i)),
                        _mm256_loadu_si256(a1_bases[idx].add(n_i)),
                    );
                    let b_sum = _mm256_add_epi64(
                        _mm256_loadu_si256(b0_bases[idx].add(n_i)),
                        _mm256_loadu_si256(b1_bases[idx].add(n_i)),
                    );
                    bbc_accumulate_ifma(&mut acc_lo, &mut acc_hi, a_sum, b_sum);
                }
            }
            unsafe { _mm256_storeu_si256(res_ptr.add(n_i), reduce_bbc_ifma_simd(acc_lo, acc_hi)) };
        }
    }

    for j in min_size..res_size {
        res.at_mut(res_col, j).fill(Q120bScalar([0; 4]));
    }
}
