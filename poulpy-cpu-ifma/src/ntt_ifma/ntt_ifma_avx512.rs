//! Raw AVX512-IFMA forward and inverse NTT kernels.
//!
//! These kernels are the core arithmetic engine of the IFMA backend.
//!
//! - Values stay in lazy modular ranges such as `[0, 2q)` throughout the transform.
//! - Harvey multiplication replaces the AVX2 split-precomputed multiply path.
//! - Two coefficients are processed at a time through 512-bit loads where profitable.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m256i, __m512i, _mm256_add_epi64, _mm256_and_si256, _mm256_cmpgt_epi64, _mm256_loadu_si256, _mm256_madd52hi_epu64,
    _mm256_madd52lo_epu64, _mm256_set1_epi64x, _mm256_setzero_si256, _mm256_slli_epi64, _mm256_storeu_si256, _mm256_sub_epi64,
    _mm512_add_epi64, _mm512_and_si512, _mm512_castsi256_si512, _mm512_cmpgt_epi64_mask, _mm512_inserti64x4, _mm512_loadu_si512,
    _mm512_madd52hi_epu64, _mm512_madd52lo_epu64, _mm512_mask_sub_epi64, _mm512_set1_epi64, _mm512_setzero_si512,
    _mm512_slli_epi64, _mm512_storeu_si512, _mm512_sub_epi64,
};

use std::mem::size_of;

use poulpy_cpu_ref::reference::ntt_ifma::{
    ntt::{NttIfmaTable, NttIfmaTableInv},
    primes::PrimeSetIfma,
};

// ──────────────────────────────────────────────────────────────────────────────
// SIMD arithmetic primitives
// ──────────────────────────────────────────────────────────────────────────────

/// Conditional subtract of 2q: if x >= 2q (unsigned), return x - 2q, else x.
///
/// Uses the MSB-flip trick for unsigned comparison via signed cmpgt:
/// `a >=_unsigned b`  iff  `(a ^ MSB) >=_signed (b ^ MSB)`.
#[inline]
#[target_feature(enable = "avx512vl")]
pub(crate) unsafe fn cond_sub_2q_si256(x: __m256i, q2: __m256i) -> __m256i {
    use core::arch::x86_64::_mm256_xor_si256;
    let msb = _mm256_set1_epi64x(i64::MIN);
    let x_flip = _mm256_xor_si256(x, msb);
    let q2_flip = _mm256_xor_si256(q2, msb);
    // cmpgt(q2_flip, x_flip) → all-ones where q2 > x (unsigned), i.e. x < q2
    let lt_mask = _mm256_cmpgt_epi64(q2_flip, x_flip);
    // If x < q2, don't subtract (mask is all-ones → andnot clears q2).
    // If x >= q2, subtract (mask is zero → andnot passes q2).
    use core::arch::x86_64::_mm256_andnot_si256;
    let sub_amount = _mm256_andnot_si256(lt_mask, q2); // q2 where x >= q2, 0 where x < q2
    _mm256_sub_epi64(x, sub_amount)
}

/// Harvey modular multiply — 4 lanes.
///
/// Input: `a ∈ [0, 2q)`, `omega ∈ [0, q)`.
/// Output: `r ∈ [0, 2q)` with `r ≡ a*omega (mod q)`.
///
/// `VPMADD52LUQ` / `VPMADD52HUQ` split the 104-bit product at bit 52, not bit 64.
/// We therefore reconstruct the low 64 bits as:
/// `prod_lo52 + ((prod_hi52 & ((1<<12)-1)) << 52)`.
#[inline]
#[target_feature(enable = "avx512ifma,avx512vl")]
pub(crate) unsafe fn mullo_u64_from_epu52(a: __m256i, b: __m256i) -> __m256i {
    let zero = _mm256_setzero_si256();
    let lo52 = _mm256_madd52lo_epu64(zero, a, b);
    let hi52 = _mm256_madd52hi_epu64(zero, a, b);
    let low12_mask = _mm256_set1_epi64x((1_i64 << 12) - 1);
    let hi12_shifted = _mm256_slli_epi64(_mm256_and_si256(hi52, low12_mask), 52);
    _mm256_add_epi64(lo52, hi12_shifted)
}

#[inline]
#[target_feature(enable = "avx512ifma,avx512vl")]
pub(crate) unsafe fn harvey_modmul_si256(a: __m256i, omega: __m256i, omega_quot: __m256i, q: __m256i) -> __m256i {
    unsafe {
        let zero = _mm256_setzero_si256();
        let qhat = _mm256_madd52hi_epu64(zero, a, omega_quot);
        let product_lo = mullo_u64_from_epu52(a, omega);
        let qhat_times_q = mullo_u64_from_epu52(qhat, q);
        let r = _mm256_sub_epi64(product_lo, qhat_times_q);
        // r might have wrapped negative (r > 2^63 as unsigned).
        // Since true result ∈ [0, 2q) and error is ±q, wrapped r looks like
        // a huge positive number.  Adding q fixes it.
        // Check: if r is "negative" (signed), add q.
        let neg_mask = _mm256_cmpgt_epi64(zero, r); // all-ones where r < 0 (signed)
        _mm256_add_epi64(r, _mm256_and_si256(q, neg_mask))
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// 512-bit wide primitives (2 CRT coefficients per __m512i)
// ──────────────────────────────────────────────────────────────────────────────

/// Conditional subtract of `q2` (= 2q) on 8 lanes (2 coefficients).
///
/// Uses AVX-512 native unsigned comparison via `_mm512_cmpgt_epi64_mask`
/// with the MSB-flip trick.
#[inline]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn cond_sub_2q_si512(x: __m512i, q2: __m512i) -> __m512i {
    // x < q2 (unsigned)? Use MSB-flip + signed compare.
    let msb = _mm512_set1_epi64(i64::MIN);
    let x_flip = _mm512_add_epi64(x, msb);
    let q2_flip = _mm512_add_epi64(q2, msb);
    let ge_mask = !_mm512_cmpgt_epi64_mask(q2_flip, x_flip); // lanes where x >= q2
    _mm512_mask_sub_epi64(x, ge_mask, x, q2)
}

/// Full 64-bit low product from IFMA 52-bit multiply — 8 lanes.
#[inline]
#[target_feature(enable = "avx512ifma")]
pub(crate) unsafe fn mullo_u64_from_epu52_512(a: __m512i, b: __m512i) -> __m512i {
    let zero = _mm512_setzero_si512();
    let lo52 = _mm512_madd52lo_epu64(zero, a, b);
    let hi52 = _mm512_madd52hi_epu64(zero, a, b);
    let low12_mask = _mm512_set1_epi64((1_i64 << 12) - 1);
    let hi12_shifted = _mm512_slli_epi64(_mm512_and_si512(hi52, low12_mask), 52);
    _mm512_add_epi64(lo52, hi12_shifted)
}

/// Harvey modular multiply — 8 lanes (2 coefficients).
#[inline]
#[target_feature(enable = "avx512ifma")]
pub(crate) unsafe fn harvey_modmul_si512(a: __m512i, omega: __m512i, omega_quot: __m512i, q: __m512i) -> __m512i {
    unsafe {
        let zero = _mm512_setzero_si512();
        let qhat = _mm512_madd52hi_epu64(zero, a, omega_quot);
        let product_lo = mullo_u64_from_epu52_512(a, omega);
        let qhat_times_q = mullo_u64_from_epu52_512(qhat, q);
        let r = _mm512_sub_epi64(product_lo, qhat_times_q);
        // If r is "negative" (signed), add q.
        let neg_mask = _mm512_cmpgt_epi64_mask(zero, r);
        use core::arch::x86_64::_mm512_mask_add_epi64;
        _mm512_mask_add_epi64(r, neg_mask, r, q)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// NTT butterfly kernels
// ──────────────────────────────────────────────────────────────────────────────

/// Pack two consecutive `__m256i` values into one `__m512i`.
#[inline(always)]
unsafe fn pack_512(lo: __m256i, hi: __m256i) -> __m512i {
    unsafe { _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo), hi) }
}

/// Level-0: `a[i] *= ω^i` using Harvey multiply.
/// Uses 512-bit main loop with split twiddle layout.
///
/// `po_omega`: pointer to contiguous ω values for this segment.
/// `po_quot`: pointer to contiguous ωq values for this segment.
#[target_feature(enable = "avx512ifma")]
unsafe fn ntt_iter_first_ifma(
    begin: *mut __m256i,
    end: *const __m256i,
    po_omega: *const __m256i,
    po_quot: *const __m256i,
    q: __m256i,
) {
    unsafe {
        let q_512 = pack_512(q, q);
        let n_coeffs = (end as usize - begin as usize) / size_of::<__m256i>();

        // 512-bit main loop: 2 coefficients at a time — single 512-bit load per twiddle
        let pairs = n_coeffs / 2;
        let data_512 = begin as *mut __m512i;
        let omega_512 = po_omega as *const __m512i;
        let quot_512 = po_quot as *const __m512i;
        let unrolled_pairs = pairs / 2;
        for i in 0..unrolled_pairs {
            let base = i * 2;
            let x0 = _mm512_loadu_si512(data_512.add(base));
            let omega0 = _mm512_loadu_si512(omega_512.add(base));
            let omega_quot0 = _mm512_loadu_si512(quot_512.add(base));
            let x1 = _mm512_loadu_si512(data_512.add(base + 1));
            let omega1 = _mm512_loadu_si512(omega_512.add(base + 1));
            let omega_quot1 = _mm512_loadu_si512(quot_512.add(base + 1));
            _mm512_storeu_si512(data_512.add(base), harvey_modmul_si512(x0, omega0, omega_quot0, q_512));
            _mm512_storeu_si512(data_512.add(base + 1), harvey_modmul_si512(x1, omega1, omega_quot1, q_512));
        }

        if !pairs.is_multiple_of(2) {
            let i = pairs - 1;
            let x = _mm512_loadu_si512(data_512.add(i));
            let omega = _mm512_loadu_si512(omega_512.add(i));
            let omega_quot = _mm512_loadu_si512(quot_512.add(i));
            _mm512_storeu_si512(data_512.add(i), harvey_modmul_si512(x, omega, omega_quot, q_512));
        }

        // 256-bit tail
        if !n_coeffs.is_multiple_of(2) {
            let idx = n_coeffs - 1;
            let x = _mm256_loadu_si256(begin.add(idx));
            let omega = _mm256_loadu_si256(po_omega.add(idx));
            let omega_quot = _mm256_loadu_si256(po_quot.add(idx));
            _mm256_storeu_si256(begin.add(idx), harvey_modmul_si256(x, omega, omega_quot, q));
        }
    }
}

/// Forward Cooley-Tukey butterfly with IFMA-native arithmetic.
/// Uses 512-bit inner loop with split twiddle layout.
///
/// All inputs and outputs in `[0, 2q)`.
#[target_feature(enable = "avx512ifma")]
unsafe fn ntt_iter_ifma(
    nn: usize,
    begin: *mut __m256i,
    end: *const __m256i,
    q: __m256i,
    q2: __m256i,
    po_omega: *const __m256i,
    po_quot: *const __m256i,
) {
    unsafe {
        let halfnn = nn / 2;
        let q_512 = pack_512(q, q);
        let q2_512 = pack_512(q2, q2);
        let mut data = begin;
        while (data as usize) < (end as usize) {
            let mut ptr1 = data;
            let mut ptr2 = data.add(halfnn);

            // i = 0: no twiddle
            {
                let a = _mm256_loadu_si256(ptr1);
                let b = _mm256_loadu_si256(ptr2);
                let sum = cond_sub_2q_si256(_mm256_add_epi64(a, b), q2);
                let diff = cond_sub_2q_si256(_mm256_sub_epi64(_mm256_add_epi64(a, q2), b), q2);
                _mm256_storeu_si256(ptr1, sum);
                _mm256_storeu_si256(ptr2, diff);
                ptr1 = ptr1.add(1);
                ptr2 = ptr2.add(1);
            }

            // i = 1..halfnn-1: Harvey multiply on difference (split layout)
            let remaining = halfnn - 1;

            // 512-bit pairs — single 512-bit load per twiddle
            let pairs = remaining / 2;
            let omega_512 = po_omega as *const __m512i;
            let quot_512 = po_quot as *const __m512i;
            let unrolled_pairs = pairs / 2;
            for p in 0..unrolled_pairs {
                let base = p * 2;

                let av0 = _mm512_loadu_si512(ptr1 as *const __m512i);
                let bv0 = _mm512_loadu_si512(ptr2 as *const __m512i);
                let omega0 = _mm512_loadu_si512(omega_512.add(base));
                let omega_quot0 = _mm512_loadu_si512(quot_512.add(base));
                let sum0 = cond_sub_2q_si512(_mm512_add_epi64(av0, bv0), q2_512);
                let diff0 = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(av0, q2_512), bv0), q2_512);

                let av1 = _mm512_loadu_si512(ptr1.add(2) as *const __m512i);
                let bv1 = _mm512_loadu_si512(ptr2.add(2) as *const __m512i);
                let omega1 = _mm512_loadu_si512(omega_512.add(base + 1));
                let omega_quot1 = _mm512_loadu_si512(quot_512.add(base + 1));
                let sum1 = cond_sub_2q_si512(_mm512_add_epi64(av1, bv1), q2_512);
                let diff1 = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(av1, q2_512), bv1), q2_512);

                _mm512_storeu_si512(ptr1 as *mut __m512i, sum0);
                _mm512_storeu_si512(ptr2 as *mut __m512i, harvey_modmul_si512(diff0, omega0, omega_quot0, q_512));
                _mm512_storeu_si512(ptr1.add(2) as *mut __m512i, sum1);
                _mm512_storeu_si512(
                    ptr2.add(2) as *mut __m512i,
                    harvey_modmul_si512(diff1, omega1, omega_quot1, q_512),
                );

                ptr1 = ptr1.add(4);
                ptr2 = ptr2.add(4);
            }

            if !pairs.is_multiple_of(2) {
                let tail_pair = pairs - 1;
                let av = _mm512_loadu_si512(ptr1 as *const __m512i);
                let bv = _mm512_loadu_si512(ptr2 as *const __m512i);
                let sum = cond_sub_2q_si512(_mm512_add_epi64(av, bv), q2_512);
                let diff = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(av, q2_512), bv), q2_512);
                let omega = _mm512_loadu_si512(omega_512.add(tail_pair));
                let omega_quot = _mm512_loadu_si512(quot_512.add(tail_pair));
                _mm512_storeu_si512(ptr1 as *mut __m512i, sum);
                _mm512_storeu_si512(ptr2 as *mut __m512i, harvey_modmul_si512(diff, omega, omega_quot, q_512));
                ptr1 = ptr1.add(2);
                ptr2 = ptr2.add(2);
            }

            // 256-bit tail
            if !remaining.is_multiple_of(2) {
                let tail_idx = pairs * 2;
                let a = _mm256_loadu_si256(ptr1);
                let b = _mm256_loadu_si256(ptr2);
                let sum = cond_sub_2q_si256(_mm256_add_epi64(a, b), q2);
                let diff = cond_sub_2q_si256(_mm256_sub_epi64(_mm256_add_epi64(a, q2), b), q2);
                let omega = _mm256_loadu_si256(po_omega.add(tail_idx));
                let omega_quot = _mm256_loadu_si256(po_quot.add(tail_idx));
                _mm256_storeu_si256(ptr1, sum);
                _mm256_storeu_si256(ptr2, harvey_modmul_si256(diff, omega, omega_quot, q));
            }
            data = data.add(nn);
        }
    }
}

/// Inverse Gentleman-Sande butterfly with IFMA-native arithmetic.
/// Uses 512-bit inner loop with split twiddle layout.
#[target_feature(enable = "avx512ifma")]
unsafe fn intt_iter_ifma(
    nn: usize,
    begin: *mut __m256i,
    end: *const __m256i,
    q: __m256i,
    q2: __m256i,
    po_omega: *const __m256i,
    po_quot: *const __m256i,
) {
    unsafe {
        let halfnn = nn / 2;
        let q_512 = pack_512(q, q);
        let q2_512 = pack_512(q2, q2);
        let mut data = begin;
        while (data as usize) < (end as usize) {
            let mut ptr1 = data;
            let mut ptr2 = data.add(halfnn);

            // i = 0: no twiddle
            {
                let a = _mm256_loadu_si256(ptr1);
                let b = _mm256_loadu_si256(ptr2);
                let sum = cond_sub_2q_si256(_mm256_add_epi64(a, b), q2);
                let diff = cond_sub_2q_si256(_mm256_sub_epi64(_mm256_add_epi64(a, q2), b), q2);
                _mm256_storeu_si256(ptr1, sum);
                _mm256_storeu_si256(ptr2, diff);
                ptr1 = ptr1.add(1);
                ptr2 = ptr2.add(1);
            }

            // i = 1..halfnn-1: twiddle on b BEFORE butterfly (split layout)
            let remaining = halfnn - 1;

            // 512-bit pairs — single 512-bit load per twiddle
            let pairs = remaining / 2;
            let omega_512 = po_omega as *const __m512i;
            let quot_512 = po_quot as *const __m512i;
            let unrolled_pairs = pairs / 2;
            for p in 0..unrolled_pairs {
                let base = p * 2;

                let av0 = _mm512_loadu_si512(ptr1 as *const __m512i);
                let bv0 = _mm512_loadu_si512(ptr2 as *const __m512i);
                let omega0 = _mm512_loadu_si512(omega_512.add(base));
                let omega_quot0 = _mm512_loadu_si512(quot_512.add(base));
                let bo0 = harvey_modmul_si512(bv0, omega0, omega_quot0, q_512);
                let sum0 = cond_sub_2q_si512(_mm512_add_epi64(av0, bo0), q2_512);
                let diff0 = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(av0, q2_512), bo0), q2_512);

                let av1 = _mm512_loadu_si512(ptr1.add(2) as *const __m512i);
                let bv1 = _mm512_loadu_si512(ptr2.add(2) as *const __m512i);
                let omega1 = _mm512_loadu_si512(omega_512.add(base + 1));
                let omega_quot1 = _mm512_loadu_si512(quot_512.add(base + 1));
                let bo1 = harvey_modmul_si512(bv1, omega1, omega_quot1, q_512);
                let sum1 = cond_sub_2q_si512(_mm512_add_epi64(av1, bo1), q2_512);
                let diff1 = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(av1, q2_512), bo1), q2_512);

                _mm512_storeu_si512(ptr1 as *mut __m512i, sum0);
                _mm512_storeu_si512(ptr2 as *mut __m512i, diff0);
                _mm512_storeu_si512(ptr1.add(2) as *mut __m512i, sum1);
                _mm512_storeu_si512(ptr2.add(2) as *mut __m512i, diff1);

                ptr1 = ptr1.add(4);
                ptr2 = ptr2.add(4);
            }

            if !pairs.is_multiple_of(2) {
                let tail_pair = pairs - 1;
                let av = _mm512_loadu_si512(ptr1 as *const __m512i);
                let bv = _mm512_loadu_si512(ptr2 as *const __m512i);
                let omega = _mm512_loadu_si512(omega_512.add(tail_pair));
                let omega_quot = _mm512_loadu_si512(quot_512.add(tail_pair));
                let bo = harvey_modmul_si512(bv, omega, omega_quot, q_512);
                let sum = cond_sub_2q_si512(_mm512_add_epi64(av, bo), q2_512);
                let diff = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(av, q2_512), bo), q2_512);
                _mm512_storeu_si512(ptr1 as *mut __m512i, sum);
                _mm512_storeu_si512(ptr2 as *mut __m512i, diff);
                ptr1 = ptr1.add(2);
                ptr2 = ptr2.add(2);
            }

            // 256-bit tail
            if !remaining.is_multiple_of(2) {
                let tail_idx = pairs * 2;
                let a = _mm256_loadu_si256(ptr1);
                let b = _mm256_loadu_si256(ptr2);
                let omega = _mm256_loadu_si256(po_omega.add(tail_idx));
                let omega_quot = _mm256_loadu_si256(po_quot.add(tail_idx));
                let bo = harvey_modmul_si256(b, omega, omega_quot, q);
                let sum = cond_sub_2q_si256(_mm256_add_epi64(a, bo), q2);
                let diff = cond_sub_2q_si256(_mm256_sub_epi64(_mm256_add_epi64(a, q2), bo), q2);
                _mm256_storeu_si256(ptr1, sum);
                _mm256_storeu_si256(ptr2, diff);
            }
            data = data.add(nn);
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Public: forward NTT
// ──────────────────────────────────────────────────────────────────────────────

/// Forward NTT — AVX512-IFMA accelerated, split twiddle layout.
///
/// All values kept in `[0, 2q)`.  No separate reduction passes.
#[target_feature(enable = "avx512ifma,avx512vl")]
pub(crate) unsafe fn ntt_ifma_avx512<P: PrimeSetIfma>(table: &NttIfmaTable<P>, data: &mut [u64]) {
    let n = table.n;
    if n == 1 {
        return;
    }

    unsafe {
        let begin = data.as_mut_ptr() as *mut __m256i;
        let end = begin.add(n) as *const __m256i;
        let po_base = table.powomega.as_ptr() as *const __m256i;

        let q = {
            let a = P::Q[0];
            let b = P::Q[1];
            let c = P::Q[2];
            use core::arch::x86_64::_mm256_set_epi64x;
            _mm256_set_epi64x(0, c as i64, b as i64, a as i64)
        };
        let q2 = _mm256_loadu_si256(table.q2.as_ptr() as *const __m256i);

        let mut seg_avx = 0usize; // segment base offset in __m256i units

        // Level 0: a[i] *= ω^i — split layout: n entries of ω, then n entries of ωq
        ntt_iter_first_ifma(begin, end, po_base.add(seg_avx), po_base.add(seg_avx + n), q);
        seg_avx += 2 * n;

        // Butterfly levels: nn = n, n/2, …, 2
        let mut nn = n;
        while nn >= 2 {
            let halfnn = nn / 2;
            if halfnn > 1 {
                let count = halfnn - 1;
                // Split layout: count entries of ω, then count entries of ωq
                ntt_iter_ifma(nn, begin, end, q, q2, po_base.add(seg_avx), po_base.add(seg_avx + count));
                seg_avx += 2 * count;
            } else {
                // nn == 2: add/sub only, no twiddle
                let mut ptr1 = begin;
                let mut ptr2 = begin.add(1);
                while (ptr1 as usize) < (end as usize) {
                    let a = _mm256_loadu_si256(ptr1);
                    let b = _mm256_loadu_si256(ptr2);
                    let sum = cond_sub_2q_si256(_mm256_add_epi64(a, b), q2);
                    let diff = cond_sub_2q_si256(_mm256_sub_epi64(_mm256_add_epi64(a, q2), b), q2);
                    _mm256_storeu_si256(ptr1, sum);
                    _mm256_storeu_si256(ptr2, diff);
                    ptr1 = ptr1.add(2);
                    ptr2 = ptr2.add(2);
                }
            }
            nn /= 2;
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Public: inverse NTT
// ──────────────────────────────────────────────────────────────────────────────

/// Inverse NTT — AVX512-IFMA accelerated, split twiddle layout.
#[target_feature(enable = "avx512ifma,avx512vl")]
pub(crate) unsafe fn intt_ifma_avx512<P: PrimeSetIfma>(table: &NttIfmaTableInv<P>, data: &mut [u64]) {
    let n = table.n;
    if n == 1 {
        return;
    }

    unsafe {
        let begin = data.as_mut_ptr() as *mut __m256i;
        let end = begin.add(n) as *const __m256i;
        let po_base = table.powomega.as_ptr() as *const __m256i;

        let q = {
            let a = P::Q[0];
            let b = P::Q[1];
            let c = P::Q[2];
            use core::arch::x86_64::_mm256_set_epi64x;
            _mm256_set_epi64x(0, c as i64, b as i64, a as i64)
        };
        let q2 = _mm256_loadu_si256(table.q2.as_ptr() as *const __m256i);

        let mut seg_avx = 0usize;

        // Butterfly levels: nn = 2, 4, …, n
        let mut nn = 2usize;
        while nn <= n {
            let halfnn = nn / 2;
            if halfnn > 1 {
                let count = halfnn - 1;
                intt_iter_ifma(nn, begin, end, q, q2, po_base.add(seg_avx), po_base.add(seg_avx + count));
                seg_avx += 2 * count;
            } else {
                // nn == 2: add/sub only
                let mut ptr1 = begin;
                let mut ptr2 = begin.add(1);
                while (ptr1 as usize) < (end as usize) {
                    let a = _mm256_loadu_si256(ptr1);
                    let b = _mm256_loadu_si256(ptr2);
                    let sum = cond_sub_2q_si256(_mm256_add_epi64(a, b), q2);
                    let diff = cond_sub_2q_si256(_mm256_sub_epi64(_mm256_add_epi64(a, q2), b), q2);
                    _mm256_storeu_si256(ptr1, sum);
                    _mm256_storeu_si256(ptr2, diff);
                    ptr1 = ptr1.add(2);
                    ptr2 = ptr2.add(2);
                }
            }
            nn *= 2;
        }

        // Last pass: a[i] *= ω^{-i} / n — split layout: n entries ω, then n entries ωq
        ntt_iter_first_ifma(begin, end, po_base.add(seg_avx), po_base.add(seg_avx + n), q);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use poulpy_cpu_ref::reference::ntt_ifma::{
        arithmetic::b_ifma_from_znx64_ref,
        ntt::{NttIfmaTable, NttIfmaTableInv, intt_ifma_ref, ntt_ifma_ref},
        primes::Primes40,
    };

    #[test]
    fn harvey_modmul_simd_vs_scalar() {
        use poulpy_cpu_ref::reference::ntt_ifma::ntt::{harvey_modmul, harvey_quotient};

        let q_arr = Primes40::Q;
        for &q in &q_arr {
            let omega = q / 2; // arbitrary twiddle
            let oq = harvey_quotient(omega, q);
            for &a in &[0u64, 1, q - 1, q, 2 * q - 1, q / 3, 42] {
                if a >= 2 * q {
                    continue;
                }

                let expected = harvey_modmul(a, omega, oq, q);

                // SIMD version: pack into lane 0
                let a_vec = [a as i64, 0i64, 0, 0];
                let o_vec = [omega as i64, 0i64, 0, 0];
                let oq_vec = [oq as i64, 0i64, 0, 0];
                let q_vec = [q as i64, 0i64, 0, 0];

                let got = unsafe {
                    let av = _mm256_loadu_si256(a_vec.as_ptr() as *const __m256i);
                    let ov = _mm256_loadu_si256(o_vec.as_ptr() as *const __m256i);
                    let oqv = _mm256_loadu_si256(oq_vec.as_ptr() as *const __m256i);
                    let qv = _mm256_loadu_si256(q_vec.as_ptr() as *const __m256i);
                    let r = harvey_modmul_si256(av, ov, oqv, qv);
                    let mut out = [0i64; 4];
                    _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, r);
                    out[0] as u64
                };

                assert_eq!(
                    got % q,
                    expected % q,
                    "SIMD harvey_modmul mismatch: a={a}, omega={omega}, q={q}, got={got}, expected={expected}"
                );
            }
        }
    }

    #[test]
    fn ntt_avx512_vs_ref_small_passes() {
        // Test small sizes to isolate the bug
        for log_n in 1..=7usize {
            let n = 1 << log_n;
            let fwd = NttIfmaTable::<Primes40>::new(n);
            let coeffs: Vec<i64> = (0..n as i64).map(|i| (i * 7 + 3) % 201 - 100).collect();
            let mut data_avx = vec![0u64; 4 * n];
            let mut data_ref = vec![0u64; 4 * n];
            b_ifma_from_znx64_ref(n, &mut data_avx, &coeffs);
            b_ifma_from_znx64_ref(n, &mut data_ref, &coeffs);
            unsafe { ntt_ifma_avx512::<Primes40>(&fwd, &mut data_avx) };
            ntt_ifma_ref::<Primes40>(&fwd, &mut data_ref);
            for i in 0..4 * n {
                assert_eq!(data_avx[i], data_ref[i], "n={n} idx={i}");
            }
            eprintln!("  forward n={n} OK");
        }
    }

    #[test]
    fn ntt_avx512_vs_ref() {
        for log_n in 1..=10usize {
            let n = 1 << log_n;
            let fwd = NttIfmaTable::<Primes40>::new(n);

            let coeffs: Vec<i64> = (0..n as i64).map(|i| (i * 7 + 3) % 201 - 100).collect();

            let mut data_avx = vec![0u64; 4 * n];
            let mut data_ref = vec![0u64; 4 * n];
            b_ifma_from_znx64_ref(n, &mut data_avx, &coeffs);
            b_ifma_from_znx64_ref(n, &mut data_ref, &coeffs);

            unsafe { ntt_ifma_avx512::<Primes40>(&fwd, &mut data_avx) };
            ntt_ifma_ref::<Primes40>(&fwd, &mut data_ref);

            for i in 0..4 * n {
                assert_eq!(
                    data_avx[i], data_ref[i],
                    "n={n} idx={i}: NTT AVX512 vs ref (avx={}, ref={})",
                    data_avx[i], data_ref[i]
                );
            }
        }
    }

    #[test]
    fn intt_avx512_vs_ref() {
        for log_n in 1..=10usize {
            let n = 1 << log_n;
            let fwd = NttIfmaTable::<Primes40>::new(n);
            let inv = NttIfmaTableInv::<Primes40>::new(n);

            let coeffs: Vec<i64> = (0..n as i64).map(|i| (i * 7 + 3) % 201 - 100).collect();
            let mut data = vec![0u64; 4 * n];
            b_ifma_from_znx64_ref(n, &mut data, &coeffs);
            ntt_ifma_ref::<Primes40>(&fwd, &mut data);

            let mut data_avx = data.clone();
            let mut data_ref = data.clone();

            unsafe { intt_ifma_avx512::<Primes40>(&inv, &mut data_avx) };
            intt_ifma_ref::<Primes40>(&inv, &mut data_ref);

            for i in 0..4 * n {
                assert_eq!(
                    data_avx[i], data_ref[i],
                    "n={n} idx={i}: iNTT AVX512 vs ref (avx={}, ref={})",
                    data_avx[i], data_ref[i]
                );
            }
        }
    }

    #[test]
    fn ntt_intt_avx512_roundtrip() {
        for log_n in 1..=10usize {
            let n = 1 << log_n;
            let fwd = NttIfmaTable::<Primes40>::new(n);
            let inv = NttIfmaTableInv::<Primes40>::new(n);

            let coeffs: Vec<i64> = (0..n as i64).map(|i| (i * 7 + 3) % 201 - 100).collect();
            let mut data = vec![0u64; 4 * n];
            b_ifma_from_znx64_ref(n, &mut data, &coeffs);
            let orig = data.clone();

            unsafe {
                ntt_ifma_avx512::<Primes40>(&fwd, &mut data);
                intt_ifma_avx512::<Primes40>(&inv, &mut data);
            }

            for i in 0..n {
                for k in 0..3 {
                    let o = orig[4 * i + k] % Primes40::Q[k];
                    let g = data[4 * i + k] % Primes40::Q[k];
                    assert_eq!(o, g, "n={n} i={i} k={k}: roundtrip mismatch");
                }
            }
        }
    }
}
