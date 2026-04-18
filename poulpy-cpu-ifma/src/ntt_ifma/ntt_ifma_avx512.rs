//! Raw AVX512-IFMA forward and inverse NTT kernels.
//!
//! These kernels are the core arithmetic engine of the IFMA backend.
//!
//! - Butterfly values live in `[0, 4q)`; a single final pass renormalises to `[0, 2q)`.
//! - Diff path feeds directly into Harvey without a pre-reduction (IFMA's 52-bit
//!   product absorbs inputs up to `2^52`).
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
/// Input: `a ∈ [0, 2^52)` (in practice up to `8q` under lazy reduction),
/// `omega ∈ [0, q)`.  Output: `r ∈ [0, 2q)` with `r ≡ a*omega (mod q)`.
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
        // `omega_quot = floor(ω·2^52/q)` rounds down, so `qhat` never overestimates.
        // Therefore `r = a·ω − qhat·q` is non-negative and lies in `[0, 2q)` for
        // any `a < 2^52` — no wraparound correction needed.
        _mm256_sub_epi64(product_lo, qhat_times_q)
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
        // `omega_quot = floor(ω·2^52/q)` rounds down, so `qhat` never overestimates.
        // `r = a·ω − qhat·q ∈ [0, 2q)` for `a < 2^52` — no wraparound correction.
        _mm512_sub_epi64(product_lo, qhat_times_q)
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

/// Forward Cooley-Tukey butterfly with IFMA-native lazy arithmetic.
/// Uses 512-bit inner loop with split twiddle layout.
///
/// All inputs and outputs in `[0, 4q)`.  Sum path subtracts `4q`; diff path
/// is fed directly into the Harvey multiply (which absorbs the reduction).
#[target_feature(enable = "avx512ifma")]
unsafe fn ntt_iter_ifma(
    nn: usize,
    begin: *mut __m256i,
    end: *const __m256i,
    q: __m256i,
    q4: __m256i,
    po_omega: *const __m256i,
    po_quot: *const __m256i,
) {
    unsafe {
        let halfnn = nn / 2;
        let q_512 = pack_512(q, q);
        let q4_512 = pack_512(q4, q4);
        let mut data = begin;
        while (data as usize) < (end as usize) {
            let mut ptr1 = data;
            let mut ptr2 = data.add(halfnn);

            // i = 0: no twiddle (both sides use cond_sub_4q)
            {
                let a = _mm256_loadu_si256(ptr1);
                let b = _mm256_loadu_si256(ptr2);
                let sum = cond_sub_2q_si256(_mm256_add_epi64(a, b), q4);
                let diff = cond_sub_2q_si256(_mm256_sub_epi64(_mm256_add_epi64(a, q4), b), q4);
                _mm256_storeu_si256(ptr1, sum);
                _mm256_storeu_si256(ptr2, diff);
                ptr1 = ptr1.add(1);
                ptr2 = ptr2.add(1);
            }

            // i = 1..halfnn-1: diff fed directly into Harvey (split layout)
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
                let sum0 = cond_sub_2q_si512(_mm512_add_epi64(av0, bv0), q4_512);
                let diff0 = _mm512_sub_epi64(_mm512_add_epi64(av0, q4_512), bv0);

                let av1 = _mm512_loadu_si512(ptr1.add(2) as *const __m512i);
                let bv1 = _mm512_loadu_si512(ptr2.add(2) as *const __m512i);
                let omega1 = _mm512_loadu_si512(omega_512.add(base + 1));
                let omega_quot1 = _mm512_loadu_si512(quot_512.add(base + 1));
                let sum1 = cond_sub_2q_si512(_mm512_add_epi64(av1, bv1), q4_512);
                let diff1 = _mm512_sub_epi64(_mm512_add_epi64(av1, q4_512), bv1);

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
                let sum = cond_sub_2q_si512(_mm512_add_epi64(av, bv), q4_512);
                let diff = _mm512_sub_epi64(_mm512_add_epi64(av, q4_512), bv);
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
                let sum = cond_sub_2q_si256(_mm256_add_epi64(a, b), q4);
                let diff = _mm256_sub_epi64(_mm256_add_epi64(a, q4), b);
                let omega = _mm256_loadu_si256(po_omega.add(tail_idx));
                let omega_quot = _mm256_loadu_si256(po_quot.add(tail_idx));
                _mm256_storeu_si256(ptr1, sum);
                _mm256_storeu_si256(ptr2, harvey_modmul_si256(diff, omega, omega_quot, q));
            }
            data = data.add(nn);
        }
    }
}

/// Inverse Gentleman-Sande butterfly with IFMA-native lazy arithmetic.
/// Uses 512-bit inner loop with split twiddle layout.
///
/// All inputs and outputs in `[0, 4q)`.  `b_raw ∈ [0, 4q)` is fed directly into
/// Harvey (output `∈ [0, 2q)`); sum/diff use `cond_sub_4q`.
#[target_feature(enable = "avx512ifma")]
unsafe fn intt_iter_ifma(
    nn: usize,
    begin: *mut __m256i,
    end: *const __m256i,
    q: __m256i,
    q4: __m256i,
    po_omega: *const __m256i,
    po_quot: *const __m256i,
) {
    unsafe {
        let halfnn = nn / 2;
        let q_512 = pack_512(q, q);
        let q4_512 = pack_512(q4, q4);
        let mut data = begin;
        while (data as usize) < (end as usize) {
            let mut ptr1 = data;
            let mut ptr2 = data.add(halfnn);

            // i = 0: no twiddle
            {
                let a = _mm256_loadu_si256(ptr1);
                let b = _mm256_loadu_si256(ptr2);
                let sum = cond_sub_2q_si256(_mm256_add_epi64(a, b), q4);
                let diff = cond_sub_2q_si256(_mm256_sub_epi64(_mm256_add_epi64(a, q4), b), q4);
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
                let sum0 = cond_sub_2q_si512(_mm512_add_epi64(av0, bo0), q4_512);
                let diff0 = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(av0, q4_512), bo0), q4_512);

                let av1 = _mm512_loadu_si512(ptr1.add(2) as *const __m512i);
                let bv1 = _mm512_loadu_si512(ptr2.add(2) as *const __m512i);
                let omega1 = _mm512_loadu_si512(omega_512.add(base + 1));
                let omega_quot1 = _mm512_loadu_si512(quot_512.add(base + 1));
                let bo1 = harvey_modmul_si512(bv1, omega1, omega_quot1, q_512);
                let sum1 = cond_sub_2q_si512(_mm512_add_epi64(av1, bo1), q4_512);
                let diff1 = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(av1, q4_512), bo1), q4_512);

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
                let sum = cond_sub_2q_si512(_mm512_add_epi64(av, bo), q4_512);
                let diff = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(av, q4_512), bo), q4_512);
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
                let sum = cond_sub_2q_si256(_mm256_add_epi64(a, bo), q4);
                let diff = cond_sub_2q_si256(_mm256_sub_epi64(_mm256_add_epi64(a, q4), bo), q4);
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

/// Algorithmic block size for the blocked NTT.  Once butterfly size drops to
/// this many coefficients, we stop sweeping all data per level and instead
/// run the remaining levels fully inside each block before moving on.
///
/// Chosen as a generic algorithmic constant (not tuned to a specific CPU's
/// L1).  At `32` coefficients = 4 u64 = 1KiB per CRT instance, easily fits
/// the smallest L1 caches on any modern x86-64 processor.
const NTT_BLOCK: usize = 32;

/// Forward NTT — AVX512-IFMA accelerated, split twiddle layout.
///
/// Butterfly values live in `[0, 4q)`; a final pass renormalises to `[0, 2q)`.
/// Butterfly levels larger than `NTT_BLOCK` run breadth-first; inner levels
/// (≤ `NTT_BLOCK`) are performed block-by-block, keeping the working set in
/// cache across all remaining levels.
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
        let q4 = _mm256_loadu_si256(table.q4.as_ptr() as *const __m256i);

        let mut seg_avx = 0usize;

        // Level 0: a[i] *= ω^i.
        ntt_iter_first_ifma(begin, end, po_base.add(seg_avx), po_base.add(seg_avx + n), q);
        seg_avx += 2 * n;

        // Upper butterfly levels (breadth-first) while nn > NTT_BLOCK.
        let block = NTT_BLOCK.min(n);
        let mut nn = n;
        while nn > block {
            let halfnn = nn / 2;
            let count = halfnn - 1;
            ntt_iter_ifma(nn, begin, end, q, q4, po_base.add(seg_avx), po_base.add(seg_avx + count));
            seg_avx += 2 * count;
            nn /= 2;
        }

        // Precompute segment offsets for each remaining level (nn, nn/2, …, 2).
        let mut inner_segs = [0usize; 17];
        let mut inner_nn = [0usize; 17];
        let mut num_inner = 0usize;
        {
            let mut m = nn;
            let mut s = seg_avx;
            while m >= 2 {
                inner_nn[num_inner] = m;
                inner_segs[num_inner] = s;
                let halfm = m / 2;
                if halfm > 1 {
                    s += 2 * (halfm - 1);
                }
                m /= 2;
                num_inner += 1;
            }
        }

        // Inner levels (depth-first by block): run the whole remaining level
        // sequence inside each block before moving to the next.  Each block
        // is `nn = NTT_BLOCK` coefficients; subsequent levels subdivide it.
        let mut blk_start = 0usize;
        while blk_start < n {
            let blk_begin = begin.add(blk_start);
            let blk_end = begin.add(blk_start + nn) as *const __m256i;
            for i in 0..num_inner {
                let m = inner_nn[i];
                let halfm = m / 2;
                let seg = inner_segs[i];
                if halfm > 1 {
                    let count = halfm - 1;
                    ntt_iter_ifma(m, blk_begin, blk_end, q, q4, po_base.add(seg), po_base.add(seg + count));
                } else {
                    // m == 2: add/sub only, no twiddle.
                    let mut p1 = blk_begin;
                    let mut p2 = blk_begin.add(1);
                    while (p1 as usize) < (blk_end as usize) {
                        let a = _mm256_loadu_si256(p1);
                        let b = _mm256_loadu_si256(p2);
                        let sum = cond_sub_2q_si256(_mm256_add_epi64(a, b), q4);
                        let diff = cond_sub_2q_si256(_mm256_sub_epi64(_mm256_add_epi64(a, q4), b), q4);
                        _mm256_storeu_si256(p1, sum);
                        _mm256_storeu_si256(p2, diff);
                        p1 = p1.add(2);
                        p2 = p2.add(2);
                    }
                }
            }
            blk_start += nn;
        }

        // Final normalisation: [0, 4q) → [0, 2q).  n is always a power of two
        // ≥ 2 here, so iterating 512-bit (= 2 coefficients of 4×u64) is safe.
        let q2_512 = pack_512(q2, q2);
        let ptr_512 = begin as *mut __m512i;
        let chunks = n / 2;
        for i in 0..chunks {
            let x = _mm512_loadu_si512(ptr_512.add(i));
            _mm512_storeu_si512(ptr_512.add(i), cond_sub_2q_si512(x, q2_512));
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Public: inverse NTT
// ──────────────────────────────────────────────────────────────────────────────

/// Inverse NTT — AVX512-IFMA accelerated, split twiddle layout.
///
/// Butterfly values live in `[0, 4q)`.  The final pointwise Harvey pass reduces
/// to `[0, 2q)` automatically.  Inner levels (≤ `NTT_BLOCK`) are performed
/// block-by-block to keep the working set in cache across all levels.
#[target_feature(enable = "avx512ifma,avx512vl")]
#[inline]
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
        let q4 = _mm256_loadu_si256(table.q4.as_ptr() as *const __m256i);

        let block = NTT_BLOCK.min(n);

        // Precompute segment offsets for inner (block-local) levels: nn = 2, 4, …, block.
        let mut inner_segs = [0usize; 17];
        let mut inner_nn = [0usize; 17];
        let mut num_inner = 0usize;
        let mut seg_avx = 0usize;
        {
            let mut m = 2usize;
            while m <= block {
                inner_nn[num_inner] = m;
                inner_segs[num_inner] = seg_avx;
                let halfm = m / 2;
                if halfm > 1 {
                    seg_avx += 2 * (halfm - 1);
                }
                m *= 2;
                num_inner += 1;
            }
        }

        // Inner levels (block-local): run nn = 2, 4, …, block fully inside each block
        // before moving on.  Data stays hot in cache across all these levels.
        let mut blk_start = 0usize;
        while blk_start < n {
            let blk_begin = begin.add(blk_start);
            let blk_end = begin.add(blk_start + block) as *const __m256i;
            for i in 0..num_inner {
                let m = inner_nn[i];
                let halfm = m / 2;
                let seg = inner_segs[i];
                if halfm > 1 {
                    let count = halfm - 1;
                    intt_iter_ifma(m, blk_begin, blk_end, q, q4, po_base.add(seg), po_base.add(seg + count));
                } else {
                    // m == 2: add/sub only.
                    let mut p1 = blk_begin;
                    let mut p2 = blk_begin.add(1);
                    while (p1 as usize) < (blk_end as usize) {
                        let a = _mm256_loadu_si256(p1);
                        let b = _mm256_loadu_si256(p2);
                        let sum = cond_sub_2q_si256(_mm256_add_epi64(a, b), q4);
                        let diff = cond_sub_2q_si256(_mm256_sub_epi64(_mm256_add_epi64(a, q4), b), q4);
                        _mm256_storeu_si256(p1, sum);
                        _mm256_storeu_si256(p2, diff);
                        p1 = p1.add(2);
                        p2 = p2.add(2);
                    }
                }
            }
            blk_start += block;
        }

        // Outer butterfly levels (breadth-first) for nn > block.
        let mut nn = block * 2;
        while nn <= n {
            let halfnn = nn / 2;
            let count = halfnn - 1;
            intt_iter_ifma(nn, begin, end, q, q4, po_base.add(seg_avx), po_base.add(seg_avx + count));
            seg_avx += 2 * count;
            nn *= 2;
        }

        // Last pass: a[i] *= ω^{-i} / n — Harvey absorbs [0, 4q) → [0, 2q).
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
