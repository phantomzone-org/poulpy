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

//! Q120 NTT forward and inverse — AVX2-accelerated kernels.
//!
//! Direct Rust port of `q120_ntt_avx2.c` from spqlios-arithmetic.
//!
//! Each q120b coefficient occupies exactly one `__m256i` (4 × u64, one u64 per
//! CRT prime).  The `powomega` twiddle tables from [`NttTable`] / [`NttTableInv`]
//! share the same 4-u64-per-entry layout and are used directly as `__m256i`
//! pointer arrays.
//!
//! # Algorithm
//!
//! Identical to the scalar reference in [`poulpy_hal::reference::ntt120::ntt`],
//! but the inner loops operate on 4 primes simultaneously via 256-bit SIMD.
//!
//! Split-precomputed multiplication:
//! ```text
//! split_precompmul(inp, po, h, mask)
//!   = (inp & mask) * (po & 0xFFFF_FFFF)          (low  halves, 32×32→64)
//!   + (inp >> h)   * (po >> 32)                   (high halves, 32×32→64)
//! ```
//! This avoids a full 64×64-bit multiply; the result is congruent to
//! `inp * ω mod Q` up to lazy-reduction multiples of Q.
//!
//! Lazy Barrett reduction:
//! ```text
//! modq_red(x, h, mask, cst)
//!   = (x & mask) + (x >> h) * cst
//! ```
//!
//! # Safety
//!
//! Both public functions require AVX2 support at runtime (ensured by the
//! [`NTT120Avx`](super::NTT120Avx) module constructor).

use core::arch::x86_64::{
    __m128i, __m256i, _mm_cvtsi64_si128, _mm256_add_epi64, _mm256_and_si256, _mm256_loadu_si256, _mm256_mul_epu32,
    _mm256_set1_epi64x, _mm256_srl_epi64, _mm256_srli_epi64, _mm256_storeu_si256, _mm256_sub_epi64,
};

use poulpy_hal::reference::ntt120::{
    ntt::{NttReducMeta, NttStepMeta, NttTable, NttTableInv},
    primes::PrimeSet,
};

/// Switch from level-order to block-order processing at this block size.
///
/// Matches `CHANGE_MODE_N` in `q120_ntt_avx2.c`.
const CHANGE_MODE_N: usize = 1024;

// ──────────────────────────────────────────────────────────────────────────────
// Inline AVX2 arithmetic helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Split-precomputed multiplication — 4 lanes in one `__m256i`.
///
/// Computes `(inp & mask) * (po & 0xFFFF_FFFF) + (inp >> h) * (po >> 32)`
/// for each of the 4 independent 64-bit lanes.
///
/// `h` is passed as a `__m128i` shift count (lower 64 bits = shift amount)
/// so that `_mm256_srl_epi64` (variable-count form) can be used.
///
/// Matches `split_precompmul_si256` in `q120_ntt_avx2.c`.
#[inline(always)]
unsafe fn split_precompmul_si256(inp: __m256i, po: __m256i, h: __m128i, mask: __m256i) -> __m256i {
    unsafe {
        let inp_low = _mm256_and_si256(inp, mask);
        let t1 = _mm256_mul_epu32(inp_low, po);
        let inp_high = _mm256_srl_epi64(inp, h);
        let po_high = _mm256_srli_epi64(po, 32); // constant shift — compile-time immediate
        let t2 = _mm256_mul_epu32(inp_high, po_high);
        _mm256_add_epi64(t1, t2)
    }
}

/// Lazy Barrett-style modular reduction — 4 lanes in one `__m256i`.
///
/// Computes `(x & mask) + (x >> h) * cst` for each 64-bit lane.
/// The result is congruent to `x mod Q[k]` with reduced bit width.
///
/// Matches `modq_red` in `q120_ntt_avx2.c`.
#[inline(always)]
unsafe fn modq_red_si256(x: __m256i, h: __m128i, mask: __m256i, cst: __m256i) -> __m256i {
    unsafe {
        let xh = _mm256_srl_epi64(x, h);
        let xl = _mm256_and_si256(x, mask);
        let xh_scaled = _mm256_mul_epu32(xh, cst);
        _mm256_add_epi64(xl, xh_scaled)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// NTT iteration kernels (private)
// ──────────────────────────────────────────────────────────────────────────────

/// Level-0 forward-NTT pass: multiply each coefficient `a[i]` by `ω^i`.
///
/// Each element of `data` and `powomega` is one q120b coefficient = one `__m256i`.
/// No butterfly — pure element-wise `split_precompmul`.
///
/// Matches `ntt_iter_first` in `q120_ntt_avx2.c`.
#[inline(always)]
unsafe fn ntt_iter_first(begin: *mut __m256i, end: *const __m256i, meta: &NttStepMeta, mut powomega: *const __m256i) {
    unsafe {
        let h = _mm_cvtsi64_si128(meta.half_bs as i64);
        let vmask = _mm256_set1_epi64x(meta.mask as i64);
        let mut data = begin;
        while (data as usize) < (end as usize) {
            let x = _mm256_loadu_si256(data);
            let po = _mm256_loadu_si256(powomega);
            _mm256_storeu_si256(data, split_precompmul_si256(x, po, h, vmask));
            data = data.add(1);
            powomega = powomega.add(1);
        }
    }
}

/// Level-0 forward-NTT pass with prior lazy Barrett reduction.
///
/// Like `ntt_iter_first` but each element is reduced via `modq_red` before
/// the `split_precompmul`.  Used as the final pass in the inverse NTT when
/// the accumulation bit-width would otherwise exceed 64 bits.
///
/// Matches `ntt_iter_first_red` in `q120_ntt_avx2.c`.
#[inline(always)]
unsafe fn ntt_iter_first_red(
    begin: *mut __m256i,
    end: *const __m256i,
    meta: &NttStepMeta,
    mut powomega: *const __m256i,
    reduc: &NttReducMeta,
) {
    unsafe {
        let h = _mm_cvtsi64_si128(meta.half_bs as i64);
        let vmask = _mm256_set1_epi64x(meta.mask as i64);
        let rh = _mm_cvtsi64_si128(reduc.h as i64);
        let rmask = _mm256_set1_epi64x(reduc.mask as i64);
        let rcst = _mm256_loadu_si256(reduc.modulo_red_cst.as_ptr() as *const __m256i);
        let mut data = begin;
        while (data as usize) < (end as usize) {
            let x = modq_red_si256(_mm256_loadu_si256(data), rh, rmask, rcst);
            let po = _mm256_loadu_si256(powomega);
            _mm256_storeu_si256(data, split_precompmul_si256(x, po, h, vmask));
            data = data.add(1);
            powomega = powomega.add(1);
        }
    }
}

/// Forward Cooley–Tukey (DIT) butterfly level of size `nn`, without reduction.
///
/// For each block of `nn` consecutive coefficients:
/// - `i=0`: `(a, b) → (a+b, a + q2bs - b)` — no twiddle.
/// - `i=1..halfnn-1`: `(a, b) → (a+b, split_precompmul(a + q2bs - b, ω^i))`.
///
/// Matches `ntt_iter` in `q120_ntt_avx2.c`.
#[inline(always)]
unsafe fn ntt_iter(nn: usize, begin: *mut __m256i, end: *const __m256i, meta: &NttStepMeta, powomega: *const __m256i) {
    unsafe {
        let halfnn = nn / 2;
        let vq2bs = _mm256_loadu_si256(meta.q2bs.as_ptr() as *const __m256i);
        let vmask = _mm256_set1_epi64x(meta.mask as i64);
        let h = _mm_cvtsi64_si128(meta.half_bs as i64);

        let mut data = begin;
        while (data as usize) < (end as usize) {
            let mut ptr1 = data;
            let mut ptr2 = data.add(halfnn);

            // i = 0: no twiddle
            let a = _mm256_loadu_si256(ptr1);
            let b = _mm256_loadu_si256(ptr2);
            _mm256_storeu_si256(ptr1, _mm256_add_epi64(a, b));
            _mm256_storeu_si256(ptr2, _mm256_sub_epi64(_mm256_add_epi64(a, vq2bs), b));
            ptr1 = ptr1.add(1);
            ptr2 = ptr2.add(1);

            // i = 1..halfnn-1: with twiddle on difference
            let mut po_ptr = powomega;
            for _ in 1..halfnn {
                let a = _mm256_loadu_si256(ptr1);
                let b = _mm256_loadu_si256(ptr2);
                _mm256_storeu_si256(ptr1, _mm256_add_epi64(a, b));
                let b1 = _mm256_sub_epi64(_mm256_add_epi64(a, vq2bs), b);
                let po = _mm256_loadu_si256(po_ptr);
                _mm256_storeu_si256(ptr2, split_precompmul_si256(b1, po, h, vmask));
                ptr1 = ptr1.add(1);
                ptr2 = ptr2.add(1);
                po_ptr = po_ptr.add(1);
            }
            data = data.add(nn);
        }
    }
}

/// Forward Cooley–Tukey butterfly level with prior lazy Barrett reduction.
///
/// Like `ntt_iter` but both `a` and `b` are reduced via `modq_red` before
/// each butterfly, preventing bit-width overflow across levels.
///
/// Matches `ntt_iter_red` in `q120_ntt_avx2.c`.
#[inline(always)]
unsafe fn ntt_iter_red(
    nn: usize,
    begin: *mut __m256i,
    end: *const __m256i,
    meta: &NttStepMeta,
    powomega: *const __m256i,
    reduc: &NttReducMeta,
) {
    unsafe {
        let halfnn = nn / 2;
        let vq2bs = _mm256_loadu_si256(meta.q2bs.as_ptr() as *const __m256i);
        let vmask = _mm256_set1_epi64x(meta.mask as i64);
        let h = _mm_cvtsi64_si128(meta.half_bs as i64);
        let rh = _mm_cvtsi64_si128(reduc.h as i64);
        let rmask = _mm256_set1_epi64x(reduc.mask as i64);
        let rcst = _mm256_loadu_si256(reduc.modulo_red_cst.as_ptr() as *const __m256i);

        let mut data = begin;
        while (data as usize) < (end as usize) {
            let mut ptr1 = data;
            let mut ptr2 = data.add(halfnn);

            // i = 0: no twiddle
            let a = modq_red_si256(_mm256_loadu_si256(ptr1), rh, rmask, rcst);
            let b = modq_red_si256(_mm256_loadu_si256(ptr2), rh, rmask, rcst);
            _mm256_storeu_si256(ptr1, _mm256_add_epi64(a, b));
            _mm256_storeu_si256(ptr2, _mm256_sub_epi64(_mm256_add_epi64(a, vq2bs), b));
            ptr1 = ptr1.add(1);
            ptr2 = ptr2.add(1);

            // i = 1..halfnn-1: with twiddle on difference
            let mut po_ptr = powomega;
            for _ in 1..halfnn {
                let a = modq_red_si256(_mm256_loadu_si256(ptr1), rh, rmask, rcst);
                let b = modq_red_si256(_mm256_loadu_si256(ptr2), rh, rmask, rcst);
                _mm256_storeu_si256(ptr1, _mm256_add_epi64(a, b));
                let b1 = _mm256_sub_epi64(_mm256_add_epi64(a, vq2bs), b);
                let po = _mm256_loadu_si256(po_ptr);
                _mm256_storeu_si256(ptr2, split_precompmul_si256(b1, po, h, vmask));
                ptr1 = ptr1.add(1);
                ptr2 = ptr2.add(1);
                po_ptr = po_ptr.add(1);
            }
            data = data.add(nn);
        }
    }
}

/// Inverse Gentleman–Sande (DIF) butterfly level, without reduction.
///
/// For each block of `nn` coefficients:
/// - `i=0`: `(a, b) → (a+b, a + q2bs - b)` — no twiddle.
/// - `i=1..halfnn-1`: twiddle applied to `b` **before** the butterfly:
///   `bo = split_precompmul(b, ω^{-i})`, then `(a, bo) → (a+bo, a + q2bs - bo)`.
///
/// Matches `intt_iter` in `q120_ntt_avx2.c`.
#[inline(always)]
unsafe fn intt_iter(nn: usize, begin: *mut __m256i, end: *const __m256i, meta: &NttStepMeta, powomega: *const __m256i) {
    unsafe {
        let halfnn = nn / 2;
        let vq2bs = _mm256_loadu_si256(meta.q2bs.as_ptr() as *const __m256i);
        let vmask = _mm256_set1_epi64x(meta.mask as i64);
        let h = _mm_cvtsi64_si128(meta.half_bs as i64);

        let mut data = begin;
        while (data as usize) < (end as usize) {
            let mut ptr1 = data;
            let mut ptr2 = data.add(halfnn);

            // i = 0: no twiddle
            let a = _mm256_loadu_si256(ptr1);
            let b = _mm256_loadu_si256(ptr2);
            _mm256_storeu_si256(ptr1, _mm256_add_epi64(a, b));
            _mm256_storeu_si256(ptr2, _mm256_sub_epi64(_mm256_add_epi64(a, vq2bs), b));
            ptr1 = ptr1.add(1);
            ptr2 = ptr2.add(1);

            // i = 1..halfnn-1: twiddle applied to b before butterfly
            let mut po_ptr = powomega;
            for _ in 1..halfnn {
                let a = _mm256_loadu_si256(ptr1);
                let b = _mm256_loadu_si256(ptr2);
                let po = _mm256_loadu_si256(po_ptr);
                let bo = split_precompmul_si256(b, po, h, vmask);
                _mm256_storeu_si256(ptr1, _mm256_add_epi64(a, bo));
                _mm256_storeu_si256(ptr2, _mm256_sub_epi64(_mm256_add_epi64(a, vq2bs), bo));
                ptr1 = ptr1.add(1);
                ptr2 = ptr2.add(1);
                po_ptr = po_ptr.add(1);
            }
            data = data.add(nn);
        }
    }
}

/// Inverse Gentleman–Sande butterfly level with prior lazy Barrett reduction.
///
/// Like `intt_iter` but both `a` and `b` are reduced via `modq_red` before
/// each butterfly.
///
/// Matches `intt_iter_red` in `q120_ntt_avx2.c`.
#[inline(always)]
unsafe fn intt_iter_red(
    nn: usize,
    begin: *mut __m256i,
    end: *const __m256i,
    meta: &NttStepMeta,
    powomega: *const __m256i,
    reduc: &NttReducMeta,
) {
    unsafe {
        let halfnn = nn / 2;
        let vq2bs = _mm256_loadu_si256(meta.q2bs.as_ptr() as *const __m256i);
        let vmask = _mm256_set1_epi64x(meta.mask as i64);
        let h = _mm_cvtsi64_si128(meta.half_bs as i64);
        let rh = _mm_cvtsi64_si128(reduc.h as i64);
        let rmask = _mm256_set1_epi64x(reduc.mask as i64);
        let rcst = _mm256_loadu_si256(reduc.modulo_red_cst.as_ptr() as *const __m256i);

        let mut data = begin;
        while (data as usize) < (end as usize) {
            let mut ptr1 = data;
            let mut ptr2 = data.add(halfnn);

            // i = 0: no twiddle
            let a = modq_red_si256(_mm256_loadu_si256(ptr1), rh, rmask, rcst);
            let b = modq_red_si256(_mm256_loadu_si256(ptr2), rh, rmask, rcst);
            _mm256_storeu_si256(ptr1, _mm256_add_epi64(a, b));
            _mm256_storeu_si256(ptr2, _mm256_sub_epi64(_mm256_add_epi64(a, vq2bs), b));
            ptr1 = ptr1.add(1);
            ptr2 = ptr2.add(1);

            // i = 1..halfnn-1: twiddle applied to b before butterfly
            let mut po_ptr = powomega;
            for _ in 1..halfnn {
                let a = modq_red_si256(_mm256_loadu_si256(ptr1), rh, rmask, rcst);
                let b = modq_red_si256(_mm256_loadu_si256(ptr2), rh, rmask, rcst);
                let po = _mm256_loadu_si256(po_ptr);
                let bo = split_precompmul_si256(b, po, h, vmask);
                _mm256_storeu_si256(ptr1, _mm256_add_epi64(a, bo));
                _mm256_storeu_si256(ptr2, _mm256_sub_epi64(_mm256_add_epi64(a, vq2bs), bo));
                ptr1 = ptr1.add(1);
                ptr2 = ptr2.add(1);
                po_ptr = po_ptr.add(1);
            }
            data = data.add(nn);
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Public: forward NTT
// ──────────────────────────────────────────────────────────────────────────────

/// Forward Q120 NTT — AVX2 accelerated.
///
/// Direct port of `q120_ntt_bb_avx2` from `q120_ntt_avx2.c`.
///
/// For large transforms (`n > CHANGE_MODE_N = 1024`), outer levels are processed
/// sequentially across the full array ("by-level"), then the innermost 1024-wide
/// blocks complete all remaining levels in a single pass ("by-block") to improve
/// cache locality.  For `n ≤ 1024` only the by-block phase runs.
///
/// `data` must be a `u64` slice of length `4 * table.n` in q120b layout.
///
/// # Safety
///
/// Caller must ensure AVX2 is available (guaranteed by `NTT120Avx` construction).
/// `data.len()` must be `>= 4 * table.n`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn ntt_avx2<P: PrimeSet>(table: &NttTable<P>, data: &mut [u64]) {
    let n = table.n;
    if n == 1 {
        return;
    }

    unsafe {
        let begin = data.as_mut_ptr() as *mut __m256i;
        let end = begin.add(n) as *const __m256i;
        let po_base = table.powomega.as_ptr() as *const __m256i;

        let mut meta_idx = 0usize;
        // po_avx: current offset into powomega, counted in __m256i units
        // (= groups of 4 u64, = one q120b entry).
        let mut po_avx = 0usize;

        // ── Level 0: a[i] *= ω^i (no butterfly) ──────────────────────────
        ntt_iter_first(begin, end, &table.level_metadata[meta_idx], po_base.add(po_avx));
        po_avx += n; // level 0 uses n entries
        meta_idx += 1;

        let split_nn = CHANGE_MODE_N.min(n);

        // ── By-level phase: nn = n, n/2, …, split_nn+1 ───────────────────
        let mut nn = n;
        while nn > split_nn {
            let halfnn = nn / 2;
            let meta = &table.level_metadata[meta_idx];
            if meta.reduce {
                ntt_iter_red(nn, begin, end, meta, po_base.add(po_avx), &table.reduc_metadata);
            } else {
                ntt_iter(nn, begin, end, meta, po_base.add(po_avx));
            }
            po_avx += halfnn.saturating_sub(1);
            meta_idx += 1;
            nn /= 2;
        }

        // ── By-block phase: process each split_nn-wide block independently ──
        if split_nn >= 2 {
            let meta_idx_saved = meta_idx;
            let po_avx_saved = po_avx;
            let mut it = begin;
            while (it as usize) < (end as usize) {
                let begin1 = it;
                let end1 = it.add(split_nn) as *const __m256i;
                meta_idx = meta_idx_saved;
                po_avx = po_avx_saved;
                let mut nn = split_nn;
                while nn >= 2 {
                    let halfnn = nn / 2;
                    let meta = &table.level_metadata[meta_idx];
                    if meta.reduce {
                        ntt_iter_red(nn, begin1, end1, meta, po_base.add(po_avx), &table.reduc_metadata);
                    } else {
                        ntt_iter(nn, begin1, end1, meta, po_base.add(po_avx));
                    }
                    po_avx += halfnn.saturating_sub(1);
                    meta_idx += 1;
                    nn /= 2;
                }
                it = it.add(split_nn);
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Public: inverse NTT
// ──────────────────────────────────────────────────────────────────────────────

/// Inverse Q120 NTT — AVX2 accelerated.
///
/// Direct port of `q120_intt_bb_avx2` from `q120_ntt_avx2.c`.
///
/// The inverse NTT reverses the forward butterfly order (Gentleman–Sande DIF)
/// and finalises with an element-wise multiply by `ω^{-i} * n^{-1}`, which is
/// baked into the last entry of `table.level_metadata` and `table.powomega`.
///
/// Cache-locality strategy mirrors the forward NTT: by-block for the inner
/// `split_nn` levels, by-level for the remaining outer levels.
///
/// `data` must be a `u64` slice of length `4 * table.n` in q120b layout.
///
/// # Safety
///
/// Caller must ensure AVX2 is available (guaranteed by `NTT120Avx` construction).
/// `data.len()` must be `>= 4 * table.n`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn intt_avx2<P: PrimeSet>(table: &NttTableInv<P>, data: &mut [u64]) {
    let n = table.n;
    if n == 1 {
        return;
    }

    unsafe {
        let begin = data.as_mut_ptr() as *mut __m256i;
        let end = begin.add(n) as *const __m256i;
        let po_base = table.powomega.as_ptr() as *const __m256i;

        let mut meta_idx = 0usize;
        let mut po_avx = 0usize;

        let split_nn = CHANGE_MODE_N.min(n);

        // ── By-block phase: levels nn = 2, 4, …, split_nn ────────────────
        if split_nn >= 2 {
            let meta_idx_saved = meta_idx;
            let po_avx_saved = po_avx;
            let mut it = begin;
            while (it as usize) < (end as usize) {
                let begin1 = it;
                let end1 = it.add(split_nn) as *const __m256i;
                meta_idx = meta_idx_saved;
                po_avx = po_avx_saved;
                let mut nn = 2usize;
                while nn <= split_nn {
                    let halfnn = nn / 2;
                    let meta = &table.level_metadata[meta_idx];
                    if meta.reduce {
                        intt_iter_red(nn, begin1, end1, meta, po_base.add(po_avx), &table.reduc_metadata);
                    } else {
                        intt_iter(nn, begin1, end1, meta, po_base.add(po_avx));
                    }
                    po_avx += halfnn.saturating_sub(1);
                    meta_idx += 1;
                    nn *= 2;
                }
                it = it.add(split_nn);
            }
        }

        // ── By-level phase: nn = 2*split_nn, …, n ────────────────────────
        let mut nn = 2 * split_nn;
        while nn <= n {
            let halfnn = nn / 2;
            let meta = &table.level_metadata[meta_idx];
            if meta.reduce {
                intt_iter_red(nn, begin, end, meta, po_base.add(po_avx), &table.reduc_metadata);
            } else {
                intt_iter(nn, begin, end, meta, po_base.add(po_avx));
            }
            po_avx += halfnn.saturating_sub(1);
            meta_idx += 1;
            nn *= 2;
        }

        // ── Last pass: a[i] *= ω^{-i} * n^{-1} ──────────────────────────
        let meta = &table.level_metadata[meta_idx];
        if meta.reduce {
            ntt_iter_first_red(begin, end, meta, po_base.add(po_avx), &table.reduc_metadata);
        } else {
            ntt_iter_first(begin, end, meta, po_base.add(po_avx));
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(all(test, target_feature = "avx2"))]
mod tests {
    use super::*;
    use poulpy_hal::reference::ntt120::{
        arithmetic::{b_from_znx64_ref, b_to_znx128_ref},
        ntt::{NttTable, NttTableInv, ntt_ref},
        primes::Primes30,
    };

    /// AVX2 NTT followed by AVX2 iNTT is the identity — mirrors the ref test.
    #[test]
    fn ntt_intt_identity_avx2() {
        for log_n in 1..=8usize {
            let n = 1 << log_n;
            let fwd = NttTable::<Primes30>::new(n);
            let inv = NttTableInv::<Primes30>::new(n);

            let coeffs: Vec<i64> = (0..n as i64).map(|i| (i * 7 + 3) % 201 - 100).collect();

            let mut data = vec![0u64; 4 * n];
            b_from_znx64_ref::<Primes30>(n, &mut data, &coeffs);

            let data_orig = data.clone();

            unsafe {
                ntt_avx2::<Primes30>(&fwd, &mut data);
                intt_avx2::<Primes30>(&inv, &mut data);
            }

            for i in 0..n {
                for k in 0..4 {
                    let orig = data_orig[4 * i + k] % Primes30::Q[k] as u64;
                    let got = data[4 * i + k] % Primes30::Q[k] as u64;
                    assert_eq!(orig, got, "n={n} i={i} k={k}: mismatch after AVX2 NTT+iNTT round-trip");
                }
            }
        }
    }

    /// AVX2 NTT-based convolution matches known result.
    ///
    /// a = [1, 2, 0, …], b = [3, 4, 0, …]; a*b mod (X^8+1) = [3, 10, 8, 0, …]
    #[test]
    fn ntt_convolution_avx2() {
        let n = 8usize;
        let fwd = NttTable::<Primes30>::new(n);
        let inv = NttTableInv::<Primes30>::new(n);

        let a: Vec<i64> = [1, 2, 0, 0, 0, 0, 0, 0].to_vec();
        let b: Vec<i64> = [3, 4, 0, 0, 0, 0, 0, 0].to_vec();

        let mut da = vec![0u64; 4 * n];
        let mut db = vec![0u64; 4 * n];
        b_from_znx64_ref::<Primes30>(n, &mut da, &a);
        b_from_znx64_ref::<Primes30>(n, &mut db, &b);

        unsafe {
            ntt_avx2::<Primes30>(&fwd, &mut da);
            ntt_avx2::<Primes30>(&fwd, &mut db);
        }

        // Pointwise multiply (mod each Q[k])
        let mut dc = vec![0u64; 4 * n];
        for i in 0..n {
            for k in 0..4 {
                let q = Primes30::Q[k] as u64;
                dc[4 * i + k] = (da[4 * i + k] % q * (db[4 * i + k] % q)) % q;
            }
        }

        unsafe {
            intt_avx2::<Primes30>(&inv, &mut dc);
        }

        let mut result = vec![0i128; n];
        b_to_znx128_ref::<Primes30>(n, &mut result, &dc);

        let expected: Vec<i128> = [3, 10, 8, 0, 0, 0, 0, 0].to_vec();
        assert_eq!(result, expected, "AVX2 NTT convolution mismatch");
    }

    /// AVX2 NTT output matches reference NTT output.
    #[test]
    fn ntt_avx2_vs_ref() {
        for log_n in 1..=8usize {
            let n = 1 << log_n;
            let fwd = NttTable::<Primes30>::new(n);

            let coeffs: Vec<i64> = (0..n as i64).map(|i| (i * 13 + 5) % 201 - 100).collect();

            let mut data_avx = vec![0u64; 4 * n];
            let mut data_ref = vec![0u64; 4 * n];
            b_from_znx64_ref::<Primes30>(n, &mut data_avx, &coeffs);
            b_from_znx64_ref::<Primes30>(n, &mut data_ref, &coeffs);

            unsafe { ntt_avx2::<Primes30>(&fwd, &mut data_avx) };
            ntt_ref::<Primes30>(&fwd, &mut data_ref);

            for i in 0..4 * n {
                assert_eq!(data_avx[i], data_ref[i], "n={n} idx={i}: NTT AVX2 vs ref mismatch");
            }
        }
    }
}
