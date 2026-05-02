//! NTT precomputation and reference execution for the 3-prime IFMA backend.
//!
//! # IFMA-native arithmetic model
//!
//! Lazy Harvey reduction: butterfly values are kept in `[0, 4q)` internally,
//! and normalised to `[0, 2q)` at NTT boundaries.  On the difference path of
//! each butterfly the Harvey multiplier absorbs the wider range directly —
//! inputs up to `2^52` yield outputs in `[0, 2q)` because `q < 2^42` — so a
//! pre-reduction `cond_sub` before the multiply is unnecessary.  Only the sum
//! path keeps one `cond_sub` (of `4q`) per butterfly pair.
//!
//! # Twiddle factor layout
//!
//! Twiddle factors use a **split (SoA) layout** within each NTT level segment.
//! For a segment with `m` entries, the layout is:
//! - `[ω₀, ω₁, ..., ωₘ₋₁]` — all twiddle values (4 u64 each)
//! - `[ωq₀, ωq₁, ..., ωqₘ₋₁]` — all Harvey quotients (4 u64 each)
//!
//! This enables AVX-512 kernels to load 2 consecutive ω or ωq values with a
//! single 512-bit load, instead of deinterleaving from `[ω, ωq]` pairs.
//!
//! # Data layout
//!
//! Each coefficient = 4 × u64 (3 active CRT residues + 1 padding).
//! All residues are kept in `[0, 2q)` throughout the NTT.

use std::marker::PhantomData;

use poulpy_hal::alloc_aligned;

use super::primes::{PrimeSetIfma, modq_pow64};

// ──────────────────────────────────────────────────────────────────────────────
// Precomputation data structures
// ──────────────────────────────────────────────────────────────────────────────

/// Precomputed twiddle-factor table for the forward NTT (3-prime IFMA).
///
/// No per-level metadata is needed — the IFMA-native butterfly keeps all
/// values in `[0, 2q)` without explicit reduction levels.
pub struct NttIfmaTable<P: PrimeSetIfma> {
    /// NTT size (power of two, ≤ 2^16).
    pub n: usize,
    /// `2q[k]` for each prime (lane 3 = 0).  Used for the final `[0, 4q)` → `[0, 2q)`
    /// normalisation pass and by external consumers that expect `[0, 2q)` input.
    pub q2: [u64; 4],
    /// `4q[k]` for each prime (lane 3 = 0).  Used inside butterflies under the
    /// lazy `[0, 4q)` invariant: sum path subtracts `4q`, diff path adds `4q`
    /// before subtracting `b`.
    pub q4: [u64; 4],
    /// Packed twiddle factors: each entry is 8 u64.
    /// Layout: level-0 (n entries), then butterfly levels (halfnn-1 entries each).
    pub powomega: Vec<u64>,
    _phantom: PhantomData<P>,
}

/// Precomputed twiddle-factor table for the inverse NTT (3-prime IFMA).
pub struct NttIfmaTableInv<P: PrimeSetIfma> {
    pub n: usize,
    pub q2: [u64; 4],
    pub q4: [u64; 4],
    /// Packed twiddle factors: butterfly levels (halfnn-1 entries each),
    /// then last-pass (n entries with ω^{-i}/n baked in).
    pub powomega: Vec<u64>,
    _phantom: PhantomData<P>,
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Returns the primitive `2n`-th roots of unity for each of the 3 primes.
fn fill_omegas_ifma<P: PrimeSetIfma>(n: usize) -> [u64; 3] {
    debug_assert!((1..=(1 << 16)).contains(&n), "n must be a power of two in [1, 2^16], got {n}");
    std::array::from_fn(|k| modq_pow64(P::OMEGA[k], (1i64 << 16) / n as i64, P::Q[k]))
}

/// Compute Harvey quotient: `floor(omega * 2^52 / q)`.
#[inline(always)]
pub fn harvey_quotient(omega: u64, q: u64) -> u64 {
    ((omega as u128 * (1u128 << 52)) / q as u128) as u64
}

/// Harvey modular multiply (scalar): `a * omega mod q`.
///
/// Input: `a ∈ [0, 2^52)` (in practice up to `4q` or `8q` under lazy),
/// `omega ∈ [0, q)`.  Output: `r ∈ [0, 2q)` with `r ≡ a*omega (mod q)`.
///
/// Because `omega_quot = floor(omega * 2^52 / q)` rounds down, the computed
/// `qhat` is always `≤ floor(a*omega/q)` (never an overestimate), so the raw
/// remainder `r = a*omega - qhat*q` is non-negative.  It lies in `[0, 2q)`
/// whenever `a < 2^52`, which covers all lazy-reduction ranges we use.
#[inline(always)]
pub fn harvey_modmul(a: u64, omega: u64, omega_quot: u64, q: u64) -> u64 {
    let qhat = ((a as u128 * omega_quot as u128) >> 52) as u64;
    let product_lo = (a as u128 * omega as u128) as u64; // low 64 bits (we only need mod 2^64)
    product_lo.wrapping_sub(qhat.wrapping_mul(q))
}

/// Conditional subtract: if `x >= 2q`, return `x - 2q`, else `x`.
/// Keeps values in `[0, 2q)`.
#[inline(always)]
fn cond_sub_2q(x: u64, q2: u64) -> u64 {
    if x >= q2 { x - q2 } else { x }
}

/// Store a twiddle entry into the split powomega array.
///
/// `omega_base`: start of the ω section for this level segment.
/// `quot_base`: start of the ωq section for this level segment.
/// `idx`: index of this entry within the segment.
fn store_twiddle_split<P: PrimeSetIfma>(
    powomega: &mut [u64],
    omega_base: usize,
    quot_base: usize,
    idx: usize,
    omega_vals: &[u64; 3],
) {
    let o = omega_base + 4 * idx;
    let q = quot_base + 4 * idx;
    for k in 0..3 {
        powomega[o + k] = omega_vals[k];
        powomega[q + k] = harvey_quotient(omega_vals[k], P::Q[k]);
    }
    powomega[o + 3] = 0;
    powomega[q + 3] = 0;
}

// ──────────────────────────────────────────────────────────────────────────────
// Forward NTT table construction
// ──────────────────────────────────────────────────────────────────────────────

impl<P: PrimeSetIfma> NttIfmaTable<P> {
    pub fn new(n: usize) -> Self {
        assert!(
            n.is_power_of_two() && n <= (1 << 16),
            "NTT size must be a power of two ≤ 2^16, got {n}"
        );

        let q2: [u64; 4] = [2 * P::Q[0], 2 * P::Q[1], 2 * P::Q[2], 0];
        let q4: [u64; 4] = [4 * P::Q[0], 4 * P::Q[1], 4 * P::Q[2], 0];

        let omega_vec = fill_omegas_ifma::<P>(n);

        // Allocate powomega: level-0 needs n entries, butterfly levels need sum of (halfnn-1)
        let total_entries = n
            + (0..)
                .scan(n, |nn, _| {
                    if *nn < 2 {
                        return None;
                    }
                    let h = *nn / 2;
                    *nn /= 2;
                    Some(h.saturating_sub(1))
                })
                .sum::<usize>();

        // Split layout: each segment has m entries of ω (4 u64 each) then m entries of ωq (4 u64 each)
        // Total u64 count is same: 8 * total_entries
        let mut powomega: Vec<u64> = alloc_aligned::<u64>(8 * total_entries);
        powomega.resize(8 * total_entries, 0);
        let mut seg_base = 0usize; // base offset (in u64) for current segment

        if n <= 1 {
            return Self {
                n,
                q2,
                q4,
                powomega,
                _phantom: PhantomData,
            };
        }

        // ── Level 0: a[i] *= ω^i (n entries) ────────────────────────────
        {
            let omega_base = seg_base;
            let quot_base = seg_base + 4 * n;
            let mut pow_om: [u64; 3] = [1; 3]; // ω^0 = 1
            for i in 0..n {
                store_twiddle_split::<P>(&mut powomega, omega_base, quot_base, i, &pow_om);
                for k in 0..3 {
                    pow_om[k] = ((pow_om[k] as u128 * omega_vec[k] as u128) % P::Q[k] as u128) as u64;
                }
            }
            seg_base += 8 * n;
        }

        // ── Butterfly levels: nn = n, n/2, …, 2 ─────────────────────────
        let mut nn = n;
        while nn >= 2 {
            let halfnn = nn / 2;
            if halfnn > 1 {
                let count = halfnn - 1;
                let omega_base = seg_base;
                let quot_base = seg_base + 4 * count;
                let m = n / halfnn;
                let omega_m: [u64; 3] = std::array::from_fn(|k| modq_pow64(omega_vec[k], m as i64, P::Q[k]));
                let mut pow_om = omega_m;
                for i in 0..count {
                    store_twiddle_split::<P>(&mut powomega, omega_base, quot_base, i, &pow_om);
                    for k in 0..3 {
                        pow_om[k] = ((pow_om[k] as u128 * omega_m[k] as u128) % P::Q[k] as u128) as u64;
                    }
                }
                seg_base += 8 * count;
            }
            nn /= 2;
        }

        Self {
            n,
            q2,
            q4,
            powomega,
            _phantom: PhantomData,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Inverse NTT table construction
// ──────────────────────────────────────────────────────────────────────────────

impl<P: PrimeSetIfma> NttIfmaTableInv<P> {
    pub fn new(n: usize) -> Self {
        assert!(
            n.is_power_of_two() && n <= (1 << 16),
            "NTT size must be a power of two ≤ 2^16, got {n}"
        );

        let q2: [u64; 4] = [2 * P::Q[0], 2 * P::Q[1], 2 * P::Q[2], 0];
        let q4: [u64; 4] = [4 * P::Q[0], 4 * P::Q[1], 4 * P::Q[2], 0];
        let omega_vec = fill_omegas_ifma::<P>(n);

        // butterfly levels + last pass (n entries)
        let total_entries = n
            + (0..)
                .scan(2usize, |nn, _| {
                    if *nn > n {
                        return None;
                    }
                    let h = *nn / 2;
                    *nn *= 2;
                    Some(h.saturating_sub(1))
                })
                .sum::<usize>();

        let mut powomega: Vec<u64> = alloc_aligned::<u64>(8 * total_entries);
        powomega.resize(8 * total_entries, 0);
        let mut seg_base = 0usize;

        if n <= 1 {
            return Self {
                n,
                q2,
                q4,
                powomega,
                _phantom: PhantomData,
            };
        }

        // ── Butterfly levels: nn = 2, 4, …, n ───────────────────────────
        let mut nn = 2usize;
        while nn <= n {
            let halfnn = nn / 2;
            if halfnn > 1 {
                let count = halfnn - 1;
                let omega_base = seg_base;
                let quot_base = seg_base + 4 * count;
                let m = n / halfnn;
                let omega_neg_m: [u64; 3] = std::array::from_fn(|k| modq_pow64(omega_vec[k], -(m as i64), P::Q[k]));
                let mut pow_om = omega_neg_m;
                for i in 0..count {
                    store_twiddle_split::<P>(&mut powomega, omega_base, quot_base, i, &pow_om);
                    for k in 0..3 {
                        pow_om[k] = ((pow_om[k] as u128 * omega_neg_m[k] as u128) % P::Q[k] as u128) as u64;
                    }
                }
                seg_base += 8 * count;
            }
            nn *= 2;
        }

        // ── Last pass: ω^{-i} / n (n entries) ──────────────────────────
        {
            let omega_base = seg_base;
            let quot_base = seg_base + 4 * n;
            let omega_inv: [u64; 3] = std::array::from_fn(|k| modq_pow64(omega_vec[k], -1, P::Q[k]));
            let n_inv: [u64; 3] = std::array::from_fn(|k| modq_pow64(n as u64, -1, P::Q[k]));
            let mut pow_om = n_inv; // i=0: just n^{-1}
            for i in 0..n {
                store_twiddle_split::<P>(&mut powomega, omega_base, quot_base, i, &pow_om);
                for k in 0..3 {
                    pow_om[k] = ((pow_om[k] as u128 * omega_inv[k] as u128) % P::Q[k] as u128) as u64;
                }
            }
        }

        Self {
            n,
            q2,
            q4,
            powomega,
            _phantom: PhantomData,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Reference scalar NTT execution (IFMA-native arithmetic)
// ──────────────────────────────────────────────────────────────────────────────

/// Forward NTT — scalar reference with IFMA-native lazy arithmetic.
///
/// Butterfly values live in `[0, 4q)`.  Diff path feeds directly into Harvey
/// without a pre-reduction; sum path keeps one `cond_sub(·, 4q)`.  A final
/// `cond_sub(·, 2q)` pass renormalises the output to `[0, 2q)` so downstream
/// consumers see the usual range.
pub fn ntt_ifma_ref<P: PrimeSetIfma>(table: &NttIfmaTable<P>, data: &mut [u64]) {
    let n = table.n;
    if n <= 1 {
        return;
    }

    let q2 = &table.q2;
    let q4 = &table.q4;
    let mut seg_base = 0usize; // base offset (in u64) for current segment

    // ── Level 0: a[i] *= ω^i (Harvey multiply) ──────────────────────
    {
        let omega_base = seg_base;
        let quot_base = seg_base + 4 * n;
        for i in 0..n {
            for k in 0..3 {
                let a = data[4 * i + k];
                let omega = table.powomega[omega_base + 4 * i + k];
                let omega_quot = table.powomega[quot_base + 4 * i + k];
                data[4 * i + k] = harvey_modmul(a, omega, omega_quot, P::Q[k]);
            }
        }
        seg_base += 8 * n;
    }

    // ── Butterfly levels: nn = n, n/2, …, 2 (Cooley-Tukey DIT) ──────
    let mut nn = n;
    while nn >= 2 {
        let halfnn = nn / 2;

        if halfnn > 1 {
            let count = halfnn - 1;
            let omega_base = seg_base;
            let quot_base = seg_base + 4 * count;

            let mut block_start = 0usize;
            while block_start < n {
                // i = 0: no twiddle multiply (both sides need cond_sub_4q)
                {
                    let p1 = 4 * block_start;
                    let p2 = 4 * (block_start + halfnn);
                    for k in 0..3 {
                        let a = data[p1 + k];
                        let b = data[p2 + k];
                        let sum = a + b;
                        let diff = a + q4[k] - b;
                        data[p1 + k] = cond_sub_2q(sum, q4[k]);
                        data[p2 + k] = cond_sub_2q(diff, q4[k]);
                    }
                }

                // i = 1..halfnn-1: Harvey multiply absorbs the diff-path reduction
                for i in 1..halfnn {
                    let p1 = 4 * (block_start + i);
                    let p2 = 4 * (block_start + halfnn + i);
                    let tw_idx = i - 1;
                    for k in 0..3 {
                        let a = data[p1 + k];
                        let b = data[p2 + k];
                        let sum = a + b;
                        let diff = a + q4[k] - b;
                        data[p1 + k] = cond_sub_2q(sum, q4[k]);
                        let omega = table.powomega[omega_base + 4 * tw_idx + k];
                        let omega_quot = table.powomega[quot_base + 4 * tw_idx + k];
                        data[p2 + k] = harvey_modmul(diff, omega, omega_quot, P::Q[k]);
                    }
                }

                block_start += nn;
            }

            seg_base += 8 * count;
        } else {
            // nn == 2: add/sub only, no twiddle
            let mut block_start = 0usize;
            while block_start < n {
                let p1 = 4 * block_start;
                let p2 = 4 * (block_start + 1);
                for k in 0..3 {
                    let a = data[p1 + k];
                    let b = data[p2 + k];
                    data[p1 + k] = cond_sub_2q(a + b, q4[k]);
                    data[p2 + k] = cond_sub_2q(a + q4[k] - b, q4[k]);
                }
                block_start += 2;
            }
        }

        nn /= 2;
    }

    // ── Final normalisation: [0, 4q) → [0, 2q) ──────────────────────
    for i in 0..n {
        for k in 0..3 {
            data[4 * i + k] = cond_sub_2q(data[4 * i + k], q2[k]);
        }
    }
}

/// Inverse NTT — scalar reference with IFMA-native lazy arithmetic.
///
/// Butterfly values live in `[0, 4q)`.  The final pointwise Harvey pass
/// reduces to `[0, 2q)` automatically, so no explicit renormalisation is
/// needed.
pub fn intt_ifma_ref<P: PrimeSetIfma>(table: &NttIfmaTableInv<P>, data: &mut [u64]) {
    let n = table.n;
    if n <= 1 {
        return;
    }

    let q4 = &table.q4;
    let mut seg_base = 0usize;

    // ── Butterfly levels: nn = 2, 4, …, n (Gentleman-Sande DIF) ─────
    let mut nn = 2usize;
    while nn <= n {
        let halfnn = nn / 2;

        if halfnn > 1 {
            let count = halfnn - 1;
            let omega_base = seg_base;
            let quot_base = seg_base + 4 * count;

            let mut block_start = 0usize;
            while block_start < n {
                // i = 0: no twiddle
                {
                    let p1 = 4 * block_start;
                    let p2 = 4 * (block_start + halfnn);
                    for k in 0..3 {
                        let a = data[p1 + k];
                        let b = data[p2 + k];
                        let sum = a + b;
                        let diff = a + q4[k] - b;
                        data[p1 + k] = cond_sub_2q(sum, q4[k]);
                        data[p2 + k] = cond_sub_2q(diff, q4[k]);
                    }
                }

                // i = 1..halfnn-1: twiddle on b BEFORE butterfly (b_raw ∈ [0, 4q)
                // fed directly into Harvey → bo ∈ [0, 2q); sum/diff use cond_sub_4q).
                for i in 1..halfnn {
                    let p1 = 4 * (block_start + i);
                    let p2 = 4 * (block_start + halfnn + i);
                    let tw_idx = i - 1;
                    for k in 0..3 {
                        let a = data[p1 + k];
                        let b_raw = data[p2 + k];
                        let omega = table.powomega[omega_base + 4 * tw_idx + k];
                        let omega_quot = table.powomega[quot_base + 4 * tw_idx + k];
                        let bo = harvey_modmul(b_raw, omega, omega_quot, P::Q[k]);
                        let sum = a + bo;
                        let diff = a + q4[k] - bo;
                        data[p1 + k] = cond_sub_2q(sum, q4[k]);
                        data[p2 + k] = cond_sub_2q(diff, q4[k]);
                    }
                }

                block_start += nn;
            }

            seg_base += 8 * count;
        } else {
            // nn == 2: add/sub only
            let mut block_start = 0usize;
            while block_start < n {
                let p1 = 4 * block_start;
                let p2 = 4 * (block_start + 1);
                for k in 0..3 {
                    let a = data[p1 + k];
                    let b = data[p2 + k];
                    data[p1 + k] = cond_sub_2q(a + b, q4[k]);
                    data[p2 + k] = cond_sub_2q(a + q4[k] - b, q4[k]);
                }
                block_start += 2;
            }
        }

        nn *= 2;
    }

    // ── Last pass: a[i] *= ω^{-i} / n (n entries, input ∈ [0, 4q), output ∈ [0, 2q)) ──
    {
        let omega_base = seg_base;
        let quot_base = seg_base + 4 * n;
        for i in 0..n {
            for k in 0..3 {
                let a = data[4 * i + k];
                let omega = table.powomega[omega_base + 4 * i + k];
                let omega_quot = table.powomega[quot_base + 4 * i + k];
                data[4 * i + k] = harvey_modmul(a, omega, omega_quot, P::Q[k]);
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::ntt_ifma::{
        arithmetic::{b_ifma_from_znx64_ref, b_ifma_to_znx128_ref},
        primes::Primes42,
    };

    #[test]
    fn ntt_intt_identity() {
        for log_n in 1..=10usize {
            let n = 1 << log_n;
            let fwd = NttIfmaTable::<Primes42>::new(n);
            let inv = NttIfmaTableInv::<Primes42>::new(n);

            let coeffs: Vec<i64> = (0..n as i64).map(|i| (i * 7 + 3) % 201 - 100).collect();

            let mut data = vec![0u64; 4 * n];
            b_ifma_from_znx64_ref(n, &mut data, &coeffs);
            let data_orig = data.clone();

            ntt_ifma_ref::<Primes42>(&fwd, &mut data);
            intt_ifma_ref::<Primes42>(&inv, &mut data);

            for i in 0..n {
                for k in 0..3 {
                    let orig = data_orig[4 * i + k] % Primes42::Q[k];
                    let got = data[4 * i + k] % Primes42::Q[k];
                    assert_eq!(orig, got, "n={n} i={i} k={k}: mismatch after NTT+iNTT round-trip");
                }
            }
        }
    }

    #[test]
    fn ntt_convolution() {
        let n = 8usize;
        let fwd = NttIfmaTable::<Primes42>::new(n);
        let inv = NttIfmaTableInv::<Primes42>::new(n);

        let a: Vec<i64> = vec![1, 2, 0, 0, 0, 0, 0, 0];
        let b: Vec<i64> = vec![3, 4, 0, 0, 0, 0, 0, 0];

        let mut da = vec![0u64; 4 * n];
        let mut db = vec![0u64; 4 * n];
        b_ifma_from_znx64_ref(n, &mut da, &a);
        b_ifma_from_znx64_ref(n, &mut db, &b);

        ntt_ifma_ref::<Primes42>(&fwd, &mut da);
        ntt_ifma_ref::<Primes42>(&fwd, &mut db);

        // Pointwise multiply (mod each Q[k])
        let mut dc = vec![0u64; 4 * n];
        for i in 0..n {
            for k in 0..3 {
                let q = Primes42::Q[k];
                dc[4 * i + k] = ((da[4 * i + k] % q) as u128 * (db[4 * i + k] % q) as u128 % q as u128) as u64;
            }
        }

        intt_ifma_ref::<Primes42>(&inv, &mut dc);

        let mut result = vec![0i128; n];
        b_ifma_to_znx128_ref(n, &mut result, &dc);

        let expected: Vec<i128> = vec![3, 10, 8, 0, 0, 0, 0, 0];
        assert_eq!(result, expected, "NTT convolution mismatch");
    }

    #[test]
    fn harvey_modmul_correctness() {
        for &q in &Primes42::Q {
            // Test with inputs in [0, 2q) — the IFMA-native range
            for a in [0u64, 1, q - 1, q, 2 * q - 1, q / 2, 42] {
                if a >= 2 * q {
                    continue;
                }
                for omega in [0u64, 1, q - 1, q / 2, 7] {
                    let omega_quot = harvey_quotient(omega, q);
                    let got = harvey_modmul(a, omega, omega_quot, q);
                    let expected = ((a as u128 * omega as u128) % q as u128) as u64;
                    assert!(
                        got % q == expected,
                        "harvey_modmul({a}, {omega}, q={q}): got {got} (mod q = {}), expected {expected}",
                        got % q
                    );
                    assert!(got < 2 * q, "harvey_modmul output {got} >= 2q={}", 2 * q);
                }
            }
        }
    }
}
