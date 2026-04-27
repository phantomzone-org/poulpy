//! CPU-side twiddle-factor generation for the NTT120 CUDA backend.
//!
//! Mirrors Cheddar's `NTTHandler::PopulateTwiddleFactors()`. Output is uploaded
//! to device by the caller; this module is pure host arithmetic.

/// Cheddar's `LsbSize()` for all supported `log_degree`. Always 32.
pub const LSB_SIZE: usize = 32;

/// Primes30: four ~30-bit NTT-friendly primes (Q ≈ 2^120).
const Q: [u32; 4] = [
    (1u32 << 30) - 2 * (1u32 << 17) + 1,
    (1u32 << 30) - 17 * (1u32 << 17) + 1,
    (1u32 << 30) - 23 * (1u32 << 17) + 1,
    (1u32 << 30) - 42 * (1u32 << 17) + 1,
];

/// Primitive 2^17-th roots of unity for each Primes30 prime.
const OMEGA: [u32; 4] = [1_070_907_127, 315_046_632, 309_185_662, 846_468_380];

// ── Arithmetic helpers ────────────────────────────────────────────────────────

fn mul_mod(a: u32, b: u32, q: u32) -> u32 {
    ((a as u64 * b as u64) % q as u64) as u32
}

fn pow_mod(mut a: u32, mut e: u64, q: u32) -> u32 {
    let mut res = 1u32;
    a %= q;
    while e > 0 {
        if e & 1 != 0 {
            res = mul_mod(res, a, q);
        }
        a = mul_mod(a, a, q);
        e >>= 1;
    }
    res
}

fn gcd_ext(a: i64, b: i64, x: &mut i64, y: &mut i64) -> i64 {
    if a == 0 {
        *x = 0;
        *y = 1;
        return b;
    }
    let mut x1 = 0i64;
    let mut y1 = 0i64;
    let g = gcd_ext(b % a, a, &mut x1, &mut y1);
    *x = y1 - (b / a) * x1;
    *y = x1;
    g
}

fn inv_mod(a: u32, q: u32) -> u32 {
    let mut x = 0i64;
    let mut y = 0i64;
    gcd_ext(a as i64, q as i64, &mut x, &mut y);
    let x = x % q as i64;
    if x < 0 { (x + q as i64) as u32 } else { x as u32 }
}

/// `a * 2^32 mod q` — converts a bare residue to Montgomery form.
fn to_montgomery(a: u32, q: u32) -> u32 {
    (((a as u64) << 32) % (q as u64)) as u32
}

/// `q^{-1} mod 2^32` as a signed `i32` — Cheddar's `InvModBase(q)`.
///
/// This is the Montgomery constant stored in `inv_primes`; used in
/// `__montgomery_reduction_lazy` as the `q_inv` parameter.
pub fn inv_mod_base(q: u32) -> i32 {
    // base = 2^32 - 1; remainder = 2^32 mod q
    let quotient = u32::MAX / q;
    let remainder = (u32::MAX % q) as i64 + 1;
    let mut x1 = 0i64;
    let mut y1 = 0i64;
    gcd_ext(remainder, q as i64, &mut x1, &mut y1);
    // q * (y1 - quotient * x1) ≡ 1 (mod 2^32)
    (y1 - quotient as i64 * x1) as i32
}

fn bit_reverse(v: &mut [u32]) {
    let n = v.len();
    let log_n = n.trailing_zeros() as usize;
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS as usize - log_n);
        if i < j {
            v.swap(i, j);
        }
    }
}

// ── Public output ─────────────────────────────────────────────────────────────

/// Host-side twiddle tables for one NTT120 ring dimension, ready to upload.
pub struct TwiddleTables {
    /// Forward twiddles, prime-major `[4 × n]`.
    pub twiddle_fwd: Vec<u32>,
    /// Forward MSB twiddles, prime-major `[4 × (n / LSB_SIZE)]`.
    pub twiddle_fwd_msb: Vec<u32>,
    /// Inverse twiddles, prime-major `[4 × n]`.
    pub twiddle_inv: Vec<u32>,
    /// Inverse MSB twiddles, prime-major `[4 × (n / LSB_SIZE)]`.
    pub twiddle_inv_msb: Vec<u32>,
    /// `n^{-1} mod Q[k]` in Montgomery form, one per prime.
    pub inv_n_mont: [u32; 4],
    /// `Q[k]`, one per prime.
    pub primes: [u32; 4],
    /// `Q[k]^{-1} mod 2^32` as `i32`, one per prime (Cheddar's `q_inv`).
    pub inv_primes: [i32; 4],
}

/// Generate twiddle tables for NTT size `n` (power of two, `1 ≤ n ≤ 2^16`).
///
/// Replicates Cheddar's `NTTHandler::PopulateTwiddleFactors()` for Primes30:
/// - primitive 2n-th root `psi = OMEGA[k]^(2^16/n) mod Q[k]`
/// - `psi_rev[j] = psi^j`, bit-reversed, in Montgomery form → `twiddle_fwd`
/// - `psi_inv_rev[j] = psi^{-j}`, bit-reversed, in Montgomery form → `twiddle_inv`
/// - MSB arrays: every `LSB_SIZE`-th element (Cheddar's OF-Twiddle optimisation)
pub fn generate(n: usize) -> TwiddleTables {
    assert!(
        n.is_power_of_two() && (1..=(1 << 16)).contains(&n),
        "NTT size must be a power of two in [1, 2^16], got {n}"
    );

    let msb_count = n / LSB_SIZE; // 0 when n < LSB_SIZE
    let mut twiddle_fwd = vec![0u32; 4 * n];
    let mut twiddle_fwd_msb = vec![0u32; 4 * msb_count];
    let mut twiddle_inv = vec![0u32; 4 * n];
    let mut twiddle_inv_msb = vec![0u32; 4 * msb_count];
    let mut inv_n_mont = [0u32; 4];
    let mut primes = [0u32; 4];
    let mut inv_primes_out = [0i32; 4];

    for k in 0..4 {
        let q = Q[k];

        // Primitive 2n-th root of unity: OMEGA[k]^(2^16/n)
        let exp = (1u64 << 16) / n as u64;
        let psi = pow_mod(OMEGA[k], exp, q);
        let psi_inv = inv_mod(psi, q);

        // Sequential powers, then bit-reversal (bit-reversed DIT ordering)
        let mut fwd_plain = vec![0u32; n];
        let mut inv_plain = vec![0u32; n];
        fwd_plain[0] = 1;
        inv_plain[0] = 1;
        for j in 1..n {
            fwd_plain[j] = mul_mod(fwd_plain[j - 1], psi, q);
            inv_plain[j] = mul_mod(inv_plain[j - 1], psi_inv, q);
        }
        bit_reverse(&mut fwd_plain);
        bit_reverse(&mut inv_plain);

        // Convert to Montgomery form; store prime-major
        let base = k * n;
        for j in 0..n {
            twiddle_fwd[base + j] = to_montgomery(fwd_plain[j], q);
            twiddle_inv[base + j] = to_montgomery(inv_plain[j], q);
        }

        // MSB: every LSB_SIZE-th element (one per shared-memory block)
        let base_msb = k * msb_count;
        for j in 0..msb_count {
            twiddle_fwd_msb[base_msb + j] = twiddle_fwd[base + j * LSB_SIZE];
            twiddle_inv_msb[base_msb + j] = twiddle_inv[base + j * LSB_SIZE];
        }

        inv_n_mont[k] = to_montgomery(inv_mod(n as u32, q), q);
        primes[k] = q;
        inv_primes_out[k] = inv_mod_base(q);
    }

    TwiddleTables {
        twiddle_fwd,
        twiddle_fwd_msb,
        twiddle_inv,
        twiddle_inv_msb,
        inv_n_mont,
        primes,
        inv_primes: inv_primes_out,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primes_are_ntt_friendly() {
        for &q in &Q {
            // q - 1 must be divisible by 2^17
            assert_eq!((q - 1) % (1 << 17), 0, "prime {q} not 2^17-friendly");
        }
    }

    #[test]
    fn omega_is_primitive_root() {
        for k in 0..4 {
            let q = Q[k];
            let w = OMEGA[k];
            // w^(2^17) ≡ 1 (mod q)
            assert_eq!(pow_mod(w, 1 << 17, q), 1, "prime {k}: OMEGA^2^17 != 1");
            // w^(2^16) ≡ -1 ≡ q-1 (mod q)  (hence w is a primitive 2^17-th root)
            assert_eq!(pow_mod(w, 1 << 16, q), q - 1, "prime {k}: OMEGA^2^16 != -1");
        }
    }

    #[test]
    fn inv_mod_base_is_correct() {
        for &q in &Q {
            let inv = inv_mod_base(q);
            // q * inv ≡ 1 (mod 2^32)
            let prod = (q as i64).wrapping_mul(inv as i64) as i32;
            assert_eq!(prod, 1, "inv_mod_base failed for q={q}");
        }
    }

    #[test]
    fn to_montgomery_roundtrip() {
        let q = Q[0];
        // to_montgomery(1) = R mod q = 2^32 mod q
        let r_mod_q = (1u64 << 32) % q as u64;
        assert_eq!(to_montgomery(1, q), r_mod_q as u32);
    }

    #[test]
    fn twiddles_n16_psi_order() {
        // For n=16, the primitive 2*16=32-nd root must satisfy psi^16 ≡ -1 (mod q)
        for k in 0..4 {
            let q = Q[k];
            let psi = pow_mod(OMEGA[k], (1 << 16) / 16, q);
            assert_eq!(pow_mod(psi, 16, q), q - 1, "prime {k}: psi^n != -1");
        }
    }

    #[test]
    fn generate_n16_lengths() {
        let n = 16usize;
        let t = generate(n);
        assert_eq!(t.twiddle_fwd.len(), 4 * n);
        assert_eq!(t.twiddle_inv.len(), 4 * n);
        // n=16 < LSB_SIZE=32, so MSB arrays are empty
        assert_eq!(t.twiddle_fwd_msb.len(), 0);
        assert_eq!(t.twiddle_inv_msb.len(), 0);
    }

    #[test]
    fn generate_n64_msb_lengths() {
        let n = 64usize;
        let t = generate(n);
        assert_eq!(t.twiddle_fwd.len(), 4 * n);
        assert_eq!(t.twiddle_fwd_msb.len(), 4 * (n / LSB_SIZE));
    }

    #[test]
    fn msb_is_every_lsbsize_th_element() {
        let n = 64usize;
        let t = generate(n);
        let msb_count = n / LSB_SIZE;
        for k in 0..4 {
            for j in 0..msb_count {
                assert_eq!(
                    t.twiddle_fwd_msb[k * msb_count + j],
                    t.twiddle_fwd[k * n + j * LSB_SIZE],
                    "FWD MSB mismatch at k={k} j={j}"
                );
                assert_eq!(
                    t.twiddle_inv_msb[k * msb_count + j],
                    t.twiddle_inv[k * n + j * LSB_SIZE],
                    "INV MSB mismatch at k={k} j={j}"
                );
            }
        }
    }
}
