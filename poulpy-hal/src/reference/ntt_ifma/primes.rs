/// Selects a set of three NTT-friendly primes and their associated
/// constants for the IFMA CRT representation.
///
/// The product `Q = Q[0]·Q[1]·Q[2]` is approximately 2^120.
/// All three primes support a primitive `2^17`-th root of unity, so
/// NTT sizes up to `2^16` are supported.
///
/// Unlike [`super::super::ntt120::primes::PrimeSet`] which uses `[u32; 4]`
/// for ~30-bit primes, this trait uses `[u64; 3]` for ~40-bit primes
/// that fit within IFMA52's 52-bit input window.
pub trait PrimeSetIfma: Sized + Sync + Send + 'static {
    /// The three NTT-friendly primes `[Q0, Q1, Q2]`.
    const Q: [u64; 3];

    /// `OMEGA[k]` is a primitive `2^17`-th root of unity modulo `Q[k]`.
    ///
    /// For an NTT of size `n ≤ 2^16`, the actual primitive `2n`-th root
    /// used is `modq_pow64(OMEGA[k], 2^16 / n, Q[k])`.
    const OMEGA: [u64; 3];

    /// CRT reconstruction constants (Garner's algorithm).
    ///
    /// - `CRT_CST[0]` = `inv(Q[0], Q[1])` — inverse of Q[0] modulo Q[1]
    /// - `CRT_CST[1]` = `inv(Q[0]*Q[1], Q[2])` — inverse of Q[0]*Q[1] modulo Q[2]
    /// - `CRT_CST[2]` is unused (reserved / zero)
    const CRT_CST: [u64; 3];

    /// `ceil(log2(Q[0]))`.
    ///
    /// All three primes have the same bit-size, so this constant applies
    /// to all of them.  Used during NTT precomputation to track the
    /// growth of intermediate bit-widths through the butterfly levels.
    const LOG_Q: u64;
}

/// 40-bit NTT-friendly primes with `2·2^16`-th roots of unity.
///
/// - `Q ≈ 2^120`
/// - Each prime is of the form `c·2^17 + 1` with `c·2^17 + 1 < 2^40`.
/// - Designed for use with AVX512-IFMA52 instructions (primes < 2^50).
pub struct Primes40;

impl PrimeSetIfma for Primes40 {
    const Q: [u64; 3] = [
        1_099_510_054_913, // 8388596 * 2^17 + 1
        1_099_507_695_617, // 8388578 * 2^17 + 1
        1_099_506_515_969, // 8388569 * 2^17 + 1
    ];
    const OMEGA: [u64; 3] = [58_415_410_147, 699_776_532_486, 111_593_367_472];
    // Garner CRT constants:
    // CRT_CST[0] = inv(Q[0], Q[1])
    // CRT_CST[1] = inv(Q[0]*Q[1], Q[2])
    // CRT_CST[2] = unused
    const CRT_CST: [u64; 3] = [610_837_142_644, 647_033_352_384, 0];
    const LOG_Q: u64 = 40;
}

/// Computes `x^n mod q` using square-and-multiply with 128-bit intermediates.
///
/// Handles negative exponents via `x^(-(|n| mod (q-1)))`.
pub fn modq_pow64(x: u64, n: i64, q: u64) -> u64 {
    let qm1 = (q - 1) as i64;
    // reduce exponent mod (q-1) to positive representative
    let np = ((n % qm1) + qm1) % qm1;
    let mut np = np as u64;
    let mut val_pow = x % q;
    let q128 = q as u128;
    let mut result = 1u64;
    while np > 0 {
        if np & 1 != 0 {
            result = ((result as u128 * val_pow as u128) % q128) as u64;
        }
        val_pow = ((val_pow as u128 * val_pow as u128) % q128) as u64;
        np >>= 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primes40_are_prime() {
        for &q in &Primes40::Q {
            assert!(is_prime(q), "{q} is not prime");
        }
    }

    #[test]
    fn primes40_omega_are_primitive_roots() {
        for k in 0..3 {
            let q = Primes40::Q[k];
            let omega = Primes40::OMEGA[k];
            // omega^(2^17) == 1 mod q
            assert_eq!(modq_pow64(omega, 1 << 17, q), 1, "omega[{k}]^(2^17) != 1");
            // omega^(2^16) != 1 mod q (primitive, not a 2^16-th root)
            assert_ne!(modq_pow64(omega, 1 << 16, q), 1, "omega[{k}] is not primitive");
        }
    }

    #[test]
    fn primes40_crt_roundtrip() {
        let q = Primes40::Q;
        let q01 = q[0] as u128 * q[1] as u128;
        let big_q = q01 * q[2] as u128;

        for &x in &[0i64, 1, -1, 42, i64::MAX, i64::MIN + 1] {
            let r: [u64; 3] = [
                (x.rem_euclid(q[0] as i64)) as u64,
                (x.rem_euclid(q[1] as i64)) as u64,
                (x.rem_euclid(q[2] as i64)) as u64,
            ];

            // Garner reconstruction
            let inv01 = <Primes40 as PrimeSetIfma>::CRT_CST[0];
            let inv012 = <Primes40 as PrimeSetIfma>::CRT_CST[1];

            let v0 = r[0] as u128;
            // Modular subtraction: (r[1] - v0) mod q[1]
            let r1_mod = (r[1] % q[1]) as u128;
            let v0_mod = v0 % q[1] as u128;
            let diff1 = (r1_mod + q[1] as u128 - v0_mod) % q[1] as u128;
            let v1 = ((diff1 * inv01 as u128) % q[1] as u128) as u64;

            // partial = v0 + v1*q0 (NOT reduced mod q2 yet for final sum)
            let partial_full = v0 + v1 as u128 * q[0] as u128;
            let partial_mod_q2 = partial_full % q[2] as u128;
            let r2_mod = (r[2] % q[2]) as u128;
            let diff2 = (r2_mod + q[2] as u128 - partial_mod_q2) % q[2] as u128;
            let v2 = ((diff2 * inv012 as u128) % q[2] as u128) as u64;

            let mut result = partial_full + v2 as u128 * q01;
            if result > big_q / 2 {
                result = result.wrapping_sub(big_q);
            }
            let result = result as i128;

            let expected = x as i128;
            assert_eq!(result, expected, "CRT roundtrip failed for x={x}");
        }
    }

    /// Simple trial-division primality test (for tests only).
    fn is_prime(n: u64) -> bool {
        if n < 2 {
            return false;
        }
        if n < 4 {
            return true;
        }
        if n.is_multiple_of(2) || n.is_multiple_of(3) {
            return false;
        }
        let mut i = 5u64;
        while i * i <= n {
            if n.is_multiple_of(i) || n.is_multiple_of(i + 2) {
                return false;
            }
            i += 6;
        }
        true
    }
}
