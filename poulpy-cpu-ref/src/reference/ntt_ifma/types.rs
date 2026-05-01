use super::primes::{PrimeSetIfma, Primes42};

/// Lazy-reduction bound for DFT-domain arithmetic.
///
/// In the IFMA-native model, all NTT-domain values are in `[0, 2q)`.
/// `Q_SHIFTED_IFMA[k] = 2 * Q[k]`.  Used for conditional subtract
/// after add/sub operations to keep values in range.
/// Lane 3 is zero (padding).
pub const Q_SHIFTED_IFMA: [u64; 4] = {
    let q = <Primes42 as PrimeSetIfma>::Q;
    [2 * q[0], 2 * q[1], 2 * q[2], 0]
};
