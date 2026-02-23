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

//! Q120 NTT reference implementation.
//!
//! This module is a Rust port of the `q120` component of the
//! [spqlios-arithmetic](https://github.com/tfhe/spqlios-arithmetic) library.
//! It provides a pure-scalar (no SIMD) reference implementation of the
//! number-theoretic transform (NTT) and associated arithmetic over a
//! degree-120 composite modulus `Q = Q₀·Q₁·Q₂·Q₃`.
//!
//! # Representation
//!
//! Ring elements are stored in **CRT form**: each integer is represented
//! as four residues, one per prime factor of `Q`.  Three concrete prime
//! sets (29-, 30-, and 31-bit) are provided via the [`primes::PrimeSet`]
//! trait; the 30-bit variant ([`primes::Primes30`]) is the default and
//! matches the spqlios library default.
//!
//! The concrete storage types are:
//!
//! | Type | Element | Content |
//! |------|---------|---------|
//! | [`types::Q120a`] | `[u32; 4]` | Residues in `[0, 2^32)` |
//! | [`types::Q120b`] | `[u64; 4]` | Residues in `[0, 2^64)` — NTT domain |
//! | [`types::Q120c`] | `[u32; 8]` | `(rᵢ, rᵢ·2^32 mod Qᵢ)` pairs — prepared for lazy multiply |
//!
//! An NTT vector of length `n` is stored as a flat `[u64]` slice of
//! length `4 * n` (i.e., `n` consecutive [`types::Q120b`] values).
//!
//! # Submodules
//!
//! - [`primes`]: [`primes::PrimeSet`] trait and [`primes::Primes29`] /
//!   [`primes::Primes30`] / [`primes::Primes31`] implementations.
//! - [`types`]: CRT type aliases ([`types::Q120a`], [`types::Q120b`], etc.).
//! - [`arithmetic`]: Simple element-wise operations (conversion to/from
//!   `i64` / `i128`, component-wise addition).
//! - [`mat_vec`]: Lazy-accumulation matrix–vector products
//!   ([`mat_vec::BaaMeta`], [`mat_vec::BbbMeta`], [`mat_vec::BbcMeta`]
//!   and the corresponding product functions).
//! - [`ntt`]: NTT precomputation tables ([`ntt::NttTable`],
//!   [`ntt::NttTableInv`]) and reference execution
//!   ([`ntt::ntt_ref`], [`ntt::intt_ref`]).
//!
//! # Trait overview
//!
//! The traits defined at this level mirror the `Reim*` traits in
//! [`crate::reference::fft64::reim`] and provide the NTT-domain
//! operations that a backend implementation must satisfy:
//!
//! | Trait | Description |
//! |-------|-------------|
//! | [`NttDFTExecute`] | Forward or inverse NTT execution |
//! | [`NttFromZnx64`] | Load `i64` coefficients into q120b format |
//! | [`NttToZnx128`] | CRT-reconstruct from q120b to `i128` coefficients |
//! | [`NttAdd`] | Component-wise addition of two q120b vectors |
//! | [`NttAddInplace`] | In-place component-wise addition |
//! | [`NttZero`] | Zero a q120b vector |
//! | [`NttCopy`] | Copy a q120b vector |
//! | [`NttMulBbb`] | Lazy product: q120b × q120b → q120b |
//! | [`NttMulBbc`] | Lazy product: q120b × q120c → q120b |

pub mod arithmetic;
pub mod convolution;
pub mod mat_vec;
pub mod ntt;
pub mod primes;
pub mod svp;
pub mod types;
pub mod vec_znx_big;
pub mod vec_znx_dft;
pub mod vmp;

pub use arithmetic::*;
pub use convolution::*;
pub use mat_vec::*;
pub use ntt::*;
pub use primes::*;
pub use svp::*;
pub use types::*;
pub use vec_znx_big::*;
pub use vec_znx_dft::*;
pub use vmp::*;

// ──────────────────────────────────────────────────────────────────────────────
// NTT-domain operation traits
// ──────────────────────────────────────────────────────────────────────────────

/// Execute a forward or inverse NTT using a precomputed table.
///
/// `Table` is either [`NttTable`] or [`NttTableInv`] (both generic over a
/// [`PrimeSet`]).  `data` is a flat `u64` slice of length `4 * n` in
/// q120b layout.
pub trait NttDFTExecute<Table> {
    /// Apply the NTT (or iNTT) described by `table` to `data` in place.
    fn ntt_dft_execute(table: &Table, data: &mut [u64]);
}

/// Load a polynomial from the standard `i64` coefficient representation
/// into the q120b NTT-domain format.
pub trait NttFromZnx64 {
    /// Encode the `a.len()` coefficients of `a` into `res` (q120b layout).
    ///
    /// `res` must have length `4 * a.len()`.
    fn ntt_from_znx64(res: &mut [u64], a: &[i64]);
}

/// Recover `i128` ring-element coefficients from a q120b NTT vector.
///
/// The `divisor_is_n` parameter specifies the polynomial degree `n`; it
/// is used to apply the `1/n` scaling that the inverse NTT does not
/// include automatically (see [`NttTableInv`]).
pub trait NttToZnx128 {
    /// Decode `a` (q120b layout, length `4 * n`) into `res` (`n` × `i128`).
    fn ntt_to_znx128(res: &mut [i128], divisor_is_n: usize, a: &[u64]);
}

/// Component-wise addition of two q120b vectors.
pub trait NttAdd {
    /// `res[i] = a[i] + b[i]` for each CRT component.
    ///
    /// All three slices must have the same length (a multiple of 4).
    fn ntt_add(res: &mut [u64], a: &[u64], b: &[u64]);
}

/// In-place component-wise addition of a q120b vector.
pub trait NttAddInplace {
    /// `res[i] += a[i]` for each CRT component.
    fn ntt_add_inplace(res: &mut [u64], a: &[u64]);
}

/// Zero a q120b vector.
pub trait NttZero {
    /// Set all elements of `res` to zero.
    fn ntt_zero(res: &mut [u64]);
}

/// Copy a q120b vector.
pub trait NttCopy {
    /// Copy all elements from `a` into `res`.
    fn ntt_copy(res: &mut [u64], a: &[u64]);
}

/// Lazy matrix–vector product: q120b × q120b → q120b.
///
/// Multiplies each of the `ell` rows of a column vector `b` (in q120b
/// format) by the corresponding entry in `a` (also q120b), accumulating
/// the results into `res`.
pub trait NttMulBbb {
    /// `res += a[0..ell] ⊙ b[0..ell]` using lazy modular arithmetic.
    fn ntt_mul_bbb(ell: usize, res: &mut [u64], a: &[u64], b: &[u64]);
}

/// Lazy matrix–vector product: q120b × q120c → q120b.
///
/// Like [`NttMulBbb`] but the right-hand operand `b` is in the
/// **prepared** q120c format ([`types::Q120c`]: 8 × `u32` per element),
/// which pre-stores `(r, r·2^32 mod Q)` pairs for faster multiply-accumulate.
pub trait NttMulBbc {
    /// `res += a[0..ell] ⊙ b[0..ell]` with `b` in q120c layout.
    fn ntt_mul_bbc(ell: usize, res: &mut [u64], a: &[u32], b: &[u32]);
}
