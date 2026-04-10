//! AVX512-IFMA accelerated NTT CPU backend for the Poulpy lattice cryptography library.
//!
//! This module provides [`NTTIfma`], an AVX512-IFMA accelerated backend implementation for
//! [`poulpy_hal`] that uses IFMA NTT arithmetic (CRT over three ~40-bit primes). It
//! mirrors the structure of the scalar [`poulpy_cpu_ref::NTTIfmaRef`] backend, with
//! AVX512-IFMA accelerated kernels substituted where available.
//!
//! # Current acceleration status
//!
//! | Domain | Status |
//! |-|-|
//! | Coefficient-domain (`Znx*`) | AVX-512F (reuses `crate::znx_ifma`) |
//! | NTT forward/inverse | AVX512-IFMA (`ntt_ifma_avx512` module) |
//! | mat_vec BBC product (SVP/VMP hot path) | AVX512-IFMA (`mat_vec_ifma` module) |
//! | VecZnxBig add/sub/negate | AVX-512F (`vec_znx_big_avx512` module) |
//! | VecZnxBig normalization | AVX-512F (`vec_znx_big_avx512` module) |
//!
//! # Scalar types
//!
//! - `ScalarPrep = Q120bScalar` — NTT-domain coefficients (4 × u64, 32 bytes/coeff).
//! - `ScalarBig  = i128` — CRT-reconstructed large coefficients.

mod convolution;
pub(crate) mod mat_vec_ifma;
mod module;
mod prim;
mod scratch;
mod svp;
mod vec_znx;
mod vec_znx_big;
mod vec_znx_big_avx512;
mod vec_znx_dft;
mod vmp;
mod znx;

pub(crate) mod ntt_ifma_avx512;

#[cfg(test)]
mod tests;

/// AVX512-IFMA accelerated NTT CPU backend for Poulpy HAL.
///
/// `NTTIfma` is a zero-sized marker type that selects the AVX512-IFMA accelerated IFMA backend
/// when used as the type parameter `B` in [`poulpy_hal::layouts::Module<B>`](poulpy_hal::layouts::Module)
/// and related HAL types. It implements all open extension point (OEP) traits from
/// `poulpy_hal::oep`.
///
/// # Backend characteristics
///
/// - **ScalarPrep**: `Q120bScalar` — NTT-domain coefficients stored as 4 × u64 CRT residues.
/// - **ScalarBig**: `i128` — large-coefficient ring elements use 128-bit signed integers.
/// - **Prime set**: `Primes40` (three ~40-bit primes, Q ≈ 2^120).
///
/// # CPU feature requirements
///
/// **Runtime check**: `Module::new()` verifies that the CPU supports
/// AVX512-IFMA. If the feature is missing, the constructor panics.
///
/// # Thread safety
///
/// `NTTIfma` is `Send + Sync` (derived from being a zero-sized, field-less struct).
#[derive(Debug, Clone, Copy)]
pub struct NTTIfma;
