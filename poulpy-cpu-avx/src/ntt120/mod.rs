//! AVX2-accelerated NTT120 CPU backend for the Poulpy lattice cryptography library.
//!
//! This module provides [`NTT120Avx`], an AVX2-accelerated backend implementation for
//! [`poulpy_hal`] that uses Q120 NTT arithmetic (CRT over four ~30-bit primes). It
//! mirrors the structure of the scalar [`poulpy_cpu_ref::NTT120Ref`] backend, with
//! AVX2-accelerated kernels substituted where available.
//!
//! # Current acceleration status
//!
//! | Domain | Status |
//! |-|-|
//! | Coefficient-domain (`Znx*`) | AVX2 (reuses `crate::znx_avx`) |
//! | NTT forward/inverse | Scalar (Phase 1: future work) |
//! | VecZnxBig add/sub/negate | Scalar (Phase 2: future work) |
//! | VecZnxBig normalization | Scalar (Phase 3: future work) |
//! | mat_vec BBC product | Scalar (Phase 4: future work) |
//!
//! # Scalar types
//!
//! - `ScalarPrep = Q120bScalar` — NTT-domain coefficients (4 × u64, 32 bytes/coeff).
//! - `ScalarBig  = i128` — CRT-reconstructed large coefficients.

mod convolution;
mod module;
mod prim;
mod scratch;
mod svp;
mod vec_znx;
mod vec_znx_big;
mod vec_znx_dft;
mod vmp;
mod znx;

/// AVX2-accelerated NTT120 CPU backend for Poulpy HAL.
///
/// `NTT120Avx` is a zero-sized marker type that selects the AVX2-accelerated NTT120 backend
/// when used as the type parameter `B` in [`poulpy_hal::layouts::Module<B>`](poulpy_hal::layouts::Module)
/// and related HAL types. It implements all open extension point (OEP) traits from
/// `poulpy_hal::oep`.
///
/// # Backend characteristics
///
/// - **ScalarPrep**: `Q120bScalar` — NTT-domain coefficients stored as 4 × u64 CRT residues.
/// - **ScalarBig**: `i128` — large-coefficient ring elements use 128-bit signed integers.
/// - **Prime set**: `Primes30` (four ~30-bit primes, Q ≈ 2^120).
///
/// # CPU feature requirements
///
/// **Runtime check**: [`Module::new()`](poulpy_hal::layouts::Module::new) verifies that
/// the CPU supports AVX2. If the feature is missing, the constructor panics.
///
/// # Thread safety
///
/// `NTT120Avx` is `Send + Sync` (derived from being a zero-sized, field-less struct).
#[derive(Debug, Clone, Copy)]
pub struct NTT120Avx;
