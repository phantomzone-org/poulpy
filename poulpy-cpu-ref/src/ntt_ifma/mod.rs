//! Reference NTT-IFMA CPU backend for the Poulpy lattice cryptography library.
//!
//! This crate provides [`NTTIfmaRef`], a backend implementation for [`poulpy_hal`] that uses
//! scalar 3-prime CRT NTT arithmetic (Chinese Remainder Theorem over three ~40-bit primes).
//! It is the reference IFMA implementation: portable across all CPU architectures, prioritising
//! correctness and debuggability over throughput.
//!
//! # Architecture
//!
//! `poulpy-hal` defines a hardware abstraction layer (HAL) via the [`Backend`](poulpy_hal::layouts::Backend)
//! trait and a family of _open extension point_ (OEP) traits in [`poulpy_hal::oep`]. This crate
//! implements every OEP trait for the [`NTTIfmaRef`] backend by delegating to the reference
//! functions provided by `poulpy_hal::reference::ntt_ifma`.
//!
//! The internal modules are organised by operation domain:
//!
//! | Module          | Domain                                                         |
//! |-----------------|----------------------------------------------------------------|
//! | `module`        | Backend handle lifecycle, NTT table management                 |
//! | `scratch`       | Temporary memory allocation and arena-style sub-allocation     |
//! | `znx`           | Single ring element (`Z[X]/(X^n+1)`) arithmetic               |
//! | `vec_znx`       | Vectors of ring elements (limb decomposition)                  |
//! | `vec_znx_big`   | Large-coefficient (i128) ring element vectors                  |
//! | `vec_znx_dft`   | NTT-domain ring element vectors (forward/inverse NTT)         |
//! | `convolution`   | Polynomial convolution (stub)                                  |
//! | `svp`           | Scalar-vector product in NTT domain                            |
//! | `vmp`           | Vector-matrix product in NTT domain                            |
//!
//! # Scalar types
//!
//! For the `NTTIfmaRef` backend:
//!
//! - `ScalarPrep = Q120bScalar`: coefficients in the NTT / frequency domain (32 bytes = 4 x u64).
//! - `ScalarBig  = i128`: coefficients in the large-integer (CRT-reconstructed) domain.
//!
//! # Usage
//!
//! This crate exports a single public type, [`NTTIfmaRef`], which is used as a type parameter
//! to the HAL generic types. All functionality is accessed through the trait methods defined
//! in `poulpy_hal::api`.
//!
//! # Platform support
//!
//! Compiles and runs on any target supported by the Rust standard library.
//! No platform-specific intrinsics or assembly are used.

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

pub use module::NTTIfmaRefHandle;

/// Reference (portable) CPU backend using 3-prime IFMA NTT arithmetic.
///
/// `NTTIfmaRef` is a zero-sized marker type that selects the reference NTT-IFMA CPU backend
/// when used as the type parameter `B` in [`poulpy_hal::layouts::Module<B>`](poulpy_hal::layouts::Module)
/// and related HAL types. It implements all open extension point (OEP) traits from
/// `poulpy_hal::oep` by delegating to the portable reference functions in
/// `poulpy_hal::reference::ntt_ifma`.
///
/// # Backend characteristics
///
/// - **ScalarPrep**: `Q120bScalar` -- NTT-domain coefficients stored as 4 x u64 CRT residues
///   (3 active + 1 padding).
/// - **ScalarBig**: `i128` -- large-coefficient ring elements use 128-bit signed integers.
/// - **Prime set**: `Primes40` (three ~40-bit primes, Q ~ 2^120).
/// - **NTT tables**: precomputed twiddle factors stored in the module handle
///   (`NTTIfmaRefHandle`), shared across all operations on the same module.
///
/// # Thread safety
///
/// `NTTIfmaRef` is `Send + Sync` (derived from being a zero-sized, field-less struct).
/// The `Module<NTTIfmaRef>` that holds the NTT tables is also `Send + Sync`, so modules can
/// be shared across threads.
#[derive(Debug, Clone, Copy)]
pub struct NTTIfmaRef;
