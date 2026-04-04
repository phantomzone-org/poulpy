//! AVX512-IFMA accelerated CPU backend for the Poulpy lattice cryptography library.
//!
//! This crate provides [`NTTIfma`], a high-performance NTT backend for [`poulpy_hal`]
//! that uses x86-64 AVX-512 IFMA instructions to accelerate Q120 arithmetic over
//! three approximately 40-bit CRT primes.
//!
//! Compared to the AVX2 [`NTT120Avx`](poulpy_cpu_avx) backend, which uses four
//! approximately 30-bit primes, this backend reduces the CRT width from four primes
//! to three and replaces split-precomputed multiplication with Harvey-style modular
//! multiplication based on `VPMADD52*` instructions.
//!
//! # Architecture
//!
//! `poulpy_hal` defines a hardware abstraction layer (HAL) via the
//! [`Backend`](poulpy_hal::layouts::Backend) trait and a family of open extension
//! point (OEP) traits in [`poulpy_hal::oep`]. This crate implements the NTT-side
//! OEP traits for [`NTTIfma`] using AVX-512 IFMA intrinsics on hot paths and the
//! portable reference logic where coefficient-domain code is backend-independent.
//!
//! The internal modules are organized by operation domain:
//!
//! | Module | Domain |
//! |---|---|
//! | `module` | Backend handle lifecycle and NTT table management |
//! | `scratch` | Temporary memory allocation and arena-style sub-allocation |
//! | `znx` | Single ring element arithmetic |
//! | `vec_znx` | Vectors of ring elements in coefficient form |
//! | `vec_znx_big` | Large-coefficient (`i128`) vector operations |
//! | `vec_znx_dft` | NTT-domain vector operations |
//! | `ntt_ifma_avx512` | Raw forward and inverse NTT kernels |
//! | `mat_vec_ifma` | BBC inner products and final reduction |
//! | `convolution` | Polynomial convolution in the NTT domain |
//! | `svp` | Scalar-vector product in the NTT domain |
//! | `vmp` | Vector-matrix product in the NTT domain |
//!
//! # Scalar types
//!
//! For the `NTTIfma` backend:
//!
//! - `ScalarPrep = Q120bScalar`: NTT-domain coefficients stored as four `u64`
//!   lanes per coefficient, with three active CRT residues and one padding lane.
//! - `ScalarBig = i128`: large-coefficient accumulators used after inverse NTT.
//!
//! # CPU requirements
//!
//! This backend requires x86-64 CPUs with:
//!
//! - **AVX512F**: Foundation 512-bit SIMD
//! - **AVX512IFMA**: 52-bit integer fused multiply-add (`VPMADD52LUQ`, `VPMADD52HUQ`)
//! - **AVX512VL**: Variable-length extensions for 256-bit AVX-512 operations
//!
//! Runtime CPU feature detection is performed in
//! [`Module::new()`](poulpy_hal::layouts::Module::new). If the required features are
//! not present, construction panics with a descriptive error message.
//!
//! # Compile-time requirements
//!
//! To compile this crate, enable the required AVX-512 target features:
//!
//! ```text
//! RUSTFLAGS="-C target-feature=+avx512f,+avx512ifma,+avx512vl" cargo build --release
//! ```
//!
//! Or, on supported hosts:
//!
//! ```text
//! RUSTFLAGS="-C target-cpu=native" cargo build --release --features enable-ifma
//! ```
//!
//! # Correctness guarantees
//!
//! - **Determinism**: integer-domain operations are bit-exact and match the
//!   reference IFMA arithmetic model.
//! - **Lazy modular arithmetic**: NTT-domain values are kept in bounded lazy ranges
//!   such as `[0, 2q)`, with explicit conditional-subtract steps maintaining the
//!   required invariants.
//! - **Shared cold paths**: coefficient-domain operations that are not performance
//!   critical reuse the same backend-independent logic as the reference backend.
//!
//! # Threading and concurrency
//!
//! - **`NTTIfma` is `Send + Sync`**: it is a zero-sized marker type.
//! - **`Module<NTTIfma>` is `Send + Sync`**: NTT tables are immutable after construction.
//! - **No internal locking**: synchronization is the caller's responsibility.
//!
//! # Feature flags
//!
//! - `enable-ifma` (optional): enables the AVX512-IFMA backend and enforces the
//!   corresponding compile-time target-feature checks.
//!
//! # Platform support
//!
//! - **Required**: x86-64 architecture with AVX512F, AVX512IFMA, and AVX512VL
//! - **Not supported**: non-x86-64 targets and x86-64 targets without IFMA support
//!
//! # Usage
//!
//! This crate exports a single public type, [`NTTIfma`], which is used as the
//! backend type parameter for `poulpy_hal`, `poulpy_core`, `poulpy_schemes`, and
//! `poulpy_ckks`.

// Compile-time architecture checks
#[cfg(all(feature = "enable-ifma", not(target_arch = "x86_64")))]
compile_error!("feature `enable-ifma` requires target_arch = \"x86_64\".");

#[cfg(all(feature = "enable-ifma", target_arch = "x86_64", not(target_feature = "avx512f")))]
compile_error!(
    "feature `enable-ifma` requires AVX512F. Build with RUSTFLAGS=\"-C target-feature=+avx512f,+avx512ifma,+avx512vl\"."
);

#[cfg(all(feature = "enable-ifma", target_arch = "x86_64", not(target_feature = "avx512ifma")))]
compile_error!(
    "feature `enable-ifma` requires AVX512-IFMA. Build with RUSTFLAGS=\"-C target-feature=+avx512f,+avx512ifma,+avx512vl\"."
);

#[cfg(all(feature = "enable-ifma", target_arch = "x86_64", not(target_feature = "avx512vl")))]
compile_error!(
    "feature `enable-ifma` requires AVX512VL. Build with RUSTFLAGS=\"-C target-feature=+avx512f,+avx512ifma,+avx512vl\"."
);

mod ntt_ifma;

pub use ntt_ifma::NTTIfma;
