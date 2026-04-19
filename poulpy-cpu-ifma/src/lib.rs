//! AVX-512 backends for Poulpy.
//!
//! This crate exposes:
//! - [`NTTIfma`]: NTT-domain backend over three ~40-bit CRT primes using AVX-512-IFMA.
//! - [`FFT64Ifma`]: FFT64-domain backend using AVX-512F for REIM FFT kernels.
//!
//! Layout:
//! - `znx_ifma`: shared AVX-512 ring primitives.
//! - `ntt_ifma`: NTT backend and IFMA helpers.
//! - `fft64`: FFT backend and AVX-512 FFT helpers.
//!
//! `NTTIfma` uses `Q120bScalar`/`i128`; `FFT64Ifma` uses `f64`/`i64`.
//!
//! Requirements:
//! - `FFT64Ifma`: x86-64 + AVX-512F.
//! - `NTTIfma`: x86-64 + AVX-512F + AVX-512IFMA + AVX-512VL.
//!
//! Build with:
//! ```text
//! RUSTFLAGS="-C target-feature=+avx512f,+avx512ifma,+avx512vl" cargo build --release
//! ```
//! or on a supported host:
//! ```text
//! RUSTFLAGS="-C target-cpu=native" cargo build --release --features enable-ifma
//! ```
//!
//! Hot paths use AVX-512 intrinsics; colder coefficient-domain operations reuse
//! the reference backend. Runtime checks happen in `Module::new()`.
//!
//! This crate is usually consumed through higher-level crates such as
//! `poulpy_core`, `poulpy_schemes`, or `poulpy_ckks`.

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

// Without `enable-ifma`, skip the AVX-512 backend entirely.
#[cfg(feature = "enable-ifma")]
mod fft64;
#[cfg(feature = "enable-ifma")]
mod hal_impl;
#[cfg(feature = "enable-ifma")]
mod ntt_ifma;
#[cfg(feature = "enable-ifma")]
mod znx_ifma;

#[cfg(feature = "enable-ifma")]
pub use fft64::{FFT64Ifma, ReimFFTIfma, ReimIFFTIfma};
#[cfg(feature = "enable-ifma")]
pub use ntt_ifma::NTTIfma;

#[cfg(feature = "enable-ifma")]
use poulpy_core::oep::CoreImpl;
#[cfg(feature = "enable-ifma")]
unsafe impl CoreImpl<FFT64Ifma> for FFT64Ifma {
    poulpy_core::impl_core_default_methods!(FFT64Ifma);
}

#[cfg(feature = "enable-ifma")]
unsafe impl CoreImpl<NTTIfma> for NTTIfma {
    poulpy_core::impl_core_default_methods!(NTTIfma);
}
