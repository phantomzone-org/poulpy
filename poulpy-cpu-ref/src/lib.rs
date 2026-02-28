//! Reference (portable) CPU backend for the Poulpy lattice cryptography library.
//!
//! This crate provides two backend implementations for [`poulpy_hal`]:
//!
//! - [`FFT64Ref`]: scalar `f64` FFT arithmetic — see the [`fft64`] module.
//! - [`NTT120Ref`]: scalar Q120 NTT arithmetic (CRT over four ~30-bit primes) — see the [`ntt120`] module.
//!
//! Both are canonical reference implementations: portable across all CPU architectures,
//! prioritising correctness and debuggability over throughput.
//!
//! # Platform support
//!
//! Compiles and runs on any target supported by the Rust standard library.
//! No platform-specific intrinsics or assembly are used.

pub mod fft64;
pub mod ntt120;

#[cfg(test)]
mod tests;

pub use fft64::FFT64Ref;
pub use ntt120::{NTT120Ref, NTT120RefHandle};
