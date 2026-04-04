//! Internal implementation modules for the [`NTTIfma`](crate::NTTIfma) backend.
//!
//! This module tree contains the AVX512-IFMA accelerated implementation of the
//! NTT-family backend exported by this crate. The organization mirrors the split
//! used by the AVX backend: handle management, scratch allocation, coefficient-domain
//! helpers, transform-domain helpers, and raw SIMD kernels are kept in separate
//! modules so each operation family has a clear home.
//!
//! # Internal module layout
//!
//! | Module | Responsibility |
//! |---|---|
//! | `module` | Runtime handle construction and precomputed table ownership |
//! | `scratch` | Scratch allocator integration |
//! | `znx` | Single-ring coefficient-domain primitives |
//! | `vec_znx` | Coefficient-domain vector operations |
//! | `vec_znx_big` | Large-coefficient vector operations and normalization hooks |
//! | `vec_znx_big_avx512` | Raw AVX-512 `i128` arithmetic kernels |
//! | `vec_znx_dft` | NTT-domain vector operations |
//! | `prim` | Primitive trait implementations for NTT-domain arithmetic |
//! | `ntt_ifma_avx512` | Raw forward and inverse NTT kernels |
//! | `mat_vec_ifma` | BBC inner products and SIMD final reduction |
//! | `convolution` | Convolution OEP implementations |
//! | `svp` | Scalar-vector product OEP implementations |
//! | `vmp` | Vector-matrix product OEP implementations |
//!
//! The public API of this submodule is the zero-sized marker type [`NTTIfma`].

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

/// AVX512-IFMA accelerated NTT-family backend.
///
/// `NTTIfma` is a zero-sized marker type used as the backend parameter for
/// [`Module`](poulpy_hal::layouts::Module) and related HAL containers.
///
/// The backend uses three approximately 40-bit CRT primes and AVX-512 IFMA kernels
/// for forward and inverse NTT execution, BBC inner products, convolution, SVP,
/// VMP, and selected large-integer normalization paths. Coefficient-domain cold
/// paths reuse the backend-independent reference implementation where that is the
/// intended architecture.
#[derive(Debug, Clone, Copy)]
pub struct NTTIfma;
