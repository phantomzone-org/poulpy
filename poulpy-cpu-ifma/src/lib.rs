//! AVX-512 / AVX-512-IFMA accelerated CPU backend for the Poulpy lattice
//! cryptography library.
//!
//! This crate provides two backends:
//!
//! - [`NTTIfma`] — an NTT-domain backend that uses AVX-512-IFMA (`VPMADD52*`)
//!   to accelerate Q120 arithmetic over three ~40-bit CRT primes. Used by
//!   schemes that work in the NTT domain (CKKS, NTT-domain GLWE).
//! - [`FFT64Ifma`] — an FFT64-domain backend that uses AVX-512F to accelerate
//!   the floating-point real/imaginary FFT. Used by schemes that work in the
//!   FFT domain (TFHE, CGGI, BinFHE blind rotation).
//!
//! Compared to the AVX2 backends in `poulpy-cpu-avx`:
//!
//! - **`NTTIfma` vs `NTT120Avx`** — IFMA reduces the CRT width from four
//!   ~30-bit primes to three ~40-bit primes and replaces split-precomputed
//!   multiplication with Harvey-style modular multiplication. The hot
//!   `vpmadd52luq`/`vpmadd52huq` instructions deliver one 52×52→104-bit
//!   multiply-accumulate per lane per cycle.
//! - **`FFT64Ifma` vs `FFT64Avx`** — the FFT/IFFT butterflies, the reim
//!   arithmetic, the i64↔f64 conversion, and the FFT16 base case all run on
//!   `__m512d`/`__m512i` (8 lanes), versus the 4-lane AVX2 version. The base
//!   case processes two FFT16 blocks per call in SIMD-parallel fashion.
//!
//! # Architecture
//!
//! `poulpy_hal` defines a hardware abstraction layer (HAL) via the
//! [`Backend`](poulpy_hal::layouts::Backend) trait and the single
//! [`HalImpl`](poulpy_hal::oep::HalImpl) trait. This crate implements
//! `HalImpl` for both [`NTTIfma`] and [`FFT64Ifma`] using AVX-512
//! intrinsics on hot paths and the portable reference logic where
//! coefficient-domain code is backend-independent.
//!
//! ## Crate layout
//!
//! | Module | Domain |
//! |---|---|
//! | `znx_ifma` | AVX-512 single-ring `Z[X]/(X^n+1)` primitives shared by both backends |
//! | `ntt_ifma` | NTT-domain backend implementation (`NTTIfma`) |
//! | `fft64`    | FFT64-domain backend implementation (`FFT64Ifma`) |
//!
//! Both backends share `module` (handle lifecycle) and `convolution` (AVX-512
//! kernels). `ntt_ifma` additionally keeps AVX-512 overrides for `svp`, `vmp`,
//! `vec_znx_dft` (CRT compaction / consume), and `vec_znx_big_avx512` (i128
//! arithmetic). Portable operations (scratch, vec_znx) are handled by shared
//! defaults from `poulpy-cpu-ref` via the `hal_impl/` macros.
//!
//! NTT-specific helpers (`ntt_ifma_avx512`, `mat_vec_ifma`) and FFT-specific
//! helpers (`reim`, `reim4`) live under their respective backend modules.
//!
//! # Scalar types
//!
//! | Backend | `ScalarPrep` | `ScalarBig` | Domain |
//! |---|---|---|---|
//! | [`NTTIfma`]   | `Q120bScalar` (4×u64 lanes; 3 active CRT residues + 1 padding) | `i128` | NTT |
//! | [`FFT64Ifma`] | `f64` (IEEE 754 double in REIM layout) | `i64` | FFT64 |
//!
//! # CPU requirements
//!
//! Both backends require an x86-64 CPU with the relevant instruction sets:
//!
//! - **`FFT64Ifma`** requires **AVX-512F** (foundation 512-bit SIMD).
//! - **`NTTIfma`** additionally requires **AVX-512IFMA**
//!   (`VPMADD52LUQ`, `VPMADD52HUQ`) and **AVX-512VL** (256-bit EVEX
//!   encodings used by reduction helpers).
//!
//! Runtime CPU feature detection is performed in `Module::new()`. If the
//! required features are not present, the constructor panics with a
//! descriptive error message. The crate's compile-time gate (`enable-ifma`)
//! requires all three target features at build time so the same binary can
//! host either backend.
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
//! - **Determinism (NTT)** — integer-domain operations are bit-exact and
//!   match the reference IFMA arithmetic model.
//! - **Determinism (FFT64)** — floating-point operations match the reference
//!   FFT64 implementation within < 0.5 ULP, ensuring correct rounding when
//!   converting back to integers via `reim_to_znx_i64_*`.
//! - **Lazy modular arithmetic** — NTT-domain values are kept in bounded
//!   lazy ranges such as `[0, 2q)`, with explicit conditional-subtract steps
//!   maintaining the required invariants.
//! - **Shared cold paths** — coefficient-domain operations that are not
//!   performance critical reuse the same backend-independent logic as the
//!   reference backend.
//! - **Cross-backend tested** — every `HalImpl` method is tested against the
//!   `poulpy_cpu_ref` reference backends via the shared `cross_backend_test_suite!`
//!   harness, plus per-primitive unit tests for AVX-512 helpers.
//!
//! # Threading and concurrency
//!
//! - **`NTTIfma` and `FFT64Ifma` are `Send + Sync`** — both are zero-sized
//!   marker types.
//! - **`Module<NTTIfma>` and `Module<FFT64Ifma>` are `Send + Sync`** — the
//!   precomputed twiddle tables are immutable after construction.
//! - **No internal locking** — synchronization is the caller's responsibility.
//!
//! # Feature flags
//!
//! - `enable-ifma` (optional): enables the backends and their compile-time
//!   target-feature checks. Without this feature the crate compiles to an
//!   empty shell so it can sit as an unused dependency on non-x86 builds.
//!
//! # Platform support
//!
//! - **Required**: x86-64 with AVX-512F (and additionally AVX-512IFMA +
//!   AVX-512VL for `NTTIfma`).
//! - **Not supported**: non-x86-64 targets and x86-64 targets without
//!   AVX-512F.
//!
//! # Usage
//!
//! This crate exports the two backend marker types and the FFT dispatcher
//! types used by `poulpy-bench`:
//!
//! - [`NTTIfma`]
//! - [`FFT64Ifma`]
//! - [`ReimFFTIfma`] / [`ReimIFFTIfma`] — backend-agnostic FFT/IFFT executors
//!   used by raw FFT benchmarks.
//!
//! Application code typically does not import this crate directly but uses
//! it via `poulpy_core`, `poulpy_schemes`, or `poulpy_ckks` with runtime
//! backend selection.

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

// Keep the crate as a true opt-in backend: without `enable-ifma`, none of the
// AVX-512 modules or their unit tests are compiled.
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
