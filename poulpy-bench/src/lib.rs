//! Shared helpers for `poulpy-bench` benchmark binaries.
//!
//! The two macros below iterate over every available backend and call the
//! supplied benchmark function once per backend, passing the backend type as
//! the first generic argument and a human-readable label as the last argument.
//!
//! # Syntax
//!
//! ```text
//! for_each_fft_backend!(path [, leading_arg]* ; criterion_ref)
//! for_each_ntt_backend!(path [, leading_arg]* ; criterion_ref)
//! ```
//!
//! The macro expands to one call per available backend:
//!
//! ```text
//! path::<BackendType>(leading_args..., criterion_ref, "backend-label");
//! ```
//!
//! Leading args (e.g. parameter structs) are forwarded verbatim before the
//! Criterion reference; the backend label is always appended last.
//!
//! # Adding a new backend
//!
//! 1. Add the new backend crate to `[dependencies]` (optionally gated by a
//!    feature flag).
//! 2. Add one line in the appropriate macro below, e.g.:
//!    ```text
//!    #[cfg(feature = "enable-gpu")]
//!    path::<poulpy_gpu::FFT64Gpu>($($arg,)* $c, "fft64-gpu");
//!    ```
//! 3. No other files need to change.

/// Invoke `$fn::<BE>(leading_args..., c, label)` for every available FFT backend.
///
/// Currently expands to:
/// - `FFT64Ref`  (always, label `"fft64-ref"`)
/// - `FFT64Avx`  (when `feature = "enable-avx"` and `target_arch = "x86_64"`, label `"fft64-avx"`)
#[macro_export]
macro_rules! for_each_fft_backend {
    ($fn:path $(, $arg:expr)* ; $c:expr) => {{
        $fn::<poulpy_cpu_ref::FFT64Ref>($($arg,)* $c, "fft64-ref");
        #[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
        $fn::<poulpy_cpu_avx::FFT64Avx>($($arg,)* $c, "fft64-avx");
    }};
}

/// Invoke `$fn::<BE>(leading_args..., c, label)` for every available NTT backend.
///
/// Currently expands to:
/// - `NTT120Ref`  (always, label `"ntt120-ref"`)
/// - `NTT120Avx`  (when `feature = "enable-avx"` and `target_arch = "x86_64"`, label `"ntt120-avx"`)
#[macro_export]
macro_rules! for_each_ntt_backend {
    ($fn:path $(, $arg:expr)* ; $c:expr) => {{
        $fn::<poulpy_cpu_ref::NTT120Ref>($($arg,)* $c, "ntt120-ref");
        #[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
        $fn::<poulpy_cpu_avx::NTT120Avx>($($arg,)* $c, "ntt120-avx");
    }};
}
