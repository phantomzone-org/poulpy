// ─────────────────────────────────────────────────────────────
// Build the backend **only when ALL conditions are satisfied**
// ─────────────────────────────────────────────────────────────
#![cfg(all(feature = "enable-avx", target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]

// If the user enables this backend but targets a non-x86_64 CPU → abort
#[cfg(all(feature = "enable-avx", not(target_arch = "x86_64")))]
compile_error!("feature `enable-avx` requires target_arch = \"x86_64\".");

// If the user enables this backend but AVX2 isn't enabled in the target → abort
#[cfg(all(feature = "enable-avx", target_arch = "x86_64", not(target_feature = "avx2")))]
compile_error!("feature `enable-avx` requires AVX2. Build with RUSTFLAGS=\"-C target-feature=+avx2\".");

// If the user enables this backend but FMA isn't enabled in the target → abort
#[cfg(all(feature = "enable-avx", target_arch = "x86_64", not(target_feature = "fma")))]
compile_error!("feature `enable-avx` requires FMA. Build with RUSTFLAGS=\"-C target-feature=+fma\".");

mod convolution;
mod module;
mod reim;
mod reim4;
mod scratch;
mod svp;
mod vec_znx;
mod vec_znx_big;
mod vec_znx_dft;
mod vmp;
mod znx_avx;

pub struct FFT64Avx {}
pub use reim::*;

#[cfg(test)]
pub mod tests;
