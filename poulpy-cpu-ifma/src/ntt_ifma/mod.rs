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

pub(crate) mod convolution;
pub(crate) mod mat_vec_ifma;
mod module;
mod prim;
mod svp;
mod vec_znx_big_avx512;
pub(crate) mod vec_znx_dft;
pub(crate) mod vmp;
mod znx;

pub(crate) mod ntt_ifma_avx512;

#[cfg(test)]
mod tests;

/// AVX512-IFMA accelerated NTT CPU backend for Poulpy HAL.
///
/// `NTTIfma` is a zero-sized marker type that selects the AVX512-IFMA accelerated IFMA backend
/// when used as the type parameter `B` in [`poulpy_hal::layouts::Module<B>`](poulpy_hal::layouts::Module)
/// and related HAL types. It implements the unified [`HalImpl`](poulpy_hal::oep::HalImpl) trait
/// via macros in `hal_impl.rs`.
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

use poulpy_cpu_ref::reference::ntt120::{I128BigOps, I128NormalizeOps};

use vec_znx_big_avx512::{
    nfc_final_step_inplace_avx512, nfc_final_step_inplace_scalar, nfc_middle_step_avx512, nfc_middle_step_inplace_avx512,
    nfc_middle_step_inplace_scalar, nfc_middle_step_scalar, vi128_add_avx512, vi128_add_inplace_avx512, vi128_add_small_avx512,
    vi128_add_small_inplace_avx512, vi128_from_small_avx512, vi128_neg_from_small_avx512, vi128_negate_avx512,
    vi128_negate_inplace_avx512, vi128_sub_avx512, vi128_sub_inplace_avx512, vi128_sub_negate_inplace_avx512,
    vi128_sub_small_a_avx512, vi128_sub_small_b_avx512, vi128_sub_small_inplace_avx512, vi128_sub_small_negate_inplace_avx512,
};

impl I128BigOps for NTTIfma {
    #[inline(always)]
    fn i128_add(res: &mut [i128], a: &[i128], b: &[i128]) {
        unsafe { vi128_add_avx512(res.len(), res, a, b) }
    }
    #[inline(always)]
    fn i128_add_inplace(res: &mut [i128], a: &[i128]) {
        unsafe { vi128_add_inplace_avx512(res.len(), res, a) }
    }
    #[inline(always)]
    fn i128_add_small(res: &mut [i128], a: &[i128], b: &[i64]) {
        unsafe { vi128_add_small_avx512(res.len(), res, a, b) }
    }
    #[inline(always)]
    fn i128_add_small_inplace(res: &mut [i128], a: &[i64]) {
        unsafe { vi128_add_small_inplace_avx512(res.len(), res, a) }
    }
    #[inline(always)]
    fn i128_sub(res: &mut [i128], a: &[i128], b: &[i128]) {
        unsafe { vi128_sub_avx512(res.len(), res, a, b) }
    }
    #[inline(always)]
    fn i128_sub_inplace(res: &mut [i128], a: &[i128]) {
        unsafe { vi128_sub_inplace_avx512(res.len(), res, a) }
    }
    #[inline(always)]
    fn i128_sub_negate_inplace(res: &mut [i128], a: &[i128]) {
        unsafe { vi128_sub_negate_inplace_avx512(res.len(), res, a) }
    }
    #[inline(always)]
    fn i128_sub_small_a(res: &mut [i128], a: &[i64], b: &[i128]) {
        unsafe { vi128_sub_small_a_avx512(res.len(), res, a, b) }
    }
    #[inline(always)]
    fn i128_sub_small_b(res: &mut [i128], a: &[i128], b: &[i64]) {
        unsafe { vi128_sub_small_b_avx512(res.len(), res, a, b) }
    }
    #[inline(always)]
    fn i128_sub_small_inplace(res: &mut [i128], a: &[i64]) {
        unsafe { vi128_sub_small_inplace_avx512(res.len(), res, a) }
    }
    #[inline(always)]
    fn i128_sub_small_negate_inplace(res: &mut [i128], a: &[i64]) {
        unsafe { vi128_sub_small_negate_inplace_avx512(res.len(), res, a) }
    }
    #[inline(always)]
    fn i128_negate(res: &mut [i128], a: &[i128]) {
        unsafe { vi128_negate_avx512(res.len(), res, a) }
    }
    #[inline(always)]
    fn i128_negate_inplace(res: &mut [i128]) {
        unsafe { vi128_negate_inplace_avx512(res.len(), res) }
    }
    #[inline(always)]
    fn i128_neg_from_small(res: &mut [i128], a: &[i64]) {
        unsafe { vi128_neg_from_small_avx512(res.len(), res, a) }
    }
    #[inline(always)]
    fn i128_from_small(res: &mut [i128], a: &[i64]) {
        unsafe { vi128_from_small_avx512(res.len(), res, a) }
    }
}

impl I128NormalizeOps for NTTIfma {
    #[inline(always)]
    fn nfc_middle_step(base2k: usize, lsh: usize, res: &mut [i64], a: &[i128], carry: &mut [i128]) {
        if base2k <= 64 && res.len() >= 8 {
            unsafe { nfc_middle_step_avx512(base2k as u32, lsh as u32, res.len(), res, a, carry) }
        } else {
            nfc_middle_step_scalar(base2k, lsh, res, a, carry);
        }
    }
    #[inline(always)]
    fn nfc_middle_step_inplace(base2k: usize, lsh: usize, res: &mut [i64], carry: &mut [i128]) {
        if base2k <= 64 && res.len() >= 8 {
            unsafe { nfc_middle_step_inplace_avx512(base2k as u32, lsh as u32, res.len(), res, carry) }
        } else {
            nfc_middle_step_inplace_scalar(base2k, lsh, res, carry);
        }
    }
    #[inline(always)]
    fn nfc_final_step_inplace(base2k: usize, lsh: usize, res: &mut [i64], carry: &mut [i128]) {
        if base2k <= 64 && res.len() >= 8 {
            unsafe { nfc_final_step_inplace_avx512(base2k as u32, lsh as u32, res.len(), res, carry) }
        } else {
            nfc_final_step_inplace_scalar(base2k, lsh, res, carry);
        }
    }
}
