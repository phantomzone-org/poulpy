// ----------------------------------------------------------------------
// DISCLAIMER
//
// This module contains code that has been directly ported from the
// spqlios-arithmetic library
// (https://github.com/tfhe/spqlios-arithmetic), which is licensed
// under the Apache License, Version 2.0.
//
// The porting process from C to Rust was done with minimal changes
// in order to preserve the semantics and performance characteristics
// of the original implementation.
//
// Both Poulpy and spqlios-arithmetic are distributed under the terms
// of the Apache License, Version 2.0. See the LICENSE file for details.
//
// ----------------------------------------------------------------------

#![allow(bad_asm_style)]

mod conversion;
mod fft_ref;
mod fft_vec;
mod ifft_ref;
mod table_fft;
mod table_ifft;
mod zero;

use std::fmt::Debug;

pub use conversion::*;
pub use fft_ref::*;
pub use fft_vec::*;
pub use ifft_ref::*;
pub use table_fft::*;
pub use table_ifft::*;
pub use zero::*;

#[inline(always)]
pub(crate) fn as_arr<const size: usize, R: Float + FloatConst>(x: &[R]) -> &[R; size] {
    debug_assert!(x.len() >= size, "x.len():{} < size:{}", x.len(), size);
    unsafe { &*(x.as_ptr() as *const [R; size]) }
}

#[inline(always)]
pub(crate) fn as_arr_mut<const size: usize, R: Float + FloatConst>(x: &mut [R]) -> &mut [R; size] {
    debug_assert!(x.len() >= size);
    unsafe { &mut *(x.as_mut_ptr() as *mut [R; size]) }
}

use rand_distr::num_traits::{Float, FloatConst};
#[inline(always)]
pub(crate) fn frac_rev_bits<R: Float + FloatConst>(x: usize) -> R {
    let half: R = R::from(0.5).unwrap();

    match x {
        0 => R::zero(),
        1 => half,
        _ => {
            if x.is_multiple_of(2) {
                frac_rev_bits::<R>(x >> 1) * half
            } else {
                frac_rev_bits::<R>(x >> 1) * half + half
            }
        }
    }
}

pub trait FFTModuleHandle<F: Float + FloatConst + Debug> {
    /// Returns a shared reference to the forward FFT twiddle table.
    ///
    /// # Complexity
    ///
    /// O(1) — simple pointer dereference.
    fn get_fft_table(&self) -> &ReimFFTTable<F>;

    /// Returns a shared reference to the inverse FFT twiddle table.
    ///
    /// # Complexity
    ///
    /// O(1) — simple pointer dereference.
    fn get_ifft_table(&self) -> &ReimIFFTTable<F>;
}

/// Backend-side accessor for FFT/IFFT twiddle tables stored in `B::Handle`.
///
/// Implement this on a backend type `B` (not on `Module<B>`) to avoid orphan-rule
/// violations in downstream crates. The blanket `impl FFTModuleHandle<F> for Module<B>`
/// below bridges this into the public API automatically.
pub trait FFTHandleAccessor<F: Float + FloatConst + Debug>: crate::layouts::Backend {
    fn get_fft_table_from_handle(handle: &Self::Handle) -> &ReimFFTTable<F>;
    fn get_ifft_table_from_handle(handle: &Self::Handle) -> &ReimIFFTTable<F>;
}

impl<F: Float + FloatConst + Debug, B: FFTHandleAccessor<F>> FFTModuleHandle<F> for crate::layouts::Module<B> {
    fn get_fft_table(&self) -> &ReimFFTTable<F> {
        B::get_fft_table_from_handle(unsafe { &*self.ptr() })
    }
    fn get_ifft_table(&self) -> &ReimIFFTTable<F> {
        B::get_ifft_table_from_handle(unsafe { &*self.ptr() })
    }
}

pub trait ReimFFTExecute<D, T> {
    fn reim_dft_execute(table: &D, data: &mut [T]);
}

pub trait ReimArith {
    fn reim_from_znx(res: &mut [f64], a: &[i64]) {
        reim_from_znx_i64_ref(res, a)
    }

    fn reim_to_znx(res: &mut [i64], divisor: f64, a: &[f64]) {
        reim_to_znx_i64_ref(res, divisor, a)
    }

    fn reim_to_znx_inplace(res: &mut [f64], divisor: f64) {
        reim_to_znx_i64_inplace_ref(res, divisor)
    }

    fn reim_add(res: &mut [f64], a: &[f64], b: &[f64]) {
        reim_add_ref(res, a, b)
    }

    fn reim_add_inplace(res: &mut [f64], a: &[f64]) {
        reim_add_inplace_ref(res, a)
    }

    fn reim_sub(res: &mut [f64], a: &[f64], b: &[f64]) {
        reim_sub_ref(res, a, b)
    }

    fn reim_sub_inplace(res: &mut [f64], a: &[f64]) {
        reim_sub_inplace_ref(res, a)
    }

    fn reim_sub_negate_inplace(res: &mut [f64], a: &[f64]) {
        reim_sub_negate_inplace_ref(res, a)
    }

    fn reim_negate(res: &mut [f64], a: &[f64]) {
        reim_negate_ref(res, a)
    }

    fn reim_negate_inplace(res: &mut [f64]) {
        reim_negate_inplace_ref(res)
    }

    fn reim_mul(res: &mut [f64], a: &[f64], b: &[f64]) {
        reim_mul_ref(res, a, b)
    }

    fn reim_mul_inplace(res: &mut [f64], a: &[f64]) {
        reim_mul_inplace_ref(res, a)
    }

    fn reim_addmul(res: &mut [f64], a: &[f64], b: &[f64]) {
        reim_addmul_ref(res, a, b)
    }

    fn reim_copy(res: &mut [f64], a: &[f64]) {
        reim_copy_ref(res, a)
    }

    fn reim_zero(res: &mut [f64]) {
        reim_zero_ref(res)
    }
}
