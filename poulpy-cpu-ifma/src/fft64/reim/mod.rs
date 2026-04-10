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

mod conversion;
mod fft_avx512;
mod fft_vec_avx512;
mod ifft_avx512;

pub(crate) use conversion::*;
pub(crate) use fft_vec_avx512::*;

use poulpy_hal::reference::fft64::reim::{ReimFFTExecute, ReimFFTTable, ReimIFFTTable};
use rand_distr::num_traits::{Float, FloatConst};

use crate::fft64::reim::{fft_avx512::fft_avx512, ifft_avx512::ifft_avx512};

#[inline(always)]
pub(crate) fn as_arr<const SIZE: usize, R: Float + FloatConst>(x: &[R]) -> &[R; SIZE] {
    debug_assert!(x.len() >= SIZE);
    unsafe { &*(x.as_ptr() as *const [R; SIZE]) }
}

pub struct ReimFFTIfma;

impl ReimFFTExecute<ReimFFTTable<f64>, f64> for ReimFFTIfma {
    #[inline(always)]
    fn reim_dft_execute(table: &ReimFFTTable<f64>, data: &mut [f64]) {
        unsafe {
            fft_avx512(table.m(), table.omg(), data);
        }
    }
}

pub struct ReimIFFTIfma;

impl ReimFFTExecute<ReimIFFTTable<f64>, f64> for ReimIFFTIfma {
    #[inline(always)]
    fn reim_dft_execute(table: &ReimIFFTTable<f64>, data: &mut [f64]) {
        unsafe {
            ifft_avx512(table.m(), table.omg(), data);
        }
    }
}
