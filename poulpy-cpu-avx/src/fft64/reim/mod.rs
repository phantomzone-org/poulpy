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
mod fft_avx2_fma;
mod fft_vec_avx2_fma;
mod ifft_avx2_fma;

use std::arch::global_asm;

pub(crate) use conversion::*;
pub(crate) use fft_vec_avx2_fma::*;

use poulpy_hal::reference::fft64::reim::{ReimDFTExecute, ReimFFTTable, ReimIFFTTable};
use rand_distr::num_traits::{Float, FloatConst};

use crate::fft64::reim::{fft_avx2_fma::fft_avx2_fma, ifft_avx2_fma::ifft_avx2_fma};

global_asm!(include_str!("fft16_avx2_fma.s"), include_str!("ifft16_avx2_fma.s"));

#[inline(always)]
pub(crate) fn as_arr<const SIZE: usize, R: Float + FloatConst>(x: &[R]) -> &[R; SIZE] {
    debug_assert!(x.len() >= SIZE);
    unsafe { &*(x.as_ptr() as *const [R; SIZE]) }
}

#[inline(always)]
pub(crate) fn as_arr_mut<const SIZE: usize, R: Float + FloatConst>(x: &mut [R]) -> &mut [R; SIZE] {
    debug_assert!(x.len() >= SIZE);
    unsafe { &mut *(x.as_mut_ptr() as *mut [R; SIZE]) }
}

pub struct ReimFFTAvx;

impl ReimDFTExecute<ReimFFTTable<f64>, f64> for ReimFFTAvx {
    #[inline(always)]
    fn reim_dft_execute(table: &ReimFFTTable<f64>, data: &mut [f64]) {
        unsafe {
            fft_avx2_fma(table.m(), table.omg(), data);
        }
    }
}

pub struct ReimIFFTAvx;

impl ReimDFTExecute<ReimIFFTTable<f64>, f64> for ReimIFFTAvx {
    #[inline(always)]
    fn reim_dft_execute(table: &ReimIFFTTable<f64>, data: &mut [f64]) {
        unsafe {
            ifft_avx2_fma(table.m(), table.omg(), data);
        }
    }
}
