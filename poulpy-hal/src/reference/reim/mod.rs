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
mod fft_ref;
mod fft_vec;
mod fft_vec_avx2_fma;
mod ifft_avx2_fma;
mod ifft_ref;
mod table_fft;
mod table_ifft;
mod zero;

use std::arch::global_asm;

pub use conversion::*;
pub use fft_vec::*;
pub use fft_vec_avx2_fma::*;
pub use table_fft::*;
pub use table_ifft::*;
pub use zero::*;

global_asm!(
    include_str!("fft16_avx2_fma.s"),
    include_str!("ifft16_avx2_fma.s")
);

#[inline(always)]
pub(crate) fn as_arr<const size: usize, R: Float + FloatConst>(x: &[R]) -> &[R; size] {
    debug_assert!(x.len() >= size);
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

pub trait ReimDFTExecute<D, T> {
    fn reim_dft_execute(table: &D, data: &mut [T]);
}

pub trait ReimArithmetic {
    fn reim_add(res: &mut [f64], a: &[f64], b: &[f64]);
    fn reim_add_inplace(res: &mut [f64], a: &[f64]);
    fn reim_sub(res: &mut [f64], a: &[f64], b: &[f64]);
    fn reim_sub_ab_inplace(res: &mut [f64], a: &[f64]);
    fn reim_sub_ba_inplace(res: &mut [f64], a: &[f64]);
    fn reim_negate(res: &mut [f64], a: &[f64]);
    fn reim_negate_inplace(res: &mut [f64]);
    fn reim_mul(res: &mut [f64], a: &[f64], b: &[f64]);
    fn reim_mul_inplace(res: &mut [f64], a: &[f64]);
    fn reim_addmul(res: &mut [f64], a: &[f64], b: &[f64]);
    fn reim_copy(res: &mut [f64], a: &[f64]);
    fn reim_zero(res: &mut [f64]);
}

pub trait ReimConv {
    fn reim_from_znx_i64(res: &mut [f64], a: &[i64]);
    fn reim_to_znx_i64(res: &mut [i64], divisor: f64, a: &[f64]);
    fn reim_to_znx_i64_inplace(res: &mut [f64], divisor: f64);
}
