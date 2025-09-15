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

pub use conversion::*;
pub use fft_ref::*;
pub use fft_vec::*;
pub use ifft_ref::*;
pub use table_fft::*;
pub use table_ifft::*;
pub use zero::*;

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

pub trait ReimFromZnx {
    fn reim_from_znx(res: &mut [f64], a: &[i64]);
}

pub trait ReimToZnx {
    fn reim_to_znx(res: &mut [i64], divisor: f64, a: &[f64]);
}

pub trait ReimToZnxInplace {
    fn reim_to_znx_inplace(res: &mut [f64], divisor: f64);
}

pub trait ReimAdd {
    fn reim_add(res: &mut [f64], a: &[f64], b: &[f64]);
}

pub trait ReimAddInplace {
    fn reim_add_inplace(res: &mut [f64], a: &[f64]);
}

pub trait ReimSub {
    fn reim_sub(res: &mut [f64], a: &[f64], b: &[f64]);
}

pub trait ReimSubABInplace {
    fn reim_sub_ab_inplace(res: &mut [f64], a: &[f64]);
}

pub trait ReimSubBAInplace {
    fn reim_sub_ba_inplace(res: &mut [f64], a: &[f64]);
}

pub trait ReimNegate {
    fn reim_negate(res: &mut [f64], a: &[f64]);
}

pub trait ReimNegateInplace {
    fn reim_negate_inplace(res: &mut [f64]);
}

pub trait ReimMul {
    fn reim_mul(res: &mut [f64], a: &[f64], b: &[f64]);
}

pub trait ReimMulInplace {
    fn reim_mul_inplace(res: &mut [f64], a: &[f64]);
}

pub trait ReimAddMul {
    fn reim_addmul(res: &mut [f64], a: &[f64], b: &[f64]);
}

pub trait ReimCopy {
    fn reim_copy(res: &mut [f64], a: &[f64]);
}

pub trait ReimZero {
    fn reim_zero(res: &mut [f64]);
}
