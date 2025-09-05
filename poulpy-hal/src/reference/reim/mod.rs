//! ----------------------------------------------------------------------
//! DISCLAIMER
//! This module is a direct port of the C implementation from
//! spqlios-arithmetic (https://github.com/tfhe/spqlios-arithmetic).
//! Functions have been translated to Rust with minimal changes
//! to preserve semantics and performance.
//! ----------------------------------------------------------------------

#![allow(bad_asm_style)]

mod table_fft;
mod table_ifft;

use std::arch::global_asm;

pub use table_fft::*;
pub use table_ifft::*;

mod fft_avx2_fma;
mod fft_ref;
mod ifft_avx2_fma;
mod ifft_ref;

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
        0 => return R::zero(),
        1 => return half,
        _ => {
            if x.is_multiple_of(2) {
                frac_rev_bits::<R>(x >> 1) * half
            } else {
                frac_rev_bits::<R>(x >> 1) * half + half
            }
        }
    }
}
