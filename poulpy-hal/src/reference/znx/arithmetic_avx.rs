use crate::reference::znx::{
    ZnxArithmetic,
    add::{znx_add_avx, znx_add_inplace_avx},
    automorphism::znx_automorphism_avx,
    copy::znx_copy_ref,
    neg::{znx_negate_avx, znx_negate_inplace_avx},
    rotate::znx_rotate,
    sub::{znx_sub_ab_inplace_avx, znx_sub_avx, znx_sub_ba_inplace_avx},
    switch_ring::znx_switch_ring_avx,
    zero::znx_zero_ref,
};

pub struct ZnxArithmeticAvx;

impl ZnxArithmetic for ZnxArithmeticAvx {
    #[inline(always)]
    fn znx_add(res: &mut [i64], a: &[i64], b: &[i64]) {
        unsafe {
            znx_add_avx(res, a, b);
        }
    }
    #[inline(always)]
    fn znx_sub(res: &mut [i64], a: &[i64], b: &[i64]) {
        unsafe {
            znx_sub_avx(res, a, b);
        }
    }
    #[inline(always)]
    fn znx_sub_ab_inplace(res: &mut [i64], a: &[i64]) {
        unsafe {
            znx_sub_ab_inplace_avx(res, a);
        }
    }
    #[inline(always)]
    fn znx_sub_ba_inplace(res: &mut [i64], a: &[i64]) {
        unsafe {
            znx_sub_ba_inplace_avx(res, a);
        }
    }
    #[inline(always)]
    fn znx_add_inplace(res: &mut [i64], a: &[i64]) {
        unsafe {
            znx_add_inplace_avx(res, a);
        }
    }
    #[inline(always)]
    fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]) {
        unsafe {
            znx_automorphism_avx(p, res, a);
        }
    }
    #[inline(always)]
    fn znx_copy(res: &mut [i64], a: &[i64]) {
        znx_copy_ref(res, a);
    }
    #[inline(always)]
    fn znx_negate(res: &mut [i64], src: &[i64]) {
        unsafe {
            znx_negate_avx(res, src);
        }
    }
    #[inline(always)]
    fn znx_negate_inplace(res: &mut [i64]) {
        unsafe {
            znx_negate_inplace_avx(res);
        }
    }
    #[inline(always)]
    fn znx_rotate(p: i64, res: &mut [i64], src: &[i64]) {
        znx_rotate::<Self>(p, res, src);
    }
    #[inline(always)]
    fn znx_zero(res: &mut [i64]) {
        znx_zero_ref(res);
    }
    #[inline(always)]
    fn znx_switch_ring(res: &mut [i64], a: &[i64]) {
        unsafe {
            znx_switch_ring_avx(res, a);
        }
    }
}
