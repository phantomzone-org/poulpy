use crate::reference::znx::{
    ZnxArithmetic,
    add::{znx_add_inplace_ref, znx_add_ref},
    automorphism::znx_automorphism_ref,
    copy::znx_copy_ref,
    neg::{znx_negate_inplace_ref, znx_negate_ref},
    rotate::znx_rotate,
    sub::{znx_sub_ab_inplace_ref, znx_sub_ba_inplace_ref, znx_sub_ref},
    switch_ring::znx_switch_ring_ref,
    zero::znx_zero_ref,
};

pub struct ZnxArithmeticRef;

impl ZnxArithmetic for ZnxArithmeticRef {
    #[inline(always)]
    fn znx_add(res: &mut [i64], a: &[i64], b: &[i64]) {
        znx_add_ref(res, a, b);
    }
    #[inline(always)]
    fn znx_sub(res: &mut [i64], a: &[i64], b: &[i64]) {
        znx_sub_ref(res, a, b);
    }
    #[inline(always)]
    fn znx_sub_ab_inplace(res: &mut [i64], a: &[i64]) {
        znx_sub_ab_inplace_ref(res, a);
    }
    #[inline(always)]
    fn znx_sub_ba_inplace(res: &mut [i64], a: &[i64]) {
        znx_sub_ba_inplace_ref(res, a);
    }
    #[inline(always)]
    fn znx_add_inplace(res: &mut [i64], a: &[i64]) {
        znx_add_inplace_ref(res, a);
    }
    #[inline(always)]
    fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]) {
        znx_automorphism_ref(p, res, a);
    }
    #[inline(always)]
    fn znx_copy(res: &mut [i64], a: &[i64]) {
        znx_copy_ref(res, a);
    }
    #[inline(always)]
    fn znx_negate(res: &mut [i64], src: &[i64]) {
        znx_negate_ref(res, src);
    }
    #[inline(always)]
    fn znx_negate_inplace(res: &mut [i64]) {
        znx_negate_inplace_ref(res);
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
        znx_switch_ring_ref(res, a);
    }
}
