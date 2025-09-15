use crate::reference::znx::{
    ZnxAdd, ZnxAddInplace, ZnxAutomorphism, ZnxCopy, ZnxNegate, ZnxNegateInplace, ZnxNormalizeFinalStep,
    ZnxNormalizeFinalStepInplace, ZnxNormalizeFirstStep, ZnxNormalizeFirstStepCarryOnly, ZnxNormalizeFirstStepInplace,
    ZnxNormalizeMiddleStep, ZnxNormalizeMiddleStepCarryOnly, ZnxNormalizeMiddleStepInplace, ZnxSub, ZnxSubABInplace,
    ZnxSubBAInplace, ZnxSwitchRing, ZnxZero,
    add::{znx_add_inplace_ref, znx_add_ref},
    automorphism::znx_automorphism_ref,
    copy::znx_copy_ref,
    neg::{znx_negate_inplace_ref, znx_negate_ref},
    normalization::{
        znx_normalize_final_step_inplace_ref, znx_normalize_final_step_ref, znx_normalize_first_step_carry_only_ref,
        znx_normalize_first_step_inplace_ref, znx_normalize_first_step_ref, znx_normalize_middle_step_carry_only_ref,
        znx_normalize_middle_step_inplace_ref, znx_normalize_middle_step_ref,
    },
    sub::{znx_sub_ab_inplace_ref, znx_sub_ba_inplace_ref, znx_sub_ref},
    switch_ring::znx_switch_ring_ref,
    zero::znx_zero_ref,
};

pub struct ZnxRef {}

impl ZnxAdd for ZnxRef {
    #[inline(always)]
    fn znx_add(res: &mut [i64], a: &[i64], b: &[i64]) {
        znx_add_ref(res, a, b);
    }
}

impl ZnxAddInplace for ZnxRef {
    #[inline(always)]
    fn znx_add_inplace(res: &mut [i64], a: &[i64]) {
        znx_add_inplace_ref(res, a);
    }
}

impl ZnxSub for ZnxRef {
    #[inline(always)]
    fn znx_sub(res: &mut [i64], a: &[i64], b: &[i64]) {
        znx_sub_ref(res, a, b);
    }
}

impl ZnxSubABInplace for ZnxRef {
    #[inline(always)]
    fn znx_sub_ab_inplace(res: &mut [i64], a: &[i64]) {
        znx_sub_ab_inplace_ref(res, a);
    }
}

impl ZnxSubBAInplace for ZnxRef {
    #[inline(always)]
    fn znx_sub_ba_inplace(res: &mut [i64], a: &[i64]) {
        znx_sub_ba_inplace_ref(res, a);
    }
}

impl ZnxAutomorphism for ZnxRef {
    #[inline(always)]
    fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]) {
        znx_automorphism_ref(p, res, a);
    }
}

impl ZnxCopy for ZnxRef {
    #[inline(always)]
    fn znx_copy(res: &mut [i64], a: &[i64]) {
        znx_copy_ref(res, a);
    }
}

impl ZnxNegate for ZnxRef {
    #[inline(always)]
    fn znx_negate(res: &mut [i64], src: &[i64]) {
        znx_negate_ref(res, src);
    }
}

impl ZnxNegateInplace for ZnxRef {
    #[inline(always)]
    fn znx_negate_inplace(res: &mut [i64]) {
        znx_negate_inplace_ref(res);
    }
}

impl ZnxZero for ZnxRef {
    #[inline(always)]
    fn znx_zero(res: &mut [i64]) {
        znx_zero_ref(res);
    }
}

impl ZnxSwitchRing for ZnxRef {
    #[inline(always)]
    fn znx_switch_ring(res: &mut [i64], a: &[i64]) {
        znx_switch_ring_ref(res, a);
    }
}

impl ZnxNormalizeFinalStep for ZnxRef {
    #[inline(always)]
    fn znx_normalize_final_step(basek: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        znx_normalize_final_step_ref(basek, lsh, x, a, carry);
    }
}

impl ZnxNormalizeFinalStepInplace for ZnxRef {
    #[inline(always)]
    fn znx_normalize_final_step_inplace(basek: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        znx_normalize_final_step_inplace_ref(basek, lsh, x, carry);
    }
}

impl ZnxNormalizeFirstStep for ZnxRef {
    #[inline(always)]
    fn znx_normalize_first_step(basek: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        znx_normalize_first_step_ref(basek, lsh, x, a, carry);
    }
}

impl ZnxNormalizeFirstStepCarryOnly for ZnxRef {
    #[inline(always)]
    fn znx_normalize_first_step_carry_only(basek: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
        znx_normalize_first_step_carry_only_ref(basek, lsh, x, carry);
    }
}

impl ZnxNormalizeFirstStepInplace for ZnxRef {
    #[inline(always)]
    fn znx_normalize_first_step_inplace(basek: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        znx_normalize_first_step_inplace_ref(basek, lsh, x, carry);
    }
}

impl ZnxNormalizeMiddleStep for ZnxRef {
    #[inline(always)]
    fn znx_normalize_middle_step(basek: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        znx_normalize_middle_step_ref(basek, lsh, x, a, carry);
    }
}

impl ZnxNormalizeMiddleStepCarryOnly for ZnxRef {
    #[inline(always)]
    fn znx_normalize_middle_step_carry_only(basek: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
        znx_normalize_middle_step_carry_only_ref(basek, lsh, x, carry);
    }
}

impl ZnxNormalizeMiddleStepInplace for ZnxRef {
    #[inline(always)]
    fn znx_normalize_middle_step_inplace(basek: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        znx_normalize_middle_step_inplace_ref(basek, lsh, x, carry);
    }
}
