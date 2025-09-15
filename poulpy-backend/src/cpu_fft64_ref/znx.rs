use poulpy_hal::reference::znx::{
    ZnxAdd, ZnxAddInplace, ZnxAutomorphism, ZnxCopy, ZnxNegate, ZnxNegateInplace, ZnxNormalizeFinalStep,
    ZnxNormalizeFinalStepInplace, ZnxNormalizeFirstStep, ZnxNormalizeFirstStepCarryOnly, ZnxNormalizeFirstStepInplace,
    ZnxNormalizeMiddleStep, ZnxNormalizeMiddleStepCarryOnly, ZnxNormalizeMiddleStepInplace, ZnxRotate, ZnxSub, ZnxSubABInplace,
    ZnxSubBAInplace, ZnxSwitchRing, ZnxZero, znx_add_inplace_ref, znx_add_ref, znx_automorphism_ref, znx_copy_ref,
    znx_negate_inplace_ref, znx_negate_ref, znx_normalize_final_step_inplace_ref, znx_normalize_final_step_ref,
    znx_normalize_first_step_carry_only_ref, znx_normalize_first_step_inplace_ref, znx_normalize_first_step_ref,
    znx_normalize_middle_step_carry_only_ref, znx_normalize_middle_step_inplace_ref, znx_normalize_middle_step_ref, znx_rotate,
    znx_sub_ab_inplace_ref, znx_sub_ba_inplace_ref, znx_sub_ref, znx_switch_ring_ref, znx_zero_ref,
};

use crate::cpu_fft64_ref::FFT64Ref;

impl ZnxAdd for FFT64Ref {
    #[inline(always)]
    fn znx_add(res: &mut [i64], a: &[i64], b: &[i64]) {
        znx_add_ref(res, a, b);
    }
}

impl ZnxAddInplace for FFT64Ref {
    #[inline(always)]
    fn znx_add_inplace(res: &mut [i64], a: &[i64]) {
        znx_add_inplace_ref(res, a);
    }
}

impl ZnxSub for FFT64Ref {
    #[inline(always)]
    fn znx_sub(res: &mut [i64], a: &[i64], b: &[i64]) {
        znx_sub_ref(res, a, b);
    }
}

impl ZnxSubABInplace for FFT64Ref {
    #[inline(always)]
    fn znx_sub_ab_inplace(res: &mut [i64], a: &[i64]) {
        znx_sub_ab_inplace_ref(res, a);
    }
}

impl ZnxSubBAInplace for FFT64Ref {
    #[inline(always)]
    fn znx_sub_ba_inplace(res: &mut [i64], a: &[i64]) {
        znx_sub_ba_inplace_ref(res, a);
    }
}

impl ZnxAutomorphism for FFT64Ref {
    #[inline(always)]
    fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]) {
        znx_automorphism_ref(p, res, a);
    }
}

impl ZnxCopy for FFT64Ref {
    #[inline(always)]
    fn znx_copy(res: &mut [i64], a: &[i64]) {
        znx_copy_ref(res, a);
    }
}

impl ZnxNegate for FFT64Ref {
    #[inline(always)]
    fn znx_negate(res: &mut [i64], src: &[i64]) {
        znx_negate_ref(res, src);
    }
}

impl ZnxNegateInplace for FFT64Ref {
    #[inline(always)]
    fn znx_negate_inplace(res: &mut [i64]) {
        znx_negate_inplace_ref(res);
    }
}

impl ZnxRotate for FFT64Ref {
    #[inline(always)]
    fn znx_rotate(p: i64, res: &mut [i64], src: &[i64]) {
        znx_rotate::<Self>(p, res, src);
    }
}

impl ZnxZero for FFT64Ref {
    #[inline(always)]
    fn znx_zero(res: &mut [i64]) {
        znx_zero_ref(res);
    }
}

impl ZnxSwitchRing for FFT64Ref {
    #[inline(always)]
    fn znx_switch_ring(res: &mut [i64], a: &[i64]) {
        znx_switch_ring_ref(res, a);
    }
}

impl ZnxNormalizeFinalStep for FFT64Ref {
    #[inline(always)]
    fn znx_normalize_final_step(basek: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        znx_normalize_final_step_ref(basek, lsh, x, a, carry);
    }
}

impl ZnxNormalizeFinalStepInplace for FFT64Ref {
    #[inline(always)]
    fn znx_normalize_final_step_inplace(basek: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        znx_normalize_final_step_inplace_ref(basek, lsh, x, carry);
    }
}

impl ZnxNormalizeFirstStep for FFT64Ref {
    #[inline(always)]
    fn znx_normalize_first_step(basek: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        znx_normalize_first_step_ref(basek, lsh, x, a, carry);
    }
}

impl ZnxNormalizeFirstStepCarryOnly for FFT64Ref {
    #[inline(always)]
    fn znx_normalize_first_step_carry_only(basek: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
        znx_normalize_first_step_carry_only_ref(basek, lsh, x, carry);
    }
}

impl ZnxNormalizeFirstStepInplace for FFT64Ref {
    #[inline(always)]
    fn znx_normalize_first_step_inplace(basek: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        znx_normalize_first_step_inplace_ref(basek, lsh, x, carry);
    }
}

impl ZnxNormalizeMiddleStep for FFT64Ref {
    #[inline(always)]
    fn znx_normalize_middle_step(basek: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        znx_normalize_middle_step_ref(basek, lsh, x, a, carry);
    }
}

impl ZnxNormalizeMiddleStepCarryOnly for FFT64Ref {
    #[inline(always)]
    fn znx_normalize_middle_step_carry_only(basek: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
        znx_normalize_middle_step_carry_only_ref(basek, lsh, x, carry);
    }
}

impl ZnxNormalizeMiddleStepInplace for FFT64Ref {
    #[inline(always)]
    fn znx_normalize_middle_step_inplace(basek: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        znx_normalize_middle_step_inplace_ref(basek, lsh, x, carry);
    }
}
