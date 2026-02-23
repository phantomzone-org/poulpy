//! Single ring element (`Z[X]/(X^n+1)`) arithmetic for [`NTT120Ref`](crate::NTT120Ref).
//!
//! Implements the `Znx*` traits from `poulpy_hal::reference::znx`. All implementations
//! delegate to the same `_ref` functions as `poulpy-cpu-ref` — these operate on plain
//! `&[i64]` slices, which are backend-independent.

use poulpy_hal::reference::znx::{
    ZnxAdd, ZnxAddInplace, ZnxAutomorphism, ZnxCopy, ZnxExtractDigitAddMul, ZnxMulAddPowerOfTwo, ZnxMulPowerOfTwo,
    ZnxMulPowerOfTwoInplace, ZnxNegate, ZnxNegateInplace, ZnxNormalizeDigit, ZnxNormalizeFinalStep, ZnxNormalizeFinalStepInplace,
    ZnxNormalizeFirstStep, ZnxNormalizeFirstStepCarryOnly, ZnxNormalizeFirstStepInplace, ZnxNormalizeMiddleStep,
    ZnxNormalizeMiddleStepCarryOnly, ZnxNormalizeMiddleStepInplace, ZnxRotate, ZnxSub, ZnxSubInplace, ZnxSubNegateInplace,
    ZnxSwitchRing, ZnxZero, znx_add_inplace_ref, znx_add_ref, znx_automorphism_ref, znx_copy_ref, znx_extract_digit_addmul_ref,
    znx_mul_add_power_of_two_ref, znx_mul_power_of_two_inplace_ref, znx_mul_power_of_two_ref, znx_negate_inplace_ref,
    znx_negate_ref, znx_normalize_digit_ref, znx_normalize_final_step_inplace_ref, znx_normalize_final_step_ref,
    znx_normalize_first_step_carry_only_ref, znx_normalize_first_step_inplace_ref, znx_normalize_first_step_ref,
    znx_normalize_middle_step_carry_only_ref, znx_normalize_middle_step_inplace_ref, znx_normalize_middle_step_ref, znx_rotate,
    znx_sub_inplace_ref, znx_sub_negate_inplace_ref, znx_sub_ref, znx_switch_ring_ref, znx_zero_ref,
};

use crate::NTT120Ref;

impl ZnxAdd for NTT120Ref {
    #[inline(always)]
    fn znx_add(res: &mut [i64], a: &[i64], b: &[i64]) {
        znx_add_ref(res, a, b);
    }
}

impl ZnxAddInplace for NTT120Ref {
    #[inline(always)]
    fn znx_add_inplace(res: &mut [i64], a: &[i64]) {
        znx_add_inplace_ref(res, a);
    }
}

impl ZnxSub for NTT120Ref {
    #[inline(always)]
    fn znx_sub(res: &mut [i64], a: &[i64], b: &[i64]) {
        znx_sub_ref(res, a, b);
    }
}

impl ZnxSubInplace for NTT120Ref {
    #[inline(always)]
    fn znx_sub_inplace(res: &mut [i64], a: &[i64]) {
        znx_sub_inplace_ref(res, a);
    }
}

impl ZnxSubNegateInplace for NTT120Ref {
    #[inline(always)]
    fn znx_sub_negate_inplace(res: &mut [i64], a: &[i64]) {
        znx_sub_negate_inplace_ref(res, a);
    }
}

impl ZnxMulAddPowerOfTwo for NTT120Ref {
    #[inline(always)]
    fn znx_muladd_power_of_two(k: i64, res: &mut [i64], a: &[i64]) {
        znx_mul_add_power_of_two_ref(k, res, a);
    }
}

impl ZnxMulPowerOfTwo for NTT120Ref {
    #[inline(always)]
    fn znx_mul_power_of_two(k: i64, res: &mut [i64], a: &[i64]) {
        znx_mul_power_of_two_ref(k, res, a);
    }
}

impl ZnxMulPowerOfTwoInplace for NTT120Ref {
    #[inline(always)]
    fn znx_mul_power_of_two_inplace(k: i64, res: &mut [i64]) {
        znx_mul_power_of_two_inplace_ref(k, res);
    }
}

impl ZnxAutomorphism for NTT120Ref {
    #[inline(always)]
    fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]) {
        znx_automorphism_ref(p, res, a);
    }
}

impl ZnxCopy for NTT120Ref {
    #[inline(always)]
    fn znx_copy(res: &mut [i64], a: &[i64]) {
        znx_copy_ref(res, a);
    }
}

impl ZnxNegate for NTT120Ref {
    #[inline(always)]
    fn znx_negate(res: &mut [i64], src: &[i64]) {
        znx_negate_ref(res, src);
    }
}

impl ZnxNegateInplace for NTT120Ref {
    #[inline(always)]
    fn znx_negate_inplace(res: &mut [i64]) {
        znx_negate_inplace_ref(res);
    }
}

impl ZnxRotate for NTT120Ref {
    #[inline(always)]
    fn znx_rotate(p: i64, res: &mut [i64], src: &[i64]) {
        znx_rotate::<Self>(p, res, src);
    }
}

impl ZnxZero for NTT120Ref {
    #[inline(always)]
    fn znx_zero(res: &mut [i64]) {
        znx_zero_ref(res);
    }
}

impl ZnxSwitchRing for NTT120Ref {
    #[inline(always)]
    fn znx_switch_ring(res: &mut [i64], a: &[i64]) {
        znx_switch_ring_ref(res, a);
    }
}

impl ZnxNormalizeFinalStep for NTT120Ref {
    #[inline(always)]
    fn znx_normalize_final_step(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        znx_normalize_final_step_ref(base2k, lsh, x, a, carry);
    }
}

impl ZnxNormalizeFinalStepInplace for NTT120Ref {
    #[inline(always)]
    fn znx_normalize_final_step_inplace(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        znx_normalize_final_step_inplace_ref(base2k, lsh, x, carry);
    }
}

impl ZnxNormalizeFirstStep for NTT120Ref {
    #[inline(always)]
    fn znx_normalize_first_step(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        znx_normalize_first_step_ref(base2k, lsh, x, a, carry);
    }
}

impl ZnxNormalizeFirstStepCarryOnly for NTT120Ref {
    #[inline(always)]
    fn znx_normalize_first_step_carry_only(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
        znx_normalize_first_step_carry_only_ref(base2k, lsh, x, carry);
    }
}

impl ZnxNormalizeFirstStepInplace for NTT120Ref {
    #[inline(always)]
    fn znx_normalize_first_step_inplace(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        znx_normalize_first_step_inplace_ref(base2k, lsh, x, carry);
    }
}

impl ZnxNormalizeMiddleStep for NTT120Ref {
    #[inline(always)]
    fn znx_normalize_middle_step(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        znx_normalize_middle_step_ref(base2k, lsh, x, a, carry);
    }
}

impl ZnxNormalizeMiddleStepCarryOnly for NTT120Ref {
    #[inline(always)]
    fn znx_normalize_middle_step_carry_only(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
        znx_normalize_middle_step_carry_only_ref(base2k, lsh, x, carry);
    }
}

impl ZnxNormalizeMiddleStepInplace for NTT120Ref {
    #[inline(always)]
    fn znx_normalize_middle_step_inplace(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        znx_normalize_middle_step_inplace_ref(base2k, lsh, x, carry);
    }
}

impl ZnxExtractDigitAddMul for NTT120Ref {
    #[inline(always)]
    fn znx_extract_digit_addmul(base2k: usize, lsh: usize, res: &mut [i64], src: &mut [i64]) {
        znx_extract_digit_addmul_ref(base2k, lsh, res, src);
    }
}

impl ZnxNormalizeDigit for NTT120Ref {
    #[inline(always)]
    fn znx_normalize_digit(base2k: usize, res: &mut [i64], src: &mut [i64]) {
        znx_normalize_digit_ref(base2k, res, src);
    }
}
