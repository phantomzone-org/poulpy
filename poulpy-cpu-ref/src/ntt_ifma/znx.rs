//! Single ring element (`Z[X]/(X^n+1)`) arithmetic for [`NTTIfmaRef`](crate::NTTIfmaRef).
//!
//! Implements the `Znx*` traits from `poulpy_hal::reference::znx`. All implementations
//! delegate to the same `_ref` functions as `poulpy-cpu-ref` -- these operate on plain
//! `&[i64]` slices, which are backend-independent.

use poulpy_hal::reference::znx::{
    ZnxAdd, ZnxAddInplace, ZnxAutomorphism, ZnxCopy, ZnxExtractDigitAddMul, ZnxMulAddPowerOfTwo, ZnxMulPowerOfTwo,
    ZnxMulPowerOfTwoInplace, ZnxNegate, ZnxNegateInplace, ZnxNormalizeDigit, ZnxNormalizeFinalStep, ZnxNormalizeFinalStepInplace,
    ZnxNormalizeFinalStepSub, ZnxNormalizeFirstStep, ZnxNormalizeFirstStepCarryOnly, ZnxNormalizeFirstStepInplace,
    ZnxNormalizeMiddleStep, ZnxNormalizeMiddleStepCarryOnly, ZnxNormalizeMiddleStepInplace, ZnxNormalizeMiddleStepSub, ZnxRotate,
    ZnxSub, ZnxSubInplace, ZnxSubNegateInplace, ZnxSwitchRing, ZnxZero, znx_add_inplace_ref, znx_add_ref, znx_automorphism_ref,
    znx_copy_ref, znx_extract_digit_addmul_ref, znx_mul_add_power_of_two_ref, znx_mul_power_of_two_inplace_ref,
    znx_mul_power_of_two_ref, znx_negate_inplace_ref, znx_negate_ref, znx_normalize_digit_ref,
    znx_normalize_final_step_inplace_ref, znx_normalize_final_step_ref, znx_normalize_final_step_sub_ref,
    znx_normalize_first_step_carry_only_ref, znx_normalize_first_step_inplace_ref, znx_normalize_first_step_ref,
    znx_normalize_middle_step_carry_only_ref, znx_normalize_middle_step_inplace_ref, znx_normalize_middle_step_ref,
    znx_normalize_middle_step_sub_ref, znx_rotate, znx_sub_inplace_ref, znx_sub_negate_inplace_ref, znx_sub_ref,
    znx_switch_ring_ref, znx_zero_ref,
};

use crate::NTTIfmaRef;

impl ZnxAdd for NTTIfmaRef {
    #[inline(always)]
    fn znx_add(res: &mut [i64], a: &[i64], b: &[i64]) {
        znx_add_ref(res, a, b);
    }
}

impl ZnxAddInplace for NTTIfmaRef {
    #[inline(always)]
    fn znx_add_inplace(res: &mut [i64], a: &[i64]) {
        znx_add_inplace_ref(res, a);
    }
}

impl ZnxSub for NTTIfmaRef {
    #[inline(always)]
    fn znx_sub(res: &mut [i64], a: &[i64], b: &[i64]) {
        znx_sub_ref(res, a, b);
    }
}

impl ZnxSubInplace for NTTIfmaRef {
    #[inline(always)]
    fn znx_sub_inplace(res: &mut [i64], a: &[i64]) {
        znx_sub_inplace_ref(res, a);
    }
}

impl ZnxSubNegateInplace for NTTIfmaRef {
    #[inline(always)]
    fn znx_sub_negate_inplace(res: &mut [i64], a: &[i64]) {
        znx_sub_negate_inplace_ref(res, a);
    }
}

impl ZnxMulAddPowerOfTwo for NTTIfmaRef {
    #[inline(always)]
    fn znx_muladd_power_of_two(k: i64, res: &mut [i64], a: &[i64]) {
        znx_mul_add_power_of_two_ref(k, res, a);
    }
}

impl ZnxMulPowerOfTwo for NTTIfmaRef {
    #[inline(always)]
    fn znx_mul_power_of_two(k: i64, res: &mut [i64], a: &[i64]) {
        znx_mul_power_of_two_ref(k, res, a);
    }
}

impl ZnxMulPowerOfTwoInplace for NTTIfmaRef {
    #[inline(always)]
    fn znx_mul_power_of_two_inplace(k: i64, res: &mut [i64]) {
        znx_mul_power_of_two_inplace_ref(k, res);
    }
}

impl ZnxAutomorphism for NTTIfmaRef {
    #[inline(always)]
    fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]) {
        znx_automorphism_ref(p, res, a);
    }
}

impl ZnxCopy for NTTIfmaRef {
    #[inline(always)]
    fn znx_copy(res: &mut [i64], a: &[i64]) {
        znx_copy_ref(res, a);
    }
}

impl ZnxNegate for NTTIfmaRef {
    #[inline(always)]
    fn znx_negate(res: &mut [i64], src: &[i64]) {
        znx_negate_ref(res, src);
    }
}

impl ZnxNegateInplace for NTTIfmaRef {
    #[inline(always)]
    fn znx_negate_inplace(res: &mut [i64]) {
        znx_negate_inplace_ref(res);
    }
}

impl ZnxRotate for NTTIfmaRef {
    #[inline(always)]
    fn znx_rotate(p: i64, res: &mut [i64], src: &[i64]) {
        znx_rotate::<Self>(p, res, src);
    }
}

impl ZnxZero for NTTIfmaRef {
    #[inline(always)]
    fn znx_zero(res: &mut [i64]) {
        znx_zero_ref(res);
    }
}

impl ZnxSwitchRing for NTTIfmaRef {
    #[inline(always)]
    fn znx_switch_ring(res: &mut [i64], a: &[i64]) {
        znx_switch_ring_ref(res, a);
    }
}

impl ZnxNormalizeFinalStep for NTTIfmaRef {
    #[inline(always)]
    fn znx_normalize_final_step<const OVERWRITE: bool>(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        znx_normalize_final_step_ref::<OVERWRITE>(base2k, lsh, x, a, carry);
    }
}

impl ZnxNormalizeFinalStepSub for NTTIfmaRef {
    #[inline(always)]
    fn znx_normalize_final_step_sub(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        znx_normalize_final_step_sub_ref(base2k, lsh, x, a, carry);
    }
}

impl ZnxNormalizeFinalStepInplace for NTTIfmaRef {
    #[inline(always)]
    fn znx_normalize_final_step_inplace(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        znx_normalize_final_step_inplace_ref(base2k, lsh, x, carry);
    }
}

impl ZnxNormalizeFirstStep for NTTIfmaRef {
    #[inline(always)]
    fn znx_normalize_first_step<const OVERWRITE: bool>(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        znx_normalize_first_step_ref::<OVERWRITE>(base2k, lsh, x, a, carry);
    }
}

impl ZnxNormalizeFirstStepCarryOnly for NTTIfmaRef {
    #[inline(always)]
    fn znx_normalize_first_step_carry_only(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
        znx_normalize_first_step_carry_only_ref(base2k, lsh, x, carry);
    }
}

impl ZnxNormalizeFirstStepInplace for NTTIfmaRef {
    #[inline(always)]
    fn znx_normalize_first_step_inplace(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        znx_normalize_first_step_inplace_ref(base2k, lsh, x, carry);
    }
}

impl ZnxNormalizeMiddleStep for NTTIfmaRef {
    #[inline(always)]
    fn znx_normalize_middle_step<const OVERWRITE: bool>(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        znx_normalize_middle_step_ref::<OVERWRITE>(base2k, lsh, x, a, carry);
    }
}

impl ZnxNormalizeMiddleStepSub for NTTIfmaRef {
    #[inline(always)]
    fn znx_normalize_middle_step_sub(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        znx_normalize_middle_step_sub_ref(base2k, lsh, x, a, carry);
    }
}

impl ZnxNormalizeMiddleStepCarryOnly for NTTIfmaRef {
    #[inline(always)]
    fn znx_normalize_middle_step_carry_only(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
        znx_normalize_middle_step_carry_only_ref(base2k, lsh, x, carry);
    }
}

impl ZnxNormalizeMiddleStepInplace for NTTIfmaRef {
    #[inline(always)]
    fn znx_normalize_middle_step_inplace(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        znx_normalize_middle_step_inplace_ref(base2k, lsh, x, carry);
    }
}

impl ZnxExtractDigitAddMul for NTTIfmaRef {
    #[inline(always)]
    fn znx_extract_digit_addmul(base2k: usize, lsh: usize, res: &mut [i64], src: &mut [i64]) {
        znx_extract_digit_addmul_ref(base2k, lsh, res, src);
    }
}

impl ZnxNormalizeDigit for NTTIfmaRef {
    #[inline(always)]
    fn znx_normalize_digit(base2k: usize, res: &mut [i64], src: &mut [i64]) {
        znx_normalize_digit_ref(base2k, res, src);
    }
}
