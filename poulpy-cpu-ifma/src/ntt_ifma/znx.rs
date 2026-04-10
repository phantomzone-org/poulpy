//! Single ring element (`Z[X]/(X^n+1)`) arithmetic for [`NTTIfma`](super::NTTIfma).
//!
//! Implements the `Znx*` traits from `poulpy_hal::reference::znx`. All implementations
//! delegate to the AVX512-accelerated functions in `crate::znx_ifma` (same kernels used
//! by `FFT64Ifma`). These operate on plain `&[i64]` slices and are backend-independent.

use poulpy_hal::reference::znx::{
    ZnxAdd, ZnxAddInplace, ZnxAutomorphism, ZnxCopy, ZnxExtractDigitAddMul, ZnxMulAddPowerOfTwo, ZnxMulPowerOfTwo,
    ZnxMulPowerOfTwoInplace, ZnxNegate, ZnxNegateInplace, ZnxNormalizeDigit, ZnxNormalizeFinalStep, ZnxNormalizeFinalStepInplace,
    ZnxNormalizeFirstStep, ZnxNormalizeFirstStepCarryOnly, ZnxNormalizeFirstStepInplace, ZnxNormalizeMiddleStep,
    ZnxNormalizeMiddleStepCarryOnly, ZnxNormalizeMiddleStepInplace, ZnxRotate, ZnxSub, ZnxSubInplace, ZnxSubNegateInplace,
    ZnxSwitchRing, ZnxZero, znx_copy_ref, znx_rotate, znx_zero_ref,
};

use crate::znx_ifma::{
    znx_add_ifma, znx_add_inplace_ifma, znx_automorphism_ifma, znx_extract_digit_addmul_ifma, znx_mul_add_power_of_two_ifma,
    znx_mul_power_of_two_ifma, znx_mul_power_of_two_inplace_ifma, znx_negate_ifma, znx_negate_inplace_ifma,
    znx_normalize_digit_ifma, znx_normalize_final_step_ifma, znx_normalize_final_step_inplace_ifma,
    znx_normalize_first_step_carry_only_ifma, znx_normalize_first_step_ifma, znx_normalize_first_step_inplace_ifma,
    znx_normalize_middle_step_carry_only_ifma, znx_normalize_middle_step_ifma, znx_normalize_middle_step_inplace_ifma,
    znx_sub_ifma, znx_sub_inplace_ifma, znx_sub_negate_inplace_ifma, znx_switch_ring_ifma,
};

use super::NTTIfma;

impl ZnxAdd for NTTIfma {
    #[inline(always)]
    fn znx_add(res: &mut [i64], a: &[i64], b: &[i64]) {
        unsafe { znx_add_ifma(res, a, b) }
    }
}

impl ZnxAddInplace for NTTIfma {
    #[inline(always)]
    fn znx_add_inplace(res: &mut [i64], a: &[i64]) {
        unsafe { znx_add_inplace_ifma(res, a) }
    }
}

impl ZnxSub for NTTIfma {
    #[inline(always)]
    fn znx_sub(res: &mut [i64], a: &[i64], b: &[i64]) {
        unsafe { znx_sub_ifma(res, a, b) }
    }
}

impl ZnxSubInplace for NTTIfma {
    #[inline(always)]
    fn znx_sub_inplace(res: &mut [i64], a: &[i64]) {
        unsafe { znx_sub_inplace_ifma(res, a) }
    }
}

impl ZnxSubNegateInplace for NTTIfma {
    #[inline(always)]
    fn znx_sub_negate_inplace(res: &mut [i64], a: &[i64]) {
        unsafe { znx_sub_negate_inplace_ifma(res, a) }
    }
}

impl ZnxMulAddPowerOfTwo for NTTIfma {
    #[inline(always)]
    fn znx_muladd_power_of_two(k: i64, res: &mut [i64], a: &[i64]) {
        unsafe { znx_mul_add_power_of_two_ifma(k, res, a) }
    }
}

impl ZnxMulPowerOfTwo for NTTIfma {
    #[inline(always)]
    fn znx_mul_power_of_two(k: i64, res: &mut [i64], a: &[i64]) {
        unsafe { znx_mul_power_of_two_ifma(k, res, a) }
    }
}

impl ZnxMulPowerOfTwoInplace for NTTIfma {
    #[inline(always)]
    fn znx_mul_power_of_two_inplace(k: i64, res: &mut [i64]) {
        unsafe { znx_mul_power_of_two_inplace_ifma(k, res) }
    }
}

impl ZnxAutomorphism for NTTIfma {
    #[inline(always)]
    fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]) {
        unsafe { znx_automorphism_ifma(p, res, a) }
    }
}

impl ZnxCopy for NTTIfma {
    #[inline(always)]
    fn znx_copy(res: &mut [i64], a: &[i64]) {
        znx_copy_ref(res, a);
    }
}

impl ZnxNegate for NTTIfma {
    #[inline(always)]
    fn znx_negate(res: &mut [i64], src: &[i64]) {
        unsafe { znx_negate_ifma(res, src) }
    }
}

impl ZnxNegateInplace for NTTIfma {
    #[inline(always)]
    fn znx_negate_inplace(res: &mut [i64]) {
        unsafe { znx_negate_inplace_ifma(res) }
    }
}

impl ZnxRotate for NTTIfma {
    #[inline(always)]
    fn znx_rotate(p: i64, res: &mut [i64], src: &[i64]) {
        znx_rotate::<Self>(p, res, src);
    }
}

impl ZnxZero for NTTIfma {
    #[inline(always)]
    fn znx_zero(res: &mut [i64]) {
        znx_zero_ref(res);
    }
}

impl ZnxSwitchRing for NTTIfma {
    #[inline(always)]
    fn znx_switch_ring(res: &mut [i64], a: &[i64]) {
        unsafe { znx_switch_ring_ifma(res, a) }
    }
}

impl ZnxNormalizeFinalStep for NTTIfma {
    #[inline(always)]
    fn znx_normalize_final_step(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        unsafe { znx_normalize_final_step_ifma(base2k, lsh, x, a, carry) }
    }
}

impl ZnxNormalizeFinalStepInplace for NTTIfma {
    #[inline(always)]
    fn znx_normalize_final_step_inplace(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        unsafe { znx_normalize_final_step_inplace_ifma(base2k, lsh, x, carry) }
    }
}

impl ZnxNormalizeFirstStep for NTTIfma {
    #[inline(always)]
    fn znx_normalize_first_step(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        unsafe { znx_normalize_first_step_ifma(base2k, lsh, x, a, carry) }
    }
}

impl ZnxNormalizeFirstStepCarryOnly for NTTIfma {
    #[inline(always)]
    fn znx_normalize_first_step_carry_only(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
        unsafe { znx_normalize_first_step_carry_only_ifma(base2k, lsh, x, carry) }
    }
}

impl ZnxNormalizeFirstStepInplace for NTTIfma {
    #[inline(always)]
    fn znx_normalize_first_step_inplace(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        unsafe { znx_normalize_first_step_inplace_ifma(base2k, lsh, x, carry) }
    }
}

impl ZnxNormalizeMiddleStep for NTTIfma {
    #[inline(always)]
    fn znx_normalize_middle_step(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        unsafe { znx_normalize_middle_step_ifma(base2k, lsh, x, a, carry) }
    }
}

impl ZnxNormalizeMiddleStepCarryOnly for NTTIfma {
    #[inline(always)]
    fn znx_normalize_middle_step_carry_only(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
        unsafe { znx_normalize_middle_step_carry_only_ifma(base2k, lsh, x, carry) }
    }
}

impl ZnxNormalizeMiddleStepInplace for NTTIfma {
    #[inline(always)]
    fn znx_normalize_middle_step_inplace(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        unsafe { znx_normalize_middle_step_inplace_ifma(base2k, lsh, x, carry) }
    }
}

impl ZnxExtractDigitAddMul for NTTIfma {
    #[inline(always)]
    fn znx_extract_digit_addmul(base2k: usize, lsh: usize, res: &mut [i64], src: &mut [i64]) {
        unsafe { znx_extract_digit_addmul_ifma(base2k, lsh, res, src) }
    }
}

impl ZnxNormalizeDigit for NTTIfma {
    #[inline(always)]
    fn znx_normalize_digit(base2k: usize, res: &mut [i64], src: &mut [i64]) {
        unsafe { znx_normalize_digit_ifma(base2k, res, src) }
    }
}
