mod add;
mod arithmetic_ref;
mod automorphism;
mod copy;
mod mul;
mod neg;
mod normalization;
mod rotate;
mod sampling;
mod sub;
mod switch_ring;
mod zero;

pub use add::*;
pub use arithmetic_ref::*;
pub use automorphism::*;
pub use copy::*;
pub use mul::*;
pub use neg::*;
pub use normalization::*;
pub use rotate::*;
pub use sub::*;
pub use switch_ring::*;
pub use zero::*;

pub use sampling::*;

pub trait ZnxAdd {
    fn znx_add(res: &mut [i64], a: &[i64], b: &[i64]);
}

pub trait ZnxAddInplace {
    fn znx_add_inplace(res: &mut [i64], a: &[i64]);
}

pub trait ZnxSub {
    fn znx_sub(res: &mut [i64], a: &[i64], b: &[i64]);
}

pub trait ZnxSubABInplace {
    fn znx_sub_ab_inplace(res: &mut [i64], a: &[i64]);
}

pub trait ZnxSubBAInplace {
    fn znx_sub_ba_inplace(res: &mut [i64], a: &[i64]);
}

pub trait ZnxAutomorphism {
    fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]);
}

pub trait ZnxCopy {
    fn znx_copy(res: &mut [i64], a: &[i64]);
}

pub trait ZnxNegate {
    fn znx_negate(res: &mut [i64], src: &[i64]);
}

pub trait ZnxNegateInplace {
    fn znx_negate_inplace(res: &mut [i64]);
}

pub trait ZnxRotate {
    fn znx_rotate(p: i64, res: &mut [i64], src: &[i64]);
}

pub trait ZnxZero {
    fn znx_zero(res: &mut [i64]);
}

pub trait ZnxMulPowerOfTwo {
    fn znx_mul_power_of_two(k: i64, res: &mut [i64], a: &[i64]);
}

pub trait ZnxMulAddPowerOfTwo {
    fn znx_muladd_power_of_two(k: i64, res: &mut [i64], a: &[i64]);
}

pub trait ZnxMulPowerOfTwoInplace {
    fn znx_mul_power_of_two_inplace(k: i64, res: &mut [i64]);
}

pub trait ZnxSwitchRing {
    fn znx_switch_ring(res: &mut [i64], a: &[i64]);
}

pub trait ZnxNormalizeFirstStepCarryOnly {
    fn znx_normalize_first_step_carry_only(basek: usize, lsh: usize, x: &[i64], carry: &mut [i64]);
}

pub trait ZnxNormalizeFirstStepInplace {
    fn znx_normalize_first_step_inplace(basek: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]);
}

pub trait ZnxNormalizeFirstStep {
    fn znx_normalize_first_step(basek: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]);
}

pub trait ZnxNormalizeMiddleStepCarryOnly {
    fn znx_normalize_middle_step_carry_only(basek: usize, lsh: usize, x: &[i64], carry: &mut [i64]);
}

pub trait ZnxNormalizeMiddleStepInplace {
    fn znx_normalize_middle_step_inplace<const OVERWRITE: bool>(basek: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]);
}

pub trait ZnxNormalizeMiddleStep {
    fn znx_normalize_middle_step(basek: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]);
}

pub trait ZnxNormalizeFinalStepInplace {
    fn znx_normalize_final_step_inplace<const OVERWRITE: bool>(basek: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]);
}

pub trait ZnxNormalizeFinalStep {
    fn znx_normalize_final_step(basek: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]);
}
