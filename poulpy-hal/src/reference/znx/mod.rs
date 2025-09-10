pub(crate) mod add;
pub(crate) mod arithmetic_avx;
pub(crate) mod arithmetic_ref;
pub(crate) mod automorphism;
pub(crate) mod copy;
pub(crate) mod neg;
pub(crate) mod normalize_avx;
pub(crate) mod normalize_ref;
pub(crate) mod rotate;
pub(crate) mod sampling;
pub(crate) mod sub;
pub(crate) mod switch_ring;
pub(crate) mod zero;

pub use arithmetic_avx::*;
pub use arithmetic_ref::*;
pub use normalize_avx::*;
pub use normalize_ref::*;
pub use sampling::*;

pub trait ZnxArithmetic {
    fn znx_add(res: &mut [i64], a: &[i64], b: &[i64]);
    fn znx_sub(res: &mut [i64], a: &[i64], b: &[i64]);
    fn znx_sub_ab_inplace(res: &mut [i64], a: &[i64]);
    fn znx_sub_ba_inplace(res: &mut [i64], a: &[i64]);
    fn znx_add_inplace(res: &mut [i64], a: &[i64]);
    fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]);
    fn znx_copy(res: &mut [i64], a: &[i64]);
    fn znx_negate(res: &mut [i64], src: &[i64]);
    fn znx_negate_inplace(res: &mut [i64]);
    fn znx_rotate(p: i64, res: &mut [i64], src: &[i64]);
    fn znx_zero(res: &mut [i64]);
    fn znx_switch_ring(res: &mut [i64], a: &[i64]);
}

pub trait ZnxNormalize {
    fn znx_normalize_first_step_carry_only(basek: usize, lsh: usize, x: &[i64], carry: &mut [i64]);
    fn znx_normalize_first_step_inplace(basek: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]);
    fn znx_normalize_first_step(basek: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]);
    fn znx_normalize_middle_step_carry_only(basek: usize, lsh: usize, x: &[i64], carry: &mut [i64]);
    fn znx_normalize_middle_step_inplace(basek: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]);
    fn znx_normalize_middle_step(basek: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]);
    fn znx_normalize_final_step_inplace(basek: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]);
    fn znx_normalize_final_step(basek: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]);
}
