use std::ptr::NonNull;

use poulpy_hal::{
    layouts::{Backend, Module},
    oep::ModuleNewImpl,
    reference::znx::{
        ZnxCopy, ZnxNormalizeFinalStep, ZnxNormalizeFinalStepInplace, ZnxNormalizeFirstStep, ZnxNormalizeFirstStepCarryOnly,
        ZnxNormalizeFirstStepInplace, ZnxNormalizeMiddleStep, ZnxNormalizeMiddleStepCarryOnly, ZnxNormalizeMiddleStepInplace,
        ZnxRotate, ZnxSwitchRing, ZnxZero, znx_copy_ref, znx_normalize_final_step_inplace_ref, znx_normalize_final_step_ref,
        znx_normalize_first_step_carry_only_ref, znx_normalize_first_step_inplace_ref, znx_normalize_first_step_ref,
        znx_normalize_middle_step_carry_only_ref, znx_normalize_middle_step_inplace_ref, znx_normalize_middle_step_ref,
        znx_switch_ring_ref, znx_zero_ref,
    },
};

use crate::cpu_spqlios::{
    FFT64Spqlios,
    ffi::module::{MODULE, delete_module_info, new_module_info},
    znx::znx_rotate_i64,
};

impl Backend for FFT64Spqlios {
    type ScalarPrep = f64;
    type ScalarBig = i64;
    type Handle = MODULE;
    unsafe fn destroy(handle: NonNull<Self::Handle>) {
        unsafe { delete_module_info(handle.as_ptr()) }
    }

    fn layout_big_word_count() -> usize {
        1
    }

    fn layout_prep_word_count() -> usize {
        1
    }
}

unsafe impl ModuleNewImpl<Self> for FFT64Spqlios {
    fn new_impl(n: u64) -> Module<Self> {
        unsafe { Module::from_raw_parts(new_module_info(n, 0), n) }
    }
}

impl ZnxCopy for FFT64Spqlios {
    fn znx_copy(res: &mut [i64], a: &[i64]) {
        znx_copy_ref(res, a);
    }
}

impl ZnxZero for FFT64Spqlios {
    fn znx_zero(res: &mut [i64]) {
        znx_zero_ref(res);
    }
}

impl ZnxSwitchRing for FFT64Spqlios {
    fn znx_switch_ring(res: &mut [i64], a: &[i64]) {
        znx_switch_ring_ref(res, a);
    }
}

impl ZnxRotate for FFT64Spqlios {
    fn znx_rotate(p: i64, res: &mut [i64], src: &[i64]) {
        unsafe {
            znx_rotate_i64(res.len() as u64, p, res.as_mut_ptr(), src.as_ptr());
        }
    }
}

impl ZnxNormalizeFinalStep for FFT64Spqlios {
    #[inline(always)]
    fn znx_normalize_final_step(basek: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        znx_normalize_final_step_ref(basek, lsh, x, a, carry);
    }
}

impl ZnxNormalizeFinalStepInplace for FFT64Spqlios {
    #[inline(always)]
    fn znx_normalize_final_step_inplace(basek: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        znx_normalize_final_step_inplace_ref(basek, lsh, x, carry);
    }
}

impl ZnxNormalizeFirstStep for FFT64Spqlios {
    #[inline(always)]
    fn znx_normalize_first_step(basek: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        znx_normalize_first_step_ref(basek, lsh, x, a, carry);
    }
}

impl ZnxNormalizeFirstStepCarryOnly for FFT64Spqlios {
    #[inline(always)]
    fn znx_normalize_first_step_carry_only(basek: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
        znx_normalize_first_step_carry_only_ref(basek, lsh, x, carry);
    }
}

impl ZnxNormalizeFirstStepInplace for FFT64Spqlios {
    #[inline(always)]
    fn znx_normalize_first_step_inplace(basek: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        znx_normalize_first_step_inplace_ref(basek, lsh, x, carry);
    }
}

impl ZnxNormalizeMiddleStep for FFT64Spqlios {
    #[inline(always)]
    fn znx_normalize_middle_step(basek: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        znx_normalize_middle_step_ref(basek, lsh, x, a, carry);
    }
}

impl ZnxNormalizeMiddleStepCarryOnly for FFT64Spqlios {
    #[inline(always)]
    fn znx_normalize_middle_step_carry_only(basek: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
        znx_normalize_middle_step_carry_only_ref(basek, lsh, x, carry);
    }
}

impl ZnxNormalizeMiddleStepInplace for FFT64Spqlios {
    #[inline(always)]
    fn znx_normalize_middle_step_inplace(basek: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        znx_normalize_middle_step_inplace_ref(basek, lsh, x, carry);
    }
}
