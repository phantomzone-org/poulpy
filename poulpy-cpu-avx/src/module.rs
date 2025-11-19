use std::ptr::NonNull;

use poulpy_hal::{
    layouts::{Backend, Module},
    oep::ModuleNewImpl,
    reference::{
        fft64::{
            reim::{
                ReimAdd, ReimAddInplace, ReimAddMul, ReimCopy, ReimDFTExecute, ReimFFTTable, ReimFromZnx, ReimIFFTTable, ReimMul,
                ReimMulInplace, ReimNegate, ReimNegateInplace, ReimSub, ReimSubInplace, ReimSubNegateInplace, ReimToZnx,
                ReimToZnxInplace, ReimZero, reim_copy_ref, reim_zero_ref,
            },
            reim4::{
                Reim4Extract1Blk, Reim4Mat1ColProd, Reim4Mat2Cols2ndColProd, Reim4Mat2ColsProd, Reim4Save1Blk, Reim4Save2Blks,
            },
        },
        znx::{
            ZnxAdd, ZnxAddInplace, ZnxAutomorphism, ZnxCopy, ZnxExtractDigitAddMul, ZnxMulAddPowerOfTwo, ZnxMulPowerOfTwo,
            ZnxMulPowerOfTwoInplace, ZnxNegate, ZnxNegateInplace, ZnxNormalizeDigit, ZnxNormalizeFinalStep,
            ZnxNormalizeFinalStepInplace, ZnxNormalizeFirstStep, ZnxNormalizeFirstStepCarryOnly, ZnxNormalizeFirstStepInplace,
            ZnxNormalizeMiddleStep, ZnxNormalizeMiddleStepCarryOnly, ZnxNormalizeMiddleStepInplace, ZnxRotate, ZnxSub,
            ZnxSubInplace, ZnxSubNegateInplace, ZnxSwitchRing, ZnxZero, znx_copy_ref, znx_rotate, znx_zero_ref,
        },
    },
};

use crate::{
    FFT64Avx,
    reim::{
        ReimFFTAvx, ReimIFFTAvx, reim_add_avx2_fma, reim_add_inplace_avx2_fma, reim_addmul_avx2_fma, reim_from_znx_i64_bnd50_fma,
        reim_mul_avx2_fma, reim_mul_inplace_avx2_fma, reim_negate_avx2_fma, reim_negate_inplace_avx2_fma, reim_sub_avx2_fma,
        reim_sub_inplace_avx2_fma, reim_sub_negate_inplace_avx2_fma, reim_to_znx_i64_inplace_bnd63_avx2_fma,
    },
    reim_to_znx_i64_bnd63_avx2_fma,
    reim4::{
        reim4_extract_1blk_from_reim_avx, reim4_save_1blk_to_reim_avx, reim4_save_2blk_to_reim_avx,
        reim4_vec_mat1col_product_avx, reim4_vec_mat2cols_2ndcol_product_avx, reim4_vec_mat2cols_product_avx,
    },
    znx_avx::{
        znx_add_avx, znx_add_inplace_avx, znx_automorphism_avx, znx_extract_digit_addmul_avx, znx_mul_add_power_of_two_avx,
        znx_mul_power_of_two_avx, znx_mul_power_of_two_inplace_avx, znx_negate_avx, znx_negate_inplace_avx,
        znx_normalize_digit_avx, znx_normalize_final_step_avx, znx_normalize_final_step_inplace_avx,
        znx_normalize_first_step_avx, znx_normalize_first_step_carry_only_avx, znx_normalize_first_step_inplace_avx,
        znx_normalize_middle_step_avx, znx_normalize_middle_step_carry_only_avx, znx_normalize_middle_step_inplace_avx,
        znx_sub_avx, znx_sub_inplace_avx, znx_sub_negate_inplace_avx, znx_switch_ring_avx,
    },
};

#[repr(C)]
pub struct FFT64AvxHandle {
    table_fft: ReimFFTTable<f64>,
    table_ifft: ReimIFFTTable<f64>,
}

impl Backend for FFT64Avx {
    type ScalarPrep = f64;
    type ScalarBig = i64;
    type Handle = FFT64AvxHandle;
    unsafe fn destroy(handle: NonNull<Self::Handle>) {
        unsafe {
            drop(Box::from_raw(handle.as_ptr()));
        }
    }

    fn layout_big_word_count() -> usize {
        1
    }

    fn layout_prep_word_count() -> usize {
        1
    }
}

unsafe impl ModuleNewImpl<Self> for FFT64Avx {
    fn new_impl(n: u64) -> Module<Self> {
        if !std::arch::is_x86_feature_detected!("avx")
            || !std::arch::is_x86_feature_detected!("avx2")
            || !std::arch::is_x86_feature_detected!("fma")
        {
            panic!("arch must support avx2, avx and fma")
        }

        let handle: FFT64AvxHandle = FFT64AvxHandle {
            table_fft: ReimFFTTable::new(n as usize >> 1),
            table_ifft: ReimIFFTTable::new(n as usize >> 1),
        };
        // Leak Box to get a stable NonNull pointer
        let ptr: NonNull<FFT64AvxHandle> = NonNull::from(Box::leak(Box::new(handle)));
        unsafe { Module::from_nonnull(ptr, n) }
    }
}

pub trait FFT64ModuleHandle {
    fn get_fft_table(&self) -> &ReimFFTTable<f64>;
    fn get_ifft_table(&self) -> &ReimIFFTTable<f64>;
}

impl FFT64ModuleHandle for Module<FFT64Avx> {
    fn get_fft_table(&self) -> &ReimFFTTable<f64> {
        let h: &FFT64AvxHandle = unsafe { &*self.ptr() };
        &h.table_fft
    }
    fn get_ifft_table(&self) -> &ReimIFFTTable<f64> {
        let h: &FFT64AvxHandle = unsafe { &*self.ptr() };
        &h.table_ifft
    }
}

impl ZnxAdd for FFT64Avx {
    #[inline(always)]
    fn znx_add(res: &mut [i64], a: &[i64], b: &[i64]) {
        unsafe {
            znx_add_avx(res, a, b);
        }
    }
}

impl ZnxAddInplace for FFT64Avx {
    #[inline(always)]
    fn znx_add_inplace(res: &mut [i64], a: &[i64]) {
        unsafe {
            znx_add_inplace_avx(res, a);
        }
    }
}

impl ZnxSub for FFT64Avx {
    #[inline(always)]
    fn znx_sub(res: &mut [i64], a: &[i64], b: &[i64]) {
        unsafe {
            znx_sub_avx(res, a, b);
        }
    }
}

impl ZnxSubInplace for FFT64Avx {
    #[inline(always)]
    fn znx_sub_inplace(res: &mut [i64], a: &[i64]) {
        unsafe {
            znx_sub_inplace_avx(res, a);
        }
    }
}

impl ZnxSubNegateInplace for FFT64Avx {
    #[inline(always)]
    fn znx_sub_negate_inplace(res: &mut [i64], a: &[i64]) {
        unsafe {
            znx_sub_negate_inplace_avx(res, a);
        }
    }
}

impl ZnxAutomorphism for FFT64Avx {
    #[inline(always)]
    fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]) {
        unsafe {
            znx_automorphism_avx(p, res, a);
        }
    }
}

impl ZnxCopy for FFT64Avx {
    #[inline(always)]
    fn znx_copy(res: &mut [i64], a: &[i64]) {
        znx_copy_ref(res, a);
    }
}

impl ZnxNegate for FFT64Avx {
    #[inline(always)]
    fn znx_negate(res: &mut [i64], src: &[i64]) {
        unsafe {
            znx_negate_avx(res, src);
        }
    }
}

impl ZnxNegateInplace for FFT64Avx {
    #[inline(always)]
    fn znx_negate_inplace(res: &mut [i64]) {
        unsafe {
            znx_negate_inplace_avx(res);
        }
    }
}

impl ZnxMulAddPowerOfTwo for FFT64Avx {
    #[inline(always)]
    fn znx_muladd_power_of_two(k: i64, res: &mut [i64], a: &[i64]) {
        unsafe {
            znx_mul_add_power_of_two_avx(k, res, a);
        }
    }
}

impl ZnxMulPowerOfTwo for FFT64Avx {
    #[inline(always)]
    fn znx_mul_power_of_two(k: i64, res: &mut [i64], a: &[i64]) {
        unsafe {
            znx_mul_power_of_two_avx(k, res, a);
        }
    }
}

impl ZnxMulPowerOfTwoInplace for FFT64Avx {
    #[inline(always)]
    fn znx_mul_power_of_two_inplace(k: i64, res: &mut [i64]) {
        unsafe {
            znx_mul_power_of_two_inplace_avx(k, res);
        }
    }
}

impl ZnxRotate for FFT64Avx {
    #[inline(always)]
    fn znx_rotate(p: i64, res: &mut [i64], src: &[i64]) {
        znx_rotate::<Self>(p, res, src);
    }
}

impl ZnxZero for FFT64Avx {
    #[inline(always)]
    fn znx_zero(res: &mut [i64]) {
        znx_zero_ref(res);
    }
}

impl ZnxSwitchRing for FFT64Avx {
    #[inline(always)]
    fn znx_switch_ring(res: &mut [i64], a: &[i64]) {
        unsafe {
            znx_switch_ring_avx(res, a);
        }
    }
}

impl ZnxNormalizeFinalStep for FFT64Avx {
    #[inline(always)]
    fn znx_normalize_final_step(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        unsafe {
            znx_normalize_final_step_avx(base2k, lsh, x, a, carry);
        }
    }
}

impl ZnxNormalizeFinalStepInplace for FFT64Avx {
    #[inline(always)]
    fn znx_normalize_final_step_inplace(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        unsafe {
            znx_normalize_final_step_inplace_avx(base2k, lsh, x, carry);
        }
    }
}

impl ZnxNormalizeFirstStep for FFT64Avx {
    #[inline(always)]
    fn znx_normalize_first_step(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        unsafe {
            znx_normalize_first_step_avx(base2k, lsh, x, a, carry);
        }
    }
}

impl ZnxNormalizeFirstStepCarryOnly for FFT64Avx {
    #[inline(always)]
    fn znx_normalize_first_step_carry_only(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
        unsafe {
            znx_normalize_first_step_carry_only_avx(base2k, lsh, x, carry);
        }
    }
}

impl ZnxNormalizeFirstStepInplace for FFT64Avx {
    #[inline(always)]
    fn znx_normalize_first_step_inplace(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        unsafe {
            znx_normalize_first_step_inplace_avx(base2k, lsh, x, carry);
        }
    }
}

impl ZnxNormalizeMiddleStep for FFT64Avx {
    #[inline(always)]
    fn znx_normalize_middle_step(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        unsafe {
            znx_normalize_middle_step_avx(base2k, lsh, x, a, carry);
        }
    }
}

impl ZnxNormalizeMiddleStepCarryOnly for FFT64Avx {
    #[inline(always)]
    fn znx_normalize_middle_step_carry_only(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
        unsafe {
            znx_normalize_middle_step_carry_only_avx(base2k, lsh, x, carry);
        }
    }
}

impl ZnxNormalizeMiddleStepInplace for FFT64Avx {
    #[inline(always)]
    fn znx_normalize_middle_step_inplace(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        unsafe {
            znx_normalize_middle_step_inplace_avx(base2k, lsh, x, carry);
        }
    }
}

impl ZnxExtractDigitAddMul for FFT64Avx {
    #[inline(always)]
    fn znx_extract_digit_addmul(base2k: usize, lsh: usize, res: &mut [i64], src: &mut [i64]) {
        unsafe {
            znx_extract_digit_addmul_avx(base2k, lsh, res, src);
        }
    }
}

impl ZnxNormalizeDigit for FFT64Avx {
    #[inline(always)]
    fn znx_normalize_digit(base2k: usize, res: &mut [i64], src: &mut [i64]) {
        unsafe {
            znx_normalize_digit_avx(base2k, res, src);
        }
    }
}

impl ReimDFTExecute<ReimFFTTable<f64>, f64> for FFT64Avx {
    #[inline(always)]
    fn reim_dft_execute(table: &ReimFFTTable<f64>, data: &mut [f64]) {
        ReimFFTAvx::reim_dft_execute(table, data);
    }
}

impl ReimDFTExecute<ReimIFFTTable<f64>, f64> for FFT64Avx {
    #[inline(always)]
    fn reim_dft_execute(table: &ReimIFFTTable<f64>, data: &mut [f64]) {
        ReimIFFTAvx::reim_dft_execute(table, data);
    }
}

impl ReimFromZnx for FFT64Avx {
    #[inline(always)]
    fn reim_from_znx(res: &mut [f64], a: &[i64]) {
        unsafe {
            reim_from_znx_i64_bnd50_fma(res, a);
        }
    }
}

impl ReimToZnx for FFT64Avx {
    #[inline(always)]
    fn reim_to_znx(res: &mut [i64], divisor: f64, a: &[f64]) {
        unsafe {
            reim_to_znx_i64_bnd63_avx2_fma(res, divisor, a);
        }
    }
}

impl ReimToZnxInplace for FFT64Avx {
    #[inline(always)]
    fn reim_to_znx_inplace(res: &mut [f64], divisor: f64) {
        unsafe {
            reim_to_znx_i64_inplace_bnd63_avx2_fma(res, divisor);
        }
    }
}

impl ReimAdd for FFT64Avx {
    #[inline(always)]
    fn reim_add(res: &mut [f64], a: &[f64], b: &[f64]) {
        unsafe {
            reim_add_avx2_fma(res, a, b);
        }
    }
}

impl ReimAddInplace for FFT64Avx {
    #[inline(always)]
    fn reim_add_inplace(res: &mut [f64], a: &[f64]) {
        unsafe {
            reim_add_inplace_avx2_fma(res, a);
        }
    }
}

impl ReimSub for FFT64Avx {
    #[inline(always)]
    fn reim_sub(res: &mut [f64], a: &[f64], b: &[f64]) {
        unsafe {
            reim_sub_avx2_fma(res, a, b);
        }
    }
}

impl ReimSubInplace for FFT64Avx {
    #[inline(always)]
    fn reim_sub_inplace(res: &mut [f64], a: &[f64]) {
        unsafe {
            reim_sub_inplace_avx2_fma(res, a);
        }
    }
}

impl ReimSubNegateInplace for FFT64Avx {
    #[inline(always)]
    fn reim_sub_negate_inplace(res: &mut [f64], a: &[f64]) {
        unsafe {
            reim_sub_negate_inplace_avx2_fma(res, a);
        }
    }
}

impl ReimNegate for FFT64Avx {
    #[inline(always)]
    fn reim_negate(res: &mut [f64], a: &[f64]) {
        unsafe {
            reim_negate_avx2_fma(res, a);
        }
    }
}

impl ReimNegateInplace for FFT64Avx {
    #[inline(always)]
    fn reim_negate_inplace(res: &mut [f64]) {
        unsafe {
            reim_negate_inplace_avx2_fma(res);
        }
    }
}

impl ReimMul for FFT64Avx {
    #[inline(always)]
    fn reim_mul(res: &mut [f64], a: &[f64], b: &[f64]) {
        unsafe {
            reim_mul_avx2_fma(res, a, b);
        }
    }
}

impl ReimMulInplace for FFT64Avx {
    #[inline(always)]
    fn reim_mul_inplace(res: &mut [f64], a: &[f64]) {
        unsafe {
            reim_mul_inplace_avx2_fma(res, a);
        }
    }
}

impl ReimAddMul for FFT64Avx {
    #[inline(always)]
    fn reim_addmul(res: &mut [f64], a: &[f64], b: &[f64]) {
        unsafe {
            reim_addmul_avx2_fma(res, a, b);
        }
    }
}

impl ReimCopy for FFT64Avx {
    #[inline(always)]
    fn reim_copy(res: &mut [f64], a: &[f64]) {
        reim_copy_ref(res, a);
    }
}

impl ReimZero for FFT64Avx {
    #[inline(always)]
    fn reim_zero(res: &mut [f64]) {
        reim_zero_ref(res);
    }
}

impl Reim4Extract1Blk for FFT64Avx {
    #[inline(always)]
    fn reim4_extract_1blk(m: usize, rows: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
        unsafe {
            reim4_extract_1blk_from_reim_avx(m, rows, blk, dst, src);
        }
    }
}

impl Reim4Save1Blk for FFT64Avx {
    #[inline(always)]
    fn reim4_save_1blk<const OVERWRITE: bool>(m: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
        unsafe {
            reim4_save_1blk_to_reim_avx::<OVERWRITE>(m, blk, dst, src);
        }
    }
}

impl Reim4Save2Blks for FFT64Avx {
    #[inline(always)]
    fn reim4_save_2blks<const OVERWRITE: bool>(m: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
        unsafe {
            reim4_save_2blk_to_reim_avx::<OVERWRITE>(m, blk, dst, src);
        }
    }
}

impl Reim4Mat1ColProd for FFT64Avx {
    #[inline(always)]
    fn reim4_mat1col_prod(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
        unsafe {
            reim4_vec_mat1col_product_avx(nrows, dst, u, v);
        }
    }
}

impl Reim4Mat2ColsProd for FFT64Avx {
    #[inline(always)]
    fn reim4_mat2cols_prod(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
        unsafe {
            reim4_vec_mat2cols_product_avx(nrows, dst, u, v);
        }
    }
}

impl Reim4Mat2Cols2ndColProd for FFT64Avx {
    #[inline(always)]
    fn reim4_mat2cols_2ndcol_prod(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
        unsafe {
            reim4_vec_mat2cols_2ndcol_product_avx(nrows, dst, u, v);
        }
    }
}
