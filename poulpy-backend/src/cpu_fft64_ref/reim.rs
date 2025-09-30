use poulpy_hal::reference::fft64::{
    reim::{
        ReimAdd, ReimAddInplace, ReimAddMul, ReimCopy, ReimDFTExecute, ReimFFTTable, ReimFromZnx, ReimIFFTTable, ReimMul,
        ReimMulInplace, ReimNegate, ReimNegateInplace, ReimSub, ReimSubInplace, ReimSubNegateInplace, ReimToZnx,
        ReimToZnxInplace, ReimZero, fft_ref, ifft_ref, reim_add_inplace_ref, reim_add_ref, reim_addmul_ref, reim_copy_ref,
        reim_from_znx_i64_ref, reim_mul_inplace_ref, reim_mul_ref, reim_negate_inplace_ref, reim_negate_ref,
        reim_sub_inplace_ref, reim_sub_negate_inplace_ref, reim_sub_ref, reim_to_znx_i64_inplace_ref, reim_to_znx_i64_ref,
        reim_zero_ref,
    },
    reim4::{
        Reim4Extract1Blk, Reim4Mat1ColProd, Reim4Mat2Cols2ndColProd, Reim4Mat2ColsProd, Reim4Save1Blk, Reim4Save2Blks,
        reim4_extract_1blk_from_reim_ref, reim4_save_1blk_to_reim_ref, reim4_save_2blk_to_reim_ref,
        reim4_vec_mat1col_product_ref, reim4_vec_mat2cols_2ndcol_product_ref, reim4_vec_mat2cols_product_ref,
    },
};

use crate::FFT64Ref;

impl ReimDFTExecute<ReimFFTTable<f64>, f64> for FFT64Ref {
    fn reim_dft_execute(table: &ReimFFTTable<f64>, data: &mut [f64]) {
        fft_ref(table.m(), table.omg(), data);
    }
}

impl ReimDFTExecute<ReimIFFTTable<f64>, f64> for FFT64Ref {
    fn reim_dft_execute(table: &ReimIFFTTable<f64>, data: &mut [f64]) {
        ifft_ref(table.m(), table.omg(), data);
    }
}

impl ReimFromZnx for FFT64Ref {
    #[inline(always)]
    fn reim_from_znx(res: &mut [f64], a: &[i64]) {
        reim_from_znx_i64_ref(res, a);
    }
}

impl ReimToZnx for FFT64Ref {
    #[inline(always)]
    fn reim_to_znx(res: &mut [i64], divisor: f64, a: &[f64]) {
        reim_to_znx_i64_ref(res, divisor, a);
    }
}

impl ReimToZnxInplace for FFT64Ref {
    #[inline(always)]
    fn reim_to_znx_inplace(res: &mut [f64], divisor: f64) {
        reim_to_znx_i64_inplace_ref(res, divisor);
    }
}

impl ReimAdd for FFT64Ref {
    #[inline(always)]
    fn reim_add(res: &mut [f64], a: &[f64], b: &[f64]) {
        reim_add_ref(res, a, b);
    }
}

impl ReimAddInplace for FFT64Ref {
    #[inline(always)]
    fn reim_add_inplace(res: &mut [f64], a: &[f64]) {
        reim_add_inplace_ref(res, a);
    }
}

impl ReimSub for FFT64Ref {
    #[inline(always)]
    fn reim_sub(res: &mut [f64], a: &[f64], b: &[f64]) {
        reim_sub_ref(res, a, b);
    }
}

impl ReimSubInplace for FFT64Ref {
    #[inline(always)]
    fn reim_sub_inplace(res: &mut [f64], a: &[f64]) {
        reim_sub_inplace_ref(res, a);
    }
}

impl ReimSubNegateInplace for FFT64Ref {
    #[inline(always)]
    fn reim_sub_negate_inplace(res: &mut [f64], a: &[f64]) {
        reim_sub_negate_inplace_ref(res, a);
    }
}

impl ReimNegate for FFT64Ref {
    #[inline(always)]
    fn reim_negate(res: &mut [f64], a: &[f64]) {
        reim_negate_ref(res, a);
    }
}

impl ReimNegateInplace for FFT64Ref {
    #[inline(always)]
    fn reim_negate_inplace(res: &mut [f64]) {
        reim_negate_inplace_ref(res);
    }
}

impl ReimMul for FFT64Ref {
    #[inline(always)]
    fn reim_mul(res: &mut [f64], a: &[f64], b: &[f64]) {
        reim_mul_ref(res, a, b);
    }
}

impl ReimMulInplace for FFT64Ref {
    #[inline(always)]
    fn reim_mul_inplace(res: &mut [f64], a: &[f64]) {
        reim_mul_inplace_ref(res, a);
    }
}

impl ReimAddMul for FFT64Ref {
    #[inline(always)]
    fn reim_addmul(res: &mut [f64], a: &[f64], b: &[f64]) {
        reim_addmul_ref(res, a, b);
    }
}

impl ReimCopy for FFT64Ref {
    #[inline(always)]
    fn reim_copy(res: &mut [f64], a: &[f64]) {
        reim_copy_ref(res, a);
    }
}

impl ReimZero for FFT64Ref {
    #[inline(always)]
    fn reim_zero(res: &mut [f64]) {
        reim_zero_ref(res);
    }
}

impl Reim4Extract1Blk for FFT64Ref {
    #[inline(always)]
    fn reim4_extract_1blk(m: usize, rows: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
        reim4_extract_1blk_from_reim_ref(m, rows, blk, dst, src);
    }
}

impl Reim4Save1Blk for FFT64Ref {
    #[inline(always)]
    fn reim4_save_1blk<const OVERWRITE: bool>(m: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
        reim4_save_1blk_to_reim_ref::<OVERWRITE>(m, blk, dst, src);
    }
}

impl Reim4Save2Blks for FFT64Ref {
    #[inline(always)]
    fn reim4_save_2blks<const OVERWRITE: bool>(m: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
        reim4_save_2blk_to_reim_ref::<OVERWRITE>(m, blk, dst, src);
    }
}

impl Reim4Mat1ColProd for FFT64Ref {
    #[inline(always)]
    fn reim4_mat1col_prod(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
        reim4_vec_mat1col_product_ref(nrows, dst, u, v);
    }
}

impl Reim4Mat2ColsProd for FFT64Ref {
    #[inline(always)]
    fn reim4_mat2cols_prod(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
        reim4_vec_mat2cols_product_ref(nrows, dst, u, v);
    }
}

impl Reim4Mat2Cols2ndColProd for FFT64Ref {
    #[inline(always)]
    fn reim4_mat2cols_2ndcol_prod(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
        reim4_vec_mat2cols_2ndcol_product_ref(nrows, dst, u, v);
    }
}
