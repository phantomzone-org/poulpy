mod arithmetic_ref;

pub use arithmetic_ref::*;

use crate::reference::fft64::reim::as_arr_mut;

pub trait Reim4BlkMatVec {
    fn reim4_extract_1blk_contiguous(m: usize, rows: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
        reim4_extract_1blk_from_reim_contiguous_ref(m, rows, blk, dst, src)
    }

    fn reim4_save_1blk_contiguous(m: usize, rows: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
        reim4_save_1blk_to_reim_contiguous_ref(m, rows, blk, dst, src)
    }

    fn reim4_save_1blk<const OVERWRITE: bool>(m: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
        reim4_save_1blk_to_reim_ref::<OVERWRITE>(m, blk, dst, src)
    }

    fn reim4_save_2blks<const OVERWRITE: bool>(m: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
        reim4_save_2blk_to_reim_ref::<OVERWRITE>(m, blk, dst, src)
    }

    fn reim4_mat1col_prod(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
        reim4_vec_mat1col_product_ref(nrows, dst, u, v)
    }

    fn reim4_mat2cols_prod(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
        reim4_vec_mat2cols_product_ref(nrows, dst, u, v)
    }

    fn reim4_mat2cols_2ndcol_prod(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
        reim4_vec_mat2cols_2ndcol_product_ref(nrows, dst, u, v)
    }
}

pub trait Reim4Convolution {
    fn reim4_convolution_1coeff(k: usize, dst: &mut [f64; 8], a: &[f64], a_size: usize, b: &[f64], b_size: usize) {
        reim4_convolution_1coeff_ref(k, dst, a, a_size, b, b_size)
    }

    fn reim4_convolution_2coeffs(k: usize, dst: &mut [f64; 16], a: &[f64], a_size: usize, b: &[f64], b_size: usize) {
        reim4_convolution_2coeffs_ref(k, dst, a, a_size, b, b_size)
    }

    fn reim4_convolution(dst: &mut [f64], dst_size: usize, offset: usize, a: &[f64], a_size: usize, b: &[f64], b_size: usize) {
        assert!(a_size > 0);
        assert!(b_size > 0);

        for k in (0..dst_size - 1).step_by(2) {
            Self::reim4_convolution_2coeffs(k + offset, as_arr_mut(&mut dst[8 * k..]), a, a_size, b, b_size);
        }

        if !dst_size.is_multiple_of(2) {
            let k: usize = dst_size - 1;
            Self::reim4_convolution_1coeff(k + offset, as_arr_mut(&mut dst[8 * k..]), a, a_size, b, b_size);
        }
    }

    fn reim4_convolution_by_real_const_1coeff(k: usize, dst: &mut [f64; 8], a: &[f64], a_size: usize, b: &[f64]) {
        reim4_convolution_by_real_const_1coeff_ref(k, dst, a, a_size, b)
    }

    fn reim4_convolution_by_real_const_2coeffs(k: usize, dst: &mut [f64; 16], a: &[f64], a_size: usize, b: &[f64]) {
        reim4_convolution_by_real_const_2coeffs_ref(k, dst, a, a_size, b)
    }

    fn reim4_convolution_by_real_const(dst: &mut [f64], dst_size: usize, offset: usize, a: &[f64], a_size: usize, b: &[f64]) {
        assert!(a_size > 0);

        for k in (0..dst_size - 1).step_by(2) {
            Self::reim4_convolution_by_real_const_2coeffs(k + offset, as_arr_mut(&mut dst[8 * k..]), a, a_size, b);
        }

        if !dst_size.is_multiple_of(2) {
            let k: usize = dst_size - 1;
            Self::reim4_convolution_by_real_const_1coeff(k + offset, as_arr_mut(&mut dst[8 * k..]), a, a_size, b);
        }
    }
}
