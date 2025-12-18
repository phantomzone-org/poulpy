mod arithmetic_ref;

pub use arithmetic_ref::*;

use crate::{layouts::Backend, reference::fft64::reim::as_arr_mut};

pub trait Reim4Extract1BlkContiguous {
    fn reim4_extract_1blk_contiguous(m: usize, rows: usize, blk: usize, dst: &mut [f64], src: &[f64]);
}

pub trait Reim4Save1BlkContiguous {
    fn reim4_save_1blk_contiguous(m: usize, rows: usize, blk: usize, dst: &mut [f64], src: &[f64]);
}

pub trait Reim4Save1Blk {
    fn reim4_save_1blk<const OVERWRITE: bool>(m: usize, blk: usize, dst: &mut [f64], src: &[f64]);
}

pub trait Reim4Save2Blks {
    fn reim4_save_2blks<const OVERWRITE: bool>(m: usize, blk: usize, dst: &mut [f64], src: &[f64]);
}

pub trait Reim4Mat1ColProd {
    fn reim4_mat1col_prod(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]);
}

pub trait Reim4Mat2ColsProd {
    fn reim4_mat2cols_prod(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]);
}

pub trait Reim4Mat2Cols2ndColProd {
    fn reim4_mat2cols_2ndcol_prod(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]);
}

pub trait Reim4Convolution1Coeff {
    fn reim4_convolution_1coeff(k: usize, dst: &mut [f64; 8], a: &[f64], a_size: usize, b: &[f64], b_size: usize);
}

pub trait Reim4Convolution2Coeffs {
    fn reim4_convolution_2coeffs(k: usize, dst: &mut [f64; 16], a: &[f64], a_size: usize, b: &[f64], b_size: usize);
}

pub trait Reim4ConvolutionByRealConst1Coeff {
    fn reim4_convolution_by_real_const_1coeff(k: usize, dst: &mut [f64; 8], a: &[f64], a_size: usize, b: &[f64]);
}

pub trait Reim4ConvolutionByRealConst2Coeffs {
    fn reim4_convolution_by_real_const_2coeffs(k: usize, dst: &mut [f64; 16], a: &[f64], a_size: usize, b: &[f64]);
}

impl<BE: Backend> Reim4Convolution<BE> for BE where Self: Reim4Convolution1Coeff + Reim4Convolution2Coeffs {}

pub trait Reim4Convolution<BE: Backend>
where
    BE: Reim4Convolution1Coeff + Reim4Convolution2Coeffs,
{
    fn reim4_convolution(dst: &mut [f64], dst_size: usize, offset: usize, a: &[f64], a_size: usize, b: &[f64], b_size: usize) {
        assert!(a_size > 0);
        assert!(b_size > 0);

        for k in (0..dst_size - 1).step_by(2) {
            BE::reim4_convolution_2coeffs(k + offset, as_arr_mut(&mut dst[8 * k..]), a, a_size, b, b_size);
        }

        if !dst_size.is_multiple_of(2) {
            let k: usize = dst_size - 1;
            BE::reim4_convolution_1coeff(k + offset, as_arr_mut(&mut dst[8 * k..]), a, a_size, b, b_size);
        }
    }
}

impl<BE: Backend> Reim4ConvolutionByRealConst<BE> for BE where
    Self: Reim4ConvolutionByRealConst1Coeff + Reim4ConvolutionByRealConst2Coeffs
{
}

pub trait Reim4ConvolutionByRealConst<BE: Backend>
where
    BE: Reim4ConvolutionByRealConst1Coeff + Reim4ConvolutionByRealConst2Coeffs,
{
    fn reim4_convolution_by_real_const(dst: &mut [f64], dst_size: usize, offset: usize, a: &[f64], a_size: usize, b: &[f64]) {
        assert!(a_size > 0);

        for k in (0..dst_size - 1).step_by(2) {
            BE::reim4_convolution_by_real_const_2coeffs(k + offset, as_arr_mut(&mut dst[8 * k..]), a, a_size, b);
        }

        if !dst_size.is_multiple_of(2) {
            let k: usize = dst_size - 1;
            BE::reim4_convolution_by_real_const_1coeff(k + offset, as_arr_mut(&mut dst[8 * k..]), a, a_size, b);
        }
    }
}
