mod arithmetic_avx;
mod arithmetic_ref;

pub use arithmetic_avx::*;
pub use arithmetic_ref::*;

pub trait Reim4Blk {
    fn reim4_extract_1blk_from_reim(m: usize, rows: usize, blk: usize, dst: &mut [f64], src: &[f64]);
    fn reim4_save_1blk_to_reim(m: usize, blk: usize, dst: &mut [f64], src: &[f64]);
    fn reim4_save_2blk_to_reim(m: usize, blk: usize, dst: &mut [f64], src: &[f64]);
    fn reim4_vec_mat1col_product(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]);
    fn reim4_vec_mat2cols_product(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]);
}
