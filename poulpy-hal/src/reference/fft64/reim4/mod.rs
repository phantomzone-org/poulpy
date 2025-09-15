mod arithmetic_ref;

pub use arithmetic_ref::*;

pub trait Reim4Extract1Blk {
    fn reim4_extract_1blk(m: usize, rows: usize, blk: usize, dst: &mut [f64], src: &[f64]);
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
