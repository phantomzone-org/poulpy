use crate::layouts::{
    Backend, Data, Scratch, VecZnxBig, VecZnxBigToMut, VecZnxDft, VecZnxDftOwned, VecZnxDftToMut, VecZnxDftToRef, VecZnxToRef,
};

pub trait VecZnxDftAlloc<B: Backend> {
    fn vec_znx_dft_alloc(&self, cols: usize, size: usize) -> VecZnxDftOwned<B>;
}

pub trait VecZnxDftFromBytes<B: Backend> {
    fn vec_znx_dft_from_bytes(&self, cols: usize, size: usize, bytes: Vec<u8>) -> VecZnxDftOwned<B>;
}

pub trait VecZnxDftBytesOf {
    fn bytes_of_vec_znx_dft(&self, cols: usize, size: usize) -> usize;
}

pub trait VecZnxDftApply<B: Backend> {
    fn vec_znx_dft_apply<R, A>(&self, step: usize, offset: usize, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxToRef;
}

pub trait VecZnxIdftApplyTmpBytes {
    fn vec_znx_idft_apply_tmp_bytes(&self) -> usize;
}

pub trait VecZnxIdftApply<B: Backend> {
    fn vec_znx_idft_apply<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, scratch: &mut Scratch<B>)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxDftToRef<B>;
}

pub trait VecZnxIdftApplyTmpA<B: Backend> {
    fn vec_znx_idft_apply_tmpa<R, A>(&self, res: &mut R, res_col: usize, a: &mut A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxDftToMut<B>;
}

pub trait VecZnxIdftApplyConsume<B: Backend> {
    fn vec_znx_idft_apply_consume<D: Data>(&self, a: VecZnxDft<D, B>) -> VecZnxBig<D, B>
    where
        VecZnxDft<D, B>: VecZnxDftToMut<B>;
}

pub trait VecZnxDftAdd<B: Backend> {
    fn vec_znx_dft_add<R, A, D>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &D, b_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
        D: VecZnxDftToRef<B>;
}

pub trait VecZnxDftAddInplace<B: Backend> {
    fn vec_znx_dft_add_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>;
}

pub trait VecZnxDftAddScaledInplace<B: Backend> {
    fn vec_znx_dft_add_scaled_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, a_scale: i64)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>;
}

pub trait VecZnxDftSub<B: Backend> {
    fn vec_znx_dft_sub<R, A, D>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &D, b_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
        D: VecZnxDftToRef<B>;
}

pub trait VecZnxDftSubInplace<B: Backend> {
    fn vec_znx_dft_sub_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>;
}

pub trait VecZnxDftSubNegateInplace<B: Backend> {
    fn vec_znx_dft_sub_negate_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>;
}

pub trait VecZnxDftCopy<B: Backend> {
    fn vec_znx_dft_copy<R, A>(&self, step: usize, offset: usize, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>;
}

pub trait VecZnxDftZero<B: Backend> {
    fn vec_znx_dft_zero<R>(&self, res: &mut R)
    where
        R: VecZnxDftToMut<B>;
}
