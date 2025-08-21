use crate::layouts::{
    Backend, Data, Scratch, VecZnxBig, VecZnxBigToMut, VecZnxDft, VecZnxDftOwned, VecZnxDftToMut, VecZnxDftToRef, VecZnxToRef,
};

pub trait VecZnxDftAlloc<B: Backend> {
    fn vec_znx_dft_alloc(&self, cols: usize, size: usize) -> VecZnxDftOwned<B>;
}

pub trait VecZnxDftFromBytes<B: Backend> {
    fn vec_znx_dft_from_bytes(&self, cols: usize, size: usize, bytes: Vec<u8>) -> VecZnxDftOwned<B>;
}

pub trait VecZnxDftAllocBytes {
    fn vec_znx_dft_alloc_bytes(&self, cols: usize, size: usize) -> usize;
}

pub trait VecZnxDftToVecZnxBigTmpBytes {
    fn vec_znx_dft_to_vec_znx_big_tmp_bytes(&self) -> usize;
}

pub trait VecZnxDftToVecZnxBig<B: Backend> {
    fn vec_znx_dft_to_vec_znx_big<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, scratch: &mut Scratch<B>)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxDftToRef<B>;
}

pub trait VecZnxDftToVecZnxBigTmpA<B: Backend> {
    fn vec_znx_dft_to_vec_znx_big_tmp_a<R, A>(&self, res: &mut R, res_col: usize, a: &mut A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxDftToMut<B>;
}

pub trait VecZnxDftToVecZnxBigConsume<B: Backend> {
    fn vec_znx_dft_to_vec_znx_big_consume<D: Data>(&self, a: VecZnxDft<D, B>) -> VecZnxBig<D, B>
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

pub trait VecZnxDftSub<B: Backend> {
    fn vec_znx_dft_sub<R, A, D>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &D, b_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
        D: VecZnxDftToRef<B>;
}

pub trait VecZnxDftSubABInplace<B: Backend> {
    fn vec_znx_dft_sub_ab_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>;
}

pub trait VecZnxDftSubBAInplace<B: Backend> {
    fn vec_znx_dft_sub_ba_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
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

pub trait VecZnxDftFromVecZnx<B: Backend> {
    fn vec_znx_dft_from_vec_znx<R, A>(&self, step: usize, offset: usize, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxToRef;
}

pub trait VecZnxDftZero<B: Backend> {
    fn vec_znx_dft_zero<R>(&self, res: &mut R)
    where
        R: VecZnxDftToMut<B>;
}
