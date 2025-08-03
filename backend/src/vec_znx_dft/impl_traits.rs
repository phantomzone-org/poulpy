use crate::{
    Backend, Module, Scratch, VecZnxBig, VecZnxBigToMut, VecZnxDft, VecZnxDftOwned, VecZnxDftToMut, VecZnxDftToRef, VecZnxToRef,
};

pub unsafe trait VecZnxDftAllocImpl<B: Backend> {
    fn vec_znx_dft_alloc_impl(n: usize, cols: usize, size: usize) -> VecZnxDftOwned<B>;
}

pub unsafe trait VecZnxDftFromBytesImpl<B: Backend> {
    fn vec_znx_dft_from_bytes_impl(n: usize, cols: usize, size: usize, bytes: Vec<u8>) -> VecZnxDftOwned<B>;
}

pub unsafe trait VecZnxDftAllocBytesImpl<B: Backend> {
    fn vec_znx_dft_alloc_bytes_impl(n: usize, cols: usize, size: usize) -> usize;
}

pub unsafe trait VecZnxDftToVecZnxBigTmpBytesImpl<B: Backend> {
    fn vec_znx_dft_to_vec_znx_big_tmp_bytes_impl(module: &Module<B>) -> usize;
}

pub unsafe trait VecZnxDftToVecZnxBigImpl<B: Backend> {
    fn vec_znx_dft_to_vec_znx_big_impl<R, A>(
        module: &Module<B>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxBigToMut<B>,
        A: VecZnxDftToRef<B>;
}

pub unsafe trait VecZnxDftToVecZnxBigTmpAImpl<B: Backend> {
    fn vec_znx_dft_to_vec_znx_big_tmp_a_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &mut A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxDftToMut<B>;
}

pub unsafe trait VecZnxDftToVecZnxBigConsumeImpl<B: Backend> {
    fn vec_znx_dft_to_vec_znx_big_consume_impl<D>(module: &Module<B>, a: VecZnxDft<D, B>) -> VecZnxBig<D, B>
    where
        VecZnxDft<D, B>: VecZnxDftToMut<B>;
}

pub unsafe trait VecZnxDftAddImpl<B: Backend> {
    fn vec_znx_dft_add_impl<R, A, D>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &D, b_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
        D: VecZnxDftToRef<B>;
}

pub unsafe trait VecZnxDftAddInplaceImpl<B: Backend> {
    fn vec_znx_dft_add_inplace_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>;
}

pub unsafe trait VecZnxDftSubImpl<B: Backend> {
    fn vec_znx_dft_sub_impl<R, A, D>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &D, b_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
        D: VecZnxDftToRef<B>;
}

pub unsafe trait VecZnxDftSubABInplaceImpl<B: Backend> {
    fn vec_znx_dft_sub_ab_inplace_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>;
}

pub unsafe trait VecZnxDftSubBAInplaceImpl<B: Backend> {
    fn vec_znx_dft_sub_ba_inplace_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>;
}

pub unsafe trait VecZnxDftCopyImpl<B: Backend> {
    fn vec_znx_dft_copy_impl<R, A>(
        module: &Module<B>,
        step: usize,
        offset: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
    ) where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>;
}

pub unsafe trait VecZnxDftFromVecZnxImpl<B: Backend> {
    fn vec_znx_dft_from_vec_znx_impl<R, A>(
        module: &Module<B>,
        step: usize,
        offset: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
    ) where
        R: VecZnxDftToMut<B>,
        A: VecZnxToRef;
}

pub unsafe trait VecZnxDftZeroImpl<B: Backend> {
    fn vec_znx_dft_zero_impl<R>(module: &Module<B>, res: &mut R)
    where
        R: VecZnxDftToMut<B>;
}
