use crate::{
    Backend, Module, Scratch, VecZnxBig, VecZnxBigToMut, VecZnxDft, VecZnxDftAdd, VecZnxDftAddImpl, VecZnxDftAddInplace,
    VecZnxDftAddInplaceImpl, VecZnxDftAlloc, VecZnxDftAllocBytes, VecZnxDftAllocBytesImpl, VecZnxDftAllocImpl, VecZnxDftCopy,
    VecZnxDftCopyImpl, VecZnxDftFromBytes, VecZnxDftFromBytesImpl, VecZnxDftFromVecZnx, VecZnxDftFromVecZnxImpl, VecZnxDftOwned,
    VecZnxDftSub, VecZnxDftSubABInplace, VecZnxDftSubABInplaceImpl, VecZnxDftSubBAInplace, VecZnxDftSubBAInplaceImpl,
    VecZnxDftSubImpl, VecZnxDftToMut, VecZnxDftToRef, VecZnxDftToVecZnxBig, VecZnxDftToVecZnxBigConsume,
    VecZnxDftToVecZnxBigConsumeImpl, VecZnxDftToVecZnxBigImpl, VecZnxDftToVecZnxBigTmpA, VecZnxDftToVecZnxBigTmpAImpl,
    VecZnxDftToVecZnxBigTmpBytes, VecZnxDftToVecZnxBigTmpBytesImpl, VecZnxDftZero, VecZnxDftZeroImpl, VecZnxToRef,
};

impl<B: Backend> VecZnxDftFromBytes<B> for Module<B>
where
    (): VecZnxDftFromBytesImpl<B>,
{
    fn vec_znx_dft_from_bytes(&self, cols: usize, size: usize, bytes: Vec<u8>) -> VecZnxDftOwned<B> {
        <() as VecZnxDftFromBytesImpl<B>>::vec_znx_dft_from_bytes_impl(self, cols, size, bytes)
    }
}

impl<B: Backend> VecZnxDftAllocBytes for Module<B>
where
    (): VecZnxDftAllocBytesImpl<B>,
{
    fn vec_znx_dft_alloc_bytes(&self, cols: usize, size: usize) -> usize {
        <() as VecZnxDftAllocBytesImpl<B>>::vec_znx_dft_alloc_bytes_impl(self, cols, size)
    }
}

impl<B: Backend> VecZnxDftAlloc<B> for Module<B>
where
    (): VecZnxDftAllocImpl<B>,
{
    fn vec_znx_dft_alloc(&self, cols: usize, size: usize) -> VecZnxDftOwned<B> {
        <() as VecZnxDftAllocImpl<B>>::vec_znx_dft_alloc_impl(self, cols, size)
    }
}

impl<B: Backend> VecZnxDftToVecZnxBigTmpBytes for Module<B>
where
    (): VecZnxDftToVecZnxBigTmpBytesImpl<B>,
{
    fn vec_znx_dft_to_vec_znx_big_tmp_bytes(&self) -> usize {
        <() as VecZnxDftToVecZnxBigTmpBytesImpl<B>>::vec_znx_dft_to_vec_znx_big_tmp_bytes_impl(self)
    }
}

impl<B: Backend> VecZnxDftToVecZnxBig<B> for Module<B>
where
    (): VecZnxDftToVecZnxBigImpl<B>,
{
    fn vec_znx_dft_to_vec_znx_big<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, scratch: &mut Scratch)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxDftToRef<B>,
    {
        <() as VecZnxDftToVecZnxBigImpl<B>>::vec_znx_dft_to_vec_znx_big_impl(self, res, res_col, a, a_col, scratch);
    }
}

impl<B: Backend> VecZnxDftToVecZnxBigTmpA<B> for Module<B>
where
    (): VecZnxDftToVecZnxBigTmpAImpl<B>,
{
    fn vec_znx_dft_to_vec_znx_big_tmp_a<R, A>(&self, res: &mut R, res_col: usize, a: &mut A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxDftToMut<B>,
    {
        <() as VecZnxDftToVecZnxBigTmpAImpl<B>>::vec_znx_dft_to_vec_znx_big_tmp_a_impl(self, res, res_col, a, a_col);
    }
}

impl<B: Backend> VecZnxDftToVecZnxBigConsume<B> for Module<B>
where
    (): VecZnxDftToVecZnxBigConsumeImpl<B>,
{
    fn vec_znx_dft_to_vec_znx_big_consume<D>(&self, a: VecZnxDft<D, B>) -> VecZnxBig<D, B>
    where
        VecZnxDft<D, B>: VecZnxDftToMut<B>,
    {
        <() as VecZnxDftToVecZnxBigConsumeImpl<B>>::vec_znx_dft_to_vec_znx_big_consume_impl(self, a)
    }
}

impl<B: Backend> VecZnxDftFromVecZnx<B> for Module<B>
where
    (): VecZnxDftFromVecZnxImpl<B>,
{
    fn vec_znx_dft_from_vec_znx<R, A>(&self, step: usize, offset: usize, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxToRef,
    {
        <() as VecZnxDftFromVecZnxImpl<B>>::vec_znx_dft_from_vec_znx_impl(self, step, offset, res, res_col, a, a_col);
    }
}

impl<B: Backend> VecZnxDftAdd<B> for Module<B>
where
    (): VecZnxDftAddImpl<B>,
{
    fn vec_znx_dft_add<R, A, D>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &D, b_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
        D: VecZnxDftToRef<B>,
    {
        <() as VecZnxDftAddImpl<B>>::vec_znx_dft_add_impl(self, res, res_col, a, a_col, b, b_col);
    }
}

impl<B: Backend> VecZnxDftAddInplace<B> for Module<B>
where
    (): VecZnxDftAddInplaceImpl<B>,
{
    fn vec_znx_dft_add_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
    {
        <() as VecZnxDftAddInplaceImpl<B>>::vec_znx_dft_add_inplace_impl(self, res, res_col, a, a_col);
    }
}

impl<B: Backend> VecZnxDftSub<B> for Module<B>
where
    (): VecZnxDftSubImpl<B>,
{
    fn vec_znx_dft_sub<R, A, D>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &D, b_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
        D: VecZnxDftToRef<B>,
    {
        <() as VecZnxDftSubImpl<B>>::vec_znx_dft_sub_impl(self, res, res_col, a, a_col, b, b_col);
    }
}

impl<B: Backend> VecZnxDftSubABInplace<B> for Module<B>
where
    (): VecZnxDftSubABInplaceImpl<B>,
{
    fn vec_znx_dft_sub_ab_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
    {
        <() as VecZnxDftSubABInplaceImpl<B>>::vec_znx_dft_sub_ab_inplace_impl(self, res, res_col, a, a_col);
    }
}

impl<B: Backend> VecZnxDftSubBAInplace<B> for Module<B>
where
    (): VecZnxDftSubBAInplaceImpl<B>,
{
    fn vec_znx_dft_sub_ba_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
    {
        <() as VecZnxDftSubBAInplaceImpl<B>>::vec_znx_dft_sub_ba_inplace_impl(self, res, res_col, a, a_col);
    }
}

impl<B: Backend> VecZnxDftCopy<B> for Module<B>
where
    (): VecZnxDftCopyImpl<B>,
{
    fn vec_znx_dft_copy<R, A>(&self, step: usize, offset: usize, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
    {
        <() as VecZnxDftCopyImpl<B>>::vec_znx_dft_copy_impl(self, step, offset, res, res_col, a, a_col);
    }
}

impl<B: Backend> VecZnxDftZero<B> for Module<B>
where
    (): VecZnxDftZeroImpl<B>,
{
    fn vec_znx_dft_zero<R>(&self, res: &mut R)
    where
        R: VecZnxDftToMut<B>,
    {
        <() as VecZnxDftZeroImpl<B>>::vec_znx_dft_zero_impl(self, res);
    }
}
