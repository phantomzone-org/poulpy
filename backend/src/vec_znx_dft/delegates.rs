use crate::{
    Backend, Module, Scratch, VecZnxBig, VecZnxBigToMut, VecZnxDft, VecZnxDftAdd, VecZnxDftAddImpl, VecZnxDftAddInplace,
    VecZnxDftAddInplaceImpl, VecZnxDftAlloc, VecZnxDftAllocBytes, VecZnxDftAllocBytesImpl, VecZnxDftAllocImpl, VecZnxDftCopy,
    VecZnxDftCopyImpl, VecZnxDftFromBytes, VecZnxDftFromBytesImpl, VecZnxDftFromVecZnx, VecZnxDftFromVecZnxImpl, VecZnxDftOwned,
    VecZnxDftSub, VecZnxDftSubABInplace, VecZnxDftSubABInplaceImpl, VecZnxDftSubBAInplace, VecZnxDftSubBAInplaceImpl,
    VecZnxDftSubImpl, VecZnxDftToMut, VecZnxDftToRef, VecZnxDftToVecZnxBig, VecZnxDftToVecZnxBigConsume,
    VecZnxDftToVecZnxBigConsumeImpl, VecZnxDftToVecZnxBigImpl, VecZnxDftToVecZnxBigTmpA, VecZnxDftToVecZnxBigTmpAImpl,
    VecZnxDftToVecZnxBigTmpBytes, VecZnxDftToVecZnxBigTmpBytesImpl, VecZnxDftZero, VecZnxDftZeroImpl, VecZnxToRef,
};

impl<B> VecZnxDftFromBytes<B> for Module<B>
where
    B: Backend + VecZnxDftFromBytesImpl<B>,
{
    fn vec_znx_dft_from_bytes(&self, cols: usize, size: usize, bytes: Vec<u8>) -> VecZnxDftOwned<B> {
        B::vec_znx_dft_from_bytes_impl(self.n(), cols, size, bytes)
    }
}

impl<B> VecZnxDftAllocBytes for Module<B>
where
    B: Backend + VecZnxDftAllocBytesImpl<B>,
{
    fn vec_znx_dft_alloc_bytes(&self, cols: usize, size: usize) -> usize {
        B::vec_znx_dft_alloc_bytes_impl(self.n(), cols, size)
    }
}

impl<B> VecZnxDftAlloc<B> for Module<B>
where
    B: Backend + VecZnxDftAllocImpl<B>,
{
    fn vec_znx_dft_alloc(&self, cols: usize, size: usize) -> VecZnxDftOwned<B> {
        B::vec_znx_dft_alloc_impl(self.n(), cols, size)
    }
}

impl<B> VecZnxDftToVecZnxBigTmpBytes for Module<B>
where
    B: Backend + VecZnxDftToVecZnxBigTmpBytesImpl<B>,
{
    fn vec_znx_dft_to_vec_znx_big_tmp_bytes(&self) -> usize {
        B::vec_znx_dft_to_vec_znx_big_tmp_bytes_impl(self)
    }
}

impl<B> VecZnxDftToVecZnxBig<B> for Module<B>
where
    B: Backend + VecZnxDftToVecZnxBigImpl<B>,
{
    fn vec_znx_dft_to_vec_znx_big<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, scratch: &mut Scratch<B>)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxDftToRef<B>,
    {
        B::vec_znx_dft_to_vec_znx_big_impl(self, res, res_col, a, a_col, scratch);
    }
}

impl<B> VecZnxDftToVecZnxBigTmpA<B> for Module<B>
where
    B: Backend + VecZnxDftToVecZnxBigTmpAImpl<B>,
{
    fn vec_znx_dft_to_vec_znx_big_tmp_a<R, A>(&self, res: &mut R, res_col: usize, a: &mut A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxDftToMut<B>,
    {
        B::vec_znx_dft_to_vec_znx_big_tmp_a_impl(self, res, res_col, a, a_col);
    }
}

impl<B> VecZnxDftToVecZnxBigConsume<B> for Module<B>
where
    B: Backend + VecZnxDftToVecZnxBigConsumeImpl<B>,
{
    fn vec_znx_dft_to_vec_znx_big_consume<D>(&self, a: VecZnxDft<D, B>) -> VecZnxBig<D, B>
    where
        VecZnxDft<D, B>: VecZnxDftToMut<B>,
    {
        B::vec_znx_dft_to_vec_znx_big_consume_impl(self, a)
    }
}

impl<B> VecZnxDftFromVecZnx<B> for Module<B>
where
    B: Backend + VecZnxDftFromVecZnxImpl<B>,
{
    fn vec_znx_dft_from_vec_znx<R, A>(&self, step: usize, offset: usize, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxToRef,
    {
        B::vec_znx_dft_from_vec_znx_impl(self, step, offset, res, res_col, a, a_col);
    }
}

impl<B> VecZnxDftAdd<B> for Module<B>
where
    B: Backend + VecZnxDftAddImpl<B>,
{
    fn vec_znx_dft_add<R, A, D>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &D, b_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
        D: VecZnxDftToRef<B>,
    {
        B::vec_znx_dft_add_impl(self, res, res_col, a, a_col, b, b_col);
    }
}

impl<B> VecZnxDftAddInplace<B> for Module<B>
where
    B: Backend + VecZnxDftAddInplaceImpl<B>,
{
    fn vec_znx_dft_add_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
    {
        B::vec_znx_dft_add_inplace_impl(self, res, res_col, a, a_col);
    }
}

impl<B> VecZnxDftSub<B> for Module<B>
where
    B: Backend + VecZnxDftSubImpl<B>,
{
    fn vec_znx_dft_sub<R, A, D>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &D, b_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
        D: VecZnxDftToRef<B>,
    {
        B::vec_znx_dft_sub_impl(self, res, res_col, a, a_col, b, b_col);
    }
}

impl<B> VecZnxDftSubABInplace<B> for Module<B>
where
    B: Backend + VecZnxDftSubABInplaceImpl<B>,
{
    fn vec_znx_dft_sub_ab_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
    {
        B::vec_znx_dft_sub_ab_inplace_impl(self, res, res_col, a, a_col);
    }
}

impl<B> VecZnxDftSubBAInplace<B> for Module<B>
where
    B: Backend + VecZnxDftSubBAInplaceImpl<B>,
{
    fn vec_znx_dft_sub_ba_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
    {
        B::vec_znx_dft_sub_ba_inplace_impl(self, res, res_col, a, a_col);
    }
}

impl<B> VecZnxDftCopy<B> for Module<B>
where
    B: Backend + VecZnxDftCopyImpl<B>,
{
    fn vec_znx_dft_copy<R, A>(&self, step: usize, offset: usize, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
    {
        B::vec_znx_dft_copy_impl(self, step, offset, res, res_col, a, a_col);
    }
}

impl<B> VecZnxDftZero<B> for Module<B>
where
    B: Backend + VecZnxDftZeroImpl<B>,
{
    fn vec_znx_dft_zero<R>(&self, res: &mut R)
    where
        R: VecZnxDftToMut<B>,
    {
        B::vec_znx_dft_zero_impl(self, res);
    }
}
