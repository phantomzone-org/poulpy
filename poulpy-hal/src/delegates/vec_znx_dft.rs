use crate::{
    api::{
        VecZnxDftAdd, VecZnxDftAddInplace, VecZnxDftAlloc, VecZnxDftAllocBytes, VecZnxDftApply, VecZnxDftCopy,
        VecZnxDftFromBytes, VecZnxDftSub, VecZnxDftSubABInplace, VecZnxDftSubBAInplace, VecZnxDftZero, VecZnxIdftApply,
        VecZnxIdftApplyConsume, VecZnxIdftApplyTmpA, VecZnxIdftApplyTmpBytes,
    },
    layouts::{
        Backend, Data, Module, Scratch, VecZnxBig, VecZnxBigToMut, VecZnxDft, VecZnxDftOwned, VecZnxDftToMut, VecZnxDftToRef,
        VecZnxToRef,
    },
    oep::{
        DFTImpl, VecZnxDftAddImpl, VecZnxDftAddInplaceImpl, VecZnxDftAllocBytesImpl, VecZnxDftAllocImpl, VecZnxDftCopyImpl,
        VecZnxDftFromBytesImpl, VecZnxDftSubABInplaceImpl, VecZnxDftSubBAInplaceImpl, VecZnxDftSubImpl, VecZnxDftZeroImpl,
        VecZnxIdftApplyConsumeImpl, VecZnxIdftApplyImpl, VecZnxIdftApplyTmpAImpl, VecZnxIdftApplyTmpBytesImpl,
    },
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

impl<B> VecZnxIdftApplyTmpBytes for Module<B>
where
    B: Backend + VecZnxIdftApplyTmpBytesImpl<B>,
{
    fn vec_znx_idft_apply_tmp_bytes(&self) -> usize {
        B::vec_znx_idft_apply_tmp_bytes_impl(self)
    }
}

impl<B> VecZnxIdftApply<B> for Module<B>
where
    B: Backend + VecZnxIdftApplyImpl<B>,
{
    fn vec_znx_idft_apply<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, scratch: &mut Scratch<B>)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxDftToRef<B>,
    {
        B::vec_znx_idft_apply_impl(self, res, res_col, a, a_col, scratch);
    }
}

impl<B> VecZnxIdftApplyTmpA<B> for Module<B>
where
    B: Backend + VecZnxIdftApplyTmpAImpl<B>,
{
    fn vec_znx_idft_apply_tmpa<R, A>(&self, res: &mut R, res_col: usize, a: &mut A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxDftToMut<B>,
    {
        B::vec_znx_idft_apply_tmpa_impl(self, res, res_col, a, a_col);
    }
}

impl<B> VecZnxIdftApplyConsume<B> for Module<B>
where
    B: Backend + VecZnxIdftApplyConsumeImpl<B>,
{
    fn vec_znx_idft_apply_consume<D: Data>(&self, a: VecZnxDft<D, B>) -> VecZnxBig<D, B>
    where
        VecZnxDft<D, B>: VecZnxDftToMut<B>,
    {
        B::vec_znx_idft_apply_consume_impl(self, a)
    }
}

impl<B> VecZnxDftApply<B> for Module<B>
where
    B: Backend + DFTImpl<B>,
{
    fn vec_znx_dft_apply<R, A>(&self, step: usize, offset: usize, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxToRef,
    {
        B::dft_impl(self, step, offset, res, res_col, a, a_col);
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
