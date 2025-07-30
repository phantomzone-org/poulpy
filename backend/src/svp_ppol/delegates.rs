use crate::{
    Backend, Module, ScalarZnxToRef, SvpApply, SvpApplyImpl, SvpApplyInplace, SvpApplyInplaceImpl, SvpPPolAlloc,
    SvpPPolAllocBytes, SvpPPolAllocBytesImpl, SvpPPolAllocImpl, SvpPPolFromBytes, SvpPPolFromBytesImpl, SvpPPolOwned,
    SvpPPolToMut, SvpPPolToRef, SvpPrepare, SvpPrepareImpl, VecZnxDftToMut, VecZnxDftToRef,
};

impl<B: Backend> SvpPPolFromBytes<B> for Module<B>
where
    (): SvpPPolFromBytesImpl<B>,
{
    fn svp_ppol_from_bytes(&self, cols: usize, bytes: Vec<u8>) -> SvpPPolOwned<B> {
        <() as SvpPPolFromBytesImpl<B>>::svp_ppol_from_bytes_impl(self, cols, bytes)
    }
}

impl<B: Backend> SvpPPolAlloc<B> for Module<B>
where
    (): SvpPPolAllocImpl<B>,
{
    fn svp_ppol_alloc(&self, cols: usize) -> SvpPPolOwned<B> {
        <() as SvpPPolAllocImpl<B>>::svp_ppol_alloc_impl(self, cols)
    }
}

impl<B: Backend> SvpPPolAllocBytes for Module<B>
where
    (): SvpPPolAllocBytesImpl<B>,
{
    fn svp_ppol_alloc_bytes(&self, cols: usize) -> usize {
        <() as SvpPPolAllocBytesImpl<B>>::svp_ppol_alloc_bytes_impl(self, cols)
    }
}

impl<B: Backend> SvpPrepare<B> for Module<B>
where
    (): SvpPrepareImpl<B>,
{
    fn svp_prepare<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: SvpPPolToMut<B>,
        A: ScalarZnxToRef,
    {
        <() as SvpPrepareImpl<B>>::svp_prepare_impl(self, res, res_col, a, a_col);
    }
}

impl<B: Backend> SvpApply<B> for Module<B>
where
    (): SvpApplyImpl<B>,
{
    fn svp_apply<R, A, C>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: SvpPPolToRef<B>,
        C: VecZnxDftToRef<B>,
    {
        <() as SvpApplyImpl<B>>::svp_apply_impl(self, res, res_col, a, a_col, b, b_col);
    }
}

impl<B: Backend> SvpApplyInplace<B> for Module<B>
where
    (): SvpApplyInplaceImpl<B>,
{
    fn svp_apply_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: SvpPPolToRef<B>,
    {
        <() as SvpApplyInplaceImpl<B>>::svp_apply_inplace_impl(self, res, res_col, a, a_col);
    }
}
