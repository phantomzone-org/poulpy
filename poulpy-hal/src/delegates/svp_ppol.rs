use crate::{
    api::{SvpApply, SvpApplyInplace, SvpPPolAlloc, SvpPPolAllocBytes, SvpPPolFromBytes, SvpPrepare},
    layouts::{Backend, Module, ScalarZnxToRef, SvpPPolOwned, SvpPPolToMut, SvpPPolToRef, VecZnxDftToMut, VecZnxDftToRef},
    oep::{SvpApplyImpl, SvpApplyInplaceImpl, SvpPPolAllocBytesImpl, SvpPPolAllocImpl, SvpPPolFromBytesImpl, SvpPrepareImpl},
};

impl<B> SvpPPolFromBytes<B> for Module<B>
where
    B: Backend + SvpPPolFromBytesImpl<B>,
{
    fn svp_ppol_from_bytes(&self, cols: usize, bytes: Vec<u8>) -> SvpPPolOwned<B> {
        B::svp_ppol_from_bytes_impl(self.n(), cols, bytes)
    }
}

impl<B> SvpPPolAlloc<B> for Module<B>
where
    B: Backend + SvpPPolAllocImpl<B>,
{
    fn svp_ppol_alloc(&self, cols: usize) -> SvpPPolOwned<B> {
        B::svp_ppol_alloc_impl(self.n(), cols)
    }
}

impl<B> SvpPPolAllocBytes for Module<B>
where
    B: Backend + SvpPPolAllocBytesImpl<B>,
{
    fn svp_ppol_alloc_bytes(&self, cols: usize) -> usize {
        B::svp_ppol_alloc_bytes_impl(self.n(), cols)
    }
}

impl<B> SvpPrepare<B> for Module<B>
where
    B: Backend + SvpPrepareImpl<B>,
{
    fn svp_prepare<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: SvpPPolToMut<B>,
        A: ScalarZnxToRef,
    {
        B::svp_prepare_impl(self, res, res_col, a, a_col);
    }
}

impl<B> SvpApply<B> for Module<B>
where
    B: Backend + SvpApplyImpl<B>,
{
    fn svp_apply<R, A, C>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: SvpPPolToRef<B>,
        C: VecZnxDftToRef<B>,
    {
        B::svp_apply_impl(self, res, res_col, a, a_col, b, b_col);
    }
}

impl<B> SvpApplyInplace<B> for Module<B>
where
    B: Backend + SvpApplyInplaceImpl,
{
    fn svp_apply_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: SvpPPolToRef<B>,
    {
        B::svp_apply_inplace_impl(self, res, res_col, a, a_col);
    }
}
