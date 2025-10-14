use crate::{
    api::{
        SvpApplyDft, SvpApplyDftToDft, SvpApplyDftToDftAdd, SvpApplyDftToDftInplace, SvpPPolAlloc, SvpPPolBytesOf,
        SvpPPolFromBytes, SvpPrepare,
    },
    layouts::{
        Backend, Module, ScalarZnxToRef, SvpPPolOwned, SvpPPolToMut, SvpPPolToRef, VecZnxDftToMut, VecZnxDftToRef, VecZnxToRef,
    },
    oep::{
        SvpApplyDftImpl, SvpApplyDftToDftAddImpl, SvpApplyDftToDftImpl, SvpApplyDftToDftInplaceImpl, SvpPPolAllocBytesImpl,
        SvpPPolAllocImpl, SvpPPolFromBytesImpl, SvpPrepareImpl,
    },
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

impl<B> SvpPPolBytesOf for Module<B>
where
    B: Backend + SvpPPolAllocBytesImpl<B>,
{
    fn bytes_of_svp_ppol(&self, cols: usize) -> usize {
        B::svp_ppol_bytes_of_impl(self.n(), cols)
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

impl<B> SvpApplyDft<B> for Module<B>
where
    B: Backend + SvpApplyDftImpl<B>,
{
    fn svp_apply_dft<R, A, C>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: SvpPPolToRef<B>,
        C: VecZnxToRef,
    {
        B::svp_apply_dft_impl(self, res, res_col, a, a_col, b, b_col);
    }
}

impl<B> SvpApplyDftToDft<B> for Module<B>
where
    B: Backend + SvpApplyDftToDftImpl<B>,
{
    fn svp_apply_dft_to_dft<R, A, C>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: SvpPPolToRef<B>,
        C: VecZnxDftToRef<B>,
    {
        B::svp_apply_dft_to_dft_impl(self, res, res_col, a, a_col, b, b_col);
    }
}

impl<B> SvpApplyDftToDftAdd<B> for Module<B>
where
    B: Backend + SvpApplyDftToDftAddImpl<B>,
{
    fn svp_apply_dft_to_dft_add<R, A, C>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: SvpPPolToRef<B>,
        C: VecZnxDftToRef<B>,
    {
        B::svp_apply_dft_to_dft_add_impl(self, res, res_col, a, a_col, b, b_col);
    }
}

impl<B> SvpApplyDftToDftInplace<B> for Module<B>
where
    B: Backend + SvpApplyDftToDftInplaceImpl,
{
    fn svp_apply_dft_to_dft_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: SvpPPolToRef<B>,
    {
        B::svp_apply_dft_to_dft_inplace_impl(self, res, res_col, a, a_col);
    }
}
