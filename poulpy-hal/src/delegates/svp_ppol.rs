use crate::{
    api::{SvpApplyDft, SvpApplyDftToDft, SvpApplyDftToDftInplace, SvpPPolAlloc, SvpPPolBytesOf, SvpPrepare},
    layouts::{
        Backend, Module, ScalarZnxToRef, SvpPPolOwned, SvpPPolToMut, SvpPPolToRef, VecZnxDftToMut, VecZnxDftToRef,
        VecZnxToRef,
    },
    oep::{SvpApplyDftImpl, SvpApplyDftToDftImpl, SvpApplyDftToDftInplaceImpl, SvpPrepareImpl},
};

impl<B: Backend> SvpPPolAlloc<B> for Module<B> {
    fn svp_ppol_alloc(&self, cols: usize) -> SvpPPolOwned<B> {
        SvpPPolOwned::alloc(self.n(), cols)
    }
}

impl<B: Backend> SvpPPolBytesOf for Module<B> {
    fn bytes_of_svp_ppol(&self, cols: usize) -> usize {
        B::bytes_of_svp_ppol(self.n(), cols)
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
