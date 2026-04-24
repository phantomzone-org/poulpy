use crate::{
    api::{SvpApplyDft, SvpApplyDftToDft, SvpApplyDftToDftAssign, SvpPPolAlloc, SvpPPolBytesOf, SvpPrepare},
    layouts::{
        Backend, Module, ScalarZnxToRef, SvpPPolBackendMut, SvpPPolBackendRef, SvpPPolOwned, VecZnxDftToMut, VecZnxDftToRef,
        VecZnxToRef,
    },
    oep::HalSvpImpl,
};

macro_rules! impl_svp_delegate {
    ($trait:ty, $($body:item)+) => {
        impl<B> $trait for Module<B>
        where
            B: Backend + HalSvpImpl<B>,
        {
            $($body)+
        }
    };
}

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

impl_svp_delegate!(
    SvpPrepare<B>,
    fn svp_prepare<A>(&self, res: &mut SvpPPolBackendMut<'_, B>, res_col: usize, a: &A, a_col: usize)
    where
        A: ScalarZnxToRef,
    {
        <B as HalSvpImpl<B>>::svp_prepare(self, res, res_col, a, a_col);
    }
);

impl_svp_delegate!(
    SvpApplyDft<B>,
    fn svp_apply_dft<R, C>(&self, res: &mut R, res_col: usize, a: &SvpPPolBackendRef<'_, B>, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxDftToMut<B>,
        C: VecZnxToRef,
    {
        <B as HalSvpImpl<B>>::svp_apply_dft(self, res, res_col, a, a_col, b, b_col);
    }
);

impl_svp_delegate!(
    SvpApplyDftToDft<B>,
    fn svp_apply_dft_to_dft<R, C>(
        &self,
        res: &mut R,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, B>,
        a_col: usize,
        b: &C,
        b_col: usize,
    ) where
        R: VecZnxDftToMut<B>,
        C: VecZnxDftToRef<B>,
    {
        <B as HalSvpImpl<B>>::svp_apply_dft_to_dft(self, res, res_col, a, a_col, b, b_col);
    }
);

impl_svp_delegate!(
    SvpApplyDftToDftInplace<B>,
    fn svp_apply_dft_to_dft_inplace<R>(&self, res: &mut R, res_col: usize, a: &SvpPPolBackendRef<'_, B>, a_col: usize)
    where
        R: VecZnxDftToMut<B>,
    {
        <B as HalSvpImpl<B>>::svp_apply_dft_to_dft_inplace(self, res, res_col, a, a_col);
    }
);
