use crate::{Backend, Module, ScalarZnxToRef, SvpPPolOwned, SvpPPolToMut, SvpPPolToRef, VecZnxDftToMut, VecZnxDftToRef};

pub unsafe trait SvpPPolFromBytesImpl<B: Backend> {
    fn svp_ppol_from_bytes_impl(n: usize, cols: usize, bytes: Vec<u8>) -> SvpPPolOwned<B>;
}

pub unsafe trait SvpPPolAllocImpl<B: Backend> {
    fn svp_ppol_alloc_impl(n: usize, cols: usize) -> SvpPPolOwned<B>;
}

pub unsafe trait SvpPPolAllocBytesImpl<B: Backend> {
    fn svp_ppol_alloc_bytes_impl(n: usize, cols: usize) -> usize;
}

pub unsafe trait SvpPrepareImpl<B: Backend> {
    fn svp_prepare_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: SvpPPolToMut<B>,
        A: ScalarZnxToRef;
}

pub unsafe trait SvpApplyImpl<B: Backend> {
    fn svp_apply_impl<R, A, C>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: SvpPPolToRef<B>,
        C: VecZnxDftToRef<B>;
}

pub unsafe trait SvpApplyInplaceImpl: Backend {
    fn svp_apply_inplace_impl<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<Self>,
        A: SvpPPolToRef<Self>;
}
