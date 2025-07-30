use crate::{Backend, Module, ScalarZnxToRef, SvpPPolOwned, SvpPPolToMut, SvpPPolToRef, VecZnxDftToMut, VecZnxDftToRef};

pub trait SvpPPolFromBytesImpl<B: Backend> {
    fn svp_ppol_from_bytes_impl(module: &Module<B>, cols: usize, bytes: Vec<u8>) -> SvpPPolOwned<B>;
}

pub trait SvpPPolAllocImpl<B: Backend> {
    fn svp_ppol_alloc_impl(module: &Module<B>, cols: usize) -> SvpPPolOwned<B>;
}

pub trait SvpPPolAllocBytesImpl<B: Backend> {
    fn svp_ppol_alloc_bytes_impl(module: &Module<B>, cols: usize) -> usize;
}

pub trait SvpPrepareImpl<B: Backend> {
    fn svp_prepare_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: SvpPPolToMut<B>,
        A: ScalarZnxToRef;
}

pub trait SvpApplyImpl<B: Backend> {
    fn svp_apply_impl<R, A, C>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: SvpPPolToRef<B>,
        C: VecZnxDftToRef<B>;
}

pub trait SvpApplyInplaceImpl<B: Backend> {
    fn svp_apply_inplace_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: SvpPPolToRef<B>;
}
