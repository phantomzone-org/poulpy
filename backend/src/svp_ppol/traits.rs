use crate::{Backend, ScalarZnxToRef, SvpPPolOwned, SvpPPolToMut, SvpPPolToRef, VecZnxDftToMut, VecZnxDftToRef};

pub trait SvpPPolFromBytes<B: Backend> {
    fn svp_ppol_from_bytes(&self, cols: usize, bytes: Vec<u8>) -> SvpPPolOwned<B>;
}

pub trait SvpPPolAlloc<B: Backend> {
    fn svp_ppol_alloc(&self, cols: usize) -> SvpPPolOwned<B>;
}

pub trait SvpPPolAllocBytes {
    fn svp_ppol_alloc_bytes(&self, cols: usize) -> usize;
}

pub trait SvpPrepare<B: Backend> {
    fn svp_prepare<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: SvpPPolToMut<B>,
        A: ScalarZnxToRef;
}

pub trait SvpApply<B: Backend> {
    fn svp_apply<R, A, C>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: SvpPPolToRef<B>,
        C: VecZnxDftToRef<B>;
}

pub trait SvpApplyInplace<B: Backend> {
    fn svp_apply_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: SvpPPolToRef<B>;
}
