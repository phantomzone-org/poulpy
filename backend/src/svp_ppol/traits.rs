use crate::{Backend, ScalarZnxToRef, SvpPPolOwned, SvpPPolToMut, SvpPPolToRef, VecZnxDftToMut, VecZnxDftToRef};

pub trait SvpPPolyFromBytes<B: Backend> {
    fn svp_ppol_from_bytes(&self, cols: usize, bytes: Vec<u8>) -> SvpPPolOwned<B>;
}

pub trait SvpPPolAlloc<B: Backend> {
    fn svp_ppol_alloc(&self, cols: usize) -> SvpPPolOwned<B>;
}

pub trait SvpPPolAllocBytes {
    fn svp_ppol_alloc_bytes(&self, cols: usize) -> usize;
}

pub trait SvpPPolPrepare<BACKEND: Backend> {
    fn svp_prepare<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: SvpPPolToMut<BACKEND>,
        A: ScalarZnxToRef;
}

pub trait SvpPPolApply<BACKEND: Backend> {
    fn svp_apply<R, A, B>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
    where
        R: VecZnxDftToMut<BACKEND>,
        A: SvpPPolToRef<BACKEND>,
        B: VecZnxDftToRef<BACKEND>;
}

pub trait SvpPPolApplyInplace<BACKEND: Backend> {
    fn svp_apply_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<BACKEND>,
        A: SvpPPolToRef<BACKEND>;
}
