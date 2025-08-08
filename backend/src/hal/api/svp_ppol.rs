use crate::hal::layouts::{Backend, ScalarZnxToRef, SvpPPolOwned, SvpPPolToMut, SvpPPolToRef, VecZnxDftToMut, VecZnxDftToRef};

/// Allocates as [crate::hal::layouts::SvpPPol].
pub trait SvpPPolAlloc<B: Backend> {
    fn svp_ppol_alloc(&self, cols: usize) -> SvpPPolOwned<B>;
}

/// Returns the size in bytes to allocate a [crate::hal::layouts::SvpPPol].
pub trait SvpPPolAllocBytes {
    fn svp_ppol_alloc_bytes(&self, cols: usize) -> usize;
}

/// Consume a vector of bytes into a [crate::hal::layouts::MatZnx].
/// User must ensure that bytes is memory aligned and that it length is equal to [SvpPPolAllocBytes].
pub trait SvpPPolFromBytes<B: Backend> {
    fn svp_ppol_from_bytes(&self, cols: usize, bytes: Vec<u8>) -> SvpPPolOwned<B>;
}

/// Prepare a [crate::hal::layouts::ScalarZnx] into an [crate::hal::layouts::SvpPPol].
pub trait SvpPrepare<B: Backend> {
    fn svp_prepare<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: SvpPPolToMut<B>,
        A: ScalarZnxToRef;
}

/// Apply a scalar-vector product between `a[a_col]` and `b[b_col]` and stores the result on `res[res_col]`.
pub trait SvpApply<B: Backend> {
    fn svp_apply<R, A, C>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: SvpPPolToRef<B>,
        C: VecZnxDftToRef<B>;
}

/// Apply a scalar-vector product between `res[res_col]` and `a[a_col]` and stores the result on `res[res_col]`.
pub trait SvpApplyInplace<B: Backend> {
    fn svp_apply_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: SvpPPolToRef<B>;
}
