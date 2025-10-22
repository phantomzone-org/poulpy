use crate::layouts::{
    Backend, ScalarZnxToRef, SvpPPolOwned, SvpPPolToMut, SvpPPolToRef, VecZnxDftToMut, VecZnxDftToRef, VecZnxToRef,
};

/// Allocates as [crate::layouts::SvpPPol].
pub trait SvpPPolAlloc<B: Backend> {
    fn svp_ppol_alloc(&self, cols: usize) -> SvpPPolOwned<B>;
}

/// Returns the size in bytes to allocate a [crate::layouts::SvpPPol].
pub trait SvpPPolBytesOf {
    fn bytes_of_svp_ppol(&self, cols: usize) -> usize;
}

/// Consume a vector of bytes into a [crate::layouts::MatZnx].
/// User must ensure that bytes is memory aligned and that it length is equal to [SvpPPolAllocBytes].
pub trait SvpPPolFromBytes<B: Backend> {
    fn svp_ppol_from_bytes(&self, cols: usize, bytes: Vec<u8>) -> SvpPPolOwned<B>;
}

/// Prepare a [crate::layouts::ScalarZnx] into an [crate::layouts::SvpPPol].
pub trait SvpPrepare<B: Backend> {
    fn svp_prepare<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: SvpPPolToMut<B>,
        A: ScalarZnxToRef;
}

/// Apply a scalar-vector product between `a[a_col]` and `b[b_col]` and stores the result on `res[res_col]`.
pub trait SvpApplyDft<B: Backend> {
    fn svp_apply_dft<R, A, C>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: SvpPPolToRef<B>,
        C: VecZnxToRef;
}

/// Apply a scalar-vector product between `a[a_col]` and `b[b_col]` and stores the result on `res[res_col]`.
pub trait SvpApplyDftToDft<B: Backend> {
    fn svp_apply_dft_to_dft<R, A, C>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: SvpPPolToRef<B>,
        C: VecZnxDftToRef<B>;
}

/// Apply a scalar-vector product between `a[a_col]` and `b[b_col]` and adds the result on `res[res_col]`.
pub trait SvpApplyDftToDftAdd<B: Backend> {
    fn svp_apply_dft_to_dft_add<R, A, C>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: SvpPPolToRef<B>,
        C: VecZnxDftToRef<B>;
}

/// Apply a scalar-vector product between `res[res_col]` and `a[a_col]` and stores the result on `res[res_col]`.
pub trait SvpApplyDftToDftInplace<B: Backend> {
    fn svp_apply_dft_to_dft_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: SvpPPolToRef<B>;
}
