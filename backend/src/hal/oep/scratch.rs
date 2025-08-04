use crate::hal::layouts::{Backend, MatZnx, ScalarZnx, Scratch, ScratchOwned, SvpPPol, VecZnx, VecZnxBig, VecZnxDft, VmpPMat};

pub unsafe trait ScratchOwnedAllocImpl<B: Backend> {
    fn scratch_owned_alloc_impl(size: usize) -> ScratchOwned<B>;
}

pub unsafe trait ScratchOwnedBorrowImpl<B: Backend> {
    fn scratch_owned_borrow_impl(scratch: &mut ScratchOwned<B>) -> &mut Scratch<B>;
}

pub unsafe trait ScratchFromBytesImpl<B: Backend> {
    fn scratch_from_bytes_impl(data: &mut [u8]) -> &mut Scratch<B>;
}

pub unsafe trait ScratchAvailableImpl<B: Backend> {
    fn scratch_available_impl(scratch: &Scratch<B>) -> usize;
}

pub unsafe trait TakeSliceImpl<B: Backend> {
    fn take_slice_impl<T>(scratch: &mut Scratch<B>, len: usize) -> (&mut [T], &mut Scratch<B>);
}

pub unsafe trait TakeScalarZnxImpl<B: Backend> {
    fn take_scalar_znx_impl(scratch: &mut Scratch<B>, n: usize, cols: usize) -> (ScalarZnx<&mut [u8]>, &mut Scratch<B>);
}

pub unsafe trait TakeSvpPPolImpl<B: Backend> {
    fn take_svp_ppol_impl(scratch: &mut Scratch<B>, n: usize, cols: usize) -> (SvpPPol<&mut [u8], B>, &mut Scratch<B>);
}

pub unsafe trait TakeVecZnxImpl<B: Backend> {
    fn take_vec_znx_impl(scratch: &mut Scratch<B>, n: usize, cols: usize, size: usize) -> (VecZnx<&mut [u8]>, &mut Scratch<B>);
}

pub unsafe trait TakeVecZnxSliceImpl<B: Backend> {
    fn take_vec_znx_slice_impl(
        scratch: &mut Scratch<B>,
        len: usize,
        n: usize,
        cols: usize,
        size: usize,
    ) -> (Vec<VecZnx<&mut [u8]>>, &mut Scratch<B>);
}

pub unsafe trait TakeVecZnxBigImpl<B: Backend> {
    fn take_vec_znx_big_impl(
        scratch: &mut Scratch<B>,
        n: usize,
        cols: usize,
        size: usize,
    ) -> (VecZnxBig<&mut [u8], B>, &mut Scratch<B>);
}

pub unsafe trait TakeVecZnxDftImpl<B: Backend> {
    fn take_vec_znx_dft_impl(
        scratch: &mut Scratch<B>,
        n: usize,
        cols: usize,
        size: usize,
    ) -> (VecZnxDft<&mut [u8], B>, &mut Scratch<B>);
}

pub unsafe trait TakeVecZnxDftSliceImpl<B: Backend> {
    fn take_vec_znx_dft_slice_impl(
        scratch: &mut Scratch<B>,
        len: usize,
        n: usize,
        cols: usize,
        size: usize,
    ) -> (Vec<VecZnxDft<&mut [u8], B>>, &mut Scratch<B>);
}

pub unsafe trait TakeVmpPMatImpl<B: Backend> {
    fn take_vmp_pmat_impl(
        scratch: &mut Scratch<B>,
        n: usize,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
    ) -> (VmpPMat<&mut [u8], B>, &mut Scratch<B>);
}

pub unsafe trait TakeMatZnxImpl<B: Backend> {
    fn take_mat_znx_impl(
        scratch: &mut Scratch<B>,
        n: usize,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
    ) -> (MatZnx<&mut [u8]>, &mut Scratch<B>);
}
