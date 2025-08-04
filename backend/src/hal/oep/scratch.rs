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

pub unsafe trait ScratchTakeSliceImpl<B: Backend> {
    fn scratch_take_slice_impl<T>(scratch: &mut Scratch<B>, len: usize) -> (&mut [T], &mut Scratch<B>);
}

pub unsafe trait ScratchTakeScalarZnxImpl<B: Backend> {
    fn scratch_take_scalar_znx_impl(scratch: &mut Scratch<B>, n: usize, cols: usize) -> (ScalarZnx<&mut [u8]>, &mut Scratch<B>);
}

pub unsafe trait ScratchTakeSvpPPolImpl<B: Backend> {
    fn scratch_take_svp_ppol_impl(scratch: &mut Scratch<B>, n: usize, cols: usize) -> (SvpPPol<&mut [u8], B>, &mut Scratch<B>);
}

pub unsafe trait ScratchTakeVecZnxImpl<B: Backend> {
    fn scratch_take_vec_znx_impl(
        scratch: &mut Scratch<B>,
        n: usize,
        cols: usize,
        size: usize,
    ) -> (VecZnx<&mut [u8]>, &mut Scratch<B>);
}

pub unsafe trait ScratchTakeVecZnxSliceImpl<B: Backend> {
    fn scratch_take_vec_znx_slice_impl(
        scratch: &mut Scratch<B>,
        len: usize,
        n: usize,
        cols: usize,
        size: usize,
    ) -> (Vec<VecZnx<&mut [u8]>>, &mut Scratch<B>);
}

pub unsafe trait ScratchTakeVecZnxBigImpl<B: Backend> {
    fn scratch_take_vec_znx_big_impl(
        scratch: &mut Scratch<B>,
        n: usize,
        cols: usize,
        size: usize,
    ) -> (VecZnxBig<&mut [u8], B>, &mut Scratch<B>);
}

pub unsafe trait ScratchTakeVecZnxDftImpl<B: Backend> {
    fn scratch_take_vec_znx_dft_impl(
        scratch: &mut Scratch<B>,
        n: usize,
        cols: usize,
        size: usize,
    ) -> (VecZnxDft<&mut [u8], B>, &mut Scratch<B>);
}

pub unsafe trait ScratchTakeVecZnxDftSliceImpl<B: Backend> {
    fn scratch_take_vec_znx_dft_slice_impl(
        scratch: &mut Scratch<B>,
        len: usize,
        n: usize,
        cols: usize,
        size: usize,
    ) -> (Vec<VecZnxDft<&mut [u8], B>>, &mut Scratch<B>);
}

pub unsafe trait ScratchTakeVmpPMatImpl<B: Backend> {
    fn scratch_take_vmp_pmat_impl(
        scratch: &mut Scratch<B>,
        n: usize,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
    ) -> (VmpPMat<&mut [u8], B>, &mut Scratch<B>);
}

pub unsafe trait ScratchTakeMatZnxImpl<B: Backend> {
    fn scratch_take_mat_znx_impl(
        scratch: &mut Scratch<B>,
        n: usize,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
    ) -> (MatZnx<&mut [u8]>, &mut Scratch<B>);
}
