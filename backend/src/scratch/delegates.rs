use crate::{
    Backend, MatZnx, Module, ScalarZnx, Scratch, ScratchAvailable, ScratchAvailableImpl, ScratchFromBytes, ScratchFromBytesImpl, ScratchOwned, ScratchOwnedAlloc, ScratchOwnedAllocImpl, ScratchOwnedBorrow, ScratchOwnedBorrowImpl, ScratchTakeMatZnx, ScratchTakeMatZnxImpl, ScratchTakeScalarZnx, ScratchTakeScalarZnxImpl, ScratchTakeSlice, ScratchTakeSliceImpl, ScratchTakeSvpPPol, ScratchTakeSvpPPolImpl, ScratchTakeVecZnx, ScratchTakeVecZnxBig, ScratchTakeVecZnxBigImpl, ScratchTakeVecZnxDft, ScratchTakeVecZnxDftImpl, ScratchTakeVecZnxDftSlice, ScratchTakeVecZnxDftSliceImpl, ScratchTakeVecZnxImpl, ScratchTakeVecZnxSlice, ScratchTakeVecZnxSliceImpl, ScratchTakeVmpPMat, ScratchTakeVmpPMatImpl, SvpPPol, VecZnx, VecZnxBig, VecZnxDft, VmpPMat
};

impl<B: Backend> ScratchOwnedAlloc<B> for ScratchOwned<B>
where
    B: Backend + ScratchOwnedAllocImpl<B>,
{
    fn alloc(size: usize) -> Self {
        B::scratch_owned_alloc_impl(size)
    }
}

impl<B: Backend> ScratchOwnedBorrow<B> for ScratchOwned<B>
where
    B: Backend + ScratchOwnedBorrowImpl<B>,
{
    fn borrow(&mut self) -> &mut super::Scratch<B> {
        B::scratch_owned_borrow_impl(self)
    }
}

impl<B: Backend> ScratchFromBytes<B> for Scratch<B>
where
    B: Backend + ScratchFromBytesImpl<B>,
{
    fn from_bytes(data: &mut [u8]) -> &mut Scratch<B> {
        B::scratch_from_bytes_impl(data)
    }
}

impl<B: Backend> ScratchAvailable for Scratch<B>
where
    B: Backend + ScratchAvailableImpl<B>,
{
    fn available(&self) -> usize {
        B::scratch_available_impl(self)
    }
}

impl<B: Backend> ScratchTakeSlice for Scratch<B>
where
    B: Backend + ScratchTakeSliceImpl<B>,
{
    fn take_slice<T>(&mut self, len: usize) -> (&mut [T], &mut Self) {
        B::scratch_take_slice_impl(self, len)
    }
}

impl<B: Backend> ScratchTakeScalarZnx<B> for Scratch<B>
where
    B: Backend + ScratchTakeScalarZnxImpl<B>,
{
    fn take_scalar_znx(&mut self, module: &Module<B>, cols: usize) -> (ScalarZnx<&mut [u8]>, &mut Self) {
        B::scratch_take_scalar_znx_impl(self, module.n(), cols)
    }
}

impl<B: Backend> ScratchTakeSvpPPol<B> for Scratch<B>
where
    B: Backend + ScratchTakeSvpPPolImpl<B>,
{
    fn take_svp_ppol(&mut self, module: &Module<B>, cols: usize) -> (SvpPPol<&mut [u8], B>, &mut Self) {
        B::scratch_take_svp_ppol_impl(self, module.n(), cols)
    }
}

impl<B: Backend> ScratchTakeVecZnx<B> for Scratch<B>
where
    B: Backend + ScratchTakeVecZnxImpl<B>,
{
    fn take_vec_znx(&mut self, module: &Module<B>, cols: usize, size: usize) -> (VecZnx<&mut [u8]>, &mut Self) {
        B::scratch_take_vec_znx_impl(self, module.n(), cols, size)
    }
}

impl<B: Backend> ScratchTakeVecZnxSlice<B> for Scratch<B>
where
    B: Backend + ScratchTakeVecZnxSliceImpl<B>,
{
    fn take_vec_znx_slice(
        &mut self,
        len: usize,
        module: &Module<B>,
        cols: usize,
        size: usize,
    ) -> (Vec<VecZnx<&mut [u8]>>, &mut Self) {
        B::scratch_take_vec_znx_slice_impl(self, len, module.n(), cols, size)
    }
}

impl<B: Backend> ScratchTakeVecZnxBig<B> for Scratch<B>
where
    B: Backend + ScratchTakeVecZnxBigImpl<B>,
{
    fn take_vec_znx_big(&mut self, module: &Module<B>, cols: usize, size: usize) -> (VecZnxBig<&mut [u8], B>, &mut Self) {
        B::scratch_take_vec_znx_big_impl(self, module.n(), cols, size)
    }
}

impl<B: Backend> ScratchTakeVecZnxDft<B> for Scratch<B>
where
    B: Backend + ScratchTakeVecZnxDftImpl<B>,
{
    fn take_vec_znx_dft(&mut self, module: &Module<B>, cols: usize, size: usize) -> (VecZnxDft<&mut [u8], B>, &mut Self) {
        B::scratch_take_vec_znx_dft_impl(self, module.n(), cols, size)
    }
}

impl<B: Backend> ScratchTakeVecZnxDftSlice<B> for Scratch<B>
where
    B: Backend + ScratchTakeVecZnxDftSliceImpl<B>,
{
    fn take_vec_znx_dft_slice(
        &mut self,
        len: usize,
        module: &Module<B>,
        cols: usize,
        size: usize,
    ) -> (Vec<VecZnxDft<&mut [u8], B>>, &mut Self) {
        B::scratch_take_vec_znx_dft_slice_impl(self, len, module.n(), cols, size)
    }
}

impl<B: Backend> ScratchTakeVmpPMat<B> for Scratch<B>
where
    B: Backend + ScratchTakeVmpPMatImpl<B>,
{
    fn take_vmp_pmat(
        &mut self,
        module: &Module<B>,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
    ) -> (VmpPMat<&mut [u8], B>, &mut Self) {
        B::scratch_take_vmp_pmat_impl(self, module.n(), rows, cols_in, cols_out, size)
    }
}

impl<B: Backend> ScratchTakeMatZnx<B> for Scratch<B>
where
    B: Backend + ScratchTakeMatZnxImpl<B>,
{
    fn take_mat_znx(
        &mut self,
        module: &Module<B>,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
    ) -> (MatZnx<&mut [u8]>, &mut Self) {
        B::scratch_take_mat_znx_impl(self, module.n(), rows, cols_in, cols_out, size)
    }
}
