use crate::{
    api::{
        ScratchAvailable, ScratchFromBytes, ScratchOwnedAlloc, ScratchOwnedBorrow, TakeLike, TakeMatZnx, TakeScalarZnx,
        TakeSlice, TakeSvpPPol, TakeVecZnx, TakeVecZnxBig, TakeVecZnxDft, TakeVecZnxDftSlice, TakeVecZnxSlice, TakeVmpPMat,
    },
    layouts::{Backend, DataRef, MatZnx, ScalarZnx, Scratch, ScratchOwned, SvpPPol, VecZnx, VecZnxBig, VecZnxDft, VmpPMat},
    oep::{
        ScratchAvailableImpl, ScratchFromBytesImpl, ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeLikeImpl, TakeMatZnxImpl,
        TakeScalarZnxImpl, TakeSliceImpl, TakeSvpPPolImpl, TakeVecZnxBigImpl, TakeVecZnxDftImpl, TakeVecZnxDftSliceImpl,
        TakeVecZnxImpl, TakeVecZnxSliceImpl, TakeVmpPMatImpl,
    },
};

impl<B> ScratchOwnedAlloc<B> for ScratchOwned<B>
where
    B: Backend + ScratchOwnedAllocImpl<B>,
{
    fn alloc(size: usize) -> Self {
        B::scratch_owned_alloc_impl(size)
    }
}

impl<B> ScratchOwnedBorrow<B> for ScratchOwned<B>
where
    B: Backend + ScratchOwnedBorrowImpl<B>,
{
    fn borrow(&mut self) -> &mut Scratch<B> {
        B::scratch_owned_borrow_impl(self)
    }
}

impl<B> ScratchFromBytes<B> for Scratch<B>
where
    B: Backend + ScratchFromBytesImpl<B>,
{
    fn from_bytes(data: &mut [u8]) -> &mut Scratch<B> {
        B::scratch_from_bytes_impl(data)
    }
}

impl<B> ScratchAvailable for Scratch<B>
where
    B: Backend + ScratchAvailableImpl<B>,
{
    fn available(&self) -> usize {
        B::scratch_available_impl(self)
    }
}

impl<B> TakeSlice for Scratch<B>
where
    B: Backend + TakeSliceImpl<B>,
{
    fn take_slice<T>(&mut self, len: usize) -> (&mut [T], &mut Self) {
        B::take_slice_impl(self, len)
    }
}

impl<B> TakeScalarZnx for Scratch<B>
where
    B: Backend + TakeScalarZnxImpl<B>,
{
    fn take_scalar_znx(&mut self, n: usize, cols: usize) -> (ScalarZnx<&mut [u8]>, &mut Self) {
        B::take_scalar_znx_impl(self, n, cols)
    }
}

impl<B> TakeSvpPPol<B> for Scratch<B>
where
    B: Backend + TakeSvpPPolImpl<B>,
{
    fn take_svp_ppol(&mut self, n: usize, cols: usize) -> (SvpPPol<&mut [u8], B>, &mut Self) {
        B::take_svp_ppol_impl(self, n, cols)
    }
}

impl<B> TakeVecZnx for Scratch<B>
where
    B: Backend + TakeVecZnxImpl<B>,
{
    fn take_vec_znx(&mut self, n: usize, cols: usize, size: usize) -> (VecZnx<&mut [u8]>, &mut Self) {
        B::take_vec_znx_impl(self, n, cols, size)
    }
}

impl<B> TakeVecZnxSlice for Scratch<B>
where
    B: Backend + TakeVecZnxSliceImpl<B>,
{
    fn take_vec_znx_slice(&mut self, len: usize, n: usize, cols: usize, size: usize) -> (Vec<VecZnx<&mut [u8]>>, &mut Self) {
        B::take_vec_znx_slice_impl(self, len, n, cols, size)
    }
}

impl<B> TakeVecZnxBig<B> for Scratch<B>
where
    B: Backend + TakeVecZnxBigImpl<B>,
{
    fn take_vec_znx_big(&mut self, n: usize, cols: usize, size: usize) -> (VecZnxBig<&mut [u8], B>, &mut Self) {
        B::take_vec_znx_big_impl(self, n, cols, size)
    }
}

impl<B> TakeVecZnxDft<B> for Scratch<B>
where
    B: Backend + TakeVecZnxDftImpl<B>,
{
    fn take_vec_znx_dft(&mut self, n: usize, cols: usize, size: usize) -> (VecZnxDft<&mut [u8], B>, &mut Self) {
        B::take_vec_znx_dft_impl(self, n, cols, size)
    }
}

impl<B> TakeVecZnxDftSlice<B> for Scratch<B>
where
    B: Backend + TakeVecZnxDftSliceImpl<B>,
{
    fn take_vec_znx_dft_slice(
        &mut self,
        len: usize,
        n: usize,
        cols: usize,
        size: usize,
    ) -> (Vec<VecZnxDft<&mut [u8], B>>, &mut Self) {
        B::take_vec_znx_dft_slice_impl(self, len, n, cols, size)
    }
}

impl<B> TakeVmpPMat<B> for Scratch<B>
where
    B: Backend + TakeVmpPMatImpl<B>,
{
    fn take_vmp_pmat(
        &mut self,
        n: usize,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
    ) -> (VmpPMat<&mut [u8], B>, &mut Self) {
        B::take_vmp_pmat_impl(self, n, rows, cols_in, cols_out, size)
    }
}

impl<B> TakeMatZnx for Scratch<B>
where
    B: Backend + TakeMatZnxImpl<B>,
{
    fn take_mat_znx(
        &mut self,
        n: usize,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
    ) -> (MatZnx<&mut [u8]>, &mut Self) {
        B::take_mat_znx_impl(self, n, rows, cols_in, cols_out, size)
    }
}

impl<'a, B: Backend, D> TakeLike<'a, B, ScalarZnx<D>> for Scratch<B>
where
    B: TakeLikeImpl<'a, B, ScalarZnx<D>, Output = ScalarZnx<&'a mut [u8]>>,
    D: DataRef,
{
    type Output = ScalarZnx<&'a mut [u8]>;
    fn take_like(&'a mut self, template: &ScalarZnx<D>) -> (Self::Output, &'a mut Self) {
        B::take_like_impl(self, template)
    }
}

impl<'a, B: Backend, D> TakeLike<'a, B, SvpPPol<D, B>> for Scratch<B>
where
    B: TakeLikeImpl<'a, B, SvpPPol<D, B>, Output = SvpPPol<&'a mut [u8], B>>,
    D: DataRef,
{
    type Output = SvpPPol<&'a mut [u8], B>;
    fn take_like(&'a mut self, template: &SvpPPol<D, B>) -> (Self::Output, &'a mut Self) {
        B::take_like_impl(self, template)
    }
}

impl<'a, B: Backend, D> TakeLike<'a, B, VecZnx<D>> for Scratch<B>
where
    B: TakeLikeImpl<'a, B, VecZnx<D>, Output = VecZnx<&'a mut [u8]>>,
    D: DataRef,
{
    type Output = VecZnx<&'a mut [u8]>;
    fn take_like(&'a mut self, template: &VecZnx<D>) -> (Self::Output, &'a mut Self) {
        B::take_like_impl(self, template)
    }
}

impl<'a, B: Backend, D> TakeLike<'a, B, VecZnxBig<D, B>> for Scratch<B>
where
    B: TakeLikeImpl<'a, B, VecZnxBig<D, B>, Output = VecZnxBig<&'a mut [u8], B>>,
    D: DataRef,
{
    type Output = VecZnxBig<&'a mut [u8], B>;
    fn take_like(&'a mut self, template: &VecZnxBig<D, B>) -> (Self::Output, &'a mut Self) {
        B::take_like_impl(self, template)
    }
}

impl<'a, B: Backend, D> TakeLike<'a, B, VecZnxDft<D, B>> for Scratch<B>
where
    B: TakeLikeImpl<'a, B, VecZnxDft<D, B>, Output = VecZnxDft<&'a mut [u8], B>>,
    D: DataRef,
{
    type Output = VecZnxDft<&'a mut [u8], B>;
    fn take_like(&'a mut self, template: &VecZnxDft<D, B>) -> (Self::Output, &'a mut Self) {
        B::take_like_impl(self, template)
    }
}

impl<'a, B: Backend, D> TakeLike<'a, B, MatZnx<D>> for Scratch<B>
where
    B: TakeLikeImpl<'a, B, MatZnx<D>, Output = MatZnx<&'a mut [u8]>>,
    D: DataRef,
{
    type Output = MatZnx<&'a mut [u8]>;
    fn take_like(&'a mut self, template: &MatZnx<D>) -> (Self::Output, &'a mut Self) {
        B::take_like_impl(self, template)
    }
}

impl<'a, B: Backend, D> TakeLike<'a, B, VmpPMat<D, B>> for Scratch<B>
where
    B: TakeLikeImpl<'a, B, VmpPMat<D, B>, Output = VmpPMat<&'a mut [u8], B>>,
    D: DataRef,
{
    type Output = VmpPMat<&'a mut [u8], B>;
    fn take_like(&'a mut self, template: &VmpPMat<D, B>) -> (Self::Output, &'a mut Self) {
        B::take_like_impl(self, template)
    }
}
