use crate::{
    api::ZnxInfos,
    layouts::{Backend, DataRef, MatZnx, ScalarZnx, Scratch, ScratchOwned, SvpPPol, VecZnx, VecZnxBig, VecZnxDft, VmpPMat},
};

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See TODO for reference code.
/// * See TODO for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait ScratchOwnedAllocImpl<B: Backend> {
    fn scratch_owned_alloc_impl(size: usize) -> ScratchOwned<B>;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See TODO for reference code.
/// * See TODO for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait ScratchOwnedBorrowImpl<B: Backend> {
    fn scratch_owned_borrow_impl(scratch: &mut ScratchOwned<B>) -> &mut Scratch<B>;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See TODO for reference code.
/// * See TODO for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait ScratchFromBytesImpl<B: Backend> {
    fn scratch_from_bytes_impl(data: &mut [u8]) -> &mut Scratch<B>;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See TODO for reference code.
/// * See TODO for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait ScratchAvailableImpl<B: Backend> {
    fn scratch_available_impl(scratch: &Scratch<B>) -> usize;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See TODO for reference code.
/// * See TODO for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait TakeSliceImpl<B: Backend> {
    fn take_slice_impl<T>(scratch: &mut Scratch<B>, len: usize) -> (&mut [T], &mut Scratch<B>);
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See TODO for reference code.
/// * See TODO for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait TakeScalarZnxImpl<B: Backend> {
    fn take_scalar_znx_impl(scratch: &mut Scratch<B>, n: usize, cols: usize) -> (ScalarZnx<&mut [u8]>, &mut Scratch<B>);
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See TODO for reference code.
/// * See TODO for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait TakeSvpPPolImpl<B: Backend> {
    fn take_svp_ppol_impl(scratch: &mut Scratch<B>, n: usize, cols: usize) -> (SvpPPol<&mut [u8], B>, &mut Scratch<B>);
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See TODO for reference code.
/// * See TODO for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait TakeVecZnxImpl<B: Backend> {
    fn take_vec_znx_impl(scratch: &mut Scratch<B>, n: usize, cols: usize, size: usize) -> (VecZnx<&mut [u8]>, &mut Scratch<B>);
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See TODO for reference code.
/// * See TODO for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait TakeVecZnxSliceImpl<B: Backend> {
    fn take_vec_znx_slice_impl(
        scratch: &mut Scratch<B>,
        len: usize,
        n: usize,
        cols: usize,
        size: usize,
    ) -> (Vec<VecZnx<&mut [u8]>>, &mut Scratch<B>);
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See TODO for reference code.
/// * See TODO for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait TakeVecZnxBigImpl<B: Backend> {
    fn take_vec_znx_big_impl(
        scratch: &mut Scratch<B>,
        n: usize,
        cols: usize,
        size: usize,
    ) -> (VecZnxBig<&mut [u8], B>, &mut Scratch<B>);
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See TODO for reference code.
/// * See TODO for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait TakeVecZnxDftImpl<B: Backend> {
    fn take_vec_znx_dft_impl(
        scratch: &mut Scratch<B>,
        n: usize,
        cols: usize,
        size: usize,
    ) -> (VecZnxDft<&mut [u8], B>, &mut Scratch<B>);
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See TODO for reference code.
/// * See TODO for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait TakeVecZnxDftSliceImpl<B: Backend> {
    fn take_vec_znx_dft_slice_impl(
        scratch: &mut Scratch<B>,
        len: usize,
        n: usize,
        cols: usize,
        size: usize,
    ) -> (Vec<VecZnxDft<&mut [u8], B>>, &mut Scratch<B>);
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See TODO for reference code.
/// * See TODO for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
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

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See TODO for reference code.
/// * See TODO for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
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

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See TODO for reference code.
/// * See TODO for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub trait TakeLikeImpl<'a, B: Backend, T> {
    type Output;
    fn take_like_impl(scratch: &'a mut Scratch<B>, template: &T) -> (Self::Output, &'a mut Scratch<B>);
}

impl<'a, B: Backend, D> TakeLikeImpl<'a, B, VmpPMat<D, B>> for B
where
    B: TakeVmpPMatImpl<B>,
    D: DataRef,
{
    type Output = VmpPMat<&'a mut [u8], B>;

    fn take_like_impl(scratch: &'a mut Scratch<B>, template: &VmpPMat<D, B>) -> (Self::Output, &'a mut Scratch<B>) {
        B::take_vmp_pmat_impl(
            scratch,
            template.n(),
            template.rows(),
            template.cols_in(),
            template.cols_out(),
            template.size(),
        )
    }
}

impl<'a, B: Backend, D> TakeLikeImpl<'a, B, MatZnx<D>> for B
where
    B: TakeMatZnxImpl<B>,
    D: DataRef,
{
    type Output = MatZnx<&'a mut [u8]>;

    fn take_like_impl(scratch: &'a mut Scratch<B>, template: &MatZnx<D>) -> (Self::Output, &'a mut Scratch<B>) {
        B::take_mat_znx_impl(
            scratch,
            template.n(),
            template.rows(),
            template.cols_in(),
            template.cols_out(),
            template.size(),
        )
    }
}

impl<'a, B: Backend, D> TakeLikeImpl<'a, B, VecZnxDft<D, B>> for B
where
    B: TakeVecZnxDftImpl<B>,
    D: DataRef,
{
    type Output = VecZnxDft<&'a mut [u8], B>;

    fn take_like_impl(scratch: &'a mut Scratch<B>, template: &VecZnxDft<D, B>) -> (Self::Output, &'a mut Scratch<B>) {
        B::take_vec_znx_dft_impl(scratch, template.n(), template.cols(), template.size())
    }
}

impl<'a, B: Backend, D> TakeLikeImpl<'a, B, VecZnxBig<D, B>> for B
where
    B: TakeVecZnxBigImpl<B>,
    D: DataRef,
{
    type Output = VecZnxBig<&'a mut [u8], B>;

    fn take_like_impl(scratch: &'a mut Scratch<B>, template: &VecZnxBig<D, B>) -> (Self::Output, &'a mut Scratch<B>) {
        B::take_vec_znx_big_impl(scratch, template.n(), template.cols(), template.size())
    }
}

impl<'a, B: Backend, D> TakeLikeImpl<'a, B, SvpPPol<D, B>> for B
where
    B: TakeSvpPPolImpl<B>,
    D: DataRef,
{
    type Output = SvpPPol<&'a mut [u8], B>;

    fn take_like_impl(scratch: &'a mut Scratch<B>, template: &SvpPPol<D, B>) -> (Self::Output, &'a mut Scratch<B>) {
        B::take_svp_ppol_impl(scratch, template.n(), template.cols())
    }
}

impl<'a, B: Backend, D> TakeLikeImpl<'a, B, VecZnx<D>> for B
where
    B: TakeVecZnxImpl<B>,
    D: DataRef,
{
    type Output = VecZnx<&'a mut [u8]>;

    fn take_like_impl(scratch: &'a mut Scratch<B>, template: &VecZnx<D>) -> (Self::Output, &'a mut Scratch<B>) {
        B::take_vec_znx_impl(scratch, template.n(), template.cols(), template.size())
    }
}

impl<'a, B: Backend, D> TakeLikeImpl<'a, B, ScalarZnx<D>> for B
where
    B: TakeScalarZnxImpl<B>,
    D: DataRef,
{
    type Output = ScalarZnx<&'a mut [u8]>;

    fn take_like_impl(scratch: &'a mut Scratch<B>, template: &ScalarZnx<D>) -> (Self::Output, &'a mut Scratch<B>) {
        B::take_scalar_znx_impl(scratch, template.n(), template.cols())
    }
}
