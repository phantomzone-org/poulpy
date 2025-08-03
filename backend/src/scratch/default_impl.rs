use std::marker::PhantomData;

use crate::{
    Backend, DEFAULTALIGN, MatZnx, ScalarZnx, ScalarZnxAllocBytesImpl, Scratch, ScratchAvailableImpl, ScratchFromBytes,
    ScratchFromBytesImpl, ScratchOwned, ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, ScratchTakeMatZnxImpl,
    ScratchTakeScalarZnxImpl, ScratchTakeSliceImpl, ScratchTakeSvpPPolImpl, ScratchTakeVecZnxBigImpl, ScratchTakeVecZnxDftImpl,
    ScratchTakeVecZnxDftSliceImpl, ScratchTakeVecZnxImpl, ScratchTakeVecZnxSliceImpl, ScratchTakeVmpPMatImpl, SvpPPol,
    SvpPPolAllocBytesImpl, VecZnx, VecZnxAllocBytesImpl, VecZnxBig, VecZnxBigAllocBytesImpl, VecZnxDft, VecZnxDftAllocBytesImpl,
    VmpPMat, VmpPMatAllocBytesImpl, alloc_aligned,
};

unsafe impl<B: Backend> ScratchOwnedAllocImpl<B> for B {
    fn scratch_owned_alloc_impl(size: usize) -> ScratchOwned<B> {
        let data: Vec<u8> = alloc_aligned(size);
        ScratchOwned {
            data,
            _phantom: PhantomData,
        }
    }
}

unsafe impl<B: Backend> ScratchOwnedBorrowImpl<B> for B {
    fn scratch_owned_borrow_impl(scratch: &mut ScratchOwned<B>) -> &mut Scratch<B> {
        Scratch::from_bytes(&mut scratch.data)
    }
}

unsafe impl<B: Backend> ScratchFromBytesImpl<B> for B {
    fn scratch_from_bytes_impl(data: &mut [u8]) -> &mut Scratch<B> {
        unsafe { &mut *(data as *mut [u8] as *mut Scratch<B>) }
    }
}

unsafe impl<B: Backend> ScratchAvailableImpl<B> for B {
    fn scratch_available_impl(scratch: &Scratch<B>) -> usize {
        let ptr: *const u8 = scratch.data.as_ptr();
        let self_len: usize = scratch.data.len();
        let aligned_offset: usize = ptr.align_offset(DEFAULTALIGN);
        self_len.saturating_sub(aligned_offset)
    }
}

unsafe impl<B: Backend> ScratchTakeSliceImpl<B> for B {
    fn scratch_take_slice_impl<T>(scratch: &mut Scratch<B>, len: usize) -> (&mut [T], &mut Scratch<B>) {
        let (take_slice, rem_slice) = take_slice_aligned(&mut scratch.data, len * std::mem::size_of::<T>());

        unsafe {
            (
                &mut *(std::ptr::slice_from_raw_parts_mut(take_slice.as_mut_ptr() as *mut T, len)),
                Scratch::from_bytes(rem_slice),
            )
        }
    }
}

unsafe impl<B: Backend> ScratchTakeScalarZnxImpl<B> for B
where
    B: ScalarZnxAllocBytesImpl<B>,
{
    fn scratch_take_scalar_znx_impl(scratch: &mut Scratch<B>, n: usize, cols: usize) -> (ScalarZnx<&mut [u8]>, &mut Scratch<B>) {
        let (take_slice, rem_slice) = take_slice_aligned(&mut scratch.data, B::scalar_znx_alloc_bytes_impl(n, cols));
        (
            ScalarZnx::from_data(take_slice, n, cols),
            Scratch::from_bytes(rem_slice),
        )
    }
}

unsafe impl<B: Backend> ScratchTakeSvpPPolImpl<B> for B
where
    B: SvpPPolAllocBytesImpl<B>,
{
    fn scratch_take_svp_ppol_impl(scratch: &mut Scratch<B>, n: usize, cols: usize) -> (SvpPPol<&mut [u8], B>, &mut Scratch<B>) {
        let (take_slice, rem_slice) = take_slice_aligned(&mut scratch.data, B::svp_ppol_alloc_bytes_impl(n, cols));
        (
            SvpPPol::from_data(take_slice, n, cols),
            Scratch::from_bytes(rem_slice),
        )
    }
}

unsafe impl<B: Backend> ScratchTakeVecZnxImpl<B> for B
where
    B: VecZnxAllocBytesImpl<B>,
{
    fn scratch_take_vec_znx_impl(
        scratch: &mut Scratch<B>,
        n: usize,
        cols: usize,
        size: usize,
    ) -> (VecZnx<&mut [u8]>, &mut Scratch<B>) {
        let (take_slice, rem_slice) = take_slice_aligned(
            &mut scratch.data,
            B::vec_znx_alloc_bytes_impl(n, cols, size),
        );
        (
            VecZnx::from_data(take_slice, n, cols, size),
            Scratch::from_bytes(rem_slice),
        )
    }
}

unsafe impl<B: Backend> ScratchTakeVecZnxBigImpl<B> for B
where
    B: VecZnxBigAllocBytesImpl<B>,
{
    fn scratch_take_vec_znx_big_impl(
        scratch: &mut Scratch<B>,
        n: usize,
        cols: usize,
        size: usize,
    ) -> (VecZnxBig<&mut [u8], B>, &mut Scratch<B>) {
        let (take_slice, rem_slice) = take_slice_aligned(
            &mut scratch.data,
            B::vec_znx_big_alloc_bytes_impl(n, cols, size),
        );
        (
            VecZnxBig::from_data(take_slice, n, cols, size),
            Scratch::from_bytes(rem_slice),
        )
    }
}

unsafe impl<B: Backend> ScratchTakeVecZnxDftImpl<B> for B
where
    B: VecZnxDftAllocBytesImpl<B>,
{
    fn scratch_take_vec_znx_dft_impl(
        scratch: &mut Scratch<B>,
        n: usize,
        cols: usize,
        size: usize,
    ) -> (VecZnxDft<&mut [u8], B>, &mut Scratch<B>) {
        let (take_slice, rem_slice) = take_slice_aligned(
            &mut scratch.data,
            B::vec_znx_dft_alloc_bytes_impl(n, cols, size),
        );

        (
            VecZnxDft::from_data(take_slice, n, cols, size),
            Scratch::from_bytes(rem_slice),
        )
    }
}

unsafe impl<B: Backend> ScratchTakeVecZnxDftSliceImpl<B> for B
where
    B: VecZnxDftAllocBytesImpl<B>,
{
    fn scratch_take_vec_znx_dft_slice_impl(
        scratch: &mut Scratch<B>,
        len: usize,
        n: usize,
        cols: usize,
        size: usize,
    ) -> (Vec<VecZnxDft<&mut [u8], B>>, &mut Scratch<B>) {
        let mut scratch: &mut Scratch<B> = scratch;
        let mut slice: Vec<VecZnxDft<&mut [u8], B>> = Vec::with_capacity(len);
        for _ in 0..len {
            let (znx, new_scratch) = B::scratch_take_vec_znx_dft_impl(scratch, n, cols, size);
            scratch = new_scratch;
            slice.push(znx);
        }
        (slice, scratch)
    }
}

unsafe impl<B: Backend> ScratchTakeVecZnxSliceImpl<B> for B {
    fn scratch_take_vec_znx_slice_impl(
        scratch: &mut Scratch<B>,
        len: usize,
        n: usize,
        cols: usize,
        size: usize,
    ) -> (Vec<VecZnx<&mut [u8]>>, &mut Scratch<B>) {
        let mut scratch: &mut Scratch<B> = scratch;
        let mut slice: Vec<VecZnx<&mut [u8]>> = Vec::with_capacity(len);
        for _ in 0..len {
            let (znx, new_scratch) = B::scratch_take_vec_znx_impl(scratch, n, cols, size);
            scratch = new_scratch;
            slice.push(znx);
        }
        (slice, scratch)
    }
}

unsafe impl<B: Backend> ScratchTakeVmpPMatImpl<B> for B
where
    B: VmpPMatAllocBytesImpl<B>,
{
    fn scratch_take_vmp_pmat_impl(
        scratch: &mut Scratch<B>,
        n: usize,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
    ) -> (VmpPMat<&mut [u8], B>, &mut Scratch<B>) {
        let (take_slice, rem_slice) = take_slice_aligned(
            &mut scratch.data,
            B::vmp_pmat_alloc_bytes_impl(n, rows, cols_in, cols_out, size),
        );
        (
            VmpPMat::from_data(take_slice, n, rows, cols_in, cols_out, size),
            Scratch::from_bytes(rem_slice),
        )
    }
}

unsafe impl<B: Backend> ScratchTakeMatZnxImpl<B> for B {
    fn scratch_take_mat_znx_impl(
        scratch: &mut Scratch<B>,
        n: usize,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
    ) -> (MatZnx<&mut [u8]>, &mut Scratch<B>) {
        let (take_slice, rem_slice) = take_slice_aligned(
            &mut scratch.data,
            MatZnx::<Vec<u8>>::bytes_of(n, rows, cols_in, cols_out, size),
        );
        (
            MatZnx::from_data(take_slice, n, rows, cols_in, cols_out, size),
            Scratch::from_bytes(rem_slice),
        )
    }
}

fn take_slice_aligned(data: &mut [u8], take_len: usize) -> (&mut [u8], &mut [u8]) {
    let ptr: *mut u8 = data.as_mut_ptr();
    let self_len: usize = data.len();

    let aligned_offset: usize = ptr.align_offset(DEFAULTALIGN);
    let aligned_len: usize = self_len.saturating_sub(aligned_offset);

    if let Some(rem_len) = aligned_len.checked_sub(take_len) {
        unsafe {
            let rem_ptr: *mut u8 = ptr.add(aligned_offset).add(take_len);
            let rem_slice: &mut [u8] = &mut *std::ptr::slice_from_raw_parts_mut(rem_ptr, rem_len);

            let take_slice: &mut [u8] = &mut *std::ptr::slice_from_raw_parts_mut(ptr.add(aligned_offset), take_len);

            return (take_slice, rem_slice);
        }
    } else {
        panic!(
            "Attempted to take {} from scratch with {} aligned bytes left",
            take_len, aligned_len,
        );
    }
}
