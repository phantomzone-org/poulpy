use crate::{
    api::{ScratchAvailable, ScratchFromBytes, ScratchOwnedAlloc, ScratchOwnedBorrow, TakeSlice},
    layouts::{Backend, Scratch, ScratchOwned},
    oep::{ScratchAvailableImpl, ScratchFromBytesImpl, ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeSliceImpl},
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
