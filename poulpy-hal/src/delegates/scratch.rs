use crate::{
    api::{ScratchAvailable, ScratchFromBytes, ScratchOwnedAlloc, ScratchOwnedBorrow, TakeSlice},
    layouts::{Backend, Scratch, ScratchOwned},
    oep::HalImpl,
};

impl<B> ScratchOwnedAlloc<B> for ScratchOwned<B>
where
    B: Backend + HalImpl<B>,
{
    fn alloc(size: usize) -> Self {
        B::scratch_owned_alloc(size)
    }
}

impl<B> ScratchOwnedBorrow<B> for ScratchOwned<B>
where
    B: Backend + HalImpl<B>,
{
    fn borrow(&mut self) -> &mut Scratch<B> {
        B::scratch_owned_borrow(self)
    }
}

impl<B> ScratchFromBytes<B> for Scratch<B>
where
    B: Backend + HalImpl<B>,
{
    fn from_bytes(data: &mut [u8]) -> &mut Scratch<B> {
        B::scratch_from_bytes(data)
    }
}

impl<B> ScratchAvailable for Scratch<B>
where
    B: Backend + HalImpl<B>,
{
    fn available(&self) -> usize {
        B::scratch_available(self)
    }
}

impl<B> TakeSlice for Scratch<B>
where
    B: Backend + HalImpl<B>,
{
    fn take_slice<T>(&mut self, len: usize) -> (&mut [T], &mut Self) {
        B::take_slice(self, len)
    }
}
