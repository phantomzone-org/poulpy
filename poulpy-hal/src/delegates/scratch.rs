use crate::{
    api::{ScratchAvailable, ScratchFromBytes, ScratchOwnedAlloc, ScratchOwnedBorrow, TakeSlice},
    layouts::{Backend, Scratch, ScratchArena, ScratchOwned},
    oep::HalScratchImpl,
};

impl<B> ScratchOwnedAlloc<B> for ScratchOwned<B>
where
    B: Backend + HalScratchImpl<B>,
{
    fn alloc(size: usize) -> Self {
        <B as HalScratchImpl<B>>::scratch_owned_alloc(size)
    }
}

impl<B> ScratchOwnedBorrow<B> for ScratchOwned<B>
where
    B: Backend + HalScratchImpl<B>,
{
    fn borrow(&mut self) -> ScratchArena<'_, B> {
        self.arena()
    }
}

impl<B> ScratchFromBytes<B> for Scratch<B>
where
    B: Backend + HalScratchImpl<B>,
{
    fn from_bytes(data: &mut [u8]) -> &mut Scratch<B> {
        <B as HalScratchImpl<B>>::scratch_from_bytes(data)
    }
}

impl<B> ScratchAvailable for Scratch<B>
where
    B: Backend + HalScratchImpl<B>,
{
    fn available(&self) -> usize {
        <B as HalScratchImpl<B>>::scratch_available(self)
    }
}

impl<B: Backend> ScratchAvailable for ScratchArena<'_, B> {
    fn available(&self) -> usize {
        ScratchArena::available(self)
    }
}

impl<B> TakeSlice for Scratch<B>
where
    B: Backend + HalScratchImpl<B>,
{
    fn take_slice<T>(&mut self, len: usize) -> (&mut [T], &mut Self) {
        <B as HalScratchImpl<B>>::take_slice(self, len)
    }
}
