macro_rules! hal_impl_scratch {
    () => {
        fn scratch_owned_alloc(size: usize) -> ScratchOwned<Self> {
            <Self as HalScratchDefaults<Self>>::scratch_owned_alloc_default(size)
        }

        fn scratch_owned_borrow(scratch: &mut ScratchOwned<Self>) -> &mut Scratch<Self> {
            <Self as HalScratchDefaults<Self>>::scratch_owned_borrow_default(scratch)
        }

        fn scratch_from_bytes(data: &mut [u8]) -> &mut Scratch<Self> {
            <Self as HalScratchDefaults<Self>>::scratch_from_bytes_default(data)
        }

        fn scratch_available(scratch: &Scratch<Self>) -> usize {
            <Self as HalScratchDefaults<Self>>::scratch_available_default(scratch)
        }

        fn take_slice<T>(scratch: &mut Scratch<Self>, len: usize) -> (&mut [T], &mut Scratch<Self>) {
            <Self as HalScratchDefaults<Self>>::take_slice_default(scratch, len)
        }
    };
}
