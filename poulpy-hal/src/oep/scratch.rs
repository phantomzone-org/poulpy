use crate::layouts::{Backend, Scratch, ScratchOwned};

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See the [poulpy-backend/src/cpu_fft64_ref/scratch.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/scratch.rs) reference implementation.
/// * See [crate::api::ScratchOwnedAlloc] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait ScratchOwnedAllocImpl<B: Backend> {
    fn scratch_owned_alloc_impl(size: usize) -> ScratchOwned<B>;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See the [poulpy-backend/src/cpu_fft64_ref/scratch.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/scratch.rs) reference implementation.
/// * See [crate::api::ScratchOwnedBorrow] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait ScratchOwnedBorrowImpl<B: Backend> {
    fn scratch_owned_borrow_impl(scratch: &mut ScratchOwned<B>) -> &mut Scratch<B>;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See the [poulpy-backend/src/cpu_fft64_ref/scratch.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/scratch.rs) reference implementation.
/// * See [crate::api::ScratchFromBytes] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait ScratchFromBytesImpl<B: Backend> {
    fn scratch_from_bytes_impl(data: &mut [u8]) -> &mut Scratch<B>;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See the [poulpy-backend/src/cpu_fft64_ref/scratch.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/scratch.rs) reference implementation.
/// * See [crate::api::ScratchAvailable] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait ScratchAvailableImpl<B: Backend> {
    fn scratch_available_impl(scratch: &Scratch<B>) -> usize;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See the [poulpy-backend/src/cpu_fft64_ref/scratch.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/scratch.rs) reference implementation.
/// * See [crate::api::ScratchOwnedAlloc] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait TakeSliceImpl<B: Backend> {
    fn take_slice_impl<T>(scratch: &mut Scratch<B>, len: usize) -> (&mut [T], &mut Scratch<B>);
}
