//! Backend extension points for [`Scratch`] memory allocation and management.

use std::marker::PhantomData;

use poulpy_hal::{
    DEFAULTALIGN,
    layouts::{Backend, DeviceBuf, Scratch, ScratchOwned},
};

#[doc(hidden)]
pub trait HalScratchDefaults<BE: Backend>: Backend {
    fn scratch_owned_alloc_default(size: usize) -> ScratchOwned<BE> {
        let data = DeviceBuf::<BE>::new(BE::alloc_bytes(size));
        ScratchOwned {
            data,
            _phantom: PhantomData,
        }
    }

    fn scratch_owned_borrow_default(scratch: &mut ScratchOwned<BE>) -> &mut Scratch<BE> {
        Self::scratch_from_bytes_default(scratch.data.as_mut())
    }

    fn scratch_from_bytes_default(data: &mut [u8]) -> &mut Scratch<BE> {
        // SAFETY: `Scratch<BE>` is `#[repr(C)]` with layout `{ PhantomData<BE>, [u8] }`.
        // `PhantomData` is zero-sized, so the byte layout is identical to `[u8]`.
        unsafe { &mut *(data as *mut [u8] as *mut Scratch<BE>) }
    }

    fn scratch_available_default(scratch: &Scratch<BE>) -> usize {
        let ptr: *const u8 = scratch.data.as_ref().as_ptr();
        let self_len: usize = scratch.data.as_ref().len();
        let aligned_offset: usize = ptr.align_offset(DEFAULTALIGN);
        self_len.saturating_sub(aligned_offset)
    }

    fn take_slice_default<T>(scratch: &mut Scratch<BE>, len: usize) -> (&mut [T], &mut Scratch<BE>) {
        debug_assert!(
            DEFAULTALIGN.is_multiple_of(std::mem::align_of::<T>()),
            "DEFAULTALIGN ({DEFAULTALIGN}) must be a multiple of align_of::<T>() ({})",
            std::mem::align_of::<T>()
        );
        let (take_slice, rem_slice) = take_slice_aligned(scratch.data.as_mut(), len * std::mem::size_of::<T>());

        // SAFETY: `take_slice` is aligned to `DEFAULTALIGN` which is a multiple of
        // `align_of::<T>()` (asserted above). Length is `len * size_of::<T>()` bytes,
        // so reinterpreting as `[T; len]` is valid. The remainder is a disjoint sub-slice.
        unsafe {
            (
                &mut *(std::ptr::slice_from_raw_parts_mut(take_slice.as_mut_ptr() as *mut T, len)),
                Self::scratch_from_bytes_default(rem_slice),
            )
        }
    }
}

impl<BE: Backend> HalScratchDefaults<BE> for BE {}

fn take_slice_aligned(data: &mut [u8], take_len: usize) -> (&mut [u8], &mut [u8]) {
    let ptr: *mut u8 = data.as_mut_ptr();
    let self_len: usize = data.len();

    let aligned_offset: usize = ptr.align_offset(DEFAULTALIGN);
    let aligned_len: usize = self_len.saturating_sub(aligned_offset);

    if let Some(rem_len) = aligned_len.checked_sub(take_len) {
        // SAFETY: `aligned_offset + take_len <= self_len`, so both sub-slices are
        // within bounds. They are non-overlapping because `rem` starts immediately
        // after `take`. The original `data` borrow is split into two disjoint parts.
        unsafe {
            let rem_ptr: *mut u8 = ptr.add(aligned_offset).add(take_len);
            let rem_slice: &mut [u8] = &mut *std::ptr::slice_from_raw_parts_mut(rem_ptr, rem_len);

            let take_slice: &mut [u8] = &mut *std::ptr::slice_from_raw_parts_mut(ptr.add(aligned_offset), take_len);

            (take_slice, rem_slice)
        }
    } else {
        panic!("Attempted to take {take_len} from scratch with {aligned_len} aligned bytes left");
    }
}
