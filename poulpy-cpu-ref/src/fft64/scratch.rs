//! Scratch (temporary) memory management for [`FFT64Ref`](crate::FFT64Ref).
//!
//! Implements the OEP scratch traits: allocation of 64-byte-aligned owned buffers,
//! borrowing as `&mut Scratch<B>`, arena-style sub-allocation via `take_slice`, and
//! available-space queries.
//!
//! # Memory layout
//!
//! All scratch buffers are allocated with 64-byte alignment ([`DEFAULTALIGN`]).
//! The `take_slice_impl` method carves typed slices from the front of the aligned region
//! and returns the remainder as a new `Scratch` reference. The remainder may not itself
//! be aligned; the next `take_slice` call will re-align internally, potentially wasting
//! up to `DEFAULTALIGN - 1` bytes.
//!
//! # Safety
//!
//! The `scratch_from_bytes_impl` reinterprets a `&mut [u8]` as `&mut Scratch<B>`.
//! This relies on `Scratch<B>` being `#[repr(C)]` with a leading zero-sized
//! `PhantomData<B>` followed by a `[u8]` DST — guaranteed by `poulpy-hal`.

use std::marker::PhantomData;

use poulpy_hal::{
    DEFAULTALIGN, alloc_aligned,
    api::ScratchFromBytes,
    layouts::{Backend, Scratch, ScratchOwned},
    oep::{ScratchAvailableImpl, ScratchFromBytesImpl, ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeSliceImpl},
};

use super::FFT64Ref;

unsafe impl<B: Backend> ScratchOwnedAllocImpl<B> for FFT64Ref {
    fn scratch_owned_alloc_impl(size: usize) -> ScratchOwned<B> {
        let data: Vec<u8> = alloc_aligned(size);
        ScratchOwned {
            data,
            _phantom: PhantomData,
        }
    }
}

unsafe impl<B: Backend> ScratchOwnedBorrowImpl<B> for FFT64Ref
where
    B: ScratchFromBytesImpl<B>,
{
    fn scratch_owned_borrow_impl(scratch: &mut ScratchOwned<B>) -> &mut Scratch<B> {
        Scratch::from_bytes(&mut scratch.data)
    }
}

unsafe impl<B: Backend> ScratchFromBytesImpl<B> for FFT64Ref {
    fn scratch_from_bytes_impl(data: &mut [u8]) -> &mut Scratch<B> {
        // SAFETY: `Scratch<B>` is `#[repr(C)]` with layout `{ PhantomData<B>, [u8] }`.
        // `PhantomData` is zero-sized, so the byte layout is identical to `[u8]`.
        unsafe { &mut *(data as *mut [u8] as *mut Scratch<B>) }
    }
}

unsafe impl<B: Backend> ScratchAvailableImpl<B> for FFT64Ref {
    fn scratch_available_impl(scratch: &Scratch<B>) -> usize {
        let ptr: *const u8 = scratch.data.as_ptr();
        let self_len: usize = scratch.data.len();
        let aligned_offset: usize = ptr.align_offset(DEFAULTALIGN);
        self_len.saturating_sub(aligned_offset)
    }
}

unsafe impl<B: Backend> TakeSliceImpl<B> for FFT64Ref
where
    B: ScratchFromBytesImpl<B>,
{
    fn take_slice_impl<T>(scratch: &mut Scratch<B>, len: usize) -> (&mut [T], &mut Scratch<B>) {
        debug_assert!(
            DEFAULTALIGN.is_multiple_of(std::mem::align_of::<T>()),
            "DEFAULTALIGN ({DEFAULTALIGN}) must be a multiple of align_of::<T>() ({})",
            std::mem::align_of::<T>()
        );
        let (take_slice, rem_slice) = take_slice_aligned(&mut scratch.data, len * std::mem::size_of::<T>());

        // SAFETY: `take_slice` is aligned to `DEFAULTALIGN` which is a multiple of
        // `align_of::<T>()` (asserted above). Length is `len * size_of::<T>()` bytes,
        // so reinterpreting as `[T; len]` is valid. The remainder is a disjoint sub-slice.
        unsafe {
            (
                &mut *(std::ptr::slice_from_raw_parts_mut(take_slice.as_mut_ptr() as *mut T, len)),
                Scratch::from_bytes(rem_slice),
            )
        }
    }
}

/// Splits `data` into an aligned prefix of `take_len` bytes and an unaligned remainder.
///
/// The returned prefix starts at the first `DEFAULTALIGN`-aligned address within `data`.
///
/// # Panics
///
/// Panics if the aligned region of `data` is smaller than `take_len`.
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
