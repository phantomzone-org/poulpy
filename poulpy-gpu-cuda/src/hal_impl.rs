//! Narrow HAL-family implementations owned by `poulpy-gpu-cuda`.
//!
//! This file is intentionally small. It documents the recommended pattern for
//! an external device backend:
//! - keep the impl surface narrow
//! - avoid inheriting host-only CPU defaults when the backend is device-only

use std::{marker::PhantomData, ptr::NonNull};

use crate::CudaGpuBackend;
use poulpy_cpu_ref::reference::fft64::module::FFT64HandleFactory;
use poulpy_hal::{
    DEFAULTALIGN,
    layouts::{Module, Scratch, ScratchOwned},
    oep::{HalModuleImpl, HalScratchImpl},
};

unsafe impl HalScratchImpl<CudaGpuBackend> for CudaGpuBackend {
    fn scratch_owned_alloc(size: usize) -> ScratchOwned<CudaGpuBackend> {
        ScratchOwned {
            data: <CudaGpuBackend as poulpy_hal::layouts::Backend>::alloc_bytes(size),
            _phantom: PhantomData,
        }
    }

    fn scratch_owned_borrow(scratch: &mut ScratchOwned<CudaGpuBackend>) -> &mut Scratch<CudaGpuBackend> {
        let _ = scratch;
        panic!("legacy Scratch borrowing is not supported for the pure device-only CUDA backend; use ScratchArena instead")
    }

    fn scratch_from_bytes(data: &mut [u8]) -> &mut Scratch<CudaGpuBackend> {
        unsafe { &mut *(data as *mut [u8] as *mut Scratch<CudaGpuBackend>) }
    }

    fn scratch_available(scratch: &Scratch<CudaGpuBackend>) -> usize {
        let ptr: *const u8 = scratch.data.as_ptr();
        let self_len: usize = scratch.data.len();
        let aligned_offset: usize = ptr.align_offset(DEFAULTALIGN);
        self_len.saturating_sub(aligned_offset)
    }

    fn take_slice<T>(scratch: &mut Scratch<CudaGpuBackend>, len: usize) -> (&mut [T], &mut Scratch<CudaGpuBackend>) {
        debug_assert!(
            DEFAULTALIGN.is_multiple_of(std::mem::align_of::<T>()),
            "DEFAULTALIGN ({DEFAULTALIGN}) must be a multiple of align_of::<T>() ({})",
            std::mem::align_of::<T>()
        );

        let data: &mut [u8] = &mut scratch.data;
        let ptr: *mut u8 = data.as_mut_ptr();
        let self_len: usize = data.len();
        let aligned_offset: usize = ptr.align_offset(DEFAULTALIGN);
        let aligned_len: usize = self_len.saturating_sub(aligned_offset);

        if let Some(rem_len) = aligned_len.checked_sub(len * std::mem::size_of::<T>()) {
            unsafe {
                let take_ptr = ptr.add(aligned_offset);
                let rem_ptr = take_ptr.add(len * std::mem::size_of::<T>());
                let take_slice = &mut *(std::ptr::slice_from_raw_parts_mut(take_ptr as *mut T, len));
                let rem_slice = &mut *(std::ptr::slice_from_raw_parts_mut(rem_ptr, rem_len) as *mut Scratch<CudaGpuBackend>);
                (take_slice, rem_slice)
            }
        } else {
            panic!(
                "Attempted to take {} from scratch with {aligned_len} aligned bytes left",
                len * std::mem::size_of::<T>()
            );
        }
    }
}

unsafe impl HalModuleImpl<CudaGpuBackend> for CudaGpuBackend {
    fn new(n: u64) -> Module<CudaGpuBackend> {
        <crate::CudaFft64Handle as FFT64HandleFactory>::assert_fft64_runtime_support();
        let handle = <crate::CudaFft64Handle as FFT64HandleFactory>::create_fft64_handle(n as usize);
        let ptr: NonNull<crate::CudaFft64Handle> = NonNull::from(Box::leak(Box::new(handle)));
        unsafe { Module::from_nonnull(ptr, n) }
    }
}
