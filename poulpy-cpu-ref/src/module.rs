//! Backend handle and module initialization for [`FFT64Ref`](crate::FFT64Ref).
//!
//! This module defines:
//!
//! - [`FFT64RefHandle`]: the opaque handle stored inside a `Module<FFT64Ref>`,
//!   holding precomputed FFT and IFFT twiddle-factor tables.
//! - The [`Backend`] trait implementation, which defines scalar types and the
//!   handle destruction path.
//! - The [`ModuleNewImpl`] implementation, which allocates the handle on the heap
//!   and transfers ownership to the `Module`.
//! - The [`FFT64ModuleHandle`] trait, which provides typed access to the FFT tables
//!   from a `Module<FFT64Ref>`. This trait is also implemented by other FFT64 backends
//!   (e.g. AVX), enabling shared downstream code.

use std::ptr::NonNull;

use poulpy_hal::{
    layouts::{Backend, Module},
    oep::ModuleNewImpl,
    reference::fft64::reim::{ReimFFTTable, ReimIFFTTable},
};

use crate::FFT64Ref;

/// Opaque handle for the [`FFT64Ref`](crate::FFT64Ref) backend.
///
/// Holds precomputed twiddle-factor tables for the forward FFT and inverse FFT
/// of size `m = n / 2`, where `n` is the ring dimension passed to
/// [`Module::new`](poulpy_hal::api::ModuleNew::new).
///
/// This struct is heap-allocated during module creation and freed when the
/// `Module<FFT64Ref>` is dropped (via [`Backend::destroy`]).
#[repr(C)]
pub struct FFT64RefHandle {
    table_fft: ReimFFTTable<f64>,
    table_ifft: ReimIFFTTable<f64>,
}

impl Backend for FFT64Ref {
    type ScalarPrep = f64;
    type ScalarBig = i64;
    type Handle = FFT64RefHandle;
    unsafe fn destroy(handle: NonNull<Self::Handle>) {
        unsafe {
            drop(Box::from_raw(handle.as_ptr()));
        }
    }

    fn layout_big_word_count() -> usize {
        1
    }

    fn layout_prep_word_count() -> usize {
        1
    }
}

/// # Safety
///
/// The returned `Module` owns the heap-allocated `FFT64RefHandle`.
/// `n` must be a power of two >= 2 (asserted by `Module::from_nonnull`).
/// The FFT tables are built for dimension `m = n / 2`.
unsafe impl ModuleNewImpl<Self> for FFT64Ref {
    fn new_impl(n: u64) -> Module<Self> {
        let handle: FFT64RefHandle = FFT64RefHandle {
            table_fft: ReimFFTTable::new(n as usize >> 1),
            table_ifft: ReimIFFTTable::new(n as usize >> 1),
        };
        // Leak Box to get a stable NonNull pointer
        let ptr: NonNull<FFT64RefHandle> = NonNull::from(Box::leak(Box::new(handle)));
        unsafe { Module::from_nonnull(ptr, n) }
    }
}

/// Provides access to the precomputed FFT and IFFT twiddle-factor tables
/// stored inside a module handle.
///
/// This trait is implemented by `Module<FFT64Ref>` (this crate) and also by
/// `Module<FFT64Avx>` in `poulpy-cpu-avx`, allowing shared code in the
/// reference functions to work across both backends.
pub trait FFT64ModuleHandle {
    /// Returns a reference to the forward-FFT twiddle-factor table.
    fn get_fft_table(&self) -> &ReimFFTTable<f64>;
    /// Returns a reference to the inverse-FFT twiddle-factor table.
    fn get_ifft_table(&self) -> &ReimIFFTTable<f64>;
}

impl FFT64ModuleHandle for Module<FFT64Ref> {
    fn get_fft_table(&self) -> &ReimFFTTable<f64> {
        // SAFETY: `self.ptr()` returns a valid pointer to `FFT64RefHandle`,
        // which was heap-allocated in `new_impl` and is kept alive by the Module.
        let h: &FFT64RefHandle = unsafe { &*self.ptr() };
        &h.table_fft
    }
    fn get_ifft_table(&self) -> &ReimIFFTTable<f64> {
        // SAFETY: same as above.
        let h: &FFT64RefHandle = unsafe { &*self.ptr() };
        &h.table_ifft
    }
}
