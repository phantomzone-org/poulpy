//! Backend handle and module initialization for [`FFT64Ref`](crate::FFT64Ref).
//!
//! This module defines:
//!
//! - [`FFT64RefHandle`]: the opaque handle stored inside a `Module<FFT64Ref>`,
//!   holding precomputed FFT and IFFT twiddle-factor tables.
//! - The [`Backend`] trait implementation, which defines scalar types and the
//!   handle destruction path.
//! - The [`FFT64HandleFactory`] implementation, which builds the handle stored
//!   inside the `Module`.
//! - The shared [`FFT64ModuleHandle`](crate::reference::fft64::module::FFT64ModuleHandle)
//!   trait from `poulpy-hal`, which provides typed access to the FFT tables from
//!   a `Module<FFT64Ref>` and other FFT64-family backends.

use std::ptr::NonNull;

use poulpy_hal::{alloc_aligned, assert_alignment, layouts::Backend};

use crate::reference::fft64::{
    module::{FFT64HandleFactory, FFTHandleProvider},
    reim::{ReimFFTTable, ReimIFFTTable},
};

use super::FFT64Ref;

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
    type OwnedBuf = Vec<u8>;
    type Handle = FFT64RefHandle;
    fn alloc_bytes(len: usize) -> Self::OwnedBuf {
        alloc_aligned::<u8>(len)
    }
    fn from_bytes(bytes: Vec<u8>) -> Self::OwnedBuf {
        assert_alignment(bytes.as_ptr());
        bytes
    }
    unsafe fn destroy(handle: NonNull<Self::Handle>) {
        unsafe {
            drop(Box::from_raw(handle.as_ptr()));
        }
    }
}

/// # Safety
///
/// The returned handle must be fully initialized for `n`.
unsafe impl FFT64HandleFactory for FFT64RefHandle {
    fn create_fft64_handle(n: usize) -> Self {
        FFT64RefHandle {
            table_fft: ReimFFTTable::new(n >> 1),
            table_ifft: ReimIFFTTable::new(n >> 1),
        }
    }
}

unsafe impl FFTHandleProvider<f64> for FFT64RefHandle {
    fn get_fft_table(&self) -> &ReimFFTTable<f64> {
        &self.table_fft
    }

    fn get_ifft_table(&self) -> &ReimIFFTTable<f64> {
        &self.table_ifft
    }
}
