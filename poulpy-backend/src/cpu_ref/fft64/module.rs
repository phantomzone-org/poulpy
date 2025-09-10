use std::ptr::NonNull;

use poulpy_hal::{
    layouts::{Backend, Module},
    oep::ModuleNewImpl,
    reference::reim::{ReimFFTTable, ReimIFFTTable},
};

pub struct FFT64 {}

#[repr(C)]
pub struct FFT64Handle {
    table_fft: ReimFFTTable<f64>,
    table_ifft: ReimIFFTTable<f64>,
}

impl Backend for FFT64 {
    type ScalarPrep = f64;
    type ScalarBig = i64;
    type Handle = FFT64Handle;
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

unsafe impl ModuleNewImpl<Self> for FFT64 {
    fn new_impl(n: u64) -> Module<Self> {
        let handle: FFT64Handle = FFT64Handle {
            table_fft: ReimFFTTable::new(n as usize >> 1),
            table_ifft: ReimIFFTTable::new(n as usize >> 1),
        };
        // Leak Box to get a stable NonNull pointer
        let ptr: NonNull<FFT64Handle> = NonNull::from(Box::leak(Box::new(handle)));
        unsafe { Module::from_nonnull(ptr, n) }
    }
}

pub trait FFT64ModuleHandle {
    fn get_fft_table(&self) -> &ReimFFTTable<f64>;
    fn get_ifft_table(&self) -> &ReimIFFTTable<f64>;
}

impl FFT64ModuleHandle for Module<FFT64> {
    fn get_fft_table(&self) -> &ReimFFTTable<f64> {
        let h: &FFT64Handle = unsafe { &*self.ptr() };
        &h.table_fft
    }
    fn get_ifft_table(&self) -> &ReimIFFTTable<f64> {
        let h: &FFT64Handle = unsafe { &*self.ptr() };
        &h.table_ifft
    }
}
