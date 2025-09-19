use std::ptr::NonNull;

use poulpy_hal::{
    layouts::{Backend, Module},
    oep::ModuleNewImpl,
};

use crate::cpu_spqlios::{
    FFT64Spqlios,
    ffi::module::{MODULE, delete_module_info, new_module_info},
};

impl Backend for FFT64Spqlios {
    type ScalarPrep = f64;
    type ScalarBig = i64;
    type Handle = MODULE;
    unsafe fn destroy(handle: NonNull<Self::Handle>) {
        unsafe { delete_module_info(handle.as_ptr()) }
    }

    fn layout_big_word_count() -> usize {
        1
    }

    fn layout_prep_word_count() -> usize {
        1
    }
}

unsafe impl ModuleNewImpl<Self> for FFT64Spqlios {
    fn new_impl(n: u64) -> Module<Self> {
        unsafe { Module::from_raw_parts(new_module_info(n, 0), n) }
    }
}
