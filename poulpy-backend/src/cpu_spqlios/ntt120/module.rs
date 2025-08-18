use std::ptr::NonNull;

use poulpy_hal::{
    layouts::{Backend, Module},
    oep::ModuleNewImpl,
};

use crate::cpu_spqlios::ffi::module::{MODULE, delete_module_info, new_module_info};

pub struct NTT120;

impl Backend for NTT120 {
    type ScalarPrep = i64;
    type ScalarBig = i128;
    type Handle = MODULE;
    unsafe fn destroy(handle: NonNull<Self::Handle>) {
        unsafe { delete_module_info(handle.as_ptr()) }
    }

    fn layout_big_word_count() -> usize {
        4
    }

    fn layout_prep_word_count() -> usize {
        1
    }
}

unsafe impl ModuleNewImpl<Self> for NTT120 {
    fn new_impl(n: u64) -> Module<Self> {
        unsafe { Module::from_raw_parts(new_module_info(n, 1), n) }
    }
}
