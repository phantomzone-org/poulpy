use std::ptr::NonNull;

use crate::{
    hal::{
        layouts::{Backend, Module},
        oep::ModuleNewImpl,
    },
    implementation::cpu_avx::ffi::module::{MODULE, delete_module_info, new_module_info},
};

pub struct FFT64;

impl Backend for FFT64 {
    type Handle = MODULE;
    unsafe fn destroy(handle: NonNull<Self::Handle>) {
        unsafe { delete_module_info(handle.as_ptr()) }
    }
}

unsafe impl ModuleNewImpl<Self> for FFT64 {
    fn new_impl(n: u64) -> Module<Self> {
        unsafe { Module::from_raw_parts(new_module_info(n, 0), n) }
    }
}
