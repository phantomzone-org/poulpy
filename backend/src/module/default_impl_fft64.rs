use std::ptr::NonNull;

use crate::{
    Backend, FFT64, Module, ModuleNewImpl,
    ffi::module::{MODULE, delete_module_info, new_module_info},
};

impl Backend for FFT64 {
    type Handle = MODULE;
    fn module_type() -> u32 {
        0
    }
    unsafe fn destroy(handle: NonNull<Self::Handle>) {
        unsafe { delete_module_info(handle.as_ptr()) }
    }
}

unsafe impl ModuleNewImpl<Self> for FFT64 {
    fn new_impl(n: u64) -> Module<Self> {
        unsafe { Module::from_raw_parts(new_module_info(n, FFT64::module_type()), n) }
    }
}
