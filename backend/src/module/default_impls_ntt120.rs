use std::ptr::NonNull;

use crate::{
    Backend, NTT120,
    ffi::module::{MODULE, delete_module_info},
};

impl Backend for NTT120 {
    type Handle = MODULE;
    fn module_type() -> u32 {
        1
    }
    unsafe fn destroy(handle: NonNull<Self::Handle>) {
        unsafe { delete_module_info(handle.as_ptr()) }
    }
}
