pub struct module_info_t {
    _unused: [u8; 0],
}

pub type module_type_t = ::std::os::raw::c_uint;
pub use self::module_type_t as MODULE_TYPE;

pub type MODULE = module_info_t;

unsafe extern "C" {
    pub unsafe fn new_module_info(N: u64, mode: MODULE_TYPE) -> *mut MODULE;
}
unsafe extern "C" {
    pub unsafe fn delete_module_info(module_info: *mut MODULE);
}
unsafe extern "C" {
    pub unsafe fn module_get_n(module: *const MODULE) -> u64;
}
