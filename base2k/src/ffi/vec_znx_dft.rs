use crate::ffi::module::MODULE;
use crate::ffi::vec_znx_big::VEC_ZNX_BIG;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct vec_znx_dft_t {
    _unused: [u8; 0],
}
pub type VEC_ZNX_DFT = vec_znx_dft_t;

unsafe extern "C" {
    pub unsafe fn bytes_of_vec_znx_dft(module: *const MODULE, size: u64) -> u64;
}
unsafe extern "C" {
    pub unsafe fn new_vec_znx_dft(module: *const MODULE, size: u64) -> *mut VEC_ZNX_DFT;
}
unsafe extern "C" {
    pub unsafe fn delete_vec_znx_dft(res: *mut VEC_ZNX_DFT);
}

unsafe extern "C" {
    pub fn vec_dft_zero(module: *const MODULE, res: *mut VEC_ZNX_DFT, res_size: u64);
}
unsafe extern "C" {
    pub fn vec_dft_add(
        module: *const MODULE,
        res: *mut VEC_ZNX_DFT,
        res_size: u64,
        a: *const VEC_ZNX_DFT,
        a_size: u64,
        b: *const VEC_ZNX_DFT,
        b_size: u64,
    );
}
unsafe extern "C" {
    pub fn vec_dft_sub(
        module: *const MODULE,
        res: *mut VEC_ZNX_DFT,
        res_size: u64,
        a: *const VEC_ZNX_DFT,
        a_size: u64,
        b: *const VEC_ZNX_DFT,
        b_size: u64,
    );
}
unsafe extern "C" {
    pub fn vec_znx_dft(
        module: *const MODULE,
        res: *mut VEC_ZNX_DFT,
        res_size: u64,
        a: *const i64,
        a_size: u64,
        a_sl: u64,
    );
}
unsafe extern "C" {
    pub fn vec_znx_idft(
        module: *const MODULE,
        res: *mut VEC_ZNX_BIG,
        res_size: u64,
        a_dft: *const VEC_ZNX_DFT,
        a_size: u64,
        tmp: *mut u8,
    );
}
unsafe extern "C" {
    pub fn vec_znx_idft_tmp_bytes(module: *const MODULE) -> u64;
}
unsafe extern "C" {
    pub fn vec_znx_idft_tmp_a(
        module: *const MODULE,
        res: *mut VEC_ZNX_BIG,
        res_size: u64,
        a_dft: *mut VEC_ZNX_DFT,
        a_size: u64,
    );
}
