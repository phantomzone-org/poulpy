use crate::ffi::module::MODULE;
use crate::ffi::vec_znx_dft::VEC_ZNX_DFT;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct svp_ppol_t {
    _unused: [u8; 0],
}
pub type SVP_PPOL = svp_ppol_t;

unsafe extern "C" {
    pub unsafe fn bytes_of_svp_ppol(module: *const MODULE) -> u64;
}
unsafe extern "C" {
    pub unsafe fn new_svp_ppol(module: *const MODULE) -> *mut SVP_PPOL;
}
unsafe extern "C" {
    pub unsafe fn delete_svp_ppol(res: *mut SVP_PPOL);
}

unsafe extern "C" {
    pub unsafe fn svp_prepare(module: *const MODULE, ppol: *mut SVP_PPOL, pol: *const i64);
}

unsafe extern "C" {
    pub unsafe fn svp_apply_dft(
        module: *const MODULE,
        res: *const VEC_ZNX_DFT,
        res_size: u64,
        ppol: *const SVP_PPOL,
        a: *const i64,
        a_size: u64,
        a_sl: u64,
    );
}
