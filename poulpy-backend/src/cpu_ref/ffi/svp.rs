use crate::cpu_ref::ffi::{module::MODULE, vec_znx_dft::VEC_ZNX_DFT};

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct svp_ppol_t {
    _unused: [u8; 0],
}
pub type SVP_PPOL = svp_ppol_t;

unsafe extern "C" {
    pub unsafe fn svp_prepare(module: *const MODULE, ppol: *mut SVP_PPOL, pol: *const i64);
}

#[allow(dead_code)]
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

unsafe extern "C" {
    pub unsafe fn svp_apply_dft_to_dft(
        module: *const MODULE,
        res: *const VEC_ZNX_DFT,
        res_size: u64,
        res_cols: u64,
        ppol: *const SVP_PPOL,
        a: *const VEC_ZNX_DFT,
        a_size: u64,
        a_cols: u64,
    );
}
