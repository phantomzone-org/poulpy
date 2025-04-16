use crate::ffi::module::MODULE;
use crate::ffi::vec_znx_big::VEC_ZNX_BIG;
use crate::ffi::vec_znx_dft::VEC_ZNX_DFT;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct vmp_pmat_t {
    _unused: [u8; 0],
}

// [rows][cols] = [#Decomposition][#Limbs]
pub type VMP_PMAT = vmp_pmat_t;

unsafe extern "C" {
    pub unsafe fn bytes_of_vmp_pmat(module: *const MODULE, nrows: u64, ncols: u64) -> u64;
}
unsafe extern "C" {
    pub unsafe fn new_vmp_pmat(module: *const MODULE, nrows: u64, ncols: u64) -> *mut VMP_PMAT;
}
unsafe extern "C" {
    pub unsafe fn delete_vmp_pmat(res: *mut VMP_PMAT);
}

unsafe extern "C" {
    pub unsafe fn vmp_apply_dft(
        module: *const MODULE,
        res: *mut VEC_ZNX_DFT,
        res_size: u64,
        a: *const i64,
        a_size: u64,
        a_sl: u64,
        pmat: *const VMP_PMAT,
        nrows: u64,
        ncols: u64,
        tmp_space: *mut u8,
    );
}

unsafe extern "C" {
    pub unsafe fn vmp_apply_dft_tmp_bytes(
        module: *const MODULE,
        res_size: u64,
        a_size: u64,
        nrows: u64,
        ncols: u64,
    ) -> u64;
}

unsafe extern "C" {
    pub unsafe fn vmp_apply_dft_to_dft(
        module: *const MODULE,
        res: *mut VEC_ZNX_DFT,
        res_size: u64,
        a_dft: *const VEC_ZNX_DFT,
        a_size: u64,
        pmat: *const VMP_PMAT,
        nrows: u64,
        ncols: u64,
        tmp_space: *mut u8,
    );
}

unsafe extern "C" {
    pub unsafe fn vmp_apply_dft_to_dft_tmp_bytes(
        module: *const MODULE,
        res_size: u64,
        a_size: u64,
        nrows: u64,
        ncols: u64,
    ) -> u64;
}

unsafe extern "C" {
    pub unsafe fn vmp_prepare_contiguous(
        module: *const MODULE,
        pmat: *mut VMP_PMAT,
        mat: *const i64,
        nrows: u64,
        ncols: u64,
        tmp_space: *mut u8,
    );
}

unsafe extern "C" {
    pub unsafe fn vmp_prepare_dblptr(
        module: *const MODULE,
        pmat: *mut VMP_PMAT,
        mat: *const *const i64,
        nrows: u64,
        ncols: u64,
        tmp_space: *mut u8,
    );
}

unsafe extern "C" {
    pub unsafe fn vmp_prepare_row(
        module: *const MODULE,
        pmat: *mut VMP_PMAT,
        row: *const i64,
        row_i: u64,
        nrows: u64,
        ncols: u64,
        tmp_space: *mut u8,
    );
}

unsafe extern "C" {
    pub unsafe fn vmp_prepare_row_dft(
        module: *const MODULE,
        pmat: *mut VMP_PMAT,
        row: *const VEC_ZNX_DFT,
        row_i: u64,
        nrows: u64,
        ncols: u64,
    );
}

unsafe extern "C" {
    pub unsafe fn vmp_extract_row_dft(
        module: *const MODULE,
        res: *mut VEC_ZNX_DFT,
        pmat: *const VMP_PMAT,
        row_i: u64,
        nrows: u64,
        ncols: u64,
    );
}

unsafe extern "C" {
    pub unsafe fn vmp_extract_row(
        module: *const MODULE,
        res: *mut VEC_ZNX_BIG,
        pmat: *const VMP_PMAT,
        row_i: u64,
        nrows: u64,
        ncols: u64,
    );
}

unsafe extern "C" {
    pub unsafe fn vmp_prepare_tmp_bytes(module: *const MODULE, nrows: u64, ncols: u64) -> u64;
}
