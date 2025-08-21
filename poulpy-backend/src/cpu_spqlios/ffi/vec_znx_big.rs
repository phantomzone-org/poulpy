use crate::cpu_spqlios::ffi::module::MODULE;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct vec_znx_big_t {
    _unused: [u8; 0],
}
pub type VEC_ZNX_BIG = vec_znx_big_t;

unsafe extern "C" {
    pub unsafe fn vec_znx_big_add(
        module: *const MODULE,
        res: *mut VEC_ZNX_BIG,
        res_size: u64,
        a: *const VEC_ZNX_BIG,
        a_size: u64,
        b: *const VEC_ZNX_BIG,
        b_size: u64,
    );
}
unsafe extern "C" {
    pub unsafe fn vec_znx_big_add_small(
        module: *const MODULE,
        res: *mut VEC_ZNX_BIG,
        res_size: u64,
        a: *const VEC_ZNX_BIG,
        a_size: u64,
        b: *const i64,
        b_size: u64,
        b_sl: u64,
    );
}
unsafe extern "C" {
    pub unsafe fn vec_znx_big_add_small2(
        module: *const MODULE,
        res: *mut VEC_ZNX_BIG,
        res_size: u64,
        a: *const i64,
        a_size: u64,
        a_sl: u64,
        b: *const i64,
        b_size: u64,
        b_sl: u64,
    );
}
unsafe extern "C" {
    pub unsafe fn vec_znx_big_sub(
        module: *const MODULE,
        res: *mut VEC_ZNX_BIG,
        res_size: u64,
        a: *const VEC_ZNX_BIG,
        a_size: u64,
        b: *const VEC_ZNX_BIG,
        b_size: u64,
    );
}
unsafe extern "C" {
    pub unsafe fn vec_znx_big_sub_small_b(
        module: *const MODULE,
        res: *mut VEC_ZNX_BIG,
        res_size: u64,
        a: *const VEC_ZNX_BIG,
        a_size: u64,
        b: *const i64,
        b_size: u64,
        b_sl: u64,
    );
}
unsafe extern "C" {
    pub unsafe fn vec_znx_big_sub_small_a(
        module: *const MODULE,
        res: *mut VEC_ZNX_BIG,
        res_size: u64,
        a: *const i64,
        a_size: u64,
        a_sl: u64,
        b: *const VEC_ZNX_BIG,
        b_size: u64,
    );
}
unsafe extern "C" {
    pub unsafe fn vec_znx_big_sub_small2(
        module: *const MODULE,
        res: *mut VEC_ZNX_BIG,
        res_size: u64,
        a: *const i64,
        a_size: u64,
        a_sl: u64,
        b: *const i64,
        b_size: u64,
        b_sl: u64,
    );
}

unsafe extern "C" {
    pub unsafe fn vec_znx_big_normalize_base2k_tmp_bytes(module: *const MODULE) -> u64;
}

unsafe extern "C" {
    pub unsafe fn vec_znx_big_normalize_base2k(
        module: *const MODULE,
        log2_base2k: u64,
        res: *mut i64,
        res_size: u64,
        res_sl: u64,
        a: *const VEC_ZNX_BIG,
        a_size: u64,
        tmp_space: *mut u8,
    );
}

unsafe extern "C" {
    pub unsafe fn vec_znx_big_range_normalize_base2k(
        module: *const MODULE,
        log2_base2k: u64,
        res: *mut i64,
        res_size: u64,
        res_sl: u64,
        a: *const VEC_ZNX_BIG,
        a_range_begin: u64,
        a_range_xend: u64,
        a_range_step: u64,
        tmp_space: *mut u8,
    );
}

unsafe extern "C" {
    pub unsafe fn vec_znx_big_range_normalize_base2k_tmp_bytes(module: *const MODULE) -> u64;
}

unsafe extern "C" {
    pub unsafe fn vec_znx_big_automorphism(
        module: *const MODULE,
        p: i64,
        res: *mut VEC_ZNX_BIG,
        res_size: u64,
        a: *const VEC_ZNX_BIG,
        a_size: u64,
    );
}

unsafe extern "C" {
    pub unsafe fn vec_znx_big_rotate(
        module: *const MODULE,
        p: i64,
        res: *mut VEC_ZNX_BIG,
        res_size: u64,
        a: *const VEC_ZNX_BIG,
        a_size: u64,
    );
}
