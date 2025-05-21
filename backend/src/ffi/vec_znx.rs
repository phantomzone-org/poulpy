use crate::ffi::module::MODULE;

unsafe extern "C" {
    pub unsafe fn vec_znx_add(
        module: *const MODULE,
        res: *mut i64,
        res_size: u64,
        res_sl: u64,
        a: *const i64,
        a_size: u64,
        a_sl: u64,
        b: *const i64,
        b_size: u64,
        b_sl: u64,
    );
}

unsafe extern "C" {
    pub unsafe fn vec_znx_automorphism(
        module: *const MODULE,
        p: i64,
        res: *mut i64,
        res_size: u64,
        res_sl: u64,
        a: *const i64,
        a_size: u64,
        a_sl: u64,
    );
}

unsafe extern "C" {
    pub unsafe fn vec_znx_negate(
        module: *const MODULE,
        res: *mut i64,
        res_size: u64,
        res_sl: u64,
        a: *const i64,
        a_size: u64,
        a_sl: u64,
    );
}

unsafe extern "C" {
    pub unsafe fn vec_znx_rotate(
        module: *const MODULE,
        p: i64,
        res: *mut i64,
        res_size: u64,
        res_sl: u64,
        a: *const i64,
        a_size: u64,
        a_sl: u64,
    );
}

unsafe extern "C" {
    pub unsafe fn vec_znx_sub(
        module: *const MODULE,
        res: *mut i64,
        res_size: u64,
        res_sl: u64,
        a: *const i64,
        a_size: u64,
        a_sl: u64,
        b: *const i64,
        b_size: u64,
        b_sl: u64,
    );
}

unsafe extern "C" {
    pub unsafe fn vec_znx_zero(module: *const MODULE, res: *mut i64, res_size: u64, res_sl: u64);
}
unsafe extern "C" {
    pub unsafe fn vec_znx_copy(
        module: *const MODULE,
        res: *mut i64,
        res_size: u64,
        res_sl: u64,
        a: *const i64,
        a_size: u64,
        a_sl: u64,
    );
}

unsafe extern "C" {
    pub unsafe fn vec_znx_normalize_base2k(
        module: *const MODULE,
        base2k: u64,
        res: *mut i64,
        res_size: u64,
        res_sl: u64,
        a: *const i64,
        a_size: u64,
        a_sl: u64,
        tmp_space: *mut u8,
    );
}
unsafe extern "C" {
    pub unsafe fn vec_znx_normalize_base2k_tmp_bytes(module: *const MODULE) -> u64;
}
