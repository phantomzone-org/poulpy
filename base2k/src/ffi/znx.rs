use crate::ffi::module::MODULE;

unsafe extern "C" {
    pub unsafe fn znx_add_i64_ref(nn: u64, res: *mut i64, a: *const i64, b: *const i64);
}
unsafe extern "C" {
    pub unsafe fn znx_add_i64_avx(nn: u64, res: *mut i64, a: *const i64, b: *const i64);
}
unsafe extern "C" {
    pub unsafe fn znx_sub_i64_ref(nn: u64, res: *mut i64, a: *const i64, b: *const i64);
}
unsafe extern "C" {
    pub unsafe fn znx_sub_i64_avx(nn: u64, res: *mut i64, a: *const i64, b: *const i64);
}
unsafe extern "C" {
    pub unsafe fn znx_negate_i64_ref(nn: u64, res: *mut i64, a: *const i64);
}
unsafe extern "C" {
    pub unsafe fn znx_negate_i64_avx(nn: u64, res: *mut i64, a: *const i64);
}
unsafe extern "C" {
    pub unsafe fn znx_copy_i64_ref(nn: u64, res: *mut i64, a: *const i64);
}
unsafe extern "C" {
    pub unsafe fn znx_zero_i64_ref(nn: u64, res: *mut i64);
}
unsafe extern "C" {
    pub unsafe fn rnx_divide_by_m_ref(nn: u64, m: f64, res: *mut f64, a: *const f64);
}
unsafe extern "C" {
    pub unsafe fn rnx_divide_by_m_avx(nn: u64, m: f64, res: *mut f64, a: *const f64);
}
unsafe extern "C" {
    pub unsafe fn rnx_rotate_f64(nn: u64, p: i64, res: *mut f64, in_: *const f64);
}
unsafe extern "C" {
    pub unsafe fn znx_rotate_i64(nn: u64, p: i64, res: *mut i64, in_: *const i64);
}
unsafe extern "C" {
    pub unsafe fn rnx_rotate_inplace_f64(nn: u64, p: i64, res: *mut f64);
}
unsafe extern "C" {
    pub unsafe fn znx_rotate_inplace_i64(nn: u64, p: i64, res: *mut i64);
}
unsafe extern "C" {
    pub unsafe fn rnx_automorphism_f64(nn: u64, p: i64, res: *mut f64, in_: *const f64);
}
unsafe extern "C" {
    pub unsafe fn znx_automorphism_i64(nn: u64, p: i64, res: *mut i64, in_: *const i64);
}
unsafe extern "C" {
    pub unsafe fn rnx_automorphism_inplace_f64(nn: u64, p: i64, res: *mut f64);
}
unsafe extern "C" {
    pub unsafe fn znx_automorphism_inplace_i64(nn: u64, p: i64, res: *mut i64);
}
unsafe extern "C" {
    pub unsafe fn rnx_mul_xp_minus_one(nn: u64, p: i64, res: *mut f64, in_: *const f64);
}
unsafe extern "C" {
    pub unsafe fn znx_mul_xp_minus_one(nn: u64, p: i64, res: *mut i64, in_: *const i64);
}
unsafe extern "C" {
    pub unsafe fn rnx_mul_xp_minus_one_inplace(nn: u64, p: i64, res: *mut f64);
}
unsafe extern "C" {
    pub unsafe fn znx_normalize(
        nn: u64,
        base_k: u64,
        out: *mut i64,
        carry_out: *mut i64,
        in_: *const i64,
        carry_in: *const i64,
    );
}

unsafe extern "C" {
    pub unsafe fn znx_small_single_product(
        module: *const MODULE,
        res: *mut i64,
        a: *const i64,
        b: *const i64,
        tmp: *mut u8,
    );
}

unsafe extern "C" {
    pub unsafe fn znx_small_single_product_tmp_bytes(module: *const MODULE) -> u64;
}
