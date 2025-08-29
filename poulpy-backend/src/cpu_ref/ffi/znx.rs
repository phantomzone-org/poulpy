unsafe extern "C" {
    pub unsafe fn znx_rotate_i64(nn: u64, p: i64, res: *mut i64, in_: *const i64);
}

unsafe extern "C" {
    pub unsafe fn znx_rotate_inplace_i64(nn: u64, p: i64, res: *mut i64);
}
