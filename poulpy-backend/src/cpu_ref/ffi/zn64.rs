unsafe extern "C" {
    pub unsafe fn zn64_normalize_base2k_ref(
        n: u64,
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
