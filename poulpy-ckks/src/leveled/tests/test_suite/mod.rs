#[derive(Clone, Copy)]
pub struct CKKSTestParams {
    pub base2k: u32,
    pub k: u32,
    pub log_delta: u32,
    pub hw: usize,
}

pub const NTT120_PARAMS: CKKSTestParams = CKKSTestParams {
    base2k: 52,
    k: 17 * 52,
    log_delta: 40,
    hw: 192,
};

pub const FFT64_PARAMS: CKKSTestParams = CKKSTestParams {
    base2k: 19,
    k: 46 * 19,
    log_delta: 40,
    hw: 192,
};

pub mod arithmetic;
pub mod drop_level;
pub mod encrypt_decrypt;
