#[derive(Clone, Copy)]
pub struct CKKSTestParams {
    pub n: u32,
    pub base2k: u32,
    pub k: u32,
    pub log_delta: u32,
    pub hw: usize,
}

pub const NTT120_PARAMS: CKKSTestParams = CKKSTestParams {
    n: 65536,
    base2k: 52,
    k: 29 * 52, // 1508 bits
    log_delta: 40,
    hw: 192,
};

pub mod add;
pub mod encryption;
mod helpers;
pub mod level;
pub mod mul;
pub mod neg;
pub mod precision;
pub mod sub;
