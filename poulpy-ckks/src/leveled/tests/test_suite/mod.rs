use poulpy_core::layouts::{Base2K, Degree, Dnum, Dsize, GLWELayout, GLWETensorKeyLayout, Rank, TorusPrecision};

#[derive(Clone, Copy)]
pub struct CKKSTestParams {
    pub n: u32,
    pub base2k: u32,
    pub k: u32,
    pub log_delta: u32,
    pub hw: usize,
    pub dsize: u32,
}

impl CKKSTestParams {
    pub fn glwe_layout(&self) -> GLWELayout {
        GLWELayout {
            n: Degree(self.n),
            base2k: Base2K(self.base2k),
            k: TorusPrecision(self.k),
            rank: Rank(1),
        }
    }

    pub fn tsk_layout(&self) -> GLWETensorKeyLayout {
        let dnum = self.k.div_ceil(self.dsize * self.base2k);
        let k = self.k + self.dsize * self.base2k;
        GLWETensorKeyLayout {
            n: Degree(self.n),
            base2k: Base2K(self.base2k),
            k: TorusPrecision(k),
            rank: Rank(1),
            dsize: Dsize(self.dsize),
            dnum: Dnum(dnum),
        }
    }
}

pub const NTT120_PARAMS: CKKSTestParams = CKKSTestParams {
    n: 8192,
    base2k: 52,
    k: 8 * 52, // 1508 bits
    log_delta: 40,
    hw: 192,
    dsize: 1, // ceil(sqrt(29))
};

pub mod add;
pub mod encryption;
mod helpers;
pub mod level;
pub mod mul;
pub mod neg;
pub mod precision;
pub mod sub;
