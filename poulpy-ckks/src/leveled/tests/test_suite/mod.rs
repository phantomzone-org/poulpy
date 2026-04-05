//! Backend-generic CKKS test suite.
//!
//! All test functions are generic over `BE: Backend` and take a
//! [`TestContext`](helpers::TestContext) that owns the module, secret key,
//! and optional evaluation keys.  The backend-specific test harnesses
//! (`ntt120_ref`, `ntt120_avx`) instantiate and invoke these functions.

use poulpy_core::layouts::{
    Base2K, Degree, Dnum, Dsize, GLWEAutomorphismKeyLayout, GLWELayout, GLWETensorKeyLayout, Rank, TorusPrecision,
};

/// Shared CKKS parameter set for test instantiation.
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

    pub fn atk_layout(&self) -> GLWEAutomorphismKeyLayout {
        let dnum = self.k.div_ceil(self.dsize * self.base2k);
        let k = self.k + self.dsize * self.base2k;
        GLWEAutomorphismKeyLayout {
            n: Degree(self.n),
            base2k: Base2K(self.base2k),
            k: TorusPrecision(k),
            rank: Rank(1),
            dsize: Dsize(self.dsize),
            dnum: Dnum(dnum),
        }
    }
}

/// NTT120 parameter set.
pub const NTT120_PARAMS: CKKSTestParams = CKKSTestParams {
    n: 8192,
    base2k: 52,
    k: 8 * 52, // 416
    log_delta: 40,
    hw: 192,
    dsize: 1,
};

pub mod add;
pub mod align;
pub mod composition;
pub mod conjugate;
pub mod encryption;
pub mod helpers;
pub mod level;
pub mod metadata;
pub mod mul;
pub mod neg;
pub mod plaintext_prepared;
pub mod rotate;
pub mod sub;
