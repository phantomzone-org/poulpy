//! Backend-generic CKKS test suite.
//!
//! All test functions are generic over `BE: Backend` and take a
//! [`TestContext`](helpers::TestContext) that owns the module, secret key,
//! and optional evaluation keys.  The backend-specific test harnesses
//! (`ntt120_ref`, `ntt120_avx`) instantiate and invoke these functions.

use poulpy_core::{
    EncryptionLayout,
    layouts::{GLWEAutomorphismKeyLayout, GLWELayout, GLWETensorKeyLayout, Rank},
};

use crate::CKKS;

/// Shared CKKS parameter set for test instantiation.
#[derive(Clone, Copy)]
pub struct CKKSTestParams {
    pub n: usize,
    pub base2k: usize,
    pub k: usize,
    pub prec: CKKS,
    pub hw: usize,
    pub dsize: usize,
}

impl CKKSTestParams {
    pub fn glwe_layout(&self) -> EncryptionLayout<GLWELayout> {
        EncryptionLayout::new_from_default_sigma(GLWELayout {
            n: self.n.into(),
            base2k: self.base2k.into(),
            k: self.k.into(),
            rank: Rank(1),
        })
        .unwrap()
    }

    pub fn tsk_layout(&self) -> EncryptionLayout<GLWETensorKeyLayout> {
        let k = self.k + self.dsize * self.base2k;
        let dnum = k.div_ceil(self.dsize * self.base2k);
        EncryptionLayout::new_from_default_sigma(GLWETensorKeyLayout {
            n: self.n.into(),
            base2k: self.base2k.into(),
            k: k.into(),
            rank: Rank(1),
            dsize: self.dsize.into(),
            dnum: dnum.into(),
        })
        .unwrap()
    }

    pub fn atk_layout(&self) -> EncryptionLayout<GLWEAutomorphismKeyLayout> {
        let k = self.k + self.dsize * self.base2k;
        let dnum = k.div_ceil(self.dsize * self.base2k);
        EncryptionLayout::new_from_default_sigma(GLWEAutomorphismKeyLayout {
            n: self.n.into(),
            base2k: self.base2k.into(),
            k: k.into(),
            rank: Rank(1),
            dsize: self.dsize.into(),
            dnum: dnum.into(),
        })
        .unwrap()
    }
}

/// NTT120 parameter set.
pub const NTT120_PARAMS: CKKSTestParams = CKKSTestParams {
    n: 256,
    base2k: 52,
    k: 8 * 52 + 1,
    prec: CKKS {
        log_decimal: 40,
        log_hom_rem: 30,
    },
    hw: 192,
    dsize: 1,
};

pub mod add;
//pub mod composition;
pub mod conjugate;
pub mod encryption;
pub mod errors;
pub mod helpers;
//pub mod level;
//pub mod metadata;
pub mod mul;
pub mod neg;
pub mod pow2;
pub mod rotate;
pub mod sub;
