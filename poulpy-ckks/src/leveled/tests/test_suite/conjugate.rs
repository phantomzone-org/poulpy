//! Conjugation tests (out-of-place and in-place).
//!
//! # Test inventory
//!
//! ## Operations-layer conjugation (`CKKSCiphertext::conjugate`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_conjugate`] | out-of-place conjugation |
//!
//! ## Operations-layer conjugation (`CKKSCiphertext::conjugate_inplace`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_conjugate_inplace`] | in-place conjugation |

use crate::layouts::PrecisionInfos;

use super::helpers::TestContext;
use poulpy_core::{GLWEAutomorphism, GLWEDecrypt, GLWEEncryptSk, ScratchTakeCore, layouts::GLWESecretPreparedFactory};
use poulpy_hal::{
    api::{
        ModuleN, ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxCopy, VecZnxLsh, VecZnxLshInplace,
        VecZnxNormalize, VecZnxNormalizeTmpBytes, VecZnxRshAdd,
    },
    layouts::{Backend, Module, Scratch, ScratchOwned},
};

// ─── conjugation out-of-place (CKKSCiphertext::conjugate) ───────────────────

/// Conjugation out-of-place: real part preserved, imaginary part negated.
pub fn test_conjugate<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEAutomorphism<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxCopy
        + VecZnxLsh<BE>
        + VecZnxLshInplace<BE>
        + VecZnxRshAdd<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(&ctx.re1, &ctx.im1, scratch.borrow());
    let (want_re, want_im) = ctx.want_conjugate();
    let conj_key = ctx.atk(-1);

    let mut ct_res = ctx.alloc_ct();
    ct_res.conjugate(&ctx.module, &ct1, conj_key, scratch.borrow());

    assert_eq!(
        ct_res.log_hom_rem(),
        ct1.log_hom_rem(),
        "conjugate: log_hom_rem must equal input"
    );
    ctx.assert_decrypt_precision("conjugate", &ct_res, &want_re, &want_im, 20.0, scratch.borrow());
}

// ─── conjugation in-place (CKKSCiphertext::conjugate_inplace) ───────────────

/// Conjugation in-place: real part preserved, imaginary part negated.
pub fn test_conjugate_inplace<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEAutomorphism<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxCopy
        + VecZnxLsh<BE>
        + VecZnxLshInplace<BE>
        + VecZnxRshAdd<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    let mut scratch = ctx.alloc_scratch();
    let mut ct = ctx.encrypt(&ctx.re1, &ctx.im1, scratch.borrow());
    let expected_delta = ct.log_hom_rem();
    let (want_re, want_im) = ctx.want_conjugate();
    let conj_key = ctx.atk(-1);

    ct.conjugate_inplace(&ctx.module, conj_key, scratch.borrow());

    assert_eq!(
        ct.log_hom_rem(),
        expected_delta,
        "conjugate_inplace: log_hom_rem must be unchanged"
    );
    ctx.assert_decrypt_precision("conjugate_inplace", &ct, &want_re, &want_im, 20.0, scratch.borrow());
}
