//! Negation tests (out-of-place and in-place).
//!
//! # Test inventory
//!
//! ## Operations-layer negation (`CKKSCiphertext::neg`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_neg`] | out-of-place negation |
//!
//! ## Operations-layer negation (`CKKSCiphertext::neg_inplace`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_neg_inplace`] | in-place negation |

use super::helpers::TestContext;
use poulpy_core::{
    GLWEDecrypt, GLWEEncryptSk, ScratchTakeCore, layouts::GLWESecretPreparedFactory,
};
use poulpy_hal::{
    api::{
        ModuleN, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxCopy, VecZnxLsh, VecZnxLshInplace, VecZnxNegate,
        VecZnxNegateInplace, VecZnxNormalize, VecZnxNormalizeTmpBytes, VecZnxRshAdd,
    },
    layouts::{Backend, Module, Scratch, ScratchOwned},
};

// ─── negation out-of-place (CKKSCiphertext::neg) ────────────────────────────

/// Negation out-of-place.
pub fn test_neg<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxNegate
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxCopy
        + VecZnxLsh<BE>
        + VecZnxLshInplace<BE>
        + VecZnxRshAdd<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(&ctx.re1, &ctx.im1, scratch.borrow());
    let (want_re, want_im) = ctx.want_neg();

    let mut ct_res = ctx.alloc_ct();
    ct_res.neg(&ctx.module, &ct1);

    assert_eq!(ct_res.log_delta, ct1.log_delta, "neg: log_delta must equal input");
    ctx.assert_decrypt_precision("neg", &ct_res, &want_re, &want_im, 20.0, scratch.borrow());
}

// ─── negation in-place (CKKSCiphertext::neg_inplace) ────────────────────────

/// Negation in-place.
pub fn test_neg_inplace<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxNegateInplace
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxCopy
        + VecZnxLsh<BE>
        + VecZnxLshInplace<BE>
        + VecZnxRshAdd<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let mut scratch = ctx.alloc_scratch();
    let mut ct = ctx.encrypt(&ctx.re1, &ctx.im1, scratch.borrow());
    let expected_delta = ct.log_delta;
    let (want_re, want_im) = ctx.want_neg();

    ct.neg_inplace(&ctx.module);

    assert_eq!(ct.log_delta, expected_delta, "neg_inplace: log_delta must be unchanged");
    ctx.assert_decrypt_precision("neg_inplace", &ct, &want_re, &want_im, 20.0, scratch.borrow());
}
