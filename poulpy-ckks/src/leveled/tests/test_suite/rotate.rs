//! Slot rotation tests (out-of-place and in-place).
//!
//! # Test inventory
//!
//! ## Operations-layer rotation (`CKKSCiphertext::rotate`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_rotate`] | out-of-place rotation for each requested shift |
//!
//! ## Operations-layer rotation (`CKKSCiphertext::rotate_inplace`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_rotate_inplace`] | in-place rotation for each requested shift |

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

// ─── rotation out-of-place (CKKSCiphertext::rotate) ─────────────────────────

/// Rotation out-of-place: slot values are cyclically shifted.
pub fn test_rotate<BE: Backend>(ctx: &TestContext<BE>, rotations: &[i64])
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
    let ct = ctx.encrypt(&ctx.re1, &ctx.im1, scratch.borrow());

    for &r in rotations {
        let (want_re, want_im) = ctx.want_rotate(r);

        let mut ct_res = ctx.alloc_ct();
        ct_res.rotate(&ctx.module, &ct, r, ctx.atks(), scratch.borrow());

        assert_eq!(
            ct_res.log_hom_rem(),
            ct.log_hom_rem(),
            "rotate({r}): log_hom_rem must equal input"
        );
        ctx.assert_decrypt_precision(&format!("rotate({r})"), &ct_res, &want_re, &want_im, 20.0, scratch.borrow());
    }
}

// ─── rotation in-place (CKKSCiphertext::rotate_inplace) ─────────────────────

/// Rotation in-place: slot values are cyclically shifted.
pub fn test_rotate_inplace<BE: Backend>(ctx: &TestContext<BE>, rotations: &[i64])
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

    for &r in rotations {
        let (want_re, want_im) = ctx.want_rotate(r);

        let mut ct = ctx.encrypt(&ctx.re1, &ctx.im1, scratch.borrow());
        let expected_delta = ct.log_hom_rem();

        ct.rotate_inplace(&ctx.module, r, ctx.atks(), scratch.borrow());

        assert_eq!(
            ct.log_hom_rem(),
            expected_delta,
            "rotate_inplace({r}): log_hom_rem must be unchanged"
        );
        ctx.assert_decrypt_precision(
            &format!("rotate_inplace({r})"),
            &ct,
            &want_re,
            &want_im,
            20.0,
            scratch.borrow(),
        );
    }
}
