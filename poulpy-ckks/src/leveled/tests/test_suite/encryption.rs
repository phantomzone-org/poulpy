//! Encrypt / decrypt round-trip test.

use super::helpers::{TestContext, assert_precision};
use poulpy_core::{GLWEDecrypt, GLWEEncryptSk, GLWEShift, ScratchTakeCore, layouts::GLWESecretPreparedFactory};
use poulpy_hal::{
    api::{
        ModuleN, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxCopy, VecZnxLsh, VecZnxLshInplace, VecZnxNormalize,
        VecZnxNormalizeTmpBytes, VecZnxRshAdd,
    },
    layouts::{Backend, Module, Scratch, ScratchOwned},
};

/// Verifies that encrypt → decrypt → decode recovers the original message.
pub fn test_encrypt_decrypt<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWEShift<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxRshAdd<BE>
        + VecZnxLsh<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxLshInplace<BE>
        + VecZnxCopy,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let mut scratch = ctx.alloc_scratch();
    let ct = ctx.encrypt(&ctx.re1, &ctx.im1, scratch.borrow());
    let (re_out, im_out) = ctx.decrypt_decode(&ct, scratch.borrow());
    assert_precision("encrypt_decrypt re", &re_out, &ctx.re1, 20.0);
    assert_precision("encrypt_decrypt im", &im_out, &ctx.im1, 20.0);
}
