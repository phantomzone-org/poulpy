//! Encrypt / decrypt round-trip test.

use super::helpers::{TestContext, assert_precision};
use poulpy_core::{GLWEDecrypt, GLWEEncryptSk, ScratchTakeCore, layouts::GLWESecretPreparedFactory};
use poulpy_hal::{
    api::{ModuleN, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxNormalize, VecZnxNormalizeTmpBytes},
    layouts::{Backend, Module, Scratch, ScratchOwned},
};

/// Verifies that encrypt → decrypt → decode recovers the original message.
pub fn test_encrypt_decrypt<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let mut scratch = ctx.alloc_scratch();
    let ct = ctx.encrypt(&ctx.re1, &ctx.im1, &mut scratch);
    let (re_out, im_out) = ctx.decrypt_decode(&ct, &mut scratch);
    assert_precision("encrypt_decrypt re", &re_out, &ctx.re1, 20.0);
    assert_precision("encrypt_decrypt im", &im_out, &ctx.im1, 20.0);
}
