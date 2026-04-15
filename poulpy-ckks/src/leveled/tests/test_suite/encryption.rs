//! Encrypt / decrypt round-trip test.

use super::helpers::{TestCiphertextBackend as Backend, TestContext, assert_ct_meta, assert_precision};
use poulpy_hal::api::ScratchOwnedBorrow;

/// Verifies that encrypt → decrypt → decode recovers the original message.
pub fn test_encrypt_decrypt<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    assert_ct_meta(
        "encrypt_decrypt",
        &ct,
        ctx.meta().log_decimal,
        ctx.max_k() - ctx.meta().log_decimal,
    );
    let (re_out, im_out) = ctx.decrypt_decode(&ct, scratch.borrow());
    assert_precision("encrypt_decrypt re", &re_out, &ctx.re1, 20.0);
    assert_precision("encrypt_decrypt im", &im_out, &ctx.im1, 20.0);
}
