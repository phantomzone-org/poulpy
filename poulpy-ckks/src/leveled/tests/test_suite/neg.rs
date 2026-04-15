//! Negation tests (out-of-place and in-place).
//!
//! # Test inventory
//!
//! ## Operations-layer negation (`GLWE<_, CKKS>::neg`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_neg`] | out-of-place negation |
//!
//! ## Operations-layer negation (`GLWE<_, CKKS>::neg_inplace`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_neg_inplace`] | in-place negation |

use crate::leveled::operations::neg::CKKSNegOps;

use super::helpers::{TestContext, TestNegBackend as Backend};
use anyhow::Result;
use poulpy_hal::api::ScratchOwnedBorrow;

// ─── negation out-of-place (GLWE<_, CKKS>::neg) ────────────────────────────

/// Negation out-of-place.
pub fn test_neg_aligned<BE: Backend>(ctx: &TestContext<BE>) -> Result<()> {
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let (want_re, want_im) = ctx.want_neg();
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ct_res.neg(&ctx.module, &ct1, scratch.borrow())?;
    ctx.assert_decrypt_precision("neg", &ct_res, &want_re, &want_im, 20.0, scratch.borrow());
    Ok(())
}

/// Negation out-of-place.
pub fn test_neg_smaller_output<BE: Backend>(ctx: &TestContext<BE>) -> Result<()> {
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let (want_re, want_im) = ctx.want_neg();
    let mut ct_res = ctx.alloc_ct(ctx.max_k() - ctx.base2k().as_usize() - 1);
    ct_res.neg(&ctx.module, &ct1, scratch.borrow())?;
    ctx.assert_decrypt_precision("neg", &ct_res, &want_re, &want_im, 20.0, scratch.borrow());

    Ok(())
}

// ─── negation in-place (GLWE<_, CKKS>::neg_inplace) ────────────────────────

/// Negation in-place.
pub fn test_neg_inplace<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let mut ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let (want_re, want_im) = ctx.want_neg();
    ct.neg_inplace(&ctx.module);
    ctx.assert_decrypt_precision("neg_inplace", &ct, &want_re, &want_im, 20.0, scratch.borrow());
}
