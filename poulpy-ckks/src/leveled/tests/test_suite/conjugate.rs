//! Conjugation tests (out-of-place and in-place).
//!
//! # Test inventory
//!
//! ## Operations-layer conjugation (`GLWE<_, CKKS>::conjugate`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_conjugate`] | out-of-place conjugation |
//!
//! ## Operations-layer conjugation (`GLWE<_, CKKS>::conjugate_inplace`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_conjugate_inplace`] | in-place conjugation |

use crate::leveled::operations::conjugate::CKKSConjugateOps;

use super::helpers::{TestContext, TestRotateBackend as Backend};
use poulpy_hal::api::ScratchOwnedBorrow;

// ─── conjugation out-of-place (GLWE<_, CKKS>::conjugate) ───────────────────

/// Conjugation out-of-place: real part preserved, imaginary part negated.
pub fn test_conjugate_aligned<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let (want_re, want_im) = ctx.want_conjugate();
    let conj_key = ctx.atk(-1);
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ct_res.conjugate(&ctx.module, &ct1, conj_key, scratch.borrow()).unwrap();
    ctx.assert_decrypt_precision("conjugate", &ct_res, &want_re, &want_im, 20.0, scratch.borrow());
}

/// Conjugation out-of-place: real part preserved, imaginary part negated.
pub fn test_conjugate_smaller_output<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let (want_re, want_im) = ctx.want_conjugate();
    let conj_key = ctx.atk(-1);
    let mut ct_res = ctx.alloc_ct(ctx.max_k() - ctx.base2k().as_usize() - 1);
    ct_res.conjugate(&ctx.module, &ct1, conj_key, scratch.borrow()).unwrap();
    ctx.assert_decrypt_precision("conjugate", &ct_res, &want_re, &want_im, 20.0, scratch.borrow());
}

// ─── conjugation in-place (GLWE<_, CKKS>::conjugate_inplace) ───────────────

/// Conjugation in-place: real part preserved, imaginary part negated.
pub fn test_conjugate_inplace<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let mut ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let (want_re, want_im) = ctx.want_conjugate();
    let conj_key = ctx.atk(-1);
    ct.conjugate_inplace(&ctx.module, conj_key, scratch.borrow());
    ctx.assert_decrypt_precision("conjugate_inplace", &ct, &want_re, &want_im, 20.0, scratch.borrow());
}
