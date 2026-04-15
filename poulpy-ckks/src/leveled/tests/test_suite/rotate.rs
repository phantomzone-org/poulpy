//! Slot rotation tests (out-of-place and in-place).
//!
//! # Test inventory
//!
//! ## Operations-layer rotation (`GLWE<_, CKKS>::rotate`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_rotate`] | out-of-place rotation for each requested shift |
//!
//! ## Operations-layer rotation (`GLWE<_, CKKS>::rotate_inplace`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_rotate_inplace`] | in-place rotation for each requested shift |

use crate::leveled::operations::rotate::CKKSRotateOps;

use super::helpers::{TestContext, TestRotateBackend as Backend};
use poulpy_hal::api::ScratchOwnedBorrow;

// ─── rotation out-of-place (GLWE<_, CKKS>::rotate) ─────────────────────────

/// Rotation out-of-place: slot values are cyclically shifted.
pub fn test_rotate_aligned<BE: Backend>(ctx: &TestContext<BE>, rotations: &[i64]) {
    let mut scratch = ctx.alloc_scratch();
    let ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    for &r in rotations {
        let (want_re, want_im) = ctx.want_rotate(r);
        let mut ct_res = ctx.alloc_ct(ctx.max_k());
        ct_res.rotate(&ctx.module, &ct, r, ctx.atks(), scratch.borrow()).unwrap();
        ctx.assert_decrypt_precision(&format!("rotate({r})"), &ct_res, &want_re, &want_im, 20.0, scratch.borrow());
    }
}

/// Rotation out-of-place: slot values are cyclically shifted.
pub fn test_rotate_smaller_output<BE: Backend>(ctx: &TestContext<BE>, rotations: &[i64]) {
    let mut scratch = ctx.alloc_scratch();
    let ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    for &r in rotations {
        let (want_re, want_im) = ctx.want_rotate(r);
        let mut ct_res = ctx.alloc_ct(ctx.max_k() - ctx.base2k().as_usize() - 1);
        ct_res.rotate(&ctx.module, &ct, r, ctx.atks(), scratch.borrow()).unwrap();
        ctx.assert_decrypt_precision(&format!("rotate({r})"), &ct_res, &want_re, &want_im, 20.0, scratch.borrow());
    }
}

// ─── rotation in-place (GLWE<_, CKKS>::rotate_inplace) ─────────────────────

/// Rotation in-place: slot values are cyclically shifted.
pub fn test_rotate_inplace<BE: Backend>(ctx: &TestContext<BE>, rotations: &[i64]) {
    let mut scratch = ctx.alloc_scratch();
    for &r in rotations {
        let (want_re, want_im) = ctx.want_rotate(r);
        let mut ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
        ct.rotate_inplace(&ctx.module, r, ctx.atks(), scratch.borrow());
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
