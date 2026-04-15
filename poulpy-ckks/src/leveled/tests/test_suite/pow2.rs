//! Multiplication and division by a power of two.
//!
//! These operations shift the GLWE payload without altering CKKS metadata
//! (`log_decimal`, `log_hom_rem`).
//!
//! # Test inventory
//!
//! ## `GLWE<_, CKKS>::mul_pow2` / `mul_pow2_inplace`
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_mul_pow2`] | out-of-place, message × 2^bits |
//! | [`test_mul_pow2_inplace`] | in-place, message × 2^bits |
//!
//! ## `GLWE<_, CKKS>::div_pow2` / `div_pow2_inplace`
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_div_pow2`] | out-of-place, message / 2^bits |
//! | [`test_div_pow2_inplace`] | in-place, message / 2^bits |

use crate::leveled::operations::pow2::CKKSPow2Ops;

use super::helpers::{TestContext, TestPow2Backend as Backend};
use poulpy_hal::api::ScratchOwnedBorrow;

const SHIFT_BITS: usize = 7;

// ─── mul_pow2 (message × 2^bits) ───────────────────────────────────────────────

/// Out-of-place multiplication by 2^bits.
pub fn test_mul_pow2_aligned<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let (want_re, want_im) = ctx.want_mul_pow2(SHIFT_BITS);
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ct_res.mul_pow2(&ctx.module, &ct, SHIFT_BITS, scratch.borrow()).unwrap();
    ctx.assert_decrypt_precision("mul_pow2", &ct_res, &want_re, &want_im, 20.0, scratch.borrow());
}

/// Out-of-place multiplication by 2^bits.
pub fn test_mul_pow2_smaller_output<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let (want_re, want_im) = ctx.want_mul_pow2(SHIFT_BITS);
    let mut ct_res = ctx.alloc_ct(ctx.max_k() - ctx.base2k().as_usize() - 1);
    ct_res.mul_pow2(&ctx.module, &ct, SHIFT_BITS, scratch.borrow()).unwrap();
    ctx.assert_decrypt_precision("mul_pow2", &ct_res, &want_re, &want_im, 20.0, scratch.borrow());
}

/// In-place multiplication by 2^bits.
pub fn test_mul_pow2_inplace<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let mut ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let (want_re, want_im) = ctx.want_mul_pow2(SHIFT_BITS);
    ct.mul_pow2_inplace(&ctx.module, SHIFT_BITS, scratch.borrow()).unwrap();
    ctx.assert_decrypt_precision("mul_pow2_inplace", &ct, &want_re, &want_im, 20.0, scratch.borrow());
}

// ─── div_pow2 (message / 2^bits) ───────────────────────────────────────────────

/// Out-of-place division by 2^bits.
pub fn test_div_pow2_aligned<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let (want_re, want_im) = ctx.want_div_pow2(SHIFT_BITS);
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ct_res.div_pow2(&ctx.module, &ct, SHIFT_BITS, scratch.borrow()).unwrap();
    ctx.assert_decrypt_precision("div_pow2", &ct_res, &want_re, &want_im, 20.0, scratch.borrow());
}

/// Out-of-place division by 2^bits.
pub fn test_div_pow2_smaller_output<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let (want_re, want_im) = ctx.want_div_pow2(SHIFT_BITS);
    let mut ct_res = ctx.alloc_ct(ctx.max_k() - ctx.base2k().as_usize() - 1);
    ct_res.div_pow2(&ctx.module, &ct, SHIFT_BITS, scratch.borrow()).unwrap();
    ctx.assert_decrypt_precision("div_pow2", &ct_res, &want_re, &want_im, 20.0, scratch.borrow());
}

/// In-place division by 2^bits.
pub fn test_div_pow2_inplace<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let mut ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let (want_re, want_im) = ctx.want_div_pow2(SHIFT_BITS);
    ct.div_pow2_inplace(SHIFT_BITS).unwrap();
    ctx.assert_decrypt_precision("div_pow2_inplace", &ct, &want_re, &want_im, 20.0, scratch.borrow());
}
