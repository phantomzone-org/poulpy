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
//! | [`test_mul_pow2_smaller_output`] | out-of-place into a smaller output buffer |
//! | [`test_mul_pow2_inplace`] | in-place, message × 2^bits |
//!
//! ## `GLWE<_, CKKS>::div_pow2` / `div_pow2_inplace`
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_div_pow2`] | out-of-place, message / 2^bits |
//! | [`test_div_pow2_smaller_output`] | out-of-place into a smaller output buffer |
//! | [`test_div_pow2_inplace`] | in-place, message / 2^bits |

use crate::{CKKSInfos, leveled::operations::pow2::CKKSPow2Ops};

use super::helpers::{TestContext, TestPow2Backend as Backend, assert_ct_meta, assert_unary_output_meta};
use poulpy_core::layouts::LWEInfos;
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
    assert_unary_output_meta("mul_pow2", &ct_res, &ct);
    ctx.assert_decrypt_precision("mul_pow2", &ct_res, &want_re, &want_im, 20.0, scratch.borrow());
}

/// Out-of-place multiplication by 2^bits.
pub fn test_mul_pow2_smaller_output<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let (want_re, want_im) = ctx.want_mul_pow2(SHIFT_BITS);
    let mut ct_res = ctx.alloc_ct(ctx.max_k() - ctx.base2k().as_usize() - 1);
    ct_res.mul_pow2(&ctx.module, &ct, SHIFT_BITS, scratch.borrow()).unwrap();
    assert_unary_output_meta("mul_pow2 smaller_output", &ct_res, &ct);
    ctx.assert_decrypt_precision("mul_pow2", &ct_res, &want_re, &want_im, 20.0, scratch.borrow());
}

/// In-place multiplication by 2^bits.
pub fn test_mul_pow2_inplace<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let mut ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let (want_re, want_im) = ctx.want_mul_pow2(SHIFT_BITS);
    let expected_log_decimal = ct.log_decimal();
    let expected_log_hom_rem = ct.log_hom_rem();
    ct.mul_pow2_inplace(&ctx.module, SHIFT_BITS, scratch.borrow()).unwrap();
    assert_ct_meta("mul_pow2_inplace", &ct, expected_log_decimal, expected_log_hom_rem);
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
    assert_ct_meta("div_pow2", &ct_res, ct.log_decimal(), ct.log_hom_rem() - SHIFT_BITS);
    ctx.assert_decrypt_precision("div_pow2", &ct_res, &want_re, &want_im, 20.0, scratch.borrow());
}

/// Out-of-place division by 2^bits.
pub fn test_div_pow2_smaller_output<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let (want_re, want_im) = ctx.want_div_pow2(SHIFT_BITS);
    let mut ct_res = ctx.alloc_ct(ctx.max_k() - ctx.base2k().as_usize() - 1);
    ct_res.div_pow2(&ctx.module, &ct, SHIFT_BITS, scratch.borrow()).unwrap();
    let offset = ct.effective_k().saturating_sub(ct_res.max_k().as_usize());
    assert_ct_meta("div_pow2 smaller_output", &ct_res, ct.log_decimal(), ct.log_hom_rem() - SHIFT_BITS - offset);
    ctx.assert_decrypt_precision("div_pow2", &ct_res, &want_re, &want_im, 20.0, scratch.borrow());
}

/// In-place division by 2^bits.
pub fn test_div_pow2_inplace<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let mut ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let (want_re, want_im) = ctx.want_div_pow2(SHIFT_BITS);
    let expected_log_decimal = ct.log_decimal();
    let expected_log_hom_rem = ct.log_hom_rem() - SHIFT_BITS;
    ct.div_pow2_inplace(SHIFT_BITS).unwrap();
    assert_ct_meta("div_pow2_inplace", &ct, expected_log_decimal, expected_log_hom_rem);
    ctx.assert_decrypt_precision("div_pow2_inplace", &ct, &want_re, &want_im, 20.0, scratch.borrow());
}
