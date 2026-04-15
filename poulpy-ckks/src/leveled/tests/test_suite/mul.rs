//! Multiplication tests: ct × ct and ct² (square).
//!
//! # Test inventory
//!
//! ## ct × ct multiplication (`GLWE<_, CKKS>::mul`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_mul_ct_aligned`] | both inputs at same `log_hom_rem()` |
//! | [`test_mul_ct_delta_a_lt_b`] | `a.log_hom_rem() < b.log_hom_rem()` |
//! | [`test_mul_ct_delta_a_gt_b`] | `a.log_hom_rem() > b.log_hom_rem()` |
//! | [`test_mul_ct_smaller_output`] | output has smaller `max_k()` than inputs |
//!
//! ## ct² squaring (`GLWE<_, CKKS>::square`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_square_ct_aligned`] | square at default precision |
//! | [`test_square_ct_rescaled_input`] | square after a rescale (reduced `log_hom_rem()`) |
//! | [`test_square_ct_smaller_output`] | square into smaller output buffer |

use crate::{
    CKKS, CKKSInfos,
    leveled::{
        operations::mul::CKKSMulOps,
        tests::test_suite::helpers::{TestContext, TestMulBackend as Backend},
    },
};

use poulpy_core::{GLWEDecrypt, layouts::GLWEPlaintext};
use poulpy_hal::api::ScratchOwnedBorrow;
// ─── ct × ct out-of-place (GLWE<_, CKKS>::mul) ─────────────────────────────────

/// ct × ct multiplication with both inputs at the same log_hom_rem().
pub fn test_mul_ct_aligned<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let ct2 = ctx.encrypt(ctx.max_k(), &ctx.re2, &ctx.im2, scratch.borrow());

    let (want_re, want_im) = ctx.want_mul();

    let mut pt = GLWEPlaintext::<Vec<u8>, CKKS>::alloc_from_infos(&ct1);

    println!("ct1: k:{} log_decimal:{}", ct1.effective_k(), ct1.log_decimal());
    println!("ct2: k:{} log_decimal:{}", ct2.effective_k(), ct2.log_decimal());

    ctx.module.glwe_decrypt(&ct1, &mut pt, &ctx.sk, scratch.borrow());
    println!("ct1: {pt}");
    ctx.module.glwe_decrypt(&ct2, &mut pt, &ctx.sk, scratch.borrow());
    println!("ct2: {pt}");

    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ct_res.mul(&ctx.module, &ct1, &ct2, ctx.tsk(), scratch.borrow()).unwrap();

    ctx.assert_decrypt_precision("mul_ct_aligned", &ct_res, &want_re, &want_im, 20.0, scratch.borrow());
}

/// ct × ct, a.log_hom_rem() < b.log_hom_rem() (a rescaled by one limb).
pub fn test_mul_ct_delta_a_lt_b<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(
        ctx.max_k() - ctx.base2k().as_usize() + 1,
        &ctx.re1,
        &ctx.im1,
        scratch.borrow(),
    );
    let ct2 = ctx.encrypt(ctx.max_k(), &ctx.re2, &ctx.im2, scratch.borrow());
    let (want_re, want_im) = ctx.want_mul();
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ct_res.mul(&ctx.module, &ct1, &ct2, ctx.tsk(), scratch.borrow()).unwrap();
    ctx.assert_decrypt_precision("mul_ct a_lt_b", &ct_res, &want_re, &want_im, 20.0, scratch.borrow());
}

/// ct × ct, a.log_hom_rem() > b.log_hom_rem() (b rescaled by one limb).
pub fn test_mul_ct_delta_a_gt_b<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let ct2 = ctx.encrypt(
        ctx.max_k() - ctx.base2k().as_usize() + 1,
        &ctx.re2,
        &ctx.im2,
        scratch.borrow(),
    );
    let (want_re, want_im) = ctx.want_mul();
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ct_res.mul(&ctx.module, &ct1, &ct2, ctx.tsk(), scratch.borrow()).unwrap();
    ctx.assert_decrypt_precision("mul_ct a_gt_b", &ct_res, &want_re, &want_im, 20.0, scratch.borrow());
}

/// ct × ct, output buffer has smaller max_k than inputs (offset > 0).
pub fn test_mul_ct_smaller_output<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let ct2 = ctx.encrypt(ctx.max_k(), &ctx.re2, &ctx.im2, scratch.borrow());
    let (want_re, want_im) = ctx.want_mul();
    let mut ct_res = ctx.alloc_ct(ctx.max_k() - ctx.base2k().as_usize() - 1);
    ct_res.mul(&ctx.module, &ct1, &ct2, ctx.tsk(), scratch.borrow()).unwrap();
    ctx.assert_decrypt_precision("mul_ct smaller_output", &ct_res, &want_re, &want_im, 20.0, scratch.borrow());
}

// ─── ct² squaring (GLWE<_, CKKS>::square) ───────────────────────────────────────

/// ct² at default precision (same as fresh encryption).
pub fn test_square_ct_aligned<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let (want_re, want_im) = ctx.want_square();
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ct_res.square(&ctx.module, &ct, ctx.tsk(), scratch.borrow()).unwrap();
    ctx.assert_decrypt_precision("square_ct_aligned", &ct_res, &want_re, &want_im, 20.0, scratch.borrow());
}

/// ct² into an output buffer with smaller k.
pub fn test_square_ct_smaller_output<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let (want_re, want_im) = ctx.want_square();
    let mut ct_res = ctx.alloc_ct(ctx.max_k() - ctx.base2k().as_usize() - 1);
    ct_res.square(&ctx.module, &ct, ctx.tsk(), scratch.borrow()).unwrap();
    ctx.assert_decrypt_precision("square_ct rescaled", &ct_res, &want_re, &want_im, 20.0, scratch.borrow());
}
