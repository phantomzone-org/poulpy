//! Multiplication tests: ct × ct and ct² (square).
//!
//! # Test inventory
//!
//! ## ct × ct multiplication out of place (`GLWE<_, CKKS>::mul`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_mul_ct_aligned`] | both inputs at same `log_hom_rem()` |
//! | [`test_mul_ct_delta_a_lt_b`] | `a.log_hom_rem() < b.log_hom_rem()` |
//! | [`test_mul_ct_delta_a_gt_b`] | `a.log_hom_rem() > b.log_hom_rem()` |
//! | [`test_mul_ct_smaller_output`] | output has smaller `max_k()` than inputs |
//!
//! ## ct x ct inplace ct-ct (`GLWE<_, CKKS>::mul_inplace`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_mul_ct_inplace_aligned`] | `self.log_hom_rem() == a.log_hom_rem()` |
//! | [`test_mul_ct_inplace_self_lt`] | `self.log_hom_rem() < a.log_hom_rem()` → a shifted to align with self |
//! | [`test_mul_ct_inplace_self_gt`] | `self.log_hom_rem() > a.log_hom_rem()` → self shifted to align with a |
//!
//! ## ct² squaring out of place (`GLWE<_, CKKS>::square`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_square_aligned`] | square at default precision |
//! | [`test_square_rescaled_input`] | square after a rescale (reduced `log_hom_rem()`) |
//! | [`test_square_smaller_output`] | square into smaller output buffer |
//!
//! ## ct² squaring inplace (`GLWE<_, CKKS>::square_inplace`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_square_inplace`] | square at default precision |
//!
//! ## ct x pt_znx out of place (`GLWE<_, CKKS>::mul_pt_znx`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_mul_pt_znx_aligned`] | input and output at same `log_hom_rem()` |
//! | [`test_mul_pt_znx_smaller_output`] | output at smaller `log_hom_rem()` |
//!
//! ## ct x pt_znx inplace (`GLWE<_, CKKS>::mul_pt_znx_inplace`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_mul_pt_znx_inplace`] | - |
//!
//! ## ct x pt_rnx out of place (`GLWE<_, CKKS>::mul_pt_rnx`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_mul_pt_znx_aligned`] | input and output at same `log_hom_rem()` |
//! | [`test_mul_pt_znx_smaller_output`] | output at smaller `log_hom_rem()` |
//!
//! ## ct x pt_rnx inplace (`GLWE<_, CKKS>::mul_pt_rnx_inplace`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_mul_pt_rnx_inplace`] | - |
use crate::{
    CKKSCompositionError, CKKSInfos,
    leveled::{
        operations::mul::CKKSMulOps,
        tests::test_suite::helpers::{
            TestContext, TestMulBackend as Backend, assert_ckks_error, assert_mul_ct_output_meta, assert_mul_pt_output_meta,
        },
    },
};

use poulpy_hal::api::ScratchOwnedBorrow;
// ─── ct × ct out-of-place (GLWE<_, CKKS>::mul) ─────────────────────────────────

/// ct × ct multiplication with both inputs at the same log_hom_rem().
pub fn test_mul_ct_aligned<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let ct2 = ctx.encrypt(ctx.max_k(), &ctx.re2, &ctx.im2, scratch.borrow());
    let (want_re, want_im) = ctx.want_mul();
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ct_res.mul(&ctx.module, &ct1, &ct2, ctx.tsk(), scratch.borrow()).unwrap();
    assert_mul_ct_output_meta("mul_ct_aligned", &ct_res, &ct1, &ct2);
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
    assert_mul_ct_output_meta("mul_ct a_lt_b", &ct_res, &ct1, &ct2);
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
    assert_mul_ct_output_meta("mul_ct a_gt_b", &ct_res, &ct1, &ct2);
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
    assert_mul_ct_output_meta("mul_ct smaller_output", &ct_res, &ct1, &ct2);
    ctx.assert_decrypt_precision("mul_ct smaller_output", &ct_res, &want_re, &want_im, 20.0, scratch.borrow());
}

// ─── ct × ct out-of-place (GLWE<_, CKKS>::mul_inplace) ─────────────────────────────────

/// ct × ct multiplication with both inputs at the same log_hom_rem().
pub fn test_mul_ct_inplace_aligned<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let mut ct_res = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let ct1 = ctx.encrypt(ctx.max_k(), &ctx.re2, &ctx.im2, scratch.borrow());
    let (want_re, want_im) = ctx.want_mul();
    let ct_res_meta = ct_res.meta();
    ct_res.mul_inplace(&ctx.module, &ct1, ctx.tsk(), scratch.borrow()).unwrap();
    assert_mul_ct_output_meta("mul_ct_aligned", &ct_res, &ct_res_meta, &ct1);
    ctx.assert_decrypt_precision("mul_ct_aligned", &ct_res, &want_re, &want_im, 20.0, scratch.borrow());
}

/// ct-ct in-place, self.log_hom_rem() < a.log_hom_rem() (a is shifted down to align with self).
pub fn test_mul_ct_inplace_self_lt<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let mut ct_res = ctx.encrypt(
        ctx.max_k() - ctx.base2k().as_usize() - 1,
        &ctx.re1,
        &ctx.im1,
        scratch.borrow(),
    );
    let ct1 = ctx.encrypt(ctx.max_k(), &ctx.re2, &ctx.im2, scratch.borrow());
    let (want_re, want_im) = ctx.want_mul();
    let ct_res_meta = ct_res.meta();
    ct_res.mul_inplace(&ctx.module, &ct1, ctx.tsk(), scratch.borrow()).unwrap();
    assert_mul_ct_output_meta("mul_ct_aligned", &ct_res, &ct_res_meta, &ct1);
    ctx.assert_decrypt_precision("mul_ct_aligned", &ct_res, &want_re, &want_im, 20.0, scratch.borrow());
}

/// ct-ct in-place, self.log_hom_rem() > a.log_hom_rem() (a is shifted down to align with self).
pub fn test_mul_ct_inplace_self_gt<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let mut ct_res = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let ct1 = ctx.encrypt(
        ctx.max_k() - ctx.base2k().as_usize() - 1,
        &ctx.re2,
        &ctx.im2,
        scratch.borrow(),
    );
    let (want_re, want_im) = ctx.want_mul();
    let ct_res_meta = ct_res.meta();
    ct_res.mul_inplace(&ctx.module, &ct1, ctx.tsk(), scratch.borrow()).unwrap();
    assert_mul_ct_output_meta("mul_ct_aligned", &ct_res, &ct_res_meta, &ct1);
    ctx.assert_decrypt_precision("mul_ct_aligned", &ct_res, &want_re, &want_im, 20.0, scratch.borrow());
}

// ─── ct² squaring out of place (GLWE<_, CKKS>::square) ───────────────────────────────────────

/// ct² at default precision (same as fresh encryption).
pub fn test_square_aligned<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let (want_re, want_im) = ctx.want_square();
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ct_res.square(&ctx.module, &ct, ctx.tsk(), scratch.borrow()).unwrap();
    assert_mul_ct_output_meta("square_aligned", &ct_res, &ct, &ct);
    ctx.assert_decrypt_precision("square_aligned", &ct_res, &want_re, &want_im, 20.0, scratch.borrow());
}

/// ct² after the input has already been rescaled by one limb.
pub fn test_square_rescaled_input<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let ct = ctx.encrypt(
        ctx.max_k() - ctx.base2k().as_usize() + 1,
        &ctx.re1,
        &ctx.im1,
        scratch.borrow(),
    );
    let (want_re, want_im) = ctx.want_square();
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ct_res.square(&ctx.module, &ct, ctx.tsk(), scratch.borrow()).unwrap();
    assert_mul_ct_output_meta("square_rescaled_input", &ct_res, &ct, &ct);
    ctx.assert_decrypt_precision("square_rescaled_input", &ct_res, &want_re, &want_im, 20.0, scratch.borrow());
}

/// ct² into an output buffer with smaller k.
pub fn test_square_smaller_output<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let (want_re, want_im) = ctx.want_square();
    let mut ct_res = ctx.alloc_ct(ctx.max_k() - ctx.base2k().as_usize() - 1);
    ct_res.square(&ctx.module, &ct, ctx.tsk(), scratch.borrow()).unwrap();
    assert_mul_ct_output_meta("square_smaller_output", &ct_res, &ct, &ct);
    ctx.assert_decrypt_precision(" square_smaller_output", &ct_res, &want_re, &want_im, 20.0, scratch.borrow());
}

// ─── ct² squaring inplace (GLWE<_, CKKS>::square) ───────────────────────────────────────

/// ct² inplace.
pub fn test_square_inplace<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let mut ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let (want_re, want_im) = ctx.want_square();
    let ct_in_meta = ct.meta();
    ct.square_inplace(&ctx.module, ctx.tsk(), scratch.borrow()).unwrap();
    assert_mul_ct_output_meta("square_inplace", &ct, &ct_in_meta, &ct_in_meta);
    ctx.assert_decrypt_precision("square_inplace", &ct, &want_re, &want_im, 20.0, scratch.borrow());
}

// ─── ct x pt_znx out of place (`GLWE<_, CKKS>::mul_pt_znx`)) ───────────────────────────

pub fn test_mul_pt_znx_aligned<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let pt = ctx.encode_pt_znx(&ctx.re2, &ctx.im2);
    let (want_re, want_im) = ctx.want_mul();
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ct_res.mul_pt_znx(&ctx.module, &ct, &pt, scratch.borrow()).unwrap();
    assert_mul_pt_output_meta("mul_pt_znx_aligned", &ct_res, &ct, &pt);
    ctx.assert_decrypt_precision("mul_pt_znx_aligned", &ct_res, &want_re, &want_im, 20.0, scratch.borrow());
}

pub fn test_mul_pt_znx_smaller_output<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let pt = ctx.encode_pt_znx(&ctx.re2, &ctx.im2);
    let (want_re, want_im) = ctx.want_mul();
    let mut ct_res = ctx.alloc_ct(ctx.max_k() - ctx.base2k().as_usize() - 1);
    ct_res.mul_pt_znx(&ctx.module, &ct, &pt, scratch.borrow()).unwrap();
    assert_mul_pt_output_meta("mul_pt_znx_aligned", &ct_res, &ct, &pt);
    ctx.assert_decrypt_precision("mul_pt_znx_aligned", &ct_res, &want_re, &want_im, 20.0, scratch.borrow());
}

// ─── ct x pt_znx inplace (`GLWE<_, CKKS>::mul_pt_znx_inplace`)) ───────────────────────────

pub fn test_mul_pt_znx_inplace<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let mut ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let pt = ctx.encode_pt_znx(&ctx.re2, &ctx.im2);
    let (want_re, want_im) = ctx.want_mul();
    let ct_meta = ct.meta();
    ct.mul_pt_znx_inplace(&ctx.module, &pt, scratch.borrow()).unwrap();
    assert_mul_pt_output_meta("mul_pt_znx_aligned", &ct, &ct_meta, &pt);
    ctx.assert_decrypt_precision("mul_pt_znx_aligned", &ct, &want_re, &want_im, 20.0, scratch.borrow());
}

// ─── ct x pt_rnx out of place (`GLWE<_, CKKS>::mul_pt_rnx`) ───────────────────────────

pub fn test_mul_pt_rnx_aligned<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let pt: crate::layouts::plaintext::CKKSPlaintextRnx<f64> = ctx.encode_pt_rnx(&ctx.re2, &ctx.im2);
    let (want_re, want_im) = ctx.want_mul();
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ct_res
        .mul_pt_rnx(&ctx.module, &ct, &pt, ctx.meta(), scratch.borrow())
        .unwrap();
    assert_mul_pt_output_meta("mul_pt_znx_aligned", &ct_res, &ct, &ctx.meta());
    ctx.assert_decrypt_precision("mul_pt_znx_aligned", &ct_res, &want_re, &want_im, 20.0, scratch.borrow());
}

pub fn test_mul_pt_rnx_smaller_output<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let pt = ctx.encode_pt_rnx(&ctx.re2, &ctx.im2);
    let (want_re, want_im) = ctx.want_mul();
    let mut ct_res = ctx.alloc_ct(ctx.max_k() - ctx.base2k().as_usize() - 1);
    ct_res
        .mul_pt_rnx(&ctx.module, &ct, &pt, ctx.meta(), scratch.borrow())
        .unwrap();
    assert_mul_pt_output_meta("mul_pt_znx_aligned", &ct_res, &ct, &ctx.meta());
    ctx.assert_decrypt_precision("mul_pt_znx_aligned", &ct_res, &want_re, &want_im, 20.0, scratch.borrow());
}

// ─── ct x pt_rnx inplace (`GLWE<_, CKKS>::mul_pt_rnx_inplace`)) ───────────────────────────

pub fn test_mul_pt_rnx_inplace<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let mut ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let pt = ctx.encode_pt_rnx(&ctx.re2, &ctx.im2);
    let (want_re, want_im) = ctx.want_mul();
    let ct_meta = ct.meta();
    ct.mul_pt_rnx_inplace(&ctx.module, &pt, ctx.meta(), scratch.borrow()).unwrap();
    assert_mul_pt_output_meta("mul_pt_znx_aligned", &ct, &ct_meta, &ctx.meta());
    ctx.assert_decrypt_precision("mul_pt_znx_aligned", &ct, &want_re, &want_im, 20.0, scratch.borrow());
}

/// Multiplication with inconsistent metadata must fail explicitly instead of panicking on usize underflow.
pub fn test_mul_ct_explicit_metadata_error<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let mut ct1 = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let mut ct2 = ctx.encrypt(ctx.max_k(), &ctx.re2, &ctx.im2, scratch.borrow());
    ct1.set_log_hom_rem(8).unwrap();
    ct2.set_log_hom_rem(9).unwrap();
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    let err = ct_res.mul(&ctx.module, &ct1, &ct2, ctx.tsk(), scratch.borrow()).unwrap_err();
    assert_ckks_error(
        "mul_ct_explicit_metadata_error",
        &err,
        CKKSCompositionError::MultiplicationPrecisionUnderflow {
            op: "mul",
            lhs_log_hom_rem: 8,
            rhs_log_hom_rem: 9,
            lhs_log_decimal: ctx.meta().log_decimal,
            rhs_log_decimal: ctx.meta().log_decimal,
        },
    );
}
