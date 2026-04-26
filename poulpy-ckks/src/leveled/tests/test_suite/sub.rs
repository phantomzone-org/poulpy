//! Subtraction tests: ct-ct, ct-pt, ct-const (out-of-place and in-place).
//!
//! # Test inventory
//!
//! ## Operations-layer ct-ct (`GLWE<_, CKKS>::sub`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_sub_ct_aligned`] | `a.log_hom_rem() == b.log_hom_rem()`, `offset == 0` → `glwe_sub` fast path |
//! | [`test_sub_ct_delta_a_lt_b`] | `a.log_hom_rem() < b.log_hom_rem()` → b shifted to align with a |
//! | [`test_sub_ct_delta_a_gt_b`] | `a.log_hom_rem() > b.log_hom_rem()` → a shifted to align with b |
//! | [`test_sub_ct_smaller_output`] | `offset > 0` (output one limb narrower than inputs) |
//!
//! ## Operations-layer ct-ct (`GLWE<_, CKKS>::sub_assign`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_sub_ct_assign_aligned`] | `self.log_hom_rem() == a.log_hom_rem()` |
//! | [`test_sub_ct_assign_self_lt`] | `self.log_hom_rem() < a.log_hom_rem()` → a shifted to align with self |
//! | [`test_sub_ct_assign_self_gt`] | `self.log_hom_rem() > a.log_hom_rem()` → self shifted to align with a |
//!
//! ## Operations-layer ct - ZNX plaintext (`GLWE<_, CKKS>::sub_pt_vec_znx_into[_assign]`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_sub_pt_vec_znx_assign`] | in-place, `offset == 0` |
//! | [`test_sub_pt_vec_znx`] | out-of-place, `offset == 0` |
//! | [`test_sub_pt_vec_znx_into_smaller_output`] | out-of-place, `offset > 0` (output one limb narrower) |
//!
//! ## Operations-layer ct + RNX plaintext (`GLWE<_, CKKS>::sub_pt_vec_rnx_into[_assign]`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_sub_pt_vec_rnx_assign`] | in-place, `offset == 0`, RNX → ZNX auto-conversion |
//! | [`test_sub_pt_vec_rnx`] | out-of-place, `offset == 0`, RNX → ZNX auto-conversion |
//! | [`test_sub_pt_vec_rnx_into_smaller_output`] | out-of-place, `offset > 0` (output one limb narrower) |

use crate::{CKKSInfos, layouts::plaintext::CKKSConstPlaintextConversion, leveled::api::CKKSSubOps};

use super::helpers::{
    TestContext, TestScalar, TestSubBackend as Backend, TestVector, assert_binary_output_meta, assert_ct_meta,
    assert_unary_output_meta,
};
use poulpy_hal::api::ScratchOwnedBorrow;

const CONST_RE: f64 = 0.314_159_265_358_979_3;
const CONST_IM: f64 = -0.271_828_182_845_904_5;
const DELTA_LOG_DECIMAL: usize = 12;

// ─── ct-ct out-of-place (GLWE<_, CKKS>::sub) ────────────────────────────────

/// ct-ct out-of-place, aligned (same log_hom_rem, offset == 0 → glwe_sub fast path).
pub fn test_sub_ct_aligned<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let ct2 = ctx.encrypt(ctx.max_k(), &ctx.re2, &ctx.im2, scratch.borrow());
    let (want_re, want_im) = ctx.want_sub();
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ctx.module.ckks_sub_into(&mut ct_res, &ct1, &ct2, scratch.borrow()).unwrap();
    assert_binary_output_meta("sub_ct_aligned", &ct_res, &ct1, &ct2);
    ctx.assert_decrypt_precision("sub_ct_aligned", &ct_res, &want_re, &want_im, scratch.borrow());
}

/// ct-ct out-of-place, a.log_hom_rem() < b.log_hom_rem() (b is shifted to align with a).
pub fn test_sub_ct_delta_a_lt_b<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(
        ctx.max_k() - ctx.base2k().as_usize() + 1,
        &ctx.re1,
        &ctx.im1,
        scratch.borrow(),
    );
    let ct2 = ctx.encrypt(ctx.max_k(), &ctx.re2, &ctx.im2, scratch.borrow());
    let (want_re, want_im) = ctx.want_sub();
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ctx.module.ckks_sub_into(&mut ct_res, &ct1, &ct2, scratch.borrow()).unwrap();
    assert_binary_output_meta("sub_ct a_lt_b", &ct_res, &ct1, &ct2);
    ctx.assert_decrypt_precision("sub_ct a_lt_b", &ct_res, &want_re, &want_im, scratch.borrow());
}

/// ct-ct out-of-place, a.log_hom_rem() > b.log_hom_rem() (a is shifted to align with b).
pub fn test_sub_ct_delta_a_gt_b<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let ct2 = ctx.encrypt(
        ctx.max_k() - ctx.base2k().as_usize() + 1,
        &ctx.re2,
        &ctx.im2,
        scratch.borrow(),
    );
    let (want_re, want_im) = ctx.want_sub();
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ctx.module.ckks_sub_into(&mut ct_res, &ct1, &ct2, scratch.borrow()).unwrap();
    assert_binary_output_meta("sub_ct a_gt_b", &ct_res, &ct1, &ct2);
    ctx.assert_decrypt_precision("sub_ct a_gt_b", &ct_res, &want_re, &want_im, scratch.borrow());
}

/// ct-ct out-of-place with aligned homomorphic capacity but different log_decimal.
pub fn test_sub_ct_delta_log_decimal<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let low_log_decimal = ctx.meta().log_decimal - DELTA_LOG_DECIMAL;
    let low_prec = ctx.precision_at(low_log_decimal);
    let (a_re, a_im) = ctx.quantized_vector(TestVector::First, ctx.meta().log_decimal);
    let (b_re, b_im) = ctx.quantized_vector(TestVector::Second, low_log_decimal);
    let ct1 = ctx.encrypt(ctx.max_k(), &a_re, &a_im, scratch.borrow());
    let ct2 = ctx.encrypt_with_prec(ctx.max_k() - DELTA_LOG_DECIMAL, &b_re, &b_im, low_prec, scratch.borrow());
    let (want_re, want_im) = ctx.want_sub_from(&a_re, &a_im, &b_re, &b_im);
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ctx.module.ckks_sub_into(&mut ct_res, &ct1, &ct2, scratch.borrow()).unwrap();
    assert_binary_output_meta("sub_ct delta_log_decimal", &ct_res, &ct1, &ct2);
    ctx.assert_decrypt_precision_at_log_decimal(
        "sub_ct delta_log_decimal",
        &ct_res,
        &want_re,
        &want_im,
        low_log_decimal,
        scratch.borrow(),
    );
}

/// ct-ct out-of-place, output buffer has smaller max_k than inputs (offset > 0).
///
/// Both inputs are at the same log_hom_rem.  The output is one limb narrower,
/// so both inputs are shifted by one limb before subtraction to fit.
/// Expected result log_hom_rem = input log_hom_rem − base2k.
pub fn test_sub_ct_smaller_output<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let ct2 = ctx.encrypt(ctx.max_k(), &ctx.re2, &ctx.im2, scratch.borrow());
    let (want_re, want_im) = ctx.want_sub();
    let mut ct_res = ctx.alloc_ct(ctx.max_k() - ctx.base2k().as_usize() - 1);
    ctx.module.ckks_sub_into(&mut ct_res, &ct1, &ct2, scratch.borrow()).unwrap();
    assert_binary_output_meta("sub_ct smaller_output", &ct_res, &ct1, &ct2);
    ctx.assert_decrypt_precision("sub_ct smaller_output", &ct_res, &want_re, &want_im, scratch.borrow());
}

// ─── ct-ct in-place (GLWE<_, CKKS>::sub_assign) ────────────────────────────

/// ct-ct in-place, aligned (same log_hom_rem).
pub fn test_sub_ct_assign_aligned<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let mut ct1 = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let ct2 = ctx.encrypt(ctx.max_k(), &ctx.re2, &ctx.im2, scratch.borrow());
    let (want_re, want_im) = ctx.want_sub();
    let expected_log_hom_rem = ct1.log_hom_rem().min(ct2.log_hom_rem());
    let expected_log_decimal = ct1.log_decimal().max(ct2.log_decimal());
    ctx.module.ckks_sub_assign(&mut ct1, &ct2, scratch.borrow()).unwrap();
    assert_ct_meta("sub_ct_assign_aligned", &ct1, expected_log_decimal, expected_log_hom_rem);
    ctx.assert_decrypt_precision("sub_ct_assign_aligned", &ct1, &want_re, &want_im, scratch.borrow());
}

/// ct-ct in-place, self.log_hom_rem() < a.log_hom_rem() (a is shifted down to align with self).
pub fn test_sub_ct_assign_self_lt<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let mut ct_self = ctx.encrypt(
        ctx.max_k() - ctx.base2k().as_usize() - 1,
        &ctx.re1,
        &ctx.im1,
        scratch.borrow(),
    );
    let ct_other = ctx.encrypt(ctx.max_k(), &ctx.re2, &ctx.im2, scratch.borrow());
    let (want_re, want_im) = ctx.want_sub();
    let expected_log_hom_rem = ct_self.log_hom_rem().min(ct_other.log_hom_rem());
    let expected_log_decimal = ct_self.log_decimal().max(ct_other.log_decimal());
    ctx.module.ckks_sub_assign(&mut ct_self, &ct_other, scratch.borrow()).unwrap();
    assert_ct_meta("sub_ct_assign self_lt", &ct_self, expected_log_decimal, expected_log_hom_rem);
    ctx.assert_decrypt_precision("sub_ct_assign self_lt", &ct_self, &want_re, &want_im, scratch.borrow());
}

/// ct-ct in-place, self.log_hom_rem() > a.log_hom_rem() (self is shifted down to align with a).
pub fn test_sub_ct_assign_self_gt<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let mut ct_self = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let ct_other = ctx.encrypt(
        ctx.max_k() - ctx.base2k().as_usize() - 1,
        &ctx.re2,
        &ctx.im2,
        scratch.borrow(),
    );
    let (want_re, want_im) = ctx.want_sub();
    let expected_log_hom_rem = ct_self.log_hom_rem().min(ct_other.log_hom_rem());
    let expected_log_decimal = ct_self.log_decimal().max(ct_other.log_decimal());
    ctx.module.ckks_sub_assign(&mut ct_self, &ct_other, scratch.borrow()).unwrap();
    assert_ct_meta("sub_ct_assign self_gt", &ct_self, expected_log_decimal, expected_log_hom_rem);
    ctx.assert_decrypt_precision("sub_ct_assign self_gt", &ct_self, &want_re, &want_im, scratch.borrow());
}

// ─── ct - compact ZNX plaintext (GLWE<_, CKKS>::sub_pt_vec_znx_into[_assign]) ────────

/// ct - ZNX plaintext, in-place.
pub fn test_sub_pt_vec_znx_assign<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let mut ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let pt_znx = ctx.encode_pt_znx(&ctx.re2, &ctx.im2);
    let (want_re, want_im) = ctx.want_sub();
    let expected_log_decimal = ct.log_decimal();
    let expected_log_hom_rem = ct.log_hom_rem();
    ctx.module
        .ckks_sub_pt_vec_znx_assign(&mut ct, &pt_znx, scratch.borrow())
        .unwrap();
    assert_ct_meta("sub_pt_vec_znx_assign", &ct, expected_log_decimal, expected_log_hom_rem);
    ctx.assert_decrypt_precision("sub_pt_vec_znx_assign", &ct, &want_re, &want_im, scratch.borrow());
}

/// ct - ZNX plaintext, out-of-place.
pub fn test_sub_pt_vec_znx<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let pt_znx = ctx.encode_pt_znx(&ctx.re2, &ctx.im2);
    let (want_re, want_im) = ctx.want_sub();
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ctx.module
        .ckks_sub_pt_vec_znx_into(&mut ct_res, &ct1, &pt_znx, scratch.borrow())
        .unwrap();
    assert_unary_output_meta("sub_pt_vec_znx_into", &ct_res, &ct1);
    ctx.assert_decrypt_precision("sub_pt_vec_znx_into", &ct_res, &want_re, &want_im, scratch.borrow());
}

/// ct - ZNX plaintext, out-of-place, plaintext encoded at lower decimal precision.
pub fn test_sub_pt_vec_znx_into_delta_log_decimal<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let low_log_decimal = ctx.meta().log_decimal - DELTA_LOG_DECIMAL;
    let low_prec = ctx.precision_at(low_log_decimal);
    let (a_re, a_im) = ctx.quantized_vector(TestVector::First, ctx.meta().log_decimal);
    let (b_re, b_im) = ctx.quantized_vector(TestVector::Second, low_log_decimal);
    let ct1 = ctx.encrypt(ctx.max_k(), &a_re, &a_im, scratch.borrow());
    let pt_znx = ctx.encode_pt_znx_with_prec(&b_re, &b_im, low_prec);
    let (want_re, want_im) = ctx.want_sub_from(&a_re, &a_im, &b_re, &b_im);
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ctx.module
        .ckks_sub_pt_vec_znx_into(&mut ct_res, &ct1, &pt_znx, scratch.borrow())
        .unwrap();
    assert_unary_output_meta("sub_pt_vec_znx_into delta_log_decimal", &ct_res, &ct1);
    ctx.assert_decrypt_precision_at_log_decimal(
        "sub_pt_vec_znx_into delta_log_decimal",
        &ct_res,
        &want_re,
        &want_im,
        low_log_decimal,
        scratch.borrow(),
    );
}

// ─── ct - float RNX plaintext (GLWE<_, CKKS>::sub_pt_vec_rnx_into[_assign]) ──────────

/// ct - RNX plaintext, in-place (auto-converts RNX → ZNX using scratch).
pub fn test_sub_pt_vec_rnx_assign<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let mut ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let pt_rnx = ctx.encode_pt_rnx(&ctx.re2, &ctx.im2);
    let (want_re, want_im) = ctx.want_sub();
    let expected_log_decimal = ct.log_decimal();
    let expected_log_hom_rem = ct.log_hom_rem();
    ctx.module
        .ckks_sub_pt_vec_rnx_assign(&mut ct, &pt_rnx, ctx.meta(), scratch.borrow())
        .unwrap();
    assert_ct_meta("sub_pt_vec_rnx_assign", &ct, expected_log_decimal, expected_log_hom_rem);
    ctx.assert_decrypt_precision("sub_pt_vec_rnx_assign", &ct, &want_re, &want_im, scratch.borrow());
}

/// ct - RNX plaintext, out-of-place (auto-converts RNX → ZNX using scratch).
pub fn test_sub_pt_vec_rnx<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let pt_rnx = ctx.encode_pt_rnx(&ctx.re2, &ctx.im2);
    let (want_re, want_im) = ctx.want_sub();
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ctx.module
        .ckks_sub_pt_vec_rnx_into(&mut ct_res, &ct1, &pt_rnx, ctx.meta(), scratch.borrow())
        .unwrap();
    assert_unary_output_meta("sub_pt_vec_rnx_into", &ct_res, &ct1);
    ctx.assert_decrypt_precision("sub_pt_vec_rnx_into", &ct_res, &want_re, &want_im, scratch.borrow());
}

/// ct - RNX plaintext, out-of-place, plaintext encoded at lower decimal precision.
pub fn test_sub_pt_vec_rnx_into_delta_log_decimal<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let low_log_decimal = ctx.meta().log_decimal - DELTA_LOG_DECIMAL;
    let low_prec = ctx.precision_at(low_log_decimal);
    let (a_re, a_im) = ctx.quantized_vector(TestVector::First, ctx.meta().log_decimal);
    let (b_re, b_im) = ctx.quantized_vector(TestVector::Second, low_log_decimal);
    let ct1 = ctx.encrypt(ctx.max_k(), &a_re, &a_im, scratch.borrow());
    let pt_rnx = ctx.encode_pt_rnx(&b_re, &b_im);
    let (want_re, want_im) = ctx.want_sub_from(&a_re, &a_im, &b_re, &b_im);
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ctx.module
        .ckks_sub_pt_vec_rnx_into(&mut ct_res, &ct1, &pt_rnx, low_prec, scratch.borrow())
        .unwrap();
    assert_unary_output_meta("sub_pt_vec_rnx_into delta_log_decimal", &ct_res, &ct1);
    ctx.assert_decrypt_precision_at_log_decimal(
        "sub_pt_vec_rnx_into delta_log_decimal",
        &ct_res,
        &want_re,
        &want_im,
        low_log_decimal,
        scratch.borrow(),
    );
}

/// ct - ZNX plaintext, out-of-place, output buffer has smaller max_k than `a` (offset > 0).
///
/// Exercises the lsh-then-sub path in `sub_pt_vec_znx_into`.  The output log_hom_rem must
/// equal `a.log_hom_rem() − base2k`, not the original `a.log_hom_rem()`.
pub fn test_sub_pt_vec_znx_into_smaller_output<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let pt_znx = ctx.encode_pt_znx(&ctx.re2, &ctx.im2);
    let (want_re, want_im) = ctx.want_sub();
    let mut ct_res = ctx.alloc_ct(ctx.max_k() - ctx.base2k().as_usize() - 1);
    ctx.module
        .ckks_sub_pt_vec_znx_into(&mut ct_res, &ct1, &pt_znx, scratch.borrow())
        .unwrap();
    assert_unary_output_meta("sub_pt_vec_znx_into smaller_output", &ct_res, &ct1);
    ctx.assert_decrypt_precision(
        "sub_pt_vec_znx_into smaller_output",
        &ct_res,
        &want_re,
        &want_im,
        scratch.borrow(),
    );
}

/// ct - RNX plaintext, out-of-place, output buffer has smaller max_k than `a` (offset > 0).
///
/// Same path as `test_sub_pt_vec_znx_into_smaller_output` but entered via `sub_pt_vec_rnx_into`
/// (which converts RNX → ZNX internally before delegating to `sub_pt_vec_znx_into`).
pub fn test_sub_pt_vec_rnx_into_smaller_output<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let pt_rnx = ctx.encode_pt_rnx(&ctx.re2, &ctx.im2);
    let (want_re, want_im) = ctx.want_sub();
    let mut ct_res = ctx.alloc_ct(ctx.max_k() - ctx.base2k().as_usize() - 1);
    ctx.module
        .ckks_sub_pt_vec_rnx_into(&mut ct_res, &ct1, &pt_rnx, ctx.meta(), scratch.borrow())
        .unwrap();
    assert_unary_output_meta("sub_pt_vec_rnx_into smaller_output", &ct_res, &ct1);
    ctx.assert_decrypt_precision(
        "sub_pt_vec_rnx_into smaller_output",
        &ct_res,
        &want_re,
        &want_im,
        scratch.borrow(),
    );
}

pub fn test_sub_pt_const_znx_into_aligned<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let (const_re, const_im) = ctx.quantized_const(CONST_RE, CONST_IM, ctx.meta().log_decimal);
    let want_re: Vec<F> = ctx.re1.iter().map(|x| *x - const_re).collect();
    let want_im: Vec<F> = ctx.im1.iter().map(|x| *x - const_im).collect();
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    let cst_rnx = ctx.const_rnx(Some(CONST_RE), Some(CONST_IM));
    let cst_znx = cst_rnx
        .to_znx_at_k(
            ctx.base2k(),
            ct.log_hom_rem()
                .checked_add(ctx.meta().log_decimal)
                .expect("aligned precision overflow"),
            ctx.meta().log_decimal,
        )
        .unwrap();
    ctx.module
        .ckks_sub_pt_const_znx_into(&mut ct_res, &ct, &cst_znx, scratch.borrow())
        .unwrap();
    assert_unary_output_meta("sub_pt_const_znx_into_aligned", &ct_res, &ct);
    ctx.assert_decrypt_precision("sub_pt_const_znx_into_aligned", &ct_res, &want_re, &want_im, scratch.borrow());
}
