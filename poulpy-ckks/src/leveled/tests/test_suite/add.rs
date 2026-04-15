//! Addition tests: ct+ct, ct+pt, ct+const (out-of-place and in-place).
//!
//! # Test inventory
//!
//! ## Operations-layer ct+ct (`GLWE<_, CKKS>::add`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_add_ct_aligned`] | `a.log_hom_rem() == b.log_hom_rem()`, `offset == 0` → `glwe_add` fast path |
//! | [`test_add_ct_delta_a_lt_b`] | `a.log_hom_rem() < b.log_hom_rem()` → b shifted to align with a |
//! | [`test_add_ct_delta_a_gt_b`] | `a.log_hom_rem() > b.log_hom_rem()` → a shifted to align with b |
//! | [`test_add_ct_smaller_output`] | `offset > 0` (output one limb narrower than inputs) |
//!
//! ## Operations-layer ct+ct (`GLWE<_, CKKS>::add_inplace`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_add_ct_inplace_aligned`] | `self.log_hom_rem() == a.log_hom_rem()` |
//! | [`test_add_ct_inplace_self_lt`] | `self.log_hom_rem() < a.log_hom_rem()` → a shifted to align with self |
//! | [`test_add_ct_inplace_self_gt`] | `self.log_hom_rem() > a.log_hom_rem()` → self shifted to align with a |
//!
//! ## Operations-layer ct + ZNX plaintext (`GLWE<_, CKKS>::add_pt_znx[_inplace]`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_add_pt_znx_inplace`] | in-place, `offset == 0` |
//! | [`test_add_pt_znx`] | out-of-place, `offset == 0` |
//! | [`test_add_pt_znx_smaller_output`] | out-of-place, `offset > 0` (output one limb narrower) |
//!
//! ## Operations-layer ct + RNX plaintext (`GLWE<_, CKKS>::add_pt_rnx[_inplace]`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_add_pt_rnx_inplace`] | in-place, `offset == 0`, RNX → ZNX auto-conversion |
//! | [`test_add_pt_rnx`] | out-of-place, `offset == 0`, RNX → ZNX auto-conversion |
//! | [`test_add_pt_rnx_smaller_output`] | out-of-place, `offset > 0` (output one limb narrower) |

use crate::leveled::operations::add::CKKSAddOps;

use super::helpers::{TestAddBackend as Backend, TestContext};
use poulpy_hal::api::ScratchOwnedBorrow;

// ─── ct+ct out-of-place (GLWE<_, CKKS>::add) ────────────────────────────────

/// ct+ct out-of-place, aligned (same log_hom_rem(), offset == 0 → glwe_add fast path).
pub fn test_add_ct_aligned<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let ct2 = ctx.encrypt(ctx.max_k(), &ctx.re2, &ctx.im2, scratch.borrow());
    let (want_re, want_im) = ctx.want_add();
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ct_res.add(&ctx.module, &ct1, &ct2, scratch.borrow()).unwrap();
    ctx.assert_decrypt_precision("add_ct_aligned", &ct_res, &want_re, &want_im, 20.0, scratch.borrow());
}

/// ct+ct out-of-place, a.log_hom_rem() < b.log_hom_rem() (b is shifted to align with a).
pub fn test_add_ct_delta_a_lt_b<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(
        ctx.max_k() - ctx.base2k().as_usize() + 1,
        &ctx.re1,
        &ctx.im1,
        scratch.borrow(),
    );
    let ct2 = ctx.encrypt(ctx.max_k(), &ctx.re2, &ctx.im2, scratch.borrow());
    let (want_re, want_im) = ctx.want_add();
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ct_res.add(&ctx.module, &ct1, &ct2, scratch.borrow()).unwrap();
    ctx.assert_decrypt_precision("add_ct a_lt_b", &ct_res, &want_re, &want_im, 20.0, scratch.borrow());
}

/// ct+ct out-of-place, a.log_hom_rem() > b.log_hom_rem() (a is shifted to align with b).
pub fn test_add_ct_delta_a_gt_b<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let ct2 = ctx.encrypt(
        ctx.max_k() - ctx.base2k().as_usize() + 1,
        &ctx.re2,
        &ctx.im2,
        scratch.borrow(),
    );
    let (want_re, want_im) = ctx.want_add();
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ct_res.add(&ctx.module, &ct1, &ct2, scratch.borrow()).unwrap();
    ctx.assert_decrypt_precision("add_ct a_gt_b", &ct_res, &want_re, &want_im, 20.0, scratch.borrow());
}

/// ct+ct out-of-place, output buffer has smaller max_k than inputs (offset > 0).
///
/// Both inputs are at the same log_hom_rem().  The output is one limb narrower,
/// so both inputs are shifted by one limb before addition to fit.
/// Expected result log_hom_rem() = input log_hom_rem() − base2k.
pub fn test_add_ct_aligned_smaller_output<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let ct2 = ctx.encrypt(ctx.max_k(), &ctx.re2, &ctx.im2, scratch.borrow());
    let (want_re, want_im) = ctx.want_add();
    let mut ct_res = ctx.alloc_ct(ctx.max_k() - ctx.base2k().as_usize() - 1);
    ct_res.add(&ctx.module, &ct1, &ct2, scratch.borrow()).unwrap();
    ctx.assert_decrypt_precision("add_ct smaller_output", &ct_res, &want_re, &want_im, 20.0, scratch.borrow());
}

// ─── ct+ct in-place (GLWE<_, CKKS>::add_inplace) ────────────────────────────

/// ct+ct in-place, aligned (same log_hom_rem()).
pub fn test_add_ct_inplace_aligned<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let mut ct1 = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let ct2 = ctx.encrypt(ctx.max_k(), &ctx.re2, &ctx.im2, scratch.borrow());
    let (want_re, want_im) = ctx.want_add();
    ct1.add_inplace(&ctx.module, &ct2, scratch.borrow()).unwrap();
    ctx.assert_decrypt_precision("add_ct_inplace_aligned", &ct1, &want_re, &want_im, 20.0, scratch.borrow());
}

/// ct+ct in-place, self.log_hom_rem() < a.log_hom_rem() (a is shifted down to align with self).
pub fn test_add_ct_inplace_self_lt<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let mut ct_self = ctx.encrypt(
        ctx.max_k() - ctx.base2k().as_usize() - 1,
        &ctx.re1,
        &ctx.im1,
        scratch.borrow(),
    );
    let ct_other = ctx.encrypt(ctx.max_k(), &ctx.re2, &ctx.im2, scratch.borrow());
    let (want_re, want_im) = ctx.want_add();
    ct_self.add_inplace(&ctx.module, &ct_other, scratch.borrow()).unwrap();
    ctx.assert_decrypt_precision("add_ct_inplace self_lt", &ct_self, &want_re, &want_im, 20.0, scratch.borrow());
}

/// ct+ct in-place, self.log_hom_rem() > a.log_hom_rem() (self is shifted down to align with a).
pub fn test_add_ct_inplace_self_gt<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let mut ct_self = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let ct_other = ctx.encrypt(
        ctx.max_k() - ctx.base2k().as_usize() - 1,
        &ctx.re2,
        &ctx.im2,
        scratch.borrow(),
    );
    let (want_re, want_im) = ctx.want_add();
    ct_self.add_inplace(&ctx.module, &ct_other, scratch.borrow()).unwrap();
    ctx.assert_decrypt_precision("add_ct_inplace self_gt", &ct_self, &want_re, &want_im, 20.0, scratch.borrow());
}

// ─── ct + compact ZNX plaintext (GLWE<_, CKKS>::add_pt_znx[_inplace]) ────────

/// ct + ZNX plaintext, in-place.
pub fn test_add_pt_znx_inplace<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let mut ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let pt_znx = ctx.encode_pt_znx();
    let (want_re, want_im) = ctx.want_add();
    ct.add_pt_znx_inplace(&ctx.module, &pt_znx, scratch.borrow()).unwrap();
    ctx.assert_decrypt_precision("add_pt_znx_inplace", &ct, &want_re, &want_im, 20.0, scratch.borrow());
}

/// ct + ZNX plaintext, out-of-place.
pub fn test_add_pt_znx_aligned<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let pt_znx = ctx.encode_pt_znx();
    let (want_re, want_im) = ctx.want_add();
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ct_res.add_pt_znx(&ctx.module, &ct1, &pt_znx, scratch.borrow()).unwrap();
    ctx.assert_decrypt_precision("add_pt_znx", &ct_res, &want_re, &want_im, 20.0, scratch.borrow());
}

// ─── ct + float RNX plaintext (GLWE<_, CKKS>::add_pt_rnx[_inplace]) ──────────

/// ct + RNX plaintext, in-place (auto-converts RNX → ZNX using scratch).
pub fn test_add_pt_rnx_inplace<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let mut ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let pt_rnx = ctx.encode_pt_rnx();
    let (want_re, want_im) = ctx.want_add();
    ct.add_pt_rnx_inplace(&ctx.module, &pt_rnx, ctx.meta(), scratch.borrow())
        .unwrap();
    ctx.assert_decrypt_precision("add_pt_rnx_inplace", &ct, &want_re, &want_im, 20.0, scratch.borrow());
}

/// ct + RNX plaintext, out-of-place (auto-converts RNX → ZNX using scratch).
pub fn test_add_pt_rnx_aligned<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let pt_rnx = ctx.encode_pt_rnx();
    let (want_re, want_im) = ctx.want_add();
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ct_res
        .add_pt_rnx(&ctx.module, &ct1, &pt_rnx, ctx.meta(), scratch.borrow())
        .unwrap();
    ctx.assert_decrypt_precision("add_pt_rnx", &ct_res, &want_re, &want_im, 20.0, scratch.borrow());
}

/// ct + ZNX plaintext, out-of-place, output buffer has smaller max_k than `a` (offset > 0).
///
/// Exercises the lsh-then-add path in `add_pt_znx`.  The output log_hom_rem() must
/// equal `a.log_hom_rem() − base2k`, not the original `a.log_hom_rem()`.
pub fn test_add_pt_znx_smaller_output<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let pt_znx = ctx.encode_pt_znx();
    let (want_re, want_im) = ctx.want_add();
    let mut ct_res = ctx.alloc_ct(ctx.max_k() - ctx.base2k().as_usize() - 1);
    ct_res.add_pt_znx(&ctx.module, &ct1, &pt_znx, scratch.borrow()).unwrap();
    ctx.assert_decrypt_precision(
        "add_pt_znx smaller_output",
        &ct_res,
        &want_re,
        &want_im,
        20.0,
        scratch.borrow(),
    );
}

/// ct + RNX plaintext, out-of-place, output buffer has smaller max_k than `a` (offset > 0).
///
/// Same path as `test_add_pt_znx_smaller_output` but entered via `add_pt_rnx`
/// (which converts RNX → ZNX internally before delegating to `add_pt_znx`).
pub fn test_add_pt_rnx_smaller_output<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let pt_rnx = ctx.encode_pt_rnx();
    let (want_re, want_im) = ctx.want_add();
    let mut ct_res = ctx.alloc_ct(ctx.max_k() - ctx.base2k().as_usize() - 1);
    ct_res
        .add_pt_rnx(&ctx.module, &ct1, &pt_rnx, ctx.meta(), scratch.borrow())
        .unwrap();
    ctx.assert_decrypt_precision(
        "add_pt_rnx smaller_output",
        &ct_res,
        &want_re,
        &want_im,
        20.0,
        scratch.borrow(),
    );
}
