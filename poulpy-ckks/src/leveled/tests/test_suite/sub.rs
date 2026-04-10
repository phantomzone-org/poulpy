//! Subtraction tests: ct-ct, ct-pt, ct-const (out-of-place and in-place).
//!
//! # Test inventory
//!
//! ## Operations-layer ct-ct (`CKKSCiphertext::sub`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_sub_ct_aligned`] | `a.log_delta == b.log_delta`, `offset == 0` → `glwe_sub` fast path |
//! | [`test_sub_ct_delta_a_lt_b`] | `a.log_delta < b.log_delta` → b shifted to align with a |
//! | [`test_sub_ct_delta_a_gt_b`] | `a.log_delta > b.log_delta` → a shifted to align with b |
//! | [`test_sub_ct_smaller_output`] | `offset > 0` (output one limb narrower than inputs) |
//!
//! ## Operations-layer ct-ct (`CKKSCiphertext::sub_inplace`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_sub_ct_inplace_aligned`] | `self.log_delta == a.log_delta` |
//! | [`test_sub_ct_inplace_self_lt`] | `self.log_delta < a.log_delta` → a shifted to align with self |
//! | [`test_sub_ct_inplace_self_gt`] | `self.log_delta > a.log_delta` → self shifted to align with a |
//!
//! ## Operations-layer ct - ZNX plaintext (`CKKSCiphertext::sub_pt_znx[_inplace]`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_sub_pt_znx_inplace`] | in-place, `offset == 0` |
//! | [`test_sub_pt_znx`] | out-of-place, `offset == 0` |
//! | [`test_sub_pt_znx_smaller_output`] | out-of-place, `offset > 0` (output one limb narrower) |
//!
//! ## Operations-layer ct + RNX plaintext (`CKKSCiphertext::sub_pt_rnx[_inplace]`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_sub_pt_rnx_inplace`] | in-place, `offset == 0`, RNX → ZNX auto-conversion |
//! | [`test_sub_pt_rnx`] | out-of-place, `offset == 0`, RNX → ZNX auto-conversion |
//! | [`test_sub_pt_rnx_smaller_output`] | out-of-place, `offset > 0` (output one limb narrower) |

use super::helpers::TestContext;
use poulpy_core::{
    GLWECopy, GLWEDecrypt, GLWEEncryptSk, GLWEShift, GLWESub, ScratchTakeCore, layouts::GLWESecretPreparedFactory,
};
use poulpy_hal::{
    api::{
        ModuleN, ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxLsh, VecZnxNormalize, VecZnxNormalizeTmpBytes,
        VecZnxRshAdd, VecZnxRshSub,
    },
    layouts::{Backend, Module, Scratch, ScratchOwned},
};

// All test functions share this where clause:
//
//   Module<BE>: ModuleN + GLWEEncryptSk<BE> + GLWEDecrypt<BE>
//             + GLWESecretPreparedFactory<BE> + GLWESub + GLWECopy + GLWEShift<BE>
//             + VecZnxNormalize<BE> + VecZnxNormalizeTmpBytes + VecZnxLsh<BE> + VecZnxRshSub<BE>
//   ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>
//   Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable

// ─── ct-ct out-of-place (CKKSCiphertext::sub) ────────────────────────────────

/// ct-ct out-of-place, aligned (same log_delta, offset == 0 → glwe_sub fast path).
pub fn test_sub_ct_aligned<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESub
        + GLWECopy
        + GLWEShift<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxLsh<BE>
        + VecZnxRshAdd<BE>
        + VecZnxRshSub<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(&ctx.re1, &ctx.im1, scratch.borrow());
    let ct2 = ctx.encrypt(&ctx.re2, &ctx.im2, scratch.borrow());
    let (want_re, want_im) = ctx.want_sub();
    let mut ct_res = ctx.alloc_ct();
    ct_res.sub(&ctx.module, &ct1, &ct2, scratch.borrow()).unwrap();

    assert_eq!(ct_res.log_delta, ct1.log_delta, "aligned: log_delta must equal inputs");
    ctx.assert_decrypt_precision("sub_ct_aligned", &ct_res, &want_re, &want_im, 20.0, scratch.borrow());
}

/// ct-ct out-of-place, a.log_delta < b.log_delta (b is shifted to align with a).
pub fn test_sub_ct_delta_a_lt_b<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESub
        + GLWECopy
        + GLWEShift<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxLsh<BE>
        + VecZnxRshAdd<BE>
        + VecZnxRshSub<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    let mut scratch = ctx.alloc_scratch();
    let mut ct1 = ctx.encrypt(&ctx.re1, &ctx.im1, scratch.borrow());
    let ct2 = ctx.encrypt(&ctx.re2, &ctx.im2, scratch.borrow());

    // Drop ct1 by one limb: ct1.log_delta becomes ct2.log_delta - base2k.
    ct1.rescale_inplace(&ctx.module, ctx.params.base2k, scratch.borrow());
    let expected_delta = ct1.log_delta;
    let (want_re, want_im) = ctx.want_sub();

    let mut ct_res = ctx.alloc_ct();
    ct_res.sub(&ctx.module, &ct1, &ct2, scratch.borrow()).unwrap();

    assert_eq!(ct_res.log_delta, expected_delta, "a_lt_b: result must use min log_delta");
    ctx.assert_decrypt_precision("sub_ct a_lt_b", &ct_res, &want_re, &want_im, 18.0, scratch.borrow());
}

/// ct-ct out-of-place, a.log_delta > b.log_delta (a is shifted to align with b).
pub fn test_sub_ct_delta_a_gt_b<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESub
        + GLWECopy
        + GLWEShift<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxLsh<BE>
        + VecZnxRshAdd<BE>
        + VecZnxRshSub<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(&ctx.re1, &ctx.im1, scratch.borrow());
    let mut ct2 = ctx.encrypt(&ctx.re2, &ctx.im2, scratch.borrow());

    // Drop ct2: ct2.log_delta becomes ct1.log_delta - base2k.
    ct2.rescale_inplace(&ctx.module, ctx.params.base2k, scratch.borrow());
    let expected_delta = ct2.log_delta;
    let (want_re, want_im) = ctx.want_sub();

    let mut ct_res = ctx.alloc_ct();
    ct_res.sub(&ctx.module, &ct1, &ct2, scratch.borrow()).unwrap();

    assert_eq!(ct_res.log_delta, expected_delta, "a_gt_b: result must use min log_delta");
    ctx.assert_decrypt_precision("sub_ct a_gt_b", &ct_res, &want_re, &want_im, 18.0, scratch.borrow());
}

/// ct-ct out-of-place, output buffer has smaller max_k than inputs (offset > 0).
///
/// Both inputs are at the same log_delta.  The output is one limb narrower,
/// so both inputs are shifted by one limb before subtraction to fit.
/// Expected result log_delta = input log_delta − base2k.
pub fn test_sub_ct_smaller_output<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESub
        + GLWECopy
        + GLWEShift<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxLsh<BE>
        + VecZnxRshAdd<BE>
        + VecZnxRshSub<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(&ctx.re1, &ctx.im1, scratch.borrow());
    let ct2 = ctx.encrypt(&ctx.re2, &ctx.im2, scratch.borrow());
    let (want_re, want_im) = ctx.want_sub();

    let mut ct_res = ctx.alloc_ct_reduced_k();
    ct_res.sub(&ctx.module, &ct1, &ct2, scratch.borrow()).unwrap();

    let expected_delta = ct1.log_delta - ctx.params.base2k;
    assert_eq!(
        ct_res.log_delta, expected_delta,
        "smaller_output: log_delta must be reduced by offset"
    );
    ctx.assert_decrypt_precision("sub_ct smaller_output", &ct_res, &want_re, &want_im, 18.0, scratch.borrow());
}

// ─── ct-ct in-place (CKKSCiphertext::sub_inplace) ────────────────────────────

/// ct-ct in-place, aligned (same log_delta).
pub fn test_sub_ct_inplace_aligned<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESub
        + GLWECopy
        + GLWEShift<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxLsh<BE>
        + VecZnxRshAdd<BE>
        + VecZnxRshSub<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    let mut scratch = ctx.alloc_scratch();
    let mut ct1 = ctx.encrypt(&ctx.re1, &ctx.im1, scratch.borrow());
    let ct2 = ctx.encrypt(&ctx.re2, &ctx.im2, scratch.borrow());
    let expected_delta = ct1.log_delta;
    let (want_re, want_im) = ctx.want_sub();

    ct1.sub_inplace(&ctx.module, &ct2, scratch.borrow()).unwrap();

    assert_eq!(
        ct1.log_delta, expected_delta,
        "sub_inplace aligned: log_delta must be unchanged"
    );
    ctx.assert_decrypt_precision("sub_ct_inplace_aligned", &ct1, &want_re, &want_im, 20.0, scratch.borrow());
}

/// ct-ct in-place, self.log_delta < a.log_delta (a is shifted down to align with self).
pub fn test_sub_ct_inplace_self_lt<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESub
        + GLWECopy
        + GLWEShift<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxLsh<BE>
        + VecZnxRshAdd<BE>
        + VecZnxRshSub<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    let mut scratch = ctx.alloc_scratch();
    let mut ct_self = ctx.encrypt(&ctx.re1, &ctx.im1, scratch.borrow());
    let ct_other = ctx.encrypt(&ctx.re2, &ctx.im2, scratch.borrow());

    // Rescale self: self.log_delta < ct_other.log_delta.
    ct_self.rescale_inplace(&ctx.module, ctx.params.base2k, scratch.borrow());
    let expected_delta = ct_self.log_delta;
    let (want_re, want_im) = ctx.want_sub();

    ct_self.sub_inplace(&ctx.module, &ct_other, scratch.borrow()).unwrap();

    assert_eq!(
        ct_self.log_delta, expected_delta,
        "sub_inplace self_lt: log_delta must stay at self's value"
    );
    ctx.assert_decrypt_precision("sub_ct_inplace self_lt", &ct_self, &want_re, &want_im, 18.0, scratch.borrow());
}

/// ct-ct in-place, self.log_delta > a.log_delta (self is shifted down to align with a).
pub fn test_sub_ct_inplace_self_gt<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESub
        + GLWECopy
        + GLWEShift<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxLsh<BE>
        + VecZnxRshAdd<BE>
        + VecZnxRshSub<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    let mut scratch = ctx.alloc_scratch();
    let mut ct_self = ctx.encrypt(&ctx.re1, &ctx.im1, scratch.borrow());
    let mut ct_other = ctx.encrypt(&ctx.re2, &ctx.im2, scratch.borrow());

    // Rescale other: ct_other.log_delta < ct_self.log_delta.
    ct_other.rescale_inplace(&ctx.module, ctx.params.base2k, scratch.borrow());
    let expected_delta = ct_other.log_delta;
    let (want_re, want_im) = ctx.want_sub();

    ct_self.sub_inplace(&ctx.module, &ct_other, scratch.borrow()).unwrap();

    assert_eq!(
        ct_self.log_delta, expected_delta,
        "sub_inplace self_gt: log_delta must drop to a's value"
    );
    ctx.assert_decrypt_precision("sub_ct_inplace self_gt", &ct_self, &want_re, &want_im, 18.0, scratch.borrow());
}

// ─── ct - compact ZNX plaintext (CKKSCiphertext::sub_pt_znx[_inplace]) ────────

/// ct - ZNX plaintext, in-place.
pub fn test_sub_pt_znx_inplace<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESub
        + GLWECopy
        + GLWEShift<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxLsh<BE>
        + VecZnxRshAdd<BE>
        + VecZnxRshSub<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    let mut scratch = ctx.alloc_scratch();
    let mut ct = ctx.encrypt(&ctx.re1, &ctx.im1, scratch.borrow());
    let pt_znx = ctx.encode_pt_znx();
    let expected_delta = ct.log_delta;
    let (want_re, want_im) = ctx.want_sub();

    ct.sub_pt_znx_inplace(&ctx.module, &pt_znx, scratch.borrow()).unwrap();

    assert_eq!(ct.log_delta, expected_delta, "sub_pt_znx_inplace must not change log_delta");
    ctx.assert_decrypt_precision("sub_pt_znx_inplace", &ct, &want_re, &want_im, 20.0, scratch.borrow());
}

/// ct - ZNX plaintext, out-of-place.
pub fn test_sub_pt_znx<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESub
        + GLWECopy
        + GLWEShift<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxLsh<BE>
        + VecZnxRshAdd<BE>
        + VecZnxRshSub<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(&ctx.re1, &ctx.im1, scratch.borrow());
    let pt_znx = ctx.encode_pt_znx();
    let (want_re, want_im) = ctx.want_sub();

    let mut ct_res = ctx.alloc_ct();
    ct_res.sub_pt_znx(&ctx.module, &ct1, &pt_znx, scratch.borrow()).unwrap();

    assert_eq!(ct_res.log_delta, ct1.log_delta, "sub_pt_znx must carry forward a's log_delta");
    ctx.assert_decrypt_precision("sub_pt_znx", &ct_res, &want_re, &want_im, 20.0, scratch.borrow());
}

// ─── ct - float RNX plaintext (CKKSCiphertext::sub_pt_rnx[_inplace]) ──────────

/// ct - RNX plaintext, in-place (auto-converts RNX → ZNX using scratch).
pub fn test_sub_pt_rnx_inplace<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESub
        + GLWECopy
        + GLWEShift<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxLsh<BE>
        + VecZnxRshAdd<BE>
        + VecZnxRshSub<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    let mut scratch = ctx.alloc_scratch();
    let mut ct = ctx.encrypt(&ctx.re1, &ctx.im1, scratch.borrow());
    let pt_rnx = ctx.encode_pt_rnx();
    let expected_delta = ct.log_delta;
    let (want_re, want_im) = ctx.want_sub();

    ct.sub_pt_rnx_inplace(&ctx.module, &pt_rnx, ctx.prec(), scratch.borrow())
        .unwrap();

    assert_eq!(ct.log_delta, expected_delta, "sub_pt_rnx_inplace must not change log_delta");
    ctx.assert_decrypt_precision("sub_pt_rnx_inplace", &ct, &want_re, &want_im, 20.0, scratch.borrow());
}

/// ct - RNX plaintext, out-of-place (auto-converts RNX → ZNX using scratch).
pub fn test_sub_pt_rnx<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESub
        + GLWECopy
        + GLWEShift<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxLsh<BE>
        + VecZnxRshAdd<BE>
        + VecZnxRshSub<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(&ctx.re1, &ctx.im1, scratch.borrow());
    let pt_rnx = ctx.encode_pt_rnx();
    let (want_re, want_im) = ctx.want_sub();

    let mut ct_res = ctx.alloc_ct();
    ct_res
        .sub_pt_rnx(&ctx.module, &ct1, &pt_rnx, ctx.prec(), scratch.borrow())
        .unwrap();

    assert_eq!(ct_res.log_delta, ct1.log_delta, "sub_pt_rnx must carry forward a's log_delta");
    ctx.assert_decrypt_precision("sub_pt_rnx", &ct_res, &want_re, &want_im, 20.0, scratch.borrow());
}

/// ct - ZNX plaintext, out-of-place, output buffer has smaller max_k than `a` (offset > 0).
///
/// Exercises the lsh-then-sub path in `sub_pt_znx`.  The output log_delta must
/// equal `a.log_delta − base2k`, not the original `a.log_delta`.
pub fn test_sub_pt_znx_smaller_output<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESub
        + GLWECopy
        + GLWEShift<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxLsh<BE>
        + VecZnxRshAdd<BE>
        + VecZnxRshSub<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(&ctx.re1, &ctx.im1, scratch.borrow());

    let pt_znx = ctx.encode_pt_znx();
    let (want_re, want_im) = ctx.want_sub();

    let mut ct_res = ctx.alloc_ct_reduced_k();
    ct_res.sub_pt_znx(&ctx.module, &ct1, &pt_znx, scratch.borrow()).unwrap();

    let expected_delta = ct1.log_delta - ctx.params.base2k;
    assert_eq!(
        ct_res.log_delta, expected_delta,
        "sub_pt_znx smaller_output: log_delta must be reduced by offset"
    );
    ctx.assert_decrypt_precision(
        "sub_pt_znx smaller_output",
        &ct_res,
        &want_re,
        &want_im,
        18.0,
        scratch.borrow(),
    );
}

/// ct - RNX plaintext, out-of-place, output buffer has smaller max_k than `a` (offset > 0).
///
/// Same path as `test_sub_pt_znx_smaller_output` but entered via `sub_pt_rnx`
/// (which converts RNX → ZNX internally before delegating to `sub_pt_znx`).
pub fn test_sub_pt_rnx_smaller_output<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESub
        + GLWECopy
        + GLWEShift<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxLsh<BE>
        + VecZnxRshAdd<BE>
        + VecZnxRshSub<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(&ctx.re1, &ctx.im1, scratch.borrow());
    let pt_rnx = ctx.encode_pt_rnx();
    let (want_re, want_im) = ctx.want_sub();

    let mut ct_res = ctx.alloc_ct_reduced_k();
    ct_res
        .sub_pt_rnx(&ctx.module, &ct1, &pt_rnx, ctx.prec(), scratch.borrow())
        .unwrap();

    let expected_delta = ct1.log_delta - ctx.params.base2k;
    assert_eq!(
        ct_res.log_delta, expected_delta,
        "sub_pt_rnx smaller_output: log_delta must be reduced by offset"
    );
    ctx.assert_decrypt_precision(
        "sub_pt_rnx smaller_output",
        &ct_res,
        &want_re,
        &want_im,
        18.0,
        scratch.borrow(),
    );
}
