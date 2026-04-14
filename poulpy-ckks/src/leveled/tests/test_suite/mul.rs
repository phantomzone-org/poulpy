//! Multiplication tests: ct × ct and ct² (square).
//!
//! # Test inventory
//!
//! ## ct × ct multiplication (`GLWE<_, CKKS>::mul`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_mul_ct_aligned`] | both inputs at same `log_hom_rem()` |
//! | `test_mul_ct_delta_a_lt_b` | *(commented out)* `a.log_hom_rem() < b.log_hom_rem()` — requires alignment |
//! | `test_mul_ct_delta_a_gt_b` | *(commented out)* `a.log_hom_rem() > b.log_hom_rem()` — requires alignment |
//! | [`test_mul_ct_smaller_output`] | output has smaller `max_k()` than inputs |
//!
//! ## ct² squaring (`GLWE<_, CKKS>::square`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_square_ct_aligned`] | square at default precision |
//! | [`test_square_ct_rescaled_input`] | square after a rescale (reduced `log_hom_rem()`) |
//! | [`test_square_ct_smaller_output`] | square into smaller output buffer |

use crate::CKKSInfos;
use crate::layouts::CKKSRescaleOps;
use crate::leveled::{
    encryption::CKKSDecrypt,
    operations::mul::CKKSMulOps,
    tests::test_suite::helpers::{TestBackend as Backend, TestContext},
};

use poulpy_core::{
    GLWEDecrypt, GLWEEncryptSk, GLWEShift, GLWETensorDecrypt, GLWETensoring, ScratchTakeCore,
    layouts::GLWESecretTensorPreparedFactory,
};
use poulpy_hal::{
    api::{
        Convolution, ModuleN, ModuleNew, ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxBigAddAssign,
        VecZnxBigNormalize, VecZnxCopy, VecZnxIdftApplyConsume, VecZnxLshInplace, VecZnxNormalize, VecZnxNormalizeTmpBytes,
        VecZnxRshAddInto,
    },
    layouts::{Module, Scratch, ScratchOwned},
};

// All test functions share this where clause:
//
//   Module<BE>: ModuleN + ModuleNew<BE> + VecZnxRshAddInto<BE>
//             + VecZnxNormalize<BE> + VecZnxNormalizeTmpBytes
//             + GLWEEncryptSk<BE> + GLWEDecrypt<BE> + GLWEShift<BE>
//             + GLWETensoring<BE> + VecZnxLshInplace<BE> + VecZnxCopy
//             + GLWESecretTensorPreparedFactory<BE> + GLWETensorDecrypt<BE>
//             + Convolution<BE> + VecZnxBigAddAssign<BE> + VecZnxBigNormalize<BE>
//             + VecZnxIdftApplyConsume<BE> + CKKSDecrypt<BE>
//   ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>
//   Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable

// ─── ct × ct out-of-place (GLWE<_, CKKS>::mul) ─────────────────────────────────

/// ct × ct multiplication with both inputs at the same log_hom_rem().
pub fn test_mul_ct_aligned<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + ModuleNew<BE>
        + VecZnxRshAddInto<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWEShift<BE>
        + GLWETensoring<BE>
        + VecZnxLshInplace<BE>
        + VecZnxCopy
        + GLWESecretTensorPreparedFactory<BE>
        + GLWETensorDecrypt<BE>
        + Convolution<BE>
        + VecZnxBigAddAssign<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxIdftApplyConsume<BE>
        + CKKSDecrypt<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(&ctx.re1, &ctx.im1, scratch.borrow());
    let ct2 = ctx.encrypt(&ctx.re2, &ctx.im2, scratch.borrow());

    let (want_re, want_im) = ctx.want_mul();

    let mut ct_res = ctx.alloc_ct();
    ct_res.mul(&ctx.module, &ct1, &ct2, ctx.tsk(), scratch.borrow()).unwrap();

    ctx.assert_decrypt_precision("mul_ct_aligned", &ct_res, &want_re, &want_im, 20.0, scratch.borrow());
}

// TODO: mul does not yet align inputs with mismatched log_hom_rem() (unlike add).
//       The offset computation uses only `a`'s metadata, so these tests produce
//       garbage when a.log_hom_rem() != b.log_hom_rem().

// /// ct × ct, a.log_hom_rem() < b.log_hom_rem() (a rescaled by one limb).
// pub fn test_mul_ct_delta_a_lt_b<BE: Backend>(ctx: &TestContext<BE>) { ... }

// /// ct × ct, a.log_hom_rem() > b.log_hom_rem() (b rescaled by one limb).
// pub fn test_mul_ct_delta_a_gt_b<BE: Backend>(ctx: &TestContext<BE>) { ... }

/// ct × ct, output buffer has smaller max_k than inputs (offset > 0).
pub fn test_mul_ct_smaller_output<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + ModuleNew<BE>
        + VecZnxRshAddInto<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWEShift<BE>
        + GLWETensoring<BE>
        + VecZnxLshInplace<BE>
        + VecZnxCopy
        + GLWESecretTensorPreparedFactory<BE>
        + GLWETensorDecrypt<BE>
        + Convolution<BE>
        + VecZnxBigAddAssign<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxIdftApplyConsume<BE>
        + CKKSDecrypt<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(&ctx.re1, &ctx.im1, scratch.borrow());
    let ct2 = ctx.encrypt(&ctx.re2, &ctx.im2, scratch.borrow());
    let (want_re, want_im) = ctx.want_mul();

    let mut ct_res = ctx.alloc_ct_reduced_k();
    ct_res.mul(&ctx.module, &ct1, &ct2, ctx.tsk(), scratch.borrow()).unwrap();

    ctx.assert_decrypt_precision("mul_ct smaller_output", &ct_res, &want_re, &want_im, 18.0, scratch.borrow());
}

// ─── ct² squaring (GLWE<_, CKKS>::square) ───────────────────────────────────────

/// ct² at default precision (same as fresh encryption).
pub fn test_square_ct_aligned<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + ModuleNew<BE>
        + VecZnxRshAddInto<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWEShift<BE>
        + GLWETensoring<BE>
        + VecZnxLshInplace<BE>
        + VecZnxCopy
        + GLWESecretTensorPreparedFactory<BE>
        + GLWETensorDecrypt<BE>
        + Convolution<BE>
        + VecZnxBigAddAssign<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxIdftApplyConsume<BE>
        + CKKSDecrypt<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    let mut scratch = ctx.alloc_scratch();
    let ct = ctx.encrypt(&ctx.re1, &ctx.im1, scratch.borrow());
    let (want_re, want_im) = ctx.want_square();

    let mut ct_res = ctx.alloc_ct();
    ct_res.square(&ctx.module, &ct, ctx.tsk(), scratch.borrow()).unwrap();

    ctx.assert_decrypt_precision("square_ct_aligned", &ct_res, &want_re, &want_im, 20.0, scratch.borrow());
}

/// ct² after rescaling the input by one limb (reduced log_hom_rem()).
pub fn test_square_ct_rescaled_input<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + ModuleNew<BE>
        + VecZnxRshAddInto<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWEShift<BE>
        + GLWETensoring<BE>
        + VecZnxLshInplace<BE>
        + VecZnxCopy
        + GLWESecretTensorPreparedFactory<BE>
        + GLWETensorDecrypt<BE>
        + Convolution<BE>
        + VecZnxBigAddAssign<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxIdftApplyConsume<BE>
        + CKKSDecrypt<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    let mut scratch = ctx.alloc_scratch();
    let mut ct = ctx.encrypt(&ctx.re1, &ctx.im1, scratch.borrow());

    // Drop by one limb before squaring.
    ct.rescale_inplace(&ctx.module, ctx.params.base2k, scratch.borrow()).unwrap();
    let (want_re, want_im) = ctx.want_square();

    let mut ct_res = ctx.alloc_ct();
    ct_res.square(&ctx.module, &ct, ctx.tsk(), scratch.borrow()).unwrap();

    let expected_lhr = ct.log_hom_rem() - ct.log_decimal();
    assert_eq!(ct_res.log_hom_rem(), expected_lhr, "rescaled: result log_hom_rem()");
    ctx.assert_decrypt_precision("square_ct rescaled", &ct_res, &want_re, &want_im, 18.0, scratch.borrow());
}

/// ct² into an output buffer with smaller max_k (one limb narrower).
pub fn test_square_ct_smaller_output<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + ModuleNew<BE>
        + VecZnxRshAddInto<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWEShift<BE>
        + GLWETensoring<BE>
        + VecZnxLshInplace<BE>
        + VecZnxCopy
        + GLWESecretTensorPreparedFactory<BE>
        + GLWETensorDecrypt<BE>
        + Convolution<BE>
        + VecZnxBigAddAssign<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxIdftApplyConsume<BE>
        + CKKSDecrypt<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    let mut scratch = ctx.alloc_scratch();
    let ct = ctx.encrypt(&ctx.re1, &ctx.im1, scratch.borrow());
    let (want_re, want_im) = ctx.want_square();

    let mut ct_res = ctx.alloc_ct_reduced_k();
    ct_res.square(&ctx.module, &ct, ctx.tsk(), scratch.borrow()).unwrap();

    ctx.assert_decrypt_precision(
        "square_ct smaller_output",
        &ct_res,
        &want_re,
        &want_im,
        18.0,
        scratch.borrow(),
    );
}
