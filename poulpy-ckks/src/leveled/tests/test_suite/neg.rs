//! Negation tests (out-of-place and in-place).

use super::helpers::{TestContext, assert_precision};
use crate::{
    layouts::ciphertext::CKKSCiphertext,
    leveled::operations::neg::{neg, neg_inplace},
};
use poulpy_core::{
    GLWEAdd, GLWEDecrypt, GLWEEncryptSk, ScratchTakeCore,
    layouts::{Base2K, Degree, GLWESecretPreparedFactory, TorusPrecision},
};
use poulpy_hal::{
    api::{
        ModuleN, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxNegate, VecZnxNegateInplace, VecZnxNormalize,
        VecZnxNormalizeTmpBytes,
    },
    layouts::{Backend, Module, Scratch, ScratchOwned},
};

/// Verifies negation (out-of-place and in-place).
pub fn test_neg<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEAdd
        + VecZnxNegate
        + VecZnxNegateInplace
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let base2k = Base2K(ctx.params.base2k);
    let k = TorusPrecision(ctx.params.k);
    let degree = Degree(ctx.params.n);
    let mut scratch = ctx.alloc_scratch();

    let ct1 = ctx.encrypt(&ctx.re1, &ctx.im1, &mut scratch);

    let want_re: Vec<f64> = ctx.re1.iter().map(|v| -v).collect();
    let want_im: Vec<f64> = ctx.im1.iter().map(|v| -v).collect();

    let mut ct_res = CKKSCiphertext::alloc(degree, base2k, k, ctx.params.log_delta);
    neg(&ctx.module, &mut ct_res, &ct1);
    let (re_out, im_out) = ctx.decrypt_decode(&ct_res, &mut scratch);
    assert_precision("neg re", &re_out, &want_re, 20.0);
    assert_precision("neg im", &im_out, &want_im, 20.0);

    let mut ct_ip = ctx.encrypt(&ctx.re1, &ctx.im1, &mut scratch);
    neg_inplace(&ctx.module, &mut ct_ip);
    let (re_ip, im_ip) = ctx.decrypt_decode(&ct_ip, &mut scratch);
    assert_precision("neg_inplace re", &re_ip, &want_re, 20.0);
    assert_precision("neg_inplace im", &im_ip, &want_im, 20.0);
}
