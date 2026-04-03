//! Prepared plaintext tests: add, sub, mul using pre-expanded plaintexts.

use super::helpers::{TestContext, assert_precision};
use crate::{
    encoding::classical::encode,
    layouts::{ciphertext::CKKSCiphertext, plaintext::CKKSPlaintext, plaintext_prepared::CKKSPlaintextPrepared},
    leveled::{
        encryption::{decrypt_tmp_bytes, encrypt_sk_tmp_bytes},
        operations::{
            add::{add_prepared_pt, add_prepared_pt_inplace},
            mul::{mul_prepared_pt, mul_prepared_pt_inplace, mul_pt_tmp_bytes},
            sub::{sub_prepared_pt, sub_prepared_pt_inplace},
        },
    },
};
use poulpy_core::{
    GLWEAdd, GLWEDecrypt, GLWEEncryptSk, GLWEMulPlain, GLWEShift, GLWESub, ScratchTakeCore,
    layouts::{Base2K, Degree, GLWESecretPreparedFactory, TorusPrecision},
};
use poulpy_hal::{
    api::{ModuleN, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxNormalize, VecZnxNormalizeTmpBytes},
    layouts::{Backend, Module, Scratch, ScratchOwned},
};

/// Verifies ct + prepared pt addition (out-of-place and in-place).
pub fn test_add_prepared_pt<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEAdd
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let m = ctx.module.n() / 2;
    let base2k = Base2K(ctx.params.base2k);
    let k = TorusPrecision(ctx.params.k);
    let degree = Degree(ctx.params.n);
    let mut scratch = ctx.alloc_scratch();

    let ct1 = ctx.encrypt(&ctx.re1, &ctx.im1, &mut scratch);
    let mut pt2 = CKKSPlaintext::alloc(degree, base2k, ctx.params.log_delta);
    encode(&mut pt2, &ctx.re2, &ctx.im2);
    let pt2_prepared = CKKSPlaintextPrepared::alloc_and_prepare(&ctx.module, degree, base2k, k, &pt2, scratch.borrow());

    let want_re: Vec<f64> = (0..m).map(|j| ctx.re1[j] + ctx.re2[j]).collect();
    let want_im: Vec<f64> = (0..m).map(|j| ctx.im1[j] + ctx.im2[j]).collect();

    let mut ct_res = CKKSCiphertext::alloc(degree, base2k, k, ctx.params.log_delta);
    add_prepared_pt(&ctx.module, &mut ct_res, &ct1, &pt2_prepared);
    let (re_out, im_out) = ctx.decrypt_decode(&ct_res, &mut scratch);
    assert_precision("add_prepared_pt re", &re_out, &want_re, 20.0);
    assert_precision("add_prepared_pt im", &im_out, &want_im, 20.0);

    let mut ct_ip = ctx.encrypt(&ctx.re1, &ctx.im1, &mut scratch);
    add_prepared_pt_inplace(&ctx.module, &mut ct_ip, &pt2_prepared);
    let (re_ip, im_ip) = ctx.decrypt_decode(&ct_ip, &mut scratch);
    assert_precision("add_prepared_pt_inplace re", &re_ip, &want_re, 20.0);
    assert_precision("add_prepared_pt_inplace im", &im_ip, &want_im, 20.0);
}

/// Verifies ct - prepared pt subtraction (out-of-place and in-place).
pub fn test_sub_prepared_pt<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESub
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let m = ctx.module.n() / 2;
    let base2k = Base2K(ctx.params.base2k);
    let k = TorusPrecision(ctx.params.k);
    let degree = Degree(ctx.params.n);
    let mut scratch = ctx.alloc_scratch();

    let ct1 = ctx.encrypt(&ctx.re1, &ctx.im1, &mut scratch);
    let mut pt2 = CKKSPlaintext::alloc(degree, base2k, ctx.params.log_delta);
    encode(&mut pt2, &ctx.re2, &ctx.im2);
    let pt2_prepared = CKKSPlaintextPrepared::alloc_and_prepare(&ctx.module, degree, base2k, k, &pt2, scratch.borrow());

    let want_re: Vec<f64> = (0..m).map(|j| ctx.re1[j] - ctx.re2[j]).collect();
    let want_im: Vec<f64> = (0..m).map(|j| ctx.im1[j] - ctx.im2[j]).collect();

    let mut ct_res = CKKSCiphertext::alloc(degree, base2k, k, ctx.params.log_delta);
    sub_prepared_pt(&ctx.module, &mut ct_res, &ct1, &pt2_prepared);
    let (re_out, im_out) = ctx.decrypt_decode(&ct_res, &mut scratch);
    assert_precision("sub_prepared_pt re", &re_out, &want_re, 20.0);
    assert_precision("sub_prepared_pt im", &im_out, &want_im, 20.0);

    let mut ct_ip = ctx.encrypt(&ctx.re1, &ctx.im1, &mut scratch);
    sub_prepared_pt_inplace(&ctx.module, &mut ct_ip, &pt2_prepared);
    let (re_ip, im_ip) = ctx.decrypt_decode(&ct_ip, &mut scratch);
    assert_precision("sub_prepared_pt_inplace re", &re_ip, &want_re, 20.0);
    assert_precision("sub_prepared_pt_inplace im", &im_ip, &want_im, 20.0);
}

/// Verifies ct × prepared pt multiplication (out-of-place and in-place).
pub fn test_mul_prepared_pt<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN + GLWEEncryptSk<BE> + GLWEDecrypt<BE> + GLWESecretPreparedFactory<BE> + GLWEMulPlain<BE> + GLWEShift<BE>,
    Module<BE>: VecZnxNormalize<BE> + VecZnxNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let m = ctx.module.n() / 2;
    let base2k = Base2K(ctx.params.base2k);
    let k = TorusPrecision(ctx.params.k);
    let degree = Degree(ctx.params.n);

    let ct_tmp = CKKSCiphertext::alloc(degree, base2k, k, ctx.params.log_delta);
    let mut scratch = ScratchOwned::<BE>::alloc(
        encrypt_sk_tmp_bytes(&ctx.module, &ct_tmp)
            .max(decrypt_tmp_bytes(&ctx.module, &ct_tmp))
            .max(mul_pt_tmp_bytes(&ctx.module, &ct_tmp)),
    );

    let mut pt2 = CKKSPlaintext::alloc(degree, base2k, ctx.params.log_delta);
    encode(&mut pt2, &ctx.re2, &ctx.im2);

    // Prepare for mul: use max_k = base2k * size
    let ct_size = k.0.div_ceil(base2k.0) as usize;
    let max_k = TorusPrecision(base2k.0 * ct_size as u32);
    let pt2_prepared = CKKSPlaintextPrepared::alloc_and_prepare(&ctx.module, degree, base2k, max_k, &pt2, scratch.borrow());

    let want_re: Vec<f64> = (0..m).map(|j| ctx.re1[j] * ctx.re2[j] - ctx.im1[j] * ctx.im2[j]).collect();
    let want_im: Vec<f64> = (0..m).map(|j| ctx.re1[j] * ctx.im2[j] + ctx.im1[j] * ctx.re2[j]).collect();

    let ct1 = ctx.encrypt(&ctx.re1, &ctx.im1, &mut scratch);
    let mut ct_res = CKKSCiphertext::alloc(degree, base2k, k, ctx.params.log_delta);
    mul_prepared_pt(&ctx.module, &mut ct_res, &ct1, &pt2_prepared, scratch.borrow());
    let (re_out, im_out) = ctx.decrypt_decode(&ct_res, &mut scratch);
    assert_precision("mul_prepared_pt re", &re_out, &want_re, 20.0);
    assert_precision("mul_prepared_pt im", &im_out, &want_im, 20.0);

    let mut ct_ip = ctx.encrypt(&ctx.re1, &ctx.im1, &mut scratch);
    mul_prepared_pt_inplace(&ctx.module, &mut ct_ip, &pt2_prepared, scratch.borrow());
    let (re_ip, im_ip) = ctx.decrypt_decode(&ct_ip, &mut scratch);
    assert_precision("mul_prepared_pt_inplace re", &re_ip, &want_re, 20.0);
    assert_precision("mul_prepared_pt_inplace im", &im_ip, &want_im, 20.0);
}
