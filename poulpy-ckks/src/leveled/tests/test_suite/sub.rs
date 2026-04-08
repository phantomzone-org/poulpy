//! Subtraction tests: ct-ct, ct-pt, ct-const (out-of-place and in-place).
use super::helpers::{TestContext, assert_precision};
use crate::layouts::{
    ciphertext::CKKSCiphertext,
    plaintext::{CKKSPlaintextConversion, CKKSPlaintextRnx, CKKSPlaintextZnx},
};
use poulpy_core::{
    GLWEAdd, GLWECopy, GLWEDecrypt, GLWEEncryptSk, GLWESub, ScratchTakeCore, layouts::GLWESecretPreparedFactory
};
use poulpy_cpu_ref::{FFT64Ref, fft64::FFT64ModuleHandle};
use poulpy_hal::{
    api::{ModuleN, ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxLsh, VecZnxNormalize, VecZnxNormalizeTmpBytes, VecZnxRshAdd},
    layouts::{Backend, Module, Scratch, ScratchOwned},
};

/// Verifies ct - ct subtraction (out-of-place and in-place).
pub fn test_sub<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEAdd
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes
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
    let ct_tmp = CKKSCiphertext::alloc(degree, base2k, k, ctx.params.log_delta);
    let mut scratch = ScratchOwned::<BE>::alloc(
        encrypt_sk_tmp_bytes(&ctx.module, &ct_tmp)
            .max(decrypt_tmp_bytes(&ctx.module, &ct_tmp))
            .max(sub_tmp_bytes(&ctx.module, &ct_tmp, &ct_tmp)),
    );

    let ct1 = ctx.encrypt(&ctx.re1, &ctx.im1, &mut scratch);
    let ct2 = ctx.encrypt(&ctx.re2, &ctx.im2, &mut scratch);

    let want_re: Vec<f64> = (0..m).map(|j| ctx.re1[j] - ctx.re2[j]).collect();
    let want_im: Vec<f64> = (0..m).map(|j| ctx.im1[j] - ctx.im2[j]).collect();

    let mut ct_res = CKKSCiphertext::alloc(degree, base2k, k, ctx.params.log_delta);
    sub(&ctx.module, &mut ct_res, &ct1, &ct2, scratch.borrow());
    let (re_out, im_out) = ctx.decrypt_decode(&ct_res, &mut scratch);
    assert_precision("sub re", &re_out, &want_re, 20.0);
    assert_precision("sub im", &im_out, &want_im, 20.0);

    let mut ct_ip = ctx.encrypt(&ctx.re1, &ctx.im1, &mut scratch);
    sub_inplace(&ctx.module, &mut ct_ip, &ct2, scratch.borrow());
    let (re_ip, im_ip) = ctx.decrypt_decode(&ct_ip, &mut scratch);
    assert_precision("sub_inplace re", &re_ip, &want_re, 20.0);
    assert_precision("sub_inplace im", &im_ip, &want_im, 20.0);

    let shifted_k = TorusPrecision(k.0 - 1);
    let mut ct2_shifted = CKKSCiphertext::alloc(degree, base2k, shifted_k, ctx.params.log_delta);
    ctx.module.glwe_align(
        &mut ct2_shifted.inner,
        shifted_k.0,
        &ct2.inner,
        ct2.offset_bits(),
        scratch.borrow(),
    );
    ct2_shifted.offset_bits = shifted_k.0;

    let mut ct_misaligned = CKKSCiphertext::alloc(degree, base2k, k, ctx.params.log_delta);
    sub(&ctx.module, &mut ct_misaligned, &ct1, &ct2_shifted, scratch.borrow());
    let (re_misaligned, im_misaligned) = ctx.decrypt_decode(&ct_misaligned, &mut scratch);
    assert_precision("sub misaligned re", &re_misaligned, &want_re, 19.0);
    assert_precision("sub misaligned im", &im_misaligned, &want_im, 19.0);
}

/// Verifies ct - compact pt subtraction (out-of-place and in-place).
pub fn test_sub_pt<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEAdd
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

    let ct_tmp = CKKSCiphertext::alloc(degree, base2k, k, ctx.params.log_delta);
    let mut scratch = ScratchOwned::<BE>::alloc(
        encrypt_sk_tmp_bytes(&ctx.module, &ct_tmp)
            .max(decrypt_tmp_bytes(&ctx.module, &ct_tmp))
            .max(sub_pt_tmp_bytes(&ctx.module, &ct_tmp)),
    );

    let ct1 = ctx.encrypt(&ctx.re1, &ctx.im1, &mut scratch);
    let mut pt2 = CKKSPlaintext::alloc(degree, base2k, ctx.params.log_delta);
    encode(&mut pt2, &ctx.re2, &ctx.im2);

    let want_re: Vec<f64> = (0..m).map(|j| ctx.re1[j] - ctx.re2[j]).collect();
    let want_im: Vec<f64> = (0..m).map(|j| ctx.im1[j] - ctx.im2[j]).collect();

    let mut ct_res = CKKSCiphertext::alloc(degree, base2k, k, ctx.params.log_delta);
    sub_pt(&ctx.module, &mut ct_res, &ct1, &pt2, scratch.borrow());
    let (re_out, im_out) = ctx.decrypt_decode(&ct_res, &mut scratch);
    assert_precision("sub_pt re", &re_out, &want_re, 20.0);
    assert_precision("sub_pt im", &im_out, &want_im, 20.0);

    let mut ct_ip = ctx.encrypt(&ctx.re1, &ctx.im1, &mut scratch);
    sub_pt_inplace(&ctx.module, &mut ct_ip, &pt2, scratch.borrow());
    let (re_ip, im_ip) = ctx.decrypt_decode(&ct_ip, &mut scratch);
    assert_precision("sub_pt_inplace re", &re_ip, &want_re, 20.0);
    assert_precision("sub_pt_inplace im", &im_ip, &want_im, 20.0);
}

/// Verifies ct - complex constant subtraction (out-of-place and in-place).
pub fn test_sub_const<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEAdd
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

    let ct_tmp = CKKSCiphertext::alloc(degree, base2k, k, ctx.params.log_delta);
    let mut scratch = ScratchOwned::<BE>::alloc(
        encrypt_sk_tmp_bytes(&ctx.module, &ct_tmp)
            .max(decrypt_tmp_bytes(&ctx.module, &ct_tmp))
            .max(sub_const_tmp_bytes(&ctx.module, &ct_tmp)),
    );
    let (c_re, c_im) = (0.7, -0.3);

    let ct1 = ctx.encrypt(&ctx.re1, &ctx.im1, &mut scratch);

    let want_re: Vec<f64> = (0..m).map(|j| ctx.re1[j] - c_re).collect();
    let want_im: Vec<f64> = (0..m).map(|j| ctx.im1[j] - c_im).collect();

    let mut ct_res = CKKSCiphertext::alloc(degree, base2k, k, ctx.params.log_delta);
    sub_const(&ctx.module, &mut ct_res, &ct1, c_re, c_im, scratch.borrow());
    let (re_out, im_out) = ctx.decrypt_decode(&ct_res, &mut scratch);
    assert_precision("sub_const re", &re_out, &want_re, 20.0);
    assert_precision("sub_const im", &im_out, &want_im, 20.0);

    let mut ct_ip = ctx.encrypt(&ctx.re1, &ctx.im1, &mut scratch);
    sub_const_inplace(&ctx.module, &mut ct_ip, c_re, c_im, scratch.borrow());
    let (re_ip, im_ip) = ctx.decrypt_decode(&ct_ip, &mut scratch);
    assert_precision("sub_const_inplace re", &re_ip, &want_re, 20.0);
    assert_precision("sub_const_inplace im", &im_ip, &want_im, 20.0);
}

/// Verifies the aligned fast-path variants: `sub_aligned` and `sub_aligned_inplace`.
pub fn test_sub_aligned<BE: Backend>(ctx: &TestContext<BE>)
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
    let ct2 = ctx.encrypt(&ctx.re2, &ctx.im2, &mut scratch);

    let want_re: Vec<f64> = (0..m).map(|j| ctx.re1[j] - ctx.re2[j]).collect();
    let want_im: Vec<f64> = (0..m).map(|j| ctx.im1[j] - ctx.im2[j]).collect();

    let mut ct_res = CKKSCiphertext::alloc(degree, base2k, k, ctx.params.log_delta);
    sub_aligned(&ctx.module, &mut ct_res, &ct1, &ct2);
    let (re_out, im_out) = ctx.decrypt_decode(&ct_res, &mut scratch);
    assert_precision("sub_aligned re", &re_out, &want_re, 20.0);
    assert_precision("sub_aligned im", &im_out, &want_im, 20.0);

    let mut ct_ip = ctx.encrypt(&ctx.re1, &ctx.im1, &mut scratch);
    sub_aligned_inplace(&ctx.module, &mut ct_ip, &ct2);
    let (re_ip, im_ip) = ctx.decrypt_decode(&ct_ip, &mut scratch);
    assert_precision("sub_aligned_inplace re", &re_ip, &want_re, 20.0);
    assert_precision("sub_aligned_inplace im", &im_ip, &want_im, 20.0);
}
