//! Multiplication tests: ct × ct, ct × pt, ct × const, ct × int, and sequential depth chains.

use super::helpers::{TestContext, assert_precision, assert_valid_ciphertext};
use crate::{
    encoding::classical::encode,
    layouts::{ciphertext::CKKSCiphertext, plaintext::CKKSPlaintext},
    leveled::{
        encryption::{decrypt_tmp_bytes, encrypt_sk, encrypt_sk_tmp_bytes},
        operations::mul::{
            mul, mul_aligned, mul_const, mul_const_inplace, mul_const_tmp_bytes, mul_pt, mul_pt_inplace, mul_pt_tmp_bytes,
            mul_tmp_bytes, square, square_tmp_bytes,
        },
    },
};
use poulpy_core::{
    GLWEAlign, GLWEDecrypt, GLWEEncryptSk, GLWEMulConst, GLWEMulPlain, GLWEShift, GLWETensorKeyEncryptSk, GLWETensoring,
    ScratchTakeCore,
    layouts::{Base2K, Degree, GLWESecretPreparedFactory, GLWETensorKeyPreparedFactory, LWEInfos, TorusPrecision},
};
use poulpy_hal::{
    api::{ModuleN, ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxNormalize, VecZnxNormalizeTmpBytes},
    layouts::{Backend, Module, Scratch, ScratchOwned},
    source::Source,
};

/// Verifies ct × ct multiplication and rescale round-trip.
pub fn test_mul<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + ModuleNew<BE>
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWETensoring<BE>
        + GLWEAlign<BE>
        + GLWEShift<BE>
        + GLWETensorKeyEncryptSk<BE>
        + GLWETensorKeyPreparedFactory<BE>,
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
            .max(mul_tmp_bytes(&ctx.module, &ct_tmp, &ct_tmp, ctx.tsk())),
    );

    let ct1 = ctx.encrypt(&ctx.re1, &ctx.im1, &mut scratch);
    let ct2 = ctx.encrypt(&ctx.re2, &ctx.im2, &mut scratch);
    assert_valid_ciphertext("mul lhs", &ct1);
    assert_valid_ciphertext("mul rhs", &ct2);

    let mut ct_res = CKKSCiphertext::alloc(degree, base2k, k, ctx.params.log_delta);
    mul(&ctx.module, &mut ct_res, &ct1, &ct2, ctx.tsk(), scratch.borrow());
    assert_valid_ciphertext("mul result", &ct_res);

    let (re_out, im_out) = ctx.decrypt_decode(&ct_res, &mut scratch);
    let want_re: Vec<f64> = (0..m).map(|j| ctx.re1[j] * ctx.re2[j] - ctx.im1[j] * ctx.im2[j]).collect();
    let want_im: Vec<f64> = (0..m).map(|j| ctx.re1[j] * ctx.im2[j] + ctx.im1[j] * ctx.re2[j]).collect();
    assert_precision("mul re", &re_out, &want_re, 20.0);
    assert_precision("mul im", &im_out, &want_im, 20.0);
}

/// Verifies the aligned fast-path variant: `mul_aligned`.
pub fn test_mul_aligned<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + ModuleNew<BE>
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWETensoring<BE>
        + GLWEAlign<BE>
        + GLWEShift<BE>
        + GLWETensorKeyEncryptSk<BE>
        + GLWETensorKeyPreparedFactory<BE>,
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
            .max(mul_tmp_bytes(&ctx.module, &ct_tmp, &ct_tmp, ctx.tsk())),
    );

    let ct1 = ctx.encrypt(&ctx.re1, &ctx.im1, &mut scratch);
    let ct2 = ctx.encrypt(&ctx.re2, &ctx.im2, &mut scratch);

    let mut ct_res = CKKSCiphertext::alloc(degree, base2k, k, ctx.params.log_delta);
    mul_aligned(&ctx.module, &mut ct_res, &ct1, &ct2, ctx.tsk(), scratch.borrow());
    assert_valid_ciphertext("mul_aligned result", &ct_res);

    let (re_out, im_out) = ctx.decrypt_decode(&ct_res, &mut scratch);
    let want_re: Vec<f64> = (0..m).map(|j| ctx.re1[j] * ctx.re2[j] - ctx.im1[j] * ctx.im2[j]).collect();
    let want_im: Vec<f64> = (0..m).map(|j| ctx.re1[j] * ctx.im2[j] + ctx.im1[j] * ctx.re2[j]).collect();
    assert_precision("mul_aligned re", &re_out, &want_re, 20.0);
    assert_precision("mul_aligned im", &im_out, &want_im, 20.0);
}

/// Verifies ct × ct squaring and rescale round-trip.
pub fn test_square<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + ModuleNew<BE>
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWETensoring<BE>
        + GLWEAlign<BE>
        + GLWEShift<BE>
        + GLWETensorKeyEncryptSk<BE>
        + GLWETensorKeyPreparedFactory<BE>,
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
            .max(square_tmp_bytes(&ctx.module, &ct_tmp, ctx.tsk())),
    );

    let ct = ctx.encrypt(&ctx.re1, &ctx.im1, &mut scratch);
    assert_valid_ciphertext("square input", &ct);

    let mut ct_res = CKKSCiphertext::alloc(degree, base2k, k, ctx.params.log_delta);
    square(&ctx.module, &mut ct_res, &ct, ctx.tsk(), scratch.borrow());
    assert_valid_ciphertext("square result", &ct_res);

    let (re_out, im_out) = ctx.decrypt_decode(&ct_res, &mut scratch);
    let want_re: Vec<f64> = (0..m).map(|j| ctx.re1[j] * ctx.re1[j] - ctx.im1[j] * ctx.im1[j]).collect();
    let want_im: Vec<f64> = (0..m).map(|j| 2.0 * ctx.re1[j] * ctx.im1[j]).collect();
    assert_precision("square re", &re_out, &want_re, 20.0);
    assert_precision("square im", &im_out, &want_im, 20.0);
}

/// Verifies ct × ct multiplication when `k_a != k_b`.
///
/// The result active precision should be `min(k_a, k_b) - rescale_bits`.
pub fn test_mul_mismatched_k<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + ModuleNew<BE>
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWETensoring<BE>
        + GLWEAlign<BE>
        + GLWEShift<BE>
        + GLWETensorKeyEncryptSk<BE>
        + GLWETensorKeyPreparedFactory<BE>,
    Module<BE>: VecZnxNormalize<BE> + VecZnxNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let m = ctx.module.n() / 2;
    let base2k = Base2K(ctx.params.base2k);
    let degree = Degree(ctx.params.n);
    let k_hi = TorusPrecision(ctx.params.k);
    let k_lo = TorusPrecision(ctx.params.k - ctx.params.base2k);

    let ct_hi = CKKSCiphertext::alloc(degree, base2k, k_hi, ctx.params.log_delta);
    let ct_lo = CKKSCiphertext::alloc(degree, base2k, k_lo, ctx.params.log_delta);
    let mut scratch = ScratchOwned::<BE>::alloc(
        encrypt_sk_tmp_bytes(&ctx.module, &ct_hi)
            .max(encrypt_sk_tmp_bytes(&ctx.module, &ct_lo))
            .max(decrypt_tmp_bytes(&ctx.module, &ct_hi))
            .max(decrypt_tmp_bytes(&ctx.module, &ct_lo))
            .max(mul_tmp_bytes(&ctx.module, &ct_hi, &ct_lo, ctx.tsk())),
    );

    let a = ctx.encrypt_with_k(&ctx.re1, &ctx.im1, k_hi, &mut scratch);
    let b = ctx.encrypt_with_k(&ctx.re2, &ctx.im2, k_lo, &mut scratch);
    assert_valid_ciphertext("mul_mismatched_k lhs", &a);
    assert_valid_ciphertext("mul_mismatched_k rhs", &b);

    let mut ct_res = CKKSCiphertext::alloc(degree, base2k, k_lo, ctx.params.log_delta);
    mul(&ctx.module, &mut ct_res, &a, &b, ctx.tsk(), scratch.borrow());
    assert_valid_ciphertext("mul_mismatched_k result", &ct_res);

    let expected_k = TorusPrecision(k_lo.0 - ctx.params.log_delta);
    assert_eq!(
        ct_res.inner.k(),
        expected_k,
        "mul should keep the smaller active precision after rescale"
    );

    let (re_out, im_out) = ctx.decrypt_decode(&ct_res, &mut scratch);
    let want_re: Vec<f64> = (0..m).map(|j| ctx.re1[j] * ctx.re2[j] - ctx.im1[j] * ctx.im2[j]).collect();
    let want_im: Vec<f64> = (0..m).map(|j| ctx.re1[j] * ctx.im2[j] + ctx.im1[j] * ctx.re2[j]).collect();
    assert_precision("mul mismatched_k re", &re_out, &want_re, 20.0);
    assert_precision("mul mismatched_k im", &im_out, &want_im, 20.0);
}

/// Verifies ct × ct multiplication when the operands use different
/// `torus_scale_bits`.
///
/// The result scale should be `max(torus_scale_bits_a, torus_scale_bits_b)`,
/// and rescale should consume `min(torus_scale_bits_a, torus_scale_bits_b)`
/// bits from `k`.
pub fn test_mul_mismatched_delta<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + ModuleNew<BE>
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWETensoring<BE>
        + GLWEAlign<BE>
        + GLWEShift<BE>
        + GLWETensorKeyEncryptSk<BE>
        + GLWETensorKeyPreparedFactory<BE>,
    Module<BE>: VecZnxNormalize<BE> + VecZnxNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let m = ctx.module.n() / 2;
    let base2k = Base2K(ctx.params.base2k);
    let degree = Degree(ctx.params.n);
    let k = TorusPrecision(ctx.params.k);
    let log_delta_hi = ctx.params.log_delta;
    let log_delta_lo = ctx.params.log_delta - 6;

    let ct_hi = CKKSCiphertext::alloc(degree, base2k, k, log_delta_hi);
    let ct_lo = CKKSCiphertext::alloc(degree, base2k, k, log_delta_lo);
    let mut scratch = ScratchOwned::<BE>::alloc(
        encrypt_sk_tmp_bytes(&ctx.module, &ct_hi)
            .max(encrypt_sk_tmp_bytes(&ctx.module, &ct_lo))
            .max(decrypt_tmp_bytes(&ctx.module, &ct_hi))
            .max(decrypt_tmp_bytes(&ctx.module, &ct_lo))
            .max(mul_tmp_bytes(&ctx.module, &ct_hi, &ct_lo, ctx.tsk())),
    );

    let mut pt_hi = CKKSPlaintext::alloc(degree, base2k, log_delta_hi);
    encode(&mut pt_hi, &ctx.re1, &ctx.im1);
    let mut pt_lo = CKKSPlaintext::alloc(degree, base2k, log_delta_lo);
    encode(&mut pt_lo, &ctx.re2, &ctx.im2);

    let mut xa = Source::new([7u8; 32]);
    let mut xe = Source::new([8u8; 32]);
    let mut a = CKKSCiphertext::alloc(degree, base2k, k, log_delta_hi);
    encrypt_sk(&ctx.module, &mut a, &pt_hi, &ctx.sk, &mut xa, &mut xe, scratch.borrow());

    let mut xa = Source::new([9u8; 32]);
    let mut xe = Source::new([10u8; 32]);
    let mut b = CKKSCiphertext::alloc(degree, base2k, k, log_delta_lo);
    encrypt_sk(&ctx.module, &mut b, &pt_lo, &ctx.sk, &mut xa, &mut xe, scratch.borrow());
    assert_valid_ciphertext("mul_mismatched_delta lhs", &a);
    assert_valid_ciphertext("mul_mismatched_delta rhs", &b);

    let mut ct_res = CKKSCiphertext::alloc(degree, base2k, k, log_delta_hi);
    mul(&ctx.module, &mut ct_res, &a, &b, ctx.tsk(), scratch.borrow());
    assert_valid_ciphertext("mul_mismatched_delta result", &ct_res);

    assert_eq!(
        ct_res.torus_scale_bits, log_delta_hi,
        "mul should keep the larger scale after rescale"
    );
    assert_eq!(
        ct_res.inner.k(),
        TorusPrecision(k.0 - log_delta_lo),
        "mul should consume the smaller scale from active k"
    );

    let (re_out, im_out) = ctx.decrypt_decode(&ct_res, &mut scratch);
    let want_re: Vec<f64> = (0..m).map(|j| ctx.re1[j] * ctx.re2[j] - ctx.im1[j] * ctx.im2[j]).collect();
    let want_im: Vec<f64> = (0..m).map(|j| ctx.re1[j] * ctx.im2[j] + ctx.im1[j] * ctx.re2[j]).collect();
    assert_precision("mul mismatched_delta re", &re_out, &want_re, 20.0);
    assert_precision("mul mismatched_delta im", &im_out, &want_im, 20.0);
}

/// Verifies ct × compact pt multiplication (out-of-place and in-place).
pub fn test_mul_pt<BE: Backend>(ctx: &TestContext<BE>)
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

    let want_re: Vec<f64> = (0..m).map(|j| ctx.re1[j] * ctx.re2[j] - ctx.im1[j] * ctx.im2[j]).collect();
    let want_im: Vec<f64> = (0..m).map(|j| ctx.re1[j] * ctx.im2[j] + ctx.im1[j] * ctx.re2[j]).collect();

    let ct1 = ctx.encrypt(&ctx.re1, &ctx.im1, &mut scratch);
    assert_valid_ciphertext("mul_pt input", &ct1);
    let mut ct_res = CKKSCiphertext::alloc(degree, base2k, k, ctx.params.log_delta);
    mul_pt(&ctx.module, &mut ct_res, &ct1, &pt2, scratch.borrow());
    assert_valid_ciphertext("mul_pt result", &ct_res);
    let (re_out, im_out) = ctx.decrypt_decode(&ct_res, &mut scratch);
    assert_precision("mul_pt re", &re_out, &want_re, 20.0);
    assert_precision("mul_pt im", &im_out, &want_im, 20.0);

    let mut ct_ip = ctx.encrypt(&ctx.re1, &ctx.im1, &mut scratch);
    mul_pt_inplace(&ctx.module, &mut ct_ip, &pt2, scratch.borrow());
    assert_valid_ciphertext("mul_pt_inplace result", &ct_ip);
    let (re_ip, im_ip) = ctx.decrypt_decode(&ct_ip, &mut scratch);
    assert_precision("mul_pt_inplace re", &re_ip, &want_re, 20.0);
    assert_precision("mul_pt_inplace im", &im_ip, &want_im, 20.0);
}

/// Verifies ct × complex constant multiplication (out-of-place and in-place).
pub fn test_mul_const<BE: Backend>(ctx: &TestContext<BE>)
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
            .max(mul_const_tmp_bytes(&ctx.module, &ct_tmp)),
    );

    let (c_re, c_im) = (0.7, -0.5);
    let want_re: Vec<f64> = (0..m).map(|j| ctx.re1[j] * c_re - ctx.im1[j] * c_im).collect();
    let want_im: Vec<f64> = (0..m).map(|j| ctx.re1[j] * c_im + ctx.im1[j] * c_re).collect();

    let ct = ctx.encrypt(&ctx.re1, &ctx.im1, &mut scratch);
    assert_valid_ciphertext("mul_const input", &ct);
    let mut ct_res = CKKSCiphertext::alloc(degree, base2k, k, ctx.params.log_delta);
    mul_const(&ctx.module, &mut ct_res, &ct, c_re, c_im, scratch.borrow());
    assert_valid_ciphertext("mul_const result", &ct_res);
    let (re_out, im_out) = ctx.decrypt_decode(&ct_res, &mut scratch);
    assert_precision("mul_const re", &re_out, &want_re, 20.0);
    assert_precision("mul_const im", &im_out, &want_im, 20.0);

    let mut ct_ip = ctx.encrypt(&ctx.re1, &ctx.im1, &mut scratch);
    mul_const_inplace(&ctx.module, &mut ct_ip, c_re, c_im, scratch.borrow());
    assert_valid_ciphertext("mul_const_inplace result", &ct_ip);
    let (re_ip, im_ip) = ctx.decrypt_decode(&ct_ip, &mut scratch);
    assert_precision("mul_const_inplace re", &re_ip, &want_re, 20.0);
    assert_precision("mul_const_inplace im", &im_ip, &want_im, 20.0);
}

/// Verifies ct × small integer multiplication (no rescale).
pub fn test_mul_int<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN + GLWEEncryptSk<BE> + GLWEDecrypt<BE> + GLWESecretPreparedFactory<BE> + GLWEMulConst<BE>,
    Module<BE>: VecZnxNormalize<BE> + VecZnxNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let m = ctx.module.n() / 2;
    let base2k = Base2K(ctx.params.base2k);
    let k = TorusPrecision(ctx.params.k);
    let degree = Degree(ctx.params.n);

    let ct_tmp = CKKSCiphertext::alloc(degree, base2k, k, ctx.params.log_delta);
    let mul_scratch = crate::leveled::operations::mul::mul_int_tmp_bytes(&ctx.module, &ct_tmp, 1);
    let mut scratch = ScratchOwned::<BE>::alloc(
        encrypt_sk_tmp_bytes(&ctx.module, &ct_tmp)
            .max(decrypt_tmp_bytes(&ctx.module, &ct_tmp))
            .max(mul_scratch),
    );

    for c in [-3i64, -1, 1, 2, 5] {
        let ct = ctx.encrypt(&ctx.re1, &ctx.im1, &mut scratch);
        assert_valid_ciphertext(&format!("mul_int input({c})"), &ct);
        let mut ct_res = CKKSCiphertext::alloc(degree, base2k, k, ctx.params.log_delta);
        crate::leveled::operations::mul::mul_int(&ctx.module, &mut ct_res, &ct, c, scratch.borrow());
        assert_valid_ciphertext(&format!("mul_int result({c})"), &ct_res);

        let (re_out, im_out) = ctx.decrypt_decode(&ct_res, &mut scratch);
        let cf = c as f64;
        let want_re: Vec<f64> = (0..m).map(|j| ctx.re1[j] * cf).collect();
        let want_im: Vec<f64> = (0..m).map(|j| ctx.im1[j] * cf).collect();
        assert_precision(&format!("mul_int({c}) re"), &re_out, &want_re, 20.0);
        assert_precision(&format!("mul_int({c}) im"), &im_out, &want_im, 20.0);
    }
}

/// Squaring chain + mul_const_inplace chain at the given depth.
///
/// The squaring chain exercises repeated ct × ct multiplication with rescale.
/// The mul_const_inplace chain exercises repeated ct × constant with rescale.
/// Both verify precision at each level.
pub fn test_sequential_mul<BE: Backend>(ctx: &TestContext<BE>, depth: usize)
where
    Module<BE>: ModuleN
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWETensoring<BE>
        + GLWEAlign<BE>
        + GLWEShift<BE>
        + GLWETensorKeyEncryptSk<BE>
        + GLWETensorKeyPreparedFactory<BE>
        + GLWEMulPlain<BE>
        + GLWEMulConst<BE>
        + poulpy_core::GLWEAdd,
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
            .max(mul_tmp_bytes(&ctx.module, &ct_tmp, &ct_tmp, ctx.tsk()))
            .max(mul_pt_tmp_bytes(&ctx.module, &ct_tmp)),
    );

    // values in (-1, 1) with nonzero imaginary part
    let re_in: Vec<f64> = (0..m).map(|i| 0.9 * (1.0 - 2.0 * (i as f64) / (m as f64))).collect();
    let im_in: Vec<f64> = (0..m).map(|i| 0.3 * (2.0 * (i as f64) / (m as f64) - 1.0)).collect();

    let mut pt_in = CKKSPlaintext::alloc(degree, base2k, ctx.params.log_delta);
    encode(&mut pt_in, &re_in, &im_in);

    // squaring chain
    let mut ct = ctx.encrypt(&re_in, &im_in, &mut scratch);
    assert_valid_ciphertext("sequential_mul square start", &ct);
    let mut want_re = re_in.clone();
    let mut want_im = im_in.clone();

    for level in 0..depth {
        assert!(ct.inner.k().0 > ctx.params.log_delta, "level {level}: k exhausted");
        let mut ct_next = CKKSCiphertext::alloc(degree, base2k, k, ctx.params.log_delta);
        mul(&ctx.module, &mut ct_next, &ct, &ct, ctx.tsk(), scratch.borrow());
        ct = ct_next;
        assert_valid_ciphertext(&format!("sequential_mul square level {level}"), &ct);

        for j in 0..m {
            let (r, i) = (want_re[j], want_im[j]);
            want_re[j] = r * r - i * i;
            want_im[j] = 2.0 * r * i;
        }

        let (re_out, im_out) = ctx.decrypt_decode(&ct, &mut scratch);
        assert_precision(&format!("seq_mul level {level} re"), &re_out, &want_re, 20.0);
        assert_precision(&format!("seq_mul level {level} im"), &im_out, &want_im, 20.0);
    }

    // mul_const_inplace chain
    let mut ct2 = ctx.encrypt(&re_in, &im_in, &mut scratch);
    assert_valid_ciphertext("sequential_mul const start", &ct2);
    let scale = 0.8f64;
    let mut want_re2 = re_in.clone();
    let mut want_im2 = im_in.clone();
    for level in 0..depth {
        if ct2.inner.k().0 <= ct2.torus_scale_bits {
            break;
        }
        mul_const_inplace(&ctx.module, &mut ct2, scale, 0.0, scratch.borrow());
        assert_valid_ciphertext(&format!("sequential_mul const level {level}"), &ct2);
        for v in want_re2.iter_mut() {
            *v *= scale;
        }
        for v in want_im2.iter_mut() {
            *v *= scale;
        }
        let (re_out, im_out) = ctx.decrypt_decode(&ct2, &mut scratch);
        assert_precision(&format!("seq_const level {level} re"), &re_out, &want_re2, 20.0);
        assert_precision(&format!("seq_const level {level} im"), &im_out, &want_im2, 20.0);
    }
}
