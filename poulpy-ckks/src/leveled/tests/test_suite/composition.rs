//! Composition tests: multi-step CKKS evaluation paths that combine primitives.

use super::helpers::{TestContext, assert_precision};
use crate::layouts::{ciphertext::CKKS, plaintext_prepared::CKKSPlaintextPrepared};
use crate::leveled::{
    encryption::{decrypt_tmp_bytes, encrypt_sk_tmp_bytes},
    operations::{
        add::{add_aligned, add_inplace, add_prepared_pt_inplace},
        align::{align_to, align_to_tmp_bytes},
        mul::{mul, mul_prepared_pt, mul_pt_tmp_bytes, mul_tmp_bytes, square, square_tmp_bytes},
    },
};
use poulpy_core::{
    GLWEAdd, GLWENormalize, GLWEDecrypt, GLWEEncryptSk, GLWEMulPlain, GLWEShift, GLWETensoring, ScratchTakeCore,
    layouts::{Base2K, Degree, GLWE, GLWESecretPreparedFactory, TorusPrecision},
};
use poulpy_hal::{
    api::{ModuleN, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxNormalize, VecZnxNormalizeTmpBytes},
    layouts::{Backend, Module, Scratch, ScratchOwned},
};

fn poly2_expected(ctx: &TestContext<impl Backend>, c0: (f64, f64), c1: (f64, f64), c2: (f64, f64)) -> (Vec<f64>, Vec<f64>) {
    let m = ctx.module.n() / 2;
    let want_re: Vec<f64> = (0..m)
        .map(|j| {
            let x_re = ctx.re1[j];
            let x_im = ctx.im1[j];
            let x2_re = x_re * x_re - x_im * x_im;
            let x2_im = 2.0 * x_re * x_im;
            c0.0 + c1.0 * x_re - c1.1 * x_im + c2.0 * x2_re - c2.1 * x2_im
        })
        .collect();
    let want_im: Vec<f64> = (0..m)
        .map(|j| {
            let x_re = ctx.re1[j];
            let x_im = ctx.im1[j];
            let x2_re = x_re * x_re - x_im * x_im;
            let x2_im = 2.0 * x_re * x_im;
            c0.1 + c1.0 * x_im + c1.1 * x_re + c2.0 * x2_im + c2.1 * x2_re
        })
        .collect();
    (want_re, want_im)
}

fn same_offset_expected(ctx: &TestContext<impl Backend>, c1: (f64, f64), c2: (f64, f64)) -> (Vec<f64>, Vec<f64>) {
    let m = ctx.module.n() / 2;
    let coeff_re = c1.0 + c2.0;
    let coeff_im = c1.1 + c2.1;
    let want_re: Vec<f64> = (0..m).map(|j| coeff_re * ctx.re1[j] - coeff_im * ctx.im1[j]).collect();
    let want_im: Vec<f64> = (0..m).map(|j| coeff_re * ctx.im1[j] + coeff_im * ctx.re1[j]).collect();
    (want_re, want_im)
}

fn mul_by_y_expected(ctx: &TestContext<impl Backend>, c0: (f64, f64), c1: (f64, f64), c2: (f64, f64)) -> (Vec<f64>, Vec<f64>) {
    let m = ctx.module.n() / 2;
    let (poly_re, poly_im) = poly2_expected(ctx, c0, c1, c2);
    let want_re: Vec<f64> = (0..m).map(|j| poly_re[j] * ctx.re2[j] - poly_im[j] * ctx.im2[j]).collect();
    let want_im: Vec<f64> = (0..m).map(|j| poly_re[j] * ctx.im2[j] + poly_im[j] * ctx.re2[j]).collect();
    (want_re, want_im)
}

/// Adding two prepared-plaintext products computed from the same branch stays accurate.
pub fn test_prepared_linear_sum<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEMulPlain<BE>
        + GLWEShift<BE>
        + GLWEAdd,
    Module<BE>: VecZnxNormalize<BE> + VecZnxNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let base2k = Base2K(ctx.params.base2k);
    let k = TorusPrecision(ctx.params.k);
    let degree = Degree(ctx.params.n);
    let ct_tmp = CKKS::alloc(degree, base2k, k, ctx.params.log_delta);
    let mut scratch = ScratchOwned::<BE>::alloc(
        encrypt_sk_tmp_bytes(&ctx.module, &ct_tmp)
            .max(decrypt_tmp_bytes(&ctx.module, &ct_tmp))
            .max(mul_pt_tmp_bytes(&ctx.module, &ct_tmp)),
    );

    let c1 = (0.625, -0.125);
    let c2 = (-0.375, 0.25);
    let pt1 = CKKSPlaintextPrepared::from_const(&ctx.module, degree, base2k, k, ctx.params.log_delta, c1, scratch.borrow());
    let pt2 = CKKSPlaintextPrepared::from_const(&ctx.module, degree, base2k, k, ctx.params.log_delta, c2, scratch.borrow());
    let (want_re, want_im) = same_offset_expected(ctx, c1, c2);

    let x = ctx.encrypt(&ctx.re1, &ctx.im1, &mut scratch);
    let mut term1 = CKKS::alloc(degree, base2k, k, ctx.params.log_delta);
    let mut term2 = CKKS::alloc(degree, base2k, k, ctx.params.log_delta);
    mul_prepared_pt(&ctx.module, &mut term1, &x, &pt1, scratch.borrow());
    mul_prepared_pt(&ctx.module, &mut term2, &x, &pt2, scratch.borrow());

    assert_eq!(
        term1.offset_bits(),
        term2.offset_bits(),
        "linear branches must share the same offset"
    );
    add_inplace(&ctx.module, &mut term1, &term2, scratch.borrow());

    let (re_out, im_out) = ctx.decrypt_decode(&term1, &mut scratch);
    assert_precision("prepared_linear_sum re", &re_out, &want_re, 20.0);
    assert_precision("prepared_linear_sum im", &im_out, &want_im, 20.0);
}

/// A prepared `c1*x + c2*x^2` sum spans mixed offsets and stays accurate.
pub fn test_prepared_poly2_sum<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEMulPlain<BE>
        + GLWEShift<BE>
        + GLWEAdd
        + GLWENormalize<BE>
        + GLWETensoring<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let tsk = ctx.tsk();
    let base2k = Base2K(ctx.params.base2k);
    let k = TorusPrecision(ctx.params.k);
    let degree = Degree(ctx.params.n);
    let ct_tmp = CKKS::alloc(degree, base2k, k, ctx.params.log_delta);
    let mut scratch = ScratchOwned::<BE>::alloc(
        encrypt_sk_tmp_bytes(&ctx.module, &ct_tmp)
            .max(decrypt_tmp_bytes(&ctx.module, &ct_tmp))
            .max(mul_pt_tmp_bytes(&ctx.module, &ct_tmp))
            .max(square_tmp_bytes(&ctx.module, &ct_tmp, tsk))
            .max(align_to_tmp_bytes(&ctx.module, &ct_tmp, k)),
    );

    let c1 = (0.625, -0.125);
    let c2 = (-0.375, 0.25);
    let pt1 = CKKSPlaintextPrepared::from_const(&ctx.module, degree, base2k, k, ctx.params.log_delta, c1, scratch.borrow());
    let pt2 = CKKSPlaintextPrepared::from_const(&ctx.module, degree, base2k, k, ctx.params.log_delta, c2, scratch.borrow());
    let (want_re, want_im) = poly2_expected(ctx, (0.0, 0.0), c1, c2);

    let x = ctx.encrypt(&ctx.re1, &ctx.im1, &mut scratch);
    let mut x2 = CKKS::alloc(degree, base2k, k, ctx.params.log_delta);
    square(&ctx.module, &mut x2, &x, tsk, scratch.borrow());

    let mut term1 = CKKS::alloc(degree, base2k, k, ctx.params.log_delta);
    let mut term2 = CKKS::alloc(degree, base2k, k, ctx.params.log_delta);
    mul_prepared_pt(&ctx.module, &mut term1, &x, &pt1, scratch.borrow());
    mul_prepared_pt(&ctx.module, &mut term2, &x2, &pt2, scratch.borrow());

    assert_eq!(
        term1.torus_scale_bits(),
        term2.torus_scale_bits(),
        "poly2 branches must remain add-compatible"
    );
    assert_ne!(
        term1.offset_bits(),
        term2.offset_bits(),
        "poly2 branches should expose different natural offsets"
    );
    add_inplace(&ctx.module, &mut term1, &term2, scratch.borrow());

    let (re_out, im_out) = ctx.decrypt_decode(&term1, &mut scratch);
    assert_precision("prepared_poly2_sum re", &re_out, &want_re, 20.0);
    assert_precision("prepared_poly2_sum im", &im_out, &want_im, 20.0);
}

/// Explicit alignment remains a no-op-compatible path for the prepared poly2 sum.
pub fn test_prepared_poly2_sum_aligned<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEMulPlain<BE>
        + GLWEShift<BE>
        + GLWEAdd
        + GLWENormalize<BE>
        + GLWETensoring<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let tsk = ctx.tsk();
    let base2k = Base2K(ctx.params.base2k);
    let k = TorusPrecision(ctx.params.k);
    let degree = Degree(ctx.params.n);
    let ct_tmp = CKKS::alloc(degree, base2k, k, ctx.params.log_delta);
    let mut scratch = ScratchOwned::<BE>::alloc(
        encrypt_sk_tmp_bytes(&ctx.module, &ct_tmp)
            .max(decrypt_tmp_bytes(&ctx.module, &ct_tmp))
            .max(mul_pt_tmp_bytes(&ctx.module, &ct_tmp))
            .max(square_tmp_bytes(&ctx.module, &ct_tmp, tsk))
            .max(align_to_tmp_bytes(&ctx.module, &ct_tmp, k)),
    );

    let c1 = (0.625, -0.125);
    let c2 = (-0.375, 0.25);
    let pt1 = CKKSPlaintextPrepared::from_const(&ctx.module, degree, base2k, k, ctx.params.log_delta, c1, scratch.borrow());
    let pt2 = CKKSPlaintextPrepared::from_const(&ctx.module, degree, base2k, k, ctx.params.log_delta, c2, scratch.borrow());
    let (want_re, want_im) = poly2_expected(ctx, (0.0, 0.0), c1, c2);

    let x = ctx.encrypt(&ctx.re1, &ctx.im1, &mut scratch);
    let mut x2 = CKKS::alloc(degree, base2k, k, ctx.params.log_delta);
    square(&ctx.module, &mut x2, &x, tsk, scratch.borrow());

    let mut term1 = CKKS::alloc(degree, base2k, k, ctx.params.log_delta);
    let mut term2 = CKKS::alloc(degree, base2k, k, ctx.params.log_delta);
    mul_prepared_pt(&ctx.module, &mut term1, &x, &pt1, scratch.borrow());
    mul_prepared_pt(&ctx.module, &mut term2, &x2, &pt2, scratch.borrow());

    let target_offset = term1.offset_bits().max(term2.offset_bits());
    let target_k = TorusPrecision(target_offset);
    let mut aligned1 = CKKS::alloc(degree, base2k, k, ctx.params.log_delta);
    let mut aligned2 = CKKS::alloc(degree, base2k, k, ctx.params.log_delta);
    let mut sum = CKKS::alloc(degree, base2k, k, ctx.params.log_delta);
    align_to(&ctx.module, &mut aligned1, &term1, target_offset, target_k, scratch.borrow());
    align_to(&ctx.module, &mut aligned2, &term2, target_offset, target_k, scratch.borrow());
    add_aligned(&ctx.module, &mut sum, &aligned1, &aligned2);

    let (re_out, im_out) = ctx.decrypt_decode(&sum, &mut scratch);
    assert_precision("prepared_poly2_sum_aligned re", &re_out, &want_re, 20.0);
    assert_precision("prepared_poly2_sum_aligned im", &im_out, &want_im, 20.0);
}

/// A prepared `c*x^2` branch keeps its natural lower offset and decrypts correctly after alignment.
pub fn test_prepared_poly2_term_align<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEMulPlain<BE>
        + GLWEShift<BE>
        + GLWENormalize<BE>
        + GLWETensoring<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let tsk = ctx.tsk();
    let m = ctx.module.n() / 2;
    let base2k = Base2K(ctx.params.base2k);
    let k = TorusPrecision(ctx.params.k);
    let degree = Degree(ctx.params.n);
    let ct_tmp = CKKS::alloc(degree, base2k, k, ctx.params.log_delta);
    let mut scratch = ScratchOwned::<BE>::alloc(
        encrypt_sk_tmp_bytes(&ctx.module, &ct_tmp)
            .max(decrypt_tmp_bytes(&ctx.module, &ct_tmp))
            .max(mul_pt_tmp_bytes(&ctx.module, &ct_tmp))
            .max(square_tmp_bytes(&ctx.module, &ct_tmp, tsk))
            .max(align_to_tmp_bytes(&ctx.module, &ct_tmp, k)),
    );

    let c2 = (-0.375, 0.25);
    let pt2 = CKKSPlaintextPrepared::from_const(&ctx.module, degree, base2k, k, ctx.params.log_delta, c2, scratch.borrow());

    let x = ctx.encrypt(&ctx.re1, &ctx.im1, &mut scratch);
    let mut x2 = CKKS::alloc(degree, base2k, k, ctx.params.log_delta);
    square(&ctx.module, &mut x2, &x, tsk, scratch.borrow());

    let mut term = CKKS::alloc(degree, base2k, k, ctx.params.log_delta);
    mul_prepared_pt(&ctx.module, &mut term, &x2, &pt2, scratch.borrow());

    let want_re: Vec<f64> = (0..m)
        .map(|j| {
            let x_re = ctx.re1[j];
            let x_im = ctx.im1[j];
            let x2_re = x_re * x_re - x_im * x_im;
            let x2_im = 2.0 * x_re * x_im;
            c2.0 * x2_re - c2.1 * x2_im
        })
        .collect();
    let want_im: Vec<f64> = (0..m)
        .map(|j| {
            let x_re = ctx.re1[j];
            let x_im = ctx.im1[j];
            let x2_re = x_re * x_re - x_im * x_im;
            let x2_im = 2.0 * x_re * x_im;
            c2.0 * x2_im + c2.1 * x2_re
        })
        .collect();

    let natural_offset = k.0 - 2 * ctx.params.log_delta;
    let align_offset = k.0 - ctx.params.log_delta;
    assert_eq!(
        term.offset_bits(),
        natural_offset,
        "x2 branch must consume visible offset precision"
    );
    let mut aligned = CKKS::alloc(degree, base2k, k, ctx.params.log_delta);
    align_to(
        &ctx.module,
        &mut aligned,
        &term,
        align_offset,
        TorusPrecision(align_offset),
        scratch.borrow(),
    );

    let (re_out, im_out) = ctx.decrypt_decode(&aligned, &mut scratch);
    assert_precision("prepared_poly2_term_align re", &re_out, &want_re, 20.0);
    assert_precision("prepared_poly2_term_align im", &im_out, &want_im, 20.0);
}

/// A prepared `c*x^2` branch decrypts correctly with its own lower-precision metadata.
pub fn test_prepared_poly2_term<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEMulPlain<BE>
        + GLWEShift<BE>
        + GLWENormalize<BE>
        + GLWETensoring<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let tsk = ctx.tsk();
    let m = ctx.module.n() / 2;
    let base2k = Base2K(ctx.params.base2k);
    let k = TorusPrecision(ctx.params.k);
    let degree = Degree(ctx.params.n);
    let ct_tmp = CKKS::alloc(degree, base2k, k, ctx.params.log_delta);
    let mut scratch = ScratchOwned::<BE>::alloc(
        encrypt_sk_tmp_bytes(&ctx.module, &ct_tmp)
            .max(decrypt_tmp_bytes(&ctx.module, &ct_tmp))
            .max(mul_pt_tmp_bytes(&ctx.module, &ct_tmp))
            .max(square_tmp_bytes(&ctx.module, &ct_tmp, tsk))
            .max(align_to_tmp_bytes(&ctx.module, &ct_tmp, k)),
    );

    let c2 = (-0.375, 0.25);
    let pt2 = CKKSPlaintextPrepared::from_const(&ctx.module, degree, base2k, k, ctx.params.log_delta, c2, scratch.borrow());

    let x = ctx.encrypt(&ctx.re1, &ctx.im1, &mut scratch);
    let mut x2 = CKKS::alloc(degree, base2k, k, ctx.params.log_delta);
    square(&ctx.module, &mut x2, &x, tsk, scratch.borrow());

    let mut term = CKKS::alloc(degree, base2k, k, ctx.params.log_delta);
    mul_prepared_pt(&ctx.module, &mut term, &x2, &pt2, scratch.borrow());

    let want_re: Vec<f64> = (0..m)
        .map(|j| {
            let x_re = ctx.re1[j];
            let x_im = ctx.im1[j];
            let x2_re = x_re * x_re - x_im * x_im;
            let x2_im = 2.0 * x_re * x_im;
            c2.0 * x2_re - c2.1 * x2_im
        })
        .collect();
    let want_im: Vec<f64> = (0..m)
        .map(|j| {
            let x_re = ctx.re1[j];
            let x_im = ctx.im1[j];
            let x2_re = x_re * x_re - x_im * x_im;
            let x2_im = 2.0 * x_re * x_im;
            c2.0 * x2_im + c2.1 * x2_re
        })
        .collect();

    assert_eq!(
        term.offset_bits(),
        k.0 - 2 * ctx.params.log_delta,
        "x2 branch must consume visible offset precision"
    );
    let (re_out, im_out) = ctx.decrypt_decode(&term, &mut scratch);
    assert_precision("prepared_poly2_term re", &re_out, &want_re, 20.0);
    assert_precision("prepared_poly2_term im", &im_out, &want_im, 20.0);
}

/// Evaluates `y * (c0 + c1*x + c2*x^2)` with encrypted `x` and `y`.
pub fn test_prepared_poly2_mul<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEMulPlain<BE>
        + GLWEShift<BE>
        + GLWEAdd
        + GLWENormalize<BE>
        + GLWETensoring<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let tsk = ctx.tsk();
    let m = ctx.module.n() / 2;
    let zero = vec![0.0; m];
    let base2k = Base2K(ctx.params.base2k);
    let k = TorusPrecision(ctx.params.k);
    let degree = Degree(ctx.params.n);
    let ct_tmp = CKKS::alloc(degree, base2k, k, ctx.params.log_delta);
    let mut scratch = ScratchOwned::<BE>::alloc(
        encrypt_sk_tmp_bytes(&ctx.module, &ct_tmp)
            .max(decrypt_tmp_bytes(&ctx.module, &ct_tmp))
            .max(mul_pt_tmp_bytes(&ctx.module, &ct_tmp))
            .max(square_tmp_bytes(&ctx.module, &ct_tmp, tsk))
            .max(mul_tmp_bytes(&ctx.module, &ct_tmp, &ct_tmp, tsk)),
    );

    let c0 = (0.125, -0.0625);
    let c1 = (0.625, -0.125);
    let c2 = (-0.375, 0.25);
    let pt0 = CKKSPlaintextPrepared::from_const(&ctx.module, degree, base2k, k, ctx.params.log_delta, c0, scratch.borrow());
    let pt1 = CKKSPlaintextPrepared::from_const(&ctx.module, degree, base2k, k, ctx.params.log_delta, c1, scratch.borrow());
    let pt2 = CKKSPlaintextPrepared::from_const(&ctx.module, degree, base2k, k, ctx.params.log_delta, c2, scratch.borrow());
    let (want_re, want_im) = mul_by_y_expected(ctx, c0, c1, c2);

    let x = ctx.encrypt(&ctx.re1, &ctx.im1, &mut scratch);
    let y = ctx.encrypt(&ctx.re2, &ctx.im2, &mut scratch);
    let mut x2 = CKKS::alloc(degree, base2k, k, ctx.params.log_delta);
    square(&ctx.module, &mut x2, &x, tsk, scratch.borrow());

    let mut poly = ctx.encrypt(&zero, &zero, &mut scratch);
    add_prepared_pt_inplace(&ctx.module, &mut poly, &pt0);

    let mut term1 = CKKS::alloc(degree, base2k, k, ctx.params.log_delta);
    let mut term2 = CKKS::alloc(degree, base2k, k, ctx.params.log_delta);
    mul_prepared_pt(&ctx.module, &mut term1, &x, &pt1, scratch.borrow());
    mul_prepared_pt(&ctx.module, &mut term2, &x2, &pt2, scratch.borrow());
    add_inplace(&ctx.module, &mut poly, &term1, scratch.borrow());
    add_inplace(&ctx.module, &mut poly, &term2, scratch.borrow());

    let mut res = CKKS::alloc(degree, base2k, k, ctx.params.log_delta);
    mul(&ctx.module, &mut res, &y, &poly, tsk, scratch.borrow());

    let (re_out, im_out) = ctx.decrypt_decode(&res, &mut scratch);
    assert_precision("prepared_poly2_mul re", &re_out, &want_re, 20.0);
    assert_precision("prepared_poly2_mul im", &im_out, &want_im, 20.0);
}
