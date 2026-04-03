//! Conjugation tests (out-of-place and in-place).

use super::helpers::{TestContext, assert_precision};
use crate::{
    layouts::ciphertext::CKKSCiphertext,
    leveled::{
        encryption::{decrypt_tmp_bytes, encrypt_sk_tmp_bytes},
        operations::conjugate::{conjugate, conjugate_inplace, conjugate_tmp_bytes},
    },
};
use poulpy_core::{
    GLWEAutomorphism, GLWEDecrypt, GLWEEncryptSk, ScratchTakeCore,
    layouts::{Base2K, Degree, GLWESecretPreparedFactory, TorusPrecision},
};
use poulpy_hal::{
    api::{ModuleN, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxNormalize, VecZnxNormalizeTmpBytes},
    layouts::{Backend, Module, Scratch, ScratchOwned},
};

/// Verifies that conjugation flips the imaginary part while preserving the
/// real part (out-of-place and in-place).
pub fn test_conjugate<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEAutomorphism<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let base2k = Base2K(ctx.params.base2k);
    let k = TorusPrecision(ctx.params.k);
    let degree = Degree(ctx.params.n);

    let conj_key = ctx.atk(-1);
    let ct_tmp = CKKSCiphertext::alloc(degree, base2k, k, ctx.params.log_delta);
    let mut scratch = ScratchOwned::<BE>::alloc(
        encrypt_sk_tmp_bytes(&ctx.module, &ct_tmp)
            .max(decrypt_tmp_bytes(&ctx.module, &ct_tmp))
            .max(conjugate_tmp_bytes(&ctx.module, &ct_tmp, conj_key)),
    );

    let ct = ctx.encrypt(&ctx.re1, &ctx.im1, &mut scratch);

    // out-of-place
    let mut ct_res = CKKSCiphertext::alloc(degree, base2k, k, ctx.params.log_delta);
    conjugate(&ctx.module, &mut ct_res, &ct, conj_key, scratch.borrow());
    let (re_out, im_out) = ctx.decrypt_decode(&ct_res, &mut scratch);

    let want_re: Vec<f64> = ctx.re1.clone();
    let want_im: Vec<f64> = ctx.im1.iter().map(|&v| -v).collect();
    assert_precision("conjugate re", &re_out, &want_re, 20.0);
    assert_precision("conjugate im", &im_out, &want_im, 20.0);

    // in-place
    let mut ct_ip = ctx.encrypt(&ctx.re1, &ctx.im1, &mut scratch);
    conjugate_inplace(&ctx.module, &mut ct_ip, conj_key, scratch.borrow());
    let (re_ip, im_ip) = ctx.decrypt_decode(&ct_ip, &mut scratch);
    assert_precision("conjugate_inplace re", &re_ip, &want_re, 20.0);
    assert_precision("conjugate_inplace im", &im_ip, &want_im, 20.0);
}
