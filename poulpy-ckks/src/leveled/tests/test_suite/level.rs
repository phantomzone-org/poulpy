//! Level-management tests: rescale-adjacent helpers and scale-preserving division.

use super::helpers::{TestContext, assert_precision};
use crate::{
    layouts::ciphertext::CKKSCiphertext,
    leveled::{
        encryption::{decrypt_tmp_bytes, encrypt_sk_tmp_bytes},
        operations::level::{div_pow2, div_pow2_inplace, div_pow2_tmp_bytes},
    },
};
use poulpy_core::{
    GLWECopy, GLWEDecrypt, GLWEEncryptSk, GLWEShift, ScratchTakeCore,
    layouts::{Base2K, Degree, GLWESecretPreparedFactory, LWEInfos, TorusPrecision},
};
use poulpy_hal::{
    api::{ModuleN, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxNormalize, VecZnxNormalizeTmpBytes},
    layouts::{Backend, Module, Scratch, ScratchOwned},
};

/// Verifies scale-preserving division by `2^bits` (out-of-place and in-place).
pub fn test_div_pow2<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + GLWECopy
        + GLWEShift<BE>
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let bits = 7usize;
    let scale = (1u64 << bits) as f64;
    let m = ctx.module.n() / 2;
    let base2k = Base2K(ctx.params.base2k);
    let k = TorusPrecision(ctx.params.k);
    let degree = Degree(ctx.params.n);

    let ct_tmp = CKKSCiphertext::alloc(degree, base2k, k, ctx.params.log_delta);
    let mut scratch = ScratchOwned::<BE>::alloc(
        encrypt_sk_tmp_bytes(&ctx.module, &ct_tmp)
            .max(decrypt_tmp_bytes(&ctx.module, &ct_tmp))
            .max(div_pow2_tmp_bytes(&ctx.module)),
    );

    let ct = ctx.encrypt(&ctx.re1, &ctx.im1, &mut scratch);
    let want_re: Vec<f64> = (0..m).map(|j| ctx.re1[j] / scale).collect();
    let want_im: Vec<f64> = (0..m).map(|j| ctx.im1[j] / scale).collect();

    let mut ct_res = CKKSCiphertext::alloc(degree, base2k, k, ctx.params.log_delta);
    div_pow2(&ctx.module, &mut ct_res, &ct, bits, scratch.borrow());
    assert_eq!(ct_res.inner.k(), ct.inner.k(), "div_pow2 must preserve k");
    assert_eq!(ct_res.inner.size(), ct.inner.size(), "div_pow2 must preserve size");
    assert_eq!(ct_res.log_delta, ct.log_delta, "div_pow2 must preserve log_delta");
    let (re_out, im_out) = ctx.decrypt_decode(&ct_res, &mut scratch);
    assert_precision("div_pow2 re", &re_out, &want_re, 20.0);
    assert_precision("div_pow2 im", &im_out, &want_im, 20.0);

    let mut ct_ip = ctx.encrypt(&ctx.re1, &ctx.im1, &mut scratch);
    div_pow2_inplace(&ctx.module, &mut ct_ip, bits, scratch.borrow());
    assert_eq!(ct_ip.inner.k(), k, "div_pow2_inplace must preserve k");
    assert_eq!(
        ct_ip.log_delta, ctx.params.log_delta,
        "div_pow2_inplace must preserve log_delta"
    );
    let (re_ip, im_ip) = ctx.decrypt_decode(&ct_ip, &mut scratch);
    assert_precision("div_pow2_inplace re", &re_ip, &want_re, 20.0);
    assert_precision("div_pow2_inplace im", &im_ip, &want_im, 20.0);
}
