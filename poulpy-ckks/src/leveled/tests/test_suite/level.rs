//! Level-management tests: rescale-adjacent helpers, scale-preserving division, and precision drops.

use super::helpers::{TestContext, TestLevelBackend, assert_precision, assert_valid_ciphertext};
use crate::{
    layouts::ciphertext::CKKS,
    leveled::{
        encryption::{decrypt_tmp_bytes, encrypt_sk_tmp_bytes},
        operations::level::{
            div_pow2, div_pow2_inplace, div_pow2_tmp_bytes, drop_scaling_precision, drop_scaling_precision_inplace,
            drop_torus_precision,
        },
    },
};
use poulpy_core::layouts::{Base2K, Degree, GLWE, LWEInfos, TorusPrecision};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, ScratchOwned},
};

/// Verifies scale-preserving division by `2^bits` (out-of-place and in-place).
pub fn test_div_pow2<BE: TestLevelBackend>(ctx: &TestContext<BE>) {
    let bits = 7usize;
    let scale = (1u64 << bits) as f64;
    let m = ctx.module.n() / 2;
    let base2k = Base2K(ctx.params.base2k);
    let k = TorusPrecision(ctx.params.k);
    let degree = Degree(ctx.params.n);

    let ct_tmp = CKKS::alloc(degree, base2k, k, ctx.params.log_delta);
    let mut scratch = ScratchOwned::<BE>::alloc(
        encrypt_sk_tmp_bytes(&ctx.module, &ct_tmp)
            .max(decrypt_tmp_bytes(&ctx.module, &ct_tmp))
            .max(div_pow2_tmp_bytes(&ctx.module)),
    );

    let ct = ctx.encrypt(&ctx.re1, &ctx.im1, &mut scratch);
    assert_valid_ciphertext("div_pow2 input", &ct);
    let want_re: Vec<f64> = (0..m).map(|j| ctx.re1[j] / scale).collect();
    let want_im: Vec<f64> = (0..m).map(|j| ctx.im1[j] / scale).collect();

    let mut ct_res = CKKS::alloc(degree, base2k, k, ctx.params.log_delta);
    div_pow2(&ctx.module, &mut ct_res, &ct, bits, scratch.borrow());
    assert_valid_ciphertext("div_pow2 result", &ct_res);
    assert_eq!(ct_res.k(), ct.k(), "div_pow2 must preserve k");
    assert_eq!(ct_res.size(), ct.size(), "div_pow2 must preserve size");
    assert_eq!(
        ct_res.torus_scale_bits(), ct.torus_scale_bits(),
        "div_pow2 must preserve torus_scale_bits"
    );
    let (re_out, im_out) = ctx.decrypt_decode(&ct_res, &mut scratch);
    assert_precision("div_pow2 re", &re_out, &want_re, 20.0);
    assert_precision("div_pow2 im", &im_out, &want_im, 20.0);

    let mut ct_ip = ctx.encrypt(&ctx.re1, &ctx.im1, &mut scratch);
    div_pow2_inplace(&ctx.module, &mut ct_ip, bits, scratch.borrow());
    assert_valid_ciphertext("div_pow2_inplace result", &ct_ip);
    assert_eq!(ct_ip.k(), k, "div_pow2_inplace must preserve k");
    assert_eq!(
        ct_ip.torus_scale_bits(), ctx.params.log_delta,
        "div_pow2_inplace must preserve torus_scale_bits"
    );
    let (re_ip, im_ip) = ctx.decrypt_decode(&ct_ip, &mut scratch);
    assert_precision("div_pow2_inplace re", &re_ip, &want_re, 20.0);
    assert_precision("div_pow2_inplace im", &im_ip, &want_im, 20.0);
}

/// Verifies division by `2^bits` while reducing `torus_scale_bits` by the same amount.
/// Decoding should still recover the original message, but with fewer scale bits.
pub fn test_drop_scaling_precision<BE: TestLevelBackend>(ctx: &TestContext<BE>) {
    let bits = 7usize;
    let m = ctx.module.n() / 2;
    let base2k = Base2K(ctx.params.base2k);
    let k = TorusPrecision(ctx.params.k);
    let degree = Degree(ctx.params.n);

    let ct_tmp = CKKS::alloc(degree, base2k, k, ctx.params.log_delta);
    let mut scratch = ScratchOwned::<BE>::alloc(
        encrypt_sk_tmp_bytes(&ctx.module, &ct_tmp)
            .max(decrypt_tmp_bytes(&ctx.module, &ct_tmp))
            .max(div_pow2_tmp_bytes(&ctx.module)),
    );

    let ct = ctx.encrypt(&ctx.re1, &ctx.im1, &mut scratch);
    assert_valid_ciphertext("drop_scaling_precision input", &ct);
    let want_re: Vec<f64> = (0..m).map(|j| ctx.re1[j]).collect();
    let want_im: Vec<f64> = (0..m).map(|j| ctx.im1[j]).collect();
    let expected_torus_scale_bits = ct.torus_scale_bits().saturating_sub(bits as u32);

    let mut ct_res = CKKS::alloc(degree, base2k, k, ctx.params.log_delta);
    drop_scaling_precision(&ctx.module, &mut ct_res, &ct, bits, scratch.borrow());
    assert_valid_ciphertext("drop_scaling_precision result", &ct_res);
    assert_eq!(ct_res.k(), ct.k(), "drop_scaling_precision must preserve k");
    assert_eq!(
        ct_res.size(),
        ct.size(),
        "drop_scaling_precision must preserve size"
    );
    assert_eq!(
        ct_res.torus_scale_bits(), expected_torus_scale_bits,
        "drop_scaling_precision must reduce torus_scale_bits by bits"
    );
    assert_eq!(
        ct_res.offset_bits(), ct.offset_bits(),
        "drop_scaling_precision must preserve offset_bits"
    );
    let (re_out, im_out) = ctx.decrypt_decode(&ct_res, &mut scratch);
    assert_precision("drop_scaling_precision re", &re_out, &want_re, 20.0);
    assert_precision("drop_scaling_precision im", &im_out, &want_im, 20.0);

    let mut ct_ip = ctx.encrypt(&ctx.re1, &ctx.im1, &mut scratch);
    drop_scaling_precision_inplace(&ctx.module, &mut ct_ip, bits, scratch.borrow());
    assert_valid_ciphertext("drop_scaling_precision_inplace result", &ct_ip);
    assert_eq!(ct_ip.k(), k, "drop_scaling_precision_inplace must preserve k");
    assert_eq!(
        ct_ip.torus_scale_bits(),
        ctx.params.log_delta - bits as u32,
        "drop_scaling_precision_inplace must reduce torus_scale_bits by bits"
    );
    assert_eq!(
        ct_ip.offset_bits(), k.0,
        "drop_scaling_precision_inplace must preserve offset_bits"
    );
    let (re_ip, im_ip) = ctx.decrypt_decode(&ct_ip, &mut scratch);
    assert_precision("drop_scaling_precision_inplace re", &re_ip, &want_re, 20.0);
    assert_precision("drop_scaling_precision_inplace im", &im_ip, &want_im, 20.0);
}

/// Verifies prefix truncation: lowering the torus precision updates `offset_bits`
/// but preserves the decoded message and `torus_scale_bits`.
pub fn test_drop_torus_precision<BE: TestLevelBackend>(ctx: &TestContext<BE>) {
    let degree = Degree(ctx.params.n);
    let base2k = Base2K(ctx.params.base2k);
    let k = TorusPrecision(ctx.params.k);

    let ct_tmp = CKKS::alloc(degree, base2k, k, ctx.params.log_delta);
    let mut scratch =
        ScratchOwned::<BE>::alloc(encrypt_sk_tmp_bytes(&ctx.module, &ct_tmp).max(decrypt_tmp_bytes(&ctx.module, &ct_tmp)));

    let mut ct = ctx.encrypt(&ctx.re1, &ctx.im1, &mut scratch);
    assert_valid_ciphertext("drop_torus_precision input", &ct);
    let old_size = ct.size();
    let old_torus_scale_bits = ct.torus_scale_bits();
    let target_k = TorusPrecision(k.0 - base2k.0);

    drop_torus_precision(&mut ct, target_k);
    assert_valid_ciphertext("drop_torus_precision result", &ct);

    assert_eq!(ct.k(), target_k, "drop_torus_precision must update k");
    assert!(ct.size() < old_size, "drop_torus_precision must shrink active size");
    assert_eq!(
        ct.offset_bits(), target_k.0,
        "drop_torus_precision must lower offset_bits to target_k"
    );
    assert_eq!(
        ct.torus_scale_bits(), old_torus_scale_bits,
        "drop_torus_precision must preserve torus_scale_bits"
    );

    let (re_out, im_out) = ctx.decrypt_decode(&ct, &mut scratch);
    assert_precision("drop_torus_precision re", &re_out, &ctx.re1, 20.0);
    assert_precision("drop_torus_precision im", &im_out, &ctx.im1, 20.0);
}
