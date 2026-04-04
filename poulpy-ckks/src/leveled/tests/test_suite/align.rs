//! Alignment tests: `align_to`, `align_to_inplace`, `are_cts_aligned`, `common_window`.

use super::helpers::{TestContext, assert_precision, assert_valid_ciphertext};
use crate::{
    layouts::ciphertext::CKKSCiphertext,
    leveled::{
        encryption::{decrypt_tmp_bytes, encrypt_sk_tmp_bytes},
        operations::align::{
            align_to, align_to_inplace, align_to_inplace_tmp_bytes, align_to_tmp_bytes, are_cts_aligned, common_window,
        },
    },
};
use poulpy_core::{
    GLWEAlign, GLWEDecrypt, GLWEEncryptSk, ScratchTakeCore,
    layouts::{Base2K, Degree, GLWESecretPreparedFactory, TorusPrecision},
};
use poulpy_hal::{
    api::{ModuleN, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxNormalize, VecZnxNormalizeTmpBytes},
    layouts::{Backend, Module, Scratch, ScratchOwned},
};

/// Tests `align_to` and `align_to_inplace`: re-windowing preserves the message.
pub fn test_align<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEAlign<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let base2k = Base2K(ctx.params.base2k);
    let k = TorusPrecision(ctx.params.k);
    let degree = Degree(ctx.params.n);

    let ct_tmp = CKKSCiphertext::alloc(degree, base2k, k, ctx.params.log_delta);
    let target_k = TorusPrecision(k.0 - base2k.0);
    let mut scratch = ScratchOwned::<BE>::alloc(
        encrypt_sk_tmp_bytes(&ctx.module, &ct_tmp)
            .max(decrypt_tmp_bytes(&ctx.module, &ct_tmp))
            .max(align_to_tmp_bytes(&ctx.module, &ct_tmp, target_k))
            .max(align_to_inplace_tmp_bytes(&ctx.module, &ct_tmp, target_k)),
    );

    let ct = ctx.encrypt(&ctx.re1, &ctx.im1, &mut scratch);
    assert_valid_ciphertext("align source", &ct);
    assert_eq!(ct.prefix_bits(), k.0);
    assert_eq!(ct.offset_bits(), k.0);

    // align_to: shrink k by one limb, offset follows
    let mut ct_aligned = CKKSCiphertext::alloc(degree, base2k, k, ctx.params.log_delta);
    align_to(&ctx.module, &mut ct_aligned, &ct, target_k.0, target_k, scratch.borrow());
    assert_valid_ciphertext("align_to result", &ct_aligned);
    assert_eq!(ct_aligned.prefix_bits(), target_k.0);
    assert_eq!(ct_aligned.offset_bits(), target_k.0);
    assert_eq!(ct_aligned.torus_scale_bits(), ct.torus_scale_bits());

    let (re_out, im_out) = ctx.decrypt_decode(&ct_aligned, &mut scratch);
    assert_precision("align_to re", &re_out, &ctx.re1, 18.0);
    assert_precision("align_to im", &im_out, &ctx.im1, 18.0);

    // align_to_inplace: same shrink, in place
    let mut ct_ip = ctx.encrypt(&ctx.re1, &ctx.im1, &mut scratch);
    align_to_inplace(&ctx.module, &mut ct_ip, target_k.0, target_k, scratch.borrow());
    assert_valid_ciphertext("align_to_inplace result", &ct_ip);
    assert_eq!(ct_ip.prefix_bits(), target_k.0);
    assert_eq!(ct_ip.offset_bits(), target_k.0);

    let (re_ip, im_ip) = ctx.decrypt_decode(&ct_ip, &mut scratch);
    assert_precision("align_to_inplace re", &re_ip, &ctx.re1, 18.0);
    assert_precision("align_to_inplace im", &im_ip, &ctx.im1, 18.0);

    // align_to_inplace no-op when already at target
    let mut ct_noop = ctx.encrypt(&ctx.re1, &ctx.im1, &mut scratch);
    align_to_inplace(&ctx.module, &mut ct_noop, k.0, k, scratch.borrow());
    assert_valid_ciphertext("align_to_inplace noop", &ct_noop);
    assert_eq!(ct_noop.prefix_bits(), k.0);
}

/// Tests `are_cts_aligned` and `common_window`.
pub fn test_alignment_predicates<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEAlign<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let base2k = Base2K(ctx.params.base2k);
    let k = TorusPrecision(ctx.params.k);
    let k_lo = TorusPrecision(k.0 - base2k.0);
    let degree = Degree(ctx.params.n);

    let ct_tmp = CKKSCiphertext::alloc(degree, base2k, k, ctx.params.log_delta);
    let mut scratch =
        ScratchOwned::<BE>::alloc(encrypt_sk_tmp_bytes(&ctx.module, &ct_tmp).max(align_to_tmp_bytes(&ctx.module, &ct_tmp, k_lo)));

    let ct_a = ctx.encrypt(&ctx.re1, &ctx.im1, &mut scratch);
    let ct_b = ctx.encrypt(&ctx.re2, &ctx.im2, &mut scratch);

    // Same parameters → aligned
    assert!(are_cts_aligned(&ct_a, &ct_b), "same-k ciphertexts must be aligned");

    // Different k → not aligned
    let mut ct_lo = CKKSCiphertext::alloc(degree, base2k, k, ctx.params.log_delta);
    align_to(&ctx.module, &mut ct_lo, &ct_b, k_lo.0, k_lo, scratch.borrow());
    assert!(!are_cts_aligned(&ct_a, &ct_lo), "different-k ciphertexts must not be aligned");

    // common_window of same-k pair
    let (off, target_k) = common_window(&ct_a, &ct_b);
    assert_eq!(off, k.0);
    assert_eq!(target_k, k);

    // common_window of mismatched pair
    let (off_mis, target_k_mis) = common_window(&ct_a, &ct_lo);
    assert_eq!(off_mis, k.0, "common offset must be the larger offset");
    let payload_a = ct_a.prefix_bits() - ct_a.offset_bits();
    let payload_lo = ct_lo.prefix_bits() - ct_lo.offset_bits();
    let expected_k = k.0 + payload_a.min(payload_lo);
    assert_eq!(target_k_mis.0, expected_k);
}
