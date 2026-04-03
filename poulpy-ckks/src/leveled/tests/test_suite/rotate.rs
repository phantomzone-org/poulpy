//! Slot rotation tests (out-of-place and in-place round-trip).

use super::helpers::{TestContext, assert_precision};
use crate::{
    layouts::ciphertext::CKKSCiphertext,
    leveled::{
        encryption::{decrypt_tmp_bytes, encrypt_sk_tmp_bytes},
        operations::rotate::{rotate, rotate_inplace, rotate_tmp_bytes},
    },
};
use poulpy_core::{
    GLWEAutomorphism, GLWEDecrypt, GLWEEncryptSk, ScratchTakeCore,
    layouts::{Base2K, Degree, GLWESecretPreparedFactory, TorusPrecision},
};
use poulpy_hal::{
    api::{ModuleN, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxNormalize, VecZnxNormalizeTmpBytes},
    layouts::{Backend, GaloisElement, Module, Scratch, ScratchOwned},
};

/// Tests rotation by verifying the round-trip property:
/// Rotate(Rotate(ct, forward_key), inverse_key) ≈ ct
pub fn test_rotate<BE: Backend>(ctx: &TestContext<BE>, rotations: &[i64])
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
    let atks = ctx.atks();

    let ct_tmp = CKKSCiphertext::alloc(degree, base2k, k, ctx.params.log_delta);
    let any_key = atks.values().next().unwrap();
    let mut scratch = ScratchOwned::<BE>::alloc(
        encrypt_sk_tmp_bytes(&ctx.module, &ct_tmp)
            .max(decrypt_tmp_bytes(&ctx.module, &ct_tmp))
            .max(rotate_tmp_bytes(&ctx.module, &ct_tmp, any_key)),
    );

    let ct = ctx.encrypt(&ctx.re1, &ctx.im1, &mut scratch);

    for &r in rotations {
        let fwd_gal = ctx.module.galois_element(r);
        let inv_gal = ctx.module.galois_element_inv(fwd_gal);
        let fwd_key = ctx.atk(fwd_gal);
        let inv_key = ctx.atk(inv_gal);

        // out-of-place round-trip: forward then inverse
        let mut ct_rot = CKKSCiphertext::alloc(degree, base2k, k, ctx.params.log_delta);
        rotate(&ctx.module, &mut ct_rot, &ct, fwd_key, scratch.borrow());

        // Verify the rotation is non-trivial (output differs from input)
        let (re_rot, _im_rot) = ctx.decrypt_decode(&ct_rot, &mut scratch);
        let max_diff: f64 = ctx
            .re1
            .iter()
            .zip(re_rot.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(max_diff > 0.01, "rotate({r}) should be non-trivial, max_diff={max_diff}");

        let mut ct_back = CKKSCiphertext::alloc(degree, base2k, k, ctx.params.log_delta);
        rotate(&ctx.module, &mut ct_back, &ct_rot, inv_key, scratch.borrow());

        let (re_back, im_back) = ctx.decrypt_decode(&ct_back, &mut scratch);
        assert_precision(&format!("rotate({r}) round-trip re"), &re_back, &ctx.re1, 20.0);
        assert_precision(&format!("rotate({r}) round-trip im"), &im_back, &ctx.im1, 20.0);

        // in-place round-trip
        let mut ct_ip = ctx.encrypt(&ctx.re1, &ctx.im1, &mut scratch);
        rotate_inplace(&ctx.module, &mut ct_ip, fwd_key, scratch.borrow());
        rotate_inplace(&ctx.module, &mut ct_ip, inv_key, scratch.borrow());
        let (re_ip, im_ip) = ctx.decrypt_decode(&ct_ip, &mut scratch);
        assert_precision(&format!("rotate_inplace({r}) round-trip re"), &re_ip, &ctx.re1, 20.0);
        assert_precision(&format!("rotate_inplace({r}) round-trip im"), &im_ip, &ctx.im1, 20.0);
    }
}
