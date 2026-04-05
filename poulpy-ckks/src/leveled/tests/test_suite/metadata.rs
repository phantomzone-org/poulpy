//! Metadata-state tests: valid CKKS `(k, offset_bits, torus_scale_bits)` sweeps.

use super::helpers::{TestContext, assert_precision, assert_valid_ciphertext};
use crate::leveled::operations::level::{drop_scaling_precision_inplace, drop_torus_precision};
use poulpy_core::layouts::{LWEInfos, TorusPrecision};
use poulpy_core::{GLWEDecrypt, GLWEEncryptSk, GLWEShift, ScratchTakeCore, layouts::GLWESecretPreparedFactory};
use poulpy_hal::{
    api::{ModuleN, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxNormalize, VecZnxNormalizeTmpBytes},
    layouts::{Backend, Module, Scratch, ScratchOwned},
};

/// Sweeps several valid `(k, offset_bits, torus_scale_bits)` states through decrypt and
/// the two precision-management paths.
pub fn test_metadata_sweep<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWEShift<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let mut scratch = ctx.alloc_scratch();
    let stress_min_bits = 12.0;
    let base2k = ctx.params.base2k;
    let min_k = ctx.params.log_delta.div_ceil(base2k) * base2k;
    let mut initial_ks = vec![min_k + base2k, ctx.params.k.saturating_sub(base2k), ctx.params.k];
    initial_ks.sort_unstable();
    initial_ks.dedup();
    initial_ks.retain(|k| *k <= ctx.params.k);

    let mut scale_drops = vec![0u32, 7, 13];
    scale_drops.retain(|bits| *bits < ctx.params.log_delta);

    for &k0 in &initial_ks {
        let ct0 = ctx.encrypt_with_k(&ctx.re1, &ctx.im1, TorusPrecision(k0), &mut scratch);
        assert_valid_ciphertext("metadata_sweep fresh", &ct0);
        assert_eq!(ct0.inner.k().0, k0, "encrypt_with_k must set k");
        assert_eq!(ct0.offset_bits, k0, "fresh encryption must initialize offset_bits to k");
        let (re0, im0) = ctx.decrypt_decode(&ct0, &mut scratch);
        assert_precision(&format!("metadata_sweep fresh re (k={k0})"), &re0, &ctx.re1, 20.0);
        assert_precision(&format!("metadata_sweep fresh im (k={k0})"), &im0, &ctx.im1, 20.0);

        for &scale_bits in scale_drops.iter().filter(|bits| **bits <= k0 - ctx.params.log_delta) {
            let mut ct_scale_first = ctx.encrypt_with_k(&ctx.re1, &ctx.im1, TorusPrecision(k0), &mut scratch);
            if scale_bits > 0 {
                drop_scaling_precision_inplace(&ctx.module, &mut ct_scale_first, scale_bits as usize, scratch.borrow());
            }
            assert_valid_ciphertext("metadata_sweep scale-first", &ct_scale_first);
            assert_eq!(
                ct_scale_first.torus_scale_bits,
                ctx.params.log_delta - scale_bits,
                "drop_scaling_precision_inplace must reduce torus_scale_bits"
            );
            assert_eq!(
                ct_scale_first.offset_bits, k0,
                "drop_scaling_precision_inplace must preserve offset_bits"
            );
            let (re_scale, im_scale) = ctx.decrypt_decode(&ct_scale_first, &mut scratch);
            assert_precision(
                &format!("metadata_sweep scale-first re (k={k0}, drop={scale_bits})"),
                &re_scale,
                &ctx.re1,
                stress_min_bits,
            );
            assert_precision(
                &format!("metadata_sweep scale-first im (k={k0}, drop={scale_bits})"),
                &im_scale,
                &ctx.im1,
                stress_min_bits,
            );

            let mut target_k = ct_scale_first.inner.k().0.saturating_sub(base2k);
            while target_k >= ct_scale_first.torus_scale_bits && target_k >= min_k {
                let mut ct = ctx.encrypt_with_k(&ctx.re1, &ctx.im1, TorusPrecision(k0), &mut scratch);
                if scale_bits > 0 {
                    drop_scaling_precision_inplace(&ctx.module, &mut ct, scale_bits as usize, scratch.borrow());
                }
                drop_torus_precision(&mut ct, TorusPrecision(target_k));
                assert_valid_ciphertext("metadata_sweep scale-then-torus", &ct);
                assert_eq!(ct.inner.k().0, target_k, "drop_torus_precision must set k");
                assert_eq!(ct.offset_bits, target_k, "drop_torus_precision must lower offset_bits with k");
                assert_eq!(
                    ct.torus_scale_bits, ct_scale_first.torus_scale_bits,
                    "drop_torus_precision must preserve torus_scale_bits"
                );
                let (re_out, im_out) = ctx.decrypt_decode(&ct, &mut scratch);
                assert_precision(
                    &format!("metadata_sweep scale-then-torus re (k={k0}, drop={scale_bits}, target_k={target_k})"),
                    &re_out,
                    &ctx.re1,
                    stress_min_bits,
                );
                assert_precision(
                    &format!("metadata_sweep scale-then-torus im (k={k0}, drop={scale_bits}, target_k={target_k})"),
                    &im_out,
                    &ctx.im1,
                    stress_min_bits,
                );
                target_k = target_k.saturating_sub(base2k);
            }
        }

        let mut target_k = k0.saturating_sub(base2k);
        while target_k >= ctx.params.log_delta && target_k > min_k {
            let mut ct_torus_first = ctx.encrypt_with_k(&ctx.re1, &ctx.im1, TorusPrecision(k0), &mut scratch);
            drop_torus_precision(&mut ct_torus_first, TorusPrecision(target_k));
            let max_scale_bits = (ct_torus_first.torus_scale_bits - 1)
                .min(7)
                .min(target_k - ctx.params.log_delta);
            if max_scale_bits > 0 {
                drop_scaling_precision_inplace(&ctx.module, &mut ct_torus_first, max_scale_bits as usize, scratch.borrow());
            }
            assert_valid_ciphertext("metadata_sweep torus-then-scale", &ct_torus_first);
            assert_eq!(
                ct_torus_first.torus_scale_bits,
                ctx.params.log_delta - max_scale_bits,
                "torus-then-scale must lower torus_scale_bits"
            );
            assert_eq!(
                ct_torus_first.offset_bits, target_k,
                "drop_scaling_precision_inplace must preserve torus-truncated offset_bits"
            );
            let (re_out, im_out) = ctx.decrypt_decode(&ct_torus_first, &mut scratch);
            assert_precision(
                &format!("metadata_sweep torus-then-scale re (k={k0}, target_k={target_k}, drop={max_scale_bits})"),
                &re_out,
                &ctx.re1,
                stress_min_bits,
            );
            assert_precision(
                &format!("metadata_sweep torus-then-scale im (k={k0}, target_k={target_k}, drop={max_scale_bits})"),
                &im_out,
                &ctx.im1,
                stress_min_bits,
            );
            target_k = target_k.saturating_sub(base2k);
        }
    }
}
