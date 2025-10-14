use poulpy_hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyDftToDftInplace, SvpPPolAlloc, SvpPPolAllocBytes, SvpPrepare,
        VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxAutomorphism, VecZnxAutomorphismInplace,
        VecZnxBigAddInplace, VecZnxBigAddSmallInplace, VecZnxBigAllocBytes, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxCopy, VecZnxDftAllocBytes, VecZnxDftApply, VecZnxFillUniform, VecZnxIdftApplyConsume, VecZnxNormalize,
        VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxSub, VecZnxSubInplace, VecZnxSubScalarInplace, VecZnxSwitchRing,
        VmpApplyDftToDft, VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes, VmpPMatAlloc, VmpPrepare,
    },
    layouts::{Backend, Module, ScratchOwned},
    oep::{
        ScratchAvailableImpl, ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeScalarZnxImpl, TakeSvpPPolImpl,
        TakeVecZnxBigImpl, TakeVecZnxDftImpl, TakeVecZnxImpl,
    },
    source::Source,
};

use crate::{
    encryption::SIGMA,
    layouts::{
        AutomorphismKey, AutomorphismKeyLayout, GGLWEInfos, GLWEPlaintext, GLWESecret,
        prepared::{AutomorphismKeyPrepared, GLWESecretPrepared, Prepare, PrepareAlloc},
    },
    noise::log2_std_noise_gglwe_product,
};

#[allow(clippy::too_many_arguments)]
pub fn test_gglwe_automorphism_key_automorphism<B>(module: &Module<B>)
where
    Module<B>: VecZnxDftAllocBytes
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VmpApplyDftToDft<B>
        + VmpApplyDftToDftAdd<B>
        + VecZnxDftApply<B>
        + VecZnxIdftApplyConsume<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxBigNormalize<B>
        + VecZnxAutomorphism
        + VecZnxAutomorphismInplace<B>
        + SvpPPolAllocBytes
        + VecZnxDftAllocBytes
        + VecZnxNormalizeTmpBytes
        + VmpPMatAlloc<B>
        + VmpPrepare<B>
        + SvpPrepare<B>
        + SvpApplyDftToDftInplace<B>
        + VecZnxAddScalarInplace
        + VecZnxFillUniform
        + VecZnxSubInplace
        + VecZnxAddInplace
        + VecZnxNormalizeInplace<B>
        + VecZnxAddNormal
        + VecZnxNormalize<B>
        + VecZnxSub
        + VecZnxSwitchRing
        + SvpPPolAlloc<B>
        + VecZnxBigAddInplace<B>
        + VecZnxSubScalarInplace,
    B: Backend
        + ScratchOwnedAllocImpl<B>
        + ScratchOwnedBorrowImpl<B>
        + ScratchAvailableImpl<B>
        + TakeScalarZnxImpl<B>
        + TakeVecZnxDftImpl<B>
        + TakeVecZnxImpl<B>
        + TakeSvpPPolImpl<B>
        + TakeVecZnxBigImpl<B>,
{
    let base2k: usize = 12;
    let k_in: usize = 60;
    let k_out: usize = 40;
    let dsize: usize = k_in.div_ceil(base2k);
    let p0: i64 = -1;
    let p1: i64 = -5;
    for rank in 1_usize..3 {
        for di in 1..dsize + 1 {
            let k_apply: usize = (dsize + di) * base2k;

            let n: usize = module.n();
            let dsize_in: usize = 1;

            let dnum_in: usize = k_in / (base2k * di);
            let dnum_out: usize = k_out / (base2k * di);
            let dnum_apply: usize = k_in.div_ceil(base2k * di);

            let auto_key_in_infos: AutomorphismKeyLayout = AutomorphismKeyLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_in.into(),
                dnum: dnum_in.into(),
                dsize: dsize_in.into(),
                rank: rank.into(),
            };

            let auto_key_out_infos: AutomorphismKeyLayout = AutomorphismKeyLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_out.into(),
                dnum: dnum_out.into(),
                dsize: dsize_in.into(),
                rank: rank.into(),
            };

            let auto_key_apply_infos: AutomorphismKeyLayout = AutomorphismKeyLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_apply.into(),
                dnum: dnum_apply.into(),
                dsize: di.into(),
                rank: rank.into(),
            };

            let mut auto_key_in: AutomorphismKey<Vec<u8>> = AutomorphismKey::alloc_from_infos(&auto_key_in_infos);
            let mut auto_key_out: AutomorphismKey<Vec<u8>> = AutomorphismKey::alloc_from_infos(&auto_key_out_infos);
            let mut auto_key_apply: AutomorphismKey<Vec<u8>> = AutomorphismKey::alloc_from_infos(&auto_key_apply_infos);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
                AutomorphismKey::encrypt_sk_scratch_space(module, &auto_key_in_infos)
                    | AutomorphismKey::encrypt_sk_scratch_space(module, &auto_key_apply_infos)
                    | AutomorphismKey::automorphism_scratch_space(
                        module,
                        &auto_key_out_infos,
                        &auto_key_in_infos,
                        &auto_key_apply_infos,
                    ),
            );

            let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&auto_key_in);
            sk.fill_ternary_prob(0.5, &mut source_xs);

            // gglwe_{s1}(s0) = s0 -> s1
            auto_key_in.encrypt_sk(
                module,
                p0,
                &sk,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            // gglwe_{s2}(s1) -> s1 -> s2
            auto_key_apply.encrypt_sk(
                module,
                p1,
                &sk,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            let mut auto_key_apply_prepared: AutomorphismKeyPrepared<Vec<u8>, B> =
                AutomorphismKeyPrepared::alloc_from_infos(module, &auto_key_apply_infos);

            auto_key_apply_prepared.prepare(module, &auto_key_apply, scratch.borrow());

            // gglwe_{s1}(s0) (x) gglwe_{s2}(s1) = gglwe_{s2}(s0)
            auto_key_out.automorphism(
                module,
                &auto_key_in,
                &auto_key_apply_prepared,
                scratch.borrow(),
            );

            let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&auto_key_out_infos);

            let mut sk_auto: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&auto_key_out_infos);
            sk_auto.fill_zero(); // Necessary to avoid panic of unfilled sk
            for i in 0..rank {
                module.vec_znx_automorphism(
                    module.galois_element_inv(p0 * p1),
                    &mut sk_auto.data.as_vec_znx_mut(),
                    i,
                    &sk.data.as_vec_znx(),
                    i,
                );
            }

            let sk_auto_dft: GLWESecretPrepared<Vec<u8>, B> = sk_auto.prepare_alloc(module, scratch.borrow());

            (0..auto_key_out.rank_in().into()).for_each(|col_i| {
                (0..auto_key_out.dnum().into()).for_each(|row_i| {
                    auto_key_out
                        .at(row_i, col_i)
                        .decrypt(module, &mut pt, &sk_auto_dft, scratch.borrow());

                    module.vec_znx_sub_scalar_inplace(
                        &mut pt.data,
                        0,
                        (dsize_in - 1) + row_i * dsize_in,
                        &sk.data,
                        col_i,
                    );

                    let noise_have: f64 = pt.data.std(base2k, 0).log2();
                    let noise_want: f64 = log2_std_noise_gglwe_product(
                        n as f64,
                        base2k * di,
                        0.5,
                        0.5,
                        0f64,
                        SIGMA * SIGMA,
                        0f64,
                        rank as f64,
                        k_out,
                        k_apply,
                    );

                    assert!(
                        noise_have < noise_want + 0.5,
                        "{noise_have} {}",
                        noise_want + 0.5
                    );
                });
            });
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn test_gglwe_automorphism_key_automorphism_inplace<B>(module: &Module<B>)
where
    Module<B>: VecZnxDftAllocBytes
        + VecZnxBigNormalize<B>
        + VecZnxDftApply<B>
        + SvpApplyDftToDftInplace<B>
        + VecZnxIdftApplyConsume<B>
        + VecZnxNormalizeTmpBytes
        + VecZnxFillUniform
        + VecZnxSubInplace
        + VecZnxAddInplace
        + VecZnxNormalizeInplace<B>
        + VecZnxAddNormal
        + VecZnxNormalize<B>
        + VecZnxSub
        + SvpPrepare<B>
        + SvpPPolAllocBytes
        + SvpPPolAlloc<B>
        + VecZnxDftAllocBytes
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VmpApplyDftToDftTmpBytes
        + VmpApplyDftToDft<B>
        + VmpApplyDftToDftAdd<B>
        + VecZnxDftApply<B>
        + VecZnxIdftApplyConsume<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxBigNormalize<B>
        + VecZnxAutomorphism
        + VecZnxSwitchRing
        + VecZnxAddScalarInplace
        + VecZnxAutomorphism
        + VecZnxAutomorphismInplace<B>
        + VecZnxDftAllocBytes
        + VecZnxBigAllocBytes
        + VecZnxDftApply<B>
        + SvpApplyDftToDftInplace<B>
        + VecZnxIdftApplyConsume<B>
        + VecZnxBigAddInplace<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxBigNormalize<B>
        + VecZnxNormalizeTmpBytes
        + VecZnxSubScalarInplace
        + VecZnxCopy
        + VmpPMatAlloc<B>
        + VmpPrepare<B>,
    B: Backend
        + ScratchOwnedAllocImpl<B>
        + ScratchOwnedBorrowImpl<B>
        + ScratchAvailableImpl<B>
        + TakeScalarZnxImpl<B>
        + TakeVecZnxDftImpl<B>
        + TakeVecZnxImpl<B>
        + TakeSvpPPolImpl<B>
        + TakeVecZnxBigImpl<B>,
{
    let base2k: usize = 12;
    let k_in: usize = 60;
    let dsize: usize = k_in.div_ceil(base2k);
    let p0: i64 = -1;
    let p1: i64 = -5;
    for rank in 1_usize..3 {
        for di in 1..dsize + 1 {
            let k_apply: usize = (dsize + di) * base2k;

            let n: usize = module.n();
            let dsize_in: usize = 1;

            let dnum_in: usize = k_in / (base2k * di);
            let dnum_apply: usize = k_in.div_ceil(base2k * di);

            let auto_key_layout: AutomorphismKeyLayout = AutomorphismKeyLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_in.into(),
                dnum: dnum_in.into(),
                dsize: dsize_in.into(),
                rank: rank.into(),
            };

            let auto_key_apply_layout: AutomorphismKeyLayout = AutomorphismKeyLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_apply.into(),
                dnum: dnum_apply.into(),
                dsize: di.into(),
                rank: rank.into(),
            };

            let mut auto_key: AutomorphismKey<Vec<u8>> = AutomorphismKey::alloc_from_infos(&auto_key_layout);
            let mut auto_key_apply: AutomorphismKey<Vec<u8>> = AutomorphismKey::alloc_from_infos(&auto_key_apply_layout);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
                AutomorphismKey::encrypt_sk_scratch_space(module, &auto_key)
                    | AutomorphismKey::encrypt_sk_scratch_space(module, &auto_key_apply)
                    | AutomorphismKey::automorphism_inplace_scratch_space(module, &auto_key, &auto_key_apply),
            );

            let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&auto_key);
            sk.fill_ternary_prob(0.5, &mut source_xs);

            // gglwe_{s1}(s0) = s0 -> s1
            auto_key.encrypt_sk(
                module,
                p0,
                &sk,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            // gglwe_{s2}(s1) -> s1 -> s2
            auto_key_apply.encrypt_sk(
                module,
                p1,
                &sk,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            let mut auto_key_apply_prepared: AutomorphismKeyPrepared<Vec<u8>, B> =
                AutomorphismKeyPrepared::alloc_from_infos(module, &auto_key_apply_layout);

            auto_key_apply_prepared.prepare(module, &auto_key_apply, scratch.borrow());

            // gglwe_{s1}(s0) (x) gglwe_{s2}(s1) = gglwe_{s2}(s0)
            auto_key.automorphism_inplace(module, &auto_key_apply_prepared, scratch.borrow());

            let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&auto_key);

            let mut sk_auto: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&auto_key);
            sk_auto.fill_zero(); // Necessary to avoid panic of unfilled sk

            for i in 0..rank {
                module.vec_znx_automorphism(
                    module.galois_element_inv(p0 * p1),
                    &mut sk_auto.data.as_vec_znx_mut(),
                    i,
                    &sk.data.as_vec_znx(),
                    i,
                );
            }

            let sk_auto_dft: GLWESecretPrepared<Vec<u8>, B> = sk_auto.prepare_alloc(module, scratch.borrow());

            (0..auto_key.rank_in().into()).for_each(|col_i| {
                (0..auto_key.dnum().into()).for_each(|row_i| {
                    auto_key
                        .at(row_i, col_i)
                        .decrypt(module, &mut pt, &sk_auto_dft, scratch.borrow());
                    module.vec_znx_sub_scalar_inplace(
                        &mut pt.data,
                        0,
                        (dsize_in - 1) + row_i * dsize_in,
                        &sk.data,
                        col_i,
                    );

                    let noise_have: f64 = pt.data.std(base2k, 0).log2();
                    let noise_want: f64 = log2_std_noise_gglwe_product(
                        n as f64,
                        base2k * di,
                        0.5,
                        0.5,
                        0f64,
                        SIGMA * SIGMA,
                        0f64,
                        rank as f64,
                        k_in,
                        k_apply,
                    );

                    assert!(
                        noise_have < noise_want + 0.5,
                        "{noise_have} {}",
                        noise_want + 0.5
                    );
                });
            });
        }
    }
}
