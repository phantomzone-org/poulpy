use poulpy_hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyDftToDftInplace, SvpPPolAlloc, SvpPPolBytesOf, SvpPrepare,
        VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxBigAddInplace, VecZnxBigAddSmallInplace,
        VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftApply, VecZnxDftBytesOf, VecZnxFillUniform,
        VecZnxIdftApplyConsume, VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxSub, VecZnxSubInplace,
        VecZnxSwitchRing, VmpApplyDftToDft, VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes, VmpPMatAlloc, VmpPrepare,
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
        GLWE, GLWELayout, GLWEPlaintext, GLWESecret, GLWESwitchingKey, GLWESwitchingKeyLayout,
        prepared::{GLWESecretPrepared, GLWESwitchingKeyPrepared, PrepareAlloc},
    },
    noise::log2_std_noise_gglwe_product,
};

#[allow(clippy::too_many_arguments)]
pub fn test_glwe_keyswitch<B>(module: &Module<B>)
where
    Module<B>: VecZnxDftBytesOf
        + VecZnxBigNormalize<B>
        + VecZnxDftApply<B>
        + SvpApplyDftToDftInplace<B>
        + VecZnxIdftApplyConsume<B>
        + VecZnxFillUniform
        + VecZnxSubInplace
        + VecZnxAddInplace
        + VecZnxNormalizeInplace<B>
        + VecZnxAddNormal
        + VecZnxNormalize<B>
        + VecZnxSub
        + SvpPrepare<B>
        + SvpPPolBytesOf
        + SvpPPolAlloc<B>
        + VecZnxBigBytesOf
        + VecZnxBigAddInplace<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxNormalizeTmpBytes
        + VecZnxAddScalarInplace
        + VmpPMatAlloc<B>
        + VmpPrepare<B>
        + VmpApplyDftToDftTmpBytes
        + VmpApplyDftToDft<B>
        + VmpApplyDftToDftAdd<B>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxSwitchRing,
    B: Backend
        + TakeVecZnxDftImpl<B>
        + TakeVecZnxBigImpl<B>
        + TakeSvpPPolImpl<B>
        + ScratchOwnedAllocImpl<B>
        + ScratchOwnedBorrowImpl<B>
        + ScratchAvailableImpl<B>
        + TakeScalarZnxImpl<B>
        + TakeVecZnxImpl<B>,
{
    let base2k: usize = 12;
    let k_in: usize = 45;
    let dsize: usize = k_in.div_ceil(base2k);

    for rank_in in 1_usize..3 {
        for rank_out in 1_usize..3 {
            for di in 1_usize..dsize + 1 {
                let k_ksk: usize = k_in + base2k * di;
                let k_out: usize = k_ksk; // better capture noise

                let n: usize = module.n();
                let dnum: usize = k_in.div_ceil(base2k * dsize);

                let glwe_in_infos: GLWELayout = GLWELayout {
                    n: n.into(),
                    base2k: base2k.into(),
                    k: k_in.into(),
                    rank: rank_in.into(),
                };

                let glwe_out_infos: GLWELayout = GLWELayout {
                    n: n.into(),
                    base2k: base2k.into(),
                    k: k_out.into(),
                    rank: rank_out.into(),
                };

                let key_apply: GLWESwitchingKeyLayout = GLWESwitchingKeyLayout {
                    n: n.into(),
                    base2k: base2k.into(),
                    k: k_ksk.into(),
                    dnum: dnum.into(),
                    dsize: di.into(),
                    rank_in: rank_in.into(),
                    rank_out: rank_out.into(),
                };

                let mut ksk: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc_from_infos(&key_apply);
                let mut glwe_in: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_in_infos);
                let mut glwe_out: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_out_infos);
                let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_in_infos);

                let mut source_xs: Source = Source::new([0u8; 32]);
                let mut source_xe: Source = Source::new([0u8; 32]);
                let mut source_xa: Source = Source::new([0u8; 32]);

                module.vec_znx_fill_uniform(base2k, &mut pt_want.data, 0, &mut source_xa);

                let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
                    GLWESwitchingKey::encrypt_sk_scratch_space(module, &key_apply)
                        | GLWE::encrypt_sk_scratch_space(module, &glwe_in_infos)
                        | GLWE::keyswitch_scratch_space(module, &glwe_out_infos, &glwe_in_infos, &key_apply),
                );

                let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc(n.into(), rank_in.into());
                sk_in.fill_ternary_prob(0.5, &mut source_xs);
                let sk_in_prepared: GLWESecretPrepared<Vec<u8>, B> = sk_in.prepare_alloc(module, scratch.borrow());

                let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc(n.into(), rank_out.into());
                sk_out.fill_ternary_prob(0.5, &mut source_xs);
                let sk_out_prepared: GLWESecretPrepared<Vec<u8>, B> = sk_out.prepare_alloc(module, scratch.borrow());

                ksk.encrypt_sk(
                    module,
                    &sk_in,
                    &sk_out,
                    &mut source_xa,
                    &mut source_xe,
                    scratch.borrow(),
                );

                glwe_in.encrypt_sk(
                    module,
                    &pt_want,
                    &sk_in_prepared,
                    &mut source_xa,
                    &mut source_xe,
                    scratch.borrow(),
                );

                let ksk_prepared: GLWESwitchingKeyPrepared<Vec<u8>, B> = ksk.prepare_alloc(module, scratch.borrow());

                glwe_out.keyswitch(module, &glwe_in, &ksk_prepared, scratch.borrow());

                let max_noise: f64 = log2_std_noise_gglwe_product(
                    module.n() as f64,
                    base2k * dsize,
                    0.5,
                    0.5,
                    0f64,
                    SIGMA * SIGMA,
                    0f64,
                    rank_in as f64,
                    k_in,
                    k_ksk,
                );

                glwe_out.assert_noise(module, &sk_out_prepared, &pt_want, max_noise + 0.5);
            }
        }
    }
}

pub fn test_glwe_keyswitch_inplace<B>(module: &Module<B>)
where
    Module<B>: VecZnxDftBytesOf
        + VecZnxBigNormalize<B>
        + VecZnxDftApply<B>
        + SvpApplyDftToDftInplace<B>
        + VecZnxIdftApplyConsume<B>
        + VecZnxFillUniform
        + VecZnxSubInplace
        + VecZnxAddInplace
        + VecZnxNormalizeInplace<B>
        + VecZnxAddNormal
        + VecZnxNormalize<B>
        + VecZnxSub
        + SvpPrepare<B>
        + SvpPPolBytesOf
        + SvpPPolAlloc<B>
        + VecZnxBigBytesOf
        + VecZnxBigAddInplace<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxNormalizeTmpBytes
        + VecZnxAddScalarInplace
        + VmpPMatAlloc<B>
        + VmpPrepare<B>
        + VmpApplyDftToDftTmpBytes
        + VmpApplyDftToDft<B>
        + VmpApplyDftToDftAdd<B>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxSwitchRing,
    B: Backend
        + TakeVecZnxDftImpl<B>
        + TakeVecZnxBigImpl<B>
        + TakeSvpPPolImpl<B>
        + ScratchOwnedAllocImpl<B>
        + ScratchOwnedBorrowImpl<B>
        + ScratchAvailableImpl<B>
        + TakeScalarZnxImpl<B>
        + TakeVecZnxImpl<B>,
{
    let base2k: usize = 12;
    let k_out: usize = 45;
    let dsize: usize = k_out.div_ceil(base2k);

    for rank in 1_usize..3 {
        for di in 1..dsize + 1 {
            let k_ksk: usize = k_out + base2k * di;

            let n: usize = module.n();
            let dnum: usize = k_out.div_ceil(base2k * dsize);

            let glwe_out_infos: GLWELayout = GLWELayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_out.into(),
                rank: rank.into(),
            };

            let key_apply_infos: GLWESwitchingKeyLayout = GLWESwitchingKeyLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_ksk.into(),
                dnum: dnum.into(),
                dsize: di.into(),
                rank_in: rank.into(),
                rank_out: rank.into(),
            };

            let mut key_apply: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc_from_infos(&key_apply_infos);
            let mut glwe_out: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_out_infos);
            let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_out_infos);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            module.vec_znx_fill_uniform(base2k, &mut pt_want.data, 0, &mut source_xa);

            let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
                GLWESwitchingKey::encrypt_sk_scratch_space(module, &key_apply_infos)
                    | GLWE::encrypt_sk_scratch_space(module, &glwe_out_infos)
                    | GLWE::keyswitch_inplace_scratch_space(module, &glwe_out_infos, &key_apply_infos),
            );

            let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc(n.into(), rank.into());
            sk_in.fill_ternary_prob(0.5, &mut source_xs);
            let sk_in_prepared: GLWESecretPrepared<Vec<u8>, B> = sk_in.prepare_alloc(module, scratch.borrow());

            let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc(n.into(), rank.into());
            sk_out.fill_ternary_prob(0.5, &mut source_xs);
            let sk_out_prepared: GLWESecretPrepared<Vec<u8>, B> = sk_out.prepare_alloc(module, scratch.borrow());

            key_apply.encrypt_sk(
                module,
                &sk_in,
                &sk_out,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            glwe_out.encrypt_sk(
                module,
                &pt_want,
                &sk_in_prepared,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            let ksk_prepared: GLWESwitchingKeyPrepared<Vec<u8>, B> = key_apply.prepare_alloc(module, scratch.borrow());

            glwe_out.keyswitch_inplace(module, &ksk_prepared, scratch.borrow());

            let max_noise: f64 = log2_std_noise_gglwe_product(
                module.n() as f64,
                base2k * dsize,
                0.5,
                0.5,
                0f64,
                SIGMA * SIGMA,
                0f64,
                rank as f64,
                k_out,
                k_ksk,
            );

            glwe_out.assert_noise(module, &sk_out_prepared, &pt_want, max_noise + 0.5);
        }
    }
}
