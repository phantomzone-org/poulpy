use poulpy_hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyDftToDftInplace, SvpPPolAlloc, SvpPPolAllocBytes, SvpPrepare,
        VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxBigAddInplace, VecZnxBigAddSmallInplace,
        VecZnxBigAllocBytes, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftAllocBytes, VecZnxDftApply,
        VecZnxFillUniform, VecZnxIdftApplyConsume, VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxSub,
        VecZnxSubInplace, VecZnxSubScalarInplace, VecZnxSwitchRing, VmpApplyDftToDft, VmpApplyDftToDftAdd,
        VmpApplyDftToDftTmpBytes, VmpPMatAlloc, VmpPrepare,
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
        GLWESecret, GLWESwitchingKey, GLWESwitchingKeyLayout,
        prepared::{GLWESecretPrepared, GLWESwitchingKeyPrepared, PrepareAlloc},
    },
    noise::log2_std_noise_gglwe_product,
};

pub fn test_gglwe_switching_key_keyswitch<B>(module: &Module<B>)
where
    Module<B>: VecZnxDftAllocBytes
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
        + SvpPPolAllocBytes
        + SvpPPolAlloc<B>
        + VecZnxBigAllocBytes
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
        + VecZnxSwitchRing
        + VecZnxSubScalarInplace,
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
    let k_in: usize = 60;
    let dsize: usize = k_in.div_ceil(base2k);

    for rank_in_s0s1 in 1_usize..3 {
        for rank_out_s0s1 in 1_usize..3 {
            for rank_out_s1s2 in 1_usize..3 {
                for di in 1_usize..dsize + 1 {
                    let k_ksk: usize = k_in + base2k * di;
                    let k_out: usize = k_ksk; // Better capture noise.

                    let n: usize = module.n();
                    let dnum: usize = k_in / base2k;
                    let dnum_apply: usize = k_in.div_ceil(base2k * di);
                    let dsize_in: usize = 1;

                    let gglwe_s0s1_infos: GLWESwitchingKeyLayout = GLWESwitchingKeyLayout {
                        n: n.into(),
                        base2k: base2k.into(),
                        k: k_in.into(),
                        dnum: dnum.into(),
                        dsize: dsize_in.into(),
                        rank_in: rank_in_s0s1.into(),
                        rank_out: rank_out_s0s1.into(),
                    };

                    let gglwe_s1s2_infos: GLWESwitchingKeyLayout = GLWESwitchingKeyLayout {
                        n: n.into(),
                        base2k: base2k.into(),
                        k: k_ksk.into(),
                        dnum: dnum_apply.into(),
                        dsize: di.into(),
                        rank_in: rank_out_s0s1.into(),
                        rank_out: rank_out_s1s2.into(),
                    };

                    let gglwe_s0s2_infos: GLWESwitchingKeyLayout = GLWESwitchingKeyLayout {
                        n: n.into(),
                        base2k: base2k.into(),
                        k: k_out.into(),
                        dnum: dnum_apply.into(),
                        dsize: dsize_in.into(),
                        rank_in: rank_in_s0s1.into(),
                        rank_out: rank_out_s1s2.into(),
                    };

                    let mut gglwe_s0s1: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc(&gglwe_s0s1_infos);
                    let mut gglwe_s1s2: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc(&gglwe_s1s2_infos);
                    let mut gglwe_s0s2: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc(&gglwe_s0s2_infos);

                    let mut source_xs: Source = Source::new([0u8; 32]);
                    let mut source_xe: Source = Source::new([0u8; 32]);
                    let mut source_xa: Source = Source::new([0u8; 32]);

                    let mut scratch_enc: ScratchOwned<B> = ScratchOwned::alloc(
                        GLWESwitchingKey::encrypt_sk_scratch_space(module, &gglwe_s0s1_infos)
                            | GLWESwitchingKey::encrypt_sk_scratch_space(module, &gglwe_s1s2_infos)
                            | GLWESwitchingKey::encrypt_sk_scratch_space(module, &gglwe_s0s2_infos),
                    );
                    let mut scratch_apply: ScratchOwned<B> = ScratchOwned::alloc(GLWESwitchingKey::keyswitch_scratch_space(
                        module,
                        &gglwe_s0s1_infos,
                        &gglwe_s0s2_infos,
                        &gglwe_s1s2_infos,
                    ));

                    let mut sk0: GLWESecret<Vec<u8>> = GLWESecret::alloc_with(n.into(), rank_in_s0s1.into());
                    sk0.fill_ternary_prob(0.5, &mut source_xs);

                    let mut sk1: GLWESecret<Vec<u8>> = GLWESecret::alloc_with(n.into(), rank_out_s0s1.into());
                    sk1.fill_ternary_prob(0.5, &mut source_xs);

                    let mut sk2: GLWESecret<Vec<u8>> = GLWESecret::alloc_with(n.into(), rank_out_s1s2.into());
                    sk2.fill_ternary_prob(0.5, &mut source_xs);
                    let sk2_prepared: GLWESecretPrepared<Vec<u8>, B> = sk2.prepare_alloc(module, scratch_apply.borrow());

                    // gglwe_{s1}(s0) = s0 -> s1
                    gglwe_s0s1.encrypt_sk(
                        module,
                        &sk0,
                        &sk1,
                        &mut source_xa,
                        &mut source_xe,
                        scratch_enc.borrow(),
                    );

                    // gglwe_{s2}(s1) -> s1 -> s2
                    gglwe_s1s2.encrypt_sk(
                        module,
                        &sk1,
                        &sk2,
                        &mut source_xa,
                        &mut source_xe,
                        scratch_enc.borrow(),
                    );

                    let gglwe_s1s2_prepared: GLWESwitchingKeyPrepared<Vec<u8>, B> =
                        gglwe_s1s2.prepare_alloc(module, scratch_apply.borrow());

                    // gglwe_{s1}(s0) (x) gglwe_{s2}(s1) = gglwe_{s2}(s0)
                    gglwe_s0s2.keyswitch(
                        module,
                        &gglwe_s0s1,
                        &gglwe_s1s2_prepared,
                        scratch_apply.borrow(),
                    );

                    let max_noise: f64 = log2_std_noise_gglwe_product(
                        n as f64,
                        base2k * di,
                        0.5,
                        0.5,
                        0f64,
                        SIGMA * SIGMA,
                        0f64,
                        rank_out_s0s1 as f64,
                        k_in,
                        k_ksk,
                    );

                    gglwe_s0s2
                        .key
                        .assert_noise(module, &sk2_prepared, &sk0.data, max_noise + 0.5);
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn test_gglwe_switching_key_keyswitch_inplace<B>(module: &Module<B>)
where
    Module<B>: VecZnxDftAllocBytes
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
        + SvpPPolAllocBytes
        + SvpPPolAlloc<B>
        + VecZnxBigAllocBytes
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
        + VecZnxSwitchRing
        + VecZnxSubScalarInplace,
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
    let k_out: usize = 60;
    let dsize: usize = k_out.div_ceil(base2k);
    for rank_in in 1_usize..3 {
        for rank_out in 1_usize..3 {
            for di in 1_usize..dsize + 1 {
                let k_ksk: usize = k_out + base2k * di;

                let n: usize = module.n();
                let dnum: usize = k_out.div_ceil(base2k * di);
                let dsize_in: usize = 1;

                let gglwe_s0s1_infos: GLWESwitchingKeyLayout = GLWESwitchingKeyLayout {
                    n: n.into(),
                    base2k: base2k.into(),
                    k: k_out.into(),
                    dnum: dnum.into(),
                    dsize: dsize_in.into(),
                    rank_in: rank_in.into(),
                    rank_out: rank_out.into(),
                };

                let gglwe_s1s2_infos: GLWESwitchingKeyLayout = GLWESwitchingKeyLayout {
                    n: n.into(),
                    base2k: base2k.into(),
                    k: k_ksk.into(),
                    dnum: dnum.into(),
                    dsize: di.into(),
                    rank_in: rank_out.into(),
                    rank_out: rank_out.into(),
                };

                let mut gglwe_s0s1: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc(&gglwe_s0s1_infos);
                let mut gglwe_s1s2: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc(&gglwe_s1s2_infos);

                let mut source_xs: Source = Source::new([0u8; 32]);
                let mut source_xe: Source = Source::new([0u8; 32]);
                let mut source_xa: Source = Source::new([0u8; 32]);

                let mut scratch_enc: ScratchOwned<B> = ScratchOwned::alloc(
                    GLWESwitchingKey::encrypt_sk_scratch_space(module, &gglwe_s0s1_infos)
                        | GLWESwitchingKey::encrypt_sk_scratch_space(module, &gglwe_s1s2_infos),
                );
                let mut scratch_apply: ScratchOwned<B> = ScratchOwned::alloc(GLWESwitchingKey::keyswitch_inplace_scratch_space(
                    module,
                    &gglwe_s0s1_infos,
                    &gglwe_s1s2_infos,
                ));

                let var_xs: f64 = 0.5;

                let mut sk0: GLWESecret<Vec<u8>> = GLWESecret::alloc_with(n.into(), rank_in.into());
                sk0.fill_ternary_prob(var_xs, &mut source_xs);

                let mut sk1: GLWESecret<Vec<u8>> = GLWESecret::alloc_with(n.into(), rank_out.into());
                sk1.fill_ternary_prob(var_xs, &mut source_xs);

                let mut sk2: GLWESecret<Vec<u8>> = GLWESecret::alloc_with(n.into(), rank_out.into());
                sk2.fill_ternary_prob(var_xs, &mut source_xs);
                let sk2_prepared: GLWESecretPrepared<Vec<u8>, B> = sk2.prepare_alloc(module, scratch_apply.borrow());

                // gglwe_{s1}(s0) = s0 -> s1
                gglwe_s0s1.encrypt_sk(
                    module,
                    &sk0,
                    &sk1,
                    &mut source_xa,
                    &mut source_xe,
                    scratch_enc.borrow(),
                );

                // gglwe_{s2}(s1) -> s1 -> s2
                gglwe_s1s2.encrypt_sk(
                    module,
                    &sk1,
                    &sk2,
                    &mut source_xa,
                    &mut source_xe,
                    scratch_enc.borrow(),
                );

                let gglwe_s1s2_prepared: GLWESwitchingKeyPrepared<Vec<u8>, B> =
                    gglwe_s1s2.prepare_alloc(module, scratch_apply.borrow());

                // gglwe_{s1}(s0) (x) gglwe_{s2}(s1) = gglwe_{s2}(s0)
                gglwe_s0s1.keyswitch_inplace(module, &gglwe_s1s2_prepared, scratch_apply.borrow());

                let gglwe_s0s2: GLWESwitchingKey<Vec<u8>> = gglwe_s0s1;

                let max_noise: f64 = log2_std_noise_gglwe_product(
                    n as f64,
                    base2k * di,
                    var_xs,
                    var_xs,
                    0f64,
                    SIGMA * SIGMA,
                    0f64,
                    rank_out as f64,
                    k_out,
                    k_ksk,
                );

                gglwe_s0s2
                    .key
                    .assert_noise(module, &sk2_prepared, &sk0.data, max_noise + 0.5);
            }
        }
    }
}
