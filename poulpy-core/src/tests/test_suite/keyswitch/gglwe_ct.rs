use poulpy_hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyDftToDftInplace, SvpPPolAlloc, SvpPPolAllocBytes, SvpPrepare,
        VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxBigAddInplace, VecZnxBigAddSmallInplace,
        VecZnxBigAllocBytes, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftAllocBytes, VecZnxDftApply,
        VecZnxFillUniform, VecZnxIdftApplyConsume, VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxSub,
        VecZnxSubABInplace, VecZnxSubScalarInplace, VecZnxSwitchRing, VmpApplyDftToDft, VmpApplyDftToDftAdd,
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
        GGLWESwitchingKey, GLWESecret,
        prepared::{GGLWESwitchingKeyPrepared, GLWESecretPrepared, PrepareAlloc},
    },
    noise::log2_std_noise_gglwe_product,
};

#[allow(clippy::too_many_arguments)]
pub fn test_gglwe_switching_key_keyswitch<B>(module: &Module<B>)
where
    Module<B>: VecZnxDftAllocBytes
        + VecZnxBigNormalize<B>
        + VecZnxDftApply<B>
        + SvpApplyDftToDftInplace<B>
        + VecZnxIdftApplyConsume<B>
        + VecZnxFillUniform
        + VecZnxSubABInplace
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
    let basek: usize = 12;
    let k_in: usize = 60;
    let digits: usize = k_in.div_ceil(basek);

    (1..3).for_each(|rank_in_s0s1| {
        (1..3).for_each(|rank_out_s0s1| {
            (1..3).for_each(|rank_out_s1s2| {
                (1..digits + 1).for_each(|di| {
                    let k_ksk: usize = k_in + basek * di;
                    let k_out: usize = k_ksk; // Better capture noise.

                    let n: usize = module.n();
                    let rows: usize = k_in / basek;
                    let rows_apply: usize = k_in.div_ceil(basek * di);
                    let digits_in: usize = 1;

                    let mut ct_gglwe_s0s1: GGLWESwitchingKey<Vec<u8>> =
                        GGLWESwitchingKey::alloc(n, basek, k_in, rows, digits_in, rank_in_s0s1, rank_out_s0s1);
                    let mut ct_gglwe_s1s2: GGLWESwitchingKey<Vec<u8>> = GGLWESwitchingKey::alloc(
                        n,
                        basek,
                        k_ksk,
                        rows_apply,
                        di,
                        rank_out_s0s1,
                        rank_out_s1s2,
                    );
                    let mut ct_gglwe_s0s2: GGLWESwitchingKey<Vec<u8>> = GGLWESwitchingKey::alloc(
                        n,
                        basek,
                        k_out,
                        rows,
                        digits_in,
                        rank_in_s0s1,
                        rank_out_s1s2,
                    );

                    let mut source_xs: Source = Source::new([0u8; 32]);
                    let mut source_xe: Source = Source::new([0u8; 32]);
                    let mut source_xa: Source = Source::new([0u8; 32]);

                    let mut scratch_enc: ScratchOwned<B> = ScratchOwned::alloc(GGLWESwitchingKey::encrypt_sk_scratch_space(
                        module,
                        basek,
                        k_ksk,
                        rank_in_s0s1 | rank_out_s0s1,
                        rank_out_s0s1 | rank_out_s1s2,
                    ));
                    let mut scratch_apply: ScratchOwned<B> = ScratchOwned::alloc(GGLWESwitchingKey::keyswitch_scratch_space(
                        module,
                        basek,
                        k_out,
                        k_in,
                        k_ksk,
                        di,
                        ct_gglwe_s1s2.rank_in(),
                        ct_gglwe_s1s2.rank_out(),
                    ));

                    let mut sk0: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank_in_s0s1);
                    sk0.fill_ternary_prob(0.5, &mut source_xs);

                    let mut sk1: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank_out_s0s1);
                    sk1.fill_ternary_prob(0.5, &mut source_xs);

                    let mut sk2: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank_out_s1s2);
                    sk2.fill_ternary_prob(0.5, &mut source_xs);
                    let sk2_prepared: GLWESecretPrepared<Vec<u8>, B> = sk2.prepare_alloc(module, scratch_apply.borrow());

                    // gglwe_{s1}(s0) = s0 -> s1
                    ct_gglwe_s0s1.encrypt_sk(
                        module,
                        &sk0,
                        &sk1,
                        &mut source_xa,
                        &mut source_xe,
                        scratch_enc.borrow(),
                    );

                    // gglwe_{s2}(s1) -> s1 -> s2
                    ct_gglwe_s1s2.encrypt_sk(
                        module,
                        &sk1,
                        &sk2,
                        &mut source_xa,
                        &mut source_xe,
                        scratch_enc.borrow(),
                    );

                    let ct_gglwe_s1s2_prepared: GGLWESwitchingKeyPrepared<Vec<u8>, B> =
                        ct_gglwe_s1s2.prepare_alloc(module, scratch_apply.borrow());

                    // gglwe_{s1}(s0) (x) gglwe_{s2}(s1) = gglwe_{s2}(s0)
                    ct_gglwe_s0s2.keyswitch(
                        module,
                        &ct_gglwe_s0s1,
                        &ct_gglwe_s1s2_prepared,
                        scratch_apply.borrow(),
                    );

                    let max_noise: f64 = log2_std_noise_gglwe_product(
                        n as f64,
                        basek * di,
                        0.5,
                        0.5,
                        0f64,
                        SIGMA * SIGMA,
                        0f64,
                        rank_out_s0s1 as f64,
                        k_in,
                        k_ksk,
                    );

                    ct_gglwe_s0s2
                        .key
                        .assert_noise(module, &sk2_prepared, &sk0.data, max_noise + 0.5);
                });
            });
        });
    });
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
        + VecZnxSubABInplace
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
    let basek: usize = 12;
    let k_ct: usize = 60;
    let digits: usize = k_ct.div_ceil(basek);
    (1..3).for_each(|rank_in| {
        (1..3).for_each(|rank_out| {
            (1..digits + 1).for_each(|di| {
                let k_ksk: usize = k_ct + basek * di;

                let n: usize = module.n();
                let rows: usize = k_ct.div_ceil(basek * di);
                let digits_in: usize = 1;

                let mut ct_gglwe_s0s1: GGLWESwitchingKey<Vec<u8>> =
                    GGLWESwitchingKey::alloc(n, basek, k_ct, rows, digits_in, rank_in, rank_out);
                let mut ct_gglwe_s1s2: GGLWESwitchingKey<Vec<u8>> =
                    GGLWESwitchingKey::alloc(n, basek, k_ksk, rows, di, rank_out, rank_out);

                let mut source_xs: Source = Source::new([0u8; 32]);
                let mut source_xe: Source = Source::new([0u8; 32]);
                let mut source_xa: Source = Source::new([0u8; 32]);

                let mut scratch_enc: ScratchOwned<B> = ScratchOwned::alloc(GGLWESwitchingKey::encrypt_sk_scratch_space(
                    module,
                    basek,
                    k_ksk,
                    rank_in | rank_out,
                    rank_out,
                ));
                let mut scratch_apply: ScratchOwned<B> = ScratchOwned::alloc(GGLWESwitchingKey::keyswitch_inplace_scratch_space(
                    module, basek, k_ct, k_ksk, di, rank_out,
                ));

                let var_xs: f64 = 0.5;

                let mut sk0: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank_in);
                sk0.fill_ternary_prob(var_xs, &mut source_xs);

                let mut sk1: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank_out);
                sk1.fill_ternary_prob(var_xs, &mut source_xs);

                let mut sk2: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank_out);
                sk2.fill_ternary_prob(var_xs, &mut source_xs);
                let sk2_prepared: GLWESecretPrepared<Vec<u8>, B> = sk2.prepare_alloc(module, scratch_apply.borrow());

                // gglwe_{s1}(s0) = s0 -> s1
                ct_gglwe_s0s1.encrypt_sk(
                    module,
                    &sk0,
                    &sk1,
                    &mut source_xa,
                    &mut source_xe,
                    scratch_enc.borrow(),
                );

                // gglwe_{s2}(s1) -> s1 -> s2
                ct_gglwe_s1s2.encrypt_sk(
                    module,
                    &sk1,
                    &sk2,
                    &mut source_xa,
                    &mut source_xe,
                    scratch_enc.borrow(),
                );

                let ct_gglwe_s1s2_prepared: GGLWESwitchingKeyPrepared<Vec<u8>, B> =
                    ct_gglwe_s1s2.prepare_alloc(module, scratch_apply.borrow());

                // gglwe_{s1}(s0) (x) gglwe_{s2}(s1) = gglwe_{s2}(s0)
                ct_gglwe_s0s1.keyswitch_inplace(module, &ct_gglwe_s1s2_prepared, scratch_apply.borrow());

                let ct_gglwe_s0s2: GGLWESwitchingKey<Vec<u8>> = ct_gglwe_s0s1;

                let max_noise: f64 = log2_std_noise_gglwe_product(
                    n as f64,
                    basek * di,
                    var_xs,
                    var_xs,
                    0f64,
                    SIGMA * SIGMA,
                    0f64,
                    rank_out as f64,
                    k_ct,
                    k_ksk,
                );

                ct_gglwe_s0s2
                    .key
                    .assert_noise(module, &sk2_prepared, &sk0.data, max_noise + 0.5);
            });
        });
    });
}
