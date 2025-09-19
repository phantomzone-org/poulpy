use poulpy_hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyDftToDft, SvpApplyDftToDftInplace, SvpPPolAlloc, SvpPPolAllocBytes,
        SvpPrepare, VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxBigAddInplace, VecZnxBigAddSmallInplace,
        VecZnxBigAlloc, VecZnxBigAllocBytes, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftAddInplace, VecZnxDftAlloc,
        VecZnxDftAllocBytes, VecZnxDftApply, VecZnxDftCopy, VecZnxFillUniform, VecZnxIdftApplyConsume, VecZnxIdftApplyTmpA,
        VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxSub, VecZnxSubABInplace, VecZnxSwitchRing,
        VmpApplyDftToDft, VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes, VmpPMatAlloc, VmpPrepare,
    },
    layouts::{Backend, Module, ScalarZnx, ScratchOwned},
    oep::{
        ScratchAvailableImpl, ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeScalarZnxImpl, TakeSvpPPolImpl,
        TakeVecZnxBigImpl, TakeVecZnxDftImpl, TakeVecZnxImpl,
    },
    source::Source,
};

use crate::{
    encryption::SIGMA,
    layouts::{
        GGLWESwitchingKey, GGLWESwitchingKeyLayout, GGLWETensorKey, GGLWETensorKeyLayout, GGSWCiphertext, GGSWCiphertextLayout,
        GLWESecret,
        prepared::{GGLWESwitchingKeyPrepared, GGLWETensorKeyPrepared, GLWESecretPrepared, PrepareAlloc},
    },
    noise::noise_ggsw_keyswitch,
};

#[allow(clippy::too_many_arguments)]
pub fn test_ggsw_keyswitch<B>(module: &Module<B>)
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
        + SvpApplyDftToDft<B>
        + VecZnxIdftApplyTmpA<B>
        + VecZnxDftCopy<B>
        + VecZnxDftAddInplace<B>
        + VecZnxBigAlloc<B>
        + VecZnxDftAlloc<B>,
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
    let k_in: usize = 54;
    let digits: usize = k_in.div_ceil(base2k);
    for rank in 1_usize..3 {
        for di in 1..digits + 1 {
            let k_ksk: usize = k_in + base2k * di;
            let k_tsk: usize = k_ksk;
            let k_out: usize = k_ksk; // Better capture noise.

            let n: usize = module.n();
            let rows: usize = k_in.div_ceil(di * base2k);

            let digits_in: usize = 1;

            let ggsw_in_infos: GGSWCiphertextLayout = GGSWCiphertextLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_in.into(),
                rows: rows.into(),
                digits: digits_in.into(),
                rank: rank.into(),
            };

            let ggsw_out_infos: GGSWCiphertextLayout = GGSWCiphertextLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_out.into(),
                rows: rows.into(),
                digits: digits_in.into(),
                rank: rank.into(),
            };

            let tsk_infos: GGLWETensorKeyLayout = GGLWETensorKeyLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_tsk.into(),
                rows: rows.into(),
                digits: di.into(),
                rank: rank.into(),
            };

            let ksk_apply_infos: GGLWESwitchingKeyLayout = GGLWESwitchingKeyLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_ksk.into(),
                rows: rows.into(),
                digits: di.into(),
                rank_in: rank.into(),
                rank_out: rank.into(),
            };

            let mut ggsw_in: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(&ggsw_in_infos);
            let mut ggsw_out: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(&ggsw_out_infos);
            let mut tsk: GGLWETensorKey<Vec<u8>> = GGLWETensorKey::alloc(&tsk_infos);
            let mut ksk: GGLWESwitchingKey<Vec<u8>> = GGLWESwitchingKey::alloc(&ksk_apply_infos);
            let mut pt_scalar: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
                GGSWCiphertext::encrypt_sk_scratch_space(module, &ggsw_in_infos)
                    | GGLWESwitchingKey::encrypt_sk_scratch_space(module, &ksk_apply_infos)
                    | GGLWETensorKey::encrypt_sk_scratch_space(module, &tsk_infos)
                    | GGSWCiphertext::keyswitch_scratch_space(
                        module,
                        &ggsw_out_infos,
                        &ggsw_in_infos,
                        &ksk_apply_infos,
                        &tsk_infos,
                    ),
            );

            let var_xs: f64 = 0.5;

            let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc_with(n.into(), rank.into());
            sk_in.fill_ternary_prob(var_xs, &mut source_xs);
            let sk_in_dft: GLWESecretPrepared<Vec<u8>, B> = sk_in.prepare_alloc(module, scratch.borrow());

            let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc_with(n.into(), rank.into());
            sk_out.fill_ternary_prob(var_xs, &mut source_xs);
            let sk_out_prepared: GLWESecretPrepared<Vec<u8>, B> = sk_out.prepare_alloc(module, scratch.borrow());

            ksk.encrypt_sk(
                module,
                &sk_in,
                &sk_out,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );
            tsk.encrypt_sk(
                module,
                &sk_out,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            pt_scalar.fill_ternary_hw(0, n, &mut source_xs);

            ggsw_in.encrypt_sk(
                module,
                &pt_scalar,
                &sk_in_dft,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            let ksk_prepared: GGLWESwitchingKeyPrepared<Vec<u8>, B> = ksk.prepare_alloc(module, scratch.borrow());
            let tsk_prepared: GGLWETensorKeyPrepared<Vec<u8>, B> = tsk.prepare_alloc(module, scratch.borrow());

            ggsw_out.keyswitch(
                module,
                &ggsw_in,
                &ksk_prepared,
                &tsk_prepared,
                scratch.borrow(),
            );

            let max_noise = |col_j: usize| -> f64 {
                noise_ggsw_keyswitch(
                    n as f64,
                    base2k * di,
                    col_j,
                    var_xs,
                    0f64,
                    SIGMA * SIGMA,
                    0f64,
                    rank as f64,
                    k_in,
                    k_ksk,
                    k_tsk,
                ) + 0.5
            };

            ggsw_out.assert_noise(module, &sk_out_prepared, &pt_scalar, max_noise);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn test_ggsw_keyswitch_inplace<B>(module: &Module<B>)
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
        + SvpApplyDftToDft<B>
        + VecZnxIdftApplyTmpA<B>
        + VecZnxDftCopy<B>
        + VecZnxDftAddInplace<B>
        + VecZnxBigAlloc<B>
        + VecZnxDftAlloc<B>,
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
    let k_out: usize = 54;
    let digits: usize = k_out.div_ceil(base2k);
    for rank in 1_usize..3 {
        for di in 1..digits + 1 {
            let k_ksk: usize = k_out + base2k * di;
            let k_tsk: usize = k_ksk;

            let n: usize = module.n();
            let rows: usize = k_out.div_ceil(di * base2k);

            let digits_in: usize = 1;

            let ggsw_out_infos: GGSWCiphertextLayout = GGSWCiphertextLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_out.into(),
                rows: rows.into(),
                digits: digits_in.into(),
                rank: rank.into(),
            };

            let tsk_infos: GGLWETensorKeyLayout = GGLWETensorKeyLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_tsk.into(),
                rows: rows.into(),
                digits: di.into(),
                rank: rank.into(),
            };

            let ksk_apply_infos: GGLWESwitchingKeyLayout = GGLWESwitchingKeyLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_ksk.into(),
                rows: rows.into(),
                digits: di.into(),
                rank_in: rank.into(),
                rank_out: rank.into(),
            };

            let mut ggsw_out: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(&ggsw_out_infos);
            let mut tsk: GGLWETensorKey<Vec<u8>> = GGLWETensorKey::alloc(&tsk_infos);
            let mut ksk: GGLWESwitchingKey<Vec<u8>> = GGLWESwitchingKey::alloc(&ksk_apply_infos);
            let mut pt_scalar: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
                GGSWCiphertext::encrypt_sk_scratch_space(module, &ggsw_out_infos)
                    | GGLWESwitchingKey::encrypt_sk_scratch_space(module, &ksk_apply_infos)
                    | GGLWETensorKey::encrypt_sk_scratch_space(module, &tsk_infos)
                    | GGSWCiphertext::keyswitch_inplace_scratch_space(module, &ggsw_out_infos, &ksk_apply_infos, &tsk_infos),
            );

            let var_xs: f64 = 0.5;

            let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc_with(n.into(), rank.into());
            sk_in.fill_ternary_prob(var_xs, &mut source_xs);
            let sk_in_dft: GLWESecretPrepared<Vec<u8>, B> = sk_in.prepare_alloc(module, scratch.borrow());

            let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc_with(n.into(), rank.into());
            sk_out.fill_ternary_prob(var_xs, &mut source_xs);
            let sk_out_prepared: GLWESecretPrepared<Vec<u8>, B> = sk_out.prepare_alloc(module, scratch.borrow());

            ksk.encrypt_sk(
                module,
                &sk_in,
                &sk_out,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );
            tsk.encrypt_sk(
                module,
                &sk_out,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            pt_scalar.fill_ternary_hw(0, n, &mut source_xs);

            ggsw_out.encrypt_sk(
                module,
                &pt_scalar,
                &sk_in_dft,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            let ksk_prepared: GGLWESwitchingKeyPrepared<Vec<u8>, B> = ksk.prepare_alloc(module, scratch.borrow());
            let tsk_prepared: GGLWETensorKeyPrepared<Vec<u8>, B> = tsk.prepare_alloc(module, scratch.borrow());

            ggsw_out.keyswitch_inplace(module, &ksk_prepared, &tsk_prepared, scratch.borrow());

            let max_noise = |col_j: usize| -> f64 {
                noise_ggsw_keyswitch(
                    n as f64,
                    base2k * di,
                    col_j,
                    var_xs,
                    0f64,
                    SIGMA * SIGMA,
                    0f64,
                    rank as f64,
                    k_out,
                    k_ksk,
                    k_tsk,
                ) + 0.5
            };

            ggsw_out.assert_noise(module, &sk_out_prepared, &pt_scalar, max_noise);
        }
    }
}
