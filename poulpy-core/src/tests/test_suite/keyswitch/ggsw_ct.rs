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
        GGLWESwitchingKey, GGLWETensorKey, GGSWCiphertext, GLWESecret,
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
        + VecZnxSwitchRing<B>
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
    let basek: usize = 12;
    let k_in: usize = 54;
    let digits: usize = k_in.div_ceil(basek);
    (1..4).for_each(|rank| {
        (1..digits + 1).for_each(|di| {
            let k_ksk: usize = k_in + basek * di;
            let k_tsk: usize = k_ksk;
            let k_out: usize = k_ksk; // Better capture noise.

            let n: usize = module.n();
            let rows: usize = k_in.div_ceil(di * basek);

            let digits_in: usize = 1;

            let mut ct_in: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(n, basek, k_in, rows, digits_in, rank);
            let mut ct_out: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(n, basek, k_out, rows, digits_in, rank);
            let mut tsk: GGLWETensorKey<Vec<u8>> = GGLWETensorKey::alloc(n, basek, k_ksk, rows, di, rank);
            let mut ksk: GGLWESwitchingKey<Vec<u8>> = GGLWESwitchingKey::alloc(n, basek, k_ksk, rows, di, rank, rank);
            let mut pt_scalar: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
                GGSWCiphertext::encrypt_sk_scratch_space(module, basek, k_in, rank)
                    | GGLWESwitchingKey::encrypt_sk_scratch_space(module, basek, k_ksk, rank, rank)
                    | GGLWETensorKey::encrypt_sk_scratch_space(module, basek, k_tsk, rank)
                    | GGSWCiphertext::keyswitch_scratch_space(module, basek, k_out, k_in, k_ksk, di, k_tsk, di, rank),
            );

            let var_xs: f64 = 0.5;

            let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
            sk_in.fill_ternary_prob(var_xs, &mut source_xs);
            let sk_in_dft: GLWESecretPrepared<Vec<u8>, B> = sk_in.prepare_alloc(module, scratch.borrow());

            let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
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

            ct_in.encrypt_sk(
                module,
                &pt_scalar,
                &sk_in_dft,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            let ksk_prepared: GGLWESwitchingKeyPrepared<Vec<u8>, B> = ksk.prepare_alloc(module, scratch.borrow());
            let tsk_prepared: GGLWETensorKeyPrepared<Vec<u8>, B> = tsk.prepare_alloc(module, scratch.borrow());

            ct_out.keyswitch(
                module,
                &ct_in,
                &ksk_prepared,
                &tsk_prepared,
                scratch.borrow(),
            );

            let max_noise = |col_j: usize| -> f64 {
                noise_ggsw_keyswitch(
                    n as f64,
                    basek * di,
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

            ct_out.assert_noise(module, &sk_out_prepared, &pt_scalar, max_noise);
        });
    });
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
        + VecZnxSwitchRing<B>
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
    let basek: usize = 12;
    let k_ct: usize = 54;
    let digits: usize = k_ct.div_ceil(basek);
    (1..4).for_each(|rank| {
        (1..digits + 1).for_each(|di| {
            let k_ksk: usize = k_ct + basek * di;
            let k_tsk: usize = k_ksk;

            let n: usize = module.n();
            let rows: usize = k_ct.div_ceil(di * basek);

            let digits_in: usize = 1;

            let mut ct: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(n, basek, k_ct, rows, digits_in, rank);
            let mut tsk: GGLWETensorKey<Vec<u8>> = GGLWETensorKey::alloc(n, basek, k_tsk, rows, di, rank);
            let mut ksk: GGLWESwitchingKey<Vec<u8>> = GGLWESwitchingKey::alloc(n, basek, k_ksk, rows, di, rank, rank);
            let mut pt_scalar: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
                GGSWCiphertext::encrypt_sk_scratch_space(module, basek, k_ct, rank)
                    | GGLWESwitchingKey::encrypt_sk_scratch_space(module, basek, k_ksk, rank, rank)
                    | GGLWETensorKey::encrypt_sk_scratch_space(module, basek, k_tsk, rank)
                    | GGSWCiphertext::keyswitch_inplace_scratch_space(module, basek, k_ct, k_ksk, di, k_tsk, di, rank),
            );

            let var_xs: f64 = 0.5;

            let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
            sk_in.fill_ternary_prob(var_xs, &mut source_xs);
            let sk_in_dft: GLWESecretPrepared<Vec<u8>, B> = sk_in.prepare_alloc(module, scratch.borrow());

            let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
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

            ct.encrypt_sk(
                module,
                &pt_scalar,
                &sk_in_dft,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            let ksk_prepared: GGLWESwitchingKeyPrepared<Vec<u8>, B> = ksk.prepare_alloc(module, scratch.borrow());
            let tsk_prepared: GGLWETensorKeyPrepared<Vec<u8>, B> = tsk.prepare_alloc(module, scratch.borrow());

            ct.keyswitch_inplace(module, &ksk_prepared, &tsk_prepared, scratch.borrow());

            let max_noise = |col_j: usize| -> f64 {
                noise_ggsw_keyswitch(
                    n as f64,
                    basek * di,
                    col_j,
                    var_xs,
                    0f64,
                    SIGMA * SIGMA,
                    0f64,
                    rank as f64,
                    k_ct,
                    k_ksk,
                    k_tsk,
                ) + 0.5
            };

            ct.assert_noise(module, &sk_out_prepared, &pt_scalar, max_noise);
        });
    });
}
