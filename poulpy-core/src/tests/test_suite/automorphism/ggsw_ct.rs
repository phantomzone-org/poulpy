use poulpy_hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyDftToDft, SvpApplyDftToDftInplace, SvpPPolAlloc, SvpPPolAllocBytes,
        SvpPrepare, VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxAutomorphism, VecZnxAutomorphismInplace,
        VecZnxBigAddInplace, VecZnxBigAddSmallInplace, VecZnxBigAlloc, VecZnxBigAllocBytes, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxCopy, VecZnxDftAddInplace, VecZnxDftAlloc, VecZnxDftAllocBytes, VecZnxDftApply,
        VecZnxDftCopy, VecZnxFillUniform, VecZnxIdftApplyConsume, VecZnxIdftApplyTmpA, VecZnxNormalize, VecZnxNormalizeInplace,
        VecZnxNormalizeTmpBytes, VecZnxSub, VecZnxSubABInplace, VecZnxSwitchRing, VmpApplyDftToDft, VmpApplyDftToDftAdd,
        VmpApplyDftToDftTmpBytes, VmpPMatAlloc, VmpPrepare,
    },
    layouts::{Backend, Module, ScalarZnx, ScratchOwned},
    oep::{
        ScratchAvailableImpl, ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeScalarZnxImpl, TakeSvpPPolImpl,
        TakeVecZnxBigImpl, TakeVecZnxDftImpl, TakeVecZnxImpl, VecZnxBigAllocBytesImpl, VecZnxDftAllocBytesImpl,
    },
    source::Source,
};

use crate::{
    encryption::SIGMA,
    layouts::{
        GGLWEAutomorphismKey, GGLWETensorKey, GGLWETensorKeyLayout, GGSWCiphertext, GGSWCiphertextLayout, GLWESecret,
        prepared::{GGLWEAutomorphismKeyPrepared, GGLWETensorKeyPrepared, GLWESecretPrepared, Prepare, PrepareAlloc},
    },
    noise::noise_ggsw_keyswitch,
};

pub fn test_ggsw_automorphism<B>(module: &Module<B>)
where
    Module<B>: VecZnxDftAllocBytes
        + VecZnxBigAllocBytes
        + VecZnxDftApply<B>
        + SvpApplyDftToDftInplace<B>
        + VecZnxIdftApplyConsume<B>
        + VecZnxBigAddInplace<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxBigNormalize<B>
        + VecZnxNormalizeTmpBytes
        + VecZnxBigAlloc<B>
        + VecZnxDftAlloc<B>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxIdftApplyTmpA<B>
        + SvpPrepare<B>
        + SvpPPolAllocBytes
        + SvpPPolAlloc<B>
        + VecZnxAddScalarInplace
        + VecZnxCopy
        + VecZnxSubABInplace
        + VmpPMatAlloc<B>
        + VmpPrepare<B>
        + VmpApplyDftToDftTmpBytes
        + VmpApplyDftToDft<B>
        + VmpApplyDftToDftAdd<B>
        + VecZnxDftCopy<B>
        + VecZnxDftAddInplace<B>
        + VecZnxFillUniform
        + VecZnxAddInplace
        + VecZnxNormalizeInplace<B>
        + VecZnxAddNormal
        + VecZnxNormalize<B>
        + VecZnxSub
        + SvpApplyDftToDft<B>
        + VecZnxSwitchRing
        + VecZnxAutomorphismInplace<B>
        + VecZnxAutomorphism,
    B: Backend
        + TakeVecZnxDftImpl<B>
        + TakeVecZnxBigImpl<B>
        + ScratchOwnedAllocImpl<B>
        + ScratchOwnedBorrowImpl<B>
        + ScratchAvailableImpl<B>
        + TakeScalarZnxImpl<B>
        + TakeVecZnxImpl<B>
        + VecZnxDftAllocBytesImpl<B>
        + VecZnxBigAllocBytesImpl<B>
        + TakeSvpPPolImpl<B>,
{
    let base2k: usize = 12;
    let k_in: usize = 54;
    let digits: usize = k_in.div_ceil(base2k);
    let p: i64 = -5;

    for rank in 1_usize..3 {
        for di in 1..digits + 1 {
            let k_ksk: usize = k_in + base2k * di;
            let k_tsk: usize = k_ksk;
            let k_out: usize = k_ksk; // Better capture noise.

            let n: usize = module.n();
            let rows: usize = k_in.div_ceil(base2k * di);
            let rows_in: usize = k_in.div_euclid(base2k * di);

            let digits_in: usize = 1;

            let ggsw_in_layout: GGSWCiphertextLayout = GGSWCiphertextLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_in.into(),
                rows: rows_in.into(),
                digits: digits_in.into(),
                rank: rank.into(),
            };

            let ggsw_out_layout: GGSWCiphertextLayout = GGSWCiphertextLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_out.into(),
                rows: rows_in.into(),
                digits: digits_in.into(),
                rank: rank.into(),
            };

            let tensor_key_layout: GGLWETensorKeyLayout = GGLWETensorKeyLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_tsk.into(),
                rows: rows.into(),
                digits: di.into(),
                rank: rank.into(),
            };

            let auto_key_layout: GGLWETensorKeyLayout = GGLWETensorKeyLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_ksk.into(),
                rows: rows.into(),
                digits: di.into(),
                rank: rank.into(),
            };

            let mut ct_in: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(&ggsw_in_layout);
            let mut ct_out: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(&ggsw_out_layout);
            let mut tensor_key: GGLWETensorKey<Vec<u8>> = GGLWETensorKey::alloc(&tensor_key_layout);
            let mut auto_key: GGLWEAutomorphismKey<Vec<u8>> = GGLWEAutomorphismKey::alloc(&auto_key_layout);
            let mut pt_scalar: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
                GGSWCiphertext::encrypt_sk_scratch_space(module, &ct_in)
                    | GGLWEAutomorphismKey::encrypt_sk_scratch_space(module, &auto_key)
                    | GGLWETensorKey::encrypt_sk_scratch_space(module, &tensor_key)
                    | GGSWCiphertext::automorphism_scratch_space(module, &ct_out, &ct_in, &auto_key, &tensor_key),
            );

            let var_xs: f64 = 0.5;

            let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(&ct_out);
            sk.fill_ternary_prob(var_xs, &mut source_xs);
            let sk_prepared: GLWESecretPrepared<Vec<u8>, B> = sk.prepare_alloc(module, scratch.borrow());

            auto_key.encrypt_sk(
                module,
                p,
                &sk,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );
            tensor_key.encrypt_sk(
                module,
                &sk,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            pt_scalar.fill_ternary_hw(0, n, &mut source_xs);

            ct_in.encrypt_sk(
                module,
                &pt_scalar,
                &sk_prepared,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            let mut auto_key_prepared: GGLWEAutomorphismKeyPrepared<Vec<u8>, B> =
                GGLWEAutomorphismKeyPrepared::alloc(module, &auto_key_layout);
            auto_key_prepared.prepare(module, &auto_key, scratch.borrow());

            let mut tsk_prepared: GGLWETensorKeyPrepared<Vec<u8>, B> = GGLWETensorKeyPrepared::alloc(module, &tensor_key_layout);
            tsk_prepared.prepare(module, &tensor_key, scratch.borrow());

            ct_out.automorphism(
                module,
                &ct_in,
                &auto_key_prepared,
                &tsk_prepared,
                scratch.borrow(),
            );

            module.vec_znx_automorphism_inplace(p, &mut pt_scalar.as_vec_znx_mut(), 0, scratch.borrow());

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

            ct_out.assert_noise(module, &sk_prepared, &pt_scalar, max_noise);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn test_ggsw_automorphism_inplace<B>(module: &Module<B>)
where
    Module<B>: VecZnxDftAllocBytes
        + VecZnxBigAllocBytes
        + VecZnxDftApply<B>
        + SvpApplyDftToDftInplace<B>
        + VecZnxIdftApplyConsume<B>
        + VecZnxBigAddInplace<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxBigNormalize<B>
        + VecZnxNormalizeTmpBytes
        + VecZnxBigAlloc<B>
        + VecZnxDftAlloc<B>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxIdftApplyTmpA<B>
        + SvpPrepare<B>
        + SvpPPolAllocBytes
        + SvpPPolAlloc<B>
        + VecZnxAddScalarInplace
        + VecZnxCopy
        + VecZnxSubABInplace
        + VmpPMatAlloc<B>
        + VmpPrepare<B>
        + VmpApplyDftToDftTmpBytes
        + VmpApplyDftToDft<B>
        + VmpApplyDftToDftAdd<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxDftCopy<B>
        + VecZnxDftAddInplace<B>
        + VecZnxAddInplace
        + VecZnxNormalizeInplace<B>
        + VecZnxAddNormal
        + VecZnxNormalize<B>
        + VecZnxSub
        + VecZnxFillUniform
        + SvpApplyDftToDft<B>
        + VecZnxSwitchRing
        + VecZnxAutomorphismInplace<B>
        + VecZnxAutomorphism,
    B: Backend
        + TakeVecZnxDftImpl<B>
        + TakeVecZnxBigImpl<B>
        + ScratchOwnedAllocImpl<B>
        + ScratchOwnedBorrowImpl<B>
        + ScratchAvailableImpl<B>
        + TakeScalarZnxImpl<B>
        + TakeVecZnxImpl<B>
        + VecZnxDftAllocBytesImpl<B>
        + VecZnxBigAllocBytesImpl<B>
        + TakeSvpPPolImpl<B>,
{
    let base2k: usize = 12;
    let k_out: usize = 54;
    let digits: usize = k_out.div_ceil(base2k);
    let p: i64 = -1;
    for rank in 1_usize..3 {
        for di in 1..digits + 1 {
            let k_ksk: usize = k_out + base2k * di;
            let k_tsk: usize = k_ksk;

            let n: usize = module.n();
            let rows: usize = k_out.div_ceil(di * base2k);
            let rows_in: usize = k_out.div_euclid(base2k * di);
            let digits_in: usize = 1;

            let ggsw_out_layout: GGSWCiphertextLayout = GGSWCiphertextLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_out.into(),
                rows: rows_in.into(),
                digits: digits_in.into(),
                rank: rank.into(),
            };

            let tensor_key_layout: GGLWETensorKeyLayout = GGLWETensorKeyLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_tsk.into(),
                rows: rows.into(),
                digits: di.into(),
                rank: rank.into(),
            };

            let auto_key_layout: GGLWETensorKeyLayout = GGLWETensorKeyLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_ksk.into(),
                rows: rows.into(),
                digits: di.into(),
                rank: rank.into(),
            };

            let mut ct: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(&ggsw_out_layout);
            let mut tensor_key: GGLWETensorKey<Vec<u8>> = GGLWETensorKey::alloc(&tensor_key_layout);
            let mut auto_key: GGLWEAutomorphismKey<Vec<u8>> = GGLWEAutomorphismKey::alloc(&auto_key_layout);
            let mut pt_scalar: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
                GGSWCiphertext::encrypt_sk_scratch_space(module, &ct)
                    | GGLWEAutomorphismKey::encrypt_sk_scratch_space(module, &auto_key)
                    | GGLWETensorKey::encrypt_sk_scratch_space(module, &tensor_key)
                    | GGSWCiphertext::automorphism_inplace_scratch_space(module, &ct, &auto_key, &tensor_key),
            );

            let var_xs: f64 = 0.5;

            let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(&ct);
            sk.fill_ternary_prob(var_xs, &mut source_xs);
            let sk_prepared: GLWESecretPrepared<Vec<u8>, B> = sk.prepare_alloc(module, scratch.borrow());

            auto_key.encrypt_sk(
                module,
                p,
                &sk,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );
            tensor_key.encrypt_sk(
                module,
                &sk,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            pt_scalar.fill_ternary_hw(0, n, &mut source_xs);

            ct.encrypt_sk(
                module,
                &pt_scalar,
                &sk_prepared,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            let mut auto_key_prepared: GGLWEAutomorphismKeyPrepared<Vec<u8>, B> =
                GGLWEAutomorphismKeyPrepared::alloc(module, &auto_key_layout);
            auto_key_prepared.prepare(module, &auto_key, scratch.borrow());

            let mut tsk_prepared: GGLWETensorKeyPrepared<Vec<u8>, B> = GGLWETensorKeyPrepared::alloc(module, &tensor_key_layout);
            tsk_prepared.prepare(module, &tensor_key, scratch.borrow());

            ct.automorphism_inplace(module, &auto_key_prepared, &tsk_prepared, scratch.borrow());

            module.vec_znx_automorphism_inplace(p, &mut pt_scalar.as_vec_znx_mut(), 0, scratch.borrow());

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

            ct.assert_noise(module, &sk_prepared, &pt_scalar, max_noise);
        }
    }
}
