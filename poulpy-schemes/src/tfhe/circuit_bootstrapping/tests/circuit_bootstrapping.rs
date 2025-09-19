use std::time::Instant;

use poulpy_hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyDftToDft, SvpApplyDftToDftInplace, SvpPPolAlloc, SvpPPolAllocBytes,
        SvpPrepare, VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxAutomorphism, VecZnxAutomorphismInplace,
        VecZnxBigAddInplace, VecZnxBigAddSmallInplace, VecZnxBigAlloc, VecZnxBigAllocBytes, VecZnxBigAutomorphismInplace,
        VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxBigSubSmallBInplace, VecZnxCopy, VecZnxDftAddInplace,
        VecZnxDftAlloc, VecZnxDftAllocBytes, VecZnxDftApply, VecZnxDftCopy, VecZnxFillUniform, VecZnxIdftApplyConsume,
        VecZnxIdftApplyTmpA, VecZnxNegateInplace, VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxRotate,
        VecZnxRotateInplace, VecZnxRotateInplaceTmpBytes, VecZnxRshInplace, VecZnxSub, VecZnxSubABInplace, VecZnxSwitchRing,
        VmpApplyDftToDft, VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes, VmpPMatAlloc, VmpPrepare, ZnAddNormal, ZnFillUniform,
        ZnNormalizeInplace,
    },
    layouts::{Backend, Module, ScalarZnx, ScratchOwned, ZnxView, ZnxViewMut},
    oep::{
        ScratchAvailableImpl, ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeMatZnxImpl, TakeScalarZnxImpl, TakeSvpPPolImpl,
        TakeVecZnxBigImpl, TakeVecZnxDftImpl, TakeVecZnxDftSliceImpl, TakeVecZnxImpl, TakeVecZnxSliceImpl,
    },
    source::Source,
};

use crate::tfhe::{
    blind_rotation::{
        BlincRotationExecute, BlindRotationAlgo, BlindRotationKey, BlindRotationKeyAlloc, BlindRotationKeyEncryptSk,
        BlindRotationKeyLayout, BlindRotationKeyPrepared,
    },
    circuit_bootstrapping::{
        CircuitBootstrappingKey, CircuitBootstrappingKeyEncryptSk, CircuitBootstrappingKeyLayout,
        CircuitBootstrappingKeyPrepared, CirtuitBootstrappingExecute,
    },
};

use poulpy_core::layouts::{
    Digits, GGLWEAutomorphismKeyLayout, GGLWETensorKeyLayout, GGSWCiphertextLayout, LWECiphertextLayout, prepared::PrepareAlloc,
};

use poulpy_core::layouts::{
    GGSWCiphertext, GLWECiphertext, GLWEPlaintext, GLWESecret, LWECiphertext, LWEPlaintext, LWESecret,
    prepared::{GGSWCiphertextPrepared, GLWESecretPrepared},
};

pub fn test_circuit_bootstrapping_to_exponent<B, BRA: BlindRotationAlgo>(module: &Module<B>)
where
    Module<B>: VecZnxFillUniform
        + VecZnxAddNormal
        + VecZnxNormalizeInplace<B>
        + VecZnxDftAllocBytes
        + VecZnxBigNormalize<B>
        + VecZnxDftApply<B>
        + SvpApplyDftToDftInplace<B>
        + VecZnxIdftApplyConsume<B>
        + VecZnxNormalizeTmpBytes
        + VecZnxSubABInplace
        + VecZnxAddInplace
        + VecZnxNormalize<B>
        + VecZnxSub
        + VecZnxAddScalarInplace
        + VecZnxAutomorphism
        + VecZnxSwitchRing
        + VecZnxBigAllocBytes
        + VecZnxIdftApplyTmpA<B>
        + SvpApplyDftToDft<B>
        + VecZnxBigAddInplace<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxBigAlloc<B>
        + VecZnxDftAlloc<B>
        + VecZnxBigNormalizeTmpBytes
        + VmpPMatAlloc<B>
        + VmpPrepare<B>
        + SvpPrepare<B>
        + SvpPPolAlloc<B>
        + VmpApplyDftToDftTmpBytes
        + VmpApplyDftToDft<B>
        + VmpApplyDftToDftAdd<B>
        + SvpPPolAllocBytes
        + VecZnxRotateInplace<B>
        + VecZnxBigAutomorphismInplace<B>
        + VecZnxRshInplace<B>
        + VecZnxDftCopy<B>
        + VecZnxNegateInplace
        + VecZnxCopy
        + VecZnxAutomorphismInplace<B>
        + VecZnxBigSubSmallBInplace<B>
        + VecZnxRotateInplaceTmpBytes
        + VecZnxBigAllocBytes
        + VecZnxDftAddInplace<B>
        + VecZnxRotate
        + ZnFillUniform
        + ZnAddNormal
        + ZnNormalizeInplace<B>,
    B: Backend
        + ScratchOwnedAllocImpl<B>
        + ScratchOwnedBorrowImpl<B>
        + TakeVecZnxDftImpl<B>
        + ScratchAvailableImpl<B>
        + TakeVecZnxImpl<B>
        + TakeScalarZnxImpl<B>
        + TakeSvpPPolImpl<B>
        + TakeVecZnxBigImpl<B>
        + TakeVecZnxDftSliceImpl<B>
        + TakeMatZnxImpl<B>
        + TakeVecZnxSliceImpl<B>,
    BlindRotationKey<Vec<u8>, BRA>: PrepareAlloc<B, BlindRotationKeyPrepared<Vec<u8>, BRA, B>>,
    BlindRotationKeyPrepared<Vec<u8>, BRA, B>: BlincRotationExecute<B>,
    BlindRotationKey<Vec<u8>, BRA>: BlindRotationKeyAlloc + BlindRotationKeyEncryptSk<B>,
{
    let n_glwe: usize = module.n();
    let base2k: usize = 17;
    let extension_factor: usize = 1;
    let rank: usize = 1;

    let n_lwe: usize = 77;
    let k_lwe_pt: usize = 4;
    let k_lwe_ct: usize = 22;
    let block_size: usize = 7;

    let k_brk: usize = 5 * base2k;
    let rows_brk: usize = 4;

    let k_atk: usize = 5 * base2k;
    let rows_atk: usize = 4;

    let k_tsk: usize = 5 * base2k;
    let rows_tsk: usize = 4;

    let k_ggsw_res: usize = 4 * base2k;
    let rows_ggsw_res: usize = 2;

    let lwe_infos: LWECiphertextLayout = LWECiphertextLayout {
        n: n_lwe.into(),
        k: k_lwe_ct.into(),
        base2k: base2k.into(),
    };

    let cbt_infos: CircuitBootstrappingKeyLayout = CircuitBootstrappingKeyLayout {
        layout_brk: BlindRotationKeyLayout {
            n_glwe: n_glwe.into(),
            n_lwe: n_lwe.into(),
            base2k: base2k.into(),
            k: k_brk.into(),
            rows: rows_brk.into(),
            rank: rank.into(),
        },
        layout_atk: GGLWEAutomorphismKeyLayout {
            n: n_glwe.into(),
            base2k: base2k.into(),
            k: k_atk.into(),
            rows: rows_atk.into(),
            rank: rank.into(),
            digits: Digits(1),
        },
        layout_tsk: GGLWETensorKeyLayout {
            n: n_glwe.into(),
            base2k: base2k.into(),
            k: k_tsk.into(),
            rows: rows_tsk.into(),
            digits: Digits(1),
            rank: rank.into(),
        },
    };

    let ggsw_infos: GGSWCiphertextLayout = GGSWCiphertextLayout {
        n: n_glwe.into(),
        base2k: base2k.into(),
        k: k_ggsw_res.into(),
        rows: rows_ggsw_res.into(),
        digits: Digits(1),
        rank: rank.into(),
    };

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(1 << 23);

    let mut source_xs: Source = Source::new([1u8; 32]);
    let mut source_xa: Source = Source::new([1u8; 32]);
    let mut source_xe: Source = Source::new([1u8; 32]);

    let mut sk_lwe: LWESecret<Vec<u8>> = LWESecret::alloc(n_lwe.into());
    sk_lwe.fill_binary_block(block_size, &mut source_xs);

    let mut sk_glwe: GLWESecret<Vec<u8>> = GLWESecret::alloc_with(n_glwe.into(), rank.into());
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);

    let sk_glwe_prepared: GLWESecretPrepared<Vec<u8>, B> = sk_glwe.prepare_alloc(module, scratch.borrow());

    let data: i64 = 1;

    let mut pt_lwe: LWEPlaintext<Vec<u8>> = LWEPlaintext::alloc_with(base2k.into(), k_lwe_pt.into());
    pt_lwe.encode_i64(data, (k_lwe_pt + 1).into());

    println!("pt_lwe: {pt_lwe}");

    let mut ct_lwe: LWECiphertext<Vec<u8>> = LWECiphertext::alloc(&lwe_infos);
    ct_lwe.encrypt_sk(module, &pt_lwe, &sk_lwe, &mut source_xa, &mut source_xe);

    let now: Instant = Instant::now();
    let cbt_key: CircuitBootstrappingKey<Vec<u8>, BRA> = CircuitBootstrappingKey::encrypt_sk(
        module,
        &sk_lwe,
        &sk_glwe,
        &cbt_infos,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );
    println!("CBT-KGEN: {} ms", now.elapsed().as_millis());

    let mut res: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(&ggsw_infos);

    let log_gap_out = 1;

    let cbt_prepared: CircuitBootstrappingKeyPrepared<Vec<u8>, BRA, B> = cbt_key.prepare_alloc(module, scratch.borrow());

    let now: Instant = Instant::now();
    cbt_prepared.execute_to_exponent(
        module,
        log_gap_out,
        &mut res,
        &ct_lwe,
        k_lwe_pt,
        extension_factor,
        scratch.borrow(),
    );
    println!("CBT: {} ms", now.elapsed().as_millis());

    // X^{data * 2^log_gap_out}
    let mut pt_ggsw: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n_glwe, 1);
    pt_ggsw.at_mut(0, 0)[0] = 1;
    module.vec_znx_rotate_inplace(
        data * (1 << log_gap_out),
        &mut pt_ggsw.as_vec_znx_mut(),
        0,
        scratch.borrow(),
    );

    res.print_noise(module, &sk_glwe_prepared, &pt_ggsw);

    let mut ct_glwe: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&ggsw_infos);
    let mut pt_glwe: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&ggsw_infos);
    pt_glwe.data.at_mut(0, 0)[0] = 1 << (base2k - 2);

    ct_glwe.encrypt_sk(
        module,
        &pt_glwe,
        &sk_glwe_prepared,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    let res_prepared: GGSWCiphertextPrepared<Vec<u8>, B> = res.prepare_alloc(module, scratch.borrow());

    ct_glwe.external_product_inplace(module, &res_prepared, scratch.borrow());

    let mut pt_res: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&ggsw_infos);
    ct_glwe.decrypt(module, &mut pt_res, &sk_glwe_prepared, scratch.borrow());

    // Parameters are set such that the first limb should be noiseless.
    let mut pt_want: Vec<i64> = vec![0i64; module.n()];
    pt_want[data as usize * (1 << log_gap_out)] = pt_glwe.data.at(0, 0)[0];
    assert_eq!(pt_res.data.at(0, 0), pt_want);
}

pub fn test_circuit_bootstrapping_to_constant<B, BRA: BlindRotationAlgo>(module: &Module<B>)
where
    Module<B>: VecZnxFillUniform
        + VecZnxAddNormal
        + VecZnxNormalizeInplace<B>
        + VecZnxDftAllocBytes
        + VecZnxBigNormalize<B>
        + VecZnxDftApply<B>
        + SvpApplyDftToDftInplace<B>
        + VecZnxIdftApplyConsume<B>
        + VecZnxNormalizeTmpBytes
        + VecZnxSubABInplace
        + VecZnxAddInplace
        + VecZnxNormalize<B>
        + VecZnxSub
        + VecZnxAddScalarInplace
        + VecZnxAutomorphism
        + VecZnxSwitchRing
        + VecZnxBigAllocBytes
        + VecZnxIdftApplyTmpA<B>
        + SvpApplyDftToDft<B>
        + VecZnxBigAddInplace<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxBigAlloc<B>
        + VecZnxDftAlloc<B>
        + VecZnxBigNormalizeTmpBytes
        + VmpPMatAlloc<B>
        + VmpPrepare<B>
        + SvpPrepare<B>
        + SvpPPolAlloc<B>
        + VmpApplyDftToDftTmpBytes
        + VmpApplyDftToDft<B>
        + VmpApplyDftToDftAdd<B>
        + SvpPPolAllocBytes
        + VecZnxRotateInplace<B>
        + VecZnxBigAutomorphismInplace<B>
        + VecZnxRotateInplaceTmpBytes
        + VecZnxRshInplace<B>
        + VecZnxDftCopy<B>
        + VecZnxNegateInplace
        + VecZnxCopy
        + VecZnxAutomorphismInplace<B>
        + VecZnxBigSubSmallBInplace<B>
        + VecZnxBigAllocBytes
        + VecZnxDftAddInplace<B>
        + VecZnxRotate
        + ZnFillUniform
        + ZnAddNormal
        + ZnNormalizeInplace<B>,
    B: Backend
        + ScratchOwnedAllocImpl<B>
        + ScratchOwnedBorrowImpl<B>
        + TakeVecZnxDftImpl<B>
        + ScratchAvailableImpl<B>
        + TakeVecZnxImpl<B>
        + TakeScalarZnxImpl<B>
        + TakeSvpPPolImpl<B>
        + TakeVecZnxBigImpl<B>
        + TakeVecZnxDftSliceImpl<B>
        + TakeMatZnxImpl<B>
        + TakeVecZnxSliceImpl<B>,
    BlindRotationKey<Vec<u8>, BRA>: PrepareAlloc<B, BlindRotationKeyPrepared<Vec<u8>, BRA, B>>,
    BlindRotationKeyPrepared<Vec<u8>, BRA, B>: BlincRotationExecute<B>,
    BlindRotationKey<Vec<u8>, BRA>: BlindRotationKeyAlloc + BlindRotationKeyEncryptSk<B>,
{
    let n_glwe: usize = module.n();
    let base2k: usize = 14;
    let extension_factor: usize = 1;
    let rank: usize = 2;

    let n_lwe: usize = 77;
    let k_lwe_pt: usize = 1;
    let k_lwe_ct: usize = 13;
    let block_size: usize = 7;

    let k_brk: usize = 5 * base2k;
    let rows_brk: usize = 3;

    let k_atk: usize = 5 * base2k;
    let rows_atk: usize = 4;

    let k_tsk: usize = 5 * base2k;
    let rows_tsk: usize = 4;

    let k_ggsw_res: usize = 4 * base2k;
    let rows_ggsw_res: usize = 3;

    let lwe_infos: LWECiphertextLayout = LWECiphertextLayout {
        n: n_lwe.into(),
        k: k_lwe_ct.into(),
        base2k: base2k.into(),
    };

    let cbt_infos: CircuitBootstrappingKeyLayout = CircuitBootstrappingKeyLayout {
        layout_brk: BlindRotationKeyLayout {
            n_glwe: n_glwe.into(),
            n_lwe: n_lwe.into(),
            base2k: base2k.into(),
            k: k_brk.into(),
            rows: rows_brk.into(),
            rank: rank.into(),
        },
        layout_atk: GGLWEAutomorphismKeyLayout {
            n: n_glwe.into(),
            base2k: base2k.into(),
            k: k_atk.into(),
            rows: rows_atk.into(),
            rank: rank.into(),
            digits: Digits(1),
        },
        layout_tsk: GGLWETensorKeyLayout {
            n: n_glwe.into(),
            base2k: base2k.into(),
            k: k_tsk.into(),
            rows: rows_tsk.into(),
            digits: Digits(1),
            rank: rank.into(),
        },
    };

    let ggsw_infos: GGSWCiphertextLayout = GGSWCiphertextLayout {
        n: n_glwe.into(),
        base2k: base2k.into(),
        k: k_ggsw_res.into(),
        rows: rows_ggsw_res.into(),
        digits: Digits(1),
        rank: rank.into(),
    };

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(1 << 23);

    let mut source_xs: Source = Source::new([1u8; 32]);
    let mut source_xa: Source = Source::new([1u8; 32]);
    let mut source_xe: Source = Source::new([1u8; 32]);

    let mut sk_lwe: LWESecret<Vec<u8>> = LWESecret::alloc(n_lwe.into());
    sk_lwe.fill_binary_block(block_size, &mut source_xs);

    let mut sk_glwe: GLWESecret<Vec<u8>> = GLWESecret::alloc_with(n_glwe.into(), rank.into());
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);

    let sk_glwe_prepared: GLWESecretPrepared<Vec<u8>, B> = sk_glwe.prepare_alloc(module, scratch.borrow());

    let data: i64 = 1;

    let mut pt_lwe: LWEPlaintext<Vec<u8>> = LWEPlaintext::alloc_with(base2k.into(), k_lwe_pt.into());
    pt_lwe.encode_i64(data, (k_lwe_pt + 1).into());

    println!("pt_lwe: {pt_lwe}");

    let mut ct_lwe: LWECiphertext<Vec<u8>> = LWECiphertext::alloc(&lwe_infos);
    ct_lwe.encrypt_sk(module, &pt_lwe, &sk_lwe, &mut source_xa, &mut source_xe);

    let now: Instant = Instant::now();
    let cbt_key: CircuitBootstrappingKey<Vec<u8>, BRA> = CircuitBootstrappingKey::encrypt_sk(
        module,
        &sk_lwe,
        &sk_glwe,
        &cbt_infos,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );
    println!("CBT-KGEN: {} ms", now.elapsed().as_millis());

    let mut res: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(&ggsw_infos);

    let cbt_prepared: CircuitBootstrappingKeyPrepared<Vec<u8>, BRA, B> = cbt_key.prepare_alloc(module, scratch.borrow());

    let now: Instant = Instant::now();
    cbt_prepared.execute_to_constant(
        module,
        &mut res,
        &ct_lwe,
        k_lwe_pt,
        extension_factor,
        scratch.borrow(),
    );
    println!("CBT: {} ms", now.elapsed().as_millis());

    // X^{data * 2^log_gap_out}
    let mut pt_ggsw: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n_glwe, 1);
    pt_ggsw.at_mut(0, 0)[0] = data;

    res.print_noise(module, &sk_glwe_prepared, &pt_ggsw);

    let mut ct_glwe: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&ggsw_infos);
    let mut pt_glwe: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&ggsw_infos);
    pt_glwe.data.at_mut(0, 0)[0] = 1 << (base2k - k_lwe_pt - 1);

    ct_glwe.encrypt_sk(
        module,
        &pt_glwe,
        &sk_glwe_prepared,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    let res_prepared: GGSWCiphertextPrepared<Vec<u8>, B> = res.prepare_alloc(module, scratch.borrow());

    ct_glwe.external_product_inplace(module, &res_prepared, scratch.borrow());

    let mut pt_res: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&ggsw_infos);
    ct_glwe.decrypt(module, &mut pt_res, &sk_glwe_prepared, scratch.borrow());

    // Parameters are set such that the first limb should be noiseless.
    let mut pt_want: Vec<i64> = vec![0i64; module.n()];
    pt_want[0] = pt_glwe.data.at(0, 0)[0] * data;
    assert_eq!(pt_res.data.at(0, 0), pt_want);
}
