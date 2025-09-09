use std::time::Instant;

use poulpy_hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApply, SvpApplyInplace, SvpPPolAlloc, SvpPPolAllocBytes, SvpPrepare,
        VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxAutomorphism, VecZnxAutomorphismInplace,
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
        BlindRotationKeyPrepared,
    },
    circuit_bootstrapping::{
        CircuitBootstrappingKey, CircuitBootstrappingKeyEncryptSk, CircuitBootstrappingKeyPrepared, CirtuitBootstrappingExecute,
    },
};

use poulpy_core::layouts::prepared::PrepareAlloc;

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
        + SvpApplyInplace<B>
        + VecZnxIdftApplyConsume<B>
        + VecZnxNormalizeTmpBytes
        + VecZnxSubABInplace
        + VecZnxAddInplace
        + VecZnxNormalize<B>
        + VecZnxSub
        + VecZnxAddScalarInplace
        + VecZnxAutomorphism
        + VecZnxSwitchRing<B>
        + VecZnxBigAllocBytes
        + VecZnxIdftApplyTmpA<B>
        + SvpApply<B>
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
    let n: usize = module.n();
    let basek: usize = 17;
    let extension_factor: usize = 1;
    let rank: usize = 1;

    let n_lwe: usize = 77;
    let k_lwe_pt: usize = 4;
    let k_lwe_ct: usize = 22;
    let block_size: usize = 7;

    let k_brk: usize = 5 * basek;
    let rows_brk: usize = 4;

    let k_trace: usize = 5 * basek;
    let rows_trace: usize = 4;

    let k_tsk: usize = 5 * basek;
    let rows_tsk: usize = 4;

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(1 << 23);

    let mut source_xs: Source = Source::new([1u8; 32]);
    let mut source_xa: Source = Source::new([1u8; 32]);
    let mut source_xe: Source = Source::new([1u8; 32]);

    let mut sk_lwe: LWESecret<Vec<u8>> = LWESecret::alloc(n_lwe);
    sk_lwe.fill_binary_block(block_size, &mut source_xs);

    let mut sk_glwe: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);

    let sk_glwe_prepared: GLWESecretPrepared<Vec<u8>, B> = sk_glwe.prepare_alloc(module, scratch.borrow());

    let data: i64 = 1;

    let mut pt_lwe: LWEPlaintext<Vec<u8>> = LWEPlaintext::alloc(basek, k_lwe_pt);
    pt_lwe.encode_i64(data, k_lwe_pt + 1);

    println!("pt_lwe: {}", pt_lwe);

    let mut ct_lwe: LWECiphertext<Vec<u8>> = LWECiphertext::alloc(n_lwe, basek, k_lwe_ct);
    ct_lwe.encrypt_sk(module, &pt_lwe, &sk_lwe, &mut source_xa, &mut source_xe);

    let now: Instant = Instant::now();
    let cbt_key: CircuitBootstrappingKey<Vec<u8>, BRA> = CircuitBootstrappingKey::encrypt_sk(
        module,
        basek,
        &sk_lwe,
        &sk_glwe,
        k_brk,
        rows_brk,
        k_trace,
        rows_trace,
        k_tsk,
        rows_tsk,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );
    println!("CBT-KGEN: {} ms", now.elapsed().as_millis());

    let k_ggsw_res: usize = 4 * basek;
    let rows_ggsw_res: usize = 2;

    let mut res: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(n, basek, k_ggsw_res, rows_ggsw_res, 1, rank);

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
    let mut pt_ggsw: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);
    pt_ggsw.at_mut(0, 0)[0] = 1;
    module.vec_znx_rotate_inplace(
        data * (1 << log_gap_out),
        &mut pt_ggsw.as_vec_znx_mut(),
        0,
        scratch.borrow(),
    );

    res.print_noise(module, &sk_glwe_prepared, &pt_ggsw);

    let k_glwe: usize = k_ggsw_res;

    let mut ct_glwe: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(n, basek, k_glwe, rank);
    let mut pt_glwe: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(n, basek, basek);
    pt_glwe.data.at_mut(0, 0)[0] = 1 << (basek - 2);

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

    let mut pt_res: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(n, basek, k_glwe);
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
        + SvpApplyInplace<B>
        + VecZnxIdftApplyConsume<B>
        + VecZnxNormalizeTmpBytes
        + VecZnxSubABInplace
        + VecZnxAddInplace
        + VecZnxNormalize<B>
        + VecZnxSub
        + VecZnxAddScalarInplace
        + VecZnxAutomorphism
        + VecZnxSwitchRing<B>
        + VecZnxBigAllocBytes
        + VecZnxIdftApplyTmpA<B>
        + SvpApply<B>
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
    let n: usize = module.n();
    let basek: usize = 14;
    let extension_factor: usize = 1;
    let rank: usize = 2;

    let n_lwe: usize = 77;
    let k_lwe_pt: usize = 1;
    let k_lwe_ct: usize = 13;
    let block_size: usize = 7;

    let k_brk: usize = 5 * basek;
    let rows_brk: usize = 3;

    let k_trace: usize = 5 * basek;
    let rows_trace: usize = 4;

    let k_tsk: usize = 5 * basek;
    let rows_tsk: usize = 4;

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(1 << 23);

    let mut source_xs: Source = Source::new([1u8; 32]);
    let mut source_xa: Source = Source::new([1u8; 32]);
    let mut source_xe: Source = Source::new([1u8; 32]);

    let mut sk_lwe: LWESecret<Vec<u8>> = LWESecret::alloc(n_lwe);
    sk_lwe.fill_binary_block(block_size, &mut source_xs);

    let mut sk_glwe: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);

    let sk_glwe_prepared: GLWESecretPrepared<Vec<u8>, B> = sk_glwe.prepare_alloc(module, scratch.borrow());

    let data: i64 = 1;

    let mut pt_lwe: LWEPlaintext<Vec<u8>> = LWEPlaintext::alloc(basek, k_lwe_pt);
    pt_lwe.encode_i64(data, k_lwe_pt + 1);

    println!("pt_lwe: {}", pt_lwe);

    let mut ct_lwe: LWECiphertext<Vec<u8>> = LWECiphertext::alloc(n_lwe, basek, k_lwe_ct);
    ct_lwe.encrypt_sk(module, &pt_lwe, &sk_lwe, &mut source_xa, &mut source_xe);

    let now: Instant = Instant::now();
    let cbt_key: CircuitBootstrappingKey<Vec<u8>, BRA> = CircuitBootstrappingKey::encrypt_sk(
        module,
        basek,
        &sk_lwe,
        &sk_glwe,
        k_brk,
        rows_brk,
        k_trace,
        rows_trace,
        k_tsk,
        rows_tsk,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );
    println!("CBT-KGEN: {} ms", now.elapsed().as_millis());

    let k_ggsw_res: usize = 4 * basek;
    let rows_ggsw_res: usize = 3;

    let mut res: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(n, basek, k_ggsw_res, rows_ggsw_res, 1, rank);

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
    let mut pt_ggsw: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);
    pt_ggsw.at_mut(0, 0)[0] = data;

    res.print_noise(module, &sk_glwe_prepared, &pt_ggsw);

    let k_glwe: usize = k_ggsw_res;

    let mut ct_glwe: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(n, basek, k_glwe, rank);
    let mut pt_glwe: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(n, basek, basek);
    pt_glwe.data.at_mut(0, 0)[0] = 1 << (basek - k_lwe_pt - 1);

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

    let mut pt_res: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(n, basek, k_glwe);
    ct_glwe.decrypt(module, &mut pt_res, &sk_glwe_prepared, scratch.borrow());

    // Parameters are set such that the first limb should be noiseless.
    let mut pt_want: Vec<i64> = vec![0i64; module.n()];
    pt_want[0] = pt_glwe.data.at(0, 0)[0] * data;
    assert_eq!(pt_res.data.at(0, 0), pt_want);
}
