use std::time::Instant;

use backend::{
    hal::{
        api::{
            ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxAutomorphism,
            VecZnxFillUniform, VecZnxNormalizeInplace, VecZnxRotateInplace, VecZnxSwithcDegree, VmpPMatAlloc, VmpPMatPrepare,
            ZnxView, ZnxViewMut,
        },
        layouts::{Backend, Module, ScalarZnx, ScratchOwned},
        oep::{
            ScratchAvailableImpl, ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeMatZnxImpl, TakeScalarZnxImpl,
            TakeSvpPPolImpl, TakeVecZnxBigImpl, TakeVecZnxDftImpl, TakeVecZnxDftSliceImpl, TakeVecZnxImpl, TakeVecZnxSliceImpl,
        },
    },
    implementation::cpu_spqlios::FFT64,
};
use sampling::source::Source;

use crate::tfhe::{
    blind_rotation::BlindRotationKeyExecLayoutFamily,
    circuit_bootstrapping::{
        CGGICircuitBootstrapFamily, CircuitBootstrappingKeyCGGI, CircuitBootstrappingKeyCGGIExec,
        circuit_bootstrap_to_constant_cggi, circuit_bootstrap_to_exponent_cggi,
    },
};

use core::trait_families::{
    GGLWEAutomorphismKeyEncryptSkFamily, GGLWETensorKeyEncryptSkFamily, GGSWAssertNoiseFamily, GGSWEncryptSkFamily,
    GLWEDecryptFamily,
};

use core::layouts::{
    GGSWCiphertext, GLWECiphertext, GLWEPlaintext, GLWESecret, LWECiphertext, LWEPlaintext, LWESecret,
    prepared::{GGSWCiphertextExec, GLWESecretExec},
};

#[test]
fn test_to_exponent() {
    let module: Module<FFT64> = Module::<FFT64>::new(256);
    to_exponent(&module);
}

fn to_exponent<B: Backend>(module: &Module<B>)
where
    Module<B>: VecZnxFillUniform
        + VecZnxAddNormal
        + VecZnxNormalizeInplace<B>
        + GGSWEncryptSkFamily<B>
        + VecZnxAddScalarInplace
        + GGLWEAutomorphismKeyEncryptSkFamily<B>
        + VecZnxAutomorphism
        + VecZnxSwithcDegree
        + GGLWETensorKeyEncryptSkFamily<B>
        + BlindRotationKeyExecLayoutFamily<B>
        + CGGICircuitBootstrapFamily<B>
        + GLWEDecryptFamily<B>
        + GGSWAssertNoiseFamily<B>
        + VmpPMatAlloc<B>
        + VmpPMatPrepare<B>,
    B: ScratchOwnedAllocImpl<B>
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
{
    let n: usize = module.n();
    let basek: usize = 17;
    let extension_factor: usize = 1;
    let rank: usize = 1;
    let sigma: f64 = 3.2;

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

    let sk_glwe_exec: GLWESecretExec<Vec<u8>, B> = GLWESecretExec::from(module, &sk_glwe);

    let data: i64 = 1;

    let mut pt_lwe: LWEPlaintext<Vec<u8>> = LWEPlaintext::alloc(basek, k_lwe_pt);
    pt_lwe.encode_i64(data, k_lwe_pt + 2);

    println!("pt_lwe: {}", pt_lwe);

    let mut ct_lwe: LWECiphertext<Vec<u8>> = LWECiphertext::alloc(n_lwe, basek, k_lwe_ct);
    ct_lwe.encrypt_sk(
        module,
        &pt_lwe,
        &sk_lwe,
        &mut source_xa,
        &mut source_xe,
        sigma,
    );

    let now: Instant = Instant::now();
    let cbt_key: CircuitBootstrappingKeyCGGI<Vec<u8>> = CircuitBootstrappingKeyCGGI::generate(
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
        sigma,
        scratch.borrow(),
    );
    println!("CBT-KGEN: {} ms", now.elapsed().as_millis());

    let k_ggsw_res: usize = 4 * basek;
    let rows_ggsw_res: usize = 2;

    let mut res: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(n, basek, k_ggsw_res, rows_ggsw_res, 1, rank);

    let log_gap_out = 1;

    let cbt_exec: CircuitBootstrappingKeyCGGIExec<Vec<u8>, B> =
        CircuitBootstrappingKeyCGGIExec::from(module, &cbt_key, scratch.borrow());

    let now: Instant = Instant::now();
    circuit_bootstrap_to_exponent_cggi(
        module,
        log_gap_out,
        &mut res,
        &ct_lwe,
        k_lwe_pt,
        extension_factor,
        &cbt_exec,
        scratch.borrow(),
    );
    println!("CBT: {} ms", now.elapsed().as_millis());

    // X^{data * 2^log_gap_out}
    let mut pt_ggsw: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);
    pt_ggsw.at_mut(0, 0)[0] = 1;
    module.vec_znx_rotate_inplace(data * (1 << log_gap_out), &mut pt_ggsw.as_vec_znx_mut(), 0);

    res.print_noise(module, &sk_glwe_exec, &pt_ggsw);

    let k_glwe: usize = k_ggsw_res;

    let mut ct_glwe: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(n, basek, k_glwe, rank);
    let mut pt_glwe: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(n, basek, basek);
    pt_glwe.data.at_mut(0, 0)[0] = 1 << (basek - 2);

    ct_glwe.encrypt_sk(
        module,
        &pt_glwe,
        &sk_glwe_exec,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let res_exec: GGSWCiphertextExec<Vec<u8>, B> = GGSWCiphertextExec::from(module, &res, scratch.borrow());

    ct_glwe.external_product_inplace(module, &res_exec, scratch.borrow());

    let mut pt_res: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(n, basek, k_glwe);
    ct_glwe.decrypt(module, &mut pt_res, &sk_glwe_exec, scratch.borrow());

    // Parameters are set such that the first limb should be noiseless.
    let mut pt_want: Vec<i64> = vec![0i64; module.n()];
    pt_want[data as usize * (1 << log_gap_out)] = pt_glwe.data.at(0, 0)[0];
    assert_eq!(pt_res.data.at(0, 0), pt_want);
}

#[test]
fn test_to_constant() {
    let module: Module<FFT64> = Module::<FFT64>::new(256);
    to_constant(&module);
}

fn to_constant<B: Backend>(module: &Module<B>)
where
    Module<B>: VecZnxFillUniform
        + VecZnxAddNormal
        + VecZnxNormalizeInplace<B>
        + GGSWEncryptSkFamily<B>
        + VecZnxAddScalarInplace
        + GGLWEAutomorphismKeyEncryptSkFamily<B>
        + VecZnxAutomorphism
        + VecZnxSwithcDegree
        + GGLWETensorKeyEncryptSkFamily<B>
        + BlindRotationKeyExecLayoutFamily<B>
        + CGGICircuitBootstrapFamily<B>
        + GLWEDecryptFamily<B>
        + GGSWAssertNoiseFamily<B>
        + VmpPMatAlloc<B>
        + VmpPMatPrepare<B>,
    B: ScratchOwnedAllocImpl<B>
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
{
    let n = module.n();
    let basek: usize = 14;
    let extension_factor: usize = 1;
    let rank: usize = 2;
    let sigma: f64 = 3.2;

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

    let sk_glwe_exec: GLWESecretExec<Vec<u8>, B> = GLWESecretExec::from(module, &sk_glwe);

    let data: i64 = 1;

    let mut pt_lwe: LWEPlaintext<Vec<u8>> = LWEPlaintext::alloc(basek, k_lwe_pt);
    pt_lwe.encode_i64(data, k_lwe_pt + 2);

    println!("pt_lwe: {}", pt_lwe);

    let mut ct_lwe: LWECiphertext<Vec<u8>> = LWECiphertext::alloc(n_lwe, basek, k_lwe_ct);
    ct_lwe.encrypt_sk(
        module,
        &pt_lwe,
        &sk_lwe,
        &mut source_xa,
        &mut source_xe,
        sigma,
    );

    let now: Instant = Instant::now();
    let cbt_key: CircuitBootstrappingKeyCGGI<Vec<u8>> = CircuitBootstrappingKeyCGGI::generate(
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
        sigma,
        scratch.borrow(),
    );
    println!("CBT-KGEN: {} ms", now.elapsed().as_millis());

    let k_ggsw_res: usize = 4 * basek;
    let rows_ggsw_res: usize = 3;

    let mut res: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(n, basek, k_ggsw_res, rows_ggsw_res, 1, rank);

    let cbt_exec: CircuitBootstrappingKeyCGGIExec<Vec<u8>, B> =
        CircuitBootstrappingKeyCGGIExec::from(module, &cbt_key, scratch.borrow());

    let now: Instant = Instant::now();
    circuit_bootstrap_to_constant_cggi(
        module,
        &mut res,
        &ct_lwe,
        k_lwe_pt,
        extension_factor,
        &cbt_exec,
        scratch.borrow(),
    );
    println!("CBT: {} ms", now.elapsed().as_millis());

    // X^{data * 2^log_gap_out}
    let mut pt_ggsw: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);
    pt_ggsw.at_mut(0, 0)[0] = data;

    res.print_noise(module, &sk_glwe_exec, &pt_ggsw);

    let k_glwe: usize = k_ggsw_res;

    let mut ct_glwe: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(n, basek, k_glwe, rank);
    let mut pt_glwe: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(n, basek, basek);
    pt_glwe.data.at_mut(0, 0)[0] = 1 << (basek - k_lwe_pt - 1);

    ct_glwe.encrypt_sk(
        module,
        &pt_glwe,
        &sk_glwe_exec,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let res_exec: GGSWCiphertextExec<Vec<u8>, B> = GGSWCiphertextExec::from(module, &res, scratch.borrow());

    ct_glwe.external_product_inplace(module, &res_exec, scratch.borrow());

    let mut pt_res: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(n, basek, k_glwe);
    ct_glwe.decrypt(module, &mut pt_res, &sk_glwe_exec, scratch.borrow());

    // Parameters are set such that the first limb should be noiseless.
    let mut pt_want: Vec<i64> = vec![0i64; module.n()];
    pt_want[0] = pt_glwe.data.at(0, 0)[0] * data;
    assert_eq!(pt_res.data.at(0, 0), pt_want);
}
