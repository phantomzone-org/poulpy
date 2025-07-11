use std::time::Instant;

use backend::{Encoding, FFT64, Module, ScalarZnxAlloc, ScratchOwned, VecZnxOps, ZnxView, ZnxViewMut};
use sampling::source::Source;

use crate::{
    FourierGLWESecret, GGSWCiphertext, GLWECiphertext, GLWEPlaintext, GLWESecret, LWECiphertext, LWESecret,
    circuit_bootstrapping::circuit_bootstrapping::{
        CircuitBootstrappingKeyCGGI, circuit_bootstrap_to_constant_cggi, circuit_bootstrap_to_exponent_cggi,
    },
    get_ggsw_noise,
    lwe::LWEPlaintext,
};

#[test]
fn to_exponent() {
    let module: Module<FFT64> = Module::<FFT64>::new(2048);
    let basek: usize = 17;
    let extension_factor: usize = 1;
    let rank: usize = 1;
    let sigma: f64 = 3.2;

    let n_lwe: usize = 1071;
    let k_lwe_pt: usize = 4;
    let k_lwe_ct: usize = 22;
    let block_size: usize = 7;

    let k_brk: usize = 5 * basek;
    let rows_brk: usize = 4;

    let k_trace: usize = 5 * basek;
    let rows_trace: usize = 4;

    let k_tsk: usize = 5 * basek;
    let rows_tsk: usize = 4;

    let mut scratch: ScratchOwned = ScratchOwned::new(1 << 23);

    let mut source_xs: Source = Source::new([1u8; 32]);
    let mut source_xa: Source = Source::new([1u8; 32]);
    let mut source_xe: Source = Source::new([1u8; 32]);

    let mut sk_lwe: LWESecret<Vec<u8>> = LWESecret::alloc(n_lwe);
    sk_lwe.fill_binary_block(block_size, &mut source_xs);

    let mut sk_glwe: GLWESecret<Vec<u8>> = GLWESecret::alloc(&module, rank);
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_glwe_fourier: FourierGLWESecret<Vec<u8>, FFT64> = FourierGLWESecret::alloc(&module, rank);
    sk_glwe_fourier.set(&module, &sk_glwe);

    let data: i64 = 1;

    let mut pt_lwe: LWEPlaintext<Vec<u8>> = LWEPlaintext::alloc(basek, k_lwe_pt);
    pt_lwe
        .data
        .encode_coeff_i64(0, basek, k_lwe_pt + 2, 0, data, k_lwe_pt);

    println!("pt_lwe: {}", pt_lwe.data);

    let mut ct_lwe: LWECiphertext<Vec<u8>> = LWECiphertext::alloc(n_lwe, basek, k_lwe_ct);
    ct_lwe.encrypt_sk(&pt_lwe, &sk_lwe, &mut source_xa, &mut source_xe, sigma);

    let now: Instant = Instant::now();
    let cbt_key: CircuitBootstrappingKeyCGGI<Vec<u8>> = CircuitBootstrappingKeyCGGI::generate(
        &module,
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

    let mut res: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::alloc(&module, basek, k_ggsw_res, rows_ggsw_res, 1, rank);

    let log_gap_out = 1;

    let now: Instant = Instant::now();
    circuit_bootstrap_to_exponent_cggi(
        &module,
        log_gap_out,
        &mut res,
        &ct_lwe,
        k_lwe_pt,
        extension_factor,
        &cbt_key,
        scratch.borrow(),
    );
    println!("CBT: {} ms", now.elapsed().as_millis());

    // X^{data * 2^log_gap_out}
    let mut pt_ggsw: backend::ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);
    pt_ggsw.at_mut(0, 0)[0] = 1;
    module.vec_znx_rotate_inplace(data * (1 << log_gap_out), &mut pt_ggsw, 0);

    let noise: Vec<f64> = get_ggsw_noise(&module, &res, &pt_ggsw, &sk_glwe_fourier);

    println!("noise: {:?}", &noise);

    let k_glwe: usize = k_ggsw_res;

    let mut ct_glwe: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, k_glwe, rank);
    let mut pt_glwe: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, basek);
    pt_glwe.data.at_mut(0, 0)[0] = 1 << (basek - 2);

    ct_glwe.encrypt_sk(
        &module,
        &pt_glwe,
        &sk_glwe_fourier,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_glwe.external_product_inplace(&module, &res, scratch.borrow());

    let mut pt_res: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_glwe);
    ct_glwe.decrypt(&module, &mut pt_res, &sk_glwe_fourier, scratch.borrow());

    // Parameters are set such that the first limb should be noiseless.
    let mut pt_want: Vec<i64> = vec![0i64; module.n()];
    pt_want[data as usize * (1 << log_gap_out)] = pt_glwe.data.at(0, 0)[0];
    assert_eq!(pt_res.data.at(0, 0), pt_want);
}

#[test]
fn to_constant() {
    let module: Module<FFT64> = Module::<FFT64>::new(1024);
    let basek: usize = 14;
    let extension_factor: usize = 1;
    let rank: usize = 2;
    let sigma: f64 = 3.2;

    let n_lwe: usize = 574;
    let k_lwe_pt: usize = 1;
    let k_lwe_ct: usize = 13;
    let block_size: usize = 7;

    let k_brk: usize = 5 * basek;
    let rows_brk: usize = 3;

    let k_trace: usize = 5 * basek;
    let rows_trace: usize = 4;

    let k_tsk: usize = 5 * basek;
    let rows_tsk: usize = 4;

    let mut scratch: ScratchOwned = ScratchOwned::new(1 << 23);

    let mut source_xs: Source = Source::new([1u8; 32]);
    let mut source_xa: Source = Source::new([1u8; 32]);
    let mut source_xe: Source = Source::new([1u8; 32]);

    let mut sk_lwe: LWESecret<Vec<u8>> = LWESecret::alloc(n_lwe);
    sk_lwe.fill_binary_block(block_size, &mut source_xs);

    let mut sk_glwe: GLWESecret<Vec<u8>> = GLWESecret::alloc(&module, rank);
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_glwe_fourier: FourierGLWESecret<Vec<u8>, FFT64> = FourierGLWESecret::alloc(&module, rank);
    sk_glwe_fourier.set(&module, &sk_glwe);

    let data: i64 = 1;

    let mut pt_lwe: LWEPlaintext<Vec<u8>> = LWEPlaintext::alloc(basek, k_lwe_pt);
    pt_lwe
        .data
        .encode_coeff_i64(0, basek, k_lwe_pt + 2, 0, data, k_lwe_pt);

    println!("pt_lwe: {}", pt_lwe.data);

    let mut ct_lwe: LWECiphertext<Vec<u8>> = LWECiphertext::alloc(n_lwe, basek, k_lwe_ct);
    ct_lwe.encrypt_sk(&pt_lwe, &sk_lwe, &mut source_xa, &mut source_xe, sigma);

    let now: Instant = Instant::now();
    let cbt_key: CircuitBootstrappingKeyCGGI<Vec<u8>> = CircuitBootstrappingKeyCGGI::generate(
        &module,
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

    let mut res: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::alloc(&module, basek, k_ggsw_res, rows_ggsw_res, 1, rank);

    let now: Instant = Instant::now();
    circuit_bootstrap_to_constant_cggi(
        &module,
        &mut res,
        &ct_lwe,
        k_lwe_pt,
        extension_factor,
        &cbt_key,
        scratch.borrow(),
    );
    println!("CBT: {} ms", now.elapsed().as_millis());

    // X^{data * 2^log_gap_out}
    let mut pt_ggsw: backend::ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);
    pt_ggsw.at_mut(0, 0)[0] = data;

    let noise: Vec<f64> = get_ggsw_noise(&module, &res, &pt_ggsw, &sk_glwe_fourier);

    println!("noise: {:?}", &noise);

    let k_glwe: usize = k_ggsw_res;

    let mut ct_glwe: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, k_glwe, rank);
    let mut pt_glwe: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, basek);
    pt_glwe.data.at_mut(0, 0)[0] = 1 << (basek - k_lwe_pt - 1);

    ct_glwe.encrypt_sk(
        &module,
        &pt_glwe,
        &sk_glwe_fourier,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_glwe.external_product_inplace(&module, &res, scratch.borrow());

    let mut pt_res: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_glwe);
    ct_glwe.decrypt(&module, &mut pt_res, &sk_glwe_fourier, scratch.borrow());

    // Parameters are set such that the first limb should be noiseless.
    let mut pt_want: Vec<i64> = vec![0i64; module.n()];
    pt_want[0] = pt_glwe.data.at(0, 0)[0] * data;
    assert_eq!(pt_res.data.at(0, 0), pt_want);
}
