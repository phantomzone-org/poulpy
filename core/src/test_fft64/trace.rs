use std::collections::HashMap;

use backend::{FFT64, FillUniform, Module, ScratchOwned, Stats, VecZnxOps, ZnxView, ZnxViewMut};
use sampling::source::Source;

use crate::{
    automorphism::AutomorphismKey,
    elem::Infos,
    glwe_ciphertext::GLWECiphertext,
    glwe_plaintext::GLWEPlaintext,
    keys::{SecretKey, SecretKeyFourier},
    test_fft64::gglwe::var_noise_gglwe_product,
};

#[test]
fn trace_inplace() {
    (1..4).for_each(|rank| {
        println!("test trace_inplace rank: {}", rank);
        test_trace_inplace(11, 8, 54, 3.2, rank);
    });
}

fn test_trace_inplace(log_n: usize, basek: usize, k: usize, sigma: f64, rank: usize) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);

    let k_autokey: usize = k + basek;

    let rows: usize = (k + basek - 1) / basek;

    let mut ct: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, k, rank);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k);
    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWECiphertext::encrypt_sk_scratch_space(&module, ct.size())
            | GLWECiphertext::decrypt_scratch_space(&module, ct.size())
            | AutomorphismKey::generate_from_sk_scratch_space(&module, rank, k_autokey)
            | GLWECiphertext::trace_inplace_scratch_space(&module, ct.size(), k_autokey, rank),
    );

    let mut sk: SecretKey<Vec<u8>> = SecretKey::alloc(&module, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::alloc(&module, rank);
    sk_dft.dft(&module, &sk);

    let mut data_want: Vec<i64> = vec![0i64; module.n()];

    data_want
        .iter_mut()
        .for_each(|x| *x = source_xa.next_i64() & 0xFF);

    pt_have
        .data
        .fill_uniform(basek, 0, pt_have.size(), &mut source_xa);

    ct.encrypt_sk(
        &module,
        &pt_have,
        &sk_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let mut auto_keys: HashMap<i64, AutomorphismKey<Vec<u8>, FFT64>> = HashMap::new();
    let gal_els: Vec<i64> = GLWECiphertext::trace_galois_elements(&module);
    gal_els.iter().for_each(|gal_el| {
        let mut key: AutomorphismKey<Vec<u8>, FFT64> = AutomorphismKey::alloc(&module, basek, k_autokey, rows, rank);
        key.generate_from_sk(
            &module,
            *gal_el,
            &sk,
            &mut source_xa,
            &mut source_xe,
            sigma,
            scratch.borrow(),
        );
        auto_keys.insert(*gal_el, key);
    });

    ct.trace_inplace(&module, 0, 5, &auto_keys, scratch.borrow());
    ct.trace_inplace(&module, 5, log_n, &auto_keys, scratch.borrow());

    (0..pt_want.size()).for_each(|i| pt_want.data.at_mut(0, i)[0] = pt_have.data.at(0, i)[0]);

    ct.decrypt(&module, &mut pt_have, &sk_dft, scratch.borrow());

    module.vec_znx_sub_ab_inplace(&mut pt_want, 0, &pt_have, 0);
    module.vec_znx_normalize_inplace(basek, &mut pt_want, 0, scratch.borrow());

    let noise_have = pt_want.data.std(0, basek).log2();

    let mut noise_want: f64 = var_noise_gglwe_product(
        module.n() as f64,
        basek,
        0.5,
        0.5,
        1.0 / 12.0,
        sigma * sigma,
        0.0,
        rank as f64,
        k,
        k_autokey,
    );
    noise_want += sigma * sigma * (-2.0 * (k) as f64).exp2();
    noise_want += module.n() as f64 * 1.0 / 12.0 * 0.5 * rank as f64 * (-2.0 * (k) as f64).exp2();
    noise_want = noise_want.sqrt().log2();

    assert!((noise_have - noise_want).abs() < 1.0);
}
