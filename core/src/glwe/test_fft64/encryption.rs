use backend::{FFT64, FillUniform, Module, ScratchOwned, Stats};
use sampling::source::Source;

use crate::{FourierGLWECiphertext, FourierGLWESecret, GLWECiphertext, GLWEOps, GLWEPlaintext, GLWEPublicKey, GLWESecret, Infos};

#[test]
fn encrypt_sk() {
    let log_n: usize = 8;
    (1..4).for_each(|rank| {
        println!("test encrypt_sk rank: {}", rank);
        test_encrypt_sk(log_n, 8, 54, 30, 3.2, rank);
    });
}

#[test]
fn encrypt_zero_sk() {
    let log_n: usize = 8;
    (1..4).for_each(|rank| {
        println!("test encrypt_zero_sk rank: {}", rank);
        test_encrypt_zero_sk(log_n, 8, 64, 3.2, rank);
    });
}

#[test]
fn encrypt_pk() {
    let log_n: usize = 8;
    (1..4).for_each(|rank| {
        println!("test encrypt_pk rank: {}", rank);
        test_encrypt_pk(log_n, 8, 64, 64, 3.2, rank)
    });
}

fn test_encrypt_sk(log_n: usize, basek: usize, k_ct: usize, k_pt: usize, sigma: f64, rank: usize) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);

    let mut ct: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, k_ct, rank);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_pt);
    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_pt);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWECiphertext::encrypt_sk_scratch_space(&module, basek, ct.k())
            | GLWECiphertext::decrypt_scratch_space(&module, basek, ct.k()),
    );

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(&module, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let sk_dft: FourierGLWESecret<Vec<u8>, FFT64> = FourierGLWESecret::from(&module, &sk);

    pt_want
        .data
        .fill_uniform(basek, 0, pt_want.size(), &mut source_xa);

    ct.encrypt_sk(
        &module,
        &pt_want,
        &sk_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct.decrypt(&module, &mut pt_have, &sk_dft, scratch.borrow());

    pt_want.sub_inplace_ab(&module, &pt_have);

    let noise_have: f64 = pt_want.data.std(0, basek) * (ct.k() as f64).exp2();
    let noise_want: f64 = sigma;

    assert!(noise_have <= noise_want + 0.2);
}

fn test_encrypt_zero_sk(log_n: usize, basek: usize, k_ct: usize, sigma: f64, rank: usize) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);

    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_ct);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([1u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(&module, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let sk_dft: FourierGLWESecret<Vec<u8>, FFT64> = FourierGLWESecret::from(&module, &sk);

    let mut ct_dft: FourierGLWECiphertext<Vec<u8>, FFT64> = FourierGLWECiphertext::alloc(&module, basek, k_ct, rank);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        FourierGLWECiphertext::decrypt_scratch_space(&module, basek, k_ct)
            | FourierGLWECiphertext::encrypt_sk_scratch_space(&module, basek, k_ct, rank),
    );

    ct_dft.encrypt_zero_sk(
        &module,
        &sk_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );
    ct_dft.decrypt(&module, &mut pt, &sk_dft, scratch.borrow());

    assert!((sigma - pt.data.std(0, basek) * (k_ct as f64).exp2()) <= 0.2);
}

fn test_encrypt_pk(log_n: usize, basek: usize, k_ct: usize, k_pk: usize, sigma: f64, rank: usize) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);

    let mut ct: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, k_ct, rank);
    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_ct);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_ct);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);
    let mut source_xu: Source = Source::new([0u8; 32]);

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(&module, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let sk_dft: FourierGLWESecret<Vec<u8>, FFT64> = FourierGLWESecret::from(&module, &sk);

    let mut pk: GLWEPublicKey<Vec<u8>, FFT64> = GLWEPublicKey::alloc(&module, basek, k_pk, rank);
    pk.generate_from_sk(&module, &sk_dft, &mut source_xa, &mut source_xe, sigma);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWECiphertext::encrypt_sk_scratch_space(&module, basek, ct.k())
            | GLWECiphertext::decrypt_scratch_space(&module, basek, ct.k())
            | GLWECiphertext::encrypt_pk_scratch_space(&module, basek, pk.k()),
    );

    pt_want
        .data
        .fill_uniform(basek, 0, pt_want.size(), &mut source_xa);

    ct.encrypt_pk(
        &module,
        &pt_want,
        &pk,
        &mut source_xu,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct.decrypt(&module, &mut pt_have, &sk_dft, scratch.borrow());

    pt_want.sub_inplace_ab(&module, &pt_have);

    let noise_have: f64 = pt_want.data.std(0, basek).log2();
    let noise_want: f64 = ((((rank as f64) + 1.0) * module.n() as f64 * 0.5 * sigma * sigma).sqrt()).log2() - (k_ct as f64);

    assert!(
        (noise_have - noise_want).abs() < 0.2,
        "{} {}",
        noise_have,
        noise_want
    );
}
