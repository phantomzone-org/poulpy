use backend::{Backend, FFT64, Module, ModuleNew, ScratchOwned, VecZnxFillUniform, VecZnxStd};
use sampling::source::Source;

use crate::{
    GLWECiphertext, GLWEDecryptFamily, GLWEEncryptPkFamily, GLWEEncryptSkFamily, GLWEOps, GLWEPlaintext, GLWEPublicKey,
    GLWEPublicKeyFamily, GLWESecret, GLWESecretExec, GLWESecretFamily, Infos,
};

#[test]
fn encrypt_sk() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    (1..4).for_each(|rank| {
        println!("test encrypt_sk rank: {}", rank);
        test_encrypt_sk(&module, 8, 54, 30, 3.2, rank);
    });
}

#[test]
fn encrypt_zero_sk() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    (1..4).for_each(|rank| {
        println!("test encrypt_zero_sk rank: {}", rank);
        test_encrypt_zero_sk(&module, 8, 64, 3.2, rank);
    });
}

#[test]
fn encrypt_pk() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    (1..4).for_each(|rank| {
        println!("test encrypt_pk rank: {}", rank);
        test_encrypt_pk(&module, 8, 64, 64, 3.2, rank)
    });
}

fn test_encrypt_sk<B: Backend>(module: &Module<B>, basek: usize, k_ct: usize, k_pt: usize, sigma: f64, rank: usize)
where
    Module<B>: GLWEEncryptSkFamily<B> + GLWEDecryptFamily<B> + GLWESecretFamily<B>,
{
    let mut ct: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(module, basek, k_ct, rank);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(module, basek, k_pt);
    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(module, basek, k_pt);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWECiphertext::encrypt_sk_scratch_space(module, basek, ct.k())
            | GLWECiphertext::decrypt_scratch_space(module, basek, ct.k()),
    );

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(module, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let sk_exec: GLWESecretExec<Vec<u8>, B> = GLWESecretExec::from(module, &sk);

    module.vec_znx_fill_uniform(basek, &mut pt_want.data, 0, k_pt, &mut source_xa);

    ct.encrypt_sk(
        module,
        &pt_want,
        &sk_exec,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct.decrypt(module, &mut pt_have, &sk_exec, scratch.borrow());

    pt_want.sub_inplace_ab(module, &pt_have);

    let noise_have: f64 = module.vec_znx_std(basek, &pt_want.data, 0) * (ct.k() as f64).exp2();
    let noise_want: f64 = sigma;

    assert!(noise_have <= noise_want + 0.2);
}

fn test_encrypt_zero_sk<B: Backend>(module: &Module<B>, basek: usize, k_ct: usize, sigma: f64, rank: usize)
where
    Module<B>: GLWEEncryptSkFamily<B> + GLWEDecryptFamily<B> + GLWESecretFamily<B>,
{
    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(module, basek, k_ct);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([1u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(module, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let sk_exec: GLWESecretExec<Vec<u8>, B> = GLWESecretExec::from(module, &sk);

    let mut ct: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(module, basek, k_ct, rank);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWECiphertext::decrypt_scratch_space(module, basek, k_ct)
            | GLWECiphertext::encrypt_sk_scratch_space(module, basek, k_ct),
    );

    ct.encrypt_zero_sk(
        module,
        &sk_exec,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );
    ct.decrypt(module, &mut pt, &sk_exec, scratch.borrow());

    assert!((sigma - module.vec_znx_std(basek, &pt.data, 0) * (k_ct as f64).exp2()) <= 0.2);
}

fn test_encrypt_pk<B: Backend>(module: &Module<B>, basek: usize, k_ct: usize, k_pk: usize, sigma: f64, rank: usize)
where
    Module<B>: GLWEDecryptFamily<B> + GLWEPublicKeyFamily<B> + GLWESecretFamily<B> + GLWEEncryptPkFamily<B>,
{
    let mut ct: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(module, basek, k_ct, rank);
    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(module, basek, k_ct);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(module, basek, k_ct);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);
    let mut source_xu: Source = Source::new([0u8; 32]);

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(module, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let sk_exec: GLWESecretExec<Vec<u8>, B> = GLWESecretExec::from(module, &sk);

    let mut pk: GLWEPublicKey<Vec<u8>, B> = GLWEPublicKey::alloc(module, basek, k_pk, rank);
    pk.generate_from_sk(module, &sk_exec, &mut source_xa, &mut source_xe, sigma);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWECiphertext::encrypt_sk_scratch_space(module, basek, ct.k())
            | GLWECiphertext::decrypt_scratch_space(module, basek, ct.k())
            | GLWECiphertext::encrypt_pk_scratch_space(module, basek, pk.k()),
    );

    module.vec_znx_fill_uniform(basek, &mut pt_want.data, 0, k_ct, &mut source_xa);

    ct.encrypt_pk(
        module,
        &pt_want,
        &pk,
        &mut source_xu,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct.decrypt(module, &mut pt_have, &sk_exec, scratch.borrow());

    pt_want.sub_inplace_ab(module, &pt_have);

    let noise_have: f64 = module.vec_znx_std(basek, &pt_want.data, 0).log2();
    let noise_want: f64 = ((((rank as f64) + 1.0) * module.n() as f64 * 0.5 * sigma * sigma).sqrt()).log2() - (k_ct as f64);

    assert!(
        (noise_have - noise_want).abs() < 0.2,
        "{} {}",
        noise_have,
        noise_want
    );
}
