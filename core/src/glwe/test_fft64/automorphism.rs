use backend::{FFT64, FillUniform, Module, ScratchOwned, Stats, VecZnxOps};

use sampling::source::Source;

use crate::{
    FourierGLWESecret, GLWEAutomorphismKey, GLWECiphertext, GLWEPlaintext, GLWESecret, Infos, div_ceil,
    noise::log2_std_noise_gglwe_product,
};

#[test]
fn apply_inplace() {
    let log_n: usize = 8;
    let basek: usize = 12;
    let k_ct: usize = 60;
    let digits: usize = div_ceil(k_ct, basek);
    (1..4).for_each(|rank| {
        (1..digits + 1).for_each(|di| {
            let k_ksk: usize = k_ct + basek * di;
            println!("test automorphism_inplace digits: {} rank: {}", di, rank);
            test_automorphism_inplace(log_n, basek, -5, k_ct, k_ksk, di, rank, 3.2);
        });
    });
}

#[test]
fn apply() {
    let log_n: usize = 8;
    let basek: usize = 12;
    let k_in: usize = 60;
    let digits: usize = div_ceil(k_in, basek);
    (1..4).for_each(|rank| {
        (1..digits + 1).for_each(|di| {
            let k_ksk: usize = k_in + basek * di;
            let k_out: usize = k_ksk; // Better capture noise.
            println!("test automorphism digits: {} rank: {}", di, rank);
            test_automorphism(log_n, basek, -5, k_out, k_in, k_ksk, di, rank, 3.2);
        })
    });
}

fn test_automorphism(
    log_n: usize,
    basek: usize,
    p: i64,
    k_out: usize,
    k_in: usize,
    k_ksk: usize,
    digits: usize,
    rank: usize,
    sigma: f64,
) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);

    let rows: usize = div_ceil(k_in, basek * digits);

    let mut autokey: GLWEAutomorphismKey<Vec<u8>, FFT64> = GLWEAutomorphismKey::alloc(&module, basek, k_ksk, rows, digits, rank);
    let mut ct_in: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, k_in, rank);
    let mut ct_out: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, k_out, rank);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_in);
    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_out);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    pt_want
        .data
        .fill_uniform(basek, 0, pt_want.size(), &mut source_xa);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWEAutomorphismKey::generate_from_sk_scratch_space(&module, basek, autokey.k(), rank)
            | GLWECiphertext::decrypt_scratch_space(&module, basek, ct_out.k())
            | GLWECiphertext::encrypt_sk_scratch_space(&module, basek, ct_in.k())
            | GLWECiphertext::automorphism_scratch_space(
                &module,
                basek,
                ct_out.k(),
                ct_in.k(),
                autokey.k(),
                digits,
                rank,
            ),
    );

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(&module, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let sk_dft: FourierGLWESecret<Vec<u8>, FFT64> = FourierGLWESecret::from(&module, &sk);

    autokey.generate_from_sk(
        &module,
        p,
        &sk,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_in.encrypt_sk(
        &module,
        &pt_want,
        &sk_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_out.automorphism(&module, &ct_in, &autokey, scratch.borrow());
    ct_out.decrypt(&module, &mut pt_have, &sk_dft, scratch.borrow());
    module.vec_znx_automorphism_inplace(p, &mut pt_want.data, 0);
    module.vec_znx_sub_ab_inplace(&mut pt_have.data, 0, &pt_want.data, 0);
    module.vec_znx_normalize_inplace(basek, &mut pt_have.data, 0, scratch.borrow());

    let noise_have: f64 = pt_have.data.std(0, basek).log2();

    println!("{}", noise_have);

    let noise_want: f64 = log2_std_noise_gglwe_product(
        module.n() as f64,
        basek * digits,
        0.5,
        0.5,
        0f64,
        sigma * sigma,
        0f64,
        rank as f64,
        k_in,
        k_ksk,
    );

    assert!(
        noise_have <= noise_want + 1.0,
        "{} {}",
        noise_have,
        noise_want
    );
}

fn test_automorphism_inplace(
    log_n: usize,
    basek: usize,
    p: i64,
    k_ct: usize,
    k_ksk: usize,
    digits: usize,
    rank: usize,
    sigma: f64,
) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);

    let rows: usize = div_ceil(k_ct, basek * digits);

    let mut autokey: GLWEAutomorphismKey<Vec<u8>, FFT64> = GLWEAutomorphismKey::alloc(&module, basek, k_ksk, rows, digits, rank);
    let mut ct: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, k_ct, rank);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_ct);
    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_ct);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    pt_want
        .data
        .fill_uniform(basek, 0, pt_want.size(), &mut source_xa);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWEAutomorphismKey::generate_from_sk_scratch_space(&module, basek, autokey.k(), rank)
            | GLWECiphertext::decrypt_scratch_space(&module, basek, ct.k())
            | GLWECiphertext::encrypt_sk_scratch_space(&module, basek, ct.k())
            | GLWECiphertext::automorphism_inplace_scratch_space(&module, basek, ct.k(), autokey.k(), digits, rank),
    );

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(&module, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let sk_dft: FourierGLWESecret<Vec<u8>, FFT64> = FourierGLWESecret::from(&module, &sk);

    autokey.generate_from_sk(
        &module,
        p,
        &sk,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct.encrypt_sk(
        &module,
        &pt_want,
        &sk_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct.automorphism_inplace(&module, &autokey, scratch.borrow());
    ct.decrypt(&module, &mut pt_have, &sk_dft, scratch.borrow());
    module.vec_znx_automorphism_inplace(p, &mut pt_want.data, 0);
    module.vec_znx_sub_ab_inplace(&mut pt_have.data, 0, &pt_want.data, 0);
    module.vec_znx_normalize_inplace(basek, &mut pt_have.data, 0, scratch.borrow());

    let noise_have: f64 = pt_have.data.std(0, basek).log2();
    let noise_want: f64 = log2_std_noise_gglwe_product(
        module.n() as f64,
        basek * digits,
        0.5,
        0.5,
        0f64,
        sigma * sigma,
        0f64,
        rank as f64,
        k_ct,
        k_ksk,
    );

    assert!(
        (noise_have - noise_want).abs() <= 0.5,
        "{} {}",
        noise_have,
        noise_want
    );
}
