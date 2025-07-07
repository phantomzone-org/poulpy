use crate::{
    FourierGLWECiphertext, FourierGLWESecret, GLWECiphertext, GLWEPlaintext, GLWESecret, GLWESwitchingKey, Infos,
    noise::log2_std_noise_gglwe_product,
};
use backend::{FFT64, FillUniform, Module, ScratchOwned, Stats, VecZnxOps};
use sampling::source::Source;

#[test]
fn apply() {
    let log_n: usize = 8;
    let basek: usize = 12;
    let k_in: usize = 45;
    let digits: usize = k_in.div_ceil(basek);
    (1..4).for_each(|rank_in| {
        (1..4).for_each(|rank_out| {
            (1..digits + 1).for_each(|di| {
                let k_ksk: usize = k_in + basek * di;
                println!(
                    "test keyswitch digits: {} rank_in: {} rank_out: {}",
                    di, rank_in, rank_out
                );
                let k_out: usize = k_ksk; // Better capture noise.
                test_apply(log_n, basek, k_in, k_out, k_ksk, di, rank_in, rank_out, 3.2);
            })
        });
    });
}

#[test]
fn apply_inplace() {
    let log_n: usize = 8;
    let basek: usize = 12;
    let k_ct: usize = 45;
    let digits: usize = k_ct.div_ceil(basek);
    (1..4).for_each(|rank| {
        (1..digits + 1).for_each(|di| {
            let k_ksk: usize = k_ct + basek * di;
            println!("test keyswitch_inplace digits: {} rank: {}", di, rank);
            test_apply_inplace(log_n, basek, k_ct, k_ksk, di, rank, 3.2);
        });
    });
}

fn test_apply(
    log_n: usize,
    basek: usize,
    k_in: usize,
    k_out: usize,
    k_ksk: usize,
    digits: usize,
    rank_in: usize,
    rank_out: usize,
    sigma: f64,
) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);

    let rows: usize = k_in.div_ceil(basek * digits);

    let mut ksk: GLWESwitchingKey<Vec<u8>, FFT64> =
        GLWESwitchingKey::alloc(&module, basek, k_ksk, rows, digits, rank_in, rank_out);
    let mut ct_glwe_in: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, k_in, rank_in);
    let mut ct_glwe_dft_in: FourierGLWECiphertext<Vec<u8>, FFT64> = FourierGLWECiphertext::alloc(&module, basek, k_in, rank_in);
    let mut ct_glwe_out: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, k_out, rank_out);
    let mut ct_glwe_dft_out: FourierGLWECiphertext<Vec<u8>, FFT64> =
        FourierGLWECiphertext::alloc(&module, basek, k_out, rank_out);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_in);
    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_out);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    // Random input plaintext
    pt_want
        .data
        .fill_uniform(basek, 0, pt_want.size(), &mut source_xa);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWESwitchingKey::encrypt_sk_scratch_space(&module, basek, k_ksk, rank_in, rank_out)
            | GLWECiphertext::decrypt_scratch_space(&module, basek, k_out)
            | GLWECiphertext::encrypt_sk_scratch_space(&module, basek, k_in)
            | FourierGLWECiphertext::keyswitch_scratch_space(
                &module,
                basek,
                ct_glwe_out.k(),
                ksk.k(),
                ct_glwe_in.k(),
                digits,
                rank_in,
                rank_out,
            ),
    );

    let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc(&module, rank_in);
    sk_in.fill_ternary_prob(0.5, &mut source_xs);
    let sk_in_dft: FourierGLWESecret<Vec<u8>, FFT64> = FourierGLWESecret::from(&module, &sk_in);

    let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc(&module, rank_out);
    sk_out.fill_ternary_prob(0.5, &mut source_xs);
    let sk_out_dft: FourierGLWESecret<Vec<u8>, FFT64> = FourierGLWESecret::from(&module, &sk_out);

    ksk.encrypt_sk(
        &module,
        &sk_in,
        &sk_out_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_glwe_in.encrypt_sk(
        &module,
        &pt_want,
        &sk_in_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_glwe_in.dft(&module, &mut ct_glwe_dft_in);
    ct_glwe_dft_out.keyswitch(&module, &ct_glwe_dft_in, &ksk, scratch.borrow());
    ct_glwe_dft_out.idft(&module, &mut ct_glwe_out, scratch.borrow());

    ct_glwe_out.decrypt(&module, &mut pt_have, &sk_out_dft, scratch.borrow());

    module.vec_znx_sub_ab_inplace(&mut pt_have.data, 0, &pt_want.data, 0);

    let noise_have: f64 = pt_have.data.std(0, basek).log2();
    let noise_want: f64 = log2_std_noise_gglwe_product(
        module.n() as f64,
        basek * digits,
        0.5,
        0.5,
        0f64,
        sigma * sigma,
        0f64,
        rank_in as f64,
        k_in,
        k_ksk,
    );

    assert!(
        (noise_have - noise_want).abs() <= 0.5,
        "{} {}",
        noise_have,
        noise_want
    );
}

fn test_apply_inplace(log_n: usize, basek: usize, k_ct: usize, k_ksk: usize, digits: usize, rank: usize, sigma: f64) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);

    let rows: usize = k_ct.div_ceil(basek * digits);

    let mut ksk: GLWESwitchingKey<Vec<u8>, FFT64> = GLWESwitchingKey::alloc(&module, basek, k_ksk, rows, digits, rank, rank);
    let mut ct_glwe: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, k_ct, rank);
    let mut ct_rlwe_dft: FourierGLWECiphertext<Vec<u8>, FFT64> = FourierGLWECiphertext::alloc(&module, basek, k_ct, rank);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_ct);
    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_ct);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    // Random input plaintext
    pt_want
        .data
        .fill_uniform(basek, 0, pt_want.size(), &mut source_xa);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWESwitchingKey::encrypt_sk_scratch_space(&module, basek, ksk.k(), rank, rank)
            | GLWECiphertext::decrypt_scratch_space(&module, basek, ct_glwe.k())
            | GLWECiphertext::encrypt_sk_scratch_space(&module, basek, ct_glwe.k())
            | FourierGLWECiphertext::keyswitch_inplace_scratch_space(&module, basek, ct_rlwe_dft.k(), ksk.k(), digits, rank),
    );

    let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc(&module, rank);
    sk_in.fill_ternary_prob(0.5, &mut source_xs);
    let sk_in_dft: FourierGLWESecret<Vec<u8>, FFT64> = FourierGLWESecret::from(&module, &sk_in);

    let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc(&module, rank);
    sk_out.fill_ternary_prob(0.5, &mut source_xs);
    let sk_out_dft: FourierGLWESecret<Vec<u8>, FFT64> = FourierGLWESecret::from(&module, &sk_out);

    ksk.encrypt_sk(
        &module,
        &sk_in,
        &sk_out_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_glwe.encrypt_sk(
        &module,
        &pt_want,
        &sk_in_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_glwe.dft(&module, &mut ct_rlwe_dft);
    ct_rlwe_dft.keyswitch_inplace(&module, &ksk, scratch.borrow());
    ct_rlwe_dft.idft(&module, &mut ct_glwe, scratch.borrow());

    ct_glwe.decrypt(&module, &mut pt_have, &sk_out_dft, scratch.borrow());

    module.vec_znx_sub_ab_inplace(&mut pt_have.data, 0, &pt_want.data, 0);

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
