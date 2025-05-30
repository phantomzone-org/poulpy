use backend::{
    Decoding, Encoding, FFT64, FillUniform, Module, ScalarZnx, ScalarZnxAlloc, ScratchOwned, Stats, VecZnxOps, ZnxViewMut,
    ZnxZero,
};
use itertools::izip;
use sampling::source::Source;

use crate::{
    GGSWCiphertext, GLWECiphertext, GLWECiphertextFourier, GLWEPlaintext, GLWEPublicKey, GLWESecret, Infos,
    automorphism::AutomorphismKey,
    keyswitch_key::GLWESwitchingKey,
    test_fft64::{gglwe::log2_std_noise_gglwe_product, ggsw::noise_ggsw_product},
};

#[test]
fn encrypt_sk() {
    (1..4).for_each(|rank| {
        println!("test encrypt_sk rank: {}", rank);
        test_encrypt_sk(11, 8, 54, 30, 3.2, rank);
    });
}

#[test]
fn encrypt_zero_sk() {
    (1..4).for_each(|rank| {
        println!("test encrypt_zero_sk rank: {}", rank);
        test_encrypt_zero_sk(11, 8, 64, 3.2, rank);
    });
}

#[test]
fn encrypt_pk() {
    (1..4).for_each(|rank| {
        println!("test encrypt_pk rank: {}", rank);
        test_encrypt_pk(11, 8, 64, 64, 3.2, rank)
    });
}

#[test]
fn keyswitch() {
    (1..4).for_each(|in_rank| {
        (1..4).for_each(|out_rank| {
            println!("test keyswitch in_rank: {} out_rank: {}", in_rank, out_rank);
            test_keyswitch(12, 12, 60, 45, 60, in_rank, out_rank, 3.2);
        });
    });
}

#[test]
fn keyswitch_inplace() {
    (1..4).for_each(|rank| {
        println!("test keyswitch_inplace rank: {}", rank);
        test_keyswitch_inplace(12, 12, 60, 45, rank, 3.2);
    });
}

#[test]
fn external_product() {
    (1..4).for_each(|rank| {
        println!("test external_product rank: {}", rank);
        test_external_product(12, 12, 60, 45, 60, rank, 3.2);
    });
}

#[test]
fn external_product_inplace() {
    (1..4).for_each(|rank| {
        println!("test external_product rank: {}", rank);
        test_external_product_inplace(12, 15, 60, 60, rank, 3.2);
    });
}

#[test]
fn automorphism_inplace() {
    (1..4).for_each(|rank| {
        println!("test automorphism_inplace rank: {}", rank);
        test_automorphism_inplace(12, 12, -5, 60, 60, rank, 3.2);
    });
}

#[test]
fn automorphism() {
    (1..4).for_each(|rank| {
        println!("test automorphism rank: {}", rank);
        test_automorphism(12, 12, -5, 60, 45, 60, rank, 3.2);
    });
}

fn test_encrypt_sk(log_n: usize, basek: usize, ct_k: usize, k_pt: usize, sigma: f64, rank: usize) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);

    let mut ct: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, ct_k, rank);
    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_pt);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWECiphertext::encrypt_sk_scratch_space(&module, basek, ct.k())
            | GLWECiphertext::decrypt_scratch_space(&module, basek, ct.k()),
    );

    let mut sk: GLWESecret<Vec<u8>, FFT64> = GLWESecret::alloc(&module, rank);
    sk.fill_ternary_prob(&module, 0.5, &mut source_xs);

    let mut data_want: Vec<i64> = vec![0i64; module.n()];

    data_want
        .iter_mut()
        .for_each(|x| *x = source_xa.next_i64() & 0xFF);

    pt.data.encode_vec_i64(0, basek, k_pt, &data_want, 10);

    ct.encrypt_sk(
        &module,
        &pt,
        &sk,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    pt.data.zero();

    ct.decrypt(&module, &mut pt, &sk, scratch.borrow());

    let mut data_have: Vec<i64> = vec![0i64; module.n()];

    pt.data
        .decode_vec_i64(0, basek, pt.size() * basek, &mut data_have);

    // TODO: properly assert the decryption noise through std(dec(ct) - pt)
    let scale: f64 = (1 << (pt.size() * basek - k_pt)) as f64;
    izip!(data_want.iter(), data_have.iter()).for_each(|(a, b)| {
        let b_scaled = (*b as f64) / scale;
        assert!(
            (*a as f64 - b_scaled).abs() < 0.1,
            "{} {}",
            *a as f64,
            b_scaled
        )
    });
}

fn test_encrypt_zero_sk(log_n: usize, basek: usize, ct_k: usize, sigma: f64, rank: usize) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);

    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, ct_k);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([1u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut sk: GLWESecret<Vec<u8>, FFT64> = GLWESecret::alloc(&module, rank);
    sk.fill_ternary_prob(&module, 0.5, &mut source_xs);

    let mut ct_dft: GLWECiphertextFourier<Vec<u8>, FFT64> = GLWECiphertextFourier::alloc(&module, basek, ct_k, rank);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWECiphertextFourier::decrypt_scratch_space(&module, basek, ct_k)
            | GLWECiphertextFourier::encrypt_sk_scratch_space(&module, basek, ct_k, rank),
    );

    ct_dft.encrypt_zero_sk(
        &module,
        &sk,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );
    ct_dft.decrypt(&module, &mut pt, &sk, scratch.borrow());

    assert!((sigma - pt.data.std(0, basek) * (ct_k as f64).exp2()) <= 0.2);
}

fn test_encrypt_pk(log_n: usize, basek: usize, ct_k: usize, k_pk: usize, sigma: f64, rank: usize) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);

    let mut ct: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, ct_k, rank);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, ct_k);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);
    let mut source_xu: Source = Source::new([0u8; 32]);

    let mut sk: GLWESecret<Vec<u8>, FFT64> = GLWESecret::alloc(&module, rank);
    sk.fill_ternary_prob(&module, 0.5, &mut source_xs);

    let mut pk: GLWEPublicKey<Vec<u8>, FFT64> = GLWEPublicKey::alloc(&module, basek, k_pk, rank);
    pk.generate_from_sk(&module, &sk, &mut source_xa, &mut source_xe, sigma);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWECiphertext::encrypt_sk_scratch_space(&module, basek, ct.k())
            | GLWECiphertext::decrypt_scratch_space(&module, basek, ct.k())
            | GLWECiphertext::encrypt_pk_scratch_space(&module, basek, pk.k()),
    );

    let mut data_want: Vec<i64> = vec![0i64; module.n()];

    data_want
        .iter_mut()
        .for_each(|x| *x = source_xa.next_i64() & 0);

    pt_want.data.encode_vec_i64(0, basek, ct_k, &data_want, 10);

    ct.encrypt_pk(
        &module,
        &pt_want,
        &pk,
        &mut source_xu,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, ct_k);

    ct.decrypt(&module, &mut pt_have, &sk, scratch.borrow());

    module.vec_znx_sub_ab_inplace(&mut pt_want.data, 0, &pt_have.data, 0);

    let noise_have: f64 = pt_want.data.std(0, basek).log2();
    let noise_want: f64 = ((((rank as f64) + 1.0) * module.n() as f64 * 0.5 * sigma * sigma).sqrt()).log2() - (ct_k as f64);

    assert!(
        (noise_have - noise_want).abs() < 0.2,
        "{} {}",
        noise_have,
        noise_want
    );
}

fn test_keyswitch(
    log_n: usize,
    basek: usize,
    k_keyswitch: usize,
    ct_k_in: usize,
    ct_k_out: usize,
    in_rank: usize,
    out_rank: usize,
    sigma: f64,
) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let rows: usize = (ct_k_in + basek - 1) / basek;

    let mut ksk: GLWESwitchingKey<Vec<u8>, FFT64> = GLWESwitchingKey::alloc(&module, basek, k_keyswitch, rows, in_rank, out_rank);
    let mut ct_in: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, ct_k_in, in_rank);
    let mut ct_out: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, ct_k_out, out_rank);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, ct_k_in);
    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, ct_k_out);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    // Random input plaintext
    pt_want
        .data
        .fill_uniform(basek, 0, pt_want.size(), &mut source_xa);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWESwitchingKey::encrypt_sk_scratch_space(&module, basek, ksk.k(), out_rank)
            | GLWECiphertext::decrypt_scratch_space(&module, basek, ct_out.k())
            | GLWECiphertext::encrypt_sk_scratch_space(&module, basek, ct_in.k())
            | GLWECiphertext::keyswitch_scratch_space(
                &module,
                basek,
                ct_out.k(),
                out_rank,
                ct_in.k(),
                in_rank,
                ksk.k(),
            ),
    );

    let mut sk_in: GLWESecret<Vec<u8>, FFT64> = GLWESecret::alloc(&module, in_rank);
    sk_in.fill_ternary_prob(&module, 0.5, &mut source_xs);

    let mut sk_out: GLWESecret<Vec<u8>, FFT64> = GLWESecret::alloc(&module, out_rank);
    sk_out.fill_ternary_prob(&module, 0.5, &mut source_xs);

    ksk.generate_from_sk(
        &module,
        &sk_in,
        &sk_out,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_in.encrypt_sk(
        &module,
        &pt_want,
        &sk_in,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_out.keyswitch(&module, &ct_in, &ksk, scratch.borrow());

    ct_out.decrypt(&module, &mut pt_have, &sk_out, scratch.borrow());

    module.vec_znx_sub_ab_inplace(&mut pt_have.data, 0, &pt_want.data, 0);

    let noise_have: f64 = pt_have.data.std(0, basek).log2();
    let noise_want: f64 = log2_std_noise_gglwe_product(
        module.n() as f64,
        basek,
        0.5,
        0.5,
        0f64,
        sigma * sigma,
        0f64,
        in_rank as f64,
        ct_k_in,
        k_keyswitch,
    );

    assert!(
        (noise_have - noise_want).abs() <= 0.1,
        "{} {}",
        noise_have,
        noise_want
    );
}

fn test_keyswitch_inplace(log_n: usize, basek: usize, k_ksk: usize, ct_k: usize, rank: usize, sigma: f64) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let rows: usize = (ct_k + basek - 1) / basek;

    let mut ct_grlwe: GLWESwitchingKey<Vec<u8>, FFT64> = GLWESwitchingKey::alloc(&module, basek, k_ksk, rows, rank, rank);
    let mut ct_glwe: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, ct_k, rank);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, ct_k);
    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, ct_k);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    // Random input plaintext
    pt_want
        .data
        .fill_uniform(basek, 0, pt_want.size(), &mut source_xa);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWESwitchingKey::encrypt_sk_scratch_space(&module, basek, ct_grlwe.k(), rank)
            | GLWECiphertext::decrypt_scratch_space(&module, basek, ct_glwe.k())
            | GLWECiphertext::encrypt_sk_scratch_space(&module, basek, ct_glwe.k())
            | GLWECiphertext::keyswitch_inplace_scratch_space(&module, basek, ct_glwe.k(), rank, ct_grlwe.k()),
    );

    let mut sk0: GLWESecret<Vec<u8>, FFT64> = GLWESecret::alloc(&module, rank);
    sk0.fill_ternary_prob(&module, 0.5, &mut source_xs);

    let mut sk1: GLWESecret<Vec<u8>, FFT64> = GLWESecret::alloc(&module, rank);
    sk1.fill_ternary_prob(&module, 0.5, &mut source_xs);

    ct_grlwe.generate_from_sk(
        &module,
        &sk0,
        &sk1,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_glwe.encrypt_sk(
        &module,
        &pt_want,
        &sk0,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_glwe.keyswitch_inplace(&module, &ct_grlwe, scratch.borrow());

    ct_glwe.decrypt(&module, &mut pt_have, &sk1, scratch.borrow());

    module.vec_znx_sub_ab_inplace(&mut pt_have.data, 0, &pt_want.data, 0);

    let noise_have: f64 = pt_have.data.std(0, basek).log2();
    let noise_want: f64 = log2_std_noise_gglwe_product(
        module.n() as f64,
        basek,
        0.5,
        0.5,
        0f64,
        sigma * sigma,
        0f64,
        rank as f64,
        ct_k,
        k_ksk,
    );

    assert!(
        (noise_have - noise_want).abs() <= 0.1,
        "{} {}",
        noise_have,
        noise_want
    );
}

fn test_automorphism(
    log_n: usize,
    basek: usize,
    p: i64,
    k_autokey: usize,
    ct_k_in: usize,
    ct_k_out: usize,
    rank: usize,
    sigma: f64,
) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let rows: usize = (ct_k_in + basek - 1) / basek;

    let mut autokey: AutomorphismKey<Vec<u8>, FFT64> = AutomorphismKey::alloc(&module, basek, k_autokey, rows, rank);
    let mut ct_in: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, ct_k_in, rank);
    let mut ct_out: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, ct_k_out, rank);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, ct_k_in);
    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, ct_k_out);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    pt_want
        .data
        .fill_uniform(basek, 0, pt_want.size(), &mut source_xa);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        AutomorphismKey::generate_from_sk_scratch_space(&module, basek, autokey.k(), rank)
            | GLWECiphertext::decrypt_scratch_space(&module, basek, ct_out.k())
            | GLWECiphertext::encrypt_sk_scratch_space(&module, basek, ct_in.k())
            | GLWECiphertext::automorphism_scratch_space(&module, basek, ct_out.k(), ct_in.k(), autokey.k(), rank),
    );

    let mut sk: GLWESecret<Vec<u8>, FFT64> = GLWESecret::alloc(&module, rank);
    sk.fill_ternary_prob(&module, 0.5, &mut source_xs);

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
        &sk,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_out.automorphism(&module, &ct_in, &autokey, scratch.borrow());
    ct_out.decrypt(&module, &mut pt_have, &sk, scratch.borrow());
    module.vec_znx_automorphism_inplace(p, &mut pt_want.data, 0);
    module.vec_znx_sub_ab_inplace(&mut pt_have.data, 0, &pt_want.data, 0);
    module.vec_znx_normalize_inplace(basek, &mut pt_have.data, 0, scratch.borrow());

    let noise_have: f64 = pt_have.data.std(0, basek).log2();

    println!("{}", noise_have);

    let noise_want: f64 = log2_std_noise_gglwe_product(
        module.n() as f64,
        basek,
        0.5,
        0.5,
        0f64,
        sigma * sigma,
        0f64,
        rank as f64,
        ct_k_in,
        k_autokey,
    );

    assert!(
        (noise_have - noise_want).abs() <= 0.1,
        "{} {}",
        noise_have,
        noise_want
    );
}

fn test_automorphism_inplace(log_n: usize, basek: usize, p: i64, k_autokey: usize, ct_k: usize, rank: usize, sigma: f64) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let rows: usize = (ct_k + basek - 1) / basek;

    let mut autokey: AutomorphismKey<Vec<u8>, FFT64> = AutomorphismKey::alloc(&module, basek, k_autokey, rows, rank);
    let mut ct: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, ct_k, rank);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, ct_k);
    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, ct_k);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    // Random input plaintext
    pt_want
        .data
        .fill_uniform(basek, 0, pt_want.size(), &mut source_xa);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        AutomorphismKey::generate_from_sk_scratch_space(&module, basek, autokey.k(), rank)
            | GLWECiphertext::decrypt_scratch_space(&module, basek, ct.k())
            | GLWECiphertext::encrypt_sk_scratch_space(&module, basek, ct.k())
            | GLWECiphertext::automorphism_inplace_scratch_space(&module, basek, ct.k(), autokey.k(), rank),
    );

    let mut sk: GLWESecret<Vec<u8>, FFT64> = GLWESecret::alloc(&module, rank);
    sk.fill_ternary_prob(&module, 0.5, &mut source_xs);

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
        &sk,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct.automorphism_inplace(&module, &autokey, scratch.borrow());
    ct.decrypt(&module, &mut pt_have, &sk, scratch.borrow());
    module.vec_znx_automorphism_inplace(p, &mut pt_want.data, 0);
    module.vec_znx_sub_ab_inplace(&mut pt_have.data, 0, &pt_want.data, 0);
    module.vec_znx_normalize_inplace(basek, &mut pt_have.data, 0, scratch.borrow());

    let noise_have: f64 = pt_have.data.std(0, basek).log2();
    let noise_want: f64 = log2_std_noise_gglwe_product(
        module.n() as f64,
        basek,
        0.5,
        0.5,
        0f64,
        sigma * sigma,
        0f64,
        rank as f64,
        ct_k,
        k_autokey,
    );

    assert!(
        (noise_have - noise_want).abs() <= 0.1,
        "{} {}",
        noise_have,
        noise_want
    );
}

fn test_external_product(log_n: usize, basek: usize, k_ggsw: usize, ct_k_in: usize, ct_k_out: usize, rank: usize, sigma: f64) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);

    let rows: usize = (ct_k_in + basek - 1) / basek;

    let mut ct_ggsw: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::alloc(&module, basek, k_ggsw, rows, rank);
    let mut ct_glwe_in: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, ct_k_in, rank);
    let mut ct_glwe_out: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, ct_k_out, rank);
    let mut pt_rgsw: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, ct_k_in);
    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, ct_k_out);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    // Random input plaintext
    pt_want
        .data
        .fill_uniform(basek, 0, pt_want.size(), &mut source_xa);

    pt_want.data.at_mut(0, 0)[1] = 1;

    let k: usize = 1;

    pt_rgsw.raw_mut()[k] = 1; // X^{k}

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GGSWCiphertext::encrypt_sk_scratch_space(&module, basek, ct_ggsw.k(), rank)
            | GLWECiphertext::decrypt_scratch_space(&module, basek, ct_glwe_out.k())
            | GLWECiphertext::encrypt_sk_scratch_space(&module, basek, ct_glwe_in.k())
            | GLWECiphertext::external_product_scratch_space(
                &module,
                basek,
                ct_glwe_out.k(),
                ct_glwe_in.k(),
                ct_ggsw.k(),
                rank,
            ),
    );

    let mut sk: GLWESecret<Vec<u8>, FFT64> = GLWESecret::alloc(&module, rank);
    sk.fill_ternary_prob(&module, 0.5, &mut source_xs);

    ct_ggsw.encrypt_sk(
        &module,
        &pt_rgsw,
        &sk,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_glwe_in.encrypt_sk(
        &module,
        &pt_want,
        &sk,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_glwe_out.external_product(&module, &ct_glwe_in, &ct_ggsw, scratch.borrow());

    ct_glwe_out.decrypt(&module, &mut pt_have, &sk, scratch.borrow());

    module.vec_znx_rotate_inplace(k as i64, &mut pt_want.data, 0);

    module.vec_znx_sub_ab_inplace(&mut pt_have.data, 0, &pt_want.data, 0);

    let noise_have: f64 = pt_have.data.std(0, basek).log2();

    let var_gct_err_lhs: f64 = sigma * sigma;
    let var_gct_err_rhs: f64 = 0f64;

    let var_msg: f64 = 1f64 / module.n() as f64; // X^{k}
    let var_a0_err: f64 = sigma * sigma;
    let var_a1_err: f64 = 1f64 / 12f64;

    let noise_want: f64 = noise_ggsw_product(
        module.n() as f64,
        basek,
        0.5,
        var_msg,
        var_a0_err,
        var_a1_err,
        var_gct_err_lhs,
        var_gct_err_rhs,
        rank as f64,
        ct_k_in,
        k_ggsw,
    );

    assert!(
        (noise_have - noise_want).abs() <= 0.1,
        "{} {}",
        noise_have,
        noise_want
    );
}

fn test_external_product_inplace(log_n: usize, basek: usize, k_ggsw: usize, ct_k: usize, rank: usize, sigma: f64) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let rows: usize = (ct_k + basek - 1) / basek;

    let mut ct_ggsw: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::alloc(&module, basek, k_ggsw, rows, rank);
    let mut ct_glwe: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, ct_k, rank);
    let mut pt_rgsw: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, ct_k);
    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, ct_k);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    // Random input plaintext
    pt_want
        .data
        .fill_uniform(basek, 0, pt_want.size(), &mut source_xa);

    pt_want.data.at_mut(0, 0)[1] = 1;

    let k: usize = 1;

    pt_rgsw.raw_mut()[k] = 1; // X^{k}

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GGSWCiphertext::encrypt_sk_scratch_space(&module, basek, ct_ggsw.k(), rank)
            | GLWECiphertext::decrypt_scratch_space(&module, basek, ct_glwe.k())
            | GLWECiphertext::encrypt_sk_scratch_space(&module, basek, ct_glwe.k())
            | GLWECiphertext::external_product_inplace_scratch_space(&module, basek, ct_glwe.k(), ct_ggsw.k(), rank),
    );

    let mut sk: GLWESecret<Vec<u8>, FFT64> = GLWESecret::alloc(&module, rank);
    sk.fill_ternary_prob(&module, 0.5, &mut source_xs);

    ct_ggsw.encrypt_sk(
        &module,
        &pt_rgsw,
        &sk,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_glwe.encrypt_sk(
        &module,
        &pt_want,
        &sk,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_glwe.external_product_inplace(&module, &ct_ggsw, scratch.borrow());

    ct_glwe.decrypt(&module, &mut pt_have, &sk, scratch.borrow());

    module.vec_znx_rotate_inplace(k as i64, &mut pt_want.data, 0);

    module.vec_znx_sub_ab_inplace(&mut pt_have.data, 0, &pt_want.data, 0);

    let noise_have: f64 = pt_have.data.std(0, basek).log2();

    let var_gct_err_lhs: f64 = sigma * sigma;
    let var_gct_err_rhs: f64 = 0f64;

    let var_msg: f64 = 1f64 / module.n() as f64; // X^{k}
    let var_a0_err: f64 = sigma * sigma;
    let var_a1_err: f64 = 1f64 / 12f64;

    let noise_want: f64 = noise_ggsw_product(
        module.n() as f64,
        basek,
        0.5,
        var_msg,
        var_a0_err,
        var_a1_err,
        var_gct_err_lhs,
        var_gct_err_rhs,
        rank as f64,
        ct_k,
        k_ggsw,
    );

    assert!(
        (noise_have - noise_want).abs() <= 0.1,
        "{} {}",
        noise_have,
        noise_want
    );
}
