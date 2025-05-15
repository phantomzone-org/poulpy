use base2k::{
    Decoding, Encoding, FFT64, FillUniform, Module, ScalarZnx, ScalarZnxAlloc, ScratchOwned, Stats, VecZnxOps, VecZnxToMut,
    ZnxViewMut, ZnxZero,
};
use itertools::izip;
use sampling::source::Source;

use crate::{
    elem::Infos,
    ggsw_ciphertext::GGSWCiphertext,
    glwe_ciphertext::GLWECiphertext,
    glwe_ciphertext_fourier::GLWECiphertextFourier,
    glwe_plaintext::GLWEPlaintext,
    keys::{GLWEPublicKey, SecretKey, SecretKeyFourier},
    keyswitch_key::GLWESwitchingKey,
    test_fft64::{gglwe::noise_gglwe_product, ggsw::noise_ggsw_product},
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
    (1..4).for_each(|rank_in| {
        (1..4).for_each(|rank_out| {
            println!("test keyswitch rank_in: {} rank_out: {}", rank_in, rank_out);
            test_keyswitch(12, 12, 60, 45, 60, rank_in, rank_out, 3.2);
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

fn test_encrypt_sk(log_n: usize, basek: usize, k_ct: usize, k_pt: usize, sigma: f64, rank: usize) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);

    let mut ct: GLWECiphertext<Vec<u8>> = GLWECiphertext::new(&module, basek, k_ct, rank);
    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::new(&module, basek, k_pt);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWECiphertext::encrypt_sk_scratch_space(&module, ct.size()) | GLWECiphertext::decrypt_scratch_space(&module, ct.size()),
    );

    let mut sk: SecretKey<Vec<u8>> = SecretKey::new(&module, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::new(&module, rank);
    sk_dft.dft(&module, &sk);

    let mut data_want: Vec<i64> = vec![0i64; module.n()];

    data_want
        .iter_mut()
        .for_each(|x| *x = source_xa.next_i64() & 0xFF);

    pt.data.encode_vec_i64(0, basek, k_pt, &data_want, 10);

    ct.encrypt_sk(
        &module,
        &pt,
        &sk_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    pt.data.zero();

    ct.decrypt(&module, &mut pt, &sk_dft, scratch.borrow());

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

fn test_encrypt_zero_sk(log_n: usize, basek: usize, k_ct: usize, sigma: f64, rank: usize) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);

    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::new(&module, basek, k_ct);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([1u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut sk: SecretKey<Vec<u8>> = SecretKey::new(&module, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let mut sk_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::new(&module, rank);
    sk_dft.dft(&module, &sk);

    let mut ct_dft: GLWECiphertextFourier<Vec<u8>, FFT64> = GLWECiphertextFourier::new(&module, basek, k_ct, rank);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWECiphertextFourier::decrypt_scratch_space(&module, ct_dft.size())
            | GLWECiphertextFourier::encrypt_sk_scratch_space(&module, rank, ct_dft.size()),
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

    let mut ct: GLWECiphertext<Vec<u8>> = GLWECiphertext::new(&module, basek, k_ct, rank);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::new(&module, basek, k_ct);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);
    let mut source_xu: Source = Source::new([0u8; 32]);

    let mut sk: SecretKey<Vec<u8>> = SecretKey::new(&module, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let mut sk_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::new(&module, rank);
    sk_dft.dft(&module, &sk);

    let mut pk: GLWEPublicKey<Vec<u8>, FFT64> = GLWEPublicKey::new(&module, basek, k_pk, rank);
    pk.generate(&module, &sk_dft, &mut source_xa, &mut source_xe, sigma);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWECiphertext::encrypt_sk_scratch_space(&module, ct.size())
            | GLWECiphertext::decrypt_scratch_space(&module, ct.size())
            | GLWECiphertext::encrypt_pk_scratch_space(&module, pk.size()),
    );

    let mut data_want: Vec<i64> = vec![0i64; module.n()];

    data_want
        .iter_mut()
        .for_each(|x| *x = source_xa.next_i64() & 0);

    pt_want.data.encode_vec_i64(0, basek, k_ct, &data_want, 10);

    ct.encrypt_pk(
        &module,
        &pt_want,
        &pk,
        &mut source_xu,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::new(&module, basek, k_ct);

    ct.decrypt(&module, &mut pt_have, &sk_dft, scratch.borrow());

    module.vec_znx_sub_ab_inplace(&mut pt_want, 0, &pt_have, 0);

    let noise_have: f64 = pt_want.data.std(0, basek).log2();
    let noise_want: f64 = ((((rank as f64) + 1.0) * module.n() as f64 * 0.5 * sigma * sigma).sqrt()).log2() - (k_ct as f64);

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
    k_ct_in: usize,
    k_ct_out: usize,
    rank_in: usize,
    rank_out: usize,
    sigma: f64,
) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let rows: usize = (k_ct_in + basek - 1) / basek;

    let mut ksk: GLWESwitchingKey<Vec<u8>, FFT64> = GLWESwitchingKey::new(&module, basek, k_keyswitch, rows, rank_in, rank_out);
    let mut ct_in: GLWECiphertext<Vec<u8>> = GLWECiphertext::new(&module, basek, k_ct_in, rank_in);
    let mut ct_out: GLWECiphertext<Vec<u8>> = GLWECiphertext::new(&module, basek, k_ct_out, rank_out);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::new(&module, basek, k_ct_in);
    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::new(&module, basek, k_ct_out);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    // Random input plaintext
    pt_want
        .data
        .fill_uniform(basek, 0, pt_want.size(), &mut source_xa);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWESwitchingKey::encrypt_sk_scratch_space(&module, rank_in, ksk.size())
            | GLWECiphertext::decrypt_scratch_space(&module, ct_out.size())
            | GLWECiphertext::encrypt_sk_scratch_space(&module, ct_in.size())
            | GLWECiphertext::keyswitch_scratch_space(
                &module,
                ct_out.size(),
                ct_in.size(),
                ksk.size(),
                rank_in,
                rank_out,
            ),
    );

    let mut sk_in: SecretKey<Vec<u8>> = SecretKey::new(&module, rank_in);
    sk_in.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_in_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::new(&module, rank_in);
    sk_in_dft.dft(&module, &sk_in);

    let mut sk_out: SecretKey<Vec<u8>> = SecretKey::new(&module, rank_out);
    sk_out.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_out_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::new(&module, rank_out);
    sk_out_dft.dft(&module, &sk_out);

    ksk.encrypt_sk(
        &module,
        &sk_in,
        &sk_out_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_in.encrypt_sk(
        &module,
        &pt_want,
        &sk_in_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_out.keyswitch(&module, &ct_in, &ksk, scratch.borrow());

    ct_out.decrypt(&module, &mut pt_have, &sk_out_dft, scratch.borrow());

    module.vec_znx_sub_ab_inplace(&mut pt_have, 0, &pt_want, 0);

    let noise_have: f64 = pt_have.data.std(0, basek).log2();
    let noise_want: f64 = noise_gglwe_product(
        module.n() as f64,
        basek,
        0.5,
        0.5,
        0f64,
        sigma * sigma,
        0f64,
        rank_in as f64,
        k_ct_in,
        k_keyswitch,
    );

    assert!(
        (noise_have - noise_want).abs() <= 0.1,
        "{} {}",
        noise_have,
        noise_want
    );
}

fn test_keyswitch_inplace(log_n: usize, basek: usize, k_ksk: usize, k_ct: usize, rank: usize, sigma: f64) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let rows: usize = (k_ct + basek - 1) / basek;

    let mut ct_grlwe: GLWESwitchingKey<Vec<u8>, FFT64> = GLWESwitchingKey::new(&module, basek, k_ksk, rows, rank, rank);
    let mut ct_rlwe: GLWECiphertext<Vec<u8>> = GLWECiphertext::new(&module, basek, k_ct, rank);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::new(&module, basek, k_ct);
    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::new(&module, basek, k_ct);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    // Random input plaintext
    pt_want
        .data
        .fill_uniform(basek, 0, pt_want.size(), &mut source_xa);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWESwitchingKey::encrypt_sk_scratch_space(&module, rank, ct_grlwe.size())
            | GLWECiphertext::decrypt_scratch_space(&module, ct_rlwe.size())
            | GLWECiphertext::encrypt_sk_scratch_space(&module, ct_rlwe.size())
            | GLWECiphertext::keyswitch_inplace_scratch_space(&module, ct_rlwe.size(), ct_grlwe.size(), rank),
    );

    let mut sk0: SecretKey<Vec<u8>> = SecretKey::new(&module, rank);
    sk0.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk0_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::new(&module, rank);
    sk0_dft.dft(&module, &sk0);

    let mut sk1: SecretKey<Vec<u8>> = SecretKey::new(&module, rank);
    sk1.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk1_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::new(&module, rank);
    sk1_dft.dft(&module, &sk1);

    ct_grlwe.encrypt_sk(
        &module,
        &sk0,
        &sk1_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_rlwe.encrypt_sk(
        &module,
        &pt_want,
        &sk0_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_rlwe.keyswitch_inplace(&module, &ct_grlwe, scratch.borrow());

    ct_rlwe.decrypt(&module, &mut pt_have, &sk1_dft, scratch.borrow());

    module.vec_znx_sub_ab_inplace(&mut pt_have, 0, &pt_want, 0);

    let noise_have: f64 = pt_have.data.std(0, basek).log2();
    let noise_want: f64 = noise_gglwe_product(
        module.n() as f64,
        basek,
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
        (noise_have - noise_want).abs() <= 0.1,
        "{} {}",
        noise_have,
        noise_want
    );
}

fn test_external_product(log_n: usize, basek: usize, k_ggsw: usize, k_ct_in: usize, k_ct_out: usize, rank: usize, sigma: f64) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);

    let rows: usize = (k_ct_in + basek - 1) / basek;

    let mut ct_rgsw: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::new(&module, basek, k_ggsw, rows, rank);
    let mut ct_rlwe_in: GLWECiphertext<Vec<u8>> = GLWECiphertext::new(&module, basek, k_ct_in, rank);
    let mut ct_rlwe_out: GLWECiphertext<Vec<u8>> = GLWECiphertext::new(&module, basek, k_ct_out, rank);
    let mut pt_rgsw: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::new(&module, basek, k_ct_in);
    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::new(&module, basek, k_ct_out);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    // Random input plaintext
    pt_want
        .data
        .fill_uniform(basek, 0, pt_want.size(), &mut source_xa);

    pt_want.to_mut().at_mut(0, 0)[1] = 1;

    let k: usize = 1;

    pt_rgsw.raw_mut()[k] = 1; // X^{k}

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GGSWCiphertext::encrypt_sk_scratch_space(&module, rank, ct_rgsw.size())
            | GLWECiphertext::decrypt_scratch_space(&module, ct_rlwe_out.size())
            | GLWECiphertext::encrypt_sk_scratch_space(&module, ct_rlwe_in.size())
            | GLWECiphertext::external_product_scratch_space(
                &module,
                ct_rlwe_out.size(),
                ct_rlwe_in.size(),
                ct_rgsw.size(),
                rank,
            ),
    );

    let mut sk: SecretKey<Vec<u8>> = SecretKey::new(&module, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::new(&module, rank);
    sk_dft.dft(&module, &sk);

    ct_rgsw.encrypt_sk(
        &module,
        &pt_rgsw,
        &sk_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_rlwe_in.encrypt_sk(
        &module,
        &pt_want,
        &sk_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_rlwe_out.external_product(&module, &ct_rlwe_in, &ct_rgsw, scratch.borrow());

    ct_rlwe_out.decrypt(&module, &mut pt_have, &sk_dft, scratch.borrow());

    module.vec_znx_rotate_inplace(k as i64, &mut pt_want, 0);

    module.vec_znx_sub_ab_inplace(&mut pt_have, 0, &pt_want, 0);

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
        k_ct_in,
        k_ggsw,
    );

    assert!(
        (noise_have - noise_want).abs() <= 0.1,
        "{} {}",
        noise_have,
        noise_want
    );
}

fn test_external_product_inplace(log_n: usize, basek: usize, k_ggsw: usize, k_ct: usize, rank: usize, sigma: f64) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let rows: usize = (k_ct + basek - 1) / basek;

    let mut ct_rgsw: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::new(&module, basek, k_ggsw, rows, rank);
    let mut ct_rlwe: GLWECiphertext<Vec<u8>> = GLWECiphertext::new(&module, basek, k_ct, rank);
    let mut pt_rgsw: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::new(&module, basek, k_ct);
    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::new(&module, basek, k_ct);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    // Random input plaintext
    pt_want
        .data
        .fill_uniform(basek, 0, pt_want.size(), &mut source_xa);

    pt_want.to_mut().at_mut(0, 0)[1] = 1;

    let k: usize = 1;

    pt_rgsw.raw_mut()[k] = 1; // X^{k}

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GGSWCiphertext::encrypt_sk_scratch_space(&module, rank, ct_rgsw.size())
            | GLWECiphertext::decrypt_scratch_space(&module, ct_rlwe.size())
            | GLWECiphertext::encrypt_sk_scratch_space(&module, ct_rlwe.size())
            | GLWECiphertext::external_product_inplace_scratch_space(&module, ct_rlwe.size(), ct_rgsw.size(), rank),
    );

    let mut sk: SecretKey<Vec<u8>> = SecretKey::new(&module, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::new(&module, rank);
    sk_dft.dft(&module, &sk);

    ct_rgsw.encrypt_sk(
        &module,
        &pt_rgsw,
        &sk_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_rlwe.encrypt_sk(
        &module,
        &pt_want,
        &sk_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_rlwe.external_product_inplace(&module, &ct_rgsw, scratch.borrow());

    ct_rlwe.decrypt(&module, &mut pt_have, &sk_dft, scratch.borrow());

    module.vec_znx_rotate_inplace(k as i64, &mut pt_want, 0);

    module.vec_znx_sub_ab_inplace(&mut pt_have, 0, &pt_want, 0);

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
        k_ct,
        k_ggsw,
    );

    assert!(
        (noise_have - noise_want).abs() <= 0.1,
        "{} {}",
        noise_have,
        noise_want
    );
}
