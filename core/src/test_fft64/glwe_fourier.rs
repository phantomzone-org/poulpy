use crate::{
    elem::Infos,
    ggsw_ciphertext::GGSWCiphertext,
    glwe_ciphertext::GLWECiphertext,
    glwe_ciphertext_fourier::GLWECiphertextFourier,
    glwe_plaintext::GLWEPlaintext,
    keys::{SecretKey, SecretKeyFourier},
    keyswitch_key::GLWESwitchingKey,
    test_fft64::{gglwe::log2_std_noise_gglwe_product, ggsw::noise_ggsw_product},
};
use backend::{FFT64, FillUniform, Module, ScalarZnx, ScalarZnxAlloc, ScratchOwned, Stats, VecZnxOps, VecZnxToMut, ZnxViewMut};
use sampling::source::Source;

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

fn test_keyswitch(
    log_n: usize,
    basek: usize,
    k_ksk: usize,
    k_ct_in: usize,
    k_ct_out: usize,
    rank_in: usize,
    rank_out: usize,
    sigma: f64,
) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);

    let rows: usize = (k_ct_in + basek - 1) / basek;

    let mut ksk: GLWESwitchingKey<Vec<u8>, FFT64> = GLWESwitchingKey::alloc(&module, basek, k_ksk, rows, rank_in, rank_out);
    let mut ct_glwe_in: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, k_ct_in, rank_in);
    let mut ct_glwe_dft_in: GLWECiphertextFourier<Vec<u8>, FFT64> = GLWECiphertextFourier::alloc(&module, basek, k_ct_in, rank_in);
    let mut ct_glwe_out: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, k_ct_out, rank_out);
    let mut ct_glwe_dft_out: GLWECiphertextFourier<Vec<u8>, FFT64> =
        GLWECiphertextFourier::alloc(&module, basek, k_ct_out, rank_out);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_ct_in);
    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_ct_out);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    // Random input plaintext
    pt_want
        .data
        .fill_uniform(basek, 0, pt_want.size(), &mut source_xa);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWESwitchingKey::encrypt_sk_scratch_space(&module, rank_out, ksk.size())
            | GLWECiphertext::decrypt_scratch_space(&module, ct_glwe_out.size())
            | GLWECiphertext::encrypt_sk_scratch_space(&module, ct_glwe_in.size())
            | GLWECiphertextFourier::keyswitch_scratch_space(
                &module,
                ct_glwe_out.size(),
                rank_out,
                ct_glwe_in.size(),
                rank_in,
                ksk.size(),
            ),
    );

    let mut sk_in: SecretKey<Vec<u8>> = SecretKey::alloc(&module, rank_in);
    sk_in.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_in_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::alloc(&module, rank_in);
    sk_in_dft.dft(&module, &sk_in);

    let mut sk_out: SecretKey<Vec<u8>> = SecretKey::alloc(&module, rank_out);
    sk_out.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_out_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::alloc(&module, rank_out);
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

    module.vec_znx_sub_ab_inplace(&mut pt_have, 0, &pt_want, 0);

    let noise_have: f64 = pt_have.data.std(0, basek).log2();
    let noise_want: f64 = log2_std_noise_gglwe_product(
        module.n() as f64,
        basek,
        0.5,
        0.5,
        0f64,
        sigma * sigma,
        0f64,
        rank_in as f64,
        k_ct_in,
        k_ksk,
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

    let mut ksk: GLWESwitchingKey<Vec<u8>, FFT64> = GLWESwitchingKey::alloc(&module, basek, k_ksk, rows, rank, rank);
    let mut ct_glwe: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, k_ct, rank);
    let mut ct_rlwe_dft: GLWECiphertextFourier<Vec<u8>, FFT64> = GLWECiphertextFourier::alloc(&module, basek, k_ct, rank);
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
        GLWESwitchingKey::encrypt_sk_scratch_space(&module, rank, ksk.size())
            | GLWECiphertext::decrypt_scratch_space(&module, ct_glwe.size())
            | GLWECiphertext::encrypt_sk_scratch_space(&module, ct_glwe.size())
            | GLWECiphertextFourier::keyswitch_inplace_scratch_space(&module, ct_rlwe_dft.size(), ksk.size(), rank),
    );

    let mut sk_in: SecretKey<Vec<u8>> = SecretKey::alloc(&module, rank);
    sk_in.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_in_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::alloc(&module, rank);
    sk_in_dft.dft(&module, &sk_in);

    let mut sk_out: SecretKey<Vec<u8>> = SecretKey::alloc(&module, rank);
    sk_out.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_out_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::alloc(&module, rank);
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

    module.vec_znx_sub_ab_inplace(&mut pt_have, 0, &pt_want, 0);

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

    let mut ct_rgsw: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::alloc(&module, basek, k_ggsw, rows, rank);
    let mut ct_in: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, k_ct_in, rank);
    let mut ct_out: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, k_ct_out, rank);
    let mut ct_in_dft: GLWECiphertextFourier<Vec<u8>, FFT64> = GLWECiphertextFourier::alloc(&module, basek, k_ct_in, rank);
    let mut ct_out_dft: GLWECiphertextFourier<Vec<u8>, FFT64> = GLWECiphertextFourier::alloc(&module, basek, k_ct_out, rank);
    let mut pt_rgsw: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_ct_in);
    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_ct_out);

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
            | GLWECiphertext::decrypt_scratch_space(&module, ct_out.size())
            | GLWECiphertext::encrypt_sk_scratch_space(&module, ct_in.size())
            | GLWECiphertextFourier::external_product_scratch_space(&module, ct_out.size(), ct_in.size(), ct_rgsw.size(), rank),
    );

    let mut sk: SecretKey<Vec<u8>> = SecretKey::alloc(&module, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::alloc(&module, rank);
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

    ct_in.encrypt_sk(
        &module,
        &pt_want,
        &sk_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_in.dft(&module, &mut ct_in_dft);
    ct_out_dft.external_product(&module, &ct_in_dft, &ct_rgsw, scratch.borrow());
    ct_out_dft.idft(&module, &mut ct_out, scratch.borrow());

    ct_out.decrypt(&module, &mut pt_have, &sk_dft, scratch.borrow());

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

    let mut ct_ggsw: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::alloc(&module, basek, k_ggsw, rows, rank);
    let mut ct: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, k_ct, rank);
    let mut ct_rlwe_dft: GLWECiphertextFourier<Vec<u8>, FFT64> = GLWECiphertextFourier::alloc(&module, basek, k_ct, rank);
    let mut pt_rgsw: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_ct);
    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_ct);

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
        GGSWCiphertext::encrypt_sk_scratch_space(&module, rank, ct_ggsw.size())
            | GLWECiphertext::decrypt_scratch_space(&module, ct.size())
            | GLWECiphertext::encrypt_sk_scratch_space(&module, ct.size())
            | GLWECiphertextFourier::external_product_inplace_scratch_space(&module, ct.size(), ct_ggsw.size(), rank),
    );

    let mut sk: SecretKey<Vec<u8>> = SecretKey::alloc(&module, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::alloc(&module, rank);
    sk_dft.dft(&module, &sk);

    ct_ggsw.encrypt_sk(
        &module,
        &pt_rgsw,
        &sk_dft,
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

    ct.dft(&module, &mut ct_rlwe_dft);
    ct_rlwe_dft.external_product_inplace(&module, &ct_ggsw, scratch.borrow());
    ct_rlwe_dft.idft(&module, &mut ct, scratch.borrow());

    ct.decrypt(&module, &mut pt_have, &sk_dft, scratch.borrow());

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
