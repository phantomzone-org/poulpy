use crate::{
    elem::Infos,
    encryption::EncryptSkScratchSpace,
    external_product::{
        ExternalProduct, ExternalProductInplace, ExternalProductInplaceScratchSpace, ExternalProductScratchSpace,
    },
    ggsw::GGSWCiphertext,
    glwe::{GLWECiphertext, GLWECiphertextFourier, GLWEPlaintext},
    keys::{SecretKey, SecretKeyFourier},
    keyswitch::{KeySwitch, KeySwitchInplace, KeySwitchInplaceScratchSpace, KeySwitchScratchSpace},
    keyswitch_key::GLWEKeySwitchKey,
    test_fft64::{grlwe::noise_grlwe_rlwe_product, rgsw::noise_rgsw_product},
};
use base2k::{FFT64, FillUniform, Module, ScalarZnx, ScalarZnxAlloc, ScratchOwned, Stats, VecZnxOps, VecZnxToMut, ZnxViewMut};
use sampling::source::Source;

#[test]
fn keyswitch() {
    let module: Module<FFT64> = Module::<FFT64>::new(2048);
    let log_base2k: usize = 12;
    let log_k_grlwe: usize = 60;
    let log_k_rlwe_in: usize = 45;
    let log_k_rlwe_out: usize = 60;
    let rows: usize = (log_k_rlwe_in + log_base2k - 1) / log_base2k;

    let sigma: f64 = 3.2;
    let bound: f64 = sigma * 6.0;

    let mut ct_grlwe: GLWEKeySwitchKey<Vec<u8>, FFT64> = GLWEKeySwitchKey::new(&module, log_base2k, log_k_grlwe, rows);
    let mut ct_rlwe_in: GLWECiphertext<Vec<u8>> = GLWECiphertext::new(&module, log_base2k, log_k_rlwe_in);
    let mut ct_rlwe_in_dft: GLWECiphertextFourier<Vec<u8>, FFT64> =
        GLWECiphertextFourier::new(&module, log_base2k, log_k_rlwe_in);
    let mut ct_rlwe_out: GLWECiphertext<Vec<u8>> = GLWECiphertext::new(&module, log_base2k, log_k_rlwe_out);
    let mut ct_rlwe_out_dft: GLWECiphertextFourier<Vec<u8>, FFT64> =
        GLWECiphertextFourier::new(&module, log_base2k, log_k_rlwe_out);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::new(&module, log_base2k, log_k_rlwe_in);
    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::new(&module, log_base2k, log_k_rlwe_out);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    // Random input plaintext
    pt_want
        .data
        .fill_uniform(log_base2k, 0, pt_want.size(), &mut source_xa);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWEKeySwitchKey::encrypt_sk_scratch_space(&module, ct_grlwe.size())
            | GLWECiphertext::decrypt_scratch_space(&module, ct_rlwe_out.size())
            | GLWECiphertext::encrypt_sk_scratch_space(&module, ct_rlwe_in.size())
            | GLWECiphertextFourier::keyswitch_scratch_space(
                &module,
                ct_rlwe_out.size(),
                ct_rlwe_in.size(),
                ct_grlwe.size(),
            ),
    );

    let mut sk0: SecretKey<Vec<u8>> = SecretKey::new(&module);
    sk0.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk0_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::new(&module);
    sk0_dft.dft(&module, &sk0);

    let mut sk1: SecretKey<Vec<u8>> = SecretKey::new(&module);
    sk1.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk1_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::new(&module);
    sk1_dft.dft(&module, &sk1);

    ct_grlwe.encrypt_sk(
        &module,
        &sk0.data,
        &sk1_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        bound,
        scratch.borrow(),
    );

    ct_rlwe_in.encrypt_sk(
        &module,
        &pt_want,
        &sk0_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        bound,
        scratch.borrow(),
    );

    ct_rlwe_in.dft(&module, &mut ct_rlwe_in_dft);
    ct_rlwe_out_dft.keyswitch(&module, &ct_rlwe_in_dft, &ct_grlwe, scratch.borrow());
    ct_rlwe_out_dft.idft(&module, &mut ct_rlwe_out, scratch.borrow());

    ct_rlwe_out.decrypt(&module, &mut pt_have, &sk1_dft, scratch.borrow());

    module.vec_znx_sub_ab_inplace(&mut pt_have, 0, &pt_want, 0);

    let noise_have: f64 = pt_have.data.std(0, log_base2k).log2();
    let noise_want: f64 = noise_grlwe_rlwe_product(
        module.n() as f64,
        log_base2k,
        0.5,
        0.5,
        0f64,
        sigma * sigma,
        0f64,
        log_k_rlwe_in,
        log_k_grlwe,
    );

    assert!(
        (noise_have - noise_want).abs() <= 0.1,
        "{} {}",
        noise_have,
        noise_want
    );
}

#[test]
fn keyswich_inplace() {
    let module: Module<FFT64> = Module::<FFT64>::new(2048);
    let log_base2k: usize = 12;
    let log_k_grlwe: usize = 60;
    let log_k_rlwe: usize = 45;
    let rows: usize = (log_k_rlwe + log_base2k - 1) / log_base2k;

    let sigma: f64 = 3.2;
    let bound: f64 = sigma * 6.0;

    let mut ct_grlwe: GLWEKeySwitchKey<Vec<u8>, FFT64> = GLWEKeySwitchKey::new(&module, log_base2k, log_k_grlwe, rows);
    let mut ct_rlwe: GLWECiphertext<Vec<u8>> = GLWECiphertext::new(&module, log_base2k, log_k_rlwe);
    let mut ct_rlwe_dft: GLWECiphertextFourier<Vec<u8>, FFT64> = GLWECiphertextFourier::new(&module, log_base2k, log_k_rlwe);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::new(&module, log_base2k, log_k_rlwe);
    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::new(&module, log_base2k, log_k_rlwe);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    // Random input plaintext
    pt_want
        .data
        .fill_uniform(log_base2k, 0, pt_want.size(), &mut source_xa);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWEKeySwitchKey::encrypt_sk_scratch_space(&module, ct_grlwe.size())
            | GLWECiphertext::decrypt_scratch_space(&module, ct_rlwe.size())
            | GLWECiphertext::encrypt_sk_scratch_space(&module, ct_rlwe.size())
            | GLWECiphertextFourier::keyswitch_inplace_scratch_space(&module, ct_rlwe_dft.size(), ct_grlwe.size()),
    );

    let mut sk0: SecretKey<Vec<u8>> = SecretKey::new(&module);
    sk0.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk0_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::new(&module);
    sk0_dft.dft(&module, &sk0);

    let mut sk1: SecretKey<Vec<u8>> = SecretKey::new(&module);
    sk1.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk1_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::new(&module);
    sk1_dft.dft(&module, &sk1);

    ct_grlwe.encrypt_sk(
        &module,
        &sk0.data,
        &sk1_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        bound,
        scratch.borrow(),
    );

    ct_rlwe.encrypt_sk(
        &module,
        &pt_want,
        &sk0_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        bound,
        scratch.borrow(),
    );

    ct_rlwe.dft(&module, &mut ct_rlwe_dft);
    ct_rlwe_dft.keyswitch_inplace(&module, &ct_grlwe, scratch.borrow());
    ct_rlwe_dft.idft(&module, &mut ct_rlwe, scratch.borrow());

    ct_rlwe.decrypt(&module, &mut pt_have, &sk1_dft, scratch.borrow());

    module.vec_znx_sub_ab_inplace(&mut pt_have, 0, &pt_want, 0);

    let noise_have: f64 = pt_have.data.std(0, log_base2k).log2();
    let noise_want: f64 = noise_grlwe_rlwe_product(
        module.n() as f64,
        log_base2k,
        0.5,
        0.5,
        0f64,
        sigma * sigma,
        0f64,
        log_k_rlwe,
        log_k_grlwe,
    );

    assert!(
        (noise_have - noise_want).abs() <= 0.1,
        "{} {}",
        noise_have,
        noise_want
    );
}

#[test]
fn external_product() {
    let module: Module<FFT64> = Module::<FFT64>::new(2048);
    let log_base2k: usize = 12;
    let log_k_grlwe: usize = 60;
    let log_k_rlwe_in: usize = 45;
    let log_k_rlwe_out: usize = 60;
    let rows: usize = (log_k_rlwe_in + log_base2k - 1) / log_base2k;

    let sigma: f64 = 3.2;
    let bound: f64 = sigma * 6.0;

    let mut ct_rgsw: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::new(&module, log_base2k, log_k_grlwe, rows);
    let mut ct_rlwe_in: GLWECiphertext<Vec<u8>> = GLWECiphertext::new(&module, log_base2k, log_k_rlwe_in);
    let mut ct_rlwe_out: GLWECiphertext<Vec<u8>> = GLWECiphertext::new(&module, log_base2k, log_k_rlwe_out);
    let mut ct_rlwe_dft_in: GLWECiphertextFourier<Vec<u8>, FFT64> =
        GLWECiphertextFourier::new(&module, log_base2k, log_k_rlwe_in);
    let mut ct_rlwe_dft_out: GLWECiphertextFourier<Vec<u8>, FFT64> =
        GLWECiphertextFourier::new(&module, log_base2k, log_k_rlwe_out);
    let mut pt_rgsw: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::new(&module, log_base2k, log_k_rlwe_in);
    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::new(&module, log_base2k, log_k_rlwe_out);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    // Random input plaintext
    pt_want
        .data
        .fill_uniform(log_base2k, 0, pt_want.size(), &mut source_xa);

    pt_want.to_mut().at_mut(0, 0)[1] = 1;

    let k: usize = 1;

    pt_rgsw.raw_mut()[k] = 1; // X^{k}

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GGSWCiphertext::encrypt_sk_scratch_space(&module, ct_rgsw.size())
            | GLWECiphertext::decrypt_scratch_space(&module, ct_rlwe_out.size())
            | GLWECiphertext::encrypt_sk_scratch_space(&module, ct_rlwe_in.size())
            | GLWECiphertext::external_product_scratch_space(
                &module,
                ct_rlwe_out.size(),
                ct_rlwe_in.size(),
                ct_rgsw.size(),
            ),
    );

    let mut sk: SecretKey<Vec<u8>> = SecretKey::new(&module);
    sk.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::new(&module);
    sk_dft.dft(&module, &sk);

    ct_rgsw.encrypt_sk(
        &module,
        &pt_rgsw,
        &sk_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        bound,
        scratch.borrow(),
    );

    ct_rlwe_in.encrypt_sk(
        &module,
        &pt_want,
        &sk_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        bound,
        scratch.borrow(),
    );

    ct_rlwe_in.dft(&module, &mut ct_rlwe_dft_in);
    ct_rlwe_dft_out.external_product(&module, &ct_rlwe_dft_in, &ct_rgsw, scratch.borrow());
    ct_rlwe_dft_out.idft(&module, &mut ct_rlwe_out, scratch.borrow());

    ct_rlwe_out.decrypt(&module, &mut pt_have, &sk_dft, scratch.borrow());

    module.vec_znx_rotate_inplace(k as i64, &mut pt_want, 0);

    module.vec_znx_sub_ab_inplace(&mut pt_have, 0, &pt_want, 0);

    let noise_have: f64 = pt_have.data.std(0, log_base2k).log2();

    let var_gct_err_lhs: f64 = sigma * sigma;
    let var_gct_err_rhs: f64 = 0f64;

    let var_msg: f64 = 1f64 / module.n() as f64; // X^{k}
    let var_a0_err: f64 = sigma * sigma;
    let var_a1_err: f64 = 1f64 / 12f64;

    let noise_want: f64 = noise_rgsw_product(
        module.n() as f64,
        log_base2k,
        0.5,
        var_msg,
        var_a0_err,
        var_a1_err,
        var_gct_err_lhs,
        var_gct_err_rhs,
        log_k_rlwe_in,
        log_k_grlwe,
    );

    assert!(
        (noise_have - noise_want).abs() <= 0.1,
        "{} {}",
        noise_have,
        noise_want
    );
}

#[test]
fn external_product_inplace() {
    let module: Module<FFT64> = Module::<FFT64>::new(2048);
    let log_base2k: usize = 12;
    let log_k_grlwe: usize = 60;
    let log_k_rlwe_in: usize = 45;
    let log_k_rlwe_out: usize = 60;
    let rows: usize = (log_k_rlwe_in + log_base2k - 1) / log_base2k;

    let sigma: f64 = 3.2;
    let bound: f64 = sigma * 6.0;

    let mut ct_rgsw: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::new(&module, log_base2k, log_k_grlwe, rows);
    let mut ct_rlwe: GLWECiphertext<Vec<u8>> = GLWECiphertext::new(&module, log_base2k, log_k_rlwe_in);
    let mut ct_rlwe_dft: GLWECiphertextFourier<Vec<u8>, FFT64> = GLWECiphertextFourier::new(&module, log_base2k, log_k_rlwe_in);
    let mut pt_rgsw: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::new(&module, log_base2k, log_k_rlwe_in);
    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::new(&module, log_base2k, log_k_rlwe_out);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    // Random input plaintext
    pt_want
        .data
        .fill_uniform(log_base2k, 0, pt_want.size(), &mut source_xa);

    pt_want.to_mut().at_mut(0, 0)[1] = 1;

    let k: usize = 1;

    pt_rgsw.raw_mut()[k] = 1; // X^{k}

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GGSWCiphertext::encrypt_sk_scratch_space(&module, ct_rgsw.size())
            | GLWECiphertext::decrypt_scratch_space(&module, ct_rlwe.size())
            | GLWECiphertext::encrypt_sk_scratch_space(&module, ct_rlwe.size())
            | GLWECiphertext::external_product_inplace_scratch_space(&module, ct_rlwe.size(), ct_rgsw.size()),
    );

    let mut sk: SecretKey<Vec<u8>> = SecretKey::new(&module);
    sk.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::new(&module);
    sk_dft.dft(&module, &sk);

    ct_rgsw.encrypt_sk(
        &module,
        &pt_rgsw,
        &sk_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        bound,
        scratch.borrow(),
    );

    ct_rlwe.encrypt_sk(
        &module,
        &pt_want,
        &sk_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        bound,
        scratch.borrow(),
    );

    ct_rlwe.dft(&module, &mut ct_rlwe_dft);
    ct_rlwe_dft.external_product_inplace(&module, &ct_rgsw, scratch.borrow());
    ct_rlwe_dft.idft(&module, &mut ct_rlwe, scratch.borrow());

    ct_rlwe.decrypt(&module, &mut pt_have, &sk_dft, scratch.borrow());

    module.vec_znx_rotate_inplace(k as i64, &mut pt_want, 0);

    module.vec_znx_sub_ab_inplace(&mut pt_have, 0, &pt_want, 0);

    let noise_have: f64 = pt_have.data.std(0, log_base2k).log2();

    let var_gct_err_lhs: f64 = sigma * sigma;
    let var_gct_err_rhs: f64 = 0f64;

    let var_msg: f64 = 1f64 / module.n() as f64; // X^{k}
    let var_a0_err: f64 = sigma * sigma;
    let var_a1_err: f64 = 1f64 / 12f64;

    let noise_want: f64 = noise_rgsw_product(
        module.n() as f64,
        log_base2k,
        0.5,
        var_msg,
        var_a0_err,
        var_a1_err,
        var_gct_err_lhs,
        var_gct_err_rhs,
        log_k_rlwe_in,
        log_k_grlwe,
    );

    assert!(
        (noise_have - noise_want).abs() <= 0.1,
        "{} {}",
        noise_have,
        noise_want
    );
}
