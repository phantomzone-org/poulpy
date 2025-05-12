use base2k::{FFT64, Module, ScalarZnx, ScalarZnxAlloc, ScratchOwned, Stats, VecZnxOps, ZnxViewMut};
use sampling::source::Source;

use crate::{
    elem::{GetRow, Infos},
    external_product::{
        ExternalProduct, ExternalProductInplace, ExternalProductInplaceScratchSpace, ExternalProductScratchSpace,
    },
    ggsw::GGSWCiphertext,
    glwe::{GLWECiphertextFourier, GLWEPlaintext},
    keys::{SecretKey, SecretKeyFourier},
    keyswitch::{KeySwitch, KeySwitchInplace, KeySwitchInplaceScratchSpace, KeySwitchScratchSpace},
    keyswitch_key::GLWEKeySwitchKey,
    test_fft64::rgsw::noise_rgsw_product,
};

#[test]
fn encrypt_sk() {
    let module: Module<FFT64> = Module::<FFT64>::new(2048);
    let log_base2k: usize = 8;
    let log_k_ct: usize = 54;
    let rows: usize = 4;

    let sigma: f64 = 3.2;
    let bound: f64 = sigma * 6.0;

    let mut ct: GLWEKeySwitchKey<Vec<u8>, FFT64> = GLWEKeySwitchKey::new(&module, log_base2k, log_k_ct, rows);
    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::new(&module, log_base2k, log_k_ct);
    let mut pt_scalar: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    pt_scalar.fill_ternary_hw(0, module.n(), &mut source_xs);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWEKeySwitchKey::encrypt_sk_scratch_space(&module, ct.size())
            | GLWECiphertextFourier::decrypt_scratch_space(&module, ct.size()),
    );

    let mut sk: SecretKey<Vec<u8>> = SecretKey::new(&module);
    sk.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::new(&module);
    sk_dft.dft(&module, &sk);

    ct.encrypt_sk(
        &module,
        &pt_scalar,
        &sk_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        bound,
        scratch.borrow(),
    );

    let mut ct_rlwe_dft: GLWECiphertextFourier<Vec<u8>, FFT64> = GLWECiphertextFourier::new(&module, log_base2k, log_k_ct);

    (0..ct.rows()).for_each(|row_i| {
        ct.get_row(&module, row_i, 0, &mut ct_rlwe_dft);
        ct_rlwe_dft.decrypt(&module, &mut pt, &sk_dft, scratch.borrow());
        module.vec_znx_sub_scalar_inplace(&mut pt, 0, row_i, &pt_scalar, 0);
        let std_pt: f64 = pt.data.std(0, log_base2k) * (log_k_ct as f64).exp2();
        assert!((sigma - std_pt).abs() <= 0.2, "{} {}", sigma, std_pt);
    });
}

#[test]
fn keyswitch() {
    let module: Module<FFT64> = Module::<FFT64>::new(2048);
    let log_base2k: usize = 12;
    let log_k_grlwe: usize = 60;
    let rows: usize = (log_k_grlwe + log_base2k - 1) / log_base2k;

    let sigma: f64 = 3.2;
    let bound: f64 = sigma * 6.0;

    let mut ct_grlwe_s0s1: GLWEKeySwitchKey<Vec<u8>, FFT64> = GLWEKeySwitchKey::new(&module, log_base2k, log_k_grlwe, rows);
    let mut ct_grlwe_s1s2: GLWEKeySwitchKey<Vec<u8>, FFT64> = GLWEKeySwitchKey::new(&module, log_base2k, log_k_grlwe, rows);
    let mut ct_grlwe_s0s2: GLWEKeySwitchKey<Vec<u8>, FFT64> = GLWEKeySwitchKey::new(&module, log_base2k, log_k_grlwe, rows);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWEKeySwitchKey::encrypt_sk_scratch_space(&module, ct_grlwe_s0s1.size())
            | GLWECiphertextFourier::decrypt_scratch_space(&module, ct_grlwe_s0s2.size())
            | GLWEKeySwitchKey::keyswitch_scratch_space(
                &module,
                ct_grlwe_s0s2.size(),
                ct_grlwe_s0s1.size(),
                ct_grlwe_s1s2.size(),
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

    let mut sk2: SecretKey<Vec<u8>> = SecretKey::new(&module);
    sk2.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk2_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::new(&module);
    sk2_dft.dft(&module, &sk2);

    // GRLWE_{s1}(s0) = s0 -> s1
    ct_grlwe_s0s1.encrypt_sk(
        &module,
        &sk0.data,
        &sk1_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        bound,
        scratch.borrow(),
    );

    // GRLWE_{s2}(s1) -> s1 -> s2
    ct_grlwe_s1s2.encrypt_sk(
        &module,
        &sk1.data,
        &sk2_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        bound,
        scratch.borrow(),
    );

    // GRLWE_{s1}(s0) (x) GRLWE_{s2}(s1) = GRLWE_{s2}(s0)
    ct_grlwe_s0s2.keyswitch(&module, &ct_grlwe_s0s1, &ct_grlwe_s1s2, scratch.borrow());

    let mut ct_rlwe_dft_s0s2: GLWECiphertextFourier<Vec<u8>, FFT64> =
        GLWECiphertextFourier::new(&module, log_base2k, log_k_grlwe);
    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::new(&module, log_base2k, log_k_grlwe);

    (0..ct_grlwe_s0s2.rows()).for_each(|row_i| {
        ct_grlwe_s0s2.get_row(&module, row_i, 0, &mut ct_rlwe_dft_s0s2);
        ct_rlwe_dft_s0s2.decrypt(&module, &mut pt, &sk2_dft, scratch.borrow());
        module.vec_znx_sub_scalar_inplace(&mut pt, 0, row_i, &sk0, 0);

        let noise_have: f64 = pt.data.std(0, log_base2k).log2();
        let noise_want: f64 = noise_grlwe_rlwe_product(
            module.n() as f64,
            log_base2k,
            0.5,
            0.5,
            0f64,
            sigma * sigma,
            0f64,
            log_k_grlwe,
            log_k_grlwe,
        );

        assert!(
            (noise_have - noise_want).abs() <= 0.1,
            "{} {}",
            noise_have,
            noise_want
        );
    });
}

#[test]
fn keyswitch_inplace() {
    let module: Module<FFT64> = Module::<FFT64>::new(2048);
    let log_base2k: usize = 12;
    let log_k_grlwe: usize = 60;
    let rows: usize = (log_k_grlwe + log_base2k - 1) / log_base2k;

    let sigma: f64 = 3.2;
    let bound: f64 = sigma * 6.0;

    let mut ct_grlwe_s0s1: GLWEKeySwitchKey<Vec<u8>, FFT64> = GLWEKeySwitchKey::new(&module, log_base2k, log_k_grlwe, rows);
    let mut ct_grlwe_s1s2: GLWEKeySwitchKey<Vec<u8>, FFT64> = GLWEKeySwitchKey::new(&module, log_base2k, log_k_grlwe, rows);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWEKeySwitchKey::encrypt_sk_scratch_space(&module, ct_grlwe_s0s1.size())
            | GLWECiphertextFourier::decrypt_scratch_space(&module, ct_grlwe_s0s1.size())
            | GLWEKeySwitchKey::keyswitch_inplace_scratch_space(&module, ct_grlwe_s0s1.size(), ct_grlwe_s1s2.size()),
    );

    let mut sk0: SecretKey<Vec<u8>> = SecretKey::new(&module);
    sk0.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk0_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::new(&module);
    sk0_dft.dft(&module, &sk0);

    let mut sk1: SecretKey<Vec<u8>> = SecretKey::new(&module);
    sk1.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk1_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::new(&module);
    sk1_dft.dft(&module, &sk1);

    let mut sk2: SecretKey<Vec<u8>> = SecretKey::new(&module);
    sk2.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk2_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::new(&module);
    sk2_dft.dft(&module, &sk2);

    // GRLWE_{s1}(s0) = s0 -> s1
    ct_grlwe_s0s1.encrypt_sk(
        &module,
        &sk0.data,
        &sk1_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        bound,
        scratch.borrow(),
    );

    // GRLWE_{s2}(s1) -> s1 -> s2
    ct_grlwe_s1s2.encrypt_sk(
        &module,
        &sk1.data,
        &sk2_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        bound,
        scratch.borrow(),
    );

    // GRLWE_{s1}(s0) (x) GRLWE_{s2}(s1) = GRLWE_{s2}(s0)
    ct_grlwe_s0s1.keyswitch_inplace(&module, &ct_grlwe_s1s2, scratch.borrow());

    let ct_grlwe_s0s2: GLWEKeySwitchKey<Vec<u8>, FFT64> = ct_grlwe_s0s1;

    let mut ct_rlwe_dft_s0s2: GLWECiphertextFourier<Vec<u8>, FFT64> =
        GLWECiphertextFourier::new(&module, log_base2k, log_k_grlwe);
    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::new(&module, log_base2k, log_k_grlwe);

    (0..ct_grlwe_s0s2.rows()).for_each(|row_i| {
        ct_grlwe_s0s2.get_row(&module, row_i, 0, &mut ct_rlwe_dft_s0s2);
        ct_rlwe_dft_s0s2.decrypt(&module, &mut pt, &sk2_dft, scratch.borrow());
        module.vec_znx_sub_scalar_inplace(&mut pt, 0, row_i, &sk0, 0);

        let noise_have: f64 = pt.data.std(0, log_base2k).log2();
        let noise_want: f64 = noise_grlwe_rlwe_product(
            module.n() as f64,
            log_base2k,
            0.5,
            0.5,
            0f64,
            sigma * sigma,
            0f64,
            log_k_grlwe,
            log_k_grlwe,
        );

        assert!(
            (noise_have - noise_want).abs() <= 0.1,
            "{} {}",
            noise_have,
            noise_want
        );
    });
}

#[test]
fn external_product() {
    let module: Module<FFT64> = Module::<FFT64>::new(2048);
    let log_base2k: usize = 12;
    let log_k_grlwe: usize = 60;
    let rows: usize = (log_k_grlwe + log_base2k - 1) / log_base2k;

    let sigma: f64 = 3.2;
    let bound: f64 = sigma * 6.0;

    let mut ct_grlwe_in: GLWEKeySwitchKey<Vec<u8>, FFT64> = GLWEKeySwitchKey::new(&module, log_base2k, log_k_grlwe, rows);
    let mut ct_grlwe_out: GLWEKeySwitchKey<Vec<u8>, FFT64> = GLWEKeySwitchKey::new(&module, log_base2k, log_k_grlwe, rows);
    let mut ct_rgsw: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::new(&module, log_base2k, log_k_grlwe, rows);

    let mut pt_rgsw: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);
    let mut pt_grlwe: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWEKeySwitchKey::encrypt_sk_scratch_space(&module, ct_grlwe_in.size())
            | GLWECiphertextFourier::decrypt_scratch_space(&module, ct_grlwe_out.size())
            | GLWEKeySwitchKey::external_product_scratch_space(
                &module,
                ct_grlwe_out.size(),
                ct_grlwe_in.size(),
                ct_rgsw.size(),
            )
            | GGSWCiphertext::encrypt_sk_scratch_space(&module, ct_rgsw.size()),
    );

    let k: usize = 1;

    pt_rgsw.raw_mut()[k] = 1; // X^{k}

    pt_grlwe.fill_ternary_prob(0, 0.5, &mut source_xs);

    let mut sk: SecretKey<Vec<u8>> = SecretKey::new(&module);
    sk.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::new(&module);
    sk_dft.dft(&module, &sk);

    // GRLWE_{s1}(s0) = s0 -> s1
    ct_grlwe_in.encrypt_sk(
        &module,
        &pt_grlwe,
        &sk_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        bound,
        scratch.borrow(),
    );

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

    // GRLWE_(m) (x) RGSW_(X^k) = GRLWE_(m * X^k)
    ct_grlwe_out.external_product(&module, &ct_grlwe_in, &ct_rgsw, scratch.borrow());

    let mut ct_rlwe_dft_s0s2: GLWECiphertextFourier<Vec<u8>, FFT64> =
        GLWECiphertextFourier::new(&module, log_base2k, log_k_grlwe);
    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::new(&module, log_base2k, log_k_grlwe);

    module.vec_znx_rotate_inplace(k as i64, &mut pt_grlwe, 0);

    (0..ct_grlwe_out.rows()).for_each(|row_i| {
        ct_grlwe_out.get_row(&module, row_i, 0, &mut ct_rlwe_dft_s0s2);
        ct_rlwe_dft_s0s2.decrypt(&module, &mut pt, &sk_dft, scratch.borrow());
        module.vec_znx_sub_scalar_inplace(&mut pt, 0, row_i, &pt_grlwe, 0);

        let noise_have: f64 = pt.data.std(0, log_base2k).log2();

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
            log_k_grlwe,
            log_k_grlwe,
        );

        assert!(
            (noise_have - noise_want).abs() <= 0.1,
            "{} {}",
            noise_have,
            noise_want
        );
    });
}

#[test]
fn external_product_inplace() {
    let module: Module<FFT64> = Module::<FFT64>::new(2048);
    let log_base2k: usize = 12;
    let log_k_grlwe: usize = 60;
    let rows: usize = (log_k_grlwe + log_base2k - 1) / log_base2k;

    let sigma: f64 = 3.2;
    let bound: f64 = sigma * 6.0;

    let mut ct_grlwe: GLWEKeySwitchKey<Vec<u8>, FFT64> = GLWEKeySwitchKey::new(&module, log_base2k, log_k_grlwe, rows);
    let mut ct_rgsw: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::new(&module, log_base2k, log_k_grlwe, rows);

    let mut pt_rgsw: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);
    let mut pt_grlwe: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWEKeySwitchKey::encrypt_sk_scratch_space(&module, ct_grlwe.size())
            | GLWECiphertextFourier::decrypt_scratch_space(&module, ct_grlwe.size())
            | GLWEKeySwitchKey::external_product_inplace_scratch_space(&module, ct_grlwe.size(), ct_rgsw.size())
            | GGSWCiphertext::encrypt_sk_scratch_space(&module, ct_rgsw.size()),
    );

    let k: usize = 1;

    pt_rgsw.raw_mut()[k] = 1; // X^{k}

    pt_grlwe.fill_ternary_prob(0, 0.5, &mut source_xs);

    let mut sk: SecretKey<Vec<u8>> = SecretKey::new(&module);
    sk.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::new(&module);
    sk_dft.dft(&module, &sk);

    // GRLWE_{s1}(s0) = s0 -> s1
    ct_grlwe.encrypt_sk(
        &module,
        &pt_grlwe,
        &sk_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        bound,
        scratch.borrow(),
    );

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

    // GRLWE_(m) (x) RGSW_(X^k) = GRLWE_(m * X^k)
    ct_grlwe.external_product_inplace(&module, &ct_rgsw, scratch.borrow());

    let mut ct_rlwe_dft_s0s2: GLWECiphertextFourier<Vec<u8>, FFT64> =
        GLWECiphertextFourier::new(&module, log_base2k, log_k_grlwe);
    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::new(&module, log_base2k, log_k_grlwe);

    module.vec_znx_rotate_inplace(k as i64, &mut pt_grlwe, 0);

    (0..ct_grlwe.rows()).for_each(|row_i| {
        ct_grlwe.get_row(&module, row_i, 0, &mut ct_rlwe_dft_s0s2);
        ct_rlwe_dft_s0s2.decrypt(&module, &mut pt, &sk_dft, scratch.borrow());
        module.vec_znx_sub_scalar_inplace(&mut pt, 0, row_i, &pt_grlwe, 0);

        let noise_have: f64 = pt.data.std(0, log_base2k).log2();

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
            log_k_grlwe,
            log_k_grlwe,
        );

        assert!(
            (noise_have - noise_want).abs() <= 0.1,
            "{} {}",
            noise_have,
            noise_want
        );
    });
}

pub(crate) fn noise_grlwe_rlwe_product(
    n: f64,
    log_base2k: usize,
    var_xs: f64,
    var_msg: f64,
    var_a_err: f64,
    var_gct_err_lhs: f64,
    var_gct_err_rhs: f64,
    a_logq: usize,
    b_logq: usize,
) -> f64 {
    let a_logq: usize = a_logq.min(b_logq);
    let a_cols: usize = (a_logq + log_base2k - 1) / log_base2k;

    let b_scale = 2.0f64.powi(b_logq as i32);
    let a_scale: f64 = 2.0f64.powi((b_logq - a_logq) as i32);

    let base: f64 = (1 << (log_base2k)) as f64;
    let var_base: f64 = base * base / 12f64;

    // lhs = a_cols * n * (var_base * var_gct_err_lhs + var_e_a * var_msg * p^2)
    // rhs = a_cols * n * var_base * var_gct_err_rhs * var_xs
    let mut noise: f64 = (a_cols as f64) * n * var_base * (var_gct_err_lhs + var_xs * var_gct_err_rhs);
    noise += var_msg * var_a_err * a_scale * a_scale * n;
    noise = noise.sqrt();
    noise /= b_scale;
    noise.log2().min(-1.0) // max noise is [-2^{-1}, 2^{-1}]
}
