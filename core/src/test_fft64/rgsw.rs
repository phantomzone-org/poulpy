use base2k::{
    FFT64, Module, ScalarZnx, ScalarZnxAlloc, ScalarZnxDftOps, ScratchOwned, Stats, VecZnxBig, VecZnxBigAlloc, VecZnxBigOps,
    VecZnxDft, VecZnxDftAlloc, VecZnxDftOps, VecZnxOps, VecZnxToMut, ZnxViewMut, ZnxZero,
};
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
    test_fft64::grlwe::noise_grlwe_rlwe_product,
};

#[test]
fn encrypt_sk() {
    let module: Module<FFT64> = Module::<FFT64>::new(2048);
    let log_base2k: usize = 8;
    let log_k_ct: usize = 54;
    let rows: usize = 4;

    let sigma: f64 = 3.2;
    let bound: f64 = sigma * 6.0;

    let mut ct: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::new(&module, log_base2k, log_k_ct, rows);
    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::new(&module, log_base2k, log_k_ct);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::new(&module, log_base2k, log_k_ct);
    let mut pt_scalar: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    pt_scalar.fill_ternary_hw(0, module.n(), &mut source_xs);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GGSWCiphertext::encrypt_sk_scratch_space(&module, ct.size())
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
    let mut pt_dft: VecZnxDft<Vec<u8>, FFT64> = module.new_vec_znx_dft(1, ct.size());
    let mut pt_big: VecZnxBig<Vec<u8>, FFT64> = module.new_vec_znx_big(1, ct.size());

    (0..ct.rank()).for_each(|col_j| {
        (0..ct.rows()).for_each(|row_i| {
            module.vec_znx_add_scalar_inplace(&mut pt_want, 0, row_i, &pt_scalar, 0);

            if col_j == 1 {
                module.vec_znx_dft(&mut pt_dft, 0, &pt_want, 0);
                module.svp_apply_inplace(&mut pt_dft, 0, &sk_dft, 0);
                module.vec_znx_idft_tmp_a(&mut pt_big, 0, &mut pt_dft, 0);
                module.vec_znx_big_normalize(log_base2k, &mut pt_want, 0, &pt_big, 0, scratch.borrow());
            }

            ct.get_row(&module, row_i, col_j, &mut ct_rlwe_dft);

            ct_rlwe_dft.decrypt(&module, &mut pt_have, &sk_dft, scratch.borrow());

            module.vec_znx_sub_ab_inplace(&mut pt_have, 0, &pt_want, 0);

            let std_pt: f64 = pt_have.data.std(0, log_base2k) * (log_k_ct as f64).exp2();
            assert!((sigma - std_pt).abs() <= 0.2, "{} {}", sigma, std_pt);

            pt_want.data.zero();
        });
    });
}

#[test]
fn keyswitch() {
    let module: Module<FFT64> = Module::<FFT64>::new(2048);
    let log_base2k: usize = 12;
    let log_k_grlwe: usize = 60;
    let log_k_rgsw_in: usize = 45;
    let log_k_rgsw_out: usize = 45;
    let rows: usize = (log_k_rgsw_in + log_base2k - 1) / log_base2k;

    let sigma: f64 = 3.2;
    let bound: f64 = sigma * 6.0;

    let mut ct_grlwe: GLWEKeySwitchKey<Vec<u8>, FFT64> = GLWEKeySwitchKey::new(&module, log_base2k, log_k_grlwe, rows);
    let mut ct_rgsw_in: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::new(&module, log_base2k, log_k_rgsw_in, rows);
    let mut ct_rgsw_out: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::new(&module, log_base2k, log_k_rgsw_out, rows);
    let mut pt_rgsw: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    // Random input plaintext
    pt_rgsw.fill_ternary_prob(0, 0.5, &mut source_xs);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWEKeySwitchKey::encrypt_sk_scratch_space(&module, ct_grlwe.size())
            | GLWECiphertextFourier::decrypt_scratch_space(&module, ct_rgsw_out.size())
            | GGSWCiphertext::encrypt_sk_scratch_space(&module, ct_rgsw_in.size())
            | GGSWCiphertext::keyswitch_scratch_space(
                &module,
                ct_rgsw_out.size(),
                ct_rgsw_in.size(),
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

    ct_rgsw_in.encrypt_sk(
        &module,
        &pt_rgsw,
        &sk0_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        bound,
        scratch.borrow(),
    );

    ct_rgsw_out.keyswitch(&module, &ct_rgsw_in, &ct_grlwe, scratch.borrow());

    let mut ct_rlwe_dft: GLWECiphertextFourier<Vec<u8>, FFT64> = GLWECiphertextFourier::new(&module, log_base2k, log_k_rgsw_out);
    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::new(&module, log_base2k, log_k_rgsw_out);
    let mut pt_dft: VecZnxDft<Vec<u8>, FFT64> = module.new_vec_znx_dft(1, ct_rgsw_out.size());
    let mut pt_big: VecZnxBig<Vec<u8>, FFT64> = module.new_vec_znx_big(1, ct_rgsw_out.size());
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::new(&module, log_base2k, log_k_rgsw_out);

    (0..ct_rgsw_out.rank()).for_each(|col_j| {
        (0..ct_rgsw_out.rows()).for_each(|row_i| {
            module.vec_znx_add_scalar_inplace(&mut pt_want, 0, row_i, &pt_rgsw, 0);

            if col_j == 1 {
                module.vec_znx_dft(&mut pt_dft, 0, &pt_want, 0);
                module.svp_apply_inplace(&mut pt_dft, 0, &sk0_dft, 0);
                module.vec_znx_idft_tmp_a(&mut pt_big, 0, &mut pt_dft, 0);
                module.vec_znx_big_normalize(log_base2k, &mut pt_want, 0, &pt_big, 0, scratch.borrow());
            }

            ct_rgsw_out.get_row(&module, row_i, col_j, &mut ct_rlwe_dft);
            ct_rlwe_dft.decrypt(&module, &mut pt, &sk1_dft, scratch.borrow());

            module.vec_znx_sub_ab_inplace(&mut pt, 0, &pt_want, 0);

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
                (noise_have - noise_want).abs() <= 0.2,
                "have: {} want: {}",
                noise_have,
                noise_want
            );

            pt_want.data.zero();
        });
    });
}

#[test]
fn keyswitch_inplace() {
    let module: Module<FFT64> = Module::<FFT64>::new(2048);
    let log_base2k: usize = 12;
    let log_k_grlwe: usize = 60;
    let log_k_rgsw: usize = 45;
    let rows: usize = (log_k_rgsw + log_base2k - 1) / log_base2k;

    let sigma: f64 = 3.2;
    let bound: f64 = sigma * 6.0;

    let mut ct_grlwe: GLWEKeySwitchKey<Vec<u8>, FFT64> = GLWEKeySwitchKey::new(&module, log_base2k, log_k_grlwe, rows);
    let mut ct_rgsw: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::new(&module, log_base2k, log_k_rgsw, rows);
    let mut pt_rgsw: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    // Random input plaintext
    pt_rgsw.fill_ternary_prob(0, 0.5, &mut source_xs);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWEKeySwitchKey::encrypt_sk_scratch_space(&module, ct_grlwe.size())
            | GLWECiphertextFourier::decrypt_scratch_space(&module, ct_rgsw.size())
            | GGSWCiphertext::encrypt_sk_scratch_space(&module, ct_rgsw.size())
            | GGSWCiphertext::keyswitch_inplace_scratch_space(&module, ct_rgsw.size(), ct_grlwe.size()),
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

    ct_rgsw.encrypt_sk(
        &module,
        &pt_rgsw,
        &sk0_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        bound,
        scratch.borrow(),
    );

    ct_rgsw.keyswitch_inplace(&module, &ct_grlwe, scratch.borrow());

    let mut ct_rlwe_dft: GLWECiphertextFourier<Vec<u8>, FFT64> = GLWECiphertextFourier::new(&module, log_base2k, log_k_rgsw);
    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::new(&module, log_base2k, log_k_rgsw);
    let mut pt_dft: VecZnxDft<Vec<u8>, FFT64> = module.new_vec_znx_dft(1, ct_rgsw.size());
    let mut pt_big: VecZnxBig<Vec<u8>, FFT64> = module.new_vec_znx_big(1, ct_rgsw.size());
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::new(&module, log_base2k, log_k_rgsw);

    (0..ct_rgsw.rank()).for_each(|col_j| {
        (0..ct_rgsw.rows()).for_each(|row_i| {
            module.vec_znx_add_scalar_inplace(&mut pt_want, 0, row_i, &pt_rgsw, 0);

            if col_j == 1 {
                module.vec_znx_dft(&mut pt_dft, 0, &pt_want, 0);
                module.svp_apply_inplace(&mut pt_dft, 0, &sk0_dft, 0);
                module.vec_znx_idft_tmp_a(&mut pt_big, 0, &mut pt_dft, 0);
                module.vec_znx_big_normalize(log_base2k, &mut pt_want, 0, &pt_big, 0, scratch.borrow());
            }

            ct_rgsw.get_row(&module, row_i, col_j, &mut ct_rlwe_dft);
            ct_rlwe_dft.decrypt(&module, &mut pt, &sk1_dft, scratch.borrow());

            module.vec_znx_sub_ab_inplace(&mut pt, 0, &pt_want, 0);

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
                (noise_have - noise_want).abs() <= 0.2,
                "have: {} want: {}",
                noise_have,
                noise_want
            );

            pt_want.data.zero();
        });
    });
}

#[test]
fn external_product() {
    let module: Module<FFT64> = Module::<FFT64>::new(2048);
    let log_base2k: usize = 12;
    let log_k_rgsw_rhs: usize = 60;
    let log_k_rgsw_lhs_in: usize = 45;
    let log_k_rgsw_lhs_out: usize = 45;
    let rows: usize = (log_k_rgsw_lhs_in + log_base2k - 1) / log_base2k;

    let sigma: f64 = 3.2;
    let bound: f64 = sigma * 6.0;

    let mut ct_rgsw_rhs: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::new(&module, log_base2k, log_k_rgsw_rhs, rows);
    let mut ct_rgsw_lhs_in: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::new(&module, log_base2k, log_k_rgsw_lhs_in, rows);
    let mut ct_rgsw_lhs_out: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::new(&module, log_base2k, log_k_rgsw_lhs_out, rows);
    let mut pt_rgsw_lhs: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);
    let mut pt_rgsw_rhs: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    // Random input plaintext
    pt_rgsw_lhs.fill_ternary_prob(0, 0.5, &mut source_xs);

    let k: usize = 1;

    pt_rgsw_rhs.to_mut().raw_mut()[k] = 1; //X^{k}

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWEKeySwitchKey::encrypt_sk_scratch_space(&module, ct_rgsw_rhs.size())
            | GLWECiphertextFourier::decrypt_scratch_space(&module, ct_rgsw_lhs_out.size())
            | GGSWCiphertext::encrypt_sk_scratch_space(&module, ct_rgsw_lhs_in.size())
            | GGSWCiphertext::external_product_scratch_space(
                &module,
                ct_rgsw_lhs_out.size(),
                ct_rgsw_lhs_in.size(),
                ct_rgsw_rhs.size(),
            ),
    );

    let mut sk: SecretKey<Vec<u8>> = SecretKey::new(&module);
    sk.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::new(&module);
    sk_dft.dft(&module, &sk);

    ct_rgsw_rhs.encrypt_sk(
        &module,
        &pt_rgsw_rhs,
        &sk_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        bound,
        scratch.borrow(),
    );

    ct_rgsw_lhs_in.encrypt_sk(
        &module,
        &pt_rgsw_lhs,
        &sk_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        bound,
        scratch.borrow(),
    );

    ct_rgsw_lhs_out.external_product(&module, &ct_rgsw_lhs_in, &ct_rgsw_rhs, scratch.borrow());

    let mut ct_rlwe_dft: GLWECiphertextFourier<Vec<u8>, FFT64> =
        GLWECiphertextFourier::new(&module, log_base2k, log_k_rgsw_lhs_out);
    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::new(&module, log_base2k, log_k_rgsw_lhs_out);
    let mut pt_dft: VecZnxDft<Vec<u8>, FFT64> = module.new_vec_znx_dft(1, ct_rgsw_lhs_out.size());
    let mut pt_big: VecZnxBig<Vec<u8>, FFT64> = module.new_vec_znx_big(1, ct_rgsw_lhs_out.size());
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::new(&module, log_base2k, log_k_rgsw_lhs_out);

    module.vec_znx_rotate_inplace(k as i64, &mut pt_rgsw_lhs, 0);

    (0..ct_rgsw_lhs_out.rank()).for_each(|col_j| {
        (0..ct_rgsw_lhs_out.rows()).for_each(|row_i| {
            module.vec_znx_add_scalar_inplace(&mut pt_want, 0, row_i, &pt_rgsw_lhs, 0);

            if col_j == 1 {
                module.vec_znx_dft(&mut pt_dft, 0, &pt_want, 0);
                module.svp_apply_inplace(&mut pt_dft, 0, &sk_dft, 0);
                module.vec_znx_idft_tmp_a(&mut pt_big, 0, &mut pt_dft, 0);
                module.vec_znx_big_normalize(log_base2k, &mut pt_want, 0, &pt_big, 0, scratch.borrow());
            }

            ct_rgsw_lhs_out.get_row(&module, row_i, col_j, &mut ct_rlwe_dft);
            ct_rlwe_dft.decrypt(&module, &mut pt, &sk_dft, scratch.borrow());

            module.vec_znx_sub_ab_inplace(&mut pt, 0, &pt_want, 0);

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
                log_k_rgsw_lhs_in,
                log_k_rgsw_rhs,
            );

            assert!(
                (noise_have - noise_want).abs() <= 0.1,
                "have: {} want: {}",
                noise_have,
                noise_want
            );

            pt_want.data.zero();
        });
    });
}

#[test]
fn external_product_inplace() {
    let module: Module<FFT64> = Module::<FFT64>::new(2048);
    let log_base2k: usize = 12;
    let log_k_rgsw_rhs: usize = 60;
    let log_k_rgsw_lhs: usize = 45;
    let rows: usize = (log_k_rgsw_lhs + log_base2k - 1) / log_base2k;

    let sigma: f64 = 3.2;
    let bound: f64 = sigma * 6.0;

    let mut ct_rgsw_rhs: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::new(&module, log_base2k, log_k_rgsw_rhs, rows);
    let mut ct_rgsw_lhs: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::new(&module, log_base2k, log_k_rgsw_lhs, rows);
    let mut pt_rgsw_lhs: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);
    let mut pt_rgsw_rhs: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    // Random input plaintext
    pt_rgsw_lhs.fill_ternary_prob(0, 0.5, &mut source_xs);

    let k: usize = 1;

    pt_rgsw_rhs.to_mut().raw_mut()[k] = 1; //X^{k}

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWEKeySwitchKey::encrypt_sk_scratch_space(&module, ct_rgsw_rhs.size())
            | GLWECiphertextFourier::decrypt_scratch_space(&module, ct_rgsw_lhs.size())
            | GGSWCiphertext::encrypt_sk_scratch_space(&module, ct_rgsw_lhs.size())
            | GGSWCiphertext::external_product_inplace_scratch_space(&module, ct_rgsw_lhs.size(), ct_rgsw_rhs.size()),
    );

    let mut sk: SecretKey<Vec<u8>> = SecretKey::new(&module);
    sk.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::new(&module);
    sk_dft.dft(&module, &sk);

    ct_rgsw_rhs.encrypt_sk(
        &module,
        &pt_rgsw_rhs,
        &sk_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        bound,
        scratch.borrow(),
    );

    ct_rgsw_lhs.encrypt_sk(
        &module,
        &pt_rgsw_lhs,
        &sk_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        bound,
        scratch.borrow(),
    );

    ct_rgsw_lhs.external_product_inplace(&module, &ct_rgsw_rhs, scratch.borrow());

    let mut ct_rlwe_dft: GLWECiphertextFourier<Vec<u8>, FFT64> = GLWECiphertextFourier::new(&module, log_base2k, log_k_rgsw_lhs);
    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::new(&module, log_base2k, log_k_rgsw_lhs);
    let mut pt_dft: VecZnxDft<Vec<u8>, FFT64> = module.new_vec_znx_dft(1, ct_rgsw_lhs.size());
    let mut pt_big: VecZnxBig<Vec<u8>, FFT64> = module.new_vec_znx_big(1, ct_rgsw_lhs.size());
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::new(&module, log_base2k, log_k_rgsw_lhs);

    module.vec_znx_rotate_inplace(k as i64, &mut pt_rgsw_lhs, 0);

    (0..ct_rgsw_lhs.rank()).for_each(|col_j| {
        (0..ct_rgsw_lhs.rows()).for_each(|row_i| {
            module.vec_znx_add_scalar_inplace(&mut pt_want, 0, row_i, &pt_rgsw_lhs, 0);

            if col_j == 1 {
                module.vec_znx_dft(&mut pt_dft, 0, &pt_want, 0);
                module.svp_apply_inplace(&mut pt_dft, 0, &sk_dft, 0);
                module.vec_znx_idft_tmp_a(&mut pt_big, 0, &mut pt_dft, 0);
                module.vec_znx_big_normalize(log_base2k, &mut pt_want, 0, &pt_big, 0, scratch.borrow());
            }

            ct_rgsw_lhs.get_row(&module, row_i, col_j, &mut ct_rlwe_dft);
            ct_rlwe_dft.decrypt(&module, &mut pt, &sk_dft, scratch.borrow());

            module.vec_znx_sub_ab_inplace(&mut pt, 0, &pt_want, 0);

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
                log_k_rgsw_lhs,
                log_k_rgsw_rhs,
            );

            assert!(
                (noise_have - noise_want).abs() <= 0.1,
                "have: {} want: {}",
                noise_have,
                noise_want
            );

            pt_want.data.zero();
        });
    });
}

pub(crate) fn noise_rgsw_product(
    n: f64,
    log_base2k: usize,
    var_xs: f64,
    var_msg: f64,
    var_a0_err: f64,
    var_a1_err: f64,
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
    let mut noise: f64 = 2.0 * (a_cols as f64) * n * var_base * (var_gct_err_lhs + var_xs * var_gct_err_rhs);
    noise += var_msg * var_a0_err * a_scale * a_scale * n;
    noise += var_msg * var_a1_err * a_scale * a_scale * n * var_xs;
    noise = noise.sqrt();
    noise /= b_scale;
    noise.log2().min(-1.0) // max noise is [-2^{-1}, 2^{-1}]
}
