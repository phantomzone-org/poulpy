use base2k::{
    FFT64, Module, ScalarZnx, ScalarZnxAlloc, ScalarZnxDftOps, ScratchOwned, Stats, VecZnxBig, VecZnxBigAlloc, VecZnxBigOps,
    VecZnxDft, VecZnxDftAlloc, VecZnxDftOps, VecZnxOps, VecZnxToMut, ZnxViewMut, ZnxZero,
};
use sampling::source::Source;

use crate::{
    elem::{GetRow, Infos},
    ggsw_ciphertext::GGSWCiphertext,
    glwe_ciphertext_fourier::GLWECiphertextFourier,
    glwe_plaintext::GLWEPlaintext,
    keys::{SecretKey, SecretKeyFourier},
    keyswitch_key::GLWESwitchingKey,
};

#[test]
fn encrypt_sk() {
    (1..4).for_each(|rank| {
        println!("test encrypt_sk rank: {}", rank);
        test_encrypt_sk(11, 8, 54, 3.2, rank);
    });
}

#[test]
fn external_product() {
    (1..4).for_each(|rank| {
        println!("test external_product rank: {}", rank);
        test_external_product(12, 12, 60, rank, 3.2);
    });
}

#[test]
fn external_product_inplace() {
    (1..4).for_each(|rank| {
        println!("test external_product rank: {}", rank);
        test_external_product_inplace(12, 15, 60, rank, 3.2);
    });
}

fn test_encrypt_sk(log_n: usize, basek: usize, k_ggsw: usize, sigma: f64, rank: usize) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);

    let rows: usize = (k_ggsw + basek - 1) / basek;

    let mut ct: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::new(&module, basek, k_ggsw, rows, rank);
    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::new(&module, basek, k_ggsw);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::new(&module, basek, k_ggsw);
    let mut pt_scalar: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    pt_scalar.fill_ternary_hw(0, module.n(), &mut source_xs);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GGSWCiphertext::encrypt_sk_scratch_space(&module, rank, ct.size())
            | GLWECiphertextFourier::decrypt_scratch_space(&module, ct.size()),
    );

    let mut sk: SecretKey<Vec<u8>> = SecretKey::new(&module, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::new(&module, rank);
    sk_dft.dft(&module, &sk);

    ct.encrypt_sk(
        &module,
        &pt_scalar,
        &sk_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let mut ct_glwe_fourier: GLWECiphertextFourier<Vec<u8>, FFT64> = GLWECiphertextFourier::new(&module, basek, k_ggsw, rank);
    let mut pt_dft: VecZnxDft<Vec<u8>, FFT64> = module.new_vec_znx_dft(1, ct.size());
    let mut pt_big: VecZnxBig<Vec<u8>, FFT64> = module.new_vec_znx_big(1, ct.size());

    (0..ct.rank() + 1).for_each(|col_j| {
        (0..ct.rows()).for_each(|row_i| {
            module.vec_znx_add_scalar_inplace(&mut pt_want, 0, row_i, &pt_scalar, 0);

            // mul with sk[col_j-1]
            if col_j > 0 {
                module.vec_znx_dft(&mut pt_dft, 0, &pt_want, 0);
                module.svp_apply_inplace(&mut pt_dft, 0, &sk_dft, col_j - 1);
                module.vec_znx_idft_tmp_a(&mut pt_big, 0, &mut pt_dft, 0);
                module.vec_znx_big_normalize(basek, &mut pt_want, 0, &pt_big, 0, scratch.borrow());
            }

            ct.get_row(&module, row_i, col_j, &mut ct_glwe_fourier);

            ct_glwe_fourier.decrypt(&module, &mut pt_have, &sk_dft, scratch.borrow());

            module.vec_znx_sub_ab_inplace(&mut pt_have, 0, &pt_want, 0);

            let std_pt: f64 = pt_have.data.std(0, basek) * (k_ggsw as f64).exp2();
            assert!((sigma - std_pt).abs() <= 0.2, "{} {}", sigma, std_pt);

            pt_want.data.zero();
        });
    });
}

fn test_external_product(log_n: usize, basek: usize, k_ggsw: usize, rank: usize, sigma: f64) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);

    let rows: usize = (k_ggsw + basek - 1) / basek;

    let mut ct_ggsw_rhs: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::new(&module, basek, k_ggsw, rows, rank);
    let mut ct_ggsw_lhs_in: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::new(&module, basek, k_ggsw, rows, rank);
    let mut ct_ggsw_lhs_out: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::new(&module, basek, k_ggsw, rows, rank);
    let mut pt_ggsw_lhs: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);
    let mut pt_ggsw_rhs: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    pt_ggsw_lhs.fill_ternary_prob(0, 0.5, &mut source_xs);

    let k: usize = 1;

    pt_ggsw_rhs.to_mut().raw_mut()[k] = 1; //X^{k}

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWESwitchingKey::encrypt_sk_scratch_space(&module, rank, ct_ggsw_rhs.size())
            | GLWECiphertextFourier::decrypt_scratch_space(&module, ct_ggsw_lhs_out.size())
            | GGSWCiphertext::encrypt_sk_scratch_space(&module, rank, ct_ggsw_lhs_in.size())
            | GGSWCiphertext::external_product_scratch_space(
                &module,
                ct_ggsw_lhs_out.size(),
                ct_ggsw_lhs_in.size(),
                ct_ggsw_rhs.size(),
                rank,
            ),
    );

    let mut sk: SecretKey<Vec<u8>> = SecretKey::new(&module, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::new(&module, rank);
    sk_dft.dft(&module, &sk);

    ct_ggsw_rhs.encrypt_sk(
        &module,
        &pt_ggsw_rhs,
        &sk_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_ggsw_lhs_in.encrypt_sk(
        &module,
        &pt_ggsw_lhs,
        &sk_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_ggsw_lhs_out.external_product(&module, &ct_ggsw_lhs_in, &ct_ggsw_rhs, scratch.borrow());

    let mut ct_glwe_fourier: GLWECiphertextFourier<Vec<u8>, FFT64> = GLWECiphertextFourier::new(&module, basek, k_ggsw, rank);
    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::new(&module, basek, k_ggsw);
    let mut pt_dft: VecZnxDft<Vec<u8>, FFT64> = module.new_vec_znx_dft(1, ct_ggsw_lhs_out.size());
    let mut pt_big: VecZnxBig<Vec<u8>, FFT64> = module.new_vec_znx_big(1, ct_ggsw_lhs_out.size());
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::new(&module, basek, k_ggsw);

    module.vec_znx_rotate_inplace(k as i64, &mut pt_ggsw_lhs, 0);

    (0..ct_ggsw_lhs_out.rank() + 1).for_each(|col_j| {
        (0..ct_ggsw_lhs_out.rows()).for_each(|row_i| {
            module.vec_znx_add_scalar_inplace(&mut pt_want, 0, row_i, &pt_ggsw_lhs, 0);

            if col_j > 0 {
                module.vec_znx_dft(&mut pt_dft, 0, &pt_want, 0);
                module.svp_apply_inplace(&mut pt_dft, 0, &sk_dft, col_j - 1);
                module.vec_znx_idft_tmp_a(&mut pt_big, 0, &mut pt_dft, 0);
                module.vec_znx_big_normalize(basek, &mut pt_want, 0, &pt_big, 0, scratch.borrow());
            }

            ct_ggsw_lhs_out.get_row(&module, row_i, col_j, &mut ct_glwe_fourier);
            ct_glwe_fourier.decrypt(&module, &mut pt, &sk_dft, scratch.borrow());

            module.vec_znx_sub_ab_inplace(&mut pt, 0, &pt_want, 0);

            let noise_have: f64 = pt.data.std(0, basek).log2();

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
                k_ggsw,
                k_ggsw,
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

fn test_external_product_inplace(log_n: usize, basek: usize, k_ggsw: usize, rank: usize, sigma: f64) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let rows: usize = (k_ggsw + basek - 1) / basek;

    let mut ct_ggsw_rhs: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::new(&module, basek, k_ggsw, rows, rank);
    let mut ct_ggsw_lhs: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::new(&module, basek, k_ggsw, rows, rank);
    let mut pt_ggsw_lhs: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);
    let mut pt_ggsw_rhs: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    pt_ggsw_lhs.fill_ternary_prob(0, 0.5, &mut source_xs);

    let k: usize = 1;

    pt_ggsw_rhs.to_mut().raw_mut()[k] = 1; //X^{k}

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWESwitchingKey::encrypt_sk_scratch_space(&module, rank, ct_ggsw_rhs.size())
            | GLWECiphertextFourier::decrypt_scratch_space(&module, ct_ggsw_lhs.size())
            | GGSWCiphertext::encrypt_sk_scratch_space(&module, rank, ct_ggsw_lhs.size())
            | GGSWCiphertext::external_product_inplace_scratch_space(&module, ct_ggsw_lhs.size(), ct_ggsw_rhs.size(), rank),
    );

    let mut sk: SecretKey<Vec<u8>> = SecretKey::new(&module, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::new(&module, rank);
    sk_dft.dft(&module, &sk);

    ct_ggsw_rhs.encrypt_sk(
        &module,
        &pt_ggsw_rhs,
        &sk_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_ggsw_lhs.encrypt_sk(
        &module,
        &pt_ggsw_lhs,
        &sk_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_ggsw_lhs.external_product_inplace(&module, &ct_ggsw_rhs, scratch.borrow());

    let mut ct_glwe_fourier: GLWECiphertextFourier<Vec<u8>, FFT64> = GLWECiphertextFourier::new(&module, basek, k_ggsw, rank);
    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::new(&module, basek, k_ggsw);
    let mut pt_dft: VecZnxDft<Vec<u8>, FFT64> = module.new_vec_znx_dft(1, ct_ggsw_lhs.size());
    let mut pt_big: VecZnxBig<Vec<u8>, FFT64> = module.new_vec_znx_big(1, ct_ggsw_lhs.size());
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::new(&module, basek, k_ggsw);

    module.vec_znx_rotate_inplace(k as i64, &mut pt_ggsw_lhs, 0);

    (0..ct_ggsw_lhs.rank() + 1).for_each(|col_j| {
        (0..ct_ggsw_lhs.rows()).for_each(|row_i| {
            module.vec_znx_add_scalar_inplace(&mut pt_want, 0, row_i, &pt_ggsw_lhs, 0);

            if col_j > 0 {
                module.vec_znx_dft(&mut pt_dft, 0, &pt_want, 0);
                module.svp_apply_inplace(&mut pt_dft, 0, &sk_dft, col_j - 1);
                module.vec_znx_idft_tmp_a(&mut pt_big, 0, &mut pt_dft, 0);
                module.vec_znx_big_normalize(basek, &mut pt_want, 0, &pt_big, 0, scratch.borrow());
            }

            ct_ggsw_lhs.get_row(&module, row_i, col_j, &mut ct_glwe_fourier);
            ct_glwe_fourier.decrypt(&module, &mut pt, &sk_dft, scratch.borrow());

            module.vec_znx_sub_ab_inplace(&mut pt, 0, &pt_want, 0);

            let noise_have: f64 = pt.data.std(0, basek).log2();

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
                k_ggsw,
                k_ggsw,
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
pub(crate) fn noise_ggsw_product(
    n: f64,
    basek: usize,
    var_xs: f64,
    var_msg: f64,
    var_a0_err: f64,
    var_a1_err: f64,
    var_gct_err_lhs: f64,
    var_gct_err_rhs: f64,
    rank: f64,
    a_logq: usize,
    b_logq: usize,
) -> f64 {
    let a_logq: usize = a_logq.min(b_logq);
    let a_cols: usize = (a_logq + basek - 1) / basek;

    let b_scale = 2.0f64.powi(b_logq as i32);
    let a_scale: f64 = 2.0f64.powi((b_logq - a_logq) as i32);

    let base: f64 = (1 << (basek)) as f64;
    let var_base: f64 = base * base / 12f64;

    // lhs = a_cols * n * (var_base * var_gct_err_lhs + var_e_a * var_msg * p^2)
    // rhs = a_cols * n * var_base * var_gct_err_rhs * var_xs
    let mut noise: f64 = (rank + 1.0) * (a_cols as f64) * n * var_base * (var_gct_err_lhs + var_xs * var_gct_err_rhs);
    noise += var_msg * var_a0_err * a_scale * a_scale * n;
    noise += var_msg * var_a1_err * a_scale * a_scale * n * var_xs * rank;
    noise = noise.sqrt();
    noise /= b_scale;
    noise.log2().min(-1.0) // max noise is [-2^{-1}, 2^{-1}]
}
