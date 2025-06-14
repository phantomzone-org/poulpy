use backend::{FFT64, Module, ScalarZnx, ScalarZnxAlloc, ScalarZnxToMut, ScratchOwned, Stats, VecZnxOps, ZnxViewMut};
use sampling::source::Source;

use crate::{
    GGSWCiphertext, GLWECiphertextFourier, GLWEPlaintext, GLWESecret, GLWESwitchingKey, GetRow, Infos,
    test_fft64::ggsw::noise_ggsw_product,
};

#[test]
fn encrypt_sk() {
    (1..4).for_each(|rank_in| {
        (1..4).for_each(|rank_out| {
            println!("test encrypt_sk rank_in rank_out: {} {}", rank_in, rank_out);
            test_encrypt_sk(12, 8, 54, 3.2, rank_in, rank_out);
        });
    });
}

#[test]
fn key_switch() {
    (1..4).for_each(|rank_in_s0s1| {
        (1..4).for_each(|rank_out_s0s1| {
            (1..4).for_each(|rank_out_s1s2| {
                println!(
                    "test key_switch : ({},{},{})",
                    rank_in_s0s1, rank_out_s0s1, rank_out_s1s2
                );
                test_key_switch(12, 15, 60, 3.2, rank_in_s0s1, rank_out_s0s1, rank_out_s1s2);
            })
        });
    });
}

#[test]
fn key_switch_inplace() {
    (1..4).for_each(|rank_in_s0s1| {
        (1..4).for_each(|rank_out_s0s1| {
            println!(
                "test key_switch_inplace : ({},{})",
                rank_in_s0s1, rank_out_s0s1
            );
            test_key_switch_inplace(12, 15, 60, 3.2, rank_in_s0s1, rank_out_s0s1);
        });
    });
}

#[test]
fn external_product() {
    (1..4).for_each(|rank_in| {
        (1..4).for_each(|rank_out| {
            println!("test external_product rank: {} {}", rank_in, rank_out);
            test_external_product(12, 12, 60, 3.2, rank_in, rank_out);
        });
    });
}

#[test]
fn external_product_inplace() {
    (1..4).for_each(|rank_in| {
        (1..4).for_each(|rank_out| {
            println!(
                "test external_product_inplace rank: {} {}",
                rank_in, rank_out
            );
            test_external_product_inplace(12, 12, 60, 3.2, rank_in, rank_out);
        });
    });
}

fn test_encrypt_sk(log_n: usize, basek: usize, k_ksk: usize, sigma: f64, rank_in: usize, rank_out: usize) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let rows = (k_ksk + basek - 1) / basek;

    let mut ksk: GLWESwitchingKey<Vec<u8>, FFT64> = GLWESwitchingKey::alloc(&module, basek, k_ksk, rows, rank_in, rank_out);
    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_ksk);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWESwitchingKey::encrypt_sk_scratch_space(&module, basek, k_ksk, rank_out)
            | GLWECiphertextFourier::decrypt_scratch_space(&module, basek, k_ksk),
    );

    let mut sk_in: GLWESecret<Vec<u8>, FFT64> = GLWESecret::alloc(&module, rank_in);
    sk_in.fill_ternary_prob(&module, 0.5, &mut source_xs);

    let mut sk_out: GLWESecret<Vec<u8>, FFT64> = GLWESecret::alloc(&module, rank_out);
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

    let mut ct_glwe_fourier: GLWECiphertextFourier<Vec<u8>, FFT64> =
        GLWECiphertextFourier::alloc(&module, basek, k_ksk, rank_out);

    (0..ksk.rank_in()).for_each(|col_i| {
        (0..ksk.rows()).for_each(|row_i| {
            ksk.get_row(&module, row_i, col_i, &mut ct_glwe_fourier);
            ct_glwe_fourier.decrypt(&module, &mut pt, &sk_out, scratch.borrow());
            module.vec_znx_sub_scalar_inplace(&mut pt.data, 0, row_i, &sk_in.data, col_i);
            let std_pt: f64 = pt.data.std(0, basek) * (k_ksk as f64).exp2();
            assert!((sigma - std_pt).abs() <= 0.2, "{} {}", sigma, std_pt);
        });
    });
}

fn test_key_switch(
    log_n: usize,
    basek: usize,
    k_ksk: usize,
    sigma: f64,
    rank_in_s0s1: usize,
    rank_out_s0s1: usize,
    rank_out_s1s2: usize,
) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let rows = (k_ksk + basek - 1) / basek;

    let mut ct_gglwe_s0s1: GLWESwitchingKey<Vec<u8>, FFT64> =
        GLWESwitchingKey::alloc(&module, basek, k_ksk, rows, rank_in_s0s1, rank_out_s0s1);
    let mut ct_gglwe_s1s2: GLWESwitchingKey<Vec<u8>, FFT64> =
        GLWESwitchingKey::alloc(&module, basek, k_ksk, rows, rank_out_s0s1, rank_out_s1s2);
    let mut ct_gglwe_s0s2: GLWESwitchingKey<Vec<u8>, FFT64> =
        GLWESwitchingKey::alloc(&module, basek, k_ksk, rows, rank_in_s0s1, rank_out_s1s2);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWESwitchingKey::encrypt_sk_scratch_space(&module, basek, k_ksk, rank_in_s0s1 | rank_out_s0s1)
            | GLWECiphertextFourier::decrypt_scratch_space(&module, basek, k_ksk)
            | GLWESwitchingKey::keyswitch_scratch_space(
                &module,
                basek,
                ct_gglwe_s0s2.k(),
                ct_gglwe_s0s2.rank(),
                ct_gglwe_s0s1.k(),
                ct_gglwe_s0s1.rank(),
                ct_gglwe_s1s2.k(),
            ),
    );

    let mut sk0: GLWESecret<Vec<u8>, FFT64> = GLWESecret::alloc(&module, rank_in_s0s1);
    sk0.fill_ternary_prob(&module, 0.5, &mut source_xs);

    let mut sk1: GLWESecret<Vec<u8>, FFT64> = GLWESecret::alloc(&module, rank_out_s0s1);
    sk1.fill_ternary_prob(&module, 0.5, &mut source_xs);

    let mut sk2: GLWESecret<Vec<u8>, FFT64> = GLWESecret::alloc(&module, rank_out_s1s2);
    sk2.fill_ternary_prob(&module, 0.5, &mut source_xs);

    // gglwe_{s1}(s0) = s0 -> s1
    ct_gglwe_s0s1.generate_from_sk(
        &module,
        &sk0,
        &sk1,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    // gglwe_{s2}(s1) -> s1 -> s2
    ct_gglwe_s1s2.generate_from_sk(
        &module,
        &sk1,
        &sk2,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    // gglwe_{s1}(s0) (x) gglwe_{s2}(s1) = gglwe_{s2}(s0)
    ct_gglwe_s0s2.keyswitch(&module, &ct_gglwe_s0s1, &ct_gglwe_s1s2, scratch.borrow());

    let mut ct_glwe_dft: GLWECiphertextFourier<Vec<u8>, FFT64> =
        GLWECiphertextFourier::alloc(&module, basek, k_ksk, rank_out_s1s2);
    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_ksk);

    (0..ct_gglwe_s0s2.rank_in()).for_each(|col_i| {
        (0..ct_gglwe_s0s2.rows()).for_each(|row_i| {
            ct_gglwe_s0s2.get_row(&module, row_i, col_i, &mut ct_glwe_dft);
            ct_glwe_dft.decrypt(&module, &mut pt, &sk2, scratch.borrow());
            module.vec_znx_sub_scalar_inplace(&mut pt.data, 0, row_i, &sk0.data, col_i);

            let noise_have: f64 = pt.data.std(0, basek).log2();
            let noise_want: f64 = log2_std_noise_gglwe_product(
                module.n() as f64,
                basek,
                0.5,
                0.5,
                0f64,
                sigma * sigma,
                0f64,
                rank_out_s0s1 as f64,
                k_ksk,
                k_ksk,
            );

            assert!(
                (noise_have - noise_want).abs() <= 0.1,
                "{} {}",
                noise_have,
                noise_want
            );
        });
    });
}

fn test_key_switch_inplace(log_n: usize, basek: usize, k_ksk: usize, sigma: f64, rank_in_s0s1: usize, rank_out_s0s1: usize) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let rows: usize = (k_ksk + basek - 1) / basek;

    let mut ct_gglwe_s0s1: GLWESwitchingKey<Vec<u8>, FFT64> =
        GLWESwitchingKey::alloc(&module, basek, k_ksk, rows, rank_in_s0s1, rank_out_s0s1);
    let mut ct_gglwe_s1s2: GLWESwitchingKey<Vec<u8>, FFT64> =
        GLWESwitchingKey::alloc(&module, basek, k_ksk, rows, rank_out_s0s1, rank_out_s0s1);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWESwitchingKey::encrypt_sk_scratch_space(&module, basek, k_ksk, rank_out_s0s1)
            | GLWECiphertextFourier::decrypt_scratch_space(&module, basek, k_ksk)
            | GLWESwitchingKey::keyswitch_inplace_scratch_space(
                &module,
                basek,
                ct_gglwe_s0s1.k(),
                ct_gglwe_s0s1.rank(),
                ct_gglwe_s1s2.k(),
            ),
    );

    let mut sk0: GLWESecret<Vec<u8>, FFT64> = GLWESecret::alloc(&module, rank_in_s0s1);
    sk0.fill_ternary_prob(&module, 0.5, &mut source_xs);

    let mut sk1: GLWESecret<Vec<u8>, FFT64> = GLWESecret::alloc(&module, rank_out_s0s1);
    sk1.fill_ternary_prob(&module, 0.5, &mut source_xs);

    let mut sk2: GLWESecret<Vec<u8>, FFT64> = GLWESecret::alloc(&module, rank_out_s0s1);
    sk2.fill_ternary_prob(&module, 0.5, &mut source_xs);

    // gglwe_{s1}(s0) = s0 -> s1
    ct_gglwe_s0s1.generate_from_sk(
        &module,
        &sk0,
        &sk1,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    // gglwe_{s2}(s1) -> s1 -> s2
    ct_gglwe_s1s2.generate_from_sk(
        &module,
        &sk1,
        &sk2,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    // gglwe_{s1}(s0) (x) gglwe_{s2}(s1) = gglwe_{s2}(s0)
    ct_gglwe_s0s1.keyswitch_inplace(&module, &ct_gglwe_s1s2, scratch.borrow());

    let ct_gglwe_s0s2: GLWESwitchingKey<Vec<u8>, FFT64> = ct_gglwe_s0s1;

    let mut ct_glwe_dft: GLWECiphertextFourier<Vec<u8>, FFT64> =
        GLWECiphertextFourier::alloc(&module, basek, k_ksk, rank_out_s0s1);
    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_ksk);

    (0..ct_gglwe_s0s2.rank_in()).for_each(|col_i| {
        (0..ct_gglwe_s0s2.rows()).for_each(|row_i| {
            ct_gglwe_s0s2.get_row(&module, row_i, col_i, &mut ct_glwe_dft);
            ct_glwe_dft.decrypt(&module, &mut pt, &sk2, scratch.borrow());
            module.vec_znx_sub_scalar_inplace(&mut pt.data, 0, row_i, &sk0.data, col_i);

            let noise_have: f64 = pt.data.std(0, basek).log2();
            let noise_want: f64 = log2_std_noise_gglwe_product(
                module.n() as f64,
                basek,
                0.5,
                0.5,
                0f64,
                sigma * sigma,
                0f64,
                rank_out_s0s1 as f64,
                k_ksk,
                k_ksk,
            );

            assert!(
                (noise_have - noise_want).abs() <= 0.1,
                "{} {}",
                noise_have,
                noise_want
            );
        });
    });
}

fn test_external_product(log_n: usize, basek: usize, k: usize, sigma: f64, rank_in: usize, rank_out: usize) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);

    let rows: usize = (k + basek - 1) / basek;

    let mut ct_gglwe_in: GLWESwitchingKey<Vec<u8>, FFT64> = GLWESwitchingKey::alloc(&module, basek, k, rows, rank_in, rank_out);
    let mut ct_gglwe_out: GLWESwitchingKey<Vec<u8>, FFT64> = GLWESwitchingKey::alloc(&module, basek, k, rows, rank_in, rank_out);
    let mut ct_rgsw: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::alloc(&module, basek, k, rows, rank_out);

    let mut pt_rgsw: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWESwitchingKey::encrypt_sk_scratch_space(&module, basek, k, rank_out)
            | GLWECiphertextFourier::decrypt_scratch_space(&module, basek, k)
            | GLWESwitchingKey::external_product_scratch_space(
                &module,
                basek,
                ct_gglwe_out.k(),
                ct_gglwe_in.k(),
                ct_rgsw.k(),
                rank_out,
            )
            | GGSWCiphertext::encrypt_sk_scratch_space(&module, basek, k, rank_out),
    );

    let r: usize = 1;

    pt_rgsw.to_mut().raw_mut()[r] = 1; // X^{r}

    let mut sk_in: GLWESecret<Vec<u8>, FFT64> = GLWESecret::alloc(&module, rank_in);
    sk_in.fill_ternary_prob(&module, 0.5, &mut source_xs);

    let mut sk_out: GLWESecret<Vec<u8>, FFT64> = GLWESecret::alloc(&module, rank_out);
    sk_out.fill_ternary_prob(&module, 0.5, &mut source_xs);

    // gglwe_{s1}(s0) = s0 -> s1
    ct_gglwe_in.generate_from_sk(
        &module,
        &sk_in,
        &sk_out,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_rgsw.encrypt_sk(
        &module,
        &pt_rgsw,
        &sk_out,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    // gglwe_(m) (x) RGSW_(X^k) = gglwe_(m * X^k)
    ct_gglwe_out.external_product(&module, &ct_gglwe_in, &ct_rgsw, scratch.borrow());

    scratch = ScratchOwned::new(
        GLWESwitchingKey::encrypt_sk_scratch_space(&module, basek, k, rank_out)
            | GLWECiphertextFourier::decrypt_scratch_space(&module, basek, k)
            | GLWESwitchingKey::external_product_scratch_space(
                &module,
                basek,
                ct_gglwe_out.k(),
                ct_gglwe_in.k(),
                ct_rgsw.k(),
                rank_out,
            )
            | GGSWCiphertext::encrypt_sk_scratch_space(&module, basek, k, rank_out),
    );

    let mut ct_glwe_dft: GLWECiphertextFourier<Vec<u8>, FFT64> = GLWECiphertextFourier::alloc(&module, basek, k, rank_out);
    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k);

    (0..rank_in).for_each(|i| {
        module.vec_znx_rotate_inplace(r as i64, &mut sk_in.data, i); // * X^{r}
    });

    (0..rank_in).for_each(|col_i| {
        (0..ct_gglwe_out.rows()).for_each(|row_i| {
            ct_gglwe_out.get_row(&module, row_i, col_i, &mut ct_glwe_dft);
            ct_glwe_dft.decrypt(&module, &mut pt, &sk_out, scratch.borrow());
            module.vec_znx_sub_scalar_inplace(&mut pt.data, 0, row_i, &sk_in.data, col_i);

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
                rank_out as f64,
                k,
                k,
            );

            assert!(
                (noise_have - noise_want).abs() <= 0.1,
                "{} {}",
                noise_have,
                noise_want
            );
        });
    });
}

fn test_external_product_inplace(log_n: usize, basek: usize, k: usize, sigma: f64, rank_in: usize, rank_out: usize) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);

    let rows: usize = (k + basek - 1) / basek;

    let mut ct_gglwe: GLWESwitchingKey<Vec<u8>, FFT64> = GLWESwitchingKey::alloc(&module, basek, k, rows, rank_in, rank_out);
    let mut ct_rgsw: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::alloc(&module, basek, k, rows, rank_out);

    let mut pt_rgsw: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWESwitchingKey::encrypt_sk_scratch_space(&module, basek, k, rank_out)
            | GLWECiphertextFourier::decrypt_scratch_space(&module, basek, k)
            | GLWESwitchingKey::external_product_inplace_scratch_space(&module, basek, k, k, rank_out)
            | GGSWCiphertext::encrypt_sk_scratch_space(&module, basek, k, rank_out),
    );

    let r: usize = 1;

    pt_rgsw.to_mut().raw_mut()[r] = 1; // X^{r}

    let mut sk_in: GLWESecret<Vec<u8>, FFT64> = GLWESecret::alloc(&module, rank_in);
    sk_in.fill_ternary_prob(&module, 0.5, &mut source_xs);

    let mut sk_out: GLWESecret<Vec<u8>, FFT64> = GLWESecret::alloc(&module, rank_out);
    sk_out.fill_ternary_prob(&module, 0.5, &mut source_xs);

    // gglwe_{s1}(s0) = s0 -> s1
    ct_gglwe.generate_from_sk(
        &module,
        &sk_in,
        &sk_out,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_rgsw.encrypt_sk(
        &module,
        &pt_rgsw,
        &sk_out,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    // gglwe_(m) (x) RGSW_(X^k) = gglwe_(m * X^k)
    ct_gglwe.external_product_inplace(&module, &ct_rgsw, scratch.borrow());

    let mut ct_glwe_dft: GLWECiphertextFourier<Vec<u8>, FFT64> = GLWECiphertextFourier::alloc(&module, basek, k, rank_out);
    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k);

    (0..rank_in).for_each(|i| {
        module.vec_znx_rotate_inplace(r as i64, &mut sk_in.data, i); // * X^{r}
    });

    (0..rank_in).for_each(|col_i| {
        (0..ct_gglwe.rows()).for_each(|row_i| {
            ct_gglwe.get_row(&module, row_i, col_i, &mut ct_glwe_dft);
            ct_glwe_dft.decrypt(&module, &mut pt, &sk_out, scratch.borrow());
            module.vec_znx_sub_scalar_inplace(&mut pt.data, 0, row_i, &sk_in.data, col_i);

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
                rank_out as f64,
                k,
                k,
            );

            assert!(
                (noise_have - noise_want).abs() <= 0.1,
                "{} {}",
                noise_have,
                noise_want
            );
        });
    });
}

pub(crate) fn var_noise_gglwe_product(
    n: f64,
    basek: usize,
    var_xs: f64,
    var_msg: f64,
    var_a_err: f64,
    var_gct_err_lhs: f64,
    var_gct_err_rhs: f64,
    rank_in: f64,
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
    let mut noise: f64 = (a_cols as f64) * n * var_base * (var_gct_err_lhs + var_xs * var_gct_err_rhs);
    noise += var_msg * var_a_err * a_scale * a_scale * n;
    noise *= rank_in;
    noise /= b_scale * b_scale;
    noise
}

pub(crate) fn log2_std_noise_gglwe_product(
    n: f64,
    basek: usize,
    var_xs: f64,
    var_msg: f64,
    var_a_err: f64,
    var_gct_err_lhs: f64,
    var_gct_err_rhs: f64,
    rank_in: f64,
    a_logq: usize,
    b_logq: usize,
) -> f64 {
    let mut noise: f64 = var_noise_gglwe_product(
        n,
        basek,
        var_xs,
        var_msg,
        var_a_err,
        var_gct_err_lhs,
        var_gct_err_rhs,
        rank_in,
        a_logq,
        b_logq,
    );
    noise = noise.sqrt();
    noise.log2().min(-1.0) // max noise is [-2^{-1}, 2^{-1}]
}
