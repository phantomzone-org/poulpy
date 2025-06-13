use backend::{FFT64, Module, ScalarZnx, ScalarZnxAlloc, ScalarZnxToMut, ScratchOwned, Stats, VecZnxOps, ZnxViewMut};
use sampling::source::Source;

use crate::{
    FourierGLWECiphertext, FourierGLWESecret, GGSWCiphertext, GLWEPlaintext, GLWESecret, GLWESwitchingKey, GetRow, Infos,
    noise::{log2_std_noise_gglwe_product, noise_ggsw_product},
};

#[test]
fn encrypt_sk() {
    let log_n: usize = 8;
    let basek: usize = 12;
    let k_ksk: usize = 54;
    let digits: usize = k_ksk / basek;
    (1..4).for_each(|rank_in| {
        (1..4).for_each(|rank_out| {
            (1..digits + 1).for_each(|di| {
                println!(
                    "test encrypt_sk digits: {} ranks: ({} {})",
                    di, rank_in, rank_out
                );
                test_encrypt_sk(log_n, basek, k_ksk, di, rank_in, rank_out, 3.2);
            });
        });
    });
}

#[test]
fn key_switch() {
    let log_n: usize = 8;
    let basek: usize = 12;
    let k_in: usize = 60;
    let digits: usize = k_in.div_ceil(basek);
    (1..4).for_each(|rank_in_s0s1| {
        (1..4).for_each(|rank_out_s0s1| {
            (1..4).for_each(|rank_out_s1s2| {
                (1..digits + 1).for_each(|di| {
                    let k_ksk: usize = k_in + basek * di;
                    println!(
                        "test key_switch digits: {} ranks: ({},{},{})",
                        di, rank_in_s0s1, rank_out_s0s1, rank_out_s1s2
                    );
                    let k_out: usize = k_ksk; // Better capture noise.
                    test_key_switch(
                        log_n,
                        basek,
                        k_out,
                        k_in,
                        k_ksk,
                        di,
                        rank_in_s0s1,
                        rank_out_s0s1,
                        rank_out_s1s2,
                        3.2,
                    );
                })
            })
        });
    });
}

#[test]
fn key_switch_inplace() {
    let log_n: usize = 8;
    let basek: usize = 12;
    let k_ct: usize = 60;
    let digits: usize = k_ct.div_ceil(basek);
    (1..4).for_each(|rank_in_s0s1| {
        (1..4).for_each(|rank_out_s0s1| {
            (1..digits + 1).for_each(|di| {
                let k_ksk: usize = k_ct + basek * di;
                println!(
                    "test key_switch_inplace digits: {} ranks: ({},{})",
                    di, rank_in_s0s1, rank_out_s0s1
                );
                test_key_switch_inplace(
                    log_n,
                    basek,
                    k_ct,
                    k_ksk,
                    di,
                    rank_in_s0s1,
                    rank_out_s0s1,
                    3.2,
                );
            });
        });
    });
}

#[test]
fn external_product() {
    let log_n: usize = 8;
    let basek: usize = 12;
    let k_in: usize = 60;
    let digits: usize = k_in.div_ceil(basek);
    (1..4).for_each(|rank_in| {
        (1..4).for_each(|rank_out| {
            (1..digits + 1).for_each(|di| {
                let k_ggsw: usize = k_in + basek * di;
                println!(
                    "test external_product digits: {} ranks: ({} {})",
                    di, rank_in, rank_out
                );
                let k_out: usize = k_in; // Better capture noise.
                test_external_product(
                    log_n, basek, k_out, k_in, k_ggsw, di, rank_in, rank_out, 3.2,
                );
            });
        });
    });
}

#[test]
fn external_product_inplace() {
    let log_n: usize = 5;
    let basek: usize = 12;
    let k_ct: usize = 60;
    let digits: usize = k_ct.div_ceil(basek);
    (1..4).for_each(|rank_in| {
        (1..4).for_each(|rank_out| {
            (1..digits).for_each(|di| {
                let k_ggsw: usize = k_ct + basek * di;
                println!(
                    "test external_product_inplace digits: {} ranks: ({} {})",
                    di, rank_in, rank_out
                );
                test_external_product_inplace(log_n, basek, k_ct, k_ggsw, di, rank_in, rank_out, 3.2);
            });
        });
    });
}

fn test_encrypt_sk(log_n: usize, basek: usize, k_ksk: usize, digits: usize, rank_in: usize, rank_out: usize, sigma: f64) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let rows: usize = (k_ksk - digits * basek) / (digits * basek);

    let mut ksk: GLWESwitchingKey<Vec<u8>, FFT64> =
        GLWESwitchingKey::alloc(&module, basek, k_ksk, rows, digits, rank_in, rank_out);
    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_ksk);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWESwitchingKey::encrypt_sk_scratch_space(&module, basek, k_ksk, rank_out)
            | FourierGLWECiphertext::decrypt_scratch_space(&module, basek, k_ksk),
    );

    let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc(&module, rank_in);
    sk_in.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc(&module, rank_out);
    sk_out.fill_ternary_prob(0.5, &mut source_xs);
    let sk_out_dft: FourierGLWESecret<Vec<u8>, FFT64> = FourierGLWESecret::from(&module, &sk_out);

    ksk.generate_from_sk(
        &module,
        &sk_in,
        &sk_out_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let mut ct_glwe_fourier: FourierGLWECiphertext<Vec<u8>, FFT64> =
        FourierGLWECiphertext::alloc(&module, basek, k_ksk, rank_out);

    (0..ksk.rank_in()).for_each(|col_i| {
        (0..ksk.rows()).for_each(|row_i| {
            ksk.get_row(&module, row_i, col_i, &mut ct_glwe_fourier);
            ct_glwe_fourier.decrypt(&module, &mut pt, &sk_out_dft, scratch.borrow());
            module.vec_znx_sub_scalar_inplace(
                &mut pt.data,
                0,
                (digits - 1) + row_i * digits,
                &sk_in.data,
                col_i,
            );
            let std_pt: f64 = pt.data.std(0, basek) * (k_ksk as f64).exp2();
            assert!((sigma - std_pt).abs() <= 0.5, "{} {}", sigma, std_pt);
        });
    });
}

fn test_key_switch(
    log_n: usize,
    basek: usize,
    k_out: usize,
    k_in: usize,
    k_ksk: usize,
    digits: usize,
    rank_in_s0s1: usize,
    rank_out_s0s1: usize,
    rank_out_s1s2: usize,
    sigma: f64,
) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let rows: usize = k_in.div_ceil(basek * digits);
    let digits_in: usize = 1;

    let mut ct_gglwe_s0s1: GLWESwitchingKey<Vec<u8>, FFT64> = GLWESwitchingKey::alloc(
        &module,
        basek,
        k_in,
        rows,
        digits_in,
        rank_in_s0s1,
        rank_out_s0s1,
    );
    let mut ct_gglwe_s1s2: GLWESwitchingKey<Vec<u8>, FFT64> = GLWESwitchingKey::alloc(
        &module,
        basek,
        k_ksk,
        rows,
        digits,
        rank_out_s0s1,
        rank_out_s1s2,
    );
    let mut ct_gglwe_s0s2: GLWESwitchingKey<Vec<u8>, FFT64> = GLWESwitchingKey::alloc(
        &module,
        basek,
        k_out,
        rows,
        digits_in,
        rank_in_s0s1,
        rank_out_s1s2,
    );

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWESwitchingKey::encrypt_sk_scratch_space(&module, basek, k_ksk, rank_in_s0s1 | rank_out_s0s1)
            | FourierGLWECiphertext::decrypt_scratch_space(&module, basek, k_out)
            | GLWESwitchingKey::keyswitch_scratch_space(
                &module,
                basek,
                k_out,
                k_in,
                k_ksk,
                digits,
                ct_gglwe_s1s2.rank_in(),
                ct_gglwe_s1s2.rank_out(),
            ),
    );

    let mut sk0: GLWESecret<Vec<u8>> = GLWESecret::alloc(&module, rank_in_s0s1);
    sk0.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk1: GLWESecret<Vec<u8>> = GLWESecret::alloc(&module, rank_out_s0s1);
    sk1.fill_ternary_prob(0.5, &mut source_xs);
    let sk1_dft: FourierGLWESecret<Vec<u8>, FFT64> = FourierGLWESecret::from(&module, &sk1);

    let mut sk2: GLWESecret<Vec<u8>> = GLWESecret::alloc(&module, rank_out_s1s2);
    sk2.fill_ternary_prob(0.5, &mut source_xs);
    let sk2_dft: FourierGLWESecret<Vec<u8>, FFT64> = FourierGLWESecret::from(&module, &sk2);

    // gglwe_{s1}(s0) = s0 -> s1
    ct_gglwe_s0s1.generate_from_sk(
        &module,
        &sk0,
        &sk1_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    // gglwe_{s2}(s1) -> s1 -> s2
    ct_gglwe_s1s2.generate_from_sk(
        &module,
        &sk1,
        &sk2_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    // gglwe_{s1}(s0) (x) gglwe_{s2}(s1) = gglwe_{s2}(s0)
    ct_gglwe_s0s2.keyswitch(&module, &ct_gglwe_s0s1, &ct_gglwe_s1s2, scratch.borrow());

    let mut ct_glwe_dft: FourierGLWECiphertext<Vec<u8>, FFT64> =
        FourierGLWECiphertext::alloc(&module, basek, k_out, rank_out_s1s2);
    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_out);

    (0..ct_gglwe_s0s2.rank_in()).for_each(|col_i| {
        (0..ct_gglwe_s0s2.rows()).for_each(|row_i| {
            ct_gglwe_s0s2.get_row(&module, row_i, col_i, &mut ct_glwe_dft);
            ct_glwe_dft.decrypt(&module, &mut pt, &sk2_dft, scratch.borrow());
            module.vec_znx_sub_scalar_inplace(
                &mut pt.data,
                0,
                (digits_in - 1) + row_i * digits_in,
                &sk0.data,
                col_i,
            );

            let noise_have: f64 = pt.data.std(0, basek).log2();
            let noise_want: f64 = log2_std_noise_gglwe_product(
                module.n() as f64,
                basek * digits,
                0.5,
                0.5,
                0f64,
                sigma * sigma,
                0f64,
                rank_out_s0s1 as f64,
                k_in,
                k_ksk,
            );

            assert!(
                (noise_have - noise_want).abs() <= 1.0,
                "{} {}",
                noise_have,
                noise_want
            );
        });
    });
}

fn test_key_switch_inplace(
    log_n: usize,
    basek: usize,
    k_ct: usize,
    k_ksk: usize,
    digits: usize,
    rank_in: usize,
    rank_out: usize,
    sigma: f64,
) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let rows: usize = k_ct.div_ceil(basek * digits);
    let digits_in: usize = 1;

    let mut ct_gglwe_s0s1: GLWESwitchingKey<Vec<u8>, FFT64> =
        GLWESwitchingKey::alloc(&module, basek, k_ct, rows, digits_in, rank_in, rank_out);
    let mut ct_gglwe_s1s2: GLWESwitchingKey<Vec<u8>, FFT64> =
        GLWESwitchingKey::alloc(&module, basek, k_ksk, rows, digits, rank_out, rank_out);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWESwitchingKey::encrypt_sk_scratch_space(&module, basek, k_ksk, rank_out)
            | FourierGLWECiphertext::decrypt_scratch_space(&module, basek, k_ksk)
            | GLWESwitchingKey::keyswitch_inplace_scratch_space(&module, basek, k_ct, k_ksk, digits, rank_out),
    );

    let mut sk0: GLWESecret<Vec<u8>> = GLWESecret::alloc(&module, rank_in);
    sk0.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk1: GLWESecret<Vec<u8>> = GLWESecret::alloc(&module, rank_out);
    sk1.fill_ternary_prob(0.5, &mut source_xs);
    let sk1_dft: FourierGLWESecret<Vec<u8>, FFT64> = FourierGLWESecret::from(&module, &sk1);

    let mut sk2: GLWESecret<Vec<u8>> = GLWESecret::alloc(&module, rank_out);
    sk2.fill_ternary_prob(0.5, &mut source_xs);
    let sk2_dft: FourierGLWESecret<Vec<u8>, FFT64> = FourierGLWESecret::from(&module, &sk2);

    // gglwe_{s1}(s0) = s0 -> s1
    ct_gglwe_s0s1.generate_from_sk(
        &module,
        &sk0,
        &sk1_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    // gglwe_{s2}(s1) -> s1 -> s2
    ct_gglwe_s1s2.generate_from_sk(
        &module,
        &sk1,
        &sk2_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    // gglwe_{s1}(s0) (x) gglwe_{s2}(s1) = gglwe_{s2}(s0)
    ct_gglwe_s0s1.keyswitch_inplace(&module, &ct_gglwe_s1s2, scratch.borrow());

    let ct_gglwe_s0s2: GLWESwitchingKey<Vec<u8>, FFT64> = ct_gglwe_s0s1;

    let mut ct_glwe_dft: FourierGLWECiphertext<Vec<u8>, FFT64> = FourierGLWECiphertext::alloc(&module, basek, k_ct, rank_out);
    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_ct);

    (0..ct_gglwe_s0s2.rank_in()).for_each(|col_i| {
        (0..ct_gglwe_s0s2.rows()).for_each(|row_i| {
            ct_gglwe_s0s2.get_row(&module, row_i, col_i, &mut ct_glwe_dft);
            ct_glwe_dft.decrypt(&module, &mut pt, &sk2_dft, scratch.borrow());
            module.vec_znx_sub_scalar_inplace(
                &mut pt.data,
                0,
                (digits_in - 1) + row_i * digits_in,
                &sk0.data,
                col_i,
            );

            let noise_have: f64 = pt.data.std(0, basek).log2();
            let noise_want: f64 = log2_std_noise_gglwe_product(
                module.n() as f64,
                basek * digits,
                0.5,
                0.5,
                0f64,
                sigma * sigma,
                0f64,
                rank_out as f64,
                k_ct,
                k_ksk,
            );

            assert!(
                (noise_have - noise_want).abs() <= 1.0,
                "{} {}",
                noise_have,
                noise_want
            );
        });
    });
}

fn test_external_product(
    log_n: usize,
    basek: usize,
    k_out: usize,
    k_in: usize,
    k_ggsw: usize,
    digits: usize,
    rank_in: usize,
    rank_out: usize,
    sigma: f64,
) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);

    let rows: usize = k_in.div_ceil(basek * digits);
    let digits_in: usize = 1;

    let mut ct_gglwe_in: GLWESwitchingKey<Vec<u8>, FFT64> =
        GLWESwitchingKey::alloc(&module, basek, k_in, rows, digits_in, rank_in, rank_out);
    let mut ct_gglwe_out: GLWESwitchingKey<Vec<u8>, FFT64> =
        GLWESwitchingKey::alloc(&module, basek, k_out, rows, digits_in, rank_in, rank_out);
    let mut ct_rgsw: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::alloc(&module, basek, k_ggsw, rows, digits, rank_out);

    let mut pt_rgsw: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWESwitchingKey::encrypt_sk_scratch_space(&module, basek, k_in, rank_out)
            | FourierGLWECiphertext::decrypt_scratch_space(&module, basek, k_out)
            | GLWESwitchingKey::external_product_scratch_space(&module, basek, k_out, k_in, k_ggsw, digits, rank_out)
            | GGSWCiphertext::encrypt_sk_scratch_space(&module, basek, k_ggsw, rank_out),
    );

    let r: usize = 1;

    pt_rgsw.to_mut().raw_mut()[r] = 1; // X^{r}

    let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc(&module, rank_in);
    sk_in.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc(&module, rank_out);
    sk_out.fill_ternary_prob(0.5, &mut source_xs);
    let sk_out_dft: FourierGLWESecret<Vec<u8>, FFT64> = FourierGLWESecret::from(&module, &sk_out);

    // gglwe_{s1}(s0) = s0 -> s1
    ct_gglwe_in.generate_from_sk(
        &module,
        &sk_in,
        &sk_out_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_rgsw.encrypt_sk(
        &module,
        &pt_rgsw,
        &sk_out_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    // gglwe_(m) (x) RGSW_(X^k) = gglwe_(m * X^k)
    ct_gglwe_out.external_product(&module, &ct_gglwe_in, &ct_rgsw, scratch.borrow());

    let mut ct_glwe_dft: FourierGLWECiphertext<Vec<u8>, FFT64> = FourierGLWECiphertext::alloc(&module, basek, k_out, rank_out);
    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_out);

    (0..rank_in).for_each(|i| {
        module.vec_znx_rotate_inplace(r as i64, &mut sk_in.data, i); // * X^{r}
    });

    (0..rank_in).for_each(|col_i| {
        (0..ct_gglwe_out.rows()).for_each(|row_i| {
            ct_gglwe_out.get_row(&module, row_i, col_i, &mut ct_glwe_dft);
            ct_glwe_dft.decrypt(&module, &mut pt, &sk_out_dft, scratch.borrow());

            module.vec_znx_sub_scalar_inplace(
                &mut pt.data,
                0,
                (digits_in - 1) + row_i * digits_in,
                &sk_in.data,
                col_i,
            );

            let noise_have: f64 = pt.data.std(0, basek).log2();

            let var_gct_err_lhs: f64 = sigma * sigma;
            let var_gct_err_rhs: f64 = 0f64;

            let var_msg: f64 = 1f64 / module.n() as f64; // X^{k}
            let var_a0_err: f64 = sigma * sigma;
            let var_a1_err: f64 = 1f64 / 12f64;

            let noise_want: f64 = noise_ggsw_product(
                module.n() as f64,
                basek * digits,
                0.5,
                var_msg,
                var_a0_err,
                var_a1_err,
                var_gct_err_lhs,
                var_gct_err_rhs,
                rank_out as f64,
                k_in,
                k_ggsw,
            );

            assert!(
                (noise_have - noise_want).abs() <= 1.0,
                "{} {}",
                noise_have,
                noise_want
            );
        });
    });
}

fn test_external_product_inplace(
    log_n: usize,
    basek: usize,
    k_ct: usize,
    k_ggsw: usize,
    digits: usize,
    rank_in: usize,
    rank_out: usize,
    sigma: f64,
) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);

    let rows: usize = k_ct.div_ceil(basek * digits);

    let digits_in: usize = 1;

    let mut ct_gglwe: GLWESwitchingKey<Vec<u8>, FFT64> =
        GLWESwitchingKey::alloc(&module, basek, k_ct, rows, digits_in, rank_in, rank_out);
    let mut ct_rgsw: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::alloc(&module, basek, k_ggsw, rows, digits, rank_out);

    let mut pt_rgsw: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWESwitchingKey::encrypt_sk_scratch_space(&module, basek, k_ct, rank_out)
            | FourierGLWECiphertext::decrypt_scratch_space(&module, basek, k_ct)
            | GLWESwitchingKey::external_product_inplace_scratch_space(&module, basek, k_ct, k_ggsw, digits, rank_out)
            | GGSWCiphertext::encrypt_sk_scratch_space(&module, basek, k_ggsw, rank_out),
    );

    let r: usize = 1;

    pt_rgsw.to_mut().raw_mut()[r] = 1; // X^{r}

    let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc(&module, rank_in);
    sk_in.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc(&module, rank_out);
    sk_out.fill_ternary_prob(0.5, &mut source_xs);
    let sk_out_dft: FourierGLWESecret<Vec<u8>, FFT64> = FourierGLWESecret::from(&module, &sk_out);

    // gglwe_{s1}(s0) = s0 -> s1
    ct_gglwe.generate_from_sk(
        &module,
        &sk_in,
        &sk_out_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_rgsw.encrypt_sk(
        &module,
        &pt_rgsw,
        &sk_out_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    // gglwe_(m) (x) RGSW_(X^k) = gglwe_(m * X^k)
    ct_gglwe.external_product_inplace(&module, &ct_rgsw, scratch.borrow());

    let mut ct_glwe_dft: FourierGLWECiphertext<Vec<u8>, FFT64> = FourierGLWECiphertext::alloc(&module, basek, k_ct, rank_out);
    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_ct);

    (0..rank_in).for_each(|i| {
        module.vec_znx_rotate_inplace(r as i64, &mut sk_in.data, i); // * X^{r}
    });

    (0..rank_in).for_each(|col_i| {
        (0..ct_gglwe.rows()).for_each(|row_i| {
            ct_gglwe.get_row(&module, row_i, col_i, &mut ct_glwe_dft);
            ct_glwe_dft.decrypt(&module, &mut pt, &sk_out_dft, scratch.borrow());

            module.vec_znx_sub_scalar_inplace(
                &mut pt.data,
                0,
                (digits_in - 1) + row_i * digits_in,
                &sk_in.data,
                col_i,
            );

            let noise_have: f64 = pt.data.std(0, basek).log2();

            let var_gct_err_lhs: f64 = sigma * sigma;
            let var_gct_err_rhs: f64 = 0f64;

            let var_msg: f64 = 1f64 / module.n() as f64; // X^{k}
            let var_a0_err: f64 = sigma * sigma;
            let var_a1_err: f64 = 1f64 / 12f64;

            let noise_want: f64 = noise_ggsw_product(
                module.n() as f64,
                basek * digits,
                0.5,
                var_msg,
                var_a0_err,
                var_a1_err,
                var_gct_err_lhs,
                var_gct_err_rhs,
                rank_out as f64,
                k_ct,
                k_ggsw,
            );

            assert!(
                (noise_have - noise_want).abs() <= 1.0,
                "{} {}",
                noise_have,
                noise_want
            );
        });
    });
}
