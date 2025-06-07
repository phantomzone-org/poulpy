use crate::{
    GGSWCiphertext, GLWECiphertext, GLWECiphertextFourier, GLWEOps, GLWEPlaintext, GLWESecret, GLWESwitchingKey, Infos, div_ceil,
    test_fft64::{log2_std_noise_gglwe_product, noise_ggsw_product},
};
use backend::{FFT64, FillUniform, Module, ScalarZnx, ScalarZnxAlloc, ScratchOwned, Stats, VecZnxOps, ZnxViewMut};
use sampling::source::Source;

#[test]
fn keyswitch() {
    let basek: usize = 12;
    let ct_k_in: usize = 45;
    let digits: usize = div_ceil(ct_k_in, basek);
    (1..4).for_each(|in_rank| {
        (1..4).for_each(|out_rank| {
            (1..digits + 1).for_each(|di| {
                let k_ksk: usize = ct_k_in + basek * di;
                println!(
                    "test keyswitch digits: {} in_rank: {} out_rank: {}",
                    di, in_rank, out_rank
                );
                test_keyswitch(12, basek, di, k_ksk, ct_k_in, k_ksk, in_rank, out_rank, 3.2);
            })
        });
    });
}

#[test]
fn keyswitch_inplace() {
    let basek: usize = 12;
    let ct_k_in: usize = 45;
    let digits: usize = div_ceil(ct_k_in, basek);
    (1..4).for_each(|rank| {
        (1..digits + 1).for_each(|di| {
            let k_ksk: usize = ct_k_in + basek * di;
            println!("test keyswitch_inplace digits: {} rank: {}", di, rank);
            test_keyswitch_inplace(12, basek, di, k_ksk, ct_k_in, rank, 3.2);
        });
    });
}

#[test]
fn external_product() {
    let basek: usize = 12;
    let ct_k_in: usize = 45;
    let digits: usize = div_ceil(ct_k_in, basek);
    (1..4).for_each(|rank| {
        (1..digits + 1).for_each(|di| {
            let k_ggsw: usize = ct_k_in + basek * di;
            println!("test external_product digits: {} rank: {}", di, rank);
            test_external_product(12, basek, di, k_ggsw, ct_k_in, k_ggsw, rank, 3.2);
        });
    });
}

#[test]
fn external_product_inplace() {
    let basek: usize = 12;
    let ct_k_in: usize = 60;
    let digits: usize = div_ceil(ct_k_in, basek);
    (1..4).for_each(|rank| {
        (1..digits + 1).for_each(|di| {
            let k_ggsw: usize = ct_k_in + basek * di;
            println!("test external_product digits: {} rank: {}", di, rank);
            test_external_product_inplace(12, basek, di, k_ggsw, ct_k_in, rank, 3.2);
        });
    });
}

fn test_keyswitch(
    log_n: usize,
    basek: usize,
    digits: usize,
    k_ksk: usize,
    k_ct_in: usize,
    k_ct_out: usize,
    rank_in: usize,
    rank_out: usize,
    sigma: f64,
) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);

    let rows: usize = (k_ct_in + basek - 1) / basek;

    let mut ksk: GLWESwitchingKey<Vec<u8>, FFT64> =
        GLWESwitchingKey::alloc(&module, basek, k_ksk, rows, digits, rank_in, rank_out);
    let mut ct_glwe_in: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, k_ct_in, rank_in);
    let mut ct_glwe_dft_in: GLWECiphertextFourier<Vec<u8>, FFT64> =
        GLWECiphertextFourier::alloc(&module, basek, k_ct_in, rank_in);
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
        GLWESwitchingKey::encrypt_sk_scratch_space(&module, basek, k_ksk, rank_out)
            | GLWECiphertext::decrypt_scratch_space(&module, basek, k_ct_out)
            | GLWECiphertext::encrypt_sk_scratch_space(&module, basek, k_ct_in)
            | GLWECiphertextFourier::keyswitch_scratch_space(
                &module,
                basek,
                ct_glwe_out.k(),
                rank_out,
                ct_glwe_in.k(),
                rank_in,
                ksk.k(),
            ),
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

    ct_glwe_in.encrypt_sk(
        &module,
        &pt_want,
        &sk_in,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_glwe_in.dft(&module, &mut ct_glwe_dft_in);
    ct_glwe_dft_out.keyswitch(&module, &ct_glwe_dft_in, &ksk, scratch.borrow());
    ct_glwe_dft_out.idft(&module, &mut ct_glwe_out, scratch.borrow());

    ct_glwe_out.decrypt(&module, &mut pt_have, &sk_out, scratch.borrow());

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

fn test_keyswitch_inplace(log_n: usize, basek: usize, digits: usize, k_ksk: usize, k_ct: usize, rank: usize, sigma: f64) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let rows: usize = (k_ct + basek - 1) / basek;

    let mut ksk: GLWESwitchingKey<Vec<u8>, FFT64> = GLWESwitchingKey::alloc(&module, basek, k_ksk, rows, digits, rank, rank);
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
        GLWESwitchingKey::encrypt_sk_scratch_space(&module, basek, ksk.k(), rank)
            | GLWECiphertext::decrypt_scratch_space(&module, basek, ct_glwe.k())
            | GLWECiphertext::encrypt_sk_scratch_space(&module, basek, ct_glwe.k())
            | GLWECiphertextFourier::keyswitch_inplace_scratch_space(&module, basek, ct_rlwe_dft.k(), ksk.k(), rank),
    );

    let mut sk_in: GLWESecret<Vec<u8>, FFT64> = GLWESecret::alloc(&module, rank);
    sk_in.fill_ternary_prob(&module, 0.5, &mut source_xs);

    let mut sk_out: GLWESecret<Vec<u8>, FFT64> = GLWESecret::alloc(&module, rank);
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

    ct_glwe.encrypt_sk(
        &module,
        &pt_want,
        &sk_in,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_glwe.dft(&module, &mut ct_rlwe_dft);
    ct_rlwe_dft.keyswitch_inplace(&module, &ksk, scratch.borrow());
    ct_rlwe_dft.idft(&module, &mut ct_glwe, scratch.borrow());

    ct_glwe.decrypt(&module, &mut pt_have, &sk_out, scratch.borrow());

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
        (noise_have - noise_want).abs() <= 0.1,
        "{} {}",
        noise_have,
        noise_want
    );
}

fn test_external_product(
    log_n: usize,
    basek: usize,
    digits: usize,
    k_ggsw: usize,
    k_ct_in: usize,
    k_ct_out: usize,
    rank: usize,
    sigma: f64,
) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);

    let rows: usize = (k_ct_in + basek - 1) / basek;

    let mut ct_ggsw: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::alloc(&module, basek, k_ggsw, rows, digits, rank);
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

    pt_want.data.at_mut(0, 0)[1] = 1;

    let k: i64 = 1;

    pt_rgsw.raw_mut()[0] = 1; // X^{0}
    module.vec_znx_rotate_inplace(k, &mut pt_rgsw, 0); // X^{k}

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GGSWCiphertext::encrypt_sk_scratch_space(&module, basek, ct_ggsw.k(), rank)
            | GLWECiphertext::decrypt_scratch_space(&module, basek, ct_out.k())
            | GLWECiphertext::encrypt_sk_scratch_space(&module, basek, ct_in.k())
            | GLWECiphertextFourier::external_product_scratch_space(&module, basek, ct_out.k(), ct_in.k(), ct_ggsw.k(), rank),
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

    ct_in.encrypt_sk(
        &module,
        &pt_want,
        &sk,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_in.dft(&module, &mut ct_in_dft);
    ct_out_dft.external_product(&module, &ct_in_dft, &ct_ggsw, scratch.borrow());
    ct_out_dft.idft(&module, &mut ct_out, scratch.borrow());

    ct_out.decrypt(&module, &mut pt_have, &sk, scratch.borrow());

    pt_want.rotate_inplace(&module, k);
    pt_have.sub_inplace_ab(&module, &pt_want);

    let noise_have: f64 = pt_have.data.std(0, basek).log2();

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

fn test_external_product_inplace(log_n: usize, basek: usize, digits: usize, k_ggsw: usize, k_ct: usize, rank: usize, sigma: f64) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let rows: usize = (k_ct + basek - 1) / basek;

    let mut ct_ggsw: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::alloc(&module, basek, k_ggsw, rows, digits, rank);
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

    pt_want.data.at_mut(0, 0)[1] = 1;

    let k: i64 = 1;

    pt_rgsw.raw_mut()[0] = 1; // X^{0}
    module.vec_znx_rotate_inplace(k, &mut pt_rgsw, 0); // X^{k}

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GGSWCiphertext::encrypt_sk_scratch_space(&module, basek, ct_ggsw.k(), rank)
            | GLWECiphertext::decrypt_scratch_space(&module, basek, ct.k())
            | GLWECiphertext::encrypt_sk_scratch_space(&module, basek, ct.k())
            | GLWECiphertextFourier::external_product_inplace_scratch_space(&module, basek, ct.k(), ct_ggsw.k(), rank),
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

    ct.encrypt_sk(
        &module,
        &pt_want,
        &sk,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct.dft(&module, &mut ct_rlwe_dft);
    ct_rlwe_dft.external_product_inplace(&module, &ct_ggsw, scratch.borrow());
    ct_rlwe_dft.idft(&module, &mut ct, scratch.borrow());

    ct.decrypt(&module, &mut pt_have, &sk, scratch.borrow());

    pt_want.rotate_inplace(&module, k);
    pt_have.sub_inplace_ab(&module, &pt_want);

    let noise_have: f64 = pt_have.data.std(0, basek).log2();

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
