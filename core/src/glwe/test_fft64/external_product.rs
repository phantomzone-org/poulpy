use backend::{FFT64, FillUniform, Module, ScalarZnx, ScalarZnxAlloc, ScratchOwned, Stats, VecZnxOps, ZnxViewMut};
use sampling::source::Source;

use crate::{
    FourierGLWESecret, GGSWCiphertext, GLWECiphertext, GLWEPlaintext, GLWESecret, Infos, noise::noise_ggsw_product,
};

#[test]
fn apply() {
    let log_n: usize = 8;
    let basek: usize = 12;
    let k_in: usize = 45;
    let digits: usize = k_in.div_ceil(basek);
    (1..4).for_each(|rank| {
        (1..digits + 1).for_each(|di| {
            let k_ggsw: usize = k_in + basek * di;
            let k_out: usize = k_ggsw; // Better capture noise
            println!("test external_product digits: {} rank: {}", di, rank);
            test_external_product(log_n, basek, k_out, k_in, k_ggsw, di, rank, 3.2);
        });
    });
}

#[test]
fn apply_inplace() {
    let log_n: usize = 8;
    let basek: usize = 12;
    let k_ct: usize = 60;
    let digits: usize = k_ct.div_ceil(basek);
    (1..4).for_each(|rank| {
        (1..digits + 1).for_each(|di| {
            let k_ggsw: usize = k_ct + basek * di;
            println!("test external_product digits: {} rank: {}", di, rank);
            test_external_product_inplace(log_n, basek, k_ct, k_ggsw, di, rank, 3.2);
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
    rank: usize,
    sigma: f64,
) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);

    let rows: usize = k_in.div_ceil(basek * digits);

    let mut ct_ggsw: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::alloc(&module, basek, k_ggsw, rows, digits, rank);
    let mut ct_glwe_in: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, k_in, rank);
    let mut ct_glwe_out: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, k_out, rank);
    let mut pt_rgsw: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_in);
    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_out);

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
                digits,
                rank,
            ),
    );

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(&module, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let sk_dft: FourierGLWESecret<Vec<u8>, FFT64> = FourierGLWESecret::from(&module, &sk);

    ct_ggsw.encrypt_sk(
        &module,
        &pt_rgsw,
        &sk_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_glwe_in.encrypt_sk(
        &module,
        &pt_want,
        &sk_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_glwe_out.external_product(&module, &ct_glwe_in, &ct_ggsw, scratch.borrow());

    ct_glwe_out.decrypt(&module, &mut pt_have, &sk_dft, scratch.borrow());

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
        basek * digits,
        0.5,
        var_msg,
        var_a0_err,
        var_a1_err,
        var_gct_err_lhs,
        var_gct_err_rhs,
        rank as f64,
        k_in,
        k_ggsw,
    );

    assert!(
        (noise_have - noise_want).abs() <= 0.5,
        "{} {}",
        noise_have,
        noise_want
    );
}

fn test_external_product_inplace(log_n: usize, basek: usize, k_ct: usize, k_ggsw: usize, digits: usize, rank: usize, sigma: f64) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let rows: usize = k_ct.div_ceil(basek * digits);

    let mut ct_ggsw: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::alloc(&module, basek, k_ggsw, rows, digits, rank);
    let mut ct_glwe: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, k_ct, rank);
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

    let k: usize = 1;

    pt_rgsw.raw_mut()[k] = 1; // X^{k}

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GGSWCiphertext::encrypt_sk_scratch_space(&module, basek, ct_ggsw.k(), rank)
            | GLWECiphertext::decrypt_scratch_space(&module, basek, ct_glwe.k())
            | GLWECiphertext::encrypt_sk_scratch_space(&module, basek, ct_glwe.k())
            | GLWECiphertext::external_product_inplace_scratch_space(&module, basek, ct_glwe.k(), ct_ggsw.k(), digits, rank),
    );

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(&module, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let sk_dft: FourierGLWESecret<Vec<u8>, FFT64> = FourierGLWESecret::from(&module, &sk);

    ct_ggsw.encrypt_sk(
        &module,
        &pt_rgsw,
        &sk_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_glwe.encrypt_sk(
        &module,
        &pt_want,
        &sk_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_glwe.external_product_inplace(&module, &ct_ggsw, scratch.borrow());

    ct_glwe.decrypt(&module, &mut pt_have, &sk_dft, scratch.borrow());

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
        (noise_have - noise_want).abs() <= 0.5,
        "{} {}",
        noise_have,
        noise_want
    );
}
