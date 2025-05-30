use backend::{FFT64, Module, ScalarZnxOps, ScratchOwned, Stats, VecZnxOps};
use sampling::source::Source;

use crate::{
    AutomorphismKey, GLWECiphertextFourier, GLWEPlaintext, GLWESecret, GetRow, Infos,
    test_fft64::gglwe::log2_std_noise_gglwe_product,
};

#[test]
fn automorphism() {
    (1..4).for_each(|rank| {
        println!("test automorphism rank: {}", rank);
        test_automorphism(-1, 5, 12, 12, 60, 3.2, rank);
    });
}

#[test]
fn automorphism_inplace() {
    (1..4).for_each(|rank| {
        println!("test automorphism_inplace rank: {}", rank);
        test_automorphism_inplace(-1, 5, 12, 12, 60, 3.2, rank);
    });
}

fn test_automorphism(p0: i64, p1: i64, log_n: usize, basek: usize, k_ksk: usize, sigma: f64, rank: usize) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let rows = (k_ksk + basek - 1) / basek;

    let mut auto_key_in: AutomorphismKey<Vec<u8>, FFT64> = AutomorphismKey::alloc(&module, basek, k_ksk, rows, rank);
    let mut auto_key_out: AutomorphismKey<Vec<u8>, FFT64> = AutomorphismKey::alloc(&module, basek, k_ksk, rows, rank);
    let mut auto_key_apply: AutomorphismKey<Vec<u8>, FFT64> = AutomorphismKey::alloc(&module, basek, k_ksk, rows, rank);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        AutomorphismKey::generate_from_sk_scratch_space(&module, basek, k_ksk, rank)
            | GLWECiphertextFourier::decrypt_scratch_space(&module, basek, k_ksk)
            | AutomorphismKey::automorphism_scratch_space(&module, basek, k_ksk, k_ksk, k_ksk, rank),
    );

    let mut sk: GLWESecret<Vec<u8>, FFT64> = GLWESecret::alloc(&module, rank);
    sk.fill_ternary_prob(&module, 0.5, &mut source_xs);

    // gglwe_{s1}(s0) = s0 -> s1
    auto_key_in.generate_from_sk(
        &module,
        p0,
        &sk,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    // gglwe_{s2}(s1) -> s1 -> s2
    auto_key_apply.generate_from_sk(
        &module,
        p1,
        &sk,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    // gglwe_{s1}(s0) (x) gglwe_{s2}(s1) = gglwe_{s2}(s0)
    auto_key_out.automorphism(&module, &auto_key_in, &auto_key_apply, scratch.borrow());

    let mut ct_glwe_dft: GLWECiphertextFourier<Vec<u8>, FFT64> = GLWECiphertextFourier::alloc(&module, basek, k_ksk, rank);
    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_ksk);

    let mut sk_auto: GLWESecret<Vec<u8>, FFT64> = GLWESecret::alloc(&module, rank);
    sk_auto.fill_zero(); // Necessary to avoid panic of unfilled sk
    (0..rank).for_each(|i| {
        module.scalar_znx_automorphism(
            module.galois_element_inv(p0 * p1),
            &mut sk_auto.data,
            i,
            &sk.data,
            i,
        );
    });

    sk_auto.prep_fourier(&module);

    (0..auto_key_out.rank_in()).for_each(|col_i| {
        (0..auto_key_out.rows()).for_each(|row_i| {
            auto_key_out.get_row(&module, row_i, col_i, &mut ct_glwe_dft);

            ct_glwe_dft.decrypt(&module, &mut pt, &sk_auto, scratch.borrow());
            module.vec_znx_sub_scalar_inplace(&mut pt.data, 0, row_i, &sk.data, col_i);

            let noise_have: f64 = pt.data.std(0, basek).log2();
            let noise_want: f64 = log2_std_noise_gglwe_product(
                module.n() as f64,
                basek,
                0.5,
                0.5,
                0f64,
                sigma * sigma,
                0f64,
                rank as f64,
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

fn test_automorphism_inplace(p0: i64, p1: i64, log_n: usize, basek: usize, k_ksk: usize, sigma: f64, rank: usize) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let rows: usize = (k_ksk + basek - 1) / basek;

    let mut auto_key: AutomorphismKey<Vec<u8>, FFT64> = AutomorphismKey::alloc(&module, basek, k_ksk, rows, rank);
    let mut auto_key_apply: AutomorphismKey<Vec<u8>, FFT64> = AutomorphismKey::alloc(&module, basek, k_ksk, rows, rank);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        AutomorphismKey::generate_from_sk_scratch_space(&module, basek, k_ksk, rank)
            | GLWECiphertextFourier::decrypt_scratch_space(&module, basek, k_ksk)
            | AutomorphismKey::automorphism_inplace_scratch_space(&module, basek, k_ksk, k_ksk, rank),
    );

    let mut sk: GLWESecret<Vec<u8>, FFT64> = GLWESecret::alloc(&module, rank);
    sk.fill_ternary_prob(&module, 0.5, &mut source_xs);

    // gglwe_{s1}(s0) = s0 -> s1
    auto_key.generate_from_sk(
        &module,
        p0,
        &sk,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    // gglwe_{s2}(s1) -> s1 -> s2
    auto_key_apply.generate_from_sk(
        &module,
        p1,
        &sk,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    // gglwe_{s1}(s0) (x) gglwe_{s2}(s1) = gglwe_{s2}(s0)
    auto_key.automorphism_inplace(&module, &auto_key_apply, scratch.borrow());

    let mut ct_glwe_dft: GLWECiphertextFourier<Vec<u8>, FFT64> = GLWECiphertextFourier::alloc(&module, basek, k_ksk, rank);
    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_ksk);

    let mut sk_auto: GLWESecret<Vec<u8>, FFT64> = GLWESecret::alloc(&module, rank);
    sk_auto.fill_zero(); // Necessary to avoid panic of unfilled sk
    (0..rank).for_each(|i| {
        module.scalar_znx_automorphism(
            module.galois_element_inv(p0 * p1),
            &mut sk_auto.data,
            i,
            &sk.data,
            i,
        );
    });

    sk_auto.prep_fourier(&module);

    (0..auto_key.rank_in()).for_each(|col_i| {
        (0..auto_key.rows()).for_each(|row_i| {
            auto_key.get_row(&module, row_i, col_i, &mut ct_glwe_dft);

            ct_glwe_dft.decrypt(&module, &mut pt, &sk_auto, scratch.borrow());
            module.vec_znx_sub_scalar_inplace(&mut pt.data, 0, row_i, &sk.data, col_i);

            let noise_have: f64 = pt.data.std(0, basek).log2();
            let noise_want: f64 = log2_std_noise_gglwe_product(
                module.n() as f64,
                basek,
                0.5,
                0.5,
                0f64,
                sigma * sigma,
                0f64,
                rank as f64,
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
