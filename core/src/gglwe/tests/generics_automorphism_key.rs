use backend::{
    hal::{
        api::{ModuleNew, ScalarZnxAutomorphism, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxStd, VecZnxSubScalarInplace},
        layouts::{Module, ScratchOwned},
    },
    implementation::cpu_spqlios::FFT64,
};
use sampling::source::Source;

use crate::{
    AutomorphismKey, AutomorphismKeyExec, GLWEPlaintext, GLWESecret, GLWESecretExec, Infos, noise::log2_std_noise_gglwe_product,
};

pub(crate) fn test_gglwe_automorphism(
    p0: i64,
    p1: i64,
    log_n: usize,
    basek: usize,
    digits: usize,
    k_in: usize,
    k_out: usize,
    k_apply: usize,
    sigma: f64,
    rank: usize,
) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);

    let digits_in: usize = 1;

    let rows_in: usize = k_in / (basek * digits);
    let rows_apply: usize = k_in.div_ceil(basek * digits);

    let mut auto_key_in: AutomorphismKey<Vec<u8>> = AutomorphismKey::alloc(&module, basek, k_in, rows_in, digits_in, rank);
    let mut auto_key_out: AutomorphismKey<Vec<u8>> = AutomorphismKey::alloc(&module, basek, k_out, rows_in, digits_in, rank);
    let mut auto_key_apply: AutomorphismKey<Vec<u8>> = AutomorphismKey::alloc(&module, basek, k_apply, rows_apply, digits, rank);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned<FFT64> = ScratchOwned::alloc(
        AutomorphismKey::encrypt_sk_scratch_space(&module, basek, k_apply, rank)
            | AutomorphismKey::automorphism_scratch_space(&module, basek, k_out, k_in, k_apply, digits, rank),
    );

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(&module, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);

    // gglwe_{s1}(s0) = s0 -> s1
    auto_key_in.encrypt_sk(
        &module,
        p0,
        &sk,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    // gglwe_{s2}(s1) -> s1 -> s2
    auto_key_apply.encrypt_sk(
        &module,
        p1,
        &sk,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let mut auto_key_apply_exec: AutomorphismKeyExec<Vec<u8>, FFT64> =
        AutomorphismKeyExec::alloc(&module, basek, k_apply, rows_apply, digits, rank);

    auto_key_apply_exec.prepare(&module, &auto_key_apply, scratch.borrow());

    // gglwe_{s1}(s0) (x) gglwe_{s2}(s1) = gglwe_{s2}(s0)
    auto_key_out.automorphism(
        &module,
        &auto_key_in,
        &auto_key_apply_exec,
        scratch.borrow(),
    );

    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_out);

    let mut sk_auto: GLWESecret<Vec<u8>> = GLWESecret::alloc(&module, rank);
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

    let sk_auto_dft: GLWESecretExec<Vec<u8>, FFT64> = GLWESecretExec::from(&module, &sk_auto);

    (0..auto_key_out.rank_in()).for_each(|col_i| {
        (0..auto_key_out.rows()).for_each(|row_i| {
            auto_key_out
                .at(row_i, col_i)
                .decrypt(&module, &mut pt, &sk_auto_dft, scratch.borrow());

            module.vec_znx_sub_scalar_inplace(
                &mut pt.data,
                0,
                (digits_in - 1) + row_i * digits_in,
                &sk.data,
                col_i,
            );

            let noise_have: f64 = module.vec_znx_std(basek, &pt.data, 0).log2();
            let noise_want: f64 = log2_std_noise_gglwe_product(
                module.n() as f64,
                basek * digits,
                0.5,
                0.5,
                0f64,
                sigma * sigma,
                0f64,
                rank as f64,
                k_out,
                k_apply,
            );

            assert!(
                noise_have < noise_want + 0.5,
                "{} {}",
                noise_have,
                noise_want
            );
        });
    });
}

pub(crate) fn test_gglwe_automorphism_inplace(
    p0: i64,
    p1: i64,
    log_n: usize,
    basek: usize,
    digits: usize,
    k_in: usize,
    k_apply: usize,
    sigma: f64,
    rank: usize,
) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);

    let digits_in: usize = 1;

    let rows_in: usize = k_in / (basek * digits);
    let rows_apply: usize = k_in.div_ceil(basek * digits);

    let mut auto_key: AutomorphismKey<Vec<u8>> = AutomorphismKey::alloc(&module, basek, k_in, rows_in, digits_in, rank);
    let mut auto_key_apply: AutomorphismKey<Vec<u8>> = AutomorphismKey::alloc(&module, basek, k_apply, rows_apply, digits, rank);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned<FFT64> = ScratchOwned::alloc(
        AutomorphismKey::encrypt_sk_scratch_space(&module, basek, k_apply, rank)
            | AutomorphismKey::automorphism_inplace_scratch_space(&module, basek, k_in, k_apply, digits, rank),
    );

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(&module, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);

    // gglwe_{s1}(s0) = s0 -> s1
    auto_key.encrypt_sk(
        &module,
        p0,
        &sk,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    // gglwe_{s2}(s1) -> s1 -> s2
    auto_key_apply.encrypt_sk(
        &module,
        p1,
        &sk,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let mut auto_key_apply_exec: AutomorphismKeyExec<Vec<u8>, FFT64> =
        AutomorphismKeyExec::alloc(&module, basek, k_apply, rows_apply, digits, rank);

    auto_key_apply_exec.prepare(&module, &auto_key_apply, scratch.borrow());

    // gglwe_{s1}(s0) (x) gglwe_{s2}(s1) = gglwe_{s2}(s0)
    auto_key.automorphism_inplace(&module, &auto_key_apply_exec, scratch.borrow());

    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_in);

    let mut sk_auto: GLWESecret<Vec<u8>> = GLWESecret::alloc(&module, rank);
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

    let sk_auto_dft: GLWESecretExec<Vec<u8>, FFT64> = GLWESecretExec::from(&module, &sk_auto);

    (0..auto_key.rank_in()).for_each(|col_i| {
        (0..auto_key.rows()).for_each(|row_i| {
            auto_key
                .at(row_i, col_i)
                .decrypt(&module, &mut pt, &sk_auto_dft, scratch.borrow());
            module.vec_znx_sub_scalar_inplace(
                &mut pt.data,
                0,
                (digits_in - 1) + row_i * digits_in,
                &sk.data,
                col_i,
            );

            let noise_have: f64 = module.vec_znx_std(basek, &pt.data, 0).log2();
            let noise_want: f64 = log2_std_noise_gglwe_product(
                module.n() as f64,
                basek * digits,
                0.5,
                0.5,
                0f64,
                sigma * sigma,
                0f64,
                rank as f64,
                k_in,
                k_apply,
            );

            assert!(
                noise_have < noise_want + 0.5,
                "{} {}",
                noise_have,
                noise_want
            );
        });
    });
}
