use backend::hal::{
    api::{
        MatZnxAlloc, ScalarZnxAlloc, ScalarZnxAllocBytes, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAddScalarInplace,
        VecZnxAlloc, VecZnxAllocBytes, VecZnxAutomorphism, VecZnxAutomorphismInplace, VecZnxCopy, VecZnxStd,
        VecZnxSubScalarInplace, VecZnxSwithcDegree,
    },
    layouts::{Backend, Module, ScratchOwned},
    oep::{
        ScratchAvailableImpl, ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeScalarZnxImpl, TakeSvpPPolImpl,
        TakeVecZnxBigImpl, TakeVecZnxDftImpl, TakeVecZnxImpl,
    },
};
use sampling::source::Source;

use crate::{
    AutomorphismKey, AutomorphismKeyCompressed, AutomorphismKeyEncryptSkFamily, AutomorphismKeyExec, GGLWEExecLayoutFamily,
    GLWEDecryptFamily, GLWEKeyswitchFamily, GLWEPlaintext, GLWESecret, GLWESecretExec, Infos,
    noise::log2_std_noise_gglwe_product,
};

pub(crate) trait AutomorphismTestModuleFamily<B: Backend> = MatZnxAlloc
    + AutomorphismKeyEncryptSkFamily<B>
    + ScalarZnxAllocBytes
    + VecZnxAllocBytes
    + GLWEKeyswitchFamily<B>
    + ScalarZnxAlloc
    + VecZnxAutomorphism
    + GGLWEExecLayoutFamily<B>
    + VecZnxSwithcDegree
    + VecZnxAddScalarInplace
    + VecZnxAutomorphism
    + VecZnxAutomorphismInplace
    + VecZnxAlloc
    + GLWEDecryptFamily<B>
    + VecZnxSubScalarInplace
    + VecZnxStd
    + VecZnxCopy;
pub(crate) trait AutomorphismTestScratchFamily<B: Backend> = ScratchOwnedAllocImpl<B>
    + ScratchOwnedBorrowImpl<B>
    + ScratchAvailableImpl<B>
    + TakeScalarZnxImpl<B>
    + TakeVecZnxDftImpl<B>
    + TakeVecZnxImpl<B>
    + TakeSvpPPolImpl<B>
    + TakeVecZnxBigImpl<B>;

pub(crate) fn test_automorphisk_key_encrypt_sk<B: Backend>(
    module: &Module<B>,
    basek: usize,
    k_ksk: usize,
    digits: usize,
    rank: usize,
    sigma: f64,
) where
    Module<B>: AutomorphismTestModuleFamily<B>,
    B: AutomorphismTestScratchFamily<B>,
{
    let rows: usize = (k_ksk - digits * basek) / (digits * basek);

    let mut atk: AutomorphismKey<Vec<u8>> = AutomorphismKey::alloc(module, basek, k_ksk, rows, digits, rank);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(AutomorphismKey::encrypt_sk_scratch_space(
        module, basek, k_ksk, rank,
    ));

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(module, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);

    let p = -5;

    atk.encrypt_sk(
        module,
        p,
        &sk,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let mut sk_out: GLWESecret<Vec<u8>> = sk.clone();
    (0..atk.rank()).for_each(|i| {
        module.vec_znx_automorphism(
            module.galois_element_inv(p),
            &mut sk_out.data.as_vec_znx_mut(),
            i,
            &sk.data.as_vec_znx(),
            i,
        );
    });
    let sk_out_exec = GLWESecretExec::from(module, &sk_out);

    atk.key
        .key
        .assert_noise(module, &sk_out_exec, &sk.data, sigma);
}

pub(crate) fn test_automorphisk_key_encrypt_sk_compressed<B: Backend>(
    module: &Module<B>,
    basek: usize,
    k_ksk: usize,
    digits: usize,
    rank: usize,
    sigma: f64,
) where
    Module<B>: AutomorphismTestModuleFamily<B>,
    B: AutomorphismTestScratchFamily<B>,
{
    let rows: usize = (k_ksk - digits * basek) / (digits * basek);

    let mut atk_compressed: AutomorphismKeyCompressed<Vec<u8>> =
        AutomorphismKeyCompressed::alloc(module, basek, k_ksk, rows, digits, rank);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(AutomorphismKey::encrypt_sk_scratch_space(
        module, basek, k_ksk, rank,
    ));

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(module, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);

    let p = -5;

    let seed_xa: [u8; 32] = [1u8; 32];

    atk_compressed.encrypt_sk(
        module,
        p,
        &sk,
        seed_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let mut sk_out: GLWESecret<Vec<u8>> = sk.clone();
    (0..atk_compressed.rank()).for_each(|i| {
        module.vec_znx_automorphism(
            module.galois_element_inv(p),
            &mut sk_out.data.as_vec_znx_mut(),
            i,
            &sk.data.as_vec_znx(),
            i,
        );
    });
    let sk_out_exec = GLWESecretExec::from(module, &sk_out);

    let mut atk: AutomorphismKey<Vec<u8>> = AutomorphismKey::alloc(module, basek, k_ksk, rows, digits, rank);
    atk.decompress(module, &atk_compressed);

    atk.key
        .key
        .assert_noise(module, &sk_out_exec, &sk.data, sigma);
}

pub(crate) fn test_gglwe_automorphism<B: Backend>(
    module: &Module<B>,
    p0: i64,
    p1: i64,
    basek: usize,
    digits: usize,
    k_in: usize,
    k_out: usize,
    k_apply: usize,
    sigma: f64,
    rank: usize,
) where
    Module<B>: AutomorphismTestModuleFamily<B>,
    B: AutomorphismTestScratchFamily<B>,
{
    let digits_in: usize = 1;

    let rows_in: usize = k_in / (basek * digits);
    let rows_apply: usize = k_in.div_ceil(basek * digits);

    let mut auto_key_in: AutomorphismKey<Vec<u8>> = AutomorphismKey::alloc(&module, basek, k_in, rows_in, digits_in, rank);
    let mut auto_key_out: AutomorphismKey<Vec<u8>> = AutomorphismKey::alloc(&module, basek, k_out, rows_in, digits_in, rank);
    let mut auto_key_apply: AutomorphismKey<Vec<u8>> = AutomorphismKey::alloc(&module, basek, k_apply, rows_apply, digits, rank);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
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

    let mut auto_key_apply_exec: AutomorphismKeyExec<Vec<u8>, B> =
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
        module.vec_znx_automorphism(
            module.galois_element_inv(p0 * p1),
            &mut sk_auto.data.as_vec_znx_mut(),
            i,
            &sk.data.as_vec_znx(),
            i,
        );
    });

    let sk_auto_dft: GLWESecretExec<Vec<u8>, B> = GLWESecretExec::from(&module, &sk_auto);

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

pub(crate) fn test_gglwe_automorphism_inplace<B: Backend>(
    module: &Module<B>,
    p0: i64,
    p1: i64,
    basek: usize,
    digits: usize,
    k_in: usize,
    k_apply: usize,
    sigma: f64,
    rank: usize,
) where
    Module<B>: AutomorphismTestModuleFamily<B>,
    B: AutomorphismTestScratchFamily<B>,
{
    let digits_in: usize = 1;

    let rows_in: usize = k_in / (basek * digits);
    let rows_apply: usize = k_in.div_ceil(basek * digits);

    let mut auto_key: AutomorphismKey<Vec<u8>> = AutomorphismKey::alloc(&module, basek, k_in, rows_in, digits_in, rank);
    let mut auto_key_apply: AutomorphismKey<Vec<u8>> = AutomorphismKey::alloc(&module, basek, k_apply, rows_apply, digits, rank);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
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

    let mut auto_key_apply_exec: AutomorphismKeyExec<Vec<u8>, B> =
        AutomorphismKeyExec::alloc(&module, basek, k_apply, rows_apply, digits, rank);

    auto_key_apply_exec.prepare(&module, &auto_key_apply, scratch.borrow());

    // gglwe_{s1}(s0) (x) gglwe_{s2}(s1) = gglwe_{s2}(s0)
    auto_key.automorphism_inplace(&module, &auto_key_apply_exec, scratch.borrow());

    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_in);

    let mut sk_auto: GLWESecret<Vec<u8>> = GLWESecret::alloc(&module, rank);
    sk_auto.fill_zero(); // Necessary to avoid panic of unfilled sk

    (0..rank).for_each(|i| {
        module.vec_znx_automorphism(
            module.galois_element_inv(p0 * p1),
            &mut sk_auto.data.as_vec_znx_mut(),
            i,
            &sk.data.as_vec_znx(),
            i,
        );
    });

    let sk_auto_dft: GLWESecretExec<Vec<u8>, B> = GLWESecretExec::from(&module, &sk_auto);

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
