use poulpy_hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyDftToDftInplace, SvpPPolAlloc, SvpPPolAllocBytes, SvpPrepare,
        VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxAutomorphism, VecZnxAutomorphismInplace,
        VecZnxBigAddInplace, VecZnxBigAddSmallInplace, VecZnxBigAllocBytes, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxCopy, VecZnxDftAllocBytes, VecZnxDftApply, VecZnxFillUniform, VecZnxIdftApplyConsume, VecZnxNormalize,
        VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxSub, VecZnxSubABInplace, VecZnxSubScalarInplace, VecZnxSwitchRing,
        VmpApplyDftToDft, VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes, VmpPMatAlloc, VmpPrepare,
    },
    layouts::{Backend, Module, ScratchOwned},
    oep::{
        ScratchAvailableImpl, ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeScalarZnxImpl, TakeSvpPPolImpl,
        TakeVecZnxBigImpl, TakeVecZnxDftImpl, TakeVecZnxImpl,
    },
    source::Source,
};

use crate::{
    encryption::SIGMA,
    layouts::{
        GGLWEAutomorphismKey, GLWEPlaintext, GLWESecret, Infos,
        prepared::{GGLWEAutomorphismKeyPrepared, GLWESecretPrepared, Prepare, PrepareAlloc},
    },
    noise::log2_std_noise_gglwe_product,
};

#[allow(clippy::too_many_arguments)]
pub fn test_gglwe_automorphism_key_automorphism<B>(module: &Module<B>)
where
    Module<B>: VecZnxDftAllocBytes
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VmpApplyDftToDft<B>
        + VmpApplyDftToDftAdd<B>
        + VecZnxDftApply<B>
        + VecZnxIdftApplyConsume<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxBigNormalize<B>
        + VecZnxAutomorphism
        + VecZnxAutomorphismInplace<B>
        + SvpPPolAllocBytes
        + VecZnxDftAllocBytes
        + VecZnxNormalizeTmpBytes
        + VmpPMatAlloc<B>
        + VmpPrepare<B>
        + SvpPrepare<B>
        + SvpApplyDftToDftInplace<B>
        + VecZnxAddScalarInplace
        + VecZnxFillUniform
        + VecZnxSubABInplace
        + VecZnxAddInplace
        + VecZnxNormalizeInplace<B>
        + VecZnxAddNormal
        + VecZnxNormalize<B>
        + VecZnxSub
        + VecZnxSwitchRing<B>
        + SvpPPolAlloc<B>
        + VecZnxBigAddInplace<B>
        + VecZnxSubScalarInplace,
    B: Backend
        + ScratchOwnedAllocImpl<B>
        + ScratchOwnedBorrowImpl<B>
        + ScratchAvailableImpl<B>
        + TakeScalarZnxImpl<B>
        + TakeVecZnxDftImpl<B>
        + TakeVecZnxImpl<B>
        + TakeSvpPPolImpl<B>
        + TakeVecZnxBigImpl<B>,
{
    let basek: usize = 12;
    let k_in: usize = 60;
    let k_out: usize = 40;
    let digits: usize = k_in.div_ceil(basek);
    let p0 = -1;
    let p1 = -5;
    (1..3).for_each(|rank| {
        (1..digits + 1).for_each(|di| {
            let k_apply: usize = (digits + di) * basek;

            let n: usize = module.n();
            let digits_in: usize = 1;

            let rows_in: usize = k_in / (basek * di);
            let rows_out: usize = k_out / (basek * di);
            let rows_apply: usize = k_in.div_ceil(basek * di);

            let mut auto_key_in: GGLWEAutomorphismKey<Vec<u8>> =
                GGLWEAutomorphismKey::alloc(n, basek, k_in, rows_in, digits_in, rank);
            let mut auto_key_out: GGLWEAutomorphismKey<Vec<u8>> =
                GGLWEAutomorphismKey::alloc(n, basek, k_out, rows_out, digits_in, rank);
            let mut auto_key_apply: GGLWEAutomorphismKey<Vec<u8>> =
                GGLWEAutomorphismKey::alloc(n, basek, k_apply, rows_apply, di, rank);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
                GGLWEAutomorphismKey::encrypt_sk_scratch_space(module, basek, k_apply, rank)
                    | GGLWEAutomorphismKey::automorphism_scratch_space(module, basek, k_out, k_in, k_apply, di, rank),
            );

            let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
            sk.fill_ternary_prob(0.5, &mut source_xs);

            // gglwe_{s1}(s0) = s0 -> s1
            auto_key_in.encrypt_sk(
                module,
                p0,
                &sk,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            // gglwe_{s2}(s1) -> s1 -> s2
            auto_key_apply.encrypt_sk(
                module,
                p1,
                &sk,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            let mut auto_key_apply_prepared: GGLWEAutomorphismKeyPrepared<Vec<u8>, B> =
                GGLWEAutomorphismKeyPrepared::alloc(module, basek, k_apply, rows_apply, di, rank);

            auto_key_apply_prepared.prepare(module, &auto_key_apply, scratch.borrow());

            // gglwe_{s1}(s0) (x) gglwe_{s2}(s1) = gglwe_{s2}(s0)
            auto_key_out.automorphism(
                module,
                &auto_key_in,
                &auto_key_apply_prepared,
                scratch.borrow(),
            );

            let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(n, basek, k_out);

            let mut sk_auto: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
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

            let sk_auto_dft: GLWESecretPrepared<Vec<u8>, B> = sk_auto.prepare_alloc(module, scratch.borrow());

            (0..auto_key_out.rank_in()).for_each(|col_i| {
                (0..auto_key_out.rows()).for_each(|row_i| {
                    auto_key_out
                        .at(row_i, col_i)
                        .decrypt(module, &mut pt, &sk_auto_dft, scratch.borrow());

                    module.vec_znx_sub_scalar_inplace(
                        &mut pt.data,
                        0,
                        (digits_in - 1) + row_i * digits_in,
                        &sk.data,
                        col_i,
                    );

                    let noise_have: f64 = pt.data.std(basek, 0).log2();
                    let noise_want: f64 = log2_std_noise_gglwe_product(
                        n as f64,
                        basek * di,
                        0.5,
                        0.5,
                        0f64,
                        SIGMA * SIGMA,
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
        });
    });
}

#[allow(clippy::too_many_arguments)]
pub fn test_gglwe_automorphism_key_automorphism_inplace<B>(module: &Module<B>)
where
    Module<B>: VecZnxDftAllocBytes
        + VecZnxBigNormalize<B>
        + VecZnxDftApply<B>
        + SvpApplyDftToDftInplace<B>
        + VecZnxIdftApplyConsume<B>
        + VecZnxNormalizeTmpBytes
        + VecZnxFillUniform
        + VecZnxSubABInplace
        + VecZnxAddInplace
        + VecZnxNormalizeInplace<B>
        + VecZnxAddNormal
        + VecZnxNormalize<B>
        + VecZnxSub
        + SvpPrepare<B>
        + SvpPPolAllocBytes
        + SvpPPolAlloc<B>
        + VecZnxDftAllocBytes
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VmpApplyDftToDftTmpBytes
        + VmpApplyDftToDft<B>
        + VmpApplyDftToDftAdd<B>
        + VecZnxDftApply<B>
        + VecZnxIdftApplyConsume<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxBigNormalize<B>
        + VecZnxAutomorphism
        + VecZnxSwitchRing<B>
        + VecZnxAddScalarInplace
        + VecZnxAutomorphism
        + VecZnxAutomorphismInplace<B>
        + VecZnxDftAllocBytes
        + VecZnxBigAllocBytes
        + VecZnxDftApply<B>
        + SvpApplyDftToDftInplace<B>
        + VecZnxIdftApplyConsume<B>
        + VecZnxBigAddInplace<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxBigNormalize<B>
        + VecZnxNormalizeTmpBytes
        + VecZnxSubScalarInplace
        + VecZnxCopy
        + VmpPMatAlloc<B>
        + VmpPrepare<B>,
    B: Backend
        + ScratchOwnedAllocImpl<B>
        + ScratchOwnedBorrowImpl<B>
        + ScratchAvailableImpl<B>
        + TakeScalarZnxImpl<B>
        + TakeVecZnxDftImpl<B>
        + TakeVecZnxImpl<B>
        + TakeSvpPPolImpl<B>
        + TakeVecZnxBigImpl<B>,
{
    let basek: usize = 12;
    let k_in: usize = 60;
    let digits: usize = k_in.div_ceil(basek);
    let p0: i64 = -1;
    let p1: i64 = -5;
    (1..3).for_each(|rank| {
        (1..digits + 1).for_each(|di| {
            println!(
                "test_gglwe_automorphism_key_automorphism_inplace: {} rank: {}",
                di, rank
            );
            let k_apply: usize = (digits + di) * basek;

            let n: usize = module.n();
            let digits_in: usize = 1;

            let rows_in: usize = k_in / (basek * di);
            let rows_apply: usize = k_in.div_ceil(basek * di);

            let mut auto_key: GGLWEAutomorphismKey<Vec<u8>> =
                GGLWEAutomorphismKey::alloc(n, basek, k_in, rows_in, digits_in, rank);
            let mut auto_key_apply: GGLWEAutomorphismKey<Vec<u8>> =
                GGLWEAutomorphismKey::alloc(n, basek, k_apply, rows_apply, di, rank);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
                GGLWEAutomorphismKey::encrypt_sk_scratch_space(module, basek, k_apply, rank)
                    | GGLWEAutomorphismKey::automorphism_inplace_scratch_space(module, basek, k_in, k_apply, di, rank),
            );

            let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
            sk.fill_ternary_prob(0.5, &mut source_xs);

            // gglwe_{s1}(s0) = s0 -> s1
            auto_key.encrypt_sk(
                module,
                p0,
                &sk,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            // gglwe_{s2}(s1) -> s1 -> s2
            auto_key_apply.encrypt_sk(
                module,
                p1,
                &sk,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            let mut auto_key_apply_prepared: GGLWEAutomorphismKeyPrepared<Vec<u8>, B> =
                GGLWEAutomorphismKeyPrepared::alloc(module, basek, k_apply, rows_apply, di, rank);

            auto_key_apply_prepared.prepare(module, &auto_key_apply, scratch.borrow());

            // gglwe_{s1}(s0) (x) gglwe_{s2}(s1) = gglwe_{s2}(s0)
            auto_key.automorphism_inplace(module, &auto_key_apply_prepared, scratch.borrow());

            let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(n, basek, k_in);

            let mut sk_auto: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
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

            let sk_auto_dft: GLWESecretPrepared<Vec<u8>, B> = sk_auto.prepare_alloc(module, scratch.borrow());

            (0..auto_key.rank_in()).for_each(|col_i| {
                (0..auto_key.rows()).for_each(|row_i| {
                    auto_key
                        .at(row_i, col_i)
                        .decrypt(module, &mut pt, &sk_auto_dft, scratch.borrow());
                    module.vec_znx_sub_scalar_inplace(
                        &mut pt.data,
                        0,
                        (digits_in - 1) + row_i * digits_in,
                        &sk.data,
                        col_i,
                    );

                    let noise_have: f64 = pt.data.std(basek, 0).log2();
                    let noise_want: f64 = log2_std_noise_gglwe_product(
                        n as f64,
                        basek * di,
                        0.5,
                        0.5,
                        0f64,
                        SIGMA * SIGMA,
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
        });
    });
}
