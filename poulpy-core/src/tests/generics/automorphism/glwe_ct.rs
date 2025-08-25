use poulpy_hal::{
    api::{
        DFT, IDFTConsume, ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyInplace, SvpPPolAlloc, SvpPPolAllocBytes, SvpPrepare,
        VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxAutomorphism, VecZnxAutomorphismInplace,
        VecZnxBigAddInplace, VecZnxBigAddSmallInplace, VecZnxBigAllocBytes, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxDftAllocBytes, VecZnxFillUniform, VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxSub,
        VecZnxSubABInplace, VecZnxSwithcDegree, VmpApply, VmpApplyAdd, VmpApplyTmpBytes, VmpPMatAlloc, VmpPrepare,
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
        GGLWEAutomorphismKey, GLWECiphertext, GLWEPlaintext, GLWESecret, Infos,
        prepared::{GGLWEAutomorphismKeyPrepared, GLWESecretPrepared, Prepare, PrepareAlloc},
    },
    noise::log2_std_noise_gglwe_product,
};

#[allow(clippy::too_many_arguments)]
pub fn test_glwe_automorphism<B>(
    module: &Module<B>,
    basek: usize,
    p: i64,
    k_out: usize,
    k_in: usize,
    k_ksk: usize,
    digits: usize,
    rank: usize,
) where
    Module<B>: VecZnxDftAllocBytes
        + VecZnxBigNormalize<B>
        + DFT<B>
        + SvpApplyInplace<B>
        + IDFTConsume<B>
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
        + VecZnxBigAllocBytes
        + VecZnxBigAddInplace<B>
        + VecZnxBigAddSmallInplace<B>
        + VmpApplyTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VmpApply<B>
        + VmpApplyAdd<B>
        + VecZnxAutomorphism
        + VecZnxSwithcDegree
        + VecZnxAddScalarInplace
        + VecZnxAutomorphismInplace
        + VmpPMatAlloc<B>
        + VmpPrepare<B>,
    B: Backend
        + TakeVecZnxDftImpl<B>
        + TakeVecZnxBigImpl<B>
        + TakeSvpPPolImpl<B>
        + ScratchOwnedAllocImpl<B>
        + ScratchOwnedBorrowImpl<B>
        + ScratchAvailableImpl<B>
        + TakeScalarZnxImpl<B>
        + TakeVecZnxImpl<B>,
{
    let n: usize = module.n();
    let rows: usize = k_in.div_ceil(basek * digits);

    let mut autokey: GGLWEAutomorphismKey<Vec<u8>> = GGLWEAutomorphismKey::alloc(n, basek, k_ksk, rows, digits, rank);
    let mut ct_in: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(n, basek, k_in, rank);
    let mut ct_out: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(n, basek, k_out, rank);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(n, basek, k_in);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    module.vec_znx_fill_uniform(basek, &mut pt_want.data, 0, k_in, &mut source_xa);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
        GGLWEAutomorphismKey::encrypt_sk_scratch_space(module, basek, autokey.k(), rank)
            | GLWECiphertext::decrypt_scratch_space(module, basek, ct_out.k())
            | GLWECiphertext::encrypt_sk_scratch_space(module, basek, ct_in.k())
            | GLWECiphertext::automorphism_scratch_space(
                module,
                basek,
                ct_out.k(),
                ct_in.k(),
                autokey.k(),
                digits,
                rank,
            ),
    );

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let sk_prepared: GLWESecretPrepared<Vec<u8>, B> = sk.prepare_alloc(module, scratch.borrow());

    autokey.encrypt_sk(
        module,
        p,
        &sk,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    ct_in.encrypt_sk(
        module,
        &pt_want,
        &sk_prepared,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    let mut autokey_prepared: GGLWEAutomorphismKeyPrepared<Vec<u8>, B> =
        GGLWEAutomorphismKeyPrepared::alloc(module, basek, k_ksk, rows, digits, rank);
    autokey_prepared.prepare(module, &autokey, scratch.borrow());

    ct_out.automorphism(module, &ct_in, &autokey_prepared, scratch.borrow());

    let max_noise: f64 = log2_std_noise_gglwe_product(
        module.n() as f64,
        basek * digits,
        0.5,
        0.5,
        0f64,
        SIGMA * SIGMA,
        0f64,
        rank as f64,
        k_in,
        k_ksk,
    );

    module.vec_znx_automorphism_inplace(p, &mut pt_want.data, 0);

    ct_out.assert_noise(module, &sk_prepared, &pt_want, max_noise + 1.0);
}

#[allow(clippy::too_many_arguments)]
pub fn test_glwe_automorphism_inplace<B>(
    module: &Module<B>,
    basek: usize,
    p: i64,
    k_ct: usize,
    k_ksk: usize,
    digits: usize,
    rank: usize,
) where
    Module<B>: VecZnxDftAllocBytes
        + VecZnxBigNormalize<B>
        + DFT<B>
        + SvpApplyInplace<B>
        + IDFTConsume<B>
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
        + VecZnxBigAllocBytes
        + VecZnxBigAddInplace<B>
        + VecZnxBigAddSmallInplace<B>
        + VmpApplyTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VmpApply<B>
        + VmpApplyAdd<B>
        + VecZnxAutomorphism
        + VecZnxSwithcDegree
        + VecZnxAddScalarInplace
        + VecZnxAutomorphismInplace
        + VmpPMatAlloc<B>
        + VmpPrepare<B>,
    B: Backend
        + TakeVecZnxDftImpl<B>
        + TakeVecZnxBigImpl<B>
        + TakeSvpPPolImpl<B>
        + ScratchOwnedAllocImpl<B>
        + ScratchOwnedBorrowImpl<B>
        + ScratchAvailableImpl<B>
        + TakeScalarZnxImpl<B>
        + TakeVecZnxImpl<B>,
{
    let n: usize = module.n();
    let rows: usize = k_ct.div_ceil(basek * digits);

    let mut autokey: GGLWEAutomorphismKey<Vec<u8>> = GGLWEAutomorphismKey::alloc(n, basek, k_ksk, rows, digits, rank);
    let mut ct: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(n, basek, k_ct, rank);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(n, basek, k_ct);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    module.vec_znx_fill_uniform(basek, &mut pt_want.data, 0, k_ct, &mut source_xa);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
        GGLWEAutomorphismKey::encrypt_sk_scratch_space(module, basek, autokey.k(), rank)
            | GLWECiphertext::decrypt_scratch_space(module, basek, ct.k())
            | GLWECiphertext::encrypt_sk_scratch_space(module, basek, ct.k())
            | GLWECiphertext::automorphism_inplace_scratch_space(module, basek, ct.k(), autokey.k(), digits, rank),
    );

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let sk_prepared: GLWESecretPrepared<Vec<u8>, B> = sk.prepare_alloc(module, scratch.borrow());

    autokey.encrypt_sk(
        module,
        p,
        &sk,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    ct.encrypt_sk(
        module,
        &pt_want,
        &sk_prepared,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    let mut autokey_prepared: GGLWEAutomorphismKeyPrepared<Vec<u8>, B> =
        GGLWEAutomorphismKeyPrepared::alloc(module, basek, k_ksk, rows, digits, rank);
    autokey_prepared.prepare(module, &autokey, scratch.borrow());

    ct.automorphism_inplace(module, &autokey_prepared, scratch.borrow());

    let max_noise: f64 = log2_std_noise_gglwe_product(
        module.n() as f64,
        basek * digits,
        0.5,
        0.5,
        0f64,
        SIGMA * SIGMA,
        0f64,
        rank as f64,
        k_ct,
        k_ksk,
    );

    module.vec_znx_automorphism_inplace(p, &mut pt_want.data, 0);

    ct.assert_noise(module, &sk_prepared, &pt_want, max_noise + 1.0);
}
