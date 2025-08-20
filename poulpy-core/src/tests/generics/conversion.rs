use poulpy_hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyInplace, SvpPPolAlloc, SvpPPolAllocBytes, SvpPrepare, VecZnxAddInplace,
        VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxAutomorphismInplace, VecZnxBigAddInplace, VecZnxBigAddSmallInplace,
        VecZnxBigAllocBytes, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftAllocBytes, VecZnxDftFromVecZnx,
        VecZnxDftToVecZnxBigConsume, VecZnxFillUniform, VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes,
        VecZnxSub, VecZnxSubABInplace, VecZnxSwithcDegree, VmpApply, VmpApplyAdd, VmpApplyTmpBytes, VmpPMatAlloc, VmpPrepare,
        ZnxView,
    },
    layouts::{Backend, Module, ScratchOwned},
    oep::{
        ScratchAvailableImpl, ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeScalarZnxImpl, TakeSvpPPolImpl,
        TakeVecZnxBigImpl, TakeVecZnxDftImpl, TakeVecZnxImpl,
    },
    source::Source,
};

use crate::layouts::{
    GLWECiphertext, GLWEPlaintext, GLWESecret, GLWEToLWESwitchingKey, Infos, LWECiphertext, LWEPlaintext, LWESecret,
    LWEToGLWESwitchingKey,
    prepared::{GLWESecretPrepared, GLWEToLWESwitchingKeyPrepared, LWEToGLWESwitchingKeyPrepared, PrepareAlloc},
};

pub fn test_lwe_to_glwe<B>(module: &Module<B>)
where
    Module<B>: VecZnxDftAllocBytes
        + VecZnxBigNormalize<B>
        + VecZnxDftFromVecZnx<B>
        + SvpApplyInplace<B>
        + VecZnxDftToVecZnxBigConsume<B>
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
        + VecZnxNormalizeTmpBytes
        + VecZnxAddScalarInplace
        + VmpPMatAlloc<B>
        + VmpPrepare<B>
        + VmpApplyTmpBytes
        + VmpApply<B>
        + VmpApplyAdd<B>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxSwithcDegree
        + VecZnxAutomorphismInplace,
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
    let basek: usize = 17;

    let rank: usize = 2;

    let n_lwe: usize = 22;
    let k_lwe_ct: usize = 2 * basek;
    let k_lwe_pt: usize = 8;

    let k_glwe_ct: usize = 3 * basek;

    let k_ksk: usize = k_lwe_ct + basek;

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
        LWEToGLWESwitchingKey::encrypt_sk_scratch_space(module, n, basek, k_ksk, rank)
            | GLWECiphertext::from_lwe_scratch_space(module, n, basek, k_lwe_ct, k_glwe_ct, k_ksk, rank)
            | GLWECiphertext::decrypt_scratch_space(module, n, basek, k_glwe_ct),
    );

    let mut sk_glwe: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);

    let sk_glwe_prepared: GLWESecretPrepared<Vec<u8>, B> = sk_glwe.prepare_alloc(module, scratch.borrow());

    let mut sk_lwe: LWESecret<Vec<u8>> = LWESecret::alloc(n_lwe);
    sk_lwe.fill_ternary_prob(0.5, &mut source_xs);

    let data: i64 = 17;

    let mut lwe_pt: LWEPlaintext<Vec<u8>> = LWEPlaintext::alloc(basek, k_lwe_pt);
    lwe_pt.encode_i64(data, k_lwe_pt);

    let mut lwe_ct: LWECiphertext<Vec<u8>> = LWECiphertext::alloc(n_lwe, basek, k_lwe_ct);
    lwe_ct.encrypt_sk(module, &lwe_pt, &sk_lwe, &mut source_xa, &mut source_xe);

    let mut ksk: LWEToGLWESwitchingKey<Vec<u8>> = LWEToGLWESwitchingKey::alloc(n, basek, k_ksk, lwe_ct.size(), rank);

    ksk.encrypt_sk(
        module,
        &sk_lwe,
        &sk_glwe,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    let mut glwe_ct: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(n, basek, k_glwe_ct, rank);

    let ksk_prepared: LWEToGLWESwitchingKeyPrepared<Vec<u8>, B> = ksk.prepare_alloc(module, scratch.borrow());

    glwe_ct.from_lwe(module, &lwe_ct, &ksk_prepared, scratch.borrow());

    let mut glwe_pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(n, basek, k_glwe_ct);
    glwe_ct.decrypt(module, &mut glwe_pt, &sk_glwe_prepared, scratch.borrow());

    assert_eq!(glwe_pt.data.at(0, 0)[0], lwe_pt.data.at(0, 0)[0]);
}

pub fn test_glwe_to_lwe<B>(module: &Module<B>)
where
    Module<B>: VecZnxDftAllocBytes
        + VecZnxBigNormalize<B>
        + VecZnxDftFromVecZnx<B>
        + SvpApplyInplace<B>
        + VecZnxDftToVecZnxBigConsume<B>
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
        + VecZnxNormalizeTmpBytes
        + VecZnxAddScalarInplace
        + VmpPMatAlloc<B>
        + VmpPrepare<B>
        + VmpApplyTmpBytes
        + VmpApply<B>
        + VmpApplyAdd<B>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxSwithcDegree
        + VecZnxAutomorphismInplace,
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
    let basek: usize = 17;

    let rank: usize = 2;

    let n_lwe: usize = 22;
    let k_lwe_ct: usize = 2 * basek;
    let k_lwe_pt: usize = 8;

    let k_glwe_ct: usize = 3 * basek;

    let k_ksk: usize = k_lwe_ct + basek;

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
        LWEToGLWESwitchingKey::encrypt_sk_scratch_space(module, n, basek, k_ksk, rank)
            | LWECiphertext::from_glwe_scratch_space(module, n, basek, k_lwe_ct, k_glwe_ct, k_ksk, rank)
            | GLWECiphertext::decrypt_scratch_space(module, n, basek, k_glwe_ct),
    );

    let mut sk_glwe: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);

    let sk_glwe_prepared: GLWESecretPrepared<Vec<u8>, B> = sk_glwe.prepare_alloc(module, scratch.borrow());

    let mut sk_lwe = LWESecret::alloc(n_lwe);
    sk_lwe.fill_ternary_prob(0.5, &mut source_xs);

    let data: i64 = 17;
    let mut glwe_pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(n, basek, k_glwe_ct);
    glwe_pt.encode_coeff_i64(data, k_lwe_pt, 0);

    let mut glwe_ct = GLWECiphertext::alloc(n, basek, k_glwe_ct, rank);
    glwe_ct.encrypt_sk(
        module,
        &glwe_pt,
        &sk_glwe_prepared,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    let mut ksk: GLWEToLWESwitchingKey<Vec<u8>> = GLWEToLWESwitchingKey::alloc(n, basek, k_ksk, glwe_ct.size(), rank);

    ksk.encrypt_sk(
        module,
        &sk_lwe,
        &sk_glwe,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    let mut lwe_ct: LWECiphertext<Vec<u8>> = LWECiphertext::alloc(n_lwe, basek, k_lwe_ct);

    let ksk_prepared: GLWEToLWESwitchingKeyPrepared<Vec<u8>, B> = ksk.prepare_alloc(module, scratch.borrow());

    lwe_ct.from_glwe(module, &glwe_ct, &ksk_prepared, scratch.borrow());

    let mut lwe_pt: LWEPlaintext<Vec<u8>> = LWEPlaintext::alloc(basek, k_lwe_ct);
    lwe_ct.decrypt(module, &mut lwe_pt, &sk_lwe);

    assert_eq!(glwe_pt.data.at(0, 0)[0], lwe_pt.data.at(0, 0)[0]);
}
