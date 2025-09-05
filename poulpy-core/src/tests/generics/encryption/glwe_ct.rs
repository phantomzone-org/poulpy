use poulpy_hal::{
    api::{
        DFT, IDFTConsume, ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApply, SvpApplyInplace, SvpPPolAlloc, SvpPPolAllocBytes,
        SvpPrepare, VecZnxAddInplace, VecZnxAddNormal, VecZnxBigAddInplace, VecZnxBigAddNormal, VecZnxBigAddSmallInplace,
        VecZnxBigAllocBytes, VecZnxBigNormalize, VecZnxCopy, VecZnxDftAlloc, VecZnxDftAllocBytes, VecZnxFillUniform,
        VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxSub, VecZnxSubABInplace,
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
        GLWECiphertext, GLWEPlaintext, GLWEPublicKey, GLWESecret, Infos,
        compressed::{Decompress, GLWECiphertextCompressed},
        prepared::{GLWEPublicKeyPrepared, GLWESecretPrepared, PrepareAlloc},
    },
    operations::GLWEOperations,
};

pub fn test_glwe_encrypt_sk<B>(module: &Module<B>, basek: usize, k_ct: usize, k_pt: usize, rank: usize)
where
    Module<B>: VecZnxDftAllocBytes
        + VecZnxBigAllocBytes
        + DFT<B>
        + SvpApplyInplace<B>
        + IDFTConsume<B>
        + VecZnxBigAddInplace<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxBigNormalize<B>
        + VecZnxNormalizeTmpBytes
        + SvpPrepare<B>
        + SvpPPolAllocBytes
        + SvpPPolAlloc<B>
        + VecZnxDftAllocBytes
        + VecZnxBigAllocBytes
        + SvpPPolAllocBytes
        + SvpPrepare<B>
        + SvpApply<B>
        + IDFTConsume<B>
        + VecZnxBigAddNormal<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxBigNormalize<B>
        + VecZnxNormalizeTmpBytes
        + VecZnxDftAllocBytes
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
        + VecZnxSub,
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
    let n = module.n();
    let mut ct: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(n, basek, k_ct, rank);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(n, basek, k_pt);
    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(n, basek, k_pt);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
        GLWECiphertext::encrypt_sk_scratch_space(module, basek, ct.k())
            | GLWECiphertext::decrypt_scratch_space(module, basek, ct.k()),
    );

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let sk_prepared: GLWESecretPrepared<Vec<u8>, B> = sk.prepare_alloc(module, scratch.borrow());

    module.vec_znx_fill_uniform(basek, &mut pt_want.data, 0, &mut source_xa);

    ct.encrypt_sk(
        module,
        &pt_want,
        &sk_prepared,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    ct.decrypt(module, &mut pt_have, &sk_prepared, scratch.borrow());

    pt_want.sub_inplace_ab(module, &pt_have);

    let noise_have: f64 = pt_want.data.std(basek, 0) * (ct.k() as f64).exp2();
    let noise_want: f64 = SIGMA;

    assert!(noise_have <= noise_want + 0.2);
}

pub fn test_glwe_compressed_encrypt_sk<B>(module: &Module<B>, basek: usize, k_ct: usize, k_pt: usize, rank: usize)
where
    Module<B>: VecZnxDftAllocBytes
        + VecZnxBigAllocBytes
        + DFT<B>
        + SvpApplyInplace<B>
        + IDFTConsume<B>
        + VecZnxBigAddInplace<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxBigNormalize<B>
        + VecZnxNormalizeTmpBytes
        + SvpPrepare<B>
        + SvpPPolAllocBytes
        + SvpPPolAlloc<B>
        + VecZnxDftAllocBytes
        + VecZnxBigAllocBytes
        + SvpPPolAllocBytes
        + SvpPrepare<B>
        + SvpApply<B>
        + IDFTConsume<B>
        + VecZnxBigAddNormal<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxBigNormalize<B>
        + VecZnxNormalizeTmpBytes
        + VecZnxDftAllocBytes
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
        + VecZnxCopy,
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
    let n = module.n();
    let mut ct_compressed: GLWECiphertextCompressed<Vec<u8>> = GLWECiphertextCompressed::alloc(n, basek, k_ct, rank);

    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(n, basek, k_pt);
    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(n, basek, k_pt);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
        GLWECiphertextCompressed::encrypt_sk_scratch_space(module, basek, k_ct)
            | GLWECiphertext::decrypt_scratch_space(module, basek, k_ct),
    );

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let sk_prepared: GLWESecretPrepared<Vec<u8>, B> = sk.prepare_alloc(module, scratch.borrow());

    module.vec_znx_fill_uniform(basek, &mut pt_want.data, 0, &mut source_xa);

    let seed_xa: [u8; 32] = [1u8; 32];

    ct_compressed.encrypt_sk(
        module,
        &pt_want,
        &sk_prepared,
        seed_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    let mut ct: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(n, basek, k_ct, rank);
    ct.decompress(module, &ct_compressed);

    ct.decrypt(module, &mut pt_have, &sk_prepared, scratch.borrow());

    pt_want.sub_inplace_ab(module, &pt_have);

    let noise_have: f64 = pt_want.data.std(basek, 0) * (ct.k() as f64).exp2();
    let noise_want: f64 = SIGMA;

    assert!(
        noise_have <= noise_want + 0.2,
        "{} <= {}",
        noise_have,
        noise_want + 0.2
    );
}

pub fn test_glwe_encrypt_zero_sk<B>(module: &Module<B>, basek: usize, k_ct: usize, rank: usize)
where
    Module<B>: VecZnxDftAllocBytes
        + VecZnxBigAllocBytes
        + DFT<B>
        + SvpApplyInplace<B>
        + IDFTConsume<B>
        + VecZnxBigAddInplace<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxBigNormalize<B>
        + VecZnxNormalizeTmpBytes
        + SvpPrepare<B>
        + SvpPPolAllocBytes
        + SvpPPolAlloc<B>
        + VecZnxDftAllocBytes
        + VecZnxBigAllocBytes
        + SvpPPolAllocBytes
        + SvpPrepare<B>
        + SvpApply<B>
        + IDFTConsume<B>
        + VecZnxBigAddNormal<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxBigNormalize<B>
        + VecZnxNormalizeTmpBytes
        + VecZnxDftAllocBytes
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
        + VecZnxSub,
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
    let n = module.n();
    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(n, basek, k_ct);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([1u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
        GLWECiphertext::decrypt_scratch_space(module, basek, k_ct)
            | GLWECiphertext::encrypt_sk_scratch_space(module, basek, k_ct),
    );

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let sk_prepared: GLWESecretPrepared<Vec<u8>, B> = sk.prepare_alloc(module, scratch.borrow());

    let mut ct: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(n, basek, k_ct, rank);

    ct.encrypt_zero_sk(
        module,
        &sk_prepared,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );
    ct.decrypt(module, &mut pt, &sk_prepared, scratch.borrow());

    assert!((SIGMA - pt.data.std(basek, 0) * (k_ct as f64).exp2()) <= 0.2);
}

pub fn test_glwe_encrypt_pk<B>(module: &Module<B>, basek: usize, k_ct: usize, k_pk: usize, rank: usize)
where
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
        + VecZnxBigAddSmallInplace<B>
        + VecZnxBigAllocBytes
        + VecZnxBigAddInplace<B>
        + VecZnxCopy
        + VecZnxDftAlloc<B>
        + SvpApply<B>
        + VecZnxBigAddNormal<B>,
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
    let mut ct: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(n, basek, k_ct, rank);
    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(n, basek, k_ct);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(n, basek, k_ct);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);
    let mut source_xu: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
        GLWECiphertext::encrypt_sk_scratch_space(module, basek, ct.k())
            | GLWECiphertext::decrypt_scratch_space(module, basek, ct.k())
            | GLWECiphertext::encrypt_pk_scratch_space(module, basek, k_pk),
    );

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let sk_prepared: GLWESecretPrepared<Vec<u8>, B> = sk.prepare_alloc(module, scratch.borrow());

    let mut pk: GLWEPublicKey<Vec<u8>> = GLWEPublicKey::alloc(n, basek, k_pk, rank);
    pk.generate_from_sk(module, &sk_prepared, &mut source_xa, &mut source_xe);

    module.vec_znx_fill_uniform(basek, &mut pt_want.data, 0, &mut source_xa);

    let pk_prepared: GLWEPublicKeyPrepared<Vec<u8>, B> = pk.prepare_alloc(module, scratch.borrow());

    ct.encrypt_pk(
        module,
        &pt_want,
        &pk_prepared,
        &mut source_xu,
        &mut source_xe,
        scratch.borrow(),
    );

    ct.decrypt(module, &mut pt_have, &sk_prepared, scratch.borrow());

    pt_want.sub_inplace_ab(module, &pt_have);

    let noise_have: f64 = pt_want.data.std(basek, 0).log2();
    let noise_want: f64 = ((((rank as f64) + 1.0) * n as f64 * 0.5 * SIGMA * SIGMA).sqrt()).log2() - (k_ct as f64);

    assert!(
        noise_have <= noise_want + 0.2,
        "{} {}",
        noise_have,
        noise_want
    );
}
