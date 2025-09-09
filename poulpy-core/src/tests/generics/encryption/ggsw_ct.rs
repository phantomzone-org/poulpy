use poulpy_hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyInplace, SvpPPolAlloc, SvpPPolAllocBytes, SvpPrepare, VecZnxAddInplace,
        VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxBigAddInplace, VecZnxBigAddSmallInplace, VecZnxBigAlloc,
        VecZnxBigAllocBytes, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxCopy, VecZnxDftAlloc, VecZnxDftAllocBytes,
        VecZnxDftApply, VecZnxFillUniform, VecZnxIdftApplyConsume, VecZnxIdftApplyTmpA, VecZnxNormalize, VecZnxNormalizeInplace,
        VecZnxNormalizeTmpBytes, VecZnxSub, VecZnxSubABInplace, VmpPMatAlloc, VmpPrepare,
    },
    layouts::{Backend, Module, ScalarZnx, ScratchOwned},
    oep::{
        ScratchAvailableImpl, ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeScalarZnxImpl, TakeSvpPPolImpl,
        TakeVecZnxBigImpl, TakeVecZnxDftImpl, TakeVecZnxImpl, VecZnxBigAllocBytesImpl, VecZnxDftAllocBytesImpl,
    },
    source::Source,
};

use crate::{
    encryption::SIGMA,
    layouts::{
        GGSWCiphertext, GLWESecret,
        compressed::{Decompress, GGSWCiphertextCompressed},
        prepared::{GLWESecretPrepared, PrepareAlloc},
    },
};

pub fn test_ggsw_encrypt_sk<B>(module: &Module<B>, basek: usize, k: usize, digits: usize, rank: usize)
where
    Module<B>: VecZnxDftAllocBytes
        + VecZnxBigNormalize<B>
        + VecZnxDftApply<B>
        + SvpApplyInplace<B>
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
        + VecZnxBigAddSmallInplace<B>
        + VecZnxAddScalarInplace
        + VecZnxBigAllocBytes
        + VecZnxBigAddInplace<B>
        + VecZnxCopy
        + VmpPMatAlloc<B>
        + VmpPrepare<B>
        + VecZnxBigAlloc<B>
        + VecZnxDftAlloc<B>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxIdftApplyTmpA<B>,
    B: Backend
        + TakeVecZnxDftImpl<B>
        + TakeVecZnxBigImpl<B>
        + TakeSvpPPolImpl<B>
        + ScratchOwnedAllocImpl<B>
        + ScratchOwnedBorrowImpl<B>
        + ScratchAvailableImpl<B>
        + TakeScalarZnxImpl<B>
        + TakeVecZnxImpl<B>
        + VecZnxDftAllocBytesImpl<B>
        + VecZnxBigAllocBytesImpl<B>
        + TakeSvpPPolImpl<B>,
{
    let n: usize = module.n();
    let rows: usize = (k - digits * basek) / (digits * basek);

    let mut ct: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(n, basek, k, rows, digits, rank);

    let mut pt_scalar: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    pt_scalar.fill_ternary_hw(0, n, &mut source_xs);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(GGSWCiphertext::encrypt_sk_scratch_space(
        module, basek, k, rank,
    ));

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let sk_prepared: GLWESecretPrepared<Vec<u8>, B> = sk.prepare_alloc(module, scratch.borrow());

    ct.encrypt_sk(
        module,
        &pt_scalar,
        &sk_prepared,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    let noise_f = |_col_i: usize| -(k as f64) + SIGMA.log2() + 0.5;

    ct.assert_noise(module, &sk_prepared, &pt_scalar, noise_f);
}

pub fn test_ggsw_compressed_encrypt_sk<B>(module: &Module<B>, basek: usize, k: usize, digits: usize, rank: usize)
where
    Module<B>: VecZnxDftAllocBytes
        + VecZnxBigNormalize<B>
        + VecZnxDftApply<B>
        + SvpApplyInplace<B>
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
        + VecZnxBigAddSmallInplace<B>
        + VecZnxAddScalarInplace
        + VecZnxBigAllocBytes
        + VecZnxBigAddInplace<B>
        + VecZnxCopy
        + VmpPMatAlloc<B>
        + VmpPrepare<B>
        + VecZnxBigAlloc<B>
        + VecZnxDftAlloc<B>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxIdftApplyTmpA<B>,
    B: Backend
        + TakeVecZnxDftImpl<B>
        + TakeVecZnxBigImpl<B>
        + ScratchOwnedAllocImpl<B>
        + ScratchOwnedBorrowImpl<B>
        + ScratchAvailableImpl<B>
        + TakeScalarZnxImpl<B>
        + TakeVecZnxImpl<B>
        + VecZnxDftAllocBytesImpl<B>
        + VecZnxBigAllocBytesImpl<B>
        + TakeSvpPPolImpl<B>,
{
    let n: usize = module.n();
    let rows: usize = (k - digits * basek) / (digits * basek);

    let mut ct_compressed: GGSWCiphertextCompressed<Vec<u8>> = GGSWCiphertextCompressed::alloc(n, basek, k, rows, digits, rank);

    let mut pt_scalar: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);

    pt_scalar.fill_ternary_hw(0, n, &mut source_xs);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(GGSWCiphertextCompressed::encrypt_sk_scratch_space(
        module, basek, k, rank,
    ));

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let sk_prepared: GLWESecretPrepared<Vec<u8>, B> = sk.prepare_alloc(module, scratch.borrow());

    let seed_xa: [u8; 32] = [1u8; 32];

    ct_compressed.encrypt_sk(
        module,
        &pt_scalar,
        &sk_prepared,
        seed_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    let noise_f = |_col_i: usize| -(k as f64) + SIGMA.log2() + 0.5;

    let mut ct: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(n, basek, k, rows, digits, rank);
    ct.decompress(module, &ct_compressed);

    ct.assert_noise(module, &sk_prepared, &pt_scalar, noise_f);
}
