use poulpy_hal::{
    api::{
        DFT, IDFTConsume, ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyInplace, SvpPPolAlloc, SvpPPolAllocBytes, SvpPrepare,
        VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxBigAddInplace, VecZnxBigAddSmallInplace,
        VecZnxBigAllocBytes, VecZnxBigNormalize, VecZnxCopy, VecZnxDftAllocBytes, VecZnxFillUniform, VecZnxNormalize,
        VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxSub, VecZnxSubABInplace, VecZnxSubScalarInplace,
        VecZnxSwithcDegree, VmpPMatAlloc, VmpPrepare,
    },
    layouts::{Backend, Module, ScratchOwned},
    oep::{
        ScratchAvailableImpl, ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeScalarZnxImpl, TakeSvpPPolImpl,
        TakeVecZnxBigImpl, TakeVecZnxDftImpl, TakeVecZnxImpl, VecZnxBigAllocBytesImpl, VecZnxDftAllocBytesImpl,
    },
    source::Source,
};

use crate::{
    encryption::SIGMA,
    layouts::{
        GGLWESwitchingKey, GLWESecret,
        compressed::{Decompress, GGLWESwitchingKeyCompressed},
        prepared::{GLWESecretPrepared, PrepareAlloc},
    },
};

pub fn test_gglwe_switching_key_encrypt_sk<B>(
    module: &Module<B>,
    basek: usize,
    k_ksk: usize,
    digits: usize,
    rank_in: usize,
    rank_out: usize,
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
        + VecZnxBigAddSmallInplace<B>
        + VecZnxSwithcDegree
        + VecZnxAddScalarInplace
        + VecZnxBigAllocBytes
        + VecZnxBigAddInplace<B>
        + VecZnxSubScalarInplace
        + VecZnxCopy
        + VmpPMatAlloc<B>
        + VmpPrepare<B>,
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
    let rows: usize = (k_ksk - digits * basek) / (digits * basek);

    let mut ksk: GGLWESwitchingKey<Vec<u8>> = GGLWESwitchingKey::alloc(n, basek, k_ksk, rows, digits, rank_in, rank_out);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(GGLWESwitchingKey::encrypt_sk_scratch_space(
        module, basek, k_ksk, rank_in, rank_out,
    ));

    let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank_in);
    sk_in.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank_out);
    sk_out.fill_ternary_prob(0.5, &mut source_xs);
    let sk_out_prepared: GLWESecretPrepared<Vec<u8>, B> = sk_out.prepare_alloc(module, scratch.borrow());

    ksk.encrypt_sk(
        module,
        &sk_in,
        &sk_out,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    ksk.key
        .assert_noise(module, &sk_out_prepared, &sk_in.data, SIGMA);
}

pub fn test_gglwe_switching_key_compressed_encrypt_sk<B>(
    module: &Module<B>,
    basek: usize,
    k_ksk: usize,
    digits: usize,
    rank_in: usize,
    rank_out: usize,
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
        + VecZnxBigAddSmallInplace<B>
        + VecZnxSwithcDegree
        + VecZnxAddScalarInplace
        + VecZnxBigAllocBytes
        + VecZnxBigAddInplace<B>
        + VecZnxSubScalarInplace
        + VecZnxCopy
        + VmpPMatAlloc<B>
        + VmpPrepare<B>,
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
    let rows: usize = (k_ksk - digits * basek) / (digits * basek);

    let mut ksk_compressed: GGLWESwitchingKeyCompressed<Vec<u8>> =
        GGLWESwitchingKeyCompressed::alloc(n, basek, k_ksk, rows, digits, rank_in, rank_out);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(GGLWESwitchingKeyCompressed::encrypt_sk_scratch_space(
        module, basek, k_ksk, rank_in, rank_out,
    ));

    let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank_in);
    sk_in.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank_out);
    sk_out.fill_ternary_prob(0.5, &mut source_xs);
    let sk_out_prepared: GLWESecretPrepared<Vec<u8>, B> = sk_out.prepare_alloc(module, scratch.borrow());

    let seed_xa = [1u8; 32];

    ksk_compressed.encrypt_sk(
        module,
        &sk_in,
        &sk_out,
        seed_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    let mut ksk: GGLWESwitchingKey<Vec<u8>> = GGLWESwitchingKey::alloc(n, basek, k_ksk, rows, digits, rank_in, rank_out);
    ksk.decompress(module, &ksk_compressed);

    ksk.key
        .assert_noise(module, &sk_out_prepared, &sk_in.data, SIGMA);
}
