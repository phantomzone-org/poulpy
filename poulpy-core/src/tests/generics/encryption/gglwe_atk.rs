use poulpy_hal::{
    api::{
        DFT, IDFTConsume, ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyInplace, SvpPPolAlloc, SvpPPolAllocBytes, SvpPrepare,
        VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxAutomorphism, VecZnxAutomorphismInplace,
        VecZnxBigAddInplace, VecZnxBigAddSmallInplace, VecZnxBigAllocBytes, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxCopy, VecZnxDftAllocBytes, VecZnxFillUniform, VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes,
        VecZnxSub, VecZnxSubABInplace, VecZnxSubScalarInplace, VecZnxSwithcDegree, VmpApply, VmpApplyAdd, VmpApplyTmpBytes,
        VmpPMatAlloc, VmpPrepare,
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
        GGLWEAutomorphismKey, GLWESecret,
        compressed::{Decompress, GGLWEAutomorphismKeyCompressed},
        prepared::{GLWESecretPrepared, PrepareAlloc},
    },
};

pub fn test_gglwe_automorphisk_key_encrypt_sk<B>(module: &Module<B>, basek: usize, k_ksk: usize, digits: usize, rank: usize)
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
        + VmpApplyTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VmpApply<B>
        + VmpApplyAdd<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxAutomorphism
        + VecZnxSwithcDegree
        + VecZnxAddScalarInplace
        + VecZnxAutomorphismInplace
        + VecZnxBigAllocBytes
        + VecZnxBigAddInplace<B>
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
    let n: usize = module.n();
    let rows: usize = (k_ksk - digits * basek) / (digits * basek);

    let mut atk: GGLWEAutomorphismKey<Vec<u8>> = GGLWEAutomorphismKey::alloc(n, basek, k_ksk, rows, digits, rank);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(GGLWEAutomorphismKey::encrypt_sk_scratch_space(
        module, basek, k_ksk, rank,
    ));

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);

    let p = -5;

    atk.encrypt_sk(
        module,
        p,
        &sk,
        &mut source_xa,
        &mut source_xe,
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
    let sk_out_prepared: GLWESecretPrepared<Vec<u8>, B> = sk_out.prepare_alloc(module, scratch.borrow());

    atk.key
        .key
        .assert_noise(module, &sk_out_prepared, &sk.data, SIGMA);
}

pub fn test_gglwe_automorphisk_key_compressed_encrypt_sk<B>(
    module: &Module<B>,
    basek: usize,
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
        + VmpApplyTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VmpApply<B>
        + VmpApplyAdd<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxAutomorphism
        + VecZnxSwithcDegree
        + VecZnxAddScalarInplace
        + VecZnxAutomorphismInplace
        + VecZnxBigAllocBytes
        + VecZnxBigAddInplace<B>
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
    let n: usize = module.n();
    let rows: usize = (k_ksk - digits * basek) / (digits * basek);

    let mut atk_compressed: GGLWEAutomorphismKeyCompressed<Vec<u8>> =
        GGLWEAutomorphismKeyCompressed::alloc(n, basek, k_ksk, rows, digits, rank);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(GGLWEAutomorphismKey::encrypt_sk_scratch_space(
        module, basek, k_ksk, rank,
    ));

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);

    let p = -5;

    let seed_xa: [u8; 32] = [1u8; 32];

    atk_compressed.encrypt_sk(module, p, &sk, seed_xa, &mut source_xe, scratch.borrow());

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
    let sk_out_prepared = sk_out.prepare_alloc(module, scratch.borrow());

    let mut atk: GGLWEAutomorphismKey<Vec<u8>> = GGLWEAutomorphismKey::alloc(n, basek, k_ksk, rows, digits, rank);
    atk.decompress(module, &atk_compressed);

    atk.key
        .key
        .assert_noise(module, &sk_out_prepared, &sk.data, SIGMA);
}
