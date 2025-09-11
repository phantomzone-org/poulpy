use poulpy_hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyDftToDft, SvpApplyDftToDftInplace, SvpPPolAlloc, SvpPPolAllocBytes,
        SvpPrepare, VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxBigAddInplace, VecZnxBigAddSmallInplace,
        VecZnxBigAlloc, VecZnxBigAllocBytes, VecZnxBigNormalize, VecZnxCopy, VecZnxDftAlloc, VecZnxDftAllocBytes, VecZnxDftApply,
        VecZnxFillUniform, VecZnxIdftApplyConsume, VecZnxIdftApplyTmpA, VecZnxNormalize, VecZnxNormalizeInplace,
        VecZnxNormalizeTmpBytes, VecZnxSub, VecZnxSubABInplace, VecZnxSubScalarInplace, VecZnxSwitchRing,
    },
    layouts::{Backend, Module, ScratchOwned, VecZnxDft},
    oep::{
        ScratchAvailableImpl, ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeScalarZnxImpl, TakeSvpPPolImpl,
        TakeVecZnxBigImpl, TakeVecZnxDftImpl, TakeVecZnxImpl, VecZnxBigAllocBytesImpl, VecZnxDftAllocBytesImpl,
    },
    source::Source,
};

use crate::{
    encryption::SIGMA,
    layouts::{
        GGLWETensorKey, GLWEPlaintext, GLWESecret, Infos,
        compressed::{Decompress, GGLWETensorKeyCompressed},
        prepared::{GLWESecretPrepared, PrepareAlloc},
    },
};

pub fn test_glwe_tensor_key_encrypt_sk<B>(module: &Module<B>, basek: usize, k: usize, rank: usize)
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
        + VecZnxBigAddSmallInplace<B>
        + VecZnxBigAllocBytes
        + VecZnxBigAddInplace<B>
        + VecZnxCopy
        + VecZnxDftAlloc<B>
        + SvpApplyDftToDft<B>
        + VecZnxBigAlloc<B>
        + VecZnxIdftApplyTmpA<B>
        + VecZnxAddScalarInplace
        + VecZnxSwitchRing<B>
        + VecZnxSubScalarInplace,
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
    let rows: usize = k / basek;

    let mut tensor_key: GGLWETensorKey<Vec<u8>> = GGLWETensorKey::alloc(n, basek, k, rows, 1, rank);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(GGLWETensorKey::encrypt_sk_scratch_space(
        module,
        basek,
        tensor_key.k(),
        rank,
    ));

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let sk_prepared: GLWESecretPrepared<Vec<u8>, B> = sk.prepare_alloc(module, scratch.borrow());

    tensor_key.encrypt_sk(
        module,
        &sk,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(n, basek, k);

    let mut sk_ij_dft = module.vec_znx_dft_alloc(1, 1);
    let mut sk_ij_big = module.vec_znx_big_alloc(1, 1);
    let mut sk_ij: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, 1);
    let mut sk_dft: VecZnxDft<Vec<u8>, B> = module.vec_znx_dft_alloc(rank, 1);

    (0..rank).for_each(|i| {
        module.vec_znx_dft_apply(1, 0, &mut sk_dft, i, &sk.data.as_vec_znx(), i);
    });

    (0..rank).for_each(|i| {
        (0..rank).for_each(|j| {
            module.svp_apply_dft_to_dft(&mut sk_ij_dft, 0, &sk_prepared.data, j, &sk_dft, i);
            module.vec_znx_idft_apply_tmpa(&mut sk_ij_big, 0, &mut sk_ij_dft, 0);
            module.vec_znx_big_normalize(
                basek,
                &mut sk_ij.data.as_vec_znx_mut(),
                0,
                &sk_ij_big,
                0,
                scratch.borrow(),
            );
            (0..tensor_key.rank_in()).for_each(|col_i| {
                (0..tensor_key.rows()).for_each(|row_i| {
                    tensor_key
                        .at(i, j)
                        .at(row_i, col_i)
                        .decrypt(module, &mut pt, &sk_prepared, scratch.borrow());

                    module.vec_znx_sub_scalar_inplace(&mut pt.data, 0, row_i, &sk_ij.data, col_i);

                    let std_pt: f64 = pt.data.std(basek, 0) * (k as f64).exp2();
                    assert!((SIGMA - std_pt).abs() <= 0.5, "{} {}", SIGMA, std_pt);
                });
            });
        })
    })
}

pub fn test_glwe_tensor_key_compressed_encrypt_sk<B>(module: &Module<B>, basek: usize, k: usize, rank: usize)
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
        + VecZnxBigAddSmallInplace<B>
        + VecZnxBigAllocBytes
        + VecZnxBigAddInplace<B>
        + VecZnxCopy
        + VecZnxDftAlloc<B>
        + SvpApplyDftToDft<B>
        + VecZnxBigAlloc<B>
        + VecZnxIdftApplyTmpA<B>
        + VecZnxAddScalarInplace
        + VecZnxSwitchRing<B>
        + VecZnxSubScalarInplace,
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
    let rows: usize = k / basek;

    let mut tensor_key_compressed: GGLWETensorKeyCompressed<Vec<u8>> =
        GGLWETensorKeyCompressed::alloc(n, basek, k, rows, 1, rank);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(GGLWETensorKeyCompressed::encrypt_sk_scratch_space(
        module,
        basek,
        tensor_key_compressed.k(),
        rank,
    ));

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let sk_prepared: GLWESecretPrepared<Vec<u8>, B> = sk.prepare_alloc(module, scratch.borrow());

    let seed_xa: [u8; 32] = [1u8; 32];

    tensor_key_compressed.encrypt_sk(module, &sk, seed_xa, &mut source_xe, scratch.borrow());

    let mut tensor_key: GGLWETensorKey<Vec<u8>> = GGLWETensorKey::alloc(n, basek, k, rows, 1, rank);
    tensor_key.decompress(module, &tensor_key_compressed);

    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(n, basek, k);

    let mut sk_ij_dft = module.vec_znx_dft_alloc(1, 1);
    let mut sk_ij_big = module.vec_znx_big_alloc(1, 1);
    let mut sk_ij: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, 1);
    let mut sk_dft: VecZnxDft<Vec<u8>, B> = module.vec_znx_dft_alloc(rank, 1);

    (0..rank).for_each(|i| {
        module.vec_znx_dft_apply(1, 0, &mut sk_dft, i, &sk.data.as_vec_znx(), i);
    });

    (0..rank).for_each(|i| {
        (0..rank).for_each(|j| {
            module.svp_apply_dft_to_dft(&mut sk_ij_dft, 0, &sk_prepared.data, j, &sk_dft, i);
            module.vec_znx_idft_apply_tmpa(&mut sk_ij_big, 0, &mut sk_ij_dft, 0);
            module.vec_znx_big_normalize(
                basek,
                &mut sk_ij.data.as_vec_znx_mut(),
                0,
                &sk_ij_big,
                0,
                scratch.borrow(),
            );
            (0..tensor_key.rank_in()).for_each(|col_i| {
                (0..tensor_key.rows()).for_each(|row_i| {
                    tensor_key
                        .at(i, j)
                        .at(row_i, col_i)
                        .decrypt(module, &mut pt, &sk_prepared, scratch.borrow());

                    module.vec_znx_sub_scalar_inplace(&mut pt.data, 0, row_i, &sk_ij.data, col_i);

                    let std_pt: f64 = pt.data.std(basek, 0) * (k as f64).exp2();
                    assert!((SIGMA - std_pt).abs() <= 0.5, "{} {}", SIGMA, std_pt);
                });
            });
        })
    })
}
