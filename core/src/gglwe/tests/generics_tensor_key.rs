use backend::hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAddScalarInplace, VecZnxBigAlloc, VecZnxCopy, VecZnxDftAlloc, VecZnxStd,
        VecZnxSubScalarInplace, VecZnxSwithcDegree,
    },
    layouts::{Backend, Module, ScratchOwned, VecZnxDft},
    oep::{
        ScratchAvailableImpl, ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeScalarZnxImpl, TakeSvpPPolImpl,
        TakeVecZnxBigImpl, TakeVecZnxDftImpl, TakeVecZnxImpl, VecZnxBigAllocBytesImpl, VecZnxDftAllocBytesImpl,
    },
};
use sampling::source::Source;

use crate::{
    GGLWEEncryptSkFamily, GGLWEExecLayoutFamily, GLWEDecryptFamily, GLWEPlaintext, GLWESecret, GLWESecretExec, GLWETensorKey,
    GLWETensorKeyCompressed, GLWETensorKeyEncryptSkFamily, Infos,
};

pub(crate) trait TestModuleFamily<B: Backend> = GGLWEEncryptSkFamily<B>
    + GLWEDecryptFamily<B>
    + VecZnxSwithcDegree
    + VecZnxAddScalarInplace
    + VecZnxStd
    + VecZnxSubScalarInplace;

pub(crate) trait TestScratchFamily<B: Backend> = TakeVecZnxDftImpl<B>
    + TakeVecZnxBigImpl<B>
    + TakeSvpPPolImpl<B>
    + ScratchOwnedAllocImpl<B>
    + ScratchOwnedBorrowImpl<B>
    + ScratchAvailableImpl<B>
    + TakeScalarZnxImpl<B>
    + TakeVecZnxImpl<B>
    + VecZnxDftAllocBytesImpl<B>
    + VecZnxBigAllocBytesImpl<B>
    + TakeSvpPPolImpl<B>;

pub(crate) fn test_tensor_key_encrypt_sk<B: Backend>(module: &Module<B>, basek: usize, k: usize, sigma: f64, rank: usize)
where
    Module<B>: TestModuleFamily<B>
        + GGLWEExecLayoutFamily<B>
        + GLWETensorKeyEncryptSkFamily<B>
        + GLWEDecryptFamily<B>
        + VecZnxDftAlloc<B>
        + VecZnxBigAlloc<B>,
    B: TestScratchFamily<B>,
{
    let n: usize = module.n();
    let rows: usize = k / basek;

    let mut tensor_key: GLWETensorKey<Vec<u8>> = GLWETensorKey::alloc(n, basek, k, rows, 1, rank);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(GLWETensorKey::encrypt_sk_scratch_space(
        module,
        n,
        basek,
        tensor_key.k(),
        rank,
    ));

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let mut sk_exec: GLWESecretExec<Vec<u8>, B> = GLWESecretExec::from(module, &sk);
    sk_exec.prepare(module, &sk);

    tensor_key.encrypt_sk(
        module,
        &sk,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(n, basek, k);

    let mut sk_ij_dft = module.vec_znx_dft_alloc(n, 1, 1);
    let mut sk_ij_big = module.vec_znx_big_alloc(n, 1, 1);
    let mut sk_ij: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, 1);
    let mut sk_dft: VecZnxDft<Vec<u8>, B> = module.vec_znx_dft_alloc(n, rank, 1);

    (0..rank).for_each(|i| {
        module.vec_znx_dft_from_vec_znx(1, 0, &mut sk_dft, i, &sk.data.as_vec_znx(), i);
    });

    (0..rank).for_each(|i| {
        (0..rank).for_each(|j| {
            module.svp_apply(&mut sk_ij_dft, 0, &sk_exec.data, j, &sk_dft, i);
            module.vec_znx_dft_to_vec_znx_big_tmp_a(&mut sk_ij_big, 0, &mut sk_ij_dft, 0);
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
                        .decrypt(module, &mut pt, &sk_exec, scratch.borrow());

                    module.vec_znx_sub_scalar_inplace(&mut pt.data, 0, row_i, &sk_ij.data, col_i);

                    let std_pt: f64 = module.vec_znx_std(basek, &pt.data, 0) * (k as f64).exp2();
                    assert!((sigma - std_pt).abs() <= 0.5, "{} {}", sigma, std_pt);
                });
            });
        })
    })
}

pub(crate) fn test_tensor_key_encrypt_sk_compressed<B: Backend>(
    module: &Module<B>,
    basek: usize,
    k: usize,
    sigma: f64,
    rank: usize,
) where
    Module<B>: TestModuleFamily<B>
        + GGLWEExecLayoutFamily<B>
        + GLWETensorKeyEncryptSkFamily<B>
        + GLWEDecryptFamily<B>
        + VecZnxDftAlloc<B>
        + VecZnxBigAlloc<B>
        + VecZnxCopy,
    B: TestScratchFamily<B>,
{
    let n: usize = module.n();
    let rows: usize = k / basek;

    let mut tensor_key_compressed: GLWETensorKeyCompressed<Vec<u8>> = GLWETensorKeyCompressed::alloc(n, basek, k, rows, 1, rank);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(GLWETensorKeyCompressed::encrypt_sk_scratch_space(
        module,
        n,
        basek,
        tensor_key_compressed.k(),
        rank,
    ));

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let mut sk_exec: GLWESecretExec<Vec<u8>, B> = GLWESecretExec::from(module, &sk);
    sk_exec.prepare(module, &sk);

    let seed_xa: [u8; 32] = [1u8; 32];

    tensor_key_compressed.encrypt_sk(
        module,
        &sk,
        seed_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let mut tensor_key: GLWETensorKey<Vec<u8>> = GLWETensorKey::alloc(n, basek, k, rows, 1, rank);
    tensor_key.decompress(module, &tensor_key_compressed);

    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(n, basek, k);

    let mut sk_ij_dft = module.vec_znx_dft_alloc(n, 1, 1);
    let mut sk_ij_big = module.vec_znx_big_alloc(n, 1, 1);
    let mut sk_ij: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, 1);
    let mut sk_dft: VecZnxDft<Vec<u8>, B> = module.vec_znx_dft_alloc(n, rank, 1);

    (0..rank).for_each(|i| {
        module.vec_znx_dft_from_vec_znx(1, 0, &mut sk_dft, i, &sk.data.as_vec_znx(), i);
    });

    (0..rank).for_each(|i| {
        (0..rank).for_each(|j| {
            module.svp_apply(&mut sk_ij_dft, 0, &sk_exec.data, j, &sk_dft, i);
            module.vec_znx_dft_to_vec_znx_big_tmp_a(&mut sk_ij_big, 0, &mut sk_ij_dft, 0);
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
                        .decrypt(module, &mut pt, &sk_exec, scratch.borrow());

                    module.vec_znx_sub_scalar_inplace(&mut pt.data, 0, row_i, &sk_ij.data, col_i);

                    let std_pt: f64 = module.vec_znx_std(basek, &pt.data, 0) * (k as f64).exp2();
                    assert!((sigma - std_pt).abs() <= 0.5, "{} {}", sigma, std_pt);
                });
            });
        })
    })
}
