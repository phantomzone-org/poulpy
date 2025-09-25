use rand::RngCore;

use crate::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxBigAlloc, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftAdd,
        VecZnxDftAddInplace, VecZnxDftAlloc, VecZnxDftApply, VecZnxDftCopy, VecZnxDftSub, VecZnxDftSubABInplace,
        VecZnxDftSubBAInplace, VecZnxIdftApply, VecZnxIdftApplyConsume, VecZnxIdftApplyTmpA, VecZnxIdftApplyTmpBytes,
    },
    layouts::{Backend, DataViewMut, DigestU64, FillUniform, Module, ScratchOwned, VecZnx, VecZnxBig, VecZnxDft},
    source::Source,
};

pub fn test_vec_znx_dft_add<BR: Backend, BT: Backend>(base2k: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: VecZnxDftAdd<BR>
        + VecZnxDftAlloc<BR>
        + VecZnxDftApply<BR>
        + VecZnxIdftApplyConsume<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxDftAdd<BT>
        + VecZnxDftAlloc<BT>
        + VecZnxDftApply<BT>
        + VecZnxIdftApplyConsume<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest = a.digest_u64();

        let mut a_dft_ref: VecZnxDft<Vec<u8>, BR> = module_ref.vec_znx_dft_alloc(cols, a_size);
        let mut a_dft_test: VecZnxDft<Vec<u8>, BT> = module_test.vec_znx_dft_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_dft_apply(1, 0, &mut a_dft_ref, j, &a, j);
            module_test.vec_znx_dft_apply(1, 0, &mut a_dft_test, j, &a, j);
        }

        assert_eq!(a.digest_u64(), a_digest);

        let a_dft_ref_digest: u64 = a_dft_ref.digest_u64();
        let a_dft_test_digest: u64 = a_dft_test.digest_u64();

        for b_size in [1, 2, 3, 4] {
            let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, b_size);
            b.fill_uniform(base2k, &mut source);
            let b_digest: u64 = b.digest_u64();

            let mut b_dft_ref: VecZnxDft<Vec<u8>, BR> = module_ref.vec_znx_dft_alloc(cols, b_size);
            let mut b_dft_test: VecZnxDft<Vec<u8>, BT> = module_test.vec_znx_dft_alloc(cols, b_size);

            for j in 0..cols {
                module_ref.vec_znx_dft_apply(1, 0, &mut b_dft_ref, j, &b, j);
                module_test.vec_znx_dft_apply(1, 0, &mut b_dft_test, j, &b, j);
            }

            assert_eq!(b.digest_u64(), b_digest);

            let b_dft_ref_digest: u64 = b_dft_ref.digest_u64();
            let b_dft_test_digest: u64 = b_dft_test.digest_u64();

            for res_size in [1, 2, 3, 4] {
                let mut res_dft_ref: VecZnxDft<Vec<u8>, BR> = module_ref.vec_znx_dft_alloc(cols, res_size);
                let mut res_dft_test: VecZnxDft<Vec<u8>, BT> = module_test.vec_znx_dft_alloc(cols, res_size);

                // Set d to garbage
                source.fill_bytes(res_dft_ref.data_mut());
                source.fill_bytes(res_dft_test.data_mut());

                // Reference
                for i in 0..cols {
                    module_ref.vec_znx_dft_add(&mut res_dft_ref, i, &a_dft_ref, i, &b_dft_ref, i);
                    module_test.vec_znx_dft_add(&mut res_dft_test, i, &a_dft_test, i, &b_dft_test, i);
                }

                assert_eq!(a_dft_ref.digest_u64(), a_dft_ref_digest);
                assert_eq!(a_dft_test.digest_u64(), a_dft_test_digest);
                assert_eq!(b_dft_ref.digest_u64(), b_dft_ref_digest);
                assert_eq!(b_dft_test.digest_u64(), b_dft_test_digest);

                let res_big_ref: VecZnxBig<Vec<u8>, BR> = module_ref.vec_znx_idft_apply_consume(res_dft_ref);
                let res_big_test: VecZnxBig<Vec<u8>, BT> = module_test.vec_znx_idft_apply_consume(res_dft_test);

                let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
                let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

                let res_ref_digest: u64 = res_big_ref.digest_u64();
                let res_test_digest: u64 = res_big_test.digest_u64();

                for j in 0..cols {
                    module_ref.vec_znx_big_normalize(
                        base2k,
                        &mut res_small_ref,
                        j,
                        base2k,
                        &res_big_ref,
                        j,
                        scratch_ref.borrow(),
                    );
                    module_test.vec_znx_big_normalize(
                        base2k,
                        &mut res_small_test,
                        j,
                        base2k,
                        &res_big_test,
                        j,
                        scratch_test.borrow(),
                    );
                }

                assert_eq!(res_big_ref.digest_u64(), res_ref_digest);
                assert_eq!(res_big_test.digest_u64(), res_test_digest);

                assert_eq!(res_small_ref, res_small_test);
            }
        }
    }
}

pub fn test_vec_znx_dft_add_inplace<BR: Backend, BT: Backend>(base2k: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: VecZnxDftAddInplace<BR>
        + VecZnxDftAlloc<BR>
        + VecZnxDftApply<BR>
        + VecZnxIdftApplyConsume<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxDftAddInplace<BT>
        + VecZnxDftAlloc<BT>
        + VecZnxDftApply<BT>
        + VecZnxIdftApplyConsume<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());
    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest = a.digest_u64();

        let mut a_dft_ref: VecZnxDft<Vec<u8>, BR> = module_ref.vec_znx_dft_alloc(cols, a_size);
        let mut a_dft_test: VecZnxDft<Vec<u8>, BT> = module_test.vec_znx_dft_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_dft_apply(1, 0, &mut a_dft_ref, j, &a, j);
            module_test.vec_znx_dft_apply(1, 0, &mut a_dft_test, j, &a, j);
        }

        assert_eq!(a.digest_u64(), a_digest);

        let a_dft_ref_digest: u64 = a_dft_ref.digest_u64();
        let a_dft_test_digest: u64 = a_dft_test.digest_u64();

        for res_size in [1, 2, 3, 4] {
            let mut res: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
            res.fill_uniform(base2k, &mut source);
            let res_digest: u64 = res.digest_u64();

            let mut res_dft_ref: VecZnxDft<Vec<u8>, BR> = module_ref.vec_znx_dft_alloc(cols, res_size);
            let mut res_dft_test: VecZnxDft<Vec<u8>, BT> = module_test.vec_znx_dft_alloc(cols, res_size);

            for j in 0..cols {
                module_ref.vec_znx_dft_apply(1, 0, &mut res_dft_ref, j, &res, j);
                module_test.vec_znx_dft_apply(1, 0, &mut res_dft_test, j, &res, j);
            }

            assert_eq!(res.digest_u64(), res_digest);

            // Reference
            for i in 0..cols {
                module_ref.vec_znx_dft_add_inplace(&mut res_dft_ref, i, &a_dft_ref, i);
                module_test.vec_znx_dft_add_inplace(&mut res_dft_test, i, &a_dft_test, i);
            }

            assert_eq!(a_dft_ref.digest_u64(), a_dft_ref_digest);
            assert_eq!(a_dft_test.digest_u64(), a_dft_test_digest);

            let res_big_ref: VecZnxBig<Vec<u8>, BR> = module_ref.vec_znx_idft_apply_consume(res_dft_ref);
            let res_big_test: VecZnxBig<Vec<u8>, BT> = module_test.vec_znx_idft_apply_consume(res_dft_test);

            let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            let res_ref_digest: u64 = res_big_ref.digest_u64();
            let res_test_digest: u64 = res_big_test.digest_u64();

            for j in 0..cols {
                module_ref.vec_znx_big_normalize(
                    base2k,
                    &mut res_small_ref,
                    j,
                    base2k,
                    &res_big_ref,
                    j,
                    scratch_ref.borrow(),
                );
                module_test.vec_znx_big_normalize(
                    base2k,
                    &mut res_small_test,
                    j,
                    base2k,
                    &res_big_test,
                    j,
                    scratch_test.borrow(),
                );
            }

            assert_eq!(res_big_ref.digest_u64(), res_ref_digest);
            assert_eq!(res_big_test.digest_u64(), res_test_digest);

            assert_eq!(res_small_ref, res_small_test);
        }
    }
}

pub fn test_vec_znx_copy<BR: Backend, BT: Backend>(base2k: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: VecZnxDftCopy<BR>
        + VecZnxDftAlloc<BR>
        + VecZnxDftApply<BR>
        + VecZnxIdftApplyConsume<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxDftCopy<BT>
        + VecZnxDftAlloc<BT>
        + VecZnxDftApply<BT>
        + VecZnxIdftApplyConsume<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    for a_size in [1, 2, 6, 11] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest = a.digest_u64();

        let mut a_dft_ref: VecZnxDft<Vec<u8>, BR> = module_ref.vec_znx_dft_alloc(cols, a_size);
        let mut a_dft_test: VecZnxDft<Vec<u8>, BT> = module_test.vec_znx_dft_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_dft_apply(1, 0, &mut a_dft_ref, j, &a, j);
            module_test.vec_znx_dft_apply(1, 0, &mut a_dft_test, j, &a, j);
        }

        assert_eq!(a.digest_u64(), a_digest);

        let a_dft_ref_digest: u64 = a_dft_ref.digest_u64();
        let a_dft_test_digest: u64 = a_dft_test.digest_u64();

        for res_size in [1, 2, 6, 11] {
            for params in [[1, 0], [1, 1], [1, 2], [2, 2]] {
                let steps: usize = params[0];
                let offset: usize = params[1];

                let mut res_dft_ref: VecZnxDft<Vec<u8>, BR> = module_ref.vec_znx_dft_alloc(cols, res_size);
                let mut res_dft_test: VecZnxDft<Vec<u8>, BT> = module_test.vec_znx_dft_alloc(cols, res_size);

                // Set d to garbage
                source.fill_bytes(res_dft_ref.data_mut());
                source.fill_bytes(res_dft_test.data_mut());

                // Reference
                for i in 0..cols {
                    module_ref.vec_znx_dft_copy(steps, offset, &mut res_dft_ref, i, &a_dft_ref, i);
                    module_test.vec_znx_dft_copy(steps, offset, &mut res_dft_test, i, &a_dft_test, i);
                }

                assert_eq!(a_dft_ref.digest_u64(), a_dft_ref_digest);
                assert_eq!(a_dft_test.digest_u64(), a_dft_test_digest);

                let res_big_ref: VecZnxBig<Vec<u8>, BR> = module_ref.vec_znx_idft_apply_consume(res_dft_ref);
                let res_big_test: VecZnxBig<Vec<u8>, BT> = module_test.vec_znx_idft_apply_consume(res_dft_test);

                let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
                let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

                let res_ref_digest: u64 = res_big_ref.digest_u64();
                let res_test_digest: u64 = res_big_test.digest_u64();

                for j in 0..cols {
                    module_ref.vec_znx_big_normalize(
                        base2k,
                        &mut res_small_ref,
                        j,
                        base2k,
                        &res_big_ref,
                        j,
                        scratch_ref.borrow(),
                    );
                    module_test.vec_znx_big_normalize(
                        base2k,
                        &mut res_small_test,
                        j,
                        base2k,
                        &res_big_test,
                        j,
                        scratch_test.borrow(),
                    );
                }

                assert_eq!(res_big_ref.digest_u64(), res_ref_digest);
                assert_eq!(res_big_test.digest_u64(), res_test_digest);

                assert_eq!(res_small_ref, res_small_test);
            }
        }
    }
}

pub fn test_vec_znx_idft_apply<BR: Backend, BT: Backend>(base2k: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: VecZnxDftApply<BR>
        + VecZnxDftAlloc<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxIdftApply<BR>,
    Module<BT>: VecZnxDftApply<BT>
        + VecZnxDftAlloc<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxIdftApply<BT>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest: u64 = a.digest_u64();

        for res_size in [1, 2, 3, 4] {
            for params in [[1, 0], [1, 1], [1, 2], [2, 2]] {
                let steps: usize = params[0];
                let offset: usize = params[1];

                let mut res_dft_ref: VecZnxDft<Vec<u8>, BR> = module_ref.vec_znx_dft_alloc(cols, a_size);
                let mut res_dft_test: VecZnxDft<Vec<u8>, BT> = module_test.vec_znx_dft_alloc(cols, a_size);

                for j in 0..cols {
                    module_ref.vec_znx_dft_apply(steps, offset, &mut res_dft_ref, j, &a, j);
                    module_test.vec_znx_dft_apply(steps, offset, &mut res_dft_test, j, &a, j);
                }

                assert_eq!(a.digest_u64(), a_digest);
                let res_dft_ref_digest: u64 = res_dft_ref.digest_u64();
                let rest_dft_test_digest: u64 = res_dft_test.digest_u64();

                let mut res_big_ref: VecZnxBig<Vec<u8>, BR> = module_ref.vec_znx_big_alloc(cols, res_size);
                let mut res_big_test: VecZnxBig<Vec<u8>, BT> = module_test.vec_znx_big_alloc(cols, res_size);

                for j in 0..cols {
                    module_ref.vec_znx_idft_apply(&mut res_big_ref, j, &res_dft_ref, j, scratch_ref.borrow());
                    module_test.vec_znx_idft_apply(
                        &mut res_big_test,
                        j,
                        &res_dft_test,
                        j,
                        scratch_test.borrow(),
                    );
                }

                assert_eq!(res_dft_ref.digest_u64(), res_dft_ref_digest);
                assert_eq!(res_dft_test.digest_u64(), rest_dft_test_digest);

                let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
                let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

                let res_ref_digest: u64 = res_big_ref.digest_u64();
                let res_test_digest: u64 = res_big_test.digest_u64();

                for j in 0..cols {
                    module_ref.vec_znx_big_normalize(
                        base2k,
                        &mut res_small_ref,
                        j,
                        base2k,
                        &res_big_ref,
                        j,
                        scratch_ref.borrow(),
                    );
                    module_test.vec_znx_big_normalize(
                        base2k,
                        &mut res_small_test,
                        j,
                        base2k,
                        &res_big_test,
                        j,
                        scratch_test.borrow(),
                    );
                }

                assert_eq!(res_big_ref.digest_u64(), res_ref_digest);
                assert_eq!(res_big_test.digest_u64(), res_test_digest);

                assert_eq!(res_small_ref, res_small_test);
            }
        }
    }
}

pub fn test_vec_znx_idft_apply_tmpa<BR: Backend, BT: Backend>(base2k: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: VecZnxDftApply<BR>
        + VecZnxDftAlloc<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxIdftApplyTmpA<BR>,
    Module<BT>: VecZnxDftApply<BT>
        + VecZnxDftAlloc<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxIdftApplyTmpA<BT>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest: u64 = a.digest_u64();

        for res_size in [1, 2, 3, 4] {
            for params in [[1, 0], [1, 1], [1, 2], [2, 2]] {
                let steps: usize = params[0];
                let offset: usize = params[1];

                let mut res_dft_ref: VecZnxDft<Vec<u8>, BR> = module_ref.vec_znx_dft_alloc(cols, a_size);
                let mut res_dft_test: VecZnxDft<Vec<u8>, BT> = module_test.vec_znx_dft_alloc(cols, a_size);

                for j in 0..cols {
                    module_ref.vec_znx_dft_apply(steps, offset, &mut res_dft_ref, j, &a, j);
                    module_test.vec_znx_dft_apply(steps, offset, &mut res_dft_test, j, &a, j);
                }

                assert_eq!(a.digest_u64(), a_digest);

                let mut res_big_ref: VecZnxBig<Vec<u8>, BR> = module_ref.vec_znx_big_alloc(cols, res_size);
                let mut res_big_test: VecZnxBig<Vec<u8>, BT> = module_test.vec_znx_big_alloc(cols, res_size);

                for j in 0..cols {
                    module_ref.vec_znx_idft_apply_tmpa(&mut res_big_ref, j, &mut res_dft_ref, j);
                    module_test.vec_znx_idft_apply_tmpa(&mut res_big_test, j, &mut res_dft_test, j);
                }

                let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
                let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

                let res_ref_digest: u64 = res_big_ref.digest_u64();
                let res_test_digest: u64 = res_big_test.digest_u64();

                for j in 0..cols {
                    module_ref.vec_znx_big_normalize(
                        base2k,
                        &mut res_small_ref,
                        j,
                        base2k,
                        &res_big_ref,
                        j,
                        scratch_ref.borrow(),
                    );
                    module_test.vec_znx_big_normalize(
                        base2k,
                        &mut res_small_test,
                        j,
                        base2k,
                        &res_big_test,
                        j,
                        scratch_test.borrow(),
                    );
                }

                assert_eq!(res_big_ref.digest_u64(), res_ref_digest);
                assert_eq!(res_big_test.digest_u64(), res_test_digest);

                assert_eq!(res_small_ref, res_small_test);
            }
        }
    }
}

pub fn test_vec_znx_idft_apply_consume<BR: Backend, BT: Backend>(base2k: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: VecZnxDftApply<BR>
        + VecZnxIdftApplyTmpBytes
        + VecZnxDftAlloc<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxIdftApplyConsume<BR>,
    Module<BT>: VecZnxDftApply<BT>
        + VecZnxIdftApplyTmpBytes
        + VecZnxDftAlloc<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxIdftApplyConsume<BT>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> =
        ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes() | module_ref.vec_znx_idft_apply_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> =
        ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes() | module_test.vec_znx_idft_apply_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest: u64 = a.digest_u64();

        for res_size in [1, 2, 3, 4] {
            for params in [[1, 0], [1, 1], [1, 2], [2, 2]] {
                let steps: usize = params[0];
                let offset: usize = params[1];

                let mut res_dft_ref: VecZnxDft<Vec<u8>, BR> = module_ref.vec_znx_dft_alloc(cols, a_size);
                let mut res_dft_test: VecZnxDft<Vec<u8>, BT> = module_test.vec_znx_dft_alloc(cols, a_size);

                for j in 0..cols {
                    module_ref.vec_znx_dft_apply(steps, offset, &mut res_dft_ref, j, &a, j);
                    module_test.vec_znx_dft_apply(steps, offset, &mut res_dft_test, j, &a, j);
                }

                assert_eq!(a.digest_u64(), a_digest);

                let res_big_ref: VecZnxBig<Vec<u8>, BR> = module_ref.vec_znx_idft_apply_consume(res_dft_ref);
                let res_big_test: VecZnxBig<Vec<u8>, BT> = module_test.vec_znx_idft_apply_consume(res_dft_test);

                let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
                let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

                let res_ref_digest: u64 = res_big_ref.digest_u64();
                let res_test_digest: u64 = res_big_test.digest_u64();

                for j in 0..cols {
                    module_ref.vec_znx_big_normalize(
                        base2k,
                        &mut res_small_ref,
                        j,
                        base2k,
                        &res_big_ref,
                        j,
                        scratch_ref.borrow(),
                    );
                    module_test.vec_znx_big_normalize(
                        base2k,
                        &mut res_small_test,
                        j,
                        base2k,
                        &res_big_test,
                        j,
                        scratch_test.borrow(),
                    );
                }

                assert_eq!(res_big_ref.digest_u64(), res_ref_digest);
                assert_eq!(res_big_test.digest_u64(), res_test_digest);

                assert_eq!(res_small_ref, res_small_test);
            }
        }
    }
}

pub fn test_vec_znx_dft_sub<BR: Backend, BT: Backend>(base2k: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: VecZnxDftSub<BR>
        + VecZnxDftAlloc<BR>
        + VecZnxDftApply<BR>
        + VecZnxIdftApplyConsume<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxDftSub<BT>
        + VecZnxDftAlloc<BT>
        + VecZnxDftApply<BT>
        + VecZnxIdftApplyConsume<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest = a.digest_u64();

        let mut a_dft_ref: VecZnxDft<Vec<u8>, BR> = module_ref.vec_znx_dft_alloc(cols, a_size);
        let mut a_dft_test: VecZnxDft<Vec<u8>, BT> = module_test.vec_znx_dft_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_dft_apply(1, 0, &mut a_dft_ref, j, &a, j);
            module_test.vec_znx_dft_apply(1, 0, &mut a_dft_test, j, &a, j);
        }

        assert_eq!(a.digest_u64(), a_digest);

        let a_dft_ref_digest: u64 = a_dft_ref.digest_u64();
        let a_dft_test_digest: u64 = a_dft_test.digest_u64();

        for b_size in [1, 2, 3, 4] {
            let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, b_size);
            b.fill_uniform(base2k, &mut source);
            let b_digest: u64 = b.digest_u64();

            let mut b_dft_ref: VecZnxDft<Vec<u8>, BR> = module_ref.vec_znx_dft_alloc(cols, b_size);
            let mut b_dft_test: VecZnxDft<Vec<u8>, BT> = module_test.vec_znx_dft_alloc(cols, b_size);

            for j in 0..cols {
                module_ref.vec_znx_dft_apply(1, 0, &mut b_dft_ref, j, &b, j);
                module_test.vec_znx_dft_apply(1, 0, &mut b_dft_test, j, &b, j);
            }

            assert_eq!(b.digest_u64(), b_digest);

            let b_dft_ref_digest: u64 = b_dft_ref.digest_u64();
            let b_dft_test_digest: u64 = b_dft_test.digest_u64();

            for res_size in [1, 2, 3, 4] {
                let mut res_dft_ref: VecZnxDft<Vec<u8>, BR> = module_ref.vec_znx_dft_alloc(cols, res_size);
                let mut res_dft_test: VecZnxDft<Vec<u8>, BT> = module_test.vec_znx_dft_alloc(cols, res_size);

                // Set d to garbage
                source.fill_bytes(res_dft_ref.data_mut());
                source.fill_bytes(res_dft_test.data_mut());

                // Reference
                for i in 0..cols {
                    module_ref.vec_znx_dft_sub(&mut res_dft_ref, i, &a_dft_ref, i, &b_dft_ref, i);
                    module_test.vec_znx_dft_sub(&mut res_dft_test, i, &a_dft_test, i, &b_dft_test, i);
                }

                assert_eq!(a_dft_ref.digest_u64(), a_dft_ref_digest);
                assert_eq!(a_dft_test.digest_u64(), a_dft_test_digest);
                assert_eq!(b_dft_ref.digest_u64(), b_dft_ref_digest);
                assert_eq!(b_dft_test.digest_u64(), b_dft_test_digest);

                let res_big_ref: VecZnxBig<Vec<u8>, BR> = module_ref.vec_znx_idft_apply_consume(res_dft_ref);
                let res_big_test: VecZnxBig<Vec<u8>, BT> = module_test.vec_znx_idft_apply_consume(res_dft_test);

                let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
                let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

                let res_ref_digest: u64 = res_big_ref.digest_u64();
                let res_test_digest: u64 = res_big_test.digest_u64();

                for j in 0..cols {
                    module_ref.vec_znx_big_normalize(
                        base2k,
                        &mut res_small_ref,
                        j,
                        base2k,
                        &res_big_ref,
                        j,
                        scratch_ref.borrow(),
                    );
                    module_test.vec_znx_big_normalize(
                        base2k,
                        &mut res_small_test,
                        j,
                        base2k,
                        &res_big_test,
                        j,
                        scratch_test.borrow(),
                    );
                }

                assert_eq!(res_big_ref.digest_u64(), res_ref_digest);
                assert_eq!(res_big_test.digest_u64(), res_test_digest);

                assert_eq!(res_small_ref, res_small_test);
            }
        }
    }
}

pub fn test_vec_znx_dft_sub_ab_inplace<BR: Backend, BT: Backend>(base2k: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: VecZnxDftSubABInplace<BR>
        + VecZnxDftAlloc<BR>
        + VecZnxDftApply<BR>
        + VecZnxIdftApplyConsume<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxDftSubABInplace<BT>
        + VecZnxDftAlloc<BT>
        + VecZnxDftApply<BT>
        + VecZnxIdftApplyConsume<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());
    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest = a.digest_u64();

        let mut a_dft_ref: VecZnxDft<Vec<u8>, BR> = module_ref.vec_znx_dft_alloc(cols, a_size);
        let mut a_dft_test: VecZnxDft<Vec<u8>, BT> = module_test.vec_znx_dft_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_dft_apply(1, 0, &mut a_dft_ref, j, &a, j);
            module_test.vec_znx_dft_apply(1, 0, &mut a_dft_test, j, &a, j);
        }

        assert_eq!(a.digest_u64(), a_digest);

        let a_dft_ref_digest: u64 = a_dft_ref.digest_u64();
        let a_dft_test_digest: u64 = a_dft_test.digest_u64();

        for res_size in [1, 2, 3, 4] {
            let mut res: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
            res.fill_uniform(base2k, &mut source);
            let res_digest: u64 = res.digest_u64();

            let mut res_dft_ref: VecZnxDft<Vec<u8>, BR> = module_ref.vec_znx_dft_alloc(cols, res_size);
            let mut res_dft_test: VecZnxDft<Vec<u8>, BT> = module_test.vec_znx_dft_alloc(cols, res_size);

            for j in 0..cols {
                module_ref.vec_znx_dft_apply(1, 0, &mut res_dft_ref, j, &res, j);
                module_test.vec_znx_dft_apply(1, 0, &mut res_dft_test, j, &res, j);
            }

            assert_eq!(res.digest_u64(), res_digest);

            // Reference
            for i in 0..cols {
                module_ref.vec_znx_dft_sub_ab_inplace(&mut res_dft_ref, i, &a_dft_ref, i);
                module_test.vec_znx_dft_sub_ab_inplace(&mut res_dft_test, i, &a_dft_test, i);
            }

            assert_eq!(a_dft_ref.digest_u64(), a_dft_ref_digest);
            assert_eq!(a_dft_test.digest_u64(), a_dft_test_digest);

            let res_big_ref: VecZnxBig<Vec<u8>, BR> = module_ref.vec_znx_idft_apply_consume(res_dft_ref);
            let res_big_test: VecZnxBig<Vec<u8>, BT> = module_test.vec_znx_idft_apply_consume(res_dft_test);

            let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            let res_ref_digest: u64 = res_big_ref.digest_u64();
            let res_test_digest: u64 = res_big_test.digest_u64();

            for j in 0..cols {
                module_ref.vec_znx_big_normalize(
                    base2k,
                    &mut res_small_ref,
                    j,
                    base2k,
                    &res_big_ref,
                    j,
                    scratch_ref.borrow(),
                );
                module_test.vec_znx_big_normalize(
                    base2k,
                    &mut res_small_test,
                    j,
                    base2k,
                    &res_big_test,
                    j,
                    scratch_test.borrow(),
                );
            }

            assert_eq!(res_big_ref.digest_u64(), res_ref_digest);
            assert_eq!(res_big_test.digest_u64(), res_test_digest);

            assert_eq!(res_small_ref, res_small_test);
        }
    }
}

pub fn test_vec_znx_dft_sub_ba_inplace<BR: Backend, BT: Backend>(base2k: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: VecZnxDftSubBAInplace<BR>
        + VecZnxDftAlloc<BR>
        + VecZnxDftApply<BR>
        + VecZnxIdftApplyConsume<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxDftSubBAInplace<BT>
        + VecZnxDftAlloc<BT>
        + VecZnxDftApply<BT>
        + VecZnxIdftApplyConsume<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());
    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest = a.digest_u64();

        let mut a_dft_ref: VecZnxDft<Vec<u8>, BR> = module_ref.vec_znx_dft_alloc(cols, a_size);
        let mut a_dft_test: VecZnxDft<Vec<u8>, BT> = module_test.vec_znx_dft_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_dft_apply(1, 0, &mut a_dft_ref, j, &a, j);
            module_test.vec_znx_dft_apply(1, 0, &mut a_dft_test, j, &a, j);
        }

        assert_eq!(a.digest_u64(), a_digest);

        let a_dft_ref_digest: u64 = a_dft_ref.digest_u64();
        let a_dft_test_digest: u64 = a_dft_test.digest_u64();

        for res_size in [1, 2, 3, 4] {
            let mut res: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
            res.fill_uniform(base2k, &mut source);
            let res_digest: u64 = res.digest_u64();

            let mut res_dft_ref: VecZnxDft<Vec<u8>, BR> = module_ref.vec_znx_dft_alloc(cols, res_size);
            let mut res_dft_test: VecZnxDft<Vec<u8>, BT> = module_test.vec_znx_dft_alloc(cols, res_size);

            for j in 0..cols {
                module_ref.vec_znx_dft_apply(1, 0, &mut res_dft_ref, j, &res, j);
                module_test.vec_znx_dft_apply(1, 0, &mut res_dft_test, j, &res, j);
            }

            assert_eq!(res.digest_u64(), res_digest);

            // Reference
            for i in 0..cols {
                module_ref.vec_znx_dft_sub_ba_inplace(&mut res_dft_ref, i, &a_dft_ref, i);
                module_test.vec_znx_dft_sub_ba_inplace(&mut res_dft_test, i, &a_dft_test, i);
            }

            assert_eq!(a_dft_ref.digest_u64(), a_dft_ref_digest);
            assert_eq!(a_dft_test.digest_u64(), a_dft_test_digest);

            let res_big_ref: VecZnxBig<Vec<u8>, BR> = module_ref.vec_znx_idft_apply_consume(res_dft_ref);
            let res_big_test: VecZnxBig<Vec<u8>, BT> = module_test.vec_znx_idft_apply_consume(res_dft_test);

            let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            let res_ref_digest: u64 = res_big_ref.digest_u64();
            let res_test_digest: u64 = res_big_test.digest_u64();

            for j in 0..cols {
                module_ref.vec_znx_big_normalize(
                    base2k,
                    &mut res_small_ref,
                    j,
                    base2k,
                    &res_big_ref,
                    j,
                    scratch_ref.borrow(),
                );
                module_test.vec_znx_big_normalize(
                    base2k,
                    &mut res_small_test,
                    j,
                    base2k,
                    &res_big_test,
                    j,
                    scratch_test.borrow(),
                );
            }

            assert_eq!(res_big_ref.digest_u64(), res_ref_digest);
            assert_eq!(res_big_test.digest_u64(), res_test_digest);

            assert_eq!(res_small_ref, res_small_test);
        }
    }
}
