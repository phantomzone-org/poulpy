use rand::RngCore;

use crate::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxBigAdd, VecZnxBigAddInplace, VecZnxBigAddSmall, VecZnxBigAddSmallInplace,
        VecZnxBigAlloc, VecZnxBigAutomorphism, VecZnxBigAutomorphismInplace, VecZnxBigAutomorphismInplaceTmpBytes,
        VecZnxBigFromSmall, VecZnxBigNegate, VecZnxBigNegateInplace, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxBigSub, VecZnxBigSubInplace, VecZnxBigSubNegateInplace, VecZnxBigSubSmallA, VecZnxBigSubSmallB,
        VecZnxBigSubSmallInplace, VecZnxBigSubSmallNegateInplace,
    },
    layouts::{Backend, DataViewMut, DigestU64, FillUniform, Module, ScratchOwned, VecZnx, VecZnxBig},
    source::Source,
};

pub fn test_vec_znx_big_add<BR: Backend, BT: Backend>(base2k: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>:
        VecZnxBigAdd<BR> + VecZnxBigAlloc<BR> + VecZnxBigFromSmall<BR> + VecZnxBigNormalize<BR> + VecZnxBigNormalizeTmpBytes,
    Module<BT>:
        VecZnxBigAdd<BT> + VecZnxBigAlloc<BT> + VecZnxBigFromSmall<BT> + VecZnxBigNormalize<BT> + VecZnxBigNormalizeTmpBytes,
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

        let mut a_ref: VecZnxBig<Vec<u8>, BR> = module_ref.vec_znx_big_alloc(cols, a_size);
        let mut a_test: VecZnxBig<Vec<u8>, BT> = module_test.vec_znx_big_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_big_from_small(&mut a_ref, j, &a, j);
            module_test.vec_znx_big_from_small(&mut a_test, j, &a, j);
        }

        assert_eq!(a.digest_u64(), a_digest);

        let a_ref_digest: u64 = a_ref.digest_u64();
        let a_test_digest: u64 = a_test.digest_u64();

        for b_size in [1, 2, 3, 4] {
            let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, b_size);
            b.fill_uniform(base2k, &mut source);
            let b_digest = b.digest_u64();

            let mut b_ref: VecZnxBig<Vec<u8>, BR> = module_ref.vec_znx_big_alloc(cols, b_size);
            let mut b_test: VecZnxBig<Vec<u8>, BT> = module_test.vec_znx_big_alloc(cols, b_size);

            for j in 0..cols {
                module_ref.vec_znx_big_from_small(&mut b_ref, j, &b, j);
                module_test.vec_znx_big_from_small(&mut b_test, j, &b, j);
            }

            assert_eq!(b.digest_u64(), b_digest);

            let b_ref_digest: u64 = b_ref.digest_u64();
            let b_test_digest: u64 = b_test.digest_u64();

            for res_size in [1, 2, 3, 4] {
                let mut res_big_ref: VecZnxBig<Vec<u8>, BR> = module_ref.vec_znx_big_alloc(cols, res_size);
                let mut res_big_test: VecZnxBig<Vec<u8>, BT> = module_test.vec_znx_big_alloc(cols, res_size);

                // Set res to garbage
                source.fill_bytes(res_big_ref.data_mut());
                source.fill_bytes(res_big_test.data_mut());

                // Reference
                for i in 0..cols {
                    module_ref.vec_znx_big_add(&mut res_big_ref, i, &a_ref, i, &b_ref, i);
                    module_test.vec_znx_big_add(&mut res_big_test, i, &a_test, i, &b_test, i);
                }

                assert_eq!(a_ref.digest_u64(), a_ref_digest);
                assert_eq!(a_test.digest_u64(), a_test_digest);
                assert_eq!(b_ref.digest_u64(), b_ref_digest);
                assert_eq!(b_test.digest_u64(), b_test_digest);

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

pub fn test_vec_znx_big_add_inplace<BR: Backend, BT: Backend>(base2k: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: VecZnxBigAddInplace<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmall<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigAddInplace<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmall<BT>
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

        let mut a_ref: VecZnxBig<Vec<u8>, BR> = module_ref.vec_znx_big_alloc(cols, a_size);
        let mut a_test: VecZnxBig<Vec<u8>, BT> = module_test.vec_znx_big_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_big_from_small(&mut a_ref, j, &a, j);
            module_test.vec_znx_big_from_small(&mut a_test, j, &a, j);
        }

        let a_ref_digest: u64 = a_ref.digest_u64();
        let a_test_digest: u64 = a_test.digest_u64();

        for res_size in [1, 2, 3, 4] {
            let mut res: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            res.fill_uniform(base2k, &mut source);

            let mut res_big_ref: VecZnxBig<Vec<u8>, BR> = module_ref.vec_znx_big_alloc(cols, res_size);
            let mut res_big_test: VecZnxBig<Vec<u8>, BT> = module_test.vec_znx_big_alloc(cols, res_size);

            for j in 0..cols {
                module_ref.vec_znx_big_from_small(&mut res_big_ref, j, &res, j);
                module_test.vec_znx_big_from_small(&mut res_big_test, j, &res, j);
            }

            for i in 0..cols {
                module_ref.vec_znx_big_add_inplace(&mut res_big_ref, i, &a_ref, i);
                module_test.vec_znx_big_add_inplace(&mut res_big_test, i, &a_test, i);
            }

            assert_eq!(a_ref.digest_u64(), a_ref_digest);
            assert_eq!(a_test.digest_u64(), a_test_digest);

            let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            let res_ref_digest: u64 = res_big_ref.digest_u64();
            let res_test_digest: u64 = res_big_test.digest_u64();

            for j in 0..cols {
                module_ref.vec_znx_big_normalize(base2k, &mut res_small_ref, j, base2k, &res_big_ref, j, scratch_ref.borrow());
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

pub fn test_vec_znx_big_add_small<BR: Backend, BT: Backend>(base2k: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>:
        VecZnxBigAddSmall<BR> + VecZnxBigAlloc<BR> + VecZnxBigFromSmall<BR> + VecZnxBigNormalize<BR> + VecZnxBigNormalizeTmpBytes,
    Module<BT>:
        VecZnxBigAddSmall<BT> + VecZnxBigAlloc<BT> + VecZnxBigFromSmall<BT> + VecZnxBigNormalize<BT> + VecZnxBigNormalizeTmpBytes,
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

        let mut a_ref: VecZnxBig<Vec<u8>, BR> = module_ref.vec_znx_big_alloc(cols, a_size);
        let mut a_test: VecZnxBig<Vec<u8>, BT> = module_test.vec_znx_big_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_big_from_small(&mut a_ref, j, &a, j);
            module_test.vec_znx_big_from_small(&mut a_test, j, &a, j);
        }

        let a_ref_digest: u64 = a_ref.digest_u64();
        let a_test_digest: u64 = a_test.digest_u64();

        for b_size in [1, 2, 3, 4] {
            let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, b_size);
            b.fill_uniform(base2k, &mut source);
            let b_digest: u64 = b.digest_u64();

            for res_size in [1, 2, 3, 4] {
                let mut res_big_ref: VecZnxBig<Vec<u8>, BR> = module_ref.vec_znx_big_alloc(cols, res_size);
                let mut res_big_test: VecZnxBig<Vec<u8>, BT> = module_test.vec_znx_big_alloc(cols, res_size);

                // Set res to garbage
                source.fill_bytes(res_big_ref.data_mut());
                source.fill_bytes(res_big_test.data_mut());

                // Reference
                for i in 0..cols {
                    module_ref.vec_znx_big_add_small(&mut res_big_ref, i, &a_ref, i, &b, i);
                    module_test.vec_znx_big_add_small(&mut res_big_test, i, &a_test, i, &b, i);
                }

                assert_eq!(a_ref.digest_u64(), a_ref_digest);
                assert_eq!(a_test.digest_u64(), a_test_digest);
                assert_eq!(b.digest_u64(), b_digest);

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

pub fn test_vec_znx_big_add_small_inplace<BR: Backend, BT: Backend>(
    base2k: usize,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxBigAddSmallInplace<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmall<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigAddSmallInplace<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmall<BT>
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

        let a_digest: u64 = a.digest_u64();

        for res_size in [1, 2, 3, 4] {
            let mut res: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            res.fill_uniform(base2k, &mut source);

            let mut res_big_ref: VecZnxBig<Vec<u8>, BR> = module_ref.vec_znx_big_alloc(cols, res_size);
            let mut res_big_test: VecZnxBig<Vec<u8>, BT> = module_test.vec_znx_big_alloc(cols, res_size);

            for j in 0..cols {
                module_ref.vec_znx_big_from_small(&mut res_big_ref, j, &res, j);
                module_test.vec_znx_big_from_small(&mut res_big_test, j, &res, j);
            }

            for i in 0..cols {
                module_ref.vec_znx_big_add_small_inplace(&mut res_big_ref, i, &a, i);
                module_test.vec_znx_big_add_small_inplace(&mut res_big_test, i, &a, i);
            }

            assert_eq!(a.digest_u64(), a_digest);

            let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            let res_ref_digest: u64 = res_big_ref.digest_u64();
            let res_test_digest: u64 = res_big_test.digest_u64();

            for j in 0..cols {
                module_ref.vec_znx_big_normalize(base2k, &mut res_small_ref, j, base2k, &res_big_ref, j, scratch_ref.borrow());
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

pub fn test_vec_znx_big_automorphism<BR: Backend, BT: Backend>(base2k: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: VecZnxBigAutomorphism<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmall<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigAutomorphism<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmall<BT>
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

        let mut a_ref: VecZnxBig<Vec<u8>, BR> = module_ref.vec_znx_big_alloc(cols, a_size);
        let mut a_test: VecZnxBig<Vec<u8>, BT> = module_test.vec_znx_big_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_big_from_small(&mut a_ref, j, &a, j);
            module_test.vec_znx_big_from_small(&mut a_test, j, &a, j);
        }

        let a_ref_digest: u64 = a_ref.digest_u64();
        let a_test_digest: u64 = a_test.digest_u64();

        for res_size in [1, 2, 3, 4] {
            for p in [-5, 5] {
                let mut res_big_ref: VecZnxBig<Vec<u8>, BR> = module_ref.vec_znx_big_alloc(cols, res_size);
                let mut res_big_test: VecZnxBig<Vec<u8>, BT> = module_test.vec_znx_big_alloc(cols, res_size);

                // Set res to garbage
                source.fill_bytes(res_big_ref.data_mut());
                source.fill_bytes(res_big_test.data_mut());

                // Reference
                for i in 0..cols {
                    module_ref.vec_znx_big_automorphism(p, &mut res_big_ref, i, &a_ref, i);
                    module_test.vec_znx_big_automorphism(p, &mut res_big_test, i, &a_test, i);
                }

                assert_eq!(a_ref.digest_u64(), a_ref_digest);
                assert_eq!(a_test.digest_u64(), a_test_digest);

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

pub fn test_vec_znx_big_automorphism_inplace<BR: Backend, BT: Backend>(
    base2k: usize,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxBigAutomorphismInplace<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmall<BR>
        + VecZnxBigAutomorphismInplaceTmpBytes
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigAutomorphismInplace<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmall<BT>
        + VecZnxBigAutomorphismInplaceTmpBytes
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(
        module_ref.vec_znx_big_automorphism_inplace_tmp_bytes() | module_ref.vec_znx_big_normalize_tmp_bytes(),
    );
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(
        module_test.vec_znx_big_automorphism_inplace_tmp_bytes() | module_test.vec_znx_big_normalize_tmp_bytes(),
    );

    for res_size in [1, 2, 3, 4] {
        let mut res: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
        res.fill_uniform(base2k, &mut source);

        let mut res_big_ref: VecZnxBig<Vec<u8>, BR> = module_ref.vec_znx_big_alloc(cols, res_size);
        let mut res_big_test: VecZnxBig<Vec<u8>, BT> = module_test.vec_znx_big_alloc(cols, res_size);

        for p in [-5, 5] {
            for j in 0..cols {
                module_ref.vec_znx_big_from_small(&mut res_big_ref, j, &res, j);
                module_test.vec_znx_big_from_small(&mut res_big_test, j, &res, j);
            }

            for i in 0..cols {
                module_ref.vec_znx_big_automorphism_inplace(p, &mut res_big_ref, i, scratch_ref.borrow());
                module_test.vec_znx_big_automorphism_inplace(p, &mut res_big_test, i, scratch_test.borrow());
            }

            let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            let res_ref_digest: u64 = res_big_ref.digest_u64();
            let res_test_digest: u64 = res_big_test.digest_u64();

            for j in 0..cols {
                module_ref.vec_znx_big_normalize(base2k, &mut res_small_ref, j, base2k, &res_big_ref, j, scratch_ref.borrow());
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

pub fn test_vec_znx_big_negate<BR: Backend, BT: Backend>(base2k: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>:
        VecZnxBigNegate<BR> + VecZnxBigAlloc<BR> + VecZnxBigFromSmall<BR> + VecZnxBigNormalize<BR> + VecZnxBigNormalizeTmpBytes,
    Module<BT>:
        VecZnxBigNegate<BT> + VecZnxBigAlloc<BT> + VecZnxBigFromSmall<BT> + VecZnxBigNormalize<BT> + VecZnxBigNormalizeTmpBytes,
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

        let mut a_ref: VecZnxBig<Vec<u8>, BR> = module_ref.vec_znx_big_alloc(cols, a_size);
        let mut a_test: VecZnxBig<Vec<u8>, BT> = module_test.vec_znx_big_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_big_from_small(&mut a_ref, j, &a, j);
            module_test.vec_znx_big_from_small(&mut a_test, j, &a, j);
        }

        let a_ref_digest: u64 = a_ref.digest_u64();
        let a_test_digest: u64 = a_test.digest_u64();

        for res_size in [1, 2, 3, 4] {
            let mut res_big_ref: VecZnxBig<Vec<u8>, BR> = module_ref.vec_znx_big_alloc(cols, res_size);
            let mut res_big_test: VecZnxBig<Vec<u8>, BT> = module_test.vec_znx_big_alloc(cols, res_size);

            // Set res to garbage
            source.fill_bytes(res_big_ref.data_mut());
            source.fill_bytes(res_big_test.data_mut());

            // Reference
            for i in 0..cols {
                module_ref.vec_znx_big_negate(&mut res_big_ref, i, &a_ref, i);
                module_test.vec_znx_big_negate(&mut res_big_test, i, &a_test, i);
            }

            assert_eq!(a_ref.digest_u64(), a_ref_digest);
            assert_eq!(a_test.digest_u64(), a_test_digest);

            let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            let res_ref_digest: u64 = res_big_ref.digest_u64();
            let res_test_digest: u64 = res_big_test.digest_u64();

            for j in 0..cols {
                module_ref.vec_znx_big_normalize(base2k, &mut res_small_ref, j, base2k, &res_big_ref, j, scratch_ref.borrow());
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

pub fn test_vec_znx_big_negate_inplace<BR: Backend, BT: Backend>(base2k: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: VecZnxBigNegateInplace<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmall<BR>
        + VecZnxBigAutomorphismInplaceTmpBytes
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigNegateInplace<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmall<BT>
        + VecZnxBigAutomorphismInplaceTmpBytes
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

    for res_size in [1, 2, 3, 4] {
        let mut res: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
        res.fill_uniform(base2k, &mut source);

        let mut res_big_ref: VecZnxBig<Vec<u8>, BR> = module_ref.vec_znx_big_alloc(cols, res_size);
        let mut res_big_test: VecZnxBig<Vec<u8>, BT> = module_test.vec_znx_big_alloc(cols, res_size);

        for j in 0..cols {
            module_ref.vec_znx_big_from_small(&mut res_big_ref, j, &res, j);
            module_test.vec_znx_big_from_small(&mut res_big_test, j, &res, j);
        }

        for i in 0..cols {
            module_ref.vec_znx_big_negate_inplace(&mut res_big_ref, i);
            module_test.vec_znx_big_negate_inplace(&mut res_big_test, i);
        }

        let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
        let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

        let res_ref_digest: u64 = res_big_ref.digest_u64();
        let res_test_digest: u64 = res_big_test.digest_u64();

        for j in 0..cols {
            module_ref.vec_znx_big_normalize(base2k, &mut res_small_ref, j, base2k, &res_big_ref, j, scratch_ref.borrow());
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

pub fn test_vec_znx_big_normalize<BR: Backend, BT: Backend>(base2k: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: VecZnxBigAlloc<BR>
        + VecZnxBigFromSmall<BR>
        + VecZnxBigAutomorphismInplaceTmpBytes
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigAlloc<BT>
        + VecZnxBigFromSmall<BT>
        + VecZnxBigAutomorphismInplaceTmpBytes
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(
        module_ref.vec_znx_big_automorphism_inplace_tmp_bytes() | module_ref.vec_znx_big_normalize_tmp_bytes(),
    );
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(
        module_test.vec_znx_big_automorphism_inplace_tmp_bytes() | module_test.vec_znx_big_normalize_tmp_bytes(),
    );

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(63, &mut source);

        let mut a_ref: VecZnxBig<Vec<u8>, BR> = module_ref.vec_znx_big_alloc(cols, a_size);
        let mut a_test: VecZnxBig<Vec<u8>, BT> = module_test.vec_znx_big_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_big_from_small(&mut a_ref, j, &a, j);
            module_test.vec_znx_big_from_small(&mut a_test, j, &a, j);
        }

        let a_ref_digest: u64 = a_ref.digest_u64();
        let a_test_digest: u64 = a_test.digest_u64();

        for res_size in [1, 2, 3, 4] {
            let mut res_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            // Set d to garbage
            source.fill_bytes(res_ref.data_mut());
            source.fill_bytes(res_test.data_mut());

            // Reference
            for j in 0..cols {
                module_ref.vec_znx_big_normalize(base2k, &mut res_ref, j, base2k, &a_ref, j, scratch_ref.borrow());
                module_test.vec_znx_big_normalize(base2k, &mut res_test, j, base2k, &a_test, j, scratch_test.borrow());
            }

            assert_eq!(a_ref.digest_u64(), a_ref_digest);
            assert_eq!(a_test.digest_u64(), a_test_digest);

            assert_eq!(res_ref, res_test);
        }
    }
}

pub fn test_vec_znx_big_sub<BR: Backend, BT: Backend>(base2k: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>:
        VecZnxBigSub<BR> + VecZnxBigAlloc<BR> + VecZnxBigFromSmall<BR> + VecZnxBigNormalize<BR> + VecZnxBigNormalizeTmpBytes,
    Module<BT>:
        VecZnxBigSub<BT> + VecZnxBigAlloc<BT> + VecZnxBigFromSmall<BT> + VecZnxBigNormalize<BT> + VecZnxBigNormalizeTmpBytes,
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

        let mut a_ref: VecZnxBig<Vec<u8>, BR> = module_ref.vec_znx_big_alloc(cols, a_size);
        let mut a_test: VecZnxBig<Vec<u8>, BT> = module_test.vec_znx_big_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_big_from_small(&mut a_ref, j, &a, j);
            module_test.vec_znx_big_from_small(&mut a_test, j, &a, j);
        }

        let a_ref_digest: u64 = a_ref.digest_u64();
        let a_test_digest: u64 = a_test.digest_u64();

        for b_size in [1, 2, 3, 4] {
            let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, b_size);
            b.fill_uniform(base2k, &mut source);

            let mut b_ref: VecZnxBig<Vec<u8>, BR> = module_ref.vec_znx_big_alloc(cols, b_size);
            let mut b_test: VecZnxBig<Vec<u8>, BT> = module_test.vec_znx_big_alloc(cols, b_size);

            for j in 0..cols {
                module_ref.vec_znx_big_from_small(&mut b_ref, j, &b, j);
                module_test.vec_znx_big_from_small(&mut b_test, j, &b, j);
            }

            let b_ref_digest: u64 = b_ref.digest_u64();
            let b_test_digest: u64 = b_test.digest_u64();

            for res_size in [1, 2, 3, 4] {
                let mut res_big_ref: VecZnxBig<Vec<u8>, BR> = module_ref.vec_znx_big_alloc(cols, res_size);
                let mut res_big_test: VecZnxBig<Vec<u8>, BT> = module_test.vec_znx_big_alloc(cols, res_size);

                // Set res to garbage
                source.fill_bytes(res_big_ref.data_mut());
                source.fill_bytes(res_big_test.data_mut());

                // Reference
                for i in 0..cols {
                    module_ref.vec_znx_big_sub(&mut res_big_ref, i, &a_ref, i, &b_ref, i);
                    module_test.vec_znx_big_sub(&mut res_big_test, i, &a_test, i, &b_test, i);
                }

                assert_eq!(a_ref.digest_u64(), a_ref_digest);
                assert_eq!(a_test.digest_u64(), a_test_digest);
                assert_eq!(b_ref.digest_u64(), b_ref_digest);
                assert_eq!(b_test.digest_u64(), b_test_digest);

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

pub fn test_vec_znx_big_sub_inplace<BR: Backend, BT: Backend>(base2k: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: VecZnxBigSubInplace<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmall<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigSubInplace<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmall<BT>
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

        let mut a_ref: VecZnxBig<Vec<u8>, BR> = module_ref.vec_znx_big_alloc(cols, a_size);
        let mut a_test: VecZnxBig<Vec<u8>, BT> = module_test.vec_znx_big_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_big_from_small(&mut a_ref, j, &a, j);
            module_test.vec_znx_big_from_small(&mut a_test, j, &a, j);
        }

        let a_ref_digest: u64 = a_ref.digest_u64();
        let a_test_digest: u64 = a_test.digest_u64();

        for res_size in [1, 2, 3, 4] {
            let mut res: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            res.fill_uniform(base2k, &mut source);

            let mut res_big_ref: VecZnxBig<Vec<u8>, BR> = module_ref.vec_znx_big_alloc(cols, res_size);
            let mut res_big_test: VecZnxBig<Vec<u8>, BT> = module_test.vec_znx_big_alloc(cols, res_size);

            for j in 0..cols {
                module_ref.vec_znx_big_from_small(&mut res_big_ref, j, &res, j);
                module_test.vec_znx_big_from_small(&mut res_big_test, j, &res, j);
            }

            for i in 0..cols {
                module_ref.vec_znx_big_sub_inplace(&mut res_big_ref, i, &a_ref, i);
                module_test.vec_znx_big_sub_inplace(&mut res_big_test, i, &a_test, i);
            }

            assert_eq!(a_ref.digest_u64(), a_ref_digest);
            assert_eq!(a_test.digest_u64(), a_test_digest);

            let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            let res_ref_digest: u64 = res_big_ref.digest_u64();
            let res_test_digest: u64 = res_big_test.digest_u64();

            for j in 0..cols {
                module_ref.vec_znx_big_normalize(base2k, &mut res_small_ref, j, base2k, &res_big_ref, j, scratch_ref.borrow());
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

pub fn test_vec_znx_big_sub_negate_inplace<BR: Backend, BT: Backend>(
    base2k: usize,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxBigSubNegateInplace<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmall<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigSubNegateInplace<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmall<BT>
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

        let mut a_ref: VecZnxBig<Vec<u8>, BR> = module_ref.vec_znx_big_alloc(cols, a_size);
        let mut a_test: VecZnxBig<Vec<u8>, BT> = module_test.vec_znx_big_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_big_from_small(&mut a_ref, j, &a, j);
            module_test.vec_znx_big_from_small(&mut a_test, j, &a, j);
        }

        let a_ref_digest: u64 = a_ref.digest_u64();
        let a_test_digest: u64 = a_test.digest_u64();

        for res_size in [1, 2, 3, 4] {
            let mut res: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            res.fill_uniform(base2k, &mut source);

            let mut res_big_ref: VecZnxBig<Vec<u8>, BR> = module_ref.vec_znx_big_alloc(cols, res_size);
            let mut res_big_test: VecZnxBig<Vec<u8>, BT> = module_test.vec_znx_big_alloc(cols, res_size);

            for j in 0..cols {
                module_ref.vec_znx_big_from_small(&mut res_big_ref, j, &res, j);
                module_test.vec_znx_big_from_small(&mut res_big_test, j, &res, j);
            }

            for i in 0..cols {
                module_ref.vec_znx_big_sub_negate_inplace(&mut res_big_ref, i, &a_ref, i);
                module_test.vec_znx_big_sub_negate_inplace(&mut res_big_test, i, &a_test, i);
            }

            assert_eq!(a_ref.digest_u64(), a_ref_digest);
            assert_eq!(a_test.digest_u64(), a_test_digest);

            let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            let res_ref_digest: u64 = res_big_ref.digest_u64();
            let res_test_digest: u64 = res_big_test.digest_u64();

            for j in 0..cols {
                module_ref.vec_znx_big_normalize(base2k, &mut res_small_ref, j, base2k, &res_big_ref, j, scratch_ref.borrow());
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

pub fn test_vec_znx_big_sub_small_a<BR: Backend, BT: Backend>(base2k: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: VecZnxBigSubSmallA<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmall<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigSubSmallA<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmall<BT>
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

        let mut a_ref: VecZnxBig<Vec<u8>, BR> = module_ref.vec_znx_big_alloc(cols, a_size);
        let mut a_test: VecZnxBig<Vec<u8>, BT> = module_test.vec_znx_big_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_big_from_small(&mut a_ref, j, &a, j);
            module_test.vec_znx_big_from_small(&mut a_test, j, &a, j);
        }

        let a_ref_digest: u64 = a_ref.digest_u64();
        let a_test_digest: u64 = a_test.digest_u64();

        for b_size in [1, 2, 3, 4] {
            let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, b_size);
            b.fill_uniform(base2k, &mut source);
            let b_digest: u64 = b.digest_u64();

            for res_size in [1, 2, 3, 4] {
                let mut res_big_ref: VecZnxBig<Vec<u8>, BR> = module_ref.vec_znx_big_alloc(cols, res_size);
                let mut res_big_test: VecZnxBig<Vec<u8>, BT> = module_test.vec_znx_big_alloc(cols, res_size);

                // Set res to garbage
                source.fill_bytes(res_big_ref.data_mut());
                source.fill_bytes(res_big_test.data_mut());

                // Reference
                for i in 0..cols {
                    module_ref.vec_znx_big_sub_small_a(&mut res_big_ref, i, &b, i, &a_ref, i);
                    module_test.vec_znx_big_sub_small_a(&mut res_big_test, i, &b, i, &a_test, i);
                }

                assert_eq!(a_ref.digest_u64(), a_ref_digest);
                assert_eq!(a_test.digest_u64(), a_test_digest);
                assert_eq!(b.digest_u64(), b_digest);

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

pub fn test_vec_znx_big_sub_small_b<BR: Backend, BT: Backend>(base2k: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: VecZnxBigSubSmallB<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmall<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigSubSmallB<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmall<BT>
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

        let mut a_ref: VecZnxBig<Vec<u8>, BR> = module_ref.vec_znx_big_alloc(cols, a_size);
        let mut a_test: VecZnxBig<Vec<u8>, BT> = module_test.vec_znx_big_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_big_from_small(&mut a_ref, j, &a, j);
            module_test.vec_znx_big_from_small(&mut a_test, j, &a, j);
        }

        let a_ref_digest: u64 = a_ref.digest_u64();
        let a_test_digest: u64 = a_test.digest_u64();

        for b_size in [1, 2, 3, 4] {
            let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, b_size);
            b.fill_uniform(base2k, &mut source);
            let b_digest: u64 = b.digest_u64();

            for res_size in [1, 2, 3, 4] {
                let mut res_big_ref: VecZnxBig<Vec<u8>, BR> = module_ref.vec_znx_big_alloc(cols, res_size);
                let mut res_big_test: VecZnxBig<Vec<u8>, BT> = module_test.vec_znx_big_alloc(cols, res_size);

                // Set res to garbage
                source.fill_bytes(res_big_ref.data_mut());
                source.fill_bytes(res_big_test.data_mut());

                // Reference
                for i in 0..cols {
                    module_ref.vec_znx_big_sub_small_b(&mut res_big_ref, i, &a_ref, i, &b, i);
                    module_test.vec_znx_big_sub_small_b(&mut res_big_test, i, &a_test, i, &b, i);
                }

                assert_eq!(a_ref.digest_u64(), a_ref_digest);
                assert_eq!(a_test.digest_u64(), a_test_digest);
                assert_eq!(b.digest_u64(), b_digest);

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

pub fn test_vec_znx_big_sub_small_a_inplace<BR: Backend, BT: Backend>(
    base2k: usize,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxBigSubSmallInplace<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmall<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigSubSmallInplace<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmall<BT>
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

        let a_digest: u64 = a.digest_u64();

        for res_size in [1, 2, 3, 4] {
            let mut res: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            res.fill_uniform(base2k, &mut source);

            let mut res_big_ref: VecZnxBig<Vec<u8>, BR> = module_ref.vec_znx_big_alloc(cols, res_size);
            let mut res_big_test: VecZnxBig<Vec<u8>, BT> = module_test.vec_znx_big_alloc(cols, res_size);

            for j in 0..cols {
                module_ref.vec_znx_big_from_small(&mut res_big_ref, j, &res, j);
                module_test.vec_znx_big_from_small(&mut res_big_test, j, &res, j);
            }

            for i in 0..cols {
                module_ref.vec_znx_big_sub_small_inplace(&mut res_big_ref, i, &a, i);
                module_test.vec_znx_big_sub_small_inplace(&mut res_big_test, i, &a, i);
            }

            assert_eq!(a.digest_u64(), a_digest);

            let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            let res_ref_digest: u64 = res_big_ref.digest_u64();
            let res_test_digest: u64 = res_big_test.digest_u64();

            for j in 0..cols {
                module_ref.vec_znx_big_normalize(base2k, &mut res_small_ref, j, base2k, &res_big_ref, j, scratch_ref.borrow());
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

pub fn test_vec_znx_big_sub_small_b_inplace<BR: Backend, BT: Backend>(
    base2k: usize,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxBigSubSmallNegateInplace<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmall<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigSubSmallNegateInplace<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmall<BT>
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

        let a_digest: u64 = a.digest_u64();

        for res_size in [1, 2, 3, 4] {
            let mut res: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            res.fill_uniform(base2k, &mut source);

            let mut res_big_ref: VecZnxBig<Vec<u8>, BR> = module_ref.vec_znx_big_alloc(cols, res_size);
            let mut res_big_test: VecZnxBig<Vec<u8>, BT> = module_test.vec_znx_big_alloc(cols, res_size);

            for j in 0..cols {
                module_ref.vec_znx_big_from_small(&mut res_big_ref, j, &res, j);
                module_test.vec_znx_big_from_small(&mut res_big_test, j, &res, j);
            }

            for i in 0..cols {
                module_ref.vec_znx_big_sub_small_negate_inplace(&mut res_big_ref, i, &a, i);
                module_test.vec_znx_big_sub_small_negate_inplace(&mut res_big_test, i, &a, i);
            }

            assert_eq!(a.digest_u64(), a_digest);

            let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            let res_ref_digest: u64 = res_big_ref.digest_u64();
            let res_test_digest: u64 = res_big_test.digest_u64();

            for j in 0..cols {
                module_ref.vec_znx_big_normalize(base2k, &mut res_small_ref, j, base2k, &res_big_ref, j, scratch_ref.borrow());
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
