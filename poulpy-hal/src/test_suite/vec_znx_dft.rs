use super::{TestParams, vec_znx_backend_mut, vec_znx_backend_ref};
use rand::Rng;

use crate::{
    api::{
        ScratchOwnedAlloc, VecZnxBigAlloc, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftAddAssign, VecZnxDftAddInto,
        VecZnxDftAlloc, VecZnxDftApply, VecZnxDftCopy, VecZnxDftSub, VecZnxDftSubInplace, VecZnxDftSubNegateInplace,
        VecZnxIdftApply, VecZnxIdftApplyTmpA, VecZnxIdftApplyTmpBytes,
    },
    layouts::{
        Backend, DataViewMut, DigestU64, FillUniform, Module, ScratchOwned, VecZnx, VecZnxBig, VecZnxBigToBackendMut,
        VecZnxBigToBackendRef, VecZnxDft, VecZnxDftToBackendMut, VecZnxDftToBackendRef,
    },
    source::Source,
};

type VecZnxDftOwned<BE> = VecZnxDft<<BE as Backend>::OwnedBuf, BE>;
type VecZnxBigOwned<BE> = VecZnxBig<<BE as Backend>::OwnedBuf, BE>;

fn idft_into_alloc<BE>(module: &Module<BE>, a: &mut VecZnxDftOwned<BE>) -> VecZnxBigOwned<BE>
where
    BE: Backend,
    Module<BE>: VecZnxBigAlloc<BE> + VecZnxIdftApplyTmpA<BE>,
{
    let cols = a.cols;
    let size = a.size;
    let mut res = module.vec_znx_big_alloc(cols, size);
    for j in 0..cols {
        let mut res_backend = res.to_backend_mut();
        let mut a_backend = a.to_backend_mut();
        module.vec_znx_idft_apply_tmpa(&mut res_backend, j, &mut a_backend, j);
    }
    res
}

pub fn test_vec_znx_dft_add_into<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>: VecZnxDftAddInto<BR>
        + VecZnxDftAlloc<BR>
        + VecZnxDftApply<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxIdftApplyTmpA<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxDftAddInto<BT>
        + VecZnxDftAlloc<BT>
        + VecZnxDftApply<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxIdftApplyTmpA<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
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

        let mut a_dft_ref: VecZnxDftOwned<BR> = module_ref.vec_znx_dft_alloc(cols, a_size);
        let mut a_dft_test: VecZnxDftOwned<BT> = module_test.vec_znx_dft_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_dft_apply(1, 0, &mut a_dft_ref.to_backend_mut(), j, &vec_znx_backend_ref::<BR>(&a), j);
            module_test.vec_znx_dft_apply(1, 0, &mut a_dft_test.to_backend_mut(), j, &vec_znx_backend_ref::<BT>(&a), j);
        }

        assert_eq!(a.digest_u64(), a_digest);

        let a_dft_ref_digest: u64 = a_dft_ref.digest_u64();
        let a_dft_test_digest: u64 = a_dft_test.digest_u64();

        for b_size in [1, 2, 3, 4] {
            let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, b_size);
            b.fill_uniform(base2k, &mut source);
            let b_digest: u64 = b.digest_u64();

            let mut b_dft_ref: VecZnxDftOwned<BR> = module_ref.vec_znx_dft_alloc(cols, b_size);
            let mut b_dft_test: VecZnxDftOwned<BT> = module_test.vec_znx_dft_alloc(cols, b_size);

            for j in 0..cols {
                module_ref.vec_znx_dft_apply(1, 0, &mut b_dft_ref.to_backend_mut(), j, &vec_znx_backend_ref::<BR>(&b), j);
                module_test.vec_znx_dft_apply(1, 0, &mut b_dft_test.to_backend_mut(), j, &vec_znx_backend_ref::<BT>(&b), j);
            }

            assert_eq!(b.digest_u64(), b_digest);

            let b_dft_ref_digest: u64 = b_dft_ref.digest_u64();
            let b_dft_test_digest: u64 = b_dft_test.digest_u64();

            for res_size in [1, 2, 3, 4] {
                let mut res_dft_ref: VecZnxDftOwned<BR> = module_ref.vec_znx_dft_alloc(cols, res_size);
                let mut res_dft_test: VecZnxDftOwned<BT> = module_test.vec_znx_dft_alloc(cols, res_size);

                // Set d to garbage
                source.fill_bytes(res_dft_ref.data_mut().as_mut());
                source.fill_bytes(res_dft_test.data_mut().as_mut());

                // Reference
                for i in 0..cols {
                    module_ref.vec_znx_dft_add_into(
                        &mut res_dft_ref.to_backend_mut(),
                        i,
                        &a_dft_ref.to_backend_ref(),
                        i,
                        &b_dft_ref.to_backend_ref(),
                        i,
                    );
                    module_test.vec_znx_dft_add_into(
                        &mut res_dft_test.to_backend_mut(),
                        i,
                        &a_dft_test.to_backend_ref(),
                        i,
                        &b_dft_test.to_backend_ref(),
                        i,
                    );
                }

                assert_eq!(a_dft_ref.digest_u64(), a_dft_ref_digest);
                assert_eq!(a_dft_test.digest_u64(), a_dft_test_digest);
                assert_eq!(b_dft_ref.digest_u64(), b_dft_ref_digest);
                assert_eq!(b_dft_test.digest_u64(), b_dft_test_digest);

                let res_big_ref = idft_into_alloc(module_ref, &mut res_dft_ref);
                let res_big_test = idft_into_alloc(module_test, &mut res_dft_test);

                let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
                let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

                for j in 0..cols {
                    module_ref.vec_znx_big_normalize(
                        &mut vec_znx_backend_mut::<BR>(&mut res_small_ref),
                        base2k,
                        0,
                        j,
                        &res_big_ref.to_backend_ref(),
                        base2k,
                        j,
                        &mut scratch_ref.arena(),
                    );
                    module_test.vec_znx_big_normalize(
                        &mut vec_znx_backend_mut::<BT>(&mut res_small_test),
                        base2k,
                        0,
                        j,
                        &res_big_test.to_backend_ref(),
                        base2k,
                        j,
                        &mut scratch_test.arena(),
                    );
                }

                assert_eq!(res_small_ref, res_small_test);
            }
        }
    }
}

pub fn test_vec_znx_dft_add_assign<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>: VecZnxDftAddAssign<BR>
        + VecZnxDftAlloc<BR>
        + VecZnxDftApply<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxIdftApplyTmpA<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxDftAddAssign<BT>
        + VecZnxDftAlloc<BT>
        + VecZnxDftApply<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxIdftApplyTmpA<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
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

        let mut a_dft_ref: VecZnxDftOwned<BR> = module_ref.vec_znx_dft_alloc(cols, a_size);
        let mut a_dft_test: VecZnxDftOwned<BT> = module_test.vec_znx_dft_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_dft_apply(1, 0, &mut a_dft_ref.to_backend_mut(), j, &vec_znx_backend_ref::<BR>(&a), j);
            module_test.vec_znx_dft_apply(1, 0, &mut a_dft_test.to_backend_mut(), j, &vec_znx_backend_ref::<BT>(&a), j);
        }

        assert_eq!(a.digest_u64(), a_digest);

        let a_dft_ref_digest: u64 = a_dft_ref.digest_u64();
        let a_dft_test_digest: u64 = a_dft_test.digest_u64();

        for res_size in [1, 2, 3, 4] {
            let mut res: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
            res.fill_uniform(base2k, &mut source);
            let res_digest: u64 = res.digest_u64();

            let mut res_dft_ref: VecZnxDftOwned<BR> = module_ref.vec_znx_dft_alloc(cols, res_size);
            let mut res_dft_test: VecZnxDftOwned<BT> = module_test.vec_znx_dft_alloc(cols, res_size);

            for j in 0..cols {
                module_ref.vec_znx_dft_apply(
                    1,
                    0,
                    &mut res_dft_ref.to_backend_mut(),
                    j,
                    &vec_znx_backend_ref::<BR>(&res),
                    j,
                );
                module_test.vec_znx_dft_apply(
                    1,
                    0,
                    &mut res_dft_test.to_backend_mut(),
                    j,
                    &vec_znx_backend_ref::<BT>(&res),
                    j,
                );
            }

            assert_eq!(res.digest_u64(), res_digest);

            // Reference
            for i in 0..cols {
                module_ref.vec_znx_dft_add_assign(&mut res_dft_ref.to_backend_mut(), i, &a_dft_ref.to_backend_ref(), i);
                module_test.vec_znx_dft_add_assign(&mut res_dft_test.to_backend_mut(), i, &a_dft_test.to_backend_ref(), i);
            }

            assert_eq!(a_dft_ref.digest_u64(), a_dft_ref_digest);
            assert_eq!(a_dft_test.digest_u64(), a_dft_test_digest);

            let res_big_ref = idft_into_alloc(module_ref, &mut res_dft_ref);
            let res_big_test = idft_into_alloc(module_test, &mut res_dft_test);

            let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            for j in 0..cols {
                module_ref.vec_znx_big_normalize(
                    &mut vec_znx_backend_mut::<BR>(&mut res_small_ref),
                    base2k,
                    0,
                    j,
                    &res_big_ref.to_backend_ref(),
                    base2k,
                    j,
                    &mut scratch_ref.arena(),
                );
                module_test.vec_znx_big_normalize(
                    &mut vec_znx_backend_mut::<BT>(&mut res_small_test),
                    base2k,
                    0,
                    j,
                    &res_big_test.to_backend_ref(),
                    base2k,
                    j,
                    &mut scratch_test.arena(),
                );
            }

            assert_eq!(res_small_ref, res_small_test);
        }
    }
}

pub fn test_vec_znx_copy<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>: VecZnxDftCopy<BR>
        + VecZnxDftAlloc<BR>
        + VecZnxDftApply<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxIdftApplyTmpA<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxDftCopy<BT>
        + VecZnxDftAlloc<BT>
        + VecZnxDftApply<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxIdftApplyTmpA<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
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

        let mut a_dft_ref: VecZnxDftOwned<BR> = module_ref.vec_znx_dft_alloc(cols, a_size);
        let mut a_dft_test: VecZnxDftOwned<BT> = module_test.vec_znx_dft_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_dft_apply(1, 0, &mut a_dft_ref.to_backend_mut(), j, &vec_znx_backend_ref::<BR>(&a), j);
            module_test.vec_znx_dft_apply(1, 0, &mut a_dft_test.to_backend_mut(), j, &vec_znx_backend_ref::<BT>(&a), j);
        }

        assert_eq!(a.digest_u64(), a_digest);

        let a_dft_ref_digest: u64 = a_dft_ref.digest_u64();
        let a_dft_test_digest: u64 = a_dft_test.digest_u64();

        for res_size in [1, 2, 6, 11] {
            for params in [[1, 0], [1, 1], [1, 2], [2, 2]] {
                let steps: usize = params[0];
                let offset: usize = params[1];

                println!("steps: {} offset: {}", steps, offset);

                let mut res_dft_ref: VecZnxDftOwned<BR> = module_ref.vec_znx_dft_alloc(cols, res_size);
                let mut res_dft_test: VecZnxDftOwned<BT> = module_test.vec_znx_dft_alloc(cols, res_size);

                // Set d to garbage
                source.fill_bytes(res_dft_ref.data_mut().as_mut());
                source.fill_bytes(res_dft_test.data_mut().as_mut());

                // Reference
                for i in 0..cols {
                    module_ref.vec_znx_dft_copy(
                        steps,
                        offset,
                        &mut res_dft_ref.to_backend_mut(),
                        i,
                        &a_dft_ref.to_backend_ref(),
                        i,
                    );
                    module_test.vec_znx_dft_copy(
                        steps,
                        offset,
                        &mut res_dft_test.to_backend_mut(),
                        i,
                        &a_dft_test.to_backend_ref(),
                        i,
                    );
                }

                assert_eq!(a_dft_ref.digest_u64(), a_dft_ref_digest);
                assert_eq!(a_dft_test.digest_u64(), a_dft_test_digest);

                let res_big_ref = idft_into_alloc(module_ref, &mut res_dft_ref);
                let res_big_test = idft_into_alloc(module_test, &mut res_dft_test);

                let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
                let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

                for j in 0..cols {
                    module_ref.vec_znx_big_normalize(
                        &mut vec_znx_backend_mut::<BR>(&mut res_small_ref),
                        base2k,
                        0,
                        j,
                        &res_big_ref.to_backend_ref(),
                        base2k,
                        j,
                        &mut scratch_ref.arena(),
                    );
                    module_test.vec_znx_big_normalize(
                        &mut vec_znx_backend_mut::<BT>(&mut res_small_test),
                        base2k,
                        0,
                        j,
                        &res_big_test.to_backend_ref(),
                        base2k,
                        j,
                        &mut scratch_test.arena(),
                    );
                }

                assert_eq!(res_small_ref, res_small_test);
            }
        }
    }
}

pub fn test_vec_znx_idft_apply<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
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
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
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

                let mut res_dft_ref: VecZnxDftOwned<BR> = module_ref.vec_znx_dft_alloc(cols, a_size);
                let mut res_dft_test: VecZnxDftOwned<BT> = module_test.vec_znx_dft_alloc(cols, a_size);

                for j in 0..cols {
                    module_ref.vec_znx_dft_apply(
                        steps,
                        offset,
                        &mut res_dft_ref.to_backend_mut(),
                        j,
                        &vec_znx_backend_ref::<BR>(&a),
                        j,
                    );
                    module_test.vec_znx_dft_apply(
                        steps,
                        offset,
                        &mut res_dft_test.to_backend_mut(),
                        j,
                        &vec_znx_backend_ref::<BT>(&a),
                        j,
                    );
                }

                assert_eq!(a.digest_u64(), a_digest);
                let res_dft_ref_digest: u64 = res_dft_ref.digest_u64();
                let rest_dft_test_digest: u64 = res_dft_test.digest_u64();

                let mut res_big_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, res_size);
                let mut res_big_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, res_size);

                for j in 0..cols {
                    module_ref.vec_znx_idft_apply(
                        &mut res_big_ref.to_backend_mut(),
                        j,
                        &res_dft_ref.to_backend_ref(),
                        j,
                        &mut scratch_ref.arena(),
                    );
                    module_test.vec_znx_idft_apply(
                        &mut res_big_test.to_backend_mut(),
                        j,
                        &res_dft_test.to_backend_ref(),
                        j,
                        &mut scratch_test.arena(),
                    );
                }

                assert_eq!(res_dft_ref.digest_u64(), res_dft_ref_digest);
                assert_eq!(res_dft_test.digest_u64(), rest_dft_test_digest);

                let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
                let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

                for j in 0..cols {
                    module_ref.vec_znx_big_normalize(
                        &mut vec_znx_backend_mut::<BR>(&mut res_small_ref),
                        base2k,
                        0,
                        j,
                        &res_big_ref.to_backend_ref(),
                        base2k,
                        j,
                        &mut scratch_ref.arena(),
                    );
                    module_test.vec_znx_big_normalize(
                        &mut vec_znx_backend_mut::<BT>(&mut res_small_test),
                        base2k,
                        0,
                        j,
                        &res_big_test.to_backend_ref(),
                        base2k,
                        j,
                        &mut scratch_test.arena(),
                    );
                }

                assert_eq!(res_small_ref, res_small_test);
            }
        }
    }
}

pub fn test_vec_znx_idft_apply_tmpa<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
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
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
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

                let mut res_dft_ref: VecZnxDftOwned<BR> = module_ref.vec_znx_dft_alloc(cols, a_size);
                let mut res_dft_test: VecZnxDftOwned<BT> = module_test.vec_znx_dft_alloc(cols, a_size);

                for j in 0..cols {
                    module_ref.vec_znx_dft_apply(
                        steps,
                        offset,
                        &mut res_dft_ref.to_backend_mut(),
                        j,
                        &vec_znx_backend_ref::<BR>(&a),
                        j,
                    );
                    module_test.vec_znx_dft_apply(
                        steps,
                        offset,
                        &mut res_dft_test.to_backend_mut(),
                        j,
                        &vec_znx_backend_ref::<BT>(&a),
                        j,
                    );
                }

                assert_eq!(a.digest_u64(), a_digest);

                let mut res_big_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, res_size);
                let mut res_big_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, res_size);

                for j in 0..cols {
                    module_ref.vec_znx_idft_apply_tmpa(
                        &mut res_big_ref.to_backend_mut(),
                        j,
                        &mut res_dft_ref.to_backend_mut(),
                        j,
                    );
                    module_test.vec_znx_idft_apply_tmpa(
                        &mut res_big_test.to_backend_mut(),
                        j,
                        &mut res_dft_test.to_backend_mut(),
                        j,
                    );
                }

                let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
                let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

                for j in 0..cols {
                    module_ref.vec_znx_big_normalize(
                        &mut vec_znx_backend_mut::<BR>(&mut res_small_ref),
                        base2k,
                        0,
                        j,
                        &res_big_ref.to_backend_ref(),
                        base2k,
                        j,
                        &mut scratch_ref.arena(),
                    );
                    module_test.vec_znx_big_normalize(
                        &mut vec_znx_backend_mut::<BT>(&mut res_small_test),
                        base2k,
                        0,
                        j,
                        &res_big_test.to_backend_ref(),
                        base2k,
                        j,
                        &mut scratch_test.arena(),
                    );
                }

                assert_eq!(res_small_ref, res_small_test);
            }
        }
    }
}

pub fn test_vec_znx_idft_apply_alloc<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>: VecZnxDftApply<BR>
        + VecZnxIdftApplyTmpBytes
        + VecZnxDftAlloc<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxBigAlloc<BR>
        + VecZnxIdftApplyTmpA<BR>,
    Module<BT>: VecZnxDftApply<BT>
        + VecZnxIdftApplyTmpBytes
        + VecZnxDftAlloc<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxBigAlloc<BT>
        + VecZnxIdftApplyTmpA<BT>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
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

                let mut res_dft_ref: VecZnxDftOwned<BR> = module_ref.vec_znx_dft_alloc(cols, a_size);
                let mut res_dft_test: VecZnxDftOwned<BT> = module_test.vec_znx_dft_alloc(cols, a_size);

                for j in 0..cols {
                    module_ref.vec_znx_dft_apply(
                        steps,
                        offset,
                        &mut res_dft_ref.to_backend_mut(),
                        j,
                        &vec_znx_backend_ref::<BR>(&a),
                        j,
                    );
                    module_test.vec_znx_dft_apply(
                        steps,
                        offset,
                        &mut res_dft_test.to_backend_mut(),
                        j,
                        &vec_znx_backend_ref::<BT>(&a),
                        j,
                    );
                }

                assert_eq!(a.digest_u64(), a_digest);

                let res_big_ref = idft_into_alloc(module_ref, &mut res_dft_ref);
                let res_big_test = idft_into_alloc(module_test, &mut res_dft_test);

                let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
                let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

                for j in 0..cols {
                    module_ref.vec_znx_big_normalize(
                        &mut vec_znx_backend_mut::<BR>(&mut res_small_ref),
                        base2k,
                        0,
                        j,
                        &res_big_ref.to_backend_ref(),
                        base2k,
                        j,
                        &mut scratch_ref.arena(),
                    );
                    module_test.vec_znx_big_normalize(
                        &mut vec_znx_backend_mut::<BT>(&mut res_small_test),
                        base2k,
                        0,
                        j,
                        &res_big_test.to_backend_ref(),
                        base2k,
                        j,
                        &mut scratch_test.arena(),
                    );
                }

                assert_eq!(res_small_ref, res_small_test);
            }
        }
    }
}

pub fn test_vec_znx_dft_sub<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>: VecZnxDftSub<BR>
        + VecZnxDftAlloc<BR>
        + VecZnxDftApply<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxIdftApplyTmpA<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxDftSub<BT>
        + VecZnxDftAlloc<BT>
        + VecZnxDftApply<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxIdftApplyTmpA<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
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

        let mut a_dft_ref: VecZnxDftOwned<BR> = module_ref.vec_znx_dft_alloc(cols, a_size);
        let mut a_dft_test: VecZnxDftOwned<BT> = module_test.vec_znx_dft_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_dft_apply(1, 0, &mut a_dft_ref.to_backend_mut(), j, &vec_znx_backend_ref::<BR>(&a), j);
            module_test.vec_znx_dft_apply(1, 0, &mut a_dft_test.to_backend_mut(), j, &vec_znx_backend_ref::<BT>(&a), j);
        }

        assert_eq!(a.digest_u64(), a_digest);

        let a_dft_ref_digest: u64 = a_dft_ref.digest_u64();
        let a_dft_test_digest: u64 = a_dft_test.digest_u64();

        for b_size in [1, 2, 3, 4] {
            let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, b_size);
            b.fill_uniform(base2k, &mut source);
            let b_digest: u64 = b.digest_u64();

            let mut b_dft_ref: VecZnxDftOwned<BR> = module_ref.vec_znx_dft_alloc(cols, b_size);
            let mut b_dft_test: VecZnxDftOwned<BT> = module_test.vec_znx_dft_alloc(cols, b_size);

            for j in 0..cols {
                module_ref.vec_znx_dft_apply(1, 0, &mut b_dft_ref.to_backend_mut(), j, &vec_znx_backend_ref::<BR>(&b), j);
                module_test.vec_znx_dft_apply(1, 0, &mut b_dft_test.to_backend_mut(), j, &vec_znx_backend_ref::<BT>(&b), j);
            }

            assert_eq!(b.digest_u64(), b_digest);

            let b_dft_ref_digest: u64 = b_dft_ref.digest_u64();
            let b_dft_test_digest: u64 = b_dft_test.digest_u64();

            for res_size in [1, 2, 3, 4] {
                let mut res_dft_ref: VecZnxDftOwned<BR> = module_ref.vec_znx_dft_alloc(cols, res_size);
                let mut res_dft_test: VecZnxDftOwned<BT> = module_test.vec_znx_dft_alloc(cols, res_size);

                // Set d to garbage
                source.fill_bytes(res_dft_ref.data_mut().as_mut());
                source.fill_bytes(res_dft_test.data_mut().as_mut());

                // Reference
                for i in 0..cols {
                    module_ref.vec_znx_dft_sub(
                        &mut res_dft_ref.to_backend_mut(),
                        i,
                        &a_dft_ref.to_backend_ref(),
                        i,
                        &b_dft_ref.to_backend_ref(),
                        i,
                    );
                    module_test.vec_znx_dft_sub(
                        &mut res_dft_test.to_backend_mut(),
                        i,
                        &a_dft_test.to_backend_ref(),
                        i,
                        &b_dft_test.to_backend_ref(),
                        i,
                    );
                }

                assert_eq!(a_dft_ref.digest_u64(), a_dft_ref_digest);
                assert_eq!(a_dft_test.digest_u64(), a_dft_test_digest);
                assert_eq!(b_dft_ref.digest_u64(), b_dft_ref_digest);
                assert_eq!(b_dft_test.digest_u64(), b_dft_test_digest);

                let res_big_ref = idft_into_alloc(module_ref, &mut res_dft_ref);
                let res_big_test = idft_into_alloc(module_test, &mut res_dft_test);

                let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
                let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

                for j in 0..cols {
                    module_ref.vec_znx_big_normalize(
                        &mut vec_znx_backend_mut::<BR>(&mut res_small_ref),
                        base2k,
                        0,
                        j,
                        &res_big_ref.to_backend_ref(),
                        base2k,
                        j,
                        &mut scratch_ref.arena(),
                    );
                    module_test.vec_znx_big_normalize(
                        &mut vec_znx_backend_mut::<BT>(&mut res_small_test),
                        base2k,
                        0,
                        j,
                        &res_big_test.to_backend_ref(),
                        base2k,
                        j,
                        &mut scratch_test.arena(),
                    );
                }

                assert_eq!(res_small_ref, res_small_test);
            }
        }
    }
}

pub fn test_vec_znx_dft_sub_inplace<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>: VecZnxDftSubInplace<BR>
        + VecZnxDftAlloc<BR>
        + VecZnxDftApply<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxIdftApplyTmpA<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxDftSubAssign<BT>
        + VecZnxDftAlloc<BT>
        + VecZnxDftApply<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxIdftApplyTmpA<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
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

        let mut a_dft_ref: VecZnxDftOwned<BR> = module_ref.vec_znx_dft_alloc(cols, a_size);
        let mut a_dft_test: VecZnxDftOwned<BT> = module_test.vec_znx_dft_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_dft_apply(1, 0, &mut a_dft_ref.to_backend_mut(), j, &vec_znx_backend_ref::<BR>(&a), j);
            module_test.vec_znx_dft_apply(1, 0, &mut a_dft_test.to_backend_mut(), j, &vec_znx_backend_ref::<BT>(&a), j);
        }

        assert_eq!(a.digest_u64(), a_digest);

        let a_dft_ref_digest: u64 = a_dft_ref.digest_u64();
        let a_dft_test_digest: u64 = a_dft_test.digest_u64();

        for res_size in [1, 2, 3, 4] {
            let mut res: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
            res.fill_uniform(base2k, &mut source);
            let res_digest: u64 = res.digest_u64();

            let mut res_dft_ref: VecZnxDftOwned<BR> = module_ref.vec_znx_dft_alloc(cols, res_size);
            let mut res_dft_test: VecZnxDftOwned<BT> = module_test.vec_znx_dft_alloc(cols, res_size);

            for j in 0..cols {
                module_ref.vec_znx_dft_apply(
                    1,
                    0,
                    &mut res_dft_ref.to_backend_mut(),
                    j,
                    &vec_znx_backend_ref::<BR>(&res),
                    j,
                );
                module_test.vec_znx_dft_apply(
                    1,
                    0,
                    &mut res_dft_test.to_backend_mut(),
                    j,
                    &vec_znx_backend_ref::<BT>(&res),
                    j,
                );
            }

            assert_eq!(res.digest_u64(), res_digest);

            // Reference
            for i in 0..cols {
                module_ref.vec_znx_dft_sub_inplace(&mut res_dft_ref.to_backend_mut(), i, &a_dft_ref.to_backend_ref(), i);
                module_test.vec_znx_dft_sub_inplace(&mut res_dft_test.to_backend_mut(), i, &a_dft_test.to_backend_ref(), i);
            }

            assert_eq!(a_dft_ref.digest_u64(), a_dft_ref_digest);
            assert_eq!(a_dft_test.digest_u64(), a_dft_test_digest);

            let res_big_ref = idft_into_alloc(module_ref, &mut res_dft_ref);
            let res_big_test = idft_into_alloc(module_test, &mut res_dft_test);

            let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            for j in 0..cols {
                module_ref.vec_znx_big_normalize(
                    &mut vec_znx_backend_mut::<BR>(&mut res_small_ref),
                    base2k,
                    0,
                    j,
                    &res_big_ref.to_backend_ref(),
                    base2k,
                    j,
                    &mut scratch_ref.arena(),
                );
                module_test.vec_znx_big_normalize(
                    &mut vec_znx_backend_mut::<BT>(&mut res_small_test),
                    base2k,
                    0,
                    j,
                    &res_big_test.to_backend_ref(),
                    base2k,
                    j,
                    &mut scratch_test.arena(),
                );
            }

            assert_eq!(res_small_ref, res_small_test);
        }
    }
}

pub fn test_vec_znx_dft_sub_negate_inplace<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>: VecZnxDftSubNegateInplace<BR>
        + VecZnxDftAlloc<BR>
        + VecZnxDftApply<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxIdftApplyTmpA<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxDftSubNegateAssign<BT>
        + VecZnxDftAlloc<BT>
        + VecZnxDftApply<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxIdftApplyTmpA<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
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

        let mut a_dft_ref: VecZnxDftOwned<BR> = module_ref.vec_znx_dft_alloc(cols, a_size);
        let mut a_dft_test: VecZnxDftOwned<BT> = module_test.vec_znx_dft_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_dft_apply(1, 0, &mut a_dft_ref.to_backend_mut(), j, &vec_znx_backend_ref::<BR>(&a), j);
            module_test.vec_znx_dft_apply(1, 0, &mut a_dft_test.to_backend_mut(), j, &vec_znx_backend_ref::<BT>(&a), j);
        }

        assert_eq!(a.digest_u64(), a_digest);

        let a_dft_ref_digest: u64 = a_dft_ref.digest_u64();
        let a_dft_test_digest: u64 = a_dft_test.digest_u64();

        for res_size in [1, 2, 3, 4] {
            let mut res: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
            res.fill_uniform(base2k, &mut source);
            let res_digest: u64 = res.digest_u64();

            let mut res_dft_ref: VecZnxDftOwned<BR> = module_ref.vec_znx_dft_alloc(cols, res_size);
            let mut res_dft_test: VecZnxDftOwned<BT> = module_test.vec_znx_dft_alloc(cols, res_size);

            for j in 0..cols {
                module_ref.vec_znx_dft_apply(
                    1,
                    0,
                    &mut res_dft_ref.to_backend_mut(),
                    j,
                    &vec_znx_backend_ref::<BR>(&res),
                    j,
                );
                module_test.vec_znx_dft_apply(
                    1,
                    0,
                    &mut res_dft_test.to_backend_mut(),
                    j,
                    &vec_znx_backend_ref::<BT>(&res),
                    j,
                );
            }

            assert_eq!(res.digest_u64(), res_digest);

            // Reference
            for i in 0..cols {
                module_ref.vec_znx_dft_sub_negate_inplace(&mut res_dft_ref.to_backend_mut(), i, &a_dft_ref.to_backend_ref(), i);
                module_test.vec_znx_dft_sub_negate_inplace(
                    &mut res_dft_test.to_backend_mut(),
                    i,
                    &a_dft_test.to_backend_ref(),
                    i,
                );
            }

            assert_eq!(a_dft_ref.digest_u64(), a_dft_ref_digest);
            assert_eq!(a_dft_test.digest_u64(), a_dft_test_digest);

            let res_big_ref = idft_into_alloc(module_ref, &mut res_dft_ref);
            let res_big_test = idft_into_alloc(module_test, &mut res_dft_test);

            let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            for j in 0..cols {
                module_ref.vec_znx_big_normalize(
                    &mut vec_znx_backend_mut::<BR>(&mut res_small_ref),
                    base2k,
                    0,
                    j,
                    &res_big_ref.to_backend_ref(),
                    base2k,
                    j,
                    &mut scratch_ref.arena(),
                );
                module_test.vec_znx_big_normalize(
                    &mut vec_znx_backend_mut::<BT>(&mut res_small_test),
                    base2k,
                    0,
                    j,
                    &res_big_test.to_backend_ref(),
                    base2k,
                    j,
                    &mut scratch_test.arena(),
                );
            }

            assert_eq!(res_small_ref, res_small_test);
        }
    }
}
