use super::{TestParams, vec_znx_backend_mut, vec_znx_backend_ref};
use rand::Rng;

use crate::{
    api::{
        ScratchOwnedAlloc, VecZnxBigAddAssign, VecZnxBigAddInto, VecZnxBigAddNormal, VecZnxBigAddNormalBackend,
        VecZnxBigAddSmallAssign, VecZnxBigAddSmallInto, VecZnxBigAlloc, VecZnxBigAutomorphism, VecZnxBigAutomorphismInplace,
        VecZnxBigAutomorphismInplaceTmpBytes, VecZnxBigFromSmall, VecZnxBigNegate, VecZnxBigNegateInplace, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxBigSub, VecZnxBigSubInplace, VecZnxBigSubNegateInplace, VecZnxBigSubSmallA,
        VecZnxBigSubSmallB, VecZnxBigSubSmallInplace, VecZnxBigSubSmallNegateInplace,
    },
    layouts::{
        Backend, DataViewMut, DigestU64, FillUniform, Module, NoiseInfos, ScratchOwned, VecZnx, VecZnxBig, VecZnxBigToBackendMut,
        VecZnxBigToBackendRef,
    },
    source::Source,
};

type VecZnxBigOwned<BE> = VecZnxBig<<BE as Backend>::OwnedBuf, BE>;

pub fn test_vec_znx_big_seed_add_normal_matches_source_wrapper<
    BR: crate::test_suite::TestBackend,
    BT: crate::test_suite::TestBackend,
>(
    params: &TestParams,
    _module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BT>: VecZnxBigAddNormal<BT> + VecZnxBigAddNormalBackend<BT> + VecZnxBigAlloc<BT>,
{
    let base2k = params.base2k;
    let size: usize = 5;
    let cols: usize = 2;
    let col_i: usize = 1;
    let noise_infos = NoiseInfos::new(2 * base2k, 3.2, 6.0 * 3.2).unwrap();

    let mut seed_source = Source::new([2u8; 32]);
    let seed = seed_source.new_seed();
    let mut wrapper_source = Source::new([2u8; 32]);

    let mut wrapper: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, size);
    let mut backend: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, size);
    module_test.vec_znx_big_add_normal(base2k, &mut wrapper.to_backend_mut(), col_i, noise_infos, &mut wrapper_source);
    module_test.vec_znx_big_add_normal_backend(base2k, &mut backend.to_backend_mut(), col_i, noise_infos, seed);
    assert_eq!(wrapper.digest_u64(), backend.digest_u64());
}

pub fn test_vec_znx_big_add_into<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>:
        VecZnxBigAddInto<BR> + VecZnxBigAlloc<BR> + VecZnxBigFromSmall<BR> + VecZnxBigNormalize<BR> + VecZnxBigNormalizeTmpBytes,
    Module<BT>:
        VecZnxBigAddInto<BT> + VecZnxBigAlloc<BT> + VecZnxBigFromSmall<BT> + VecZnxBigNormalize<BT> + VecZnxBigNormalizeTmpBytes,
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

        let mut a_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, a_size);
        let mut a_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_big_from_small(&mut a_ref.to_backend_mut(), j, &a, j);
            module_test.vec_znx_big_from_small(&mut a_test.to_backend_mut(), j, &a, j);
        }

        assert_eq!(a.digest_u64(), a_digest);

        let a_ref_digest: u64 = a_ref.digest_u64();
        let a_test_digest: u64 = a_test.digest_u64();

        for b_size in [1, 2, 3, 4] {
            let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, b_size);
            b.fill_uniform(base2k, &mut source);
            let b_digest = b.digest_u64();

            let mut b_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, b_size);
            let mut b_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, b_size);

            for j in 0..cols {
                module_ref.vec_znx_big_from_small(&mut b_ref.to_backend_mut(), j, &b, j);
                module_test.vec_znx_big_from_small(&mut b_test.to_backend_mut(), j, &b, j);
            }

            assert_eq!(b.digest_u64(), b_digest);

            let b_ref_digest: u64 = b_ref.digest_u64();
            let b_test_digest: u64 = b_test.digest_u64();

            for res_size in [1, 2, 3, 4] {
                let mut res_big_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, res_size);
                let mut res_big_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, res_size);

                // Set res to garbage
                source.fill_bytes(res_big_ref.data_mut().as_mut());
                source.fill_bytes(res_big_test.data_mut().as_mut());

                // Reference
                for i in 0..cols {
                    module_ref.vec_znx_big_add_into(
                        &mut res_big_ref.to_backend_mut(),
                        i,
                        &a_ref.to_backend_ref(),
                        i,
                        &b_ref.to_backend_ref(),
                        i,
                    );
                    module_test.vec_znx_big_add_into(
                        &mut res_big_test.to_backend_mut(),
                        i,
                        &a_test.to_backend_ref(),
                        i,
                        &b_test.to_backend_ref(),
                        i,
                    );
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

                assert_eq!(res_big_ref.digest_u64(), res_ref_digest);
                assert_eq!(res_big_test.digest_u64(), res_test_digest);

                assert_eq!(res_small_ref, res_small_test);
            }
        }
    }
}

pub fn test_vec_znx_big_add_assign<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>: VecZnxBigAddAssign<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmall<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigAddAssign<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmall<BT>
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

        let mut a_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, a_size);
        let mut a_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_big_from_small(&mut a_ref.to_backend_mut(), j, &a, j);
            module_test.vec_znx_big_from_small(&mut a_test.to_backend_mut(), j, &a, j);
        }

        let a_ref_digest: u64 = a_ref.digest_u64();
        let a_test_digest: u64 = a_test.digest_u64();

        for res_size in [1, 2, 3, 4] {
            let mut res: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            res.fill_uniform(base2k, &mut source);

            let mut res_big_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, res_size);
            let mut res_big_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, res_size);

            for j in 0..cols {
                module_ref.vec_znx_big_from_small(&mut res_big_ref.to_backend_mut(), j, &res, j);
                module_test.vec_znx_big_from_small(&mut res_big_test.to_backend_mut(), j, &res, j);
            }

            for i in 0..cols {
                module_ref.vec_znx_big_add_assign(&mut res_big_ref.to_backend_mut(), i, &a_ref.to_backend_ref(), i);
                module_test.vec_znx_big_add_assign(&mut res_big_test.to_backend_mut(), i, &a_test.to_backend_ref(), i);
            }

            assert_eq!(a_ref.digest_u64(), a_ref_digest);
            assert_eq!(a_test.digest_u64(), a_test_digest);

            let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            let res_ref_digest: u64 = res_big_ref.digest_u64();
            let res_test_digest: u64 = res_big_test.digest_u64();

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

            assert_eq!(res_big_ref.digest_u64(), res_ref_digest);
            assert_eq!(res_big_test.digest_u64(), res_test_digest);

            assert_eq!(res_small_ref, res_small_test);
        }
    }
}

pub fn test_vec_znx_big_add_small_into<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>: VecZnxBigAddSmallInto<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmall<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigAddSmallInto<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmall<BT>
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

        let mut a_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, a_size);
        let mut a_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_big_from_small(&mut a_ref.to_backend_mut(), j, &a, j);
            module_test.vec_znx_big_from_small(&mut a_test.to_backend_mut(), j, &a, j);
        }

        let a_ref_digest: u64 = a_ref.digest_u64();
        let a_test_digest: u64 = a_test.digest_u64();

        for b_size in [1, 2, 3, 4] {
            let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, b_size);
            b.fill_uniform(base2k, &mut source);
            let b_digest: u64 = b.digest_u64();

            for res_size in [1, 2, 3, 4] {
                let mut res_big_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, res_size);
                let mut res_big_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, res_size);

                // Set res to garbage
                source.fill_bytes(res_big_ref.data_mut().as_mut());
                source.fill_bytes(res_big_test.data_mut().as_mut());

                // Reference
                for i in 0..cols {
                    module_ref.vec_znx_big_add_small_into(
                        &mut res_big_ref.to_backend_mut(),
                        i,
                        &a_ref.to_backend_ref(),
                        i,
                        &b,
                        i,
                    );
                    module_test.vec_znx_big_add_small_into(
                        &mut res_big_test.to_backend_mut(),
                        i,
                        &a_test.to_backend_ref(),
                        i,
                        &b,
                        i,
                    );
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

                assert_eq!(res_big_ref.digest_u64(), res_ref_digest);
                assert_eq!(res_big_test.digest_u64(), res_test_digest);

                assert_eq!(res_small_ref, res_small_test);
            }
        }
    }
}

pub fn test_vec_znx_big_add_small_assign<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>: VecZnxBigAddSmallAssign<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmall<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigAddSmallAssign<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmall<BT>
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

        let a_digest: u64 = a.digest_u64();

        for res_size in [1, 2, 3, 4] {
            let mut res: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            res.fill_uniform(base2k, &mut source);

            let mut res_big_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, res_size);
            let mut res_big_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, res_size);

            for j in 0..cols {
                module_ref.vec_znx_big_from_small(&mut res_big_ref.to_backend_mut(), j, &res, j);
                module_test.vec_znx_big_from_small(&mut res_big_test.to_backend_mut(), j, &res, j);
            }

            for i in 0..cols {
                module_ref.vec_znx_big_add_small_assign(&mut res_big_ref.to_backend_mut(), i, &vec_znx_backend_ref::<BR>(&a), i);
                module_test.vec_znx_big_add_small_assign(
                    &mut res_big_test.to_backend_mut(),
                    i,
                    &vec_znx_backend_ref::<BT>(&a),
                    i,
                );
            }

            assert_eq!(a.digest_u64(), a_digest);

            let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            let res_ref_digest: u64 = res_big_ref.digest_u64();
            let res_test_digest: u64 = res_big_test.digest_u64();

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

            assert_eq!(res_big_ref.digest_u64(), res_ref_digest);
            assert_eq!(res_big_test.digest_u64(), res_test_digest);

            assert_eq!(res_small_ref, res_small_test);
        }
    }
}

pub fn test_vec_znx_big_automorphism<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
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

        let mut a_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, a_size);
        let mut a_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_big_from_small(&mut a_ref.to_backend_mut(), j, &a, j);
            module_test.vec_znx_big_from_small(&mut a_test.to_backend_mut(), j, &a, j);
        }

        let a_ref_digest: u64 = a_ref.digest_u64();
        let a_test_digest: u64 = a_test.digest_u64();

        for res_size in [1, 2, 3, 4] {
            for p in [-5, 5] {
                let mut res_big_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, res_size);
                let mut res_big_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, res_size);

                // Set res to garbage
                source.fill_bytes(res_big_ref.data_mut().as_mut());
                source.fill_bytes(res_big_test.data_mut().as_mut());

                // Reference
                for i in 0..cols {
                    module_ref.vec_znx_big_automorphism(p, &mut res_big_ref.to_backend_mut(), i, &a_ref.to_backend_ref(), i);
                    module_test.vec_znx_big_automorphism(p, &mut res_big_test.to_backend_mut(), i, &a_test.to_backend_ref(), i);
                }

                assert_eq!(a_ref.digest_u64(), a_ref_digest);
                assert_eq!(a_test.digest_u64(), a_test_digest);

                let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
                let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

                let res_ref_digest: u64 = res_big_ref.digest_u64();
                let res_test_digest: u64 = res_big_test.digest_u64();

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

                assert_eq!(res_big_ref.digest_u64(), res_ref_digest);
                assert_eq!(res_big_test.digest_u64(), res_test_digest);

                assert_eq!(res_small_ref, res_small_test);
            }
        }
    }
}

pub fn test_vec_znx_big_automorphism_inplace<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>: VecZnxBigAutomorphismInplace<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmall<BR>
        + VecZnxBigAutomorphismAssignTmpBytes
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigAutomorphismAssign<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmall<BT>
        + VecZnxBigAutomorphismAssignTmpBytes
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

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(
        module_ref.vec_znx_big_automorphism_assign_tmp_bytes() | module_ref.vec_znx_big_normalize_tmp_bytes(),
    );
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(
        module_test.vec_znx_big_automorphism_assign_tmp_bytes() | module_test.vec_znx_big_normalize_tmp_bytes(),
    );

    for res_size in [1, 2, 3, 4] {
        let mut res: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
        res.fill_uniform(base2k, &mut source);

        let mut res_big_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, res_size);
        let mut res_big_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, res_size);

        for p in [-5, 5] {
            for j in 0..cols {
                module_ref.vec_znx_big_from_small(&mut res_big_ref.to_backend_mut(), j, &res, j);
                module_test.vec_znx_big_from_small(&mut res_big_test.to_backend_mut(), j, &res, j);
            }

            for i in 0..cols {
                module_ref.vec_znx_big_automorphism_inplace(p, &mut res_big_ref.to_backend_mut(), i, &mut scratch_ref.arena());
                module_test.vec_znx_big_automorphism_inplace(p, &mut res_big_test.to_backend_mut(), i, &mut scratch_test.arena());
            }

            let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            let res_ref_digest: u64 = res_big_ref.digest_u64();
            let res_test_digest: u64 = res_big_test.digest_u64();

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

            assert_eq!(res_big_ref.digest_u64(), res_ref_digest);
            assert_eq!(res_big_test.digest_u64(), res_test_digest);

            assert_eq!(res_small_ref, res_small_test);
        }
    }
}

pub fn test_vec_znx_big_negate<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>:
        VecZnxBigNegate<BR> + VecZnxBigAlloc<BR> + VecZnxBigFromSmall<BR> + VecZnxBigNormalize<BR> + VecZnxBigNormalizeTmpBytes,
    Module<BT>:
        VecZnxBigNegate<BT> + VecZnxBigAlloc<BT> + VecZnxBigFromSmall<BT> + VecZnxBigNormalize<BT> + VecZnxBigNormalizeTmpBytes,
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

        let mut a_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, a_size);
        let mut a_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_big_from_small(&mut a_ref.to_backend_mut(), j, &a, j);
            module_test.vec_znx_big_from_small(&mut a_test.to_backend_mut(), j, &a, j);
        }

        let a_ref_digest: u64 = a_ref.digest_u64();
        let a_test_digest: u64 = a_test.digest_u64();

        for res_size in [1, 2, 3, 4] {
            let mut res_big_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, res_size);
            let mut res_big_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, res_size);

            // Set res to garbage
            source.fill_bytes(res_big_ref.data_mut().as_mut());
            source.fill_bytes(res_big_test.data_mut().as_mut());

            // Reference
            for i in 0..cols {
                module_ref.vec_znx_big_negate(&mut res_big_ref.to_backend_mut(), i, &a_ref.to_backend_ref(), i);
                module_test.vec_znx_big_negate(&mut res_big_test.to_backend_mut(), i, &a_test.to_backend_ref(), i);
            }

            assert_eq!(a_ref.digest_u64(), a_ref_digest);
            assert_eq!(a_test.digest_u64(), a_test_digest);

            let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            let res_ref_digest: u64 = res_big_ref.digest_u64();
            let res_test_digest: u64 = res_big_test.digest_u64();

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

            assert_eq!(res_big_ref.digest_u64(), res_ref_digest);
            assert_eq!(res_big_test.digest_u64(), res_test_digest);

            assert_eq!(res_small_ref, res_small_test);
        }
    }
}

pub fn test_vec_znx_big_negate_inplace<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>: VecZnxBigNegateInplace<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmall<BR>
        + VecZnxBigAutomorphismAssignTmpBytes
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigNegateAssign<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmall<BT>
        + VecZnxBigAutomorphismAssignTmpBytes
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

    for res_size in [1, 2, 3, 4] {
        let mut res: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
        res.fill_uniform(base2k, &mut source);

        let mut res_big_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, res_size);
        let mut res_big_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, res_size);

        for j in 0..cols {
            module_ref.vec_znx_big_from_small(&mut res_big_ref.to_backend_mut(), j, &res, j);
            module_test.vec_znx_big_from_small(&mut res_big_test.to_backend_mut(), j, &res, j);
        }

        for i in 0..cols {
            module_ref.vec_znx_big_negate_inplace(&mut res_big_ref.to_backend_mut(), i);
            module_test.vec_znx_big_negate_inplace(&mut res_big_test.to_backend_mut(), i);
        }

        let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
        let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

        let res_ref_digest: u64 = res_big_ref.digest_u64();
        let res_test_digest: u64 = res_big_test.digest_u64();

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

        assert_eq!(res_big_ref.digest_u64(), res_ref_digest);
        assert_eq!(res_big_test.digest_u64(), res_test_digest);

        assert_eq!(res_small_ref, res_small_test);
    }
}

pub fn test_vec_znx_big_normalize<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>: VecZnxBigAlloc<BR>
        + VecZnxBigFromSmall<BR>
        + VecZnxBigAutomorphismAssignTmpBytes
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigAlloc<BT>
        + VecZnxBigFromSmall<BT>
        + VecZnxBigAutomorphismAssignTmpBytes
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

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(
        module_ref.vec_znx_big_automorphism_assign_tmp_bytes() | module_ref.vec_znx_big_normalize_tmp_bytes(),
    );
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(
        module_test.vec_znx_big_automorphism_assign_tmp_bytes() | module_test.vec_znx_big_normalize_tmp_bytes(),
    );

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(63, &mut source);

        let mut a_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, a_size);
        let mut a_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_big_from_small(&mut a_ref.to_backend_mut(), j, &a, j);
            module_test.vec_znx_big_from_small(&mut a_test.to_backend_mut(), j, &a, j);
        }

        let a_ref_digest: u64 = a_ref.digest_u64();
        let a_test_digest: u64 = a_test.digest_u64();

        for res_size in [1, 2, 3, 4] {
            let mut res_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            for res_offset in -(base2k as i64)..=(base2k as i64) {
                // Set d to garbage
                source.fill_bytes(res_ref.data_mut());
                source.fill_bytes(res_test.data_mut());

                // Reference
                for j in 0..cols {
                    module_ref.vec_znx_big_normalize(
                        &mut vec_znx_backend_mut::<BR>(&mut res_ref),
                        base2k,
                        res_offset,
                        j,
                        &a_ref.to_backend_ref(),
                        base2k,
                        j,
                        &mut scratch_ref.arena(),
                    );
                    module_test.vec_znx_big_normalize(
                        &mut vec_znx_backend_mut::<BT>(&mut res_test),
                        base2k,
                        res_offset,
                        j,
                        &a_test.to_backend_ref(),
                        base2k,
                        j,
                        &mut scratch_test.arena(),
                    );
                }

                assert_eq!(a_ref.digest_u64(), a_ref_digest);
                assert_eq!(a_test.digest_u64(), a_test_digest);

                assert_eq!(res_ref, res_test);
            }
        }
    }
}

pub fn test_vec_znx_big_sub<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>:
        VecZnxBigSub<BR> + VecZnxBigAlloc<BR> + VecZnxBigFromSmall<BR> + VecZnxBigNormalize<BR> + VecZnxBigNormalizeTmpBytes,
    Module<BT>:
        VecZnxBigSub<BT> + VecZnxBigAlloc<BT> + VecZnxBigFromSmall<BT> + VecZnxBigNormalize<BT> + VecZnxBigNormalizeTmpBytes,
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

        let mut a_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, a_size);
        let mut a_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_big_from_small(&mut a_ref.to_backend_mut(), j, &a, j);
            module_test.vec_znx_big_from_small(&mut a_test.to_backend_mut(), j, &a, j);
        }

        let a_ref_digest: u64 = a_ref.digest_u64();
        let a_test_digest: u64 = a_test.digest_u64();

        for b_size in [1, 2, 3, 4] {
            let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, b_size);
            b.fill_uniform(base2k, &mut source);

            let mut b_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, b_size);
            let mut b_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, b_size);

            for j in 0..cols {
                module_ref.vec_znx_big_from_small(&mut b_ref.to_backend_mut(), j, &b, j);
                module_test.vec_znx_big_from_small(&mut b_test.to_backend_mut(), j, &b, j);
            }

            let b_ref_digest: u64 = b_ref.digest_u64();
            let b_test_digest: u64 = b_test.digest_u64();

            for res_size in [1, 2, 3, 4] {
                let mut res_big_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, res_size);
                let mut res_big_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, res_size);

                // Set res to garbage
                source.fill_bytes(res_big_ref.data_mut().as_mut());
                source.fill_bytes(res_big_test.data_mut().as_mut());

                // Reference
                for i in 0..cols {
                    module_ref.vec_znx_big_sub(
                        &mut res_big_ref.to_backend_mut(),
                        i,
                        &a_ref.to_backend_ref(),
                        i,
                        &b_ref.to_backend_ref(),
                        i,
                    );
                    module_test.vec_znx_big_sub(
                        &mut res_big_test.to_backend_mut(),
                        i,
                        &a_test.to_backend_ref(),
                        i,
                        &b_test.to_backend_ref(),
                        i,
                    );
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

                assert_eq!(res_big_ref.digest_u64(), res_ref_digest);
                assert_eq!(res_big_test.digest_u64(), res_test_digest);

                assert_eq!(res_small_ref, res_small_test);
            }
        }
    }
}

pub fn test_vec_znx_big_sub_inplace<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>: VecZnxBigSubInplace<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmall<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigSubAssign<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmall<BT>
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

        let mut a_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, a_size);
        let mut a_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_big_from_small(&mut a_ref.to_backend_mut(), j, &a, j);
            module_test.vec_znx_big_from_small(&mut a_test.to_backend_mut(), j, &a, j);
        }

        let a_ref_digest: u64 = a_ref.digest_u64();
        let a_test_digest: u64 = a_test.digest_u64();

        for res_size in [1, 2, 3, 4] {
            let mut res: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            res.fill_uniform(base2k, &mut source);

            let mut res_big_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, res_size);
            let mut res_big_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, res_size);

            for j in 0..cols {
                module_ref.vec_znx_big_from_small(&mut res_big_ref.to_backend_mut(), j, &res, j);
                module_test.vec_znx_big_from_small(&mut res_big_test.to_backend_mut(), j, &res, j);
            }

            for i in 0..cols {
                module_ref.vec_znx_big_sub_inplace(&mut res_big_ref.to_backend_mut(), i, &a_ref.to_backend_ref(), i);
                module_test.vec_znx_big_sub_inplace(&mut res_big_test.to_backend_mut(), i, &a_test.to_backend_ref(), i);
            }

            assert_eq!(a_ref.digest_u64(), a_ref_digest);
            assert_eq!(a_test.digest_u64(), a_test_digest);

            let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            let res_ref_digest: u64 = res_big_ref.digest_u64();
            let res_test_digest: u64 = res_big_test.digest_u64();

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

            assert_eq!(res_big_ref.digest_u64(), res_ref_digest);
            assert_eq!(res_big_test.digest_u64(), res_test_digest);

            assert_eq!(res_small_ref, res_small_test);
        }
    }
}

pub fn test_vec_znx_big_sub_negate_inplace<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>: VecZnxBigSubNegateInplace<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmall<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigSubNegateAssign<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmall<BT>
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

        let mut a_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, a_size);
        let mut a_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_big_from_small(&mut a_ref.to_backend_mut(), j, &a, j);
            module_test.vec_znx_big_from_small(&mut a_test.to_backend_mut(), j, &a, j);
        }

        let a_ref_digest: u64 = a_ref.digest_u64();
        let a_test_digest: u64 = a_test.digest_u64();

        for res_size in [1, 2, 3, 4] {
            let mut res: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            res.fill_uniform(base2k, &mut source);

            let mut res_big_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, res_size);
            let mut res_big_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, res_size);

            for j in 0..cols {
                module_ref.vec_znx_big_from_small(&mut res_big_ref.to_backend_mut(), j, &res, j);
                module_test.vec_znx_big_from_small(&mut res_big_test.to_backend_mut(), j, &res, j);
            }

            for i in 0..cols {
                module_ref.vec_znx_big_sub_negate_inplace(&mut res_big_ref.to_backend_mut(), i, &a_ref.to_backend_ref(), i);
                module_test.vec_znx_big_sub_negate_inplace(&mut res_big_test.to_backend_mut(), i, &a_test.to_backend_ref(), i);
            }

            assert_eq!(a_ref.digest_u64(), a_ref_digest);
            assert_eq!(a_test.digest_u64(), a_test_digest);

            let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            let res_ref_digest: u64 = res_big_ref.digest_u64();
            let res_test_digest: u64 = res_big_test.digest_u64();

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

            assert_eq!(res_big_ref.digest_u64(), res_ref_digest);
            assert_eq!(res_big_test.digest_u64(), res_test_digest);

            assert_eq!(res_small_ref, res_small_test);
        }
    }
}

pub fn test_vec_znx_big_sub_small_a<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
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

        let mut a_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, a_size);
        let mut a_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_big_from_small(&mut a_ref.to_backend_mut(), j, &a, j);
            module_test.vec_znx_big_from_small(&mut a_test.to_backend_mut(), j, &a, j);
        }

        let a_ref_digest: u64 = a_ref.digest_u64();
        let a_test_digest: u64 = a_test.digest_u64();

        for b_size in [1, 2, 3, 4] {
            let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, b_size);
            b.fill_uniform(base2k, &mut source);
            let b_digest: u64 = b.digest_u64();

            for res_size in [1, 2, 3, 4] {
                let mut res_big_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, res_size);
                let mut res_big_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, res_size);

                // Set res to garbage
                source.fill_bytes(res_big_ref.data_mut().as_mut());
                source.fill_bytes(res_big_test.data_mut().as_mut());

                // Reference
                for i in 0..cols {
                    module_ref.vec_znx_big_sub_small_a(&mut res_big_ref.to_backend_mut(), i, &b, i, &a_ref.to_backend_ref(), i);
                    module_test.vec_znx_big_sub_small_a(
                        &mut res_big_test.to_backend_mut(),
                        i,
                        &b,
                        i,
                        &a_test.to_backend_ref(),
                        i,
                    );
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

                assert_eq!(res_big_ref.digest_u64(), res_ref_digest);
                assert_eq!(res_big_test.digest_u64(), res_test_digest);

                assert_eq!(res_small_ref, res_small_test);
            }
        }
    }
}

pub fn test_vec_znx_big_sub_small_b<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
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

        let mut a_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, a_size);
        let mut a_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_big_from_small(&mut a_ref.to_backend_mut(), j, &a, j);
            module_test.vec_znx_big_from_small(&mut a_test.to_backend_mut(), j, &a, j);
        }

        let a_ref_digest: u64 = a_ref.digest_u64();
        let a_test_digest: u64 = a_test.digest_u64();

        for b_size in [1, 2, 3, 4] {
            let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, b_size);
            b.fill_uniform(base2k, &mut source);
            let b_digest: u64 = b.digest_u64();

            for res_size in [1, 2, 3, 4] {
                let mut res_big_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, res_size);
                let mut res_big_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, res_size);

                // Set res to garbage
                source.fill_bytes(res_big_ref.data_mut().as_mut());
                source.fill_bytes(res_big_test.data_mut().as_mut());

                // Reference
                for i in 0..cols {
                    module_ref.vec_znx_big_sub_small_b(&mut res_big_ref.to_backend_mut(), i, &a_ref.to_backend_ref(), i, &b, i);
                    module_test.vec_znx_big_sub_small_b(
                        &mut res_big_test.to_backend_mut(),
                        i,
                        &a_test.to_backend_ref(),
                        i,
                        &b,
                        i,
                    );
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

                assert_eq!(res_big_ref.digest_u64(), res_ref_digest);
                assert_eq!(res_big_test.digest_u64(), res_test_digest);

                assert_eq!(res_small_ref, res_small_test);
            }
        }
    }
}

pub fn test_vec_znx_big_sub_small_a_inplace<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>: VecZnxBigSubSmallInplace<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmall<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigSubSmallAssign<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmall<BT>
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

        let a_digest: u64 = a.digest_u64();

        for res_size in [1, 2, 3, 4] {
            let mut res: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            res.fill_uniform(base2k, &mut source);

            let mut res_big_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, res_size);
            let mut res_big_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, res_size);

            for j in 0..cols {
                module_ref.vec_znx_big_from_small(&mut res_big_ref.to_backend_mut(), j, &res, j);
                module_test.vec_znx_big_from_small(&mut res_big_test.to_backend_mut(), j, &res, j);
            }

            for i in 0..cols {
                module_ref.vec_znx_big_sub_small_inplace(&mut res_big_ref.to_backend_mut(), i, &vec_znx_backend_ref::<BR>(&a), i);
                module_test.vec_znx_big_sub_small_inplace(
                    &mut res_big_test.to_backend_mut(),
                    i,
                    &vec_znx_backend_ref::<BT>(&a),
                    i,
                );
            }

            assert_eq!(a.digest_u64(), a_digest);

            let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            let res_ref_digest: u64 = res_big_ref.digest_u64();
            let res_test_digest: u64 = res_big_test.digest_u64();

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

            assert_eq!(res_big_ref.digest_u64(), res_ref_digest);
            assert_eq!(res_big_test.digest_u64(), res_test_digest);

            assert_eq!(res_small_ref, res_small_test);
        }
    }
}

pub fn test_vec_znx_big_sub_small_b_inplace<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>: VecZnxBigSubSmallNegateInplace<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmall<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigSubSmallNegateAssign<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmall<BT>
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

        let a_digest: u64 = a.digest_u64();

        for res_size in [1, 2, 3, 4] {
            for res_offset in -(base2k as i64)..=(base2k as i64) {
                let mut res: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
                res.fill_uniform(base2k, &mut source);

                let mut res_big_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, res_size);
                let mut res_big_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, res_size);

                for j in 0..cols {
                    module_ref.vec_znx_big_from_small(&mut res_big_ref.to_backend_mut(), j, &res, j);
                    module_test.vec_znx_big_from_small(&mut res_big_test.to_backend_mut(), j, &res, j);
                }

                for i in 0..cols {
                    module_ref.vec_znx_big_sub_small_negate_inplace(
                        &mut res_big_ref.to_backend_mut(),
                        i,
                        &vec_znx_backend_ref::<BR>(&a),
                        i,
                    );
                    module_test.vec_znx_big_sub_small_negate_inplace(
                        &mut res_big_test.to_backend_mut(),
                        i,
                        &vec_znx_backend_ref::<BT>(&a),
                        i,
                    );
                }

                assert_eq!(a.digest_u64(), a_digest);

                let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
                let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

                let res_ref_digest: u64 = res_big_ref.digest_u64();
                let res_test_digest: u64 = res_big_test.digest_u64();

                for j in 0..cols {
                    module_ref.vec_znx_big_normalize(
                        &mut vec_znx_backend_mut::<BR>(&mut res_small_ref),
                        base2k,
                        res_offset,
                        j,
                        &res_big_ref.to_backend_ref(),
                        base2k,
                        j,
                        &mut scratch_ref.arena(),
                    );
                    module_test.vec_znx_big_normalize(
                        &mut vec_znx_backend_mut::<BT>(&mut res_small_test),
                        base2k,
                        res_offset,
                        j,
                        &res_big_test.to_backend_ref(),
                        base2k,
                        j,
                        &mut scratch_test.arena(),
                    );
                }

                assert_eq!(res_big_ref.digest_u64(), res_ref_digest);
                assert_eq!(res_big_test.digest_u64(), res_test_digest);

                assert_eq!(res_small_ref, res_small_test);
            }
        }
    }
}
