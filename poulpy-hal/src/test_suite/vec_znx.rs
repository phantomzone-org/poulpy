use super::{TestParams, scalar_znx_backend_ref, vec_znx_backend_mut, vec_znx_backend_ref};
use std::f64::consts::SQRT_2;

use crate::{
    api::{
        ModuleNew, ScalarZnxFillBinaryBlockBackend, ScalarZnxFillBinaryBlockSourceBackend, ScalarZnxFillBinaryHwBackend,
        ScalarZnxFillBinaryHwSourceBackend, ScalarZnxFillBinaryProbBackend, ScalarZnxFillBinaryProbSourceBackend,
        ScalarZnxFillTernaryHwBackend, ScalarZnxFillTernaryHwSourceBackend, ScalarZnxFillTernaryProbBackend,
        ScalarZnxFillTernaryProbSourceBackend, ScratchOwnedAlloc, VecZnxAddAssignBackend, VecZnxAddIntoBackend,
        VecZnxAddNormalSourceBackend, VecZnxAddScalarAssignBackend, VecZnxAddScalarIntoBackend, VecZnxAutomorphismBackend,
        VecZnxAutomorphismInplace, VecZnxAutomorphismInplaceTmpBytes, VecZnxCopyBackend, VecZnxCopyRangeBackend,
        VecZnxFillNormalBackend, VecZnxFillNormalSourceBackend, VecZnxFillUniformBackend, VecZnxFillUniformSourceBackend,
        VecZnxLshBackend, VecZnxLshInplaceBackend, VecZnxLshTmpBytes, VecZnxMergeRingsBackend, VecZnxMergeRingsTmpBytes,
        VecZnxMulXpMinusOneBackend, VecZnxMulXpMinusOneInplaceBackend, VecZnxMulXpMinusOneInplaceTmpBytes, VecZnxNegateBackend,
        VecZnxNegateInplaceBackend, VecZnxNormalize, VecZnxNormalizeInplaceBackend, VecZnxNormalizeTmpBytes, VecZnxRotateBackend,
        VecZnxRotateInplaceBackend, VecZnxRotateInplaceTmpBytes, VecZnxRshBackend, VecZnxRshInplaceBackend, VecZnxRshTmpBytes,
        VecZnxSplitRingBackend, VecZnxSplitRingTmpBytes, VecZnxSubBackend, VecZnxSubInplaceBackend, VecZnxSubNegateInplaceBackend,
        VecZnxSubScalarBackend, VecZnxSubScalarInplaceBackend, VecZnxSwitchRingBackend, VecZnxZeroBackend,
    },
    layouts::{
        DigestU64, FillUniform, Module, NoiseInfos, ScalarZnx, ScalarZnxToBackendMut, ScratchOwned, VecZnx, ZnxInfos, ZnxView,
        ZnxViewMut,
    },
    source::Source,
};

pub fn test_vec_znx_zero_backend_matches_wrapper<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    _module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BT>: VecZnxZeroBackend<BT>,
{
    let base2k = params.base2k;
    let n: usize = module_test.n();
    let cols: usize = 2;
    let mut source: Source = Source::new([5u8; 32]);

    for size in [1, 2, 3, 4] {
        for col_i in 0..cols {
            let mut wrapper: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);
            let mut backend: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);
            wrapper.fill_uniform(base2k, &mut source);
            backend.data.copy_from_slice(&wrapper.data);

            module_test.vec_znx_zero_backend(&mut vec_znx_backend_mut::<BT>(&mut wrapper), col_i);
            module_test.vec_znx_zero_backend(&mut vec_znx_backend_mut::<BT>(&mut backend), col_i);

            assert_eq!(wrapper, backend);
        }
    }
}

pub fn test_vec_znx_encode_vec_i64() {
    let n: usize = 32;
    let base2k: usize = 17;
    let size: usize = 5;
    for k in [1, base2k / 2, size * base2k - 5] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, 2, size);
        let mut source = Source::new([0u8; 32]);
        let raw: &mut [i64] = a.raw_mut();
        raw.iter_mut().enumerate().for_each(|(i, x)| *x = i as i64);
        (0..a.cols()).for_each(|col_i| {
            let mut have: Vec<i64> = vec![i64::default(); n];
            have.iter_mut().for_each(|x| {
                if k < 64 {
                    *x = source.next_u64n(1 << k, (1 << k) - 1) as i64;
                } else {
                    *x = source.next_i64();
                }
            });
            a.encode_vec_i64(base2k, col_i, k, &have);
            let mut want: Vec<i64> = vec![i64::default(); n];
            a.decode_vec_i64(base2k, col_i, k, &mut want);
            assert_eq!(have, want, "{:?} != {:?}", &have, &want);
        })
    }
}

pub fn test_vec_znx_add_scalar_into<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>: VecZnxAddScalarIntoBackend<BR>,
    Module<BT>: VecZnxAddScalarIntoBackend<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut a: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, cols);
    a.fill_uniform(base2k, &mut source);
    let a_digest = a.digest_u64();

    for a_size in [1, 2, 3, 4] {
        let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        b.fill_uniform(base2k, &mut source);
        let b_digest: u64 = b.digest_u64();

        for res_size in [1, 2, 3, 4] {
            let mut rest_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            // Set d to garbage
            rest_ref.fill_uniform(base2k, &mut source);
            res_test.fill_uniform(base2k, &mut source);

            // Reference
            for i in 0..cols {
                module_ref.vec_znx_add_scalar_into_backend(
                    &mut vec_znx_backend_mut::<BR>(&mut rest_ref),
                    i,
                    &scalar_znx_backend_ref::<BR>(&a),
                    i,
                    &vec_znx_backend_ref::<BR>(&b),
                    i,
                    (res_size.min(a_size)) - 1,
                );
                module_test.vec_znx_add_scalar_into_backend(
                    &mut vec_znx_backend_mut::<BT>(&mut res_test),
                    i,
                    &scalar_znx_backend_ref::<BT>(&a),
                    i,
                    &vec_znx_backend_ref::<BT>(&b),
                    i,
                    (res_size.min(a_size)) - 1,
                );
            }

            assert_eq!(b.digest_u64(), b_digest);
            assert_eq!(a.digest_u64(), a_digest);
            assert_eq!(rest_ref, res_test);
        }
    }
}

pub fn test_vec_znx_add_scalar_assign<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>: VecZnxAddScalarAssignBackend<BR>,
    Module<BT>: VecZnxAddScalarAssignBackend<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut b: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, cols);
    b.fill_uniform(base2k, &mut source);
    let b_digest: u64 = b.digest_u64();

    for res_size in [1, 2, 3, 4] {
        let mut rest_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
        let mut res_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

        rest_ref.fill_uniform(base2k, &mut source);
        res_test.raw_mut().copy_from_slice(rest_ref.raw());

        for i in 0..cols {
            module_ref.vec_znx_add_scalar_assign_backend(
                &mut vec_znx_backend_mut::<BR>(&mut rest_ref),
                i,
                res_size - 1,
                &scalar_znx_backend_ref::<BR>(&b),
                i,
            );
            module_test.vec_znx_add_scalar_assign_backend(
                &mut vec_znx_backend_mut::<BT>(&mut res_test),
                i,
                res_size - 1,
                &scalar_znx_backend_ref::<BT>(&b),
                i,
            );
        }

        assert_eq!(b.digest_u64(), b_digest);
        assert_eq!(rest_ref, res_test);
    }
}
pub fn test_vec_znx_add_into_backend_matches_reference<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>: VecZnxAddIntoBackend<BR>,
    Module<BT>: VecZnxAddIntoBackend<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();
    let cols: usize = 2;
    let mut source: Source = Source::new([13u8; 32]);

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest: u64 = a.digest_u64();

        for b_size in [1, 2, 3, 4] {
            let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, b_size);
            b.fill_uniform(base2k, &mut source);
            let b_digest: u64 = b.digest_u64();

            for res_size in [1, 2, 3, 4] {
                let mut wrapper: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
                let mut backend: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

                wrapper.fill_uniform(base2k, &mut source);
                backend.data.copy_from_slice(&wrapper.data);

                for col_i in 0..cols {
                    module_ref.vec_znx_add_into_backend(
                        &mut vec_znx_backend_mut::<BR>(&mut wrapper),
                        col_i,
                        &vec_znx_backend_ref::<BR>(&a),
                        col_i,
                        &vec_znx_backend_ref::<BR>(&b),
                        col_i,
                    );
                    module_test.vec_znx_add_into_backend(
                        &mut vec_znx_backend_mut::<BT>(&mut backend),
                        col_i,
                        &vec_znx_backend_ref::<BT>(&a),
                        col_i,
                        &vec_znx_backend_ref::<BT>(&b),
                        col_i,
                    );
                }

                assert_eq!(a.digest_u64(), a_digest);
                assert_eq!(b.digest_u64(), b_digest);
                assert_eq!(wrapper, backend);
            }
        }
    }
}

pub fn test_vec_znx_add_assign<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>: VecZnxAddAssignBackend<BR>,
    Module<BT>: VecZnxAddAssignBackend<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest: u64 = a.digest_u64();

        for res_size in [1, 2, 3, 4] {
            let mut res_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            res_ref.fill_uniform(base2k, &mut source);
            res_test.raw_mut().copy_from_slice(res_ref.raw());

            for i in 0..cols {
                module_ref.vec_znx_add_assign_backend(
                    &mut vec_znx_backend_mut::<BR>(&mut res_ref),
                    i,
                    &vec_znx_backend_ref::<BR>(&a),
                    i,
                );
                module_test.vec_znx_add_assign_backend(
                    &mut vec_znx_backend_mut::<BT>(&mut res_test),
                    i,
                    &vec_znx_backend_ref::<BT>(&a),
                    i,
                );
            }

            assert_eq!(a.digest_u64(), a_digest);
            assert_eq!(res_ref, res_test);
        }
    }
}

pub fn test_vec_znx_add_assign_backend_matches_wrapper<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    _module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BT>: VecZnxAddAssignBackend<BT>,
{
    let base2k = params.base2k;
    let n: usize = module_test.n();
    let cols: usize = 2;
    let mut source: Source = Source::new([14u8; 32]);

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest: u64 = a.digest_u64();

        for res_size in [1, 2, 3, 4] {
            let mut wrapper: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut backend: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            wrapper.fill_uniform(base2k, &mut source);
            backend.data.copy_from_slice(&wrapper.data);

            for col_i in 0..cols {
                module_test.vec_znx_add_assign_backend(
                    &mut vec_znx_backend_mut::<BT>(&mut wrapper),
                    col_i,
                    &vec_znx_backend_ref::<BT>(&a),
                    col_i,
                );
                module_test.vec_znx_add_assign_backend(
                    &mut vec_znx_backend_mut::<BT>(&mut backend),
                    col_i,
                    &vec_znx_backend_ref::<BT>(&a),
                    col_i,
                );
            }

            assert_eq!(a.digest_u64(), a_digest);
            assert_eq!(wrapper, backend);
        }
    }
}

pub fn test_vec_znx_automorphism<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>: VecZnxAutomorphismBackend<BR>,
    Module<BT>: VecZnxAutomorphismBackend<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest: u64 = a.digest_u64();

        for res_size in [1, 2, 3, 4] {
            let mut res_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            let p: i64 = -5;

            // Normalize on c
            for i in 0..cols {
                module_ref.vec_znx_automorphism_backend(
                    p,
                    &mut vec_znx_backend_mut::<BR>(&mut res_ref),
                    i,
                    &vec_znx_backend_ref::<BR>(&a),
                    i,
                );
                module_test.vec_znx_automorphism_backend(
                    p,
                    &mut vec_znx_backend_mut::<BT>(&mut res_test),
                    i,
                    &vec_znx_backend_ref::<BT>(&a),
                    i,
                );
            }

            assert_eq!(a.digest_u64(), a_digest);
            assert_eq!(res_ref, res_test);

            let p: i64 = 5;

            // Normalize on c
            for i in 0..cols {
                module_ref.vec_znx_automorphism_backend(
                    p,
                    &mut vec_znx_backend_mut::<BR>(&mut res_ref),
                    i,
                    &vec_znx_backend_ref::<BR>(&a),
                    i,
                );
                module_test.vec_znx_automorphism_backend(
                    p,
                    &mut vec_znx_backend_mut::<BT>(&mut res_test),
                    i,
                    &vec_znx_backend_ref::<BT>(&a),
                    i,
                );
            }

            assert_eq!(a.digest_u64(), a_digest);
            assert_eq!(res_ref, res_test);
        }
    }
}

pub fn test_vec_znx_automorphism_inplace<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>: VecZnxAutomorphismInplace<BR> + VecZnxAutomorphismInplaceTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    Module<BT>: VecZnxAutomorphismInplace<BT> + VecZnxAutomorphismInplaceTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_automorphism_assign_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_automorphism_assign_tmp_bytes());

    for size in [1, 2, 3, 4] {
        let mut res_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);
        let mut res_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);

        // Fill a with random i64
        res_ref.fill_uniform(base2k, &mut source);
        res_test.raw_mut().copy_from_slice(res_ref.raw());

        let p: i64 = -7;

        // Normalize on c
        for i in 0..cols {
            module_ref.vec_znx_automorphism_inplace(p, &mut vec_znx_backend_mut::<BR>(&mut res_ref), i, &mut scratch_ref.arena());
            module_test.vec_znx_automorphism_inplace(
                p,
                &mut vec_znx_backend_mut::<BT>(&mut res_test),
                i,
                &mut scratch_test.arena(),
            );
        }

        assert_eq!(res_ref, res_test);

        let p: i64 = 7;

        // Normalize on c
        for i in 0..cols {
            module_ref.vec_znx_automorphism_inplace(p, &mut vec_znx_backend_mut::<BR>(&mut res_ref), i, &mut scratch_ref.arena());
            module_test.vec_znx_automorphism_inplace(
                p,
                &mut vec_znx_backend_mut::<BT>(&mut res_test),
                i,
                &mut scratch_test.arena(),
            );
        }

        assert_eq!(res_ref, res_test);
    }
}

pub fn test_vec_znx_copy<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>: VecZnxCopyBackend<BR>,
    Module<BT>: VecZnxCopyBackend<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest: u64 = a.digest_u64();

        for res_size in [1, 2, 3, 4] {
            let mut res_0: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_1: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            // Set d to garbage
            res_0.fill_uniform(base2k, &mut source);
            res_1.fill_uniform(base2k, &mut source);

            // Reference
            for i in 0..cols {
                module_ref.vec_znx_copy_backend(
                    &mut vec_znx_backend_mut::<BR>(&mut res_0),
                    i,
                    &vec_znx_backend_ref::<BR>(&a),
                    i,
                );
                module_test.vec_znx_copy_backend(
                    &mut vec_znx_backend_mut::<BT>(&mut res_1),
                    i,
                    &vec_znx_backend_ref::<BT>(&a),
                    i,
                );
            }

            assert_eq!(a.digest_u64(), a_digest);
            assert_eq!(res_0, res_1);
        }
    }
}

pub fn test_vec_znx_copy_backend_matches_wrapper<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    _module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BT>: VecZnxCopyBackend<BT>,
{
    let base2k = params.base2k;
    let n: usize = module_test.n();
    let cols: usize = 2;
    let a_col: usize = 0;
    let res_col: usize = 1;
    let mut source: Source = Source::new([3u8; 32]);

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(base2k, &mut source);

        for res_size in [1, 2, 3, 4] {
            let mut wrapper: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut backend: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            wrapper.fill_uniform(base2k, &mut source);
            backend.data.copy_from_slice(&wrapper.data);

            module_test.vec_znx_copy_backend(
                &mut vec_znx_backend_mut::<BT>(&mut wrapper),
                res_col,
                &vec_znx_backend_ref::<BT>(&a),
                a_col,
            );
            module_test.vec_znx_copy_backend(
                &mut vec_znx_backend_mut::<BT>(&mut backend),
                res_col,
                &vec_znx_backend_ref::<BT>(&a),
                a_col,
            );

            assert_eq!(wrapper, backend);
        }
    }
}

pub fn test_vec_znx_copy_range_backend<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    _module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BT>: VecZnxCopyRangeBackend<BT>,
{
    let base2k = params.base2k;
    let n: usize = module_test.n();
    let cols: usize = 2;
    let a_col: usize = 0;
    let res_col: usize = 1;
    let mut source: Source = Source::new([13u8; 32]);

    for a_size in [1, 2, 3, 4] {
        let a_limb = a_size - 1;
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(base2k, &mut source);

        for res_size in [1, 2, 3, 4] {
            let res_limb = res_size - 1;
            let mut expected: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut actual: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            expected.fill_uniform(base2k, &mut source);
            actual.data.copy_from_slice(&expected.data);

            for (res_offset, a_offset, len) in [(0usize, 0usize, 1usize), (1, 0, 2), (0, 1, 3), (2, 4, 5)] {
                if res_offset + len > n || a_offset + len > n {
                    continue;
                }

                expected.at_mut(res_col, res_limb)[res_offset..res_offset + len]
                    .copy_from_slice(&a.at(a_col, a_limb)[a_offset..a_offset + len]);

                module_test.vec_znx_copy_range_backend(
                    &mut vec_znx_backend_mut::<BT>(&mut actual),
                    res_col,
                    res_limb,
                    res_offset,
                    &vec_znx_backend_ref::<BT>(&a),
                    a_col,
                    a_limb,
                    a_offset,
                    len,
                );
            }

            assert_eq!(expected, actual);
        }
    }
}

pub fn test_vec_znx_merge_rings<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>: VecZnxMergeRingsBackend<BR> + ModuleNew<BR> + VecZnxMergeRingsTmpBytes,
    Module<BT>: VecZnxMergeRingsBackend<BT> + ModuleNew<BT> + VecZnxMergeRingsTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;
    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_merge_rings_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_merge_rings_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: [VecZnx<Vec<u8>>; 2] = [VecZnx::alloc(n >> 1, cols, a_size), VecZnx::alloc(n >> 1, cols, a_size)];

        a.iter_mut().for_each(|ai| {
            ai.fill_uniform(base2k, &mut source);
        });

        let a_digests: [u64; 2] = [a[0].digest_u64(), a[1].digest_u64()];

        for res_size in [1, 2, 3, 4] {
            let mut res_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            res_ref.fill_uniform(base2k, &mut source);
            res_test.fill_uniform(base2k, &mut source);

            for i in 0..cols {
                let a_ref: Vec<_> = a.iter().map(vec_znx_backend_ref::<BR>).collect();
                module_ref.vec_znx_merge_rings_backend(
                    &mut vec_znx_backend_mut::<BR>(&mut res_test),
                    i,
                    &a_ref,
                    i,
                    &mut scratch_ref.arena(),
                );
                let a_test: Vec<_> = a.iter().map(vec_znx_backend_ref::<BT>).collect();
                module_test.vec_znx_merge_rings_backend(
                    &mut vec_znx_backend_mut::<BT>(&mut res_ref),
                    i,
                    &a_test,
                    i,
                    &mut scratch_test.arena(),
                );
            }

            assert_eq!([a[0].digest_u64(), a[1].digest_u64()], a_digests);
            assert_eq!(res_ref, res_test);
        }
    }
}

pub fn test_vec_znx_mul_xp_minus_one<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>: VecZnxMulXpMinusOneBackend<BR>,
    Module<BT>: VecZnxMulXpMinusOneBackend<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(base2k, &mut source);

        let a_digest: u64 = a.digest_u64();

        for res_size in [1, 2, 3, 4] {
            let mut res_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            let p: i64 = -5;

            // Normalize on c
            for i in 0..cols {
                module_ref.vec_znx_mul_xp_minus_one_backend(
                    p,
                    &mut vec_znx_backend_mut::<BR>(&mut res_ref),
                    i,
                    &vec_znx_backend_ref::<BR>(&a),
                    i,
                );
                module_test.vec_znx_mul_xp_minus_one_backend(
                    p,
                    &mut vec_znx_backend_mut::<BT>(&mut res_test),
                    i,
                    &vec_znx_backend_ref::<BT>(&a),
                    i,
                );
            }

            assert_eq!(a.digest_u64(), a_digest);
            assert_eq!(res_test, res_ref);

            let p: i64 = 5;

            // Normalize on c
            for i in 0..cols {
                module_ref.vec_znx_mul_xp_minus_one_backend(
                    p,
                    &mut vec_znx_backend_mut::<BR>(&mut res_ref),
                    i,
                    &vec_znx_backend_ref::<BR>(&a),
                    i,
                );
                module_test.vec_znx_mul_xp_minus_one_backend(
                    p,
                    &mut vec_znx_backend_mut::<BT>(&mut res_test),
                    i,
                    &vec_znx_backend_ref::<BT>(&a),
                    i,
                );
            }

            assert_eq!(a.digest_u64(), a_digest);
            assert_eq!(res_test, res_ref);
        }
    }
}

pub fn test_vec_znx_mul_xp_minus_one_inplace<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>: VecZnxMulXpMinusOneInplaceBackend<BR> + VecZnxMulXpMinusOneInplaceTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    Module<BT>: VecZnxMulXpMinusOneInplaceBackend<BT> + VecZnxMulXpMinusOneInplaceTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_mul_xp_minus_one_assign_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_mul_xp_minus_one_assign_tmp_bytes());

    for size in [1, 2, 3, 4] {
        let mut res_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);
        let mut res_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);

        // Fill a with random i64
        res_ref.fill_uniform(base2k, &mut source);
        res_test.raw_mut().copy_from_slice(res_ref.raw());

        let p: i64 = -7;

        for i in 0..cols {
            module_ref.vec_znx_mul_xp_minus_one_inplace_backend(
                p,
                &mut vec_znx_backend_mut::<BR>(&mut res_ref),
                i,
                &mut scratch_ref.arena(),
            );
            module_test.vec_znx_mul_xp_minus_one_inplace_backend(
                p,
                &mut vec_znx_backend_mut::<BT>(&mut res_test),
                i,
                &mut scratch_test.arena(),
            );
        }

        assert_eq!(res_ref, res_test);

        let p: i64 = 7;

        for i in 0..cols {
            module_ref.vec_znx_mul_xp_minus_one_inplace_backend(
                p,
                &mut vec_znx_backend_mut::<BR>(&mut res_ref),
                i,
                &mut scratch_ref.arena(),
            );
            module_test.vec_znx_mul_xp_minus_one_inplace_backend(
                p,
                &mut vec_znx_backend_mut::<BT>(&mut res_test),
                i,
                &mut scratch_test.arena(),
            );
        }

        assert_eq!(res_ref, res_test);
    }
}

pub fn test_vec_znx_negate<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>: VecZnxNegateBackend<BR>,
    Module<BT>: VecZnxNegateBackend<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest: u64 = a.digest_u64();

        for res_size in [1, 2, 3, 4] {
            let mut res_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            res_ref.fill_uniform(base2k, &mut source);
            res_test.raw_mut().copy_from_slice(res_ref.raw());

            for i in 0..cols {
                module_ref.vec_znx_negate_backend(
                    &mut vec_znx_backend_mut::<BR>(&mut res_ref),
                    i,
                    &vec_znx_backend_ref::<BR>(&a),
                    i,
                );
                module_test.vec_znx_negate_backend(
                    &mut vec_znx_backend_mut::<BT>(&mut res_test),
                    i,
                    &vec_znx_backend_ref::<BT>(&a),
                    i,
                );
            }

            assert_eq!(a.digest_u64(), a_digest);
            assert_eq!(res_ref, res_test);
        }
    }
}

pub fn test_vec_znx_negate_backend_matches_wrapper<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    _module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BT>: VecZnxNegateBackend<BT>,
{
    let base2k = params.base2k;
    let n: usize = module_test.n();
    let cols: usize = 2;
    let a_col: usize = 0;
    let res_col: usize = 1;
    let mut source: Source = Source::new([6u8; 32]);

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(base2k, &mut source);

        for res_size in [1, 2, 3, 4] {
            let mut wrapper: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut backend: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            wrapper.fill_uniform(base2k, &mut source);
            backend.data.copy_from_slice(&wrapper.data);

            module_test.vec_znx_negate_backend(
                &mut vec_znx_backend_mut::<BT>(&mut wrapper),
                res_col,
                &vec_znx_backend_ref::<BT>(&a),
                a_col,
            );
            module_test.vec_znx_negate_backend(
                &mut vec_znx_backend_mut::<BT>(&mut backend),
                res_col,
                &vec_znx_backend_ref::<BT>(&a),
                a_col,
            );

            assert_eq!(wrapper, backend);
        }
    }
}

pub fn test_vec_znx_negate_inplace<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>: VecZnxNegateInplaceBackend<BR>,
    Module<BT>: VecZnxNegateInplaceBackend<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for res_size in [1, 2, 3, 4] {
        let mut res_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
        let mut res_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

        res_ref.fill_uniform(base2k, &mut source);
        res_test.raw_mut().copy_from_slice(res_ref.raw());

        for i in 0..cols {
            module_ref.vec_znx_negate_inplace_backend(&mut vec_znx_backend_mut::<BR>(&mut res_ref), i);
            module_test.vec_znx_negate_inplace_backend(&mut vec_znx_backend_mut::<BT>(&mut res_test), i);
        }

        assert_eq!(res_ref, res_test);
    }
}

pub fn test_vec_znx_negate_inplace_backend_matches_wrapper<
    BR: crate::test_suite::TestBackend,
    BT: crate::test_suite::TestBackend,
>(
    params: &TestParams,
    _module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BT>: VecZnxNegateInplaceBackend<BT>,
{
    let base2k = params.base2k;
    let n: usize = module_test.n();
    let cols: usize = 2;
    let mut source: Source = Source::new([7u8; 32]);

    for res_size in [1, 2, 3, 4] {
        for col_i in 0..cols {
            let mut wrapper: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut backend: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            wrapper.fill_uniform(base2k, &mut source);
            backend.data.copy_from_slice(&wrapper.data);

            module_test.vec_znx_negate_inplace_backend(&mut vec_znx_backend_mut::<BT>(&mut wrapper), col_i);
            module_test.vec_znx_negate_inplace_backend(&mut vec_znx_backend_mut::<BT>(&mut backend), col_i);

            assert_eq!(wrapper, backend);
        }
    }
}

pub fn test_vec_znx_normalize<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>: VecZnxNormalize<BR> + VecZnxNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    Module<BT>: VecZnxNormalize<BT> + VecZnxNormalizeTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_normalize_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest: u64 = a.digest_u64();

        for res_size in [1, 2, 3, 4] {
            let mut res_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            for res_offset in -(base2k as i64)..=(base2k as i64) {
                // Set d to garbage
                res_ref.fill_uniform(base2k, &mut source);
                res_test.fill_uniform(base2k, &mut source);

                // Reference
                for i in 0..cols {
                    module_ref.vec_znx_normalize(
                        &mut vec_znx_backend_mut::<BR>(&mut res_ref),
                        base2k,
                        res_offset,
                        i,
                        &vec_znx_backend_ref::<BR>(&a),
                        base2k,
                        i,
                        &mut scratch_ref.arena(),
                    );
                    module_test.vec_znx_normalize(
                        &mut vec_znx_backend_mut::<BT>(&mut res_test),
                        base2k,
                        res_offset,
                        i,
                        &vec_znx_backend_ref::<BT>(&a),
                        base2k,
                        i,
                        &mut scratch_test.arena(),
                    );
                }

                assert_eq!(a.digest_u64(), a_digest);
                assert_eq!(res_ref, res_test);
            }
        }
    }
}

pub fn test_vec_znx_normalize_inplace<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>: VecZnxNormalizeInplaceBackend<BR> + VecZnxNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    Module<BT>: VecZnxNormalizeInplaceBackend<BT> + VecZnxNormalizeTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_normalize_tmp_bytes());

    for res_size in [1, 2, 3, 4] {
        let mut res_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
        let mut res_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

        res_ref.fill_uniform(base2k, &mut source);
        res_test.raw_mut().copy_from_slice(res_ref.raw());

        // Reference
        for i in 0..cols {
            module_ref.vec_znx_normalize_inplace_backend(
                base2k,
                &mut vec_znx_backend_mut::<BR>(&mut res_ref),
                i,
                &mut scratch_ref.arena(),
            );
            module_test.vec_znx_normalize_inplace_backend(
                base2k,
                &mut vec_znx_backend_mut::<BT>(&mut res_test),
                i,
                &mut scratch_test.arena(),
            );
        }

        assert_eq!(res_ref, res_test);
    }
}

pub fn test_vec_znx_rotate<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>: VecZnxRotateBackend<BR>,
    Module<BT>: VecZnxRotateBackend<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest: u64 = a.digest_u64();

        for res_size in [1, 2, 3, 4] {
            let mut res_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            let p: i64 = -5;

            // Normalize on c
            for i in 0..cols {
                module_ref.vec_znx_rotate_backend(
                    p,
                    &mut vec_znx_backend_mut::<BR>(&mut res_ref),
                    i,
                    &vec_znx_backend_ref::<BR>(&a),
                    i,
                );
                module_test.vec_znx_rotate_backend(
                    p,
                    &mut vec_znx_backend_mut::<BT>(&mut res_test),
                    i,
                    &vec_znx_backend_ref::<BT>(&a),
                    i,
                );
            }

            assert_eq!(a.digest_u64(), a_digest);
            assert_eq!(res_ref, res_test);

            let p: i64 = 5;

            // Normalize on c
            for i in 0..cols {
                module_ref.vec_znx_rotate_backend(
                    p,
                    &mut vec_znx_backend_mut::<BR>(&mut res_ref),
                    i,
                    &vec_znx_backend_ref::<BR>(&a),
                    i,
                );
                module_test.vec_znx_rotate_backend(
                    p,
                    &mut vec_znx_backend_mut::<BT>(&mut res_test),
                    i,
                    &vec_znx_backend_ref::<BT>(&a),
                    i,
                );
            }

            assert_eq!(a.digest_u64(), a_digest);
            assert_eq!(res_ref, res_test);
        }
    }
}

pub fn test_vec_znx_rotate_inplace<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>: VecZnxRotateInplaceBackend<BR> + VecZnxRotateInplaceTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    Module<BT>: VecZnxRotateInplaceBackend<BT> + VecZnxRotateInplaceTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_rotate_assign_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_rotate_assign_tmp_bytes());

    for size in [1, 2, 3, 4] {
        let mut res_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);
        let mut res_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);

        // Fill a with random i64
        res_ref.fill_uniform(base2k, &mut source);
        res_test.raw_mut().copy_from_slice(res_ref.raw());

        let p: i64 = -5;

        // Normalize on c
        for i in 0..cols {
            module_ref.vec_znx_rotate_inplace_backend(
                p,
                &mut vec_znx_backend_mut::<BR>(&mut res_ref),
                i,
                &mut scratch_ref.arena(),
            );
            module_test.vec_znx_rotate_inplace_backend(
                p,
                &mut vec_znx_backend_mut::<BT>(&mut res_test),
                i,
                &mut scratch_test.arena(),
            );
        }

        assert_eq!(res_ref, res_test);

        let p: i64 = 5;

        // Normalize on c
        for i in 0..cols {
            module_ref.vec_znx_rotate_inplace_backend(
                p,
                &mut vec_znx_backend_mut::<BR>(&mut res_ref),
                i,
                &mut scratch_ref.arena(),
            );
            module_test.vec_znx_rotate_inplace_backend(
                p,
                &mut vec_znx_backend_mut::<BT>(&mut res_test),
                i,
                &mut scratch_test.arena(),
            );
        }

        assert_eq!(res_ref, res_test);
    }
}

pub fn test_vec_znx_fill_uniform<B: crate::test_suite::TestBackend>(_params: &TestParams, module: &Module<B>)
where
    B::OwnedBuf: crate::layouts::HostDataMut,
    Module<B>: VecZnxFillUniformSourceBackend<B>,
{
    let n: usize = module.n();
    let base2k: usize = 17;
    let size: usize = 5;
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;
    let zero: Vec<i64> = vec![0; n];
    let one_12_sqrt: f64 = 0.28867513459481287;
    (0..cols).for_each(|col_i| {
        let mut a: VecZnx<_> = VecZnx::alloc(n, cols, size);
        module.vec_znx_fill_uniform_source_backend(base2k, &mut vec_znx_backend_mut::<B>(&mut a), col_i, &mut source);
        (0..cols).for_each(|col_j| {
            if col_j != col_i {
                (0..size).for_each(|limb_i| {
                    assert_eq!(a.at(col_j, limb_i), zero);
                })
            } else {
                let std: f64 = a.stats(base2k, col_i).std();
                assert!((std - one_12_sqrt).abs() < 0.01, "std={std} ~!= {one_12_sqrt}",);
            }
        })
    });
}

pub fn test_vec_znx_seed_sampling_matches_source_wrappers<B: crate::test_suite::TestBackend>(
    _params: &TestParams,
    module: &Module<B>,
) where
    B::OwnedBuf: crate::layouts::HostDataMut,
    Module<B>: VecZnxFillUniformSourceBackend<B>
        + VecZnxFillUniformBackend<B>
        + VecZnxFillNormalSourceBackend<B>
        + VecZnxFillNormalBackend<B>,
{
    let n: usize = module.n();
    let base2k: usize = 17;
    let size: usize = 5;
    let cols: usize = 2;
    let col_i: usize = 1;

    let mut seed_source = Source::new([0u8; 32]);
    let seed = seed_source.new_seed();
    let mut wrapper_source = Source::new([0u8; 32]);

    let mut wrapper_uniform: VecZnx<_> = VecZnx::alloc(n, cols, size);
    let mut backend_uniform: VecZnx<_> = VecZnx::alloc(n, cols, size);
    module.vec_znx_fill_uniform_source_backend(
        base2k,
        &mut vec_znx_backend_mut::<B>(&mut wrapper_uniform),
        col_i,
        &mut wrapper_source,
    );
    module.vec_znx_fill_uniform_backend(base2k, &mut vec_znx_backend_mut::<B>(&mut backend_uniform), col_i, seed);
    assert_eq!(wrapper_uniform, backend_uniform);

    let noise_infos = NoiseInfos::new(2 * 17, 3.2, 6.0 * 3.2).unwrap();
    let mut seed_source = Source::new([1u8; 32]);
    let seed = seed_source.new_seed();
    let mut wrapper_source = Source::new([1u8; 32]);

    let mut wrapper_normal: VecZnx<_> = VecZnx::alloc(n, cols, size);
    let mut backend_normal: VecZnx<_> = VecZnx::alloc(n, cols, size);
    module.vec_znx_fill_normal_source_backend(
        base2k,
        &mut vec_znx_backend_mut::<B>(&mut wrapper_normal),
        col_i,
        noise_infos,
        &mut wrapper_source,
    );
    module.vec_znx_fill_normal_backend(
        base2k,
        &mut vec_znx_backend_mut::<B>(&mut backend_normal),
        col_i,
        noise_infos,
        seed,
    );
    assert_eq!(wrapper_normal, backend_normal);
}

pub fn test_scalar_znx_secret_seed_sampling_matches_source_wrappers<B: crate::test_suite::TestBackend>(
    _params: &TestParams,
    module: &Module<B>,
) where
    B::OwnedBuf: crate::layouts::HostDataMut,
    Module<B>: ScalarZnxFillTernaryHwSourceBackend<B>
        + ScalarZnxFillTernaryHwBackend<B>
        + ScalarZnxFillTernaryProbSourceBackend<B>
        + ScalarZnxFillTernaryProbBackend<B>
        + ScalarZnxFillBinaryHwSourceBackend<B>
        + ScalarZnxFillBinaryHwBackend<B>
        + ScalarZnxFillBinaryProbSourceBackend<B>
        + ScalarZnxFillBinaryProbBackend<B>
        + ScalarZnxFillBinaryBlockSourceBackend<B>
        + ScalarZnxFillBinaryBlockBackend<B>,
{
    let n: usize = module.n();
    let cols: usize = 2;
    let col_i: usize = 1;

    fn check<Fw, Fb>(
        seed_bytes: [u8; 32],
        n: usize,
        cols: usize,
        mut fill_wrapper: Fw,
        mut fill_backend: Fb,
    ) where
        Fw: FnMut(&mut ScalarZnx<Vec<u8>>, &mut Source),
        Fb: FnMut(&mut ScalarZnx<Vec<u8>>, [u8; 32]),
    {
        let mut seed_source = Source::new(seed_bytes);
        let seed = seed_source.new_seed();
        let mut wrapper_source = Source::new(seed_bytes);

        let mut wrapper = ScalarZnx::alloc(n, cols);
        let mut backend = ScalarZnx::alloc(n, cols);
        fill_wrapper(&mut wrapper, &mut wrapper_source);
        fill_backend(&mut backend, seed);
        assert_eq!(wrapper, backend);
    }

    check(
        [2u8; 32],
        n,
        cols,
        move |res, source| {
            module.scalar_znx_fill_ternary_hw_source_backend(
                &mut <ScalarZnx<Vec<u8>> as ScalarZnxToBackendMut<B>>::to_backend_mut(res),
                col_i,
                n / 8,
                source,
            )
        },
        move |res, seed| {
            module.scalar_znx_fill_ternary_hw_backend(
                &mut <ScalarZnx<Vec<u8>> as ScalarZnxToBackendMut<B>>::to_backend_mut(res),
                col_i,
                n / 8,
                seed,
            )
        },
    );
    check(
        [3u8; 32],
        n,
        cols,
        move |res, source| {
            module.scalar_znx_fill_ternary_prob_source_backend(
                &mut <ScalarZnx<Vec<u8>> as ScalarZnxToBackendMut<B>>::to_backend_mut(res),
                col_i,
                0.5,
                source,
            )
        },
        move |res, seed| {
            module.scalar_znx_fill_ternary_prob_backend(
                &mut <ScalarZnx<Vec<u8>> as ScalarZnxToBackendMut<B>>::to_backend_mut(res),
                col_i,
                0.5,
                seed,
            )
        },
    );
    check(
        [4u8; 32],
        n,
        cols,
        move |res, source| {
            module.scalar_znx_fill_binary_hw_source_backend(
                &mut <ScalarZnx<Vec<u8>> as ScalarZnxToBackendMut<B>>::to_backend_mut(res),
                col_i,
                n / 8,
                source,
            )
        },
        move |res, seed| {
            module.scalar_znx_fill_binary_hw_backend(
                &mut <ScalarZnx<Vec<u8>> as ScalarZnxToBackendMut<B>>::to_backend_mut(res),
                col_i,
                n / 8,
                seed,
            )
        },
    );
    check(
        [5u8; 32],
        n,
        cols,
        move |res, source| {
            module.scalar_znx_fill_binary_prob_source_backend(
                &mut <ScalarZnx<Vec<u8>> as ScalarZnxToBackendMut<B>>::to_backend_mut(res),
                col_i,
                0.5,
                source,
            )
        },
        move |res, seed| {
            module.scalar_znx_fill_binary_prob_backend(
                &mut <ScalarZnx<Vec<u8>> as ScalarZnxToBackendMut<B>>::to_backend_mut(res),
                col_i,
                0.5,
                seed,
            )
        },
    );
    check(
        [6u8; 32],
        n,
        cols,
        move |res, source| {
            module.scalar_znx_fill_binary_block_source_backend(
                &mut <ScalarZnx<Vec<u8>> as ScalarZnxToBackendMut<B>>::to_backend_mut(res),
                col_i,
                8,
                source,
            )
        },
        move |res, seed| {
            module.scalar_znx_fill_binary_block_backend(
                &mut <ScalarZnx<Vec<u8>> as ScalarZnxToBackendMut<B>>::to_backend_mut(res),
                col_i,
                8,
                seed,
            )
        },
    );
}

pub fn test_vec_znx_fill_normal<B: crate::test_suite::TestBackend>(_params: &TestParams, module: &Module<B>)
where
    B::OwnedBuf: crate::layouts::HostDataMut,
    Module<B>: VecZnxFillNormalSourceBackend<B>,
{
    let n: usize = module.n();
    let base2k: usize = 17;
    let size: usize = 5;
    let noise_infos = NoiseInfos::new(2 * 17, 3.2, 6.0 * 3.2).unwrap();
    let mut source_xe: Source = Source::new([0u8; 32]);
    let cols: usize = 2;
    let zero: Vec<i64> = vec![0; n];
    let k_f64: f64 = (1u64 << noise_infos.k as u64) as f64;
    (0..cols).for_each(|col_i| {
        let mut a: VecZnx<_> = VecZnx::alloc(n, cols, size);
        module.vec_znx_fill_normal_source_backend(
            base2k,
            &mut vec_znx_backend_mut::<B>(&mut a),
            col_i,
            noise_infos,
            &mut source_xe,
        );
        (0..cols).for_each(|col_j| {
            if col_j != col_i {
                (0..size).for_each(|limb_i| {
                    assert_eq!(a.at(col_j, limb_i), zero);
                })
            } else {
                let std: f64 = a.stats(base2k, col_i).std() * k_f64;
                assert!((std - noise_infos.sigma).abs() < 0.1, "std={std} ~!= {}", noise_infos.sigma);
            }
        })
    });
}

pub fn test_vec_znx_add_normal<B: crate::test_suite::TestBackend>(_params: &TestParams, module: &Module<B>)
where
    B::OwnedBuf: crate::layouts::HostDataMut,
    Module<B>: VecZnxFillNormalSourceBackend<B> + VecZnxAddNormalSourceBackend<B>,
{
    let n: usize = module.n();
    let base2k: usize = 17;
    let size: usize = 5;
    let noise_infos = NoiseInfos::new(2 * 17, 3.2, 6.0 * 3.2).unwrap();
    let mut source_xe: Source = Source::new([0u8; 32]);
    let cols: usize = 2;
    let zero: Vec<i64> = vec![0; n];
    let k_f64: f64 = (1u64 << noise_infos.k as u64) as f64;
    let sqrt2: f64 = SQRT_2;
    (0..cols).for_each(|col_i| {
        let mut a: VecZnx<_> = VecZnx::alloc(n, cols, size);
        module.vec_znx_fill_normal_source_backend(
            base2k,
            &mut vec_znx_backend_mut::<B>(&mut a),
            col_i,
            noise_infos,
            &mut source_xe,
        );
        module.vec_znx_add_normal_source_backend(
            base2k,
            &mut vec_znx_backend_mut::<B>(&mut a),
            col_i,
            noise_infos,
            &mut source_xe,
        );
        (0..cols).for_each(|col_j| {
            if col_j != col_i {
                (0..size).for_each(|limb_i| {
                    assert_eq!(a.at(col_j, limb_i), zero);
                })
            } else {
                let std: f64 = a.stats(base2k, col_i).std() * k_f64;
                assert!(
                    (std - noise_infos.sigma * sqrt2).abs() < 0.1,
                    "std={std} ~!= {}",
                    noise_infos.sigma * sqrt2
                );
            }
        })
    });
}

pub fn test_vec_znx_lsh<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>: VecZnxLshBackend<BR> + VecZnxLshTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    Module<BT>: VecZnxLshBackend<BT> + VecZnxLshTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_lsh_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_lsh_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest: u64 = a.digest_u64();

        for res_size in [1, 2, 3, 4] {
            for k in 0..res_size * base2k {
                let mut res_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
                let mut res_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

                // Set d to garbage
                res_ref.fill_uniform(base2k, &mut source);
                res_test.fill_uniform(base2k, &mut source);

                // Reference
                for i in 0..cols {
                    module_ref.vec_znx_lsh_backend(
                        base2k,
                        k,
                        &mut vec_znx_backend_mut::<BR>(&mut res_ref),
                        i,
                        &vec_znx_backend_ref::<BR>(&a),
                        i,
                        &mut scratch_ref.arena(),
                    );
                    module_test.vec_znx_lsh_backend(
                        base2k,
                        k,
                        &mut vec_znx_backend_mut::<BT>(&mut res_test),
                        i,
                        &vec_znx_backend_ref::<BT>(&a),
                        i,
                        &mut scratch_test.arena(),
                    );
                }

                assert_eq!(a.digest_u64(), a_digest);
                assert_eq!(res_ref, res_test);
            }
        }
    }
}

pub fn test_vec_znx_lsh_inplace<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>: VecZnxLshInplaceBackend<BR> + VecZnxLshTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    Module<BT>: VecZnxLshInplaceBackend<BT> + VecZnxLshTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_lsh_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_lsh_tmp_bytes());

    for res_size in [1, 2, 3, 4] {
        for k in 0..base2k * res_size {
            let mut res_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            res_ref.fill_uniform(base2k, &mut source);
            res_test.raw_mut().copy_from_slice(res_ref.raw());

            for i in 0..cols {
                module_ref.vec_znx_lsh_inplace_backend(
                    base2k,
                    k,
                    &mut vec_znx_backend_mut::<BR>(&mut res_ref),
                    i,
                    &mut scratch_ref.arena(),
                );
                module_test.vec_znx_lsh_inplace_backend(
                    base2k,
                    k,
                    &mut vec_znx_backend_mut::<BT>(&mut res_test),
                    i,
                    &mut scratch_test.arena(),
                );
            }

            assert_eq!(res_ref, res_test);
        }
    }
}

pub fn test_vec_znx_rsh<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>: VecZnxRshBackend<BR> + VecZnxRshTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    Module<BT>: VecZnxRshBackend<BT> + VecZnxRshTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_rsh_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_rsh_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest: u64 = a.digest_u64();

        for res_size in [1, 2, 3, 4] {
            for k in 0..res_size * base2k {
                let mut res_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
                let mut res_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

                // Set d to garbage
                res_ref.fill_uniform(base2k, &mut source);
                res_test.fill_uniform(base2k, &mut source);

                // Reference
                for i in 0..cols {
                    module_ref.vec_znx_rsh_backend(
                        base2k,
                        k,
                        &mut vec_znx_backend_mut::<BR>(&mut res_ref),
                        i,
                        &vec_znx_backend_ref::<BR>(&a),
                        i,
                        &mut scratch_ref.arena(),
                    );
                    module_test.vec_znx_rsh_backend(
                        base2k,
                        k,
                        &mut vec_znx_backend_mut::<BT>(&mut res_test),
                        i,
                        &vec_znx_backend_ref::<BT>(&a),
                        i,
                        &mut scratch_test.arena(),
                    );
                }

                assert_eq!(a.digest_u64(), a_digest);
                assert_eq!(res_ref, res_test);
            }
        }
    }
}

pub fn test_vec_znx_rsh_inplace<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>: VecZnxRshInplaceBackend<BR> + VecZnxRshTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    Module<BT>: VecZnxRshInplaceBackend<BT> + VecZnxRshTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_rsh_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_rsh_tmp_bytes());

    for res_size in [1, 2, 3, 4] {
        for k in 0..base2k * res_size {
            let mut res_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            res_ref.fill_uniform(base2k, &mut source);
            res_test.raw_mut().copy_from_slice(res_ref.raw());

            for i in 0..cols {
                module_ref.vec_znx_rsh_inplace_backend(
                    base2k,
                    k,
                    &mut vec_znx_backend_mut::<BR>(&mut res_ref),
                    i,
                    &mut scratch_ref.arena(),
                );
                module_test.vec_znx_rsh_inplace_backend(
                    base2k,
                    k,
                    &mut vec_znx_backend_mut::<BT>(&mut res_test),
                    i,
                    &mut scratch_test.arena(),
                );
            }

            assert_eq!(res_ref, res_test);
        }
    }
}

pub fn test_vec_znx_split_ring<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>: VecZnxSplitRingBackend<BR> + ModuleNew<BR> + VecZnxSplitRingTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    Module<BT>: VecZnxSplitRingBackend<BT> + ModuleNew<BT> + VecZnxSplitRingTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;
    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_split_ring_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_split_ring_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest = a.digest_u64();

        for res_size in [1, 2, 3, 4] {
            let mut res_ref: [VecZnx<Vec<u8>>; 2] =
                [VecZnx::alloc(n >> 1, cols, res_size), VecZnx::alloc(n >> 1, cols, res_size)];

            let mut res_test: [VecZnx<Vec<u8>>; 2] =
                [VecZnx::alloc(n >> 1, cols, res_size), VecZnx::alloc(n >> 1, cols, res_size)];

            res_ref.iter_mut().for_each(|ri| {
                ri.fill_uniform(base2k, &mut source);
            });

            res_test.iter_mut().for_each(|ri| {
                ri.fill_uniform(base2k, &mut source);
            });

            for i in 0..cols {
                let mut res_ref_backend: Vec<_> = res_ref.iter_mut().map(vec_znx_backend_mut::<BR>).collect();
                module_ref.vec_znx_split_ring_backend(
                    &mut res_ref_backend,
                    i,
                    &vec_znx_backend_ref::<BR>(&a),
                    i,
                    &mut scratch_ref.arena(),
                );
                let mut res_test_backend: Vec<_> = res_test.iter_mut().map(vec_znx_backend_mut::<BT>).collect();
                module_test.vec_znx_split_ring_backend(
                    &mut res_test_backend,
                    i,
                    &vec_znx_backend_ref::<BT>(&a),
                    i,
                    &mut scratch_test.arena(),
                );
            }

            assert_eq!(a.digest_u64(), a_digest);

            for (a, b) in res_ref.iter().zip(res_test.iter()) {
                assert_eq!(a, b);
            }
        }
    }
}

pub fn test_vec_znx_sub_scalar<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>: VecZnxSubScalarBackend<BR>,
    Module<BT>: VecZnxSubScalarBackend<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut a: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, cols);
    a.fill_uniform(base2k, &mut source);
    let a_digest: u64 = a.digest_u64();

    for b_size in [1, 2, 3, 4] {
        let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, b_size);
        b.fill_uniform(base2k, &mut source);
        let b_digest: u64 = b.digest_u64();

        for res_size in [1, 2, 3, 4] {
            let mut res_0: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_1: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            // Set d to garbage
            res_0.fill_uniform(base2k, &mut source);
            res_1.fill_uniform(base2k, &mut source);

            // Reference
            for i in 0..cols {
                module_ref.vec_znx_sub_scalar_backend(
                    &mut vec_znx_backend_mut::<BR>(&mut res_0),
                    i,
                    &scalar_znx_backend_ref::<BR>(&a),
                    i,
                    &vec_znx_backend_ref::<BR>(&b),
                    i,
                    (res_size.min(b_size)) - 1,
                );
                module_test.vec_znx_sub_scalar_backend(
                    &mut vec_znx_backend_mut::<BT>(&mut res_1),
                    i,
                    &scalar_znx_backend_ref::<BT>(&a),
                    i,
                    &vec_znx_backend_ref::<BT>(&b),
                    i,
                    (res_size.min(b_size)) - 1,
                );
            }

            assert_eq!(a.digest_u64(), a_digest);
            assert_eq!(b.digest_u64(), b_digest);
            assert_eq!(res_0, res_1);
        }
    }
}

pub fn test_vec_znx_sub_scalar_inplace<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>: VecZnxSubScalarInplaceBackend<BR>,
    Module<BT>: VecZnxSubScalarInplaceBackend<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut a: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, cols);
    a.fill_uniform(base2k, &mut source);
    let a_digest: u64 = a.digest_u64();

    for res_size in [1, 2, 3, 4] {
        let mut res_0: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
        let mut res_1: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

        res_0.fill_uniform(base2k, &mut source);
        res_1.raw_mut().copy_from_slice(res_0.raw());

        for i in 0..cols {
            module_ref.vec_znx_sub_scalar_inplace_backend(
                &mut vec_znx_backend_mut::<BR>(&mut res_0),
                i,
                res_size - 1,
                &scalar_znx_backend_ref::<BR>(&a),
                i,
            );
            module_test.vec_znx_sub_scalar_inplace_backend(
                &mut vec_znx_backend_mut::<BT>(&mut res_1),
                i,
                res_size - 1,
                &scalar_znx_backend_ref::<BT>(&a),
                i,
            );
        }

        assert_eq!(a.digest_u64(), a_digest);
        assert_eq!(res_0, res_1);
    }
}

pub fn test_vec_znx_sub<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>: VecZnxSubBackend<BR>,
    Module<BT>: VecZnxSubBackend<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest: u64 = a.digest_u64();

        for b_size in [1, 2, 3, 4] {
            let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, b_size);
            b.fill_uniform(base2k, &mut source);
            let b_digest: u64 = b.digest_u64();

            for res_size in [1, 2, 3, 4] {
                let mut res_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
                let mut res_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

                // Set d to garbage
                res_ref.fill_uniform(base2k, &mut source);
                res_test.fill_uniform(base2k, &mut source);

                // Reference
                for i in 0..cols {
                    module_test.vec_znx_sub_backend(
                        &mut vec_znx_backend_mut::<BT>(&mut res_ref),
                        i,
                        &vec_znx_backend_ref::<BT>(&a),
                        i,
                        &vec_znx_backend_ref::<BT>(&b),
                        i,
                    );
                    module_ref.vec_znx_sub_backend(
                        &mut vec_znx_backend_mut::<BR>(&mut res_test),
                        i,
                        &vec_znx_backend_ref::<BR>(&a),
                        i,
                        &vec_znx_backend_ref::<BR>(&b),
                        i,
                    );
                }

                assert_eq!(a.digest_u64(), a_digest);
                assert_eq!(b.digest_u64(), b_digest);

                assert_eq!(res_ref, res_test);
            }
        }
    }
}

pub fn test_vec_znx_sub_inplace<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>: VecZnxSubInplaceBackend<BR>,
    Module<BT>: VecZnxSubInplaceBackend<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest: u64 = a.digest_u64();

        for res_size in [1, 2, 3, 4] {
            let mut res_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            res_ref.fill_uniform(base2k, &mut source);
            res_test.raw_mut().copy_from_slice(res_ref.raw());

            for i in 0..cols {
                module_test.vec_znx_sub_inplace_backend(
                    &mut vec_znx_backend_mut::<BT>(&mut res_ref),
                    i,
                    &vec_znx_backend_ref::<BT>(&a),
                    i,
                );
                module_ref.vec_znx_sub_inplace_backend(
                    &mut vec_znx_backend_mut::<BR>(&mut res_test),
                    i,
                    &vec_znx_backend_ref::<BR>(&a),
                    i,
                );
            }

            assert_eq!(a.digest_u64(), a_digest);
            assert_eq!(res_ref, res_test);
        }
    }
}

pub fn test_vec_znx_sub_negate_inplace<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>: VecZnxSubNegateInplaceBackend<BR>,
    Module<BT>: VecZnxSubNegateInplaceBackend<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest: u64 = a.digest_u64();

        for res_size in [1, 2, 3, 4] {
            let mut res_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            res_ref.fill_uniform(base2k, &mut source);
            res_test.raw_mut().copy_from_slice(res_ref.raw());

            for i in 0..cols {
                module_test.vec_znx_sub_negate_inplace_backend(
                    &mut vec_znx_backend_mut::<BT>(&mut res_ref),
                    i,
                    &vec_znx_backend_ref::<BT>(&a),
                    i,
                );
                module_ref.vec_znx_sub_negate_inplace_backend(
                    &mut vec_znx_backend_mut::<BR>(&mut res_test),
                    i,
                    &vec_znx_backend_ref::<BR>(&a),
                    i,
                );
            }

            assert_eq!(a.digest_u64(), a_digest);
            assert_eq!(res_ref, res_test);
        }
    }
}

pub fn test_vec_znx_switch_ring<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR::OwnedBuf: crate::layouts::HostDataMut,
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BR>: VecZnxSwitchRingBackend<BR>,
    Module<BT>: VecZnxSwitchRingBackend<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);

        // Fill a with random i64
        a.fill_uniform(base2k, &mut source);
        let a_digest: u64 = a.digest_u64();

        for res_size in [1, 2, 3, 4] {
            {
                let mut res_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n << 1, cols, res_size);
                let mut res_test: VecZnx<Vec<u8>> = VecZnx::alloc(n << 1, cols, res_size);

                res_ref.fill_uniform(base2k, &mut source);
                res_test.fill_uniform(base2k, &mut source);

                // Normalize on c
                for i in 0..cols {
                    module_ref.vec_znx_switch_ring_backend(
                        &mut vec_znx_backend_mut::<BR>(&mut res_ref),
                        i,
                        &vec_znx_backend_ref::<BR>(&a),
                        i,
                    );
                    module_test.vec_znx_switch_ring_backend(
                        &mut vec_znx_backend_mut::<BT>(&mut res_test),
                        i,
                        &vec_znx_backend_ref::<BT>(&a),
                        i,
                    );
                }

                assert_eq!(a.digest_u64(), a_digest);
                assert_eq!(res_ref, res_test);
            }

            {
                let mut res_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n >> 1, cols, res_size);
                let mut res_test: VecZnx<Vec<u8>> = VecZnx::alloc(n >> 1, cols, res_size);

                res_ref.fill_uniform(base2k, &mut source);
                res_test.fill_uniform(base2k, &mut source);

                // Normalize on c
                for i in 0..cols {
                    module_ref.vec_znx_switch_ring_backend(
                        &mut vec_znx_backend_mut::<BR>(&mut res_ref),
                        i,
                        &vec_znx_backend_ref::<BR>(&a),
                        i,
                    );
                    module_test.vec_znx_switch_ring_backend(
                        &mut vec_znx_backend_mut::<BT>(&mut res_test),
                        i,
                        &vec_znx_backend_ref::<BT>(&a),
                        i,
                    );
                }

                assert_eq!(a.digest_u64(), a_digest);
                assert_eq!(res_ref, res_test);
            }
        }
    }
}

pub fn test_vec_znx_switch_ring_backend_matches_wrapper<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    _module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BT::OwnedBuf: crate::layouts::HostDataMut,
    Module<BT>: VecZnxSwitchRingBackend<BT>,
{
    let base2k = params.base2k;
    let n: usize = module_test.n();
    let cols: usize = 2;
    let a_col: usize = 0;
    let res_col: usize = 1;
    let mut source: Source = Source::new([4u8; 32]);

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(base2k, &mut source);

        for res_n in [n >> 1, n << 1] {
            for res_size in [1, 2, 3, 4] {
                let mut wrapper: VecZnx<Vec<u8>> = VecZnx::alloc(res_n, cols, res_size);
                let mut backend: VecZnx<Vec<u8>> = VecZnx::alloc(res_n, cols, res_size);
                wrapper.fill_uniform(base2k, &mut source);
                backend.data.copy_from_slice(&wrapper.data);

                module_test.vec_znx_switch_ring_backend(
                    &mut vec_znx_backend_mut::<BT>(&mut wrapper),
                    res_col,
                    &vec_znx_backend_ref::<BT>(&a),
                    a_col,
                );
                module_test.vec_znx_switch_ring_backend(
                    &mut vec_znx_backend_mut::<BT>(&mut backend),
                    res_col,
                    &vec_znx_backend_ref::<BT>(&a),
                    a_col,
                );

                assert_eq!(wrapper, backend);
            }
        }
    }
}
