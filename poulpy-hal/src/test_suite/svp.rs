use rand::RngCore;

use crate::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyDft, SvpApplyDftToDft, SvpApplyDftToDftAdd, SvpApplyDftToDftInplace,
        SvpPPolAlloc, SvpPrepare, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftAlloc, VecZnxDftApply,
        VecZnxIdftApplyConsume,
    },
    layouts::{Backend, DataViewMut, DigestU64, FillUniform, Module, ScalarZnx, ScratchOwned, SvpPPol, VecZnx, VecZnxDft},
    source::Source,
};

pub fn test_svp_apply_dft<BR: Backend, BT: Backend>(basek: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: SvpPrepare<BR>
        + SvpApplyDft<BR>
        + SvpPPolAlloc<BR>
        + VecZnxDftAlloc<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxIdftApplyConsume<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: SvpPrepare<BT>
        + SvpApplyDft<BT>
        + SvpPPolAlloc<BT>
        + VecZnxDftAlloc<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxIdftApplyConsume<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let cols: usize = 2;

    let mut source: Source = Source::new([0u8; 32]);

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    let mut scalar: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, cols);
    scalar.fill_uniform(basek, &mut source);

    let scalar_digest: u64 = scalar.digest_u64();

    let mut svp_ref: SvpPPol<Vec<u8>, BR> = module_ref.svp_ppol_alloc(cols);
    let mut svp_test: SvpPPol<Vec<u8>, BT> = module_test.svp_ppol_alloc(cols);

    for j in 0..cols {
        module_ref.svp_prepare(&mut svp_ref, j, &scalar, j);
        module_test.svp_prepare(&mut svp_test, j, &scalar, j);
    }

    assert_eq!(scalar.digest_u64(), scalar_digest);

    let svp_ref_digest: u64 = svp_ref.digest_u64();
    let svp_test_digest: u64 = svp_test.digest_u64();

    for a_size in [1, 2, 3, 4] {
        // Create a random input VecZnx
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(basek, &mut source);

        let a_digest: u64 = a.digest_u64();

        for res_size in [1, 2, 3, 4] {
            // Allocate VecZnxDft from FFT64Ref and module to test
            let mut res_dft_ref: VecZnxDft<Vec<u8>, BR> = module_ref.vec_znx_dft_alloc(cols, res_size);
            let mut res_dft_test: VecZnxDft<Vec<u8>, BT> = module_test.vec_znx_dft_alloc(cols, res_size);

            // Fill output with garbage
            source.fill_bytes(res_dft_ref.data_mut());
            source.fill_bytes(res_dft_test.data_mut());

            for j in 0..cols {
                module_ref.svp_apply_dft(&mut res_dft_ref, j, &svp_ref, j, &a, j);
                module_test.svp_apply_dft(&mut res_dft_test, j, &svp_test, j, &a, j);
            }

            // Assert no change to inputs
            assert_eq!(svp_ref.digest_u64(), svp_ref_digest);
            assert_eq!(svp_test.digest_u64(), svp_test_digest);
            assert_eq!(a.digest_u64(), a_digest);

            let res_big_ref: crate::layouts::VecZnxBig<Vec<u8>, BR> = module_ref.vec_znx_idft_apply_consume(res_dft_ref);
            let res_big_test: crate::layouts::VecZnxBig<Vec<u8>, BT> = module_test.vec_znx_idft_apply_consume(res_dft_test);

            let mut res_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            for j in 0..cols {
                module_ref.vec_znx_big_normalize(
                    basek,
                    &mut res_ref,
                    j,
                    &res_big_ref,
                    j,
                    scratch_ref.borrow(),
                );
                module_test.vec_znx_big_normalize(
                    basek,
                    &mut res_test,
                    j,
                    &res_big_test,
                    j,
                    scratch_test.borrow(),
                );
            }

            assert_eq!(res_ref, res_test);
        }
    }
}

pub fn test_svp_apply_dft_to_dft<BR: Backend, BT: Backend>(basek: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: SvpPrepare<BR>
        + SvpApplyDftToDft<BR>
        + SvpPPolAlloc<BR>
        + VecZnxDftAlloc<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxDftApply<BR>
        + VecZnxIdftApplyConsume<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: SvpPrepare<BT>
        + SvpApplyDftToDft<BT>
        + SvpPPolAlloc<BT>
        + VecZnxDftAlloc<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxDftApply<BT>
        + VecZnxIdftApplyConsume<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let cols: usize = 2;

    let mut source: Source = Source::new([0u8; 32]);

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    let mut scalar: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, cols);
    scalar.fill_uniform(basek, &mut source);

    let scalar_digest: u64 = scalar.digest_u64();

    let mut svp_ref: SvpPPol<Vec<u8>, BR> = module_ref.svp_ppol_alloc(cols);
    let mut svp_test: SvpPPol<Vec<u8>, BT> = module_test.svp_ppol_alloc(cols);

    for j in 0..cols {
        module_ref.svp_prepare(&mut svp_ref, j, &scalar, j);
        module_test.svp_prepare(&mut svp_test, j, &scalar, j);
    }

    assert_eq!(scalar.digest_u64(), scalar_digest);

    let svp_ref_digest: u64 = svp_ref.digest_u64();
    let svp_test_digest: u64 = svp_test.digest_u64();

    for a_size in [3] {
        // Create a random input VecZnx
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(basek, &mut source);

        let a_digest: u64 = a.digest_u64();

        let mut a_dft_ref: VecZnxDft<Vec<u8>, BR> = module_ref.vec_znx_dft_alloc(cols, a_size);
        let mut a_dft_test: VecZnxDft<Vec<u8>, BT> = module_test.vec_znx_dft_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_dft_apply(1, 0, &mut a_dft_ref, j, &a, j);
            module_test.vec_znx_dft_apply(1, 0, &mut a_dft_test, j, &a, j);
        }

        assert_eq!(a.digest_u64(), a_digest);

        let a_dft_ref_digest: u64 = a_dft_ref.digest_u64();
        let a_dft_test_digest: u64 = a_dft_test.digest_u64();

        for res_size in [3] {
            // Allocate VecZnxDft from FFT64Ref and module to test
            let mut res_dft_ref: VecZnxDft<Vec<u8>, BR> = module_ref.vec_znx_dft_alloc(cols, res_size);
            let mut res_dft_test: VecZnxDft<Vec<u8>, BT> = module_test.vec_znx_dft_alloc(cols, res_size);

            // Fill output with garbage
            source.fill_bytes(res_dft_ref.data_mut());
            source.fill_bytes(res_dft_test.data_mut());

            for j in 0..cols {
                module_ref.svp_apply_dft_to_dft(&mut res_dft_ref, j, &svp_ref, j, &a_dft_ref, j);
                module_test.svp_apply_dft_to_dft(&mut res_dft_test, j, &svp_test, j, &a_dft_test, j);
            }

            // Assert no change to inputs
            assert_eq!(a_dft_ref.digest_u64(), a_dft_ref_digest);
            assert_eq!(a_dft_test.digest_u64(), a_dft_test_digest);
            assert_eq!(svp_ref.digest_u64(), svp_ref_digest);
            assert_eq!(svp_test.digest_u64(), svp_test_digest);
            assert_eq!(a.digest_u64(), a_digest);

            let res_big_ref: crate::layouts::VecZnxBig<Vec<u8>, BR> = module_ref.vec_znx_idft_apply_consume(res_dft_ref);
            let res_big_test: crate::layouts::VecZnxBig<Vec<u8>, BT> = module_test.vec_znx_idft_apply_consume(res_dft_test);

            let mut res_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            for j in 0..cols {
                module_ref.vec_znx_big_normalize(
                    basek,
                    &mut res_ref,
                    j,
                    &res_big_ref,
                    j,
                    scratch_ref.borrow(),
                );
                module_test.vec_znx_big_normalize(
                    basek,
                    &mut res_test,
                    j,
                    &res_big_test,
                    j,
                    scratch_test.borrow(),
                );
            }

            assert_eq!(res_ref, res_test);
        }
    }
}

pub fn test_svp_apply_dft_to_dft_add<BR: Backend, BT: Backend>(basek: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: SvpPrepare<BR>
        + SvpApplyDftToDftAdd<BR>
        + SvpPPolAlloc<BR>
        + VecZnxDftAlloc<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxDftApply<BR>
        + VecZnxIdftApplyConsume<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: SvpPrepare<BT>
        + SvpApplyDftToDftAdd<BT>
        + SvpPPolAlloc<BT>
        + VecZnxDftAlloc<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxDftApply<BT>
        + VecZnxIdftApplyConsume<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let cols: usize = 2;

    let mut source: Source = Source::new([0u8; 32]);

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    let mut scalar: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, cols);
    scalar.fill_uniform(basek, &mut source);

    let scalar_digest: u64 = scalar.digest_u64();

    let mut svp_ref: SvpPPol<Vec<u8>, BR> = module_ref.svp_ppol_alloc(cols);
    let mut svp_test: SvpPPol<Vec<u8>, BT> = module_test.svp_ppol_alloc(cols);

    for j in 0..cols {
        module_ref.svp_prepare(&mut svp_ref, j, &scalar, j);
        module_test.svp_prepare(&mut svp_test, j, &scalar, j);
    }

    assert_eq!(scalar.digest_u64(), scalar_digest);

    let svp_ref_digest: u64 = svp_ref.digest_u64();
    let svp_test_digest: u64 = svp_test.digest_u64();

    for a_size in [1, 2, 3, 4] {
        // Create a random input VecZnx
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);
        a.fill_uniform(basek, &mut source);

        let a_digest: u64 = a.digest_u64();

        let mut a_dft_ref: VecZnxDft<Vec<u8>, BR> = module_ref.vec_znx_dft_alloc(cols, a_size);
        let mut a_dft_test: VecZnxDft<Vec<u8>, BT> = module_test.vec_znx_dft_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_dft_apply(1, 0, &mut a_dft_ref, j, &a, j);
            module_test.vec_znx_dft_apply(1, 0, &mut a_dft_test, j, &a, j);
        }

        let a_dft_ref_digest: u64 = a_dft_ref.digest_u64();
        let a_dft_test_digest: u64 = a_dft_test.digest_u64();

        for res_size in [1, 2, 3, 4] {
            let mut res: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            res.fill_uniform(basek, &mut source);

            let mut res_dft_ref: VecZnxDft<Vec<u8>, BR> = module_ref.vec_znx_dft_alloc(cols, res_size);
            let mut res_dft_test: VecZnxDft<Vec<u8>, BT> = module_test.vec_znx_dft_alloc(cols, res_size);

            for j in 0..cols {
                module_ref.vec_znx_dft_apply(1, 0, &mut res_dft_ref, j, &res, j);
                module_test.vec_znx_dft_apply(1, 0, &mut res_dft_test, j, &res, j);
            }

            // Fill output with garbage
            source.fill_bytes(res_dft_ref.data_mut());
            source.fill_bytes(res_dft_test.data_mut());

            for j in 0..cols {
                module_ref.svp_apply_dft_to_dft_add(&mut res_dft_ref, j, &svp_ref, j, &a_dft_ref, j);
                module_test.svp_apply_dft_to_dft_add(&mut res_dft_test, j, &svp_test, j, &a_dft_test, j);
            }

            // Assert no change to inputs
            assert_eq!(a_dft_ref.digest_u64(), a_dft_ref_digest);
            assert_eq!(a_dft_test.digest_u64(), a_dft_test_digest);
            assert_eq!(svp_ref.digest_u64(), svp_ref_digest);
            assert_eq!(svp_test.digest_u64(), svp_test_digest);
            assert_eq!(a.digest_u64(), a_digest);

            let res_big_ref: crate::layouts::VecZnxBig<Vec<u8>, BR> = module_ref.vec_znx_idft_apply_consume(res_dft_ref);
            let res_big_test: crate::layouts::VecZnxBig<Vec<u8>, BT> = module_test.vec_znx_idft_apply_consume(res_dft_test);

            let mut res_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
            let mut res_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

            for j in 0..cols {
                module_ref.vec_znx_big_normalize(
                    basek,
                    &mut res_ref,
                    j,
                    &res_big_ref,
                    j,
                    scratch_ref.borrow(),
                );
                module_test.vec_znx_big_normalize(
                    basek,
                    &mut res_test,
                    j,
                    &res_big_test,
                    j,
                    scratch_test.borrow(),
                );
            }

            assert_eq!(res_ref, res_test);
        }
    }
}

pub fn test_svp_apply_dft_to_dft_inplace<BR: Backend, BT: Backend>(
    basek: usize,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: SvpPrepare<BR>
        + SvpApplyDftToDftInplace<BR>
        + SvpPPolAlloc<BR>
        + VecZnxDftAlloc<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxDftApply<BR>
        + VecZnxIdftApplyConsume<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: SvpPrepare<BT>
        + SvpApplyDftToDftInplace<BT>
        + SvpPPolAlloc<BT>
        + VecZnxDftAlloc<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxDftApply<BT>
        + VecZnxIdftApplyConsume<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let cols: usize = 2;

    let mut source: Source = Source::new([0u8; 32]);

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    let mut scalar: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, cols);
    scalar.fill_uniform(basek, &mut source);

    let scalar_digest: u64 = scalar.digest_u64();

    let mut svp_ref: SvpPPol<Vec<u8>, BR> = module_ref.svp_ppol_alloc(cols);
    let mut svp_test: SvpPPol<Vec<u8>, BT> = module_test.svp_ppol_alloc(cols);

    for j in 0..cols {
        module_ref.svp_prepare(&mut svp_ref, j, &scalar, j);
        module_test.svp_prepare(&mut svp_test, j, &scalar, j);
    }

    assert_eq!(scalar.digest_u64(), scalar_digest);

    let svp_ref_digest: u64 = svp_ref.digest_u64();
    let svp_test_digest: u64 = svp_test.digest_u64();

    for res_size in [1, 2, 3, 4] {
        let mut res: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
        res.fill_uniform(basek, &mut source);
        let res_digest: u64 = res.digest_u64();

        let mut res_dft_ref: VecZnxDft<Vec<u8>, BR> = module_ref.vec_znx_dft_alloc(cols, res_size);
        let mut res_dft_test: VecZnxDft<Vec<u8>, BT> = module_test.vec_znx_dft_alloc(cols, res_size);

        for j in 0..cols {
            module_ref.vec_znx_dft_apply(1, 0, &mut res_dft_ref, j, &res, j);
            module_test.vec_znx_dft_apply(1, 0, &mut res_dft_test, j, &res, j);
        }

        assert_eq!(res.digest_u64(), res_digest);

        for j in 0..cols {
            module_ref.svp_apply_dft_to_dft_inplace(&mut res_dft_ref, j, &svp_ref, j);
            module_test.svp_apply_dft_to_dft_inplace(&mut res_dft_test, j, &svp_test, j);
        }

        // Assert no change to inputs
        assert_eq!(svp_ref.digest_u64(), svp_ref_digest);
        assert_eq!(svp_test.digest_u64(), svp_test_digest);

        let res_big_ref: crate::layouts::VecZnxBig<Vec<u8>, BR> = module_ref.vec_znx_idft_apply_consume(res_dft_ref);
        let res_big_test: crate::layouts::VecZnxBig<Vec<u8>, BT> = module_test.vec_znx_idft_apply_consume(res_dft_test);

        let mut res_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
        let mut res_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

        for j in 0..cols {
            module_ref.vec_znx_big_normalize(
                basek,
                &mut res_ref,
                j,
                &res_big_ref,
                j,
                scratch_ref.borrow(),
            );
            module_test.vec_znx_big_normalize(
                basek,
                &mut res_test,
                j,
                &res_big_test,
                j,
                scratch_test.borrow(),
            );
        }

        assert_eq!(res_ref, res_test);
    }
}
