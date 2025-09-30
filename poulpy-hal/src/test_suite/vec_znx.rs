use std::f64::consts::SQRT_2;

use crate::{
    api::{
        ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAdd, VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalar,
        VecZnxAddScalarInplace, VecZnxAutomorphism, VecZnxAutomorphismInplace, VecZnxAutomorphismInplaceTmpBytes, VecZnxCopy,
        VecZnxFillNormal, VecZnxFillUniform, VecZnxLsh, VecZnxLshInplace, VecZnxLshTmpBytes, VecZnxMergeRings,
        VecZnxMergeRingsTmpBytes, VecZnxMulXpMinusOne, VecZnxMulXpMinusOneInplace, VecZnxMulXpMinusOneInplaceTmpBytes,
        VecZnxNegate, VecZnxNegateInplace, VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxRotate,
        VecZnxRotateInplace, VecZnxRotateInplaceTmpBytes, VecZnxRsh, VecZnxRshInplace, VecZnxSplitRing, VecZnxSplitRingTmpBytes,
        VecZnxSub, VecZnxSubInplace, VecZnxSubNegateInplace, VecZnxSubScalar, VecZnxSubScalarInplace, VecZnxSwitchRing,
    },
    layouts::{Backend, DigestU64, FillUniform, Module, ScalarZnx, ScratchOwned, VecZnx, ZnxInfos, ZnxView, ZnxViewMut},
    reference::znx::znx_copy_ref,
    source::Source,
};

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

pub fn test_vec_znx_add_scalar<BR: Backend, BT: Backend>(base2k: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: VecZnxAddScalar,
    Module<BT>: VecZnxAddScalar,
{
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
                module_ref.vec_znx_add_scalar(&mut rest_ref, i, &a, i, &b, i, (res_size.min(a_size)) - 1);
                module_test.vec_znx_add_scalar(&mut res_test, i, &a, i, &b, i, (res_size.min(a_size)) - 1);
            }

            assert_eq!(b.digest_u64(), b_digest);
            assert_eq!(a.digest_u64(), a_digest);
            assert_eq!(rest_ref, res_test);
        }
    }
}

pub fn test_vec_znx_add_scalar_inplace<BR: Backend, BT: Backend>(base2k: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: VecZnxAddScalarInplace,
    Module<BT>: VecZnxAddScalarInplace,
{
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
            module_ref.vec_znx_add_scalar_inplace(&mut rest_ref, i, res_size - 1, &b, i);
            module_test.vec_znx_add_scalar_inplace(&mut res_test, i, res_size - 1, &b, i);
        }

        assert_eq!(b.digest_u64(), b_digest);
        assert_eq!(rest_ref, res_test);
    }
}
pub fn test_vec_znx_add<BR: Backend, BT: Backend>(base2k: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: VecZnxAdd,
    Module<BT>: VecZnxAdd,
{
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
                    module_test.vec_znx_add(&mut res_ref, i, &a, i, &b, i);
                    module_ref.vec_znx_add(&mut res_test, i, &a, i, &b, i);
                }

                assert_eq!(a.digest_u64(), a_digest);
                assert_eq!(b.digest_u64(), b_digest);

                assert_eq!(res_ref, res_test);
            }
        }
    }
}

pub fn test_vec_znx_add_inplace<BR: Backend, BT: Backend>(base2k: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: VecZnxAddInplace,
    Module<BT>: VecZnxAddInplace,
{
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
                module_ref.vec_znx_add_inplace(&mut res_ref, i, &a, i);
                module_test.vec_znx_add_inplace(&mut res_test, i, &a, i);
            }

            assert_eq!(a.digest_u64(), a_digest);
            assert_eq!(res_ref, res_test);
        }
    }
}

pub fn test_vec_znx_automorphism<BR: Backend, BT: Backend>(base2k: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: VecZnxAutomorphism,
    Module<BT>: VecZnxAutomorphism,
{
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
                module_ref.vec_znx_automorphism(p, &mut res_ref, i, &a, i);
                module_test.vec_znx_automorphism(p, &mut res_test, i, &a, i);
            }

            assert_eq!(a.digest_u64(), a_digest);
            assert_eq!(res_ref, res_test);

            let p: i64 = 5;

            // Normalize on c
            for i in 0..cols {
                module_ref.vec_znx_automorphism(p, &mut res_ref, i, &a, i);
                module_test.vec_znx_automorphism(p, &mut res_test, i, &a, i);
            }

            assert_eq!(a.digest_u64(), a_digest);
            assert_eq!(res_ref, res_test);
        }
    }
}

pub fn test_vec_znx_automorphism_inplace<BR: Backend, BT: Backend>(
    base2k: usize,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxAutomorphismInplace<BR> + VecZnxAutomorphismInplaceTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    Module<BT>: VecZnxAutomorphismInplace<BT> + VecZnxAutomorphismInplaceTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_automorphism_inplace_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_automorphism_inplace_tmp_bytes());

    for size in [1, 2, 3, 4] {
        let mut res_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);
        let mut res_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);

        // Fill a with random i64
        res_ref.fill_uniform(base2k, &mut source);
        znx_copy_ref(res_test.raw_mut(), res_ref.raw());

        let p: i64 = -7;

        // Normalize on c
        for i in 0..cols {
            module_ref.vec_znx_automorphism_inplace(p, &mut res_ref, i, scratch_ref.borrow());
            module_test.vec_znx_automorphism_inplace(p, &mut res_test, i, scratch_test.borrow());
        }

        assert_eq!(res_ref, res_test);

        let p: i64 = 7;

        // Normalize on c
        for i in 0..cols {
            module_ref.vec_znx_automorphism_inplace(p, &mut res_ref, i, scratch_ref.borrow());
            module_test.vec_znx_automorphism_inplace(p, &mut res_test, i, scratch_test.borrow());
        }

        assert_eq!(res_ref, res_test);
    }
}

pub fn test_vec_znx_copy<BR: Backend, BT: Backend>(base2k: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: VecZnxCopy,
    Module<BT>: VecZnxCopy,
{
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
                module_ref.vec_znx_copy(&mut res_0, i, &a, i);
                module_ref.vec_znx_copy(&mut res_1, i, &a, i);
            }

            assert_eq!(a.digest_u64(), a_digest);
            assert_eq!(res_0, res_1);
        }
    }
}

pub fn test_vec_znx_merge_rings<BR: Backend, BT: Backend>(base2k: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: VecZnxMergeRings<BR> + ModuleNew<BR> + VecZnxMergeRingsTmpBytes,
    Module<BT>: VecZnxMergeRings<BT> + ModuleNew<BT> + VecZnxMergeRingsTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;
    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_merge_rings_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_merge_rings_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: [VecZnx<Vec<u8>>; 2] = [
            VecZnx::alloc(n >> 1, cols, a_size),
            VecZnx::alloc(n >> 1, cols, a_size),
        ];

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
                module_ref.vec_znx_merge_rings(&mut res_test, i, &a, i, scratch_ref.borrow());
                module_test.vec_znx_merge_rings(&mut res_ref, i, &a, i, scratch_test.borrow());
            }

            assert_eq!([a[0].digest_u64(), a[1].digest_u64()], a_digests);
            assert_eq!(res_ref, res_test);
        }
    }
}

pub fn test_vec_znx_mul_xp_minus_one<BR: Backend, BT: Backend>(base2k: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: VecZnxMulXpMinusOne,
    Module<BT>: VecZnxMulXpMinusOne,
{
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
                module_ref.vec_znx_mul_xp_minus_one(p, &mut res_ref, i, &a, i);
                module_test.vec_znx_mul_xp_minus_one(p, &mut res_test, i, &a, i);
            }

            assert_eq!(a.digest_u64(), a_digest);
            assert_eq!(res_test, res_ref);

            let p: i64 = 5;

            // Normalize on c
            for i in 0..cols {
                module_ref.vec_znx_mul_xp_minus_one(p, &mut res_ref, i, &a, i);
                module_test.vec_znx_mul_xp_minus_one(p, &mut res_test, i, &a, i);
            }

            assert_eq!(a.digest_u64(), a_digest);
            assert_eq!(res_test, res_ref);
        }
    }
}

pub fn test_vec_znx_mul_xp_minus_one_inplace<BR: Backend, BT: Backend>(
    base2k: usize,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxMulXpMinusOneInplace<BR> + VecZnxMulXpMinusOneInplaceTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    Module<BT>: VecZnxMulXpMinusOneInplace<BT> + VecZnxMulXpMinusOneInplaceTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_mul_xp_minus_one_inplace_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_mul_xp_minus_one_inplace_tmp_bytes());

    for size in [1, 2, 3, 4] {
        let mut res_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);
        let mut res_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);

        // Fill a with random i64
        res_ref.fill_uniform(base2k, &mut source);
        znx_copy_ref(res_test.raw_mut(), res_ref.raw());

        let p: i64 = -7;

        for i in 0..cols {
            module_ref.vec_znx_mul_xp_minus_one_inplace(p, &mut res_ref, i, scratch_ref.borrow());
            module_test.vec_znx_mul_xp_minus_one_inplace(p, &mut res_test, i, scratch_test.borrow());
        }

        assert_eq!(res_ref, res_test);

        let p: i64 = 7;

        for i in 0..cols {
            module_ref.vec_znx_mul_xp_minus_one_inplace(p, &mut res_ref, i, scratch_ref.borrow());
            module_test.vec_znx_mul_xp_minus_one_inplace(p, &mut res_test, i, scratch_test.borrow());
        }

        assert_eq!(res_ref, res_test);
    }
}

pub fn test_vec_znx_negate<BR: Backend, BT: Backend>(base2k: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: VecZnxNegate,
    Module<BT>: VecZnxNegate,
{
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
                module_ref.vec_znx_negate(&mut res_ref, i, &a, i);
                module_test.vec_znx_negate(&mut res_test, i, &a, i);
            }

            assert_eq!(a.digest_u64(), a_digest);
            assert_eq!(res_ref, res_test);
        }
    }
}

pub fn test_vec_znx_negate_inplace<BR: Backend, BT: Backend>(base2k: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: VecZnxNegateInplace,
    Module<BT>: VecZnxNegateInplace,
{
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
            module_ref.vec_znx_negate_inplace(&mut res_ref, i);
            module_test.vec_znx_negate_inplace(&mut res_test, i);
        }

        assert_eq!(res_ref, res_test);
    }
}

pub fn test_vec_znx_normalize<BR: Backend, BT: Backend>(base2k: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: VecZnxNormalize<BR> + VecZnxNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    Module<BT>: VecZnxNormalize<BT> + VecZnxNormalizeTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
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

            // Set d to garbage
            res_ref.fill_uniform(base2k, &mut source);
            res_test.fill_uniform(base2k, &mut source);

            // Reference
            for i in 0..cols {
                module_ref.vec_znx_normalize(base2k, &mut res_ref, i, base2k, &a, i, scratch_ref.borrow());
                module_test.vec_znx_normalize(
                    base2k,
                    &mut res_test,
                    i,
                    base2k,
                    &a,
                    i,
                    scratch_test.borrow(),
                );
            }

            assert_eq!(a.digest_u64(), a_digest);
            assert_eq!(res_ref, res_test);
        }
    }
}

pub fn test_vec_znx_normalize_inplace<BR: Backend, BT: Backend>(base2k: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: VecZnxNormalizeInplace<BR> + VecZnxNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    Module<BT>: VecZnxNormalizeInplace<BT> + VecZnxNormalizeTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
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
            module_ref.vec_znx_normalize_inplace(base2k, &mut res_ref, i, scratch_ref.borrow());
            module_test.vec_znx_normalize_inplace(base2k, &mut res_test, i, scratch_test.borrow());
        }

        assert_eq!(res_ref, res_test);
    }
}

pub fn test_vec_znx_rotate<BR: Backend, BT: Backend>(base2k: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: VecZnxRotate,
    Module<BT>: VecZnxRotate,
{
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
                module_ref.vec_znx_rotate(p, &mut res_ref, i, &a, i);
                module_test.vec_znx_rotate(p, &mut res_test, i, &a, i);
            }

            assert_eq!(a.digest_u64(), a_digest);
            assert_eq!(res_ref, res_test);

            let p: i64 = 5;

            // Normalize on c
            for i in 0..cols {
                module_ref.vec_znx_rotate(p, &mut res_ref, i, &a, i);
                module_test.vec_znx_rotate(p, &mut res_test, i, &a, i);
            }

            assert_eq!(a.digest_u64(), a_digest);
            assert_eq!(res_ref, res_test);
        }
    }
}

pub fn test_vec_znx_rotate_inplace<BR: Backend, BT: Backend>(base2k: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: VecZnxRotateInplace<BR> + VecZnxRotateInplaceTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    Module<BT>: VecZnxRotateInplace<BT> + VecZnxRotateInplaceTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_rotate_inplace_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_rotate_inplace_tmp_bytes());

    for size in [1, 2, 3, 4] {
        let mut res_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);
        let mut res_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);

        // Fill a with random i64
        res_ref.fill_uniform(base2k, &mut source);
        znx_copy_ref(res_test.raw_mut(), res_ref.raw());

        let p: i64 = -5;

        // Normalize on c
        for i in 0..cols {
            module_ref.vec_znx_rotate_inplace(p, &mut res_ref, i, scratch_ref.borrow());
            module_test.vec_znx_rotate_inplace(p, &mut res_test, i, scratch_test.borrow());
        }

        assert_eq!(res_ref, res_test);

        let p: i64 = 5;

        // Normalize on c
        for i in 0..cols {
            module_ref.vec_znx_rotate_inplace(p, &mut res_ref, i, scratch_ref.borrow());
            module_test.vec_znx_rotate_inplace(p, &mut res_test, i, scratch_test.borrow());
        }

        assert_eq!(res_ref, res_test);
    }
}

pub fn test_vec_znx_fill_uniform<B: Backend>(module: &Module<B>)
where
    Module<B>: VecZnxFillUniform,
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
        module.vec_znx_fill_uniform(base2k, &mut a, col_i, &mut source);
        (0..cols).for_each(|col_j| {
            if col_j != col_i {
                (0..size).for_each(|limb_i| {
                    assert_eq!(a.at(col_j, limb_i), zero);
                })
            } else {
                let std: f64 = a.std(base2k, col_i);
                assert!(
                    (std - one_12_sqrt).abs() < 0.01,
                    "std={std} ~!= {one_12_sqrt}",
                );
            }
        })
    });
}

pub fn test_vec_znx_fill_normal<B: Backend>(module: &Module<B>)
where
    Module<B>: VecZnxFillNormal,
{
    let n: usize = module.n();
    let base2k: usize = 17;
    let k: usize = 2 * 17;
    let size: usize = 5;
    let sigma: f64 = 3.2;
    let bound: f64 = 6.0 * sigma;
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;
    let zero: Vec<i64> = vec![0; n];
    let k_f64: f64 = (1u64 << k as u64) as f64;
    (0..cols).for_each(|col_i| {
        let mut a: VecZnx<_> = VecZnx::alloc(n, cols, size);
        module.vec_znx_fill_normal(base2k, &mut a, col_i, k, &mut source, sigma, bound);
        (0..cols).for_each(|col_j| {
            if col_j != col_i {
                (0..size).for_each(|limb_i| {
                    assert_eq!(a.at(col_j, limb_i), zero);
                })
            } else {
                let std: f64 = a.std(base2k, col_i) * k_f64;
                assert!((std - sigma).abs() < 0.1, "std={std} ~!= {sigma}");
            }
        })
    });
}

pub fn test_vec_znx_add_normal<B: Backend>(module: &Module<B>)
where
    Module<B>: VecZnxFillNormal + VecZnxAddNormal,
{
    let n: usize = module.n();
    let base2k: usize = 17;
    let k: usize = 2 * 17;
    let size: usize = 5;
    let sigma: f64 = 3.2;
    let bound: f64 = 6.0 * sigma;
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;
    let zero: Vec<i64> = vec![0; n];
    let k_f64: f64 = (1u64 << k as u64) as f64;
    let sqrt2: f64 = SQRT_2;
    (0..cols).for_each(|col_i| {
        let mut a: VecZnx<_> = VecZnx::alloc(n, cols, size);
        module.vec_znx_fill_normal(base2k, &mut a, col_i, k, &mut source, sigma, bound);
        module.vec_znx_add_normal(base2k, &mut a, col_i, k, &mut source, sigma, bound);
        (0..cols).for_each(|col_j| {
            if col_j != col_i {
                (0..size).for_each(|limb_i| {
                    assert_eq!(a.at(col_j, limb_i), zero);
                })
            } else {
                let std: f64 = a.std(base2k, col_i) * k_f64;
                assert!(
                    (std - sigma * sqrt2).abs() < 0.1,
                    "std={std} ~!= {}",
                    sigma * sqrt2
                );
            }
        })
    });
}

pub fn test_vec_znx_lsh<BR: Backend, BT: Backend>(base2k: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: VecZnxLsh<BR> + VecZnxLshTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    Module<BT>: VecZnxLsh<BT> + VecZnxLshTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
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
                    module_ref.vec_znx_lsh(base2k, k, &mut res_ref, i, &a, i, scratch_ref.borrow());
                    module_test.vec_znx_lsh(base2k, k, &mut res_test, i, &a, i, scratch_test.borrow());
                }

                assert_eq!(a.digest_u64(), a_digest);
                assert_eq!(res_ref, res_test);
            }
        }
    }
}

pub fn test_vec_znx_lsh_inplace<BR: Backend, BT: Backend>(base2k: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: VecZnxLshInplace<BR> + VecZnxLshTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    Module<BT>: VecZnxLshInplace<BT> + VecZnxLshTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
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
                module_ref.vec_znx_lsh_inplace(base2k, k, &mut res_ref, i, scratch_ref.borrow());
                module_test.vec_znx_lsh_inplace(base2k, k, &mut res_test, i, scratch_test.borrow());
            }

            assert_eq!(res_ref, res_test);
        }
    }
}

pub fn test_vec_znx_rsh<BR: Backend, BT: Backend>(base2k: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: VecZnxRsh<BR> + VecZnxLshTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    Module<BT>: VecZnxRsh<BT> + VecZnxLshTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
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
                    module_ref.vec_znx_rsh(base2k, k, &mut res_ref, i, &a, i, scratch_ref.borrow());
                    module_test.vec_znx_rsh(base2k, k, &mut res_test, i, &a, i, scratch_test.borrow());
                }

                assert_eq!(a.digest_u64(), a_digest);
                assert_eq!(res_ref, res_test);
            }
        }
    }
}

pub fn test_vec_znx_rsh_inplace<BR: Backend, BT: Backend>(base2k: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: VecZnxRshInplace<BR> + VecZnxLshTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    Module<BT>: VecZnxRshInplace<BT> + VecZnxLshTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
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
                module_ref.vec_znx_rsh_inplace(base2k, k, &mut res_ref, i, scratch_ref.borrow());
                module_test.vec_znx_rsh_inplace(base2k, k, &mut res_test, i, scratch_test.borrow());
            }

            assert_eq!(res_ref, res_test);
        }
    }
}

pub fn test_vec_znx_split_ring<BR: Backend, BT: Backend>(base2k: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: VecZnxSplitRing<BR> + ModuleNew<BR> + VecZnxSplitRingTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    Module<BT>: VecZnxSplitRing<BT> + ModuleNew<BT> + VecZnxSplitRingTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
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
            let mut res_ref: [VecZnx<Vec<u8>>; 2] = [
                VecZnx::alloc(n >> 1, cols, res_size),
                VecZnx::alloc(n >> 1, cols, res_size),
            ];

            let mut res_test: [VecZnx<Vec<u8>>; 2] = [
                VecZnx::alloc(n >> 1, cols, res_size),
                VecZnx::alloc(n >> 1, cols, res_size),
            ];

            res_ref.iter_mut().for_each(|ri| {
                ri.fill_uniform(base2k, &mut source);
            });

            res_test.iter_mut().for_each(|ri| {
                ri.fill_uniform(base2k, &mut source);
            });

            for i in 0..cols {
                module_ref.vec_znx_split_ring(&mut res_ref, i, &a, i, scratch_ref.borrow());
                module_test.vec_znx_split_ring(&mut res_test, i, &a, i, scratch_test.borrow());
            }

            assert_eq!(a.digest_u64(), a_digest);

            for (a, b) in res_ref.iter().zip(res_test.iter()) {
                assert_eq!(a, b);
            }
        }
    }
}

pub fn test_vec_znx_sub_scalar<BR: Backend, BT: Backend>(base2k: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: VecZnxSubScalar,
    Module<BT>: VecZnxSubScalar,
{
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
                module_ref.vec_znx_sub_scalar(&mut res_0, i, &a, i, &b, i, (res_size.min(b_size)) - 1);
                module_test.vec_znx_sub_scalar(&mut res_1, i, &a, i, &b, i, (res_size.min(b_size)) - 1);
            }

            assert_eq!(a.digest_u64(), a_digest);
            assert_eq!(b.digest_u64(), b_digest);
            assert_eq!(res_0, res_1);
        }
    }
}

pub fn test_vec_znx_sub_scalar_inplace<BR: Backend, BT: Backend>(base2k: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: VecZnxSubScalarInplace,
    Module<BT>: VecZnxSubScalarInplace,
{
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
            module_ref.vec_znx_sub_scalar_inplace(&mut res_0, i, res_size - 1, &a, i);
            module_test.vec_znx_sub_scalar_inplace(&mut res_1, i, res_size - 1, &a, i);
        }

        assert_eq!(a.digest_u64(), a_digest);
        assert_eq!(res_0, res_1);
    }
}

pub fn test_vec_znx_sub<BR: Backend, BT: Backend>(base2k: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: VecZnxSub,
    Module<BT>: VecZnxSub,
{
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
                    module_test.vec_znx_sub(&mut res_ref, i, &a, i, &b, i);
                    module_ref.vec_znx_sub(&mut res_test, i, &a, i, &b, i);
                }

                assert_eq!(a.digest_u64(), a_digest);
                assert_eq!(b.digest_u64(), b_digest);

                assert_eq!(res_ref, res_test);
            }
        }
    }
}

pub fn test_vec_znx_sub_inplace<BR: Backend, BT: Backend>(base2k: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: VecZnxSubInplace,
    Module<BT>: VecZnxSubInplace,
{
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
                module_test.vec_znx_sub_inplace(&mut res_ref, i, &a, i);
                module_ref.vec_znx_sub_inplace(&mut res_test, i, &a, i);
            }

            assert_eq!(a.digest_u64(), a_digest);
            assert_eq!(res_ref, res_test);
        }
    }
}

pub fn test_vec_znx_sub_negate_inplace<BR: Backend, BT: Backend>(base2k: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: VecZnxSubNegateInplace,
    Module<BT>: VecZnxSubNegateInplace,
{
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
                module_test.vec_znx_sub_negate_inplace(&mut res_ref, i, &a, i);
                module_ref.vec_znx_sub_negate_inplace(&mut res_test, i, &a, i);
            }

            assert_eq!(a.digest_u64(), a_digest);
            assert_eq!(res_ref, res_test);
        }
    }
}

pub fn test_vec_znx_switch_ring<BR: Backend, BT: Backend>(base2k: usize, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: VecZnxSwitchRing,
    Module<BT>: VecZnxSwitchRing,
{
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
                    module_ref.vec_znx_switch_ring(&mut res_ref, i, &a, i);
                    module_test.vec_znx_switch_ring(&mut res_test, i, &a, i);
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
                    module_ref.vec_znx_switch_ring(&mut res_ref, i, &a, i);
                    module_test.vec_znx_switch_ring(&mut res_test, i, &a, i);
                }

                assert_eq!(a.digest_u64(), a_digest);
                assert_eq!(res_ref, res_test);
            }
        }
    }
}
