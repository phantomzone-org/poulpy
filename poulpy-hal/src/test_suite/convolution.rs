use super::{vec_znx_backend_mut, vec_znx_backend_ref};
use rand::Rng;

use crate::{
    api::{
        CnvPVecAlloc, Convolution, ModuleN, ScratchOwnedAlloc, VecZnxAddIntoBackend, VecZnxBigAlloc, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxCopyBackend, VecZnxDftAlloc, VecZnxDftApply, VecZnxIdftApplyTmpA,
        VecZnxNormalizeInplace,
    },
    layouts::{
        Backend, CnvPVecL, CnvPVecR, FillUniform, ScratchArena, ScratchOwned, VecZnx, VecZnxBig, VecZnxBigToBackendMut,
        VecZnxBigToBackendRef, VecZnxDft, VecZnxDftToBackendMut, VecZnxToMut, VecZnxToRef, ZnxInfos, ZnxView, ZnxViewMut,
        ZnxZero,
    },
    source::Source,
};

type VecZnxBigOwned<BE> = VecZnxBig<<BE as Backend>::OwnedBuf, BE>;
type VecZnxDftOwned<BE> = VecZnxDft<<BE as Backend>::OwnedBuf, BE>;
type CnvPVecLOwned<BE> = CnvPVecL<<BE as Backend>::OwnedBuf, BE>;
type CnvPVecROwned<BE> = CnvPVecR<<BE as Backend>::OwnedBuf, BE>;

pub fn test_convolution_by_const<M, BE: crate::test_suite::TestBackend>(module: &M, base2k: usize)
where
    BE::OwnedBuf: crate::layouts::HostDataMut,
    M: ModuleN
        + Convolution<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxNormalizeAssign<BE>
        + VecZnxBigAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    let mut source: Source = Source::new([0u8; 32]);

    let a_cols: usize = 2;
    let a_size: usize = 15;
    let b_size: usize = 15;
    let res_size: usize = a_size + b_size;

    let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), a_cols, a_size);
    let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), 1, b_size);

    let mut res_want: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), 1, res_size);
    let mut res_have: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), 1, res_size);
    let mut res_big: VecZnxBigOwned<BE> = module.vec_znx_big_alloc(1, res_size);

    a.fill_uniform(17, &mut source);

    let mut b_const = vec![0i64; b_size];
    let mask = (1 << base2k) - 1;
    for (j, x) in b_const[..1].iter_mut().enumerate() {
        let r = source.next_u64() & mask;
        *x = ((r << (64 - 17)) as i64) >> (64 - 17);
        b.at_mut(0, j)[0] = *x
    }

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .cnv_by_const_apply_tmp_bytes(0, res_size, a_size, b_size)
            .max(module.vec_znx_big_normalize_tmp_bytes()),
    );

    for a_col in 0..a.cols() {
        for cnv_offset in 0..res_size {
            module.cnv_by_const_apply(cnv_offset, &mut res_big, 0, &a, a_col, &b_const, &mut scratch.arena());
            module.vec_znx_big_normalize(
                &mut vec_znx_backend_mut::<BE>(&mut res_have),
                base2k,
                0,
                0,
                &res_big.to_backend_ref(),
                base2k,
                0,
                &mut scratch.arena(),
            );

            bivariate_convolution_naive(
                module,
                base2k,
                (cnv_offset + 1) as i64,
                &mut res_want,
                0,
                &a,
                a_col,
                &b,
                0,
                &mut scratch.arena(),
            );

            assert_eq!(res_want, res_have);
        }
    }
}

pub fn test_convolution<M, BE: crate::test_suite::TestBackend>(module: &M, base2k: usize)
where
    BE::OwnedBuf: crate::layouts::HostDataMut,
    M: ModuleN
        + Convolution<BE>
        + CnvPVecAlloc<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxDftApply<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxNormalizeAssign<BE>
        + VecZnxBigAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    let mut source: Source = Source::new([0u8; 32]);

    let a_cols: usize = 2;
    let b_cols: usize = 2;
    let a_size: usize = 15;
    let b_size: usize = 15;
    let res_size: usize = a_size + b_size;

    let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), a_cols, a_size);
    let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), b_cols, b_size);

    let mut res_want: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), 1, res_size);
    let mut res_have: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), 1, res_size);
    let mut res_dft: VecZnxDftOwned<BE> = module.vec_znx_dft_alloc(1, res_size);
    let mut res_big: VecZnxBigOwned<BE> = module.vec_znx_big_alloc(1, res_size);

    a.fill_uniform(17, &mut source);
    b.fill_uniform(17, &mut source);

    let mut a_prep: CnvPVecLOwned<BE> = module.cnv_pvec_left_alloc(a_cols, a_size);
    let mut b_prep: CnvPVecROwned<BE> = module.cnv_pvec_right_alloc(b_cols, b_size);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .cnv_apply_dft_tmp_bytes(0, res_size, a_size, b_size)
            .max(module.cnv_prepare_left_tmp_bytes(res_size, a_size))
            .max(module.cnv_prepare_right_tmp_bytes(res_size, b_size))
            .max(module.vec_znx_big_normalize_tmp_bytes()),
    );

    module.cnv_prepare_left(&mut a_prep, &a, !0i64, &mut scratch.arena());
    module.cnv_prepare_right(&mut b_prep, &b, !0i64, &mut scratch.arena());

    for a_col in 0..a.cols() {
        for b_col in 0..b.cols() {
            for cnv_offset in 0..res_size {
                module.cnv_apply_dft(
                    cnv_offset,
                    &mut res_dft,
                    0,
                    &a_prep,
                    a_col,
                    &b_prep,
                    b_col,
                    &mut scratch.arena(),
                );

                module.vec_znx_idft_apply_tmpa(&mut res_big.to_backend_mut(), 0, &mut res_dft.to_backend_mut(), 0);
                module.vec_znx_big_normalize(
                    &mut vec_znx_backend_mut::<BE>(&mut res_have),
                    base2k,
                    0,
                    0,
                    &res_big.to_backend_ref(),
                    base2k,
                    0,
                    &mut scratch.arena(),
                );

                bivariate_convolution_naive(
                    module,
                    base2k,
                    (cnv_offset + 1) as i64,
                    &mut res_want,
                    0,
                    &a,
                    a_col,
                    &b,
                    b_col,
                    &mut scratch.arena(),
                );

                assert_eq!(res_want, res_have);
            }
        }
    }
}

pub fn test_convolution_pairwise<M, BE: crate::test_suite::TestBackend>(module: &M, base2k: usize)
where
    BE::OwnedBuf: crate::layouts::HostDataMut,
    M: ModuleN
        + Convolution<BE>
        + CnvPVecAlloc<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxDftApply<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxNormalizeAssign<BE>
        + VecZnxBigAlloc<BE>
        + VecZnxAddIntoBackend<BE>
        + VecZnxCopyBackend<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    let mut source: Source = Source::new([0u8; 32]);

    let cols: usize = 2;
    let a_size: usize = 15;
    let b_size: usize = 15;
    let res_size: usize = a_size + b_size;

    let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, a_size);
    let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, b_size);
    let mut tmp_a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), 1, a_size);
    let mut tmp_b: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), 1, b_size);

    let mut res_want: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), 1, res_size);
    let mut res_have: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), 1, res_size);
    let mut res_dft: VecZnxDftOwned<BE> = module.vec_znx_dft_alloc(1, res_size);
    let mut res_big: VecZnxBigOwned<BE> = module.vec_znx_big_alloc(1, res_size);

    a.fill_uniform(17, &mut source);
    b.fill_uniform(17, &mut source);

    let mut a_prep: CnvPVecLOwned<BE> = module.cnv_pvec_left_alloc(cols, a_size);
    let mut b_prep: CnvPVecROwned<BE> = module.cnv_pvec_right_alloc(cols, b_size);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .cnv_pairwise_apply_dft_tmp_bytes(res_size, 0, a_size, b_size)
            .max(module.cnv_prepare_left_tmp_bytes(res_size, a_size))
            .max(module.cnv_prepare_right_tmp_bytes(res_size, b_size))
            .max(module.vec_znx_big_normalize_tmp_bytes()),
    );

    module.cnv_prepare_left(&mut a_prep, &a, !0i64, &mut scratch.arena());
    module.cnv_prepare_right(&mut b_prep, &b, !0i64, &mut scratch.arena());

    for col_i in 0..cols {
        for col_j in 0..cols {
            for cnv_offset in 0..res_size {
                module.cnv_pairwise_apply_dft(
                    cnv_offset,
                    &mut res_dft,
                    0,
                    &a_prep,
                    &b_prep,
                    col_i,
                    col_j,
                    &mut scratch.arena(),
                );

                module.vec_znx_idft_apply_tmpa(&mut res_big.to_backend_mut(), 0, &mut res_dft.to_backend_mut(), 0);
                module.vec_znx_big_normalize(
                    &mut vec_znx_backend_mut::<BE>(&mut res_have),
                    base2k,
                    0,
                    0,
                    &res_big.to_backend_ref(),
                    base2k,
                    0,
                    &mut scratch.arena(),
                );

                if col_i != col_j {
                    module.vec_znx_add_into_backend(
                        &mut vec_znx_backend_mut::<BE>(&mut tmp_a),
                        0,
                        &vec_znx_backend_ref::<BE>(&a),
                        col_i,
                        &vec_znx_backend_ref::<BE>(&a),
                        col_j,
                    );
                    module.vec_znx_add_into_backend(
                        &mut vec_znx_backend_mut::<BE>(&mut tmp_b),
                        0,
                        &vec_znx_backend_ref::<BE>(&b),
                        col_i,
                        &vec_znx_backend_ref::<BE>(&b),
                        col_j,
                    );
                } else {
                    module.vec_znx_copy_backend(
                        &mut vec_znx_backend_mut::<BE>(&mut tmp_a),
                        0,
                        &vec_znx_backend_ref::<BE>(&a),
                        col_i,
                    );
                    module.vec_znx_copy_backend(
                        &mut vec_znx_backend_mut::<BE>(&mut tmp_b),
                        0,
                        &vec_znx_backend_ref::<BE>(&b),
                        col_j,
                    );
                }

                bivariate_convolution_naive(
                    module,
                    base2k,
                    (cnv_offset + 1) as i64,
                    &mut res_want,
                    0,
                    &tmp_a,
                    0,
                    &tmp_b,
                    0,
                    &mut scratch.arena(),
                );

                assert_eq!(res_want, res_have);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn bivariate_convolution_naive<'s, R, A, B, M, BE: crate::test_suite::TestBackend>(
    module: &M,
    base2k: usize,
    k: i64,
    res: &mut R,
    res_col: usize,
    a: &A,
    a_col: usize,
    b: &B,
    b_col: usize,
    scratch: &mut ScratchArena<'s, BE>,
) where
    BE::OwnedBuf: crate::layouts::HostDataMut,
    R: VecZnxToMut,
    A: VecZnxToRef,
    B: VecZnxToRef,
    M: VecZnxNormalizeInplace<BE>,
{
    let res: &mut VecZnx<&mut [u8]> = &mut res.to_mut();
    let a: &VecZnx<&[u8]> = &a.to_ref();
    let b: &VecZnx<&[u8]> = &b.to_ref();

    for j in 0..res.size() {
        res.zero_at(res_col, j);
    }

    for a_limb in 0..a.size() {
        for b_limb in 0..b.size() {
            let res_scale_abs = k.unsigned_abs() as usize;

            let mut res_limb: usize = a_limb + b_limb + 1;

            if k <= 0 {
                res_limb += res_scale_abs;

                if res_limb < res.size() {
                    negacyclic_convolution_naive_add(res.at_mut(res_col, res_limb), a.at(a_col, a_limb), b.at(b_col, b_limb));
                }
            } else if res_limb >= res_scale_abs {
                res_limb -= res_scale_abs;

                if res_limb < res.size() {
                    negacyclic_convolution_naive_add(res.at_mut(res_col, res_limb), a.at(a_col, a_limb), b.at(b_col, b_limb));
                }
            }
        }
    }

    module.vec_znx_normalize_assign(base2k, res, res_col, scratch);
}

fn bivariate_tensoring_naive<'s, R, A, B, M, BE: crate::test_suite::TestBackend>(
    module: &M,
    base2k: usize,
    k: i64,
    res: &mut R,
    a: &A,
    b: &B,
    scratch: &mut ScratchArena<'s, BE>,
) where
    BE::OwnedBuf: crate::layouts::HostDataMut,
    R: VecZnxToMut,
    A: VecZnxToRef,
    B: VecZnxToRef,
    M: VecZnxNormalizeInplace<BE>,
{
    let res: &mut VecZnx<&mut [u8]> = &mut res.to_mut();
    let a: &VecZnx<&[u8]> = &a.to_ref();
    let b: &VecZnx<&[u8]> = &b.to_ref();

    assert!(res.cols() >= a.cols() + b.cols() - 1);

    res.zero();

    for a_col in 0..a.cols() {
        for a_limb in 0..a.size() {
            for b_col in 0..b.cols() {
                for b_limb in 0..b.size() {
                    let res_scale_abs = k.unsigned_abs() as usize;

                    let mut res_limb: usize = a_limb + b_limb + 1;

                    if k <= 0 {
                        res_limb += res_scale_abs;

                        if res_limb < res.size() {
                            negacyclic_convolution_naive_add(
                                res.at_mut(a_col + b_col, res_limb),
                                a.at(a_col, a_limb),
                                b.at(b_col, b_limb),
                            );
                        }
                    } else if res_limb >= res_scale_abs {
                        res_limb -= res_scale_abs;

                        if res_limb < res.size() {
                            negacyclic_convolution_naive_add(
                                res.at_mut(a_col + b_col, res_limb),
                                a.at(a_col, a_limb),
                                b.at(b_col, b_limb),
                            );
                        }
                    }
                }
            }
        }
    }

    for i in 0..res.cols() {
        module.vec_znx_normalize_assign(base2k, res, i, scratch);
    }
}

fn negacyclic_convolution_naive_add(res: &mut [i64], a: &[i64], b: &[i64]) {
    let n: usize = res.len();
    for i in 0..n {
        let ai: i64 = a[i];
        let lim: usize = n - i;
        for j in 0..lim {
            res[i + j] += ai * b[j];
        }
        for j in lim..n {
            res[i + j - n] -= ai * b[j];
        }
    }
}

fn negacyclic_convolution_naive(res: &mut [i64], a: &[i64], b: &[i64]) {
    let n: usize = res.len();
    res.fill(0);
    for i in 0..n {
        let ai: i64 = a[i];
        let lim: usize = n - i;
        for j in 0..lim {
            res[i + j] += ai * b[j];
        }
        for j in lim..n {
            res[i + j - n] -= ai * b[j];
        }
    }
}
