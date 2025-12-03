use crate::{
    api::{
        CnvPVecAlloc, Convolution, ModuleN, ScratchOwnedAlloc, ScratchOwnedBorrow, ScratchTakeBasic, TakeSlice, VecZnxAdd,
        VecZnxBigAlloc, VecZnxBigNormalize, VecZnxCopy, VecZnxDftAlloc, VecZnxDftApply, VecZnxIdftApplyTmpA,
        VecZnxNormalizeInplace,
    },
    layouts::{
        Backend, CnvPVecL, CnvPVecR, FillUniform, Scratch, ScratchOwned, VecZnx, VecZnxBig, VecZnxDft, VecZnxToMut, VecZnxToRef,
        ZnxInfos, ZnxView, ZnxViewMut, ZnxZero,
    },
    source::Source,
};

pub fn test_convolution<M, BE: Backend>(module: &M)
where
    M: ModuleN
        + Convolution<BE>
        + CnvPVecAlloc<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxDftApply<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxNormalizeInplace<BE>
        + VecZnxBigAlloc<BE>,
    Scratch<BE>: ScratchTakeBasic,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let mut source: Source = Source::new([0u8; 32]);

    let base2k: usize = 12;

    let a_cols: usize = 2;
    let b_cols: usize = 2;
    let a_size: usize = 15;
    let b_size: usize = 15;
    let res_size: usize = a_size + b_size;

    let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), a_cols, a_size);
    let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), b_cols, b_size);

    let mut res_want: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), 1, res_size);
    let mut res_have: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), 1, res_size);
    let mut res_dft: VecZnxDft<Vec<u8>, BE> = module.vec_znx_dft_alloc(1, res_size);
    let mut res_big: VecZnxBig<Vec<u8>, BE> = module.vec_znx_big_alloc(1, res_size);

    a.fill_uniform(base2k, &mut source);
    b.fill_uniform(base2k, &mut source);

    let mut a_prep: CnvPVecL<Vec<u8>, BE> = module.cnv_pvec_left_alloc(a_cols, a_size);
    let mut b_prep: CnvPVecR<Vec<u8>, BE> = module.cnv_pvec_right_alloc(b_cols, b_size);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .cnv_apply_dft_tmp_bytes(res_size, 0, a_size, b_size)
            .max(module.cnv_prepare_left_tmp_bytes(res_size, a_size))
            .max(module.cnv_prepare_right_tmp_bytes(res_size, b_size)),
    );

    module.cnv_prepare_left(&mut a_prep, &a, scratch.borrow());
    module.cnv_prepare_right(&mut b_prep, &b, scratch.borrow());

    for a_col in 0..a.cols() {
        for b_col in 0..b.cols() {
            for offset in 0..res_size {
                module.cnv_apply_dft(&mut res_dft, offset, 0, &a_prep, a_col, &b_prep, b_col, scratch.borrow());

                module.vec_znx_idft_apply_tmpa(&mut res_big, 0, &mut res_dft, 0);
                module.vec_znx_big_normalize(base2k, &mut res_have, 0, base2k, &res_big, 0, scratch.borrow());

                bivariate_convolution_naive(
                    module,
                    base2k,
                    (offset + 1) as i64,
                    &mut res_want,
                    0,
                    &a,
                    a_col,
                    &b,
                    b_col,
                    scratch.borrow(),
                );

                assert_eq!(res_want, res_have);
            }
        }
    }
}

pub fn test_convolution_pairwise<M, BE: Backend>(module: &M)
where
    M: ModuleN
        + Convolution<BE>
        + CnvPVecAlloc<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxDftApply<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxNormalizeInplace<BE>
        + VecZnxBigAlloc<BE>
        + VecZnxAdd
        + VecZnxCopy,
    Scratch<BE>: ScratchTakeBasic,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let mut source: Source = Source::new([0u8; 32]);

    let base2k: usize = 12;

    let cols = 2;
    let a_size: usize = 15;
    let b_size: usize = 15;
    let res_size: usize = a_size + b_size;

    let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, a_size);
    let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, b_size);
    let mut tmp_a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), 1, a_size);
    let mut tmp_b: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), 1, b_size);

    let mut res_want: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), 1, res_size);
    let mut res_have: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), 1, res_size);
    let mut res_dft: VecZnxDft<Vec<u8>, BE> = module.vec_znx_dft_alloc(1, res_size);
    let mut res_big: VecZnxBig<Vec<u8>, BE> = module.vec_znx_big_alloc(1, res_size);

    a.fill_uniform(base2k, &mut source);
    b.fill_uniform(base2k, &mut source);

    let mut a_prep: CnvPVecL<Vec<u8>, BE> = module.cnv_pvec_left_alloc(cols, a_size);
    let mut b_prep: CnvPVecR<Vec<u8>, BE> = module.cnv_pvec_right_alloc(cols, b_size);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .cnv_pairwise_apply_dft_tmp_bytes(res_size, 0, a_size, b_size)
            .max(module.cnv_prepare_left_tmp_bytes(res_size, a_size))
            .max(module.cnv_prepare_right_tmp_bytes(res_size, b_size)),
    );

    module.cnv_prepare_left(&mut a_prep, &a, scratch.borrow());
    module.cnv_prepare_right(&mut b_prep, &b, scratch.borrow());

    for col_i in 0..cols {
        for col_j in 0..cols {
            for offset in 0..res_size {
                module.cnv_pairwise_apply_dft(&mut res_dft, offset, 0, &a_prep, &b_prep, col_i, col_j, scratch.borrow());

                module.vec_znx_idft_apply_tmpa(&mut res_big, 0, &mut res_dft, 0);
                module.vec_znx_big_normalize(base2k, &mut res_have, 0, base2k, &res_big, 0, scratch.borrow());

                if col_i != col_j {
                    module.vec_znx_add(&mut tmp_a, 0, &a, col_i, &a, col_j);
                    module.vec_znx_add(&mut tmp_b, 0, &b, col_i, &b, col_j);
                } else {
                    module.vec_znx_copy(&mut tmp_a, 0, &a, col_i);
                    module.vec_znx_copy(&mut tmp_b, 0, &b, col_j);
                }

                bivariate_convolution_naive(
                    module,
                    base2k,
                    (offset + 1) as i64,
                    &mut res_want,
                    0,
                    &tmp_a,
                    0,
                    &tmp_b,
                    0,
                    scratch.borrow(),
                );

                assert_eq!(res_want, res_have);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn bivariate_convolution_naive<R, A, B, M, BE: Backend>(
    module: &M,
    base2k: usize,
    k: i64,
    res: &mut R,
    res_col: usize,
    a: &A,
    a_col: usize,
    b: &B,
    b_col: usize,
    scratch: &mut Scratch<BE>,
) where
    R: VecZnxToMut,
    A: VecZnxToRef,
    B: VecZnxToRef,
    M: VecZnxNormalizeInplace<BE>,
    Scratch<BE>: TakeSlice,
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

    module.vec_znx_normalize_inplace(base2k, res, res_col, scratch);
}

fn bivariate_tensoring_naive<R, A, B, M, BE: Backend>(
    module: &M,
    base2k: usize,
    k: i64,
    res: &mut R,
    a: &A,
    b: &B,
    scratch: &mut Scratch<BE>,
) where
    R: VecZnxToMut,
    A: VecZnxToRef,
    B: VecZnxToRef,
    M: VecZnxNormalizeInplace<BE>,
    Scratch<BE>: TakeSlice,
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
        module.vec_znx_normalize_inplace(base2k, res, i, scratch);
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
