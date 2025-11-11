use crate::{
    api::{
        BivariateTensoring, CnvPVecAlloc, Convolution, ModuleN, ScratchOwnedAlloc, ScratchOwnedBorrow, ScratchTakeBasic,
        TakeSlice, VecZnxBigAlloc, VecZnxBigNormalize, VecZnxDftAlloc, VecZnxDftApply, VecZnxIdftApplyTmpA,
        VecZnxNormalizeInplace,
    },
    layouts::{
        Backend, CnvPVecL, CnvPVecR, FillUniform, Scratch, ScratchOwned, VecZnx, VecZnxBig, VecZnxDft, VecZnxToMut, VecZnxToRef,
        ZnxInfos, ZnxView, ZnxViewMut, ZnxZero,
    },
    source::Source,
};

pub fn test_bivariate_tensoring<M, BE: Backend>(module: &M)
where
    M: ModuleN
        + BivariateTensoring<BE>
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

    let a_cols: usize = 3;
    let b_cols: usize = 3;
    let a_size: usize = 3;
    let b_size: usize = 3;
    let c_cols: usize = a_cols + b_cols - 1;
    let c_size: usize = a_size + b_size;

    let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), a_cols, a_size);
    let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), b_cols, b_size);

    let mut c_want: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), c_cols, c_size);
    let mut c_have: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), c_cols, c_size);
    let mut c_have_dft: VecZnxDft<Vec<u8>, BE> = module.vec_znx_dft_alloc(c_cols, c_size);
    let mut c_have_big: VecZnxBig<Vec<u8>, BE> = module.vec_znx_big_alloc(c_cols, c_size);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.convolution_tmp_bytes(b_size));

    a.fill_uniform(base2k, &mut source);
    b.fill_uniform(base2k, &mut source);

    let mut b_dft: VecZnxDft<Vec<u8>, BE> = module.vec_znx_dft_alloc(b_cols, b_size);
    for i in 0..b.cols() {
        module.vec_znx_dft_apply(1, 0, &mut b_dft, i, &b, i);
    }

    for mut k in 0..(2 * c_size + 1) as i64 {
        k -= c_size as i64;

        module.bivariate_tensoring(k, &mut c_have_dft, &a, &b_dft, scratch.borrow());

        for i in 0..c_cols {
            module.vec_znx_idft_apply_tmpa(&mut c_have_big, i, &mut c_have_dft, i);
        }

        for i in 0..c_cols {
            module.vec_znx_big_normalize(
                base2k,
                &mut c_have,
                i,
                base2k,
                &c_have_big,
                i,
                scratch.borrow(),
            );
        }

        bivariate_tensoring_naive(module, base2k, k, &mut c_want, &a, &b, scratch.borrow());

        assert_eq!(c_want, c_have);
    }
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

    let a_cols: usize = 1;
    let b_cols: usize = 1;
    let a_size: usize = 31;
    let b_size: usize = 31;
    let c_cols: usize = 1;
    let c_size: usize = a_size + b_size;

    let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), a_cols, a_size);
    let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), b_cols, b_size);

    let mut c_want: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), c_cols, c_size);
    let mut c_have: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), c_cols, c_size);
    let mut c_have_dft: VecZnxDft<Vec<u8>, BE> = module.vec_znx_dft_alloc(c_cols, c_size);
    let mut c_have_big: VecZnxBig<Vec<u8>, BE> = module.vec_znx_big_alloc(c_cols, c_size);

    a.fill_uniform(base2k, &mut source);
    b.fill_uniform(base2k, &mut source);

    println!("a: {a}");
    println!("b: {b}");

    let mut b_dft: VecZnxDft<Vec<u8>, BE> = module.vec_znx_dft_alloc(b_cols, b_size);
    for i in 0..b.cols() {
        module.vec_znx_dft_apply(1, 0, &mut b_dft, i, &b, i);
    }

    let mut a_prep: CnvPVecL<Vec<u8>, BE> = module.cnv_pvec_left_alloc(1, a_size);
    let mut b_prep: CnvPVecR<Vec<u8>, BE> = module.cnv_pvec_right_alloc(1, b_size);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .cnv_apply_dft_tmp_bytes(c_size, 0, a_size, b_size)
            .max(module.cnv_prepare_left_tmp_bytes(c_size, a_size))
            .max(module.cnv_prepare_right_tmp_bytes(c_size, b_size)),
    );

    module.cnv_prepare_left(&mut a_prep, &a, scratch.borrow());
    module.cnv_prepare_right(&mut b_prep, &b, scratch.borrow());

    let offset: usize = 6;

    module.cnv_apply_dft(
        &mut c_have_dft,
        offset,
        0,
        &a_prep,
        0,
        &b_prep,
        0,
        scratch.borrow(),
    );

    for i in 0..c_cols {
        module.vec_znx_idft_apply_tmpa(&mut c_have_big, i, &mut c_have_dft, i);
    }

    println!("c_big: {c_have_big}");

    for i in 0..c_cols {
        module.vec_znx_big_normalize(
            base2k,
            &mut c_have,
            i,
            base2k,
            &c_have_big,
            i,
            scratch.borrow(),
        );
    }

    bivariate_tensoring_naive(
        module,
        base2k,
        (offset + 1) as i64,
        &mut c_want,
        &a,
        &b,
        scratch.borrow(),
    );

    assert_eq!(c_want, c_have);
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
