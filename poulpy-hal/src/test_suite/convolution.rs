use crate::{
    api::{
        Convolution, ModuleN, ScratchOwnedAlloc, ScratchOwnedBorrow, ScratchTakeBasic, TakeSlice, VecZnxBigAlloc,
        VecZnxBigNormalize, VecZnxDftAlloc, VecZnxDftApply, VecZnxIdftApplyTmpA, VecZnxNormalizeInplace,
    },
    layouts::{
        Backend, FillUniform, Scratch, ScratchOwned, VecZnx, VecZnxBig, VecZnxDft, VecZnxToMut, VecZnxToRef, ZnxInfos, ZnxView,
        ZnxViewMut, ZnxZero,
    },
    source::Source,
};

pub fn test_convolution<M, BE: Backend>(module: &M)
where
    M: ModuleN
        + Convolution<BE>
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

        module.bivariate_convolution_full(k, &mut c_have_dft, &a, &b_dft, scratch.borrow());

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

        convolution_naive(module, base2k, k, &mut c_want, &a, &b, scratch.borrow());

        assert_eq!(c_want, c_have);
    }
}

fn convolution_naive<R, A, B, M, BE: Backend>(
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
