use std::f64::consts::SQRT_2;

use crate::{
    api::VecZnxBigAddNormal,
    layouts::{
        Backend, Module, VecZnx, VecZnxBig, VecZnxBigToMut, VecZnxBigToRef, VecZnxToMut, VecZnxToRef, ZnxView, ZnxViewMut,
    },
    oep::VecZnxBigAllocBytesImpl,
    reference::{
        vec_znx::{
            vec_znx_add, vec_znx_add_inplace, vec_znx_automorphism, vec_znx_automorphism_inplace, vec_znx_negate,
            vec_znx_negate_inplace, vec_znx_normalize, vec_znx_normalize_tmp_bytes, vec_znx_sub, vec_znx_sub_inplace,
            vec_znx_sub_negate_inplace,
        },
        znx::{
            ZnxAdd, ZnxAddInplace, ZnxAutomorphism, ZnxCopy, ZnxExtractDigitAddMul, ZnxMulPowerOfTwoInplace, ZnxNegate,
            ZnxNegateInplace, ZnxNormalizeDigit, ZnxNormalizeFinalStep, ZnxNormalizeFinalStepInplace, ZnxNormalizeFirstStep,
            ZnxNormalizeFirstStepCarryOnly, ZnxNormalizeMiddleStep, ZnxNormalizeMiddleStepCarryOnly,
            ZnxNormalizeMiddleStepInplace, ZnxSub, ZnxSubInplace, ZnxSubNegateInplace, ZnxZero, znx_add_normal_f64_ref,
        },
    },
    source::Source,
};

pub fn vec_znx_big_add<R, A, B, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    BE: Backend<ScalarBig = i64> + ZnxAdd + ZnxCopy + ZnxZero,
    R: VecZnxBigToMut<BE>,
    A: VecZnxBigToRef<BE>,
    B: VecZnxBigToRef<BE>,
{
    let res: VecZnxBig<&mut [u8], BE> = res.to_mut();
    let a: VecZnxBig<&[u8], BE> = a.to_ref();
    let b: VecZnxBig<&[u8], BE> = b.to_ref();

    let mut res_vznx: VecZnx<&mut [u8]> = VecZnx {
        data: res.data,
        n: res.n,
        cols: res.cols,
        size: res.size,
        max_size: res.max_size,
    };

    let a_vznx: VecZnx<&[u8]> = VecZnx {
        data: a.data,
        n: a.n,
        cols: a.cols,
        size: a.size,
        max_size: a.max_size,
    };

    let b_vznx: VecZnx<&[u8]> = VecZnx {
        data: b.data,
        n: b.n,
        cols: b.cols,
        size: b.size,
        max_size: b.max_size,
    };

    vec_znx_add::<_, _, _, BE>(&mut res_vznx, res_col, &a_vznx, a_col, &b_vznx, b_col);
}

pub fn vec_znx_big_add_inplace<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarBig = i64> + ZnxAddInplace,
    R: VecZnxBigToMut<BE>,
    A: VecZnxBigToRef<BE>,
{
    let res: VecZnxBig<&mut [u8], BE> = res.to_mut();
    let a: VecZnxBig<&[u8], BE> = a.to_ref();

    let mut res_vznx: VecZnx<&mut [u8]> = VecZnx {
        data: res.data,
        n: res.n,
        cols: res.cols,
        size: res.size,
        max_size: res.max_size,
    };

    let a_vznx: VecZnx<&[u8]> = VecZnx {
        data: a.data,
        n: a.n,
        cols: a.cols,
        size: a.size,
        max_size: a.max_size,
    };

    vec_znx_add_inplace::<_, _, BE>(&mut res_vznx, res_col, &a_vznx, a_col);
}

pub fn vec_znx_big_add_small<R, A, B, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    BE: Backend<ScalarBig = i64> + ZnxAdd + ZnxCopy + ZnxZero,
    R: VecZnxBigToMut<BE>,
    A: VecZnxBigToRef<BE>,
    B: VecZnxToRef,
{
    let res: VecZnxBig<&mut [u8], BE> = res.to_mut();
    let a: VecZnxBig<&[u8], BE> = a.to_ref();

    let mut res_vznx: VecZnx<&mut [u8]> = VecZnx {
        data: res.data,
        n: res.n,
        cols: res.cols,
        size: res.size,
        max_size: res.max_size,
    };

    let a_vznx: VecZnx<&[u8]> = VecZnx {
        data: a.data,
        n: a.n,
        cols: a.cols,
        size: a.size,
        max_size: a.max_size,
    };

    vec_znx_add::<_, _, _, BE>(&mut res_vznx, res_col, &a_vznx, a_col, b, b_col);
}

pub fn vec_znx_big_add_small_inplace<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarBig = i64> + ZnxAddInplace,
    R: VecZnxBigToMut<BE>,
    A: VecZnxToRef,
{
    let res: VecZnxBig<&mut [u8], BE> = res.to_mut();
    let mut res_vznx: VecZnx<&mut [u8]> = VecZnx {
        data: res.data,
        n: res.n,
        cols: res.cols,
        size: res.size,
        max_size: res.max_size,
    };

    vec_znx_add_inplace::<_, _, BE>(&mut res_vznx, res_col, a, a_col);
}

pub fn vec_znx_big_automorphism_inplace_tmp_bytes(n: usize) -> usize {
    n * size_of::<i64>()
}

pub fn vec_znx_big_automorphism<R, A, BE>(p: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarBig = i64> + ZnxAutomorphism + ZnxZero,
    R: VecZnxBigToMut<BE>,
    A: VecZnxBigToRef<BE>,
{
    let res: VecZnxBig<&mut [u8], _> = res.to_mut();
    let a: VecZnxBig<&[u8], _> = a.to_ref();

    let mut res_vznx: VecZnx<&mut [u8]> = VecZnx {
        data: res.data,
        n: res.n,
        cols: res.cols,
        size: res.size,
        max_size: res.max_size,
    };

    let a_vznx: VecZnx<&[u8]> = VecZnx {
        data: a.data,
        n: a.n,
        cols: a.cols,
        size: a.size,
        max_size: a.max_size,
    };

    vec_znx_automorphism::<_, _, BE>(p, &mut res_vznx, res_col, &a_vznx, a_col);
}

pub fn vec_znx_big_automorphism_inplace<R, BE>(p: i64, res: &mut R, res_col: usize, tmp: &mut [i64])
where
    BE: Backend<ScalarBig = i64> + ZnxAutomorphism + ZnxCopy,
    R: VecZnxBigToMut<BE>,
{
    let res: VecZnxBig<&mut [u8], _> = res.to_mut();

    let mut res_vznx: VecZnx<&mut [u8]> = VecZnx {
        data: res.data,
        n: res.n,
        cols: res.cols,
        size: res.size,
        max_size: res.max_size,
    };

    vec_znx_automorphism_inplace::<_, BE>(p, &mut res_vznx, res_col, tmp);
}

pub fn vec_znx_big_negate<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarBig = i64> + ZnxNegate + ZnxZero,
    R: VecZnxBigToMut<BE>,
    A: VecZnxBigToRef<BE>,
{
    let res: VecZnxBig<&mut [u8], _> = res.to_mut();
    let a: VecZnxBig<&[u8], _> = a.to_ref();

    let mut res_vznx: VecZnx<&mut [u8]> = VecZnx {
        data: res.data,
        n: res.n,
        cols: res.cols,
        size: res.size,
        max_size: res.max_size,
    };

    let a_vznx: VecZnx<&[u8]> = VecZnx {
        data: a.data,
        n: a.n,
        cols: a.cols,
        size: a.size,
        max_size: a.max_size,
    };

    vec_znx_negate::<_, _, BE>(&mut res_vznx, res_col, &a_vznx, a_col);
}

pub fn vec_znx_big_negate_inplace<R, BE>(res: &mut R, res_col: usize)
where
    BE: Backend<ScalarBig = i64> + ZnxNegateInplace,
    R: VecZnxBigToMut<BE>,
{
    let res: VecZnxBig<&mut [u8], _> = res.to_mut();

    let mut res_vznx: VecZnx<&mut [u8]> = VecZnx {
        data: res.data,
        n: res.n,
        cols: res.cols,
        size: res.size,
        max_size: res.max_size,
    };

    vec_znx_negate_inplace::<_, BE>(&mut res_vznx, res_col);
}

pub fn vec_znx_big_normalize_tmp_bytes(n: usize) -> usize {
    vec_znx_normalize_tmp_bytes(n)
}

#[allow(clippy::too_many_arguments)]
pub fn vec_znx_big_normalize<R, A, BE>(
    res: &mut R,
    res_base2k: usize,
    res_offset: i64,
    res_col: usize,
    a: &A,
    a_base2k: usize,
    a_col: usize,
    carry: &mut [i64],
) where
    R: VecZnxToMut,
    A: VecZnxBigToRef<BE>,
    BE: Backend<ScalarBig = i64>
        + ZnxZero
        + ZnxCopy
        + ZnxAddInplace
        + ZnxMulPowerOfTwoInplace
        + ZnxNormalizeFirstStepCarryOnly
        + ZnxNormalizeMiddleStepCarryOnly
        + ZnxNormalizeMiddleStep
        + ZnxNormalizeFinalStep
        + ZnxNormalizeFirstStep
        + ZnxExtractDigitAddMul
        + ZnxNormalizeDigit
        + ZnxNormalizeMiddleStepInplace
        + ZnxNormalizeFinalStepInplace,
{
    let a: VecZnxBig<&[u8], _> = a.to_ref();
    let a_vznx: VecZnx<&[u8]> = VecZnx {
        data: a.data,
        n: a.n,
        cols: a.cols,
        size: a.size,
        max_size: a.max_size,
    };

    vec_znx_normalize::<_, _, BE>(res, res_base2k, res_offset, res_col, &a_vznx, a_base2k, a_col, carry);
}

pub fn vec_znx_big_add_normal_ref<R, B: Backend<ScalarBig = i64>>(
    base2k: usize,
    res: &mut R,
    res_col: usize,
    k: usize,
    sigma: f64,
    bound: f64,
    source: &mut Source,
) where
    R: VecZnxBigToMut<B>,
{
    let mut res: VecZnxBig<&mut [u8], B> = res.to_mut();
    assert!(
        (bound.log2().ceil() as i64) < 64,
        "invalid bound: ceil(log2(bound))={} > 63",
        (bound.log2().ceil() as i64)
    );

    let limb: usize = k.div_ceil(base2k) - 1;
    let scale: f64 = (1 << ((limb + 1) * base2k - k)) as f64;
    znx_add_normal_f64_ref(res.at_mut(res_col, limb), sigma * scale, bound * scale, source)
}

pub fn test_vec_znx_big_add_normal<B>(module: &Module<B>)
where
    Module<B>: VecZnxBigAddNormal<B>,
    B: Backend<ScalarBig = i64> + VecZnxBigAllocBytesImpl,
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
        let mut a: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(n, cols, size);
        module.vec_znx_big_add_normal(base2k, &mut a, col_i, k, &mut source, sigma, bound);
        module.vec_znx_big_add_normal(base2k, &mut a, col_i, k, &mut source, sigma, bound);
        (0..cols).for_each(|col_j| {
            if col_j != col_i {
                (0..size).for_each(|limb_i| {
                    assert_eq!(a.at(col_j, limb_i), zero);
                })
            } else {
                let std: f64 = a.stats(base2k, col_i).std() * k_f64;
                assert!((std - sigma * sqrt2).abs() < 0.1, "std={} ~!= {}", std, sigma * sqrt2);
            }
        })
    });
}

/// R <- A - B
pub fn vec_znx_big_sub<R, A, B, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    BE: Backend<ScalarBig = i64> + ZnxSub + ZnxNegate + ZnxZero + ZnxCopy,
    R: VecZnxBigToMut<BE>,
    A: VecZnxBigToRef<BE>,
    B: VecZnxBigToRef<BE>,
{
    let res: VecZnxBig<&mut [u8], BE> = res.to_mut();
    let a: VecZnxBig<&[u8], BE> = a.to_ref();
    let b: VecZnxBig<&[u8], BE> = b.to_ref();

    let mut res_vznx: VecZnx<&mut [u8]> = VecZnx {
        data: res.data,
        n: res.n,
        cols: res.cols,
        size: res.size,
        max_size: res.max_size,
    };

    let a_vznx: VecZnx<&[u8]> = VecZnx {
        data: a.data,
        n: a.n,
        cols: a.cols,
        size: a.size,
        max_size: a.max_size,
    };

    let b_vznx: VecZnx<&[u8]> = VecZnx {
        data: b.data,
        n: b.n,
        cols: b.cols,
        size: b.size,
        max_size: b.max_size,
    };

    vec_znx_sub::<_, _, _, BE>(&mut res_vznx, res_col, &a_vznx, a_col, &b_vznx, b_col);
}

/// R <- A - B
pub fn vec_znx_big_sub_inplace<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarBig = i64> + ZnxSubInplace,
    R: VecZnxBigToMut<BE>,
    A: VecZnxBigToRef<BE>,
{
    let res: VecZnxBig<&mut [u8], BE> = res.to_mut();
    let a: VecZnxBig<&[u8], BE> = a.to_ref();

    let mut res_vznx: VecZnx<&mut [u8]> = VecZnx {
        data: res.data,
        n: res.n,
        cols: res.cols,
        size: res.size,
        max_size: res.max_size,
    };

    let a_vznx: VecZnx<&[u8]> = VecZnx {
        data: a.data,
        n: a.n,
        cols: a.cols,
        size: a.size,
        max_size: a.max_size,
    };

    vec_znx_sub_inplace::<_, _, BE>(&mut res_vznx, res_col, &a_vznx, a_col);
}

/// R <- B - A
pub fn vec_znx_big_sub_negate_inplace<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarBig = i64> + ZnxSubNegateInplace + ZnxNegateInplace,
    R: VecZnxBigToMut<BE>,
    A: VecZnxBigToRef<BE>,
{
    let res: VecZnxBig<&mut [u8], BE> = res.to_mut();
    let a: VecZnxBig<&[u8], BE> = a.to_ref();

    let mut res_vznx: VecZnx<&mut [u8]> = VecZnx {
        data: res.data,
        n: res.n,
        cols: res.cols,
        size: res.size,
        max_size: res.max_size,
    };

    let a_vznx: VecZnx<&[u8]> = VecZnx {
        data: a.data,
        n: a.n,
        cols: a.cols,
        size: a.size,
        max_size: a.max_size,
    };

    vec_znx_sub_negate_inplace::<_, _, BE>(&mut res_vznx, res_col, &a_vznx, a_col);
}

/// R <- A - B
pub fn vec_znx_big_sub_small_a<R, A, B, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    BE: Backend<ScalarBig = i64> + ZnxSub + ZnxNegate + ZnxZero + ZnxCopy,
    R: VecZnxBigToMut<BE>,
    A: VecZnxToRef,
    B: VecZnxBigToRef<BE>,
{
    let res: VecZnxBig<&mut [u8], BE> = res.to_mut();
    let b: VecZnxBig<&[u8], BE> = b.to_ref();

    let mut res_vznx: VecZnx<&mut [u8]> = VecZnx {
        data: res.data,
        n: res.n,
        cols: res.cols,
        size: res.size,
        max_size: res.max_size,
    };

    let b_vznx: VecZnx<&[u8]> = VecZnx {
        data: b.data,
        n: b.n,
        cols: b.cols,
        size: b.size,
        max_size: b.max_size,
    };

    vec_znx_sub::<_, _, _, BE>(&mut res_vznx, res_col, a, a_col, &b_vznx, b_col);
}

/// R <- A - B
pub fn vec_znx_big_sub_small_b<R, A, B, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    BE: Backend<ScalarBig = i64> + ZnxSub + ZnxNegate + ZnxZero + ZnxCopy,
    R: VecZnxBigToMut<BE>,
    A: VecZnxBigToRef<BE>,
    B: VecZnxToRef,
{
    let res: VecZnxBig<&mut [u8], BE> = res.to_mut();
    let a: VecZnxBig<&[u8], BE> = a.to_ref();

    let mut res_vznx: VecZnx<&mut [u8]> = VecZnx {
        data: res.data,
        n: res.n,
        cols: res.cols,
        size: res.size,
        max_size: res.max_size,
    };

    let a_vznx: VecZnx<&[u8]> = VecZnx {
        data: a.data,
        n: a.n,
        cols: a.cols,
        size: a.size,
        max_size: a.max_size,
    };

    vec_znx_sub::<_, _, _, BE>(&mut res_vznx, res_col, &a_vznx, a_col, b, b_col);
}

///  R <- R - A
pub fn vec_znx_big_sub_small_a_inplace<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarBig = i64> + ZnxSubInplace,
    R: VecZnxBigToMut<BE>,
    A: VecZnxToRef,
{
    let res: VecZnxBig<&mut [u8], BE> = res.to_mut();

    let mut res_vznx: VecZnx<&mut [u8]> = VecZnx {
        data: res.data,
        n: res.n,
        cols: res.cols,
        size: res.size,
        max_size: res.max_size,
    };

    vec_znx_sub_inplace::<_, _, BE>(&mut res_vznx, res_col, a, a_col);
}

/// R <- A - R
pub fn vec_znx_big_sub_small_b_inplace<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarBig = i64> + ZnxSubNegateInplace + ZnxNegateInplace,
    R: VecZnxBigToMut<BE>,
    A: VecZnxToRef,
{
    let res: VecZnxBig<&mut [u8], BE> = res.to_mut();

    let mut res_vznx: VecZnx<&mut [u8]> = VecZnx {
        data: res.data,
        n: res.n,
        cols: res.cols,
        size: res.size,
        max_size: res.max_size,
    };

    vec_znx_sub_negate_inplace::<_, _, BE>(&mut res_vznx, res_col, a, a_col);
}
