use rand_distr::Distribution;
use sampling::source::Source;

use crate::{Backend, Module, Scratch, VecZnxBigOwned, VecZnxBigToMut, VecZnxBigToRef, VecZnxToMut, VecZnxToRef};

pub trait VecZnxBigAllocImpl<B: Backend> {
    fn vec_znx_big_alloc_impl(module: &Module<B>, cols: usize, size: usize) -> VecZnxBigOwned<B>;
}

pub trait VecZnxBigFromBytesImpl<B: Backend> {
    fn vec_znx_big_from_bytes_impl(module: &Module<B>, cols: usize, size: usize, bytes: Vec<u8>) -> VecZnxBigOwned<B>;
}

pub trait VecZnxBigAllocBytesImpl<B: Backend> {
    fn vec_znx_big_alloc_bytes_impl(module: &Module<B>, cols: usize, size: usize) -> usize;
}

pub trait VecZnxBigAddNormalImpl<B: Backend> {
    fn add_normal_impl<R: VecZnxBigToMut<B>>(
        module: &Module<B>,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        sigma: f64,
        bound: f64,
    );
}

pub trait VecZnxBigFillNormalImpl<B: Backend> {
    fn fill_normal_impl<R: VecZnxBigToMut<B>>(
        module: &Module<B>,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        sigma: f64,
        bound: f64,
    );
}

pub trait VecZnxBigFillDistF64Impl<B: Backend> {
    fn fill_dist_f64_impl<R: VecZnxBigToMut<B>, D: Distribution<f64>>(
        module: &Module<B>,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        dist: D,
        bound: f64,
    );
}

pub trait VecZnxBigAddDistF64Impl<B: Backend> {
    fn add_dist_f64_impl<R: VecZnxBigToMut<B>, D: Distribution<f64>>(
        module: &Module<B>,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        dist: D,
        bound: f64,
    );
}

pub trait VecZnxBigAddImpl<B: Backend> {
    fn vec_znx_big_add_impl<R, A, C>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>,
        C: VecZnxBigToRef<B>;
}

pub trait VecZnxBigAddInplaceImpl<B: Backend> {
    fn vec_znx_big_add_inplace_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>;
}

pub trait VecZnxBigAddSmallImpl<B: Backend> {
    fn vec_znx_big_add_small_impl<R, A, C>(
        module: &Module<B>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &C,
        b_col: usize,
    ) where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>,
        C: VecZnxToRef;
}

pub trait VecZnxBigAddSmallInplaceImpl<B: Backend> {
    fn vec_znx_big_add_small_inplace_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxToRef;
}

pub trait VecZnxBigSubImpl<B: Backend> {
    fn vec_znx_big_sub_impl<R, A, C>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>,
        C: VecZnxBigToRef<B>;
}

pub trait VecZnxBigSubABInplaceImpl<B: Backend> {
    fn vec_znx_big_sub_ab_inplace_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>;
}

pub trait VecZnxBigSubBAInplaceImpl<B: Backend> {
    fn vec_znx_big_sub_ba_inplace_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>;
}

pub trait VecZnxBigSubSmallAImpl<B: Backend> {
    fn vec_znx_big_sub_small_a_impl<R, A, C>(
        module: &Module<B>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &C,
        b_col: usize,
    ) where
        R: VecZnxBigToMut<B>,
        A: VecZnxToRef,
        C: VecZnxBigToRef<B>;
}

pub trait VecZnxBigSubSmallAInplaceImpl<B: Backend> {
    fn vec_znx_big_sub_small_a_inplace_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxToRef;
}

pub trait VecZnxBigSubSmallBImpl<B: Backend> {
    fn vec_znx_big_sub_small_b_impl<R, A, C>(
        module: &Module<B>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &C,
        b_col: usize,
    ) where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>,
        C: VecZnxToRef;
}

pub trait VecZnxBigSubSmallBInplaceImpl<B: Backend> {
    fn vec_znx_big_sub_small_b_inplace_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxToRef;
}

pub trait VecZnxBigNegateInplaceImpl<B: Backend> {
    fn vec_znx_big_negate_inplace_impl<A>(module: &Module<B>, a: &mut A, a_col: usize)
    where
        A: VecZnxBigToMut<B>;
}

pub trait VecZnxBigNormalizeTmpBytesImpl<B: Backend> {
    fn vec_znx_big_normalize_tmp_bytes_impl(module: &Module<B>) -> usize;
}

pub trait VecZnxBigNormalizeImpl<B: Backend> {
    fn vec_znx_big_normalize_impl<R, A>(
        module: &Module<B>,
        basek: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch,
    ) where
        R: VecZnxToMut,
        A: VecZnxBigToRef<B>;
}

pub trait VecZnxBigAutomorphismImpl<B: Backend> {
    fn vec_znx_big_automorphism_impl<R, A>(module: &Module<B>, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>;
}

pub trait VecZnxBigAutomorphismInplaceImpl<B: Backend> {
    fn vec_znx_big_automorphism_inplace_impl<A>(module: &Module<B>, k: i64, a: &mut A, a_col: usize)
    where
        A: VecZnxBigToMut<B>;
}
