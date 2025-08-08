use rand_distr::Distribution;
use sampling::source::Source;

use crate::hal::layouts::{Backend, Module, Scratch, VecZnxBigOwned, VecZnxBigToMut, VecZnxBigToRef, VecZnxToMut, VecZnxToRef};

pub unsafe trait VecZnxBigAllocImpl<B: Backend> {
    fn vec_znx_big_alloc_impl(n: usize, cols: usize, size: usize) -> VecZnxBigOwned<B>;
}

pub unsafe trait VecZnxBigFromBytesImpl<B: Backend> {
    fn vec_znx_big_from_bytes_impl(n: usize, cols: usize, size: usize, bytes: Vec<u8>) -> VecZnxBigOwned<B>;
}

pub unsafe trait VecZnxBigAllocBytesImpl<B: Backend> {
    fn vec_znx_big_alloc_bytes_impl(n: usize, cols: usize, size: usize) -> usize;
}

pub unsafe trait VecZnxBigAddNormalImpl<B: Backend> {
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

pub unsafe trait VecZnxBigFillNormalImpl<B: Backend> {
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

pub unsafe trait VecZnxBigFillDistF64Impl<B: Backend> {
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

pub unsafe trait VecZnxBigAddDistF64Impl<B: Backend> {
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

pub unsafe trait VecZnxBigAddImpl<B: Backend> {
    fn vec_znx_big_add_impl<R, A, C>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>,
        C: VecZnxBigToRef<B>;
}

pub unsafe trait VecZnxBigAddInplaceImpl<B: Backend> {
    fn vec_znx_big_add_inplace_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>;
}

pub unsafe trait VecZnxBigAddSmallImpl<B: Backend> {
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

pub unsafe trait VecZnxBigAddSmallInplaceImpl<B: Backend> {
    fn vec_znx_big_add_small_inplace_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxToRef;
}

pub unsafe trait VecZnxBigSubImpl<B: Backend> {
    fn vec_znx_big_sub_impl<R, A, C>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>,
        C: VecZnxBigToRef<B>;
}

pub unsafe trait VecZnxBigSubABInplaceImpl<B: Backend> {
    fn vec_znx_big_sub_ab_inplace_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>;
}

pub unsafe trait VecZnxBigSubBAInplaceImpl<B: Backend> {
    fn vec_znx_big_sub_ba_inplace_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>;
}

pub unsafe trait VecZnxBigSubSmallAImpl<B: Backend> {
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

pub unsafe trait VecZnxBigSubSmallAInplaceImpl<B: Backend> {
    fn vec_znx_big_sub_small_a_inplace_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxToRef;
}

pub unsafe trait VecZnxBigSubSmallBImpl<B: Backend> {
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

pub unsafe trait VecZnxBigSubSmallBInplaceImpl<B: Backend> {
    fn vec_znx_big_sub_small_b_inplace_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxToRef;
}

pub unsafe trait VecZnxBigNegateInplaceImpl<B: Backend> {
    fn vec_znx_big_negate_inplace_impl<A>(module: &Module<B>, a: &mut A, a_col: usize)
    where
        A: VecZnxBigToMut<B>;
}

pub unsafe trait VecZnxBigNormalizeTmpBytesImpl<B: Backend> {
    fn vec_znx_big_normalize_tmp_bytes_impl(module: &Module<B>, n: usize) -> usize;
}

pub unsafe trait VecZnxBigNormalizeImpl<B: Backend> {
    fn vec_znx_big_normalize_impl<R, A>(
        module: &Module<B>,
        basek: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxBigToRef<B>;
}

pub unsafe trait VecZnxBigAutomorphismImpl<B: Backend> {
    fn vec_znx_big_automorphism_impl<R, A>(module: &Module<B>, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>;
}

pub unsafe trait VecZnxBigAutomorphismInplaceImpl<B: Backend> {
    fn vec_znx_big_automorphism_inplace_impl<A>(module: &Module<B>, k: i64, a: &mut A, a_col: usize)
    where
        A: VecZnxBigToMut<B>;
}
