use rand_distr::Distribution;
use sampling::source::Source;

use crate::{Backend, Scratch, VecZnxBigOwned, VecZnxBigToMut, VecZnxBigToRef, VecZnxToMut, VecZnxToRef};

pub trait VecZnxBigNew<B: Backend> {
    fn new_vec_znx_big(&self, cols: usize, size: usize) -> VecZnxBigOwned<B>;
}

pub trait VecZnxBigFromBytes<B: Backend> {
    fn new_vec_znx_big_from_bytes(&self, cols: usize, size: usize, bytes: Vec<u8>) -> VecZnxBigOwned<B>;
}

pub trait VecZnxBigAllocBytes {
    fn vec_znx_big_alloc_bytes(&self, cols: usize, size: usize) -> usize;
}

pub trait VecZnxBigAddNormal<B: Backend> {
    fn add_normal<R: VecZnxBigToMut<B>>(
        &self,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        sigma: f64,
        bound: f64,
    );
}

pub trait VecZnxBigFillNormal<B: Backend> {
    fn fill_normal<R: VecZnxBigToMut<B>>(
        &self,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        sigma: f64,
        bound: f64,
    );
}

pub trait VecZnxBigFillDistF64<B: Backend> {
    fn fill_dist_f64<R: VecZnxBigToMut<B>, D: Distribution<f64>>(
        &self,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        dist: D,
        bound: f64,
    );
}

pub trait VecZnxBigAddDistF64<B: Backend> {
    fn add_dist_f64<R: VecZnxBigToMut<B>, D: Distribution<f64>>(
        &self,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        dist: D,
        bound: f64,
    );
}

pub trait VecZnxBigAdd<BACKEND: Backend> {
    /// Adds `a` to `b` and stores the result on `c`.
    fn vec_znx_big_add<R, A, B>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
    where
        R: VecZnxBigToMut<BACKEND>,
        A: VecZnxBigToRef<BACKEND>,
        B: VecZnxBigToRef<BACKEND>;
}

pub trait VecZnxBigAddInplace<BACKEND: Backend> {
    /// Adds `a` to `b` and stores the result on `b`.
    fn vec_znx_big_add_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<BACKEND>,
        A: VecZnxBigToRef<BACKEND>;
}

pub trait VecZnxBigAddSmall<BACKEND: Backend> {
    /// Adds `a` to `b` and stores the result on `c`.
    fn vec_znx_big_add_small<R, A, B>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
    where
        R: VecZnxBigToMut<BACKEND>,
        A: VecZnxBigToRef<BACKEND>,
        B: VecZnxToRef;
}

pub trait VecZnxBigAddSmallInplace<BACKEND: Backend> {
    /// Adds `a` to `b` and stores the result on `b`.
    fn vec_znx_big_add_small_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<BACKEND>,
        A: VecZnxToRef;
}

pub trait VecZnxBigSub<BACKEND: Backend> {
    /// Subtracts `a` to `b` and stores the result on `c`.
    fn vec_znx_big_sub<R, A, B>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
    where
        R: VecZnxBigToMut<BACKEND>,
        A: VecZnxBigToRef<BACKEND>,
        B: VecZnxBigToRef<BACKEND>;
}

pub trait VecZnxBigSubABInplace<BACKEND: Backend> {
    /// Subtracts `a` from `b` and stores the result on `b`.
    fn vec_znx_big_sub_ab_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<BACKEND>,
        A: VecZnxBigToRef<BACKEND>;
}

pub trait VecZnxBigSubBAInplace<BACKEND: Backend> {
    /// Subtracts `b` from `a` and stores the result on `b`.
    fn vec_znx_big_sub_ba_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<BACKEND>,
        A: VecZnxBigToRef<BACKEND>;
}

pub trait VecZnxBigSubSmallA<BACKEND: Backend> {
    /// Subtracts `b` from `a` and stores the result on `c`.
    fn vec_znx_big_sub_small_a<R, A, B>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
    where
        R: VecZnxBigToMut<BACKEND>,
        A: VecZnxToRef,
        B: VecZnxBigToRef<BACKEND>;
}

pub trait VecZnxBigSubSmallAInplace<BACKEND: Backend> {
    /// Subtracts `a` from `res` and stores the result on `res`.
    fn vec_znx_big_sub_small_a_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<BACKEND>,
        A: VecZnxToRef;
}

pub trait VecZnxBigSubSmallB<BACKEND: Backend> {
    /// Subtracts `b` from `a` and stores the result on `c`.
    fn vec_znx_big_sub_small_b<R, A, B>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
    where
        R: VecZnxBigToMut<BACKEND>,
        A: VecZnxBigToRef<BACKEND>,
        B: VecZnxToRef;
}

pub trait VecZnxBigSubSmallBInplace<BACKEND: Backend> {
    /// Subtracts `res` from `a` and stores the result on `res`.
    fn vec_znx_big_sub_small_b_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<BACKEND>,
        A: VecZnxToRef;
}

pub trait VecZnxBigNegateInplace<BACKEND: Backend> {
    fn vec_znx_big_negate_inplace<A>(&self, a: &mut A, a_col: usize)
    where
        A: VecZnxBigToMut<BACKEND>;
}

pub trait VecZnxBigNormalize<BACKEND: Backend> {
    fn vec_znx_big_normalize_scratch_bytes(&self) -> usize;
    fn vec_znx_big_normalize<R, A>(&self, basek: usize, res: &mut R, res_col: usize, a: &A, a_col: usize, scratch: &mut Scratch)
    where
        R: VecZnxToMut,
        A: VecZnxBigToRef<BACKEND>;
}

pub trait VecZnxBigAutomorphism<BACKEND: Backend> {
    /// Applies the automorphism X^i -> X^ik on `a` and stores the result on `b`.
    fn vec_znx_big_automorphism<R, A>(&self, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<BACKEND>,
        A: VecZnxBigToRef<BACKEND>;
}

pub trait VecZnxBigAutomorphismInplace<BACKEND: Backend> {
    /// Applies the automorphism X^i -> X^ik on `a` and stores the result on `a`.
    fn vec_znx_big_automorphism_inplace<A>(&self, k: i64, a: &mut A, a_col: usize)
    where
        A: VecZnxBigToMut<BACKEND>;
}
