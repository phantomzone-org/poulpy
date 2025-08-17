use rand_distr::Distribution;
use sampling::source::Source;

use crate::hal::layouts::{Backend, Scratch, VecZnxBigOwned, VecZnxBigToMut, VecZnxBigToRef, VecZnxToMut, VecZnxToRef};

/// Allocates as [crate::hal::layouts::VecZnxBig].
pub trait VecZnxBigAlloc<B: Backend> {
    fn vec_znx_big_alloc(&self, n: usize, cols: usize, size: usize) -> VecZnxBigOwned<B>;
}

/// Returns the size in bytes to allocate a [crate::hal::layouts::VecZnxBig].
pub trait VecZnxBigAllocBytes {
    fn vec_znx_big_alloc_bytes(&self, n: usize, cols: usize, size: usize) -> usize;
}

/// Consume a vector of bytes into a [crate::hal::layouts::VecZnxBig].
/// User must ensure that bytes is memory aligned and that it length is equal to [VecZnxBigAllocBytes].
pub trait VecZnxBigFromBytes<B: Backend> {
    fn vec_znx_big_from_bytes(&self, n: usize, cols: usize, size: usize, bytes: Vec<u8>) -> VecZnxBigOwned<B>;
}

#[allow(clippy::too_many_arguments)]
/// Add a discrete normal distribution on res.
///
/// # Arguments
/// * `basek`: base two logarithm of the bivariate representation
/// * `res`: receiver.
/// * `res_col`: column of the receiver on which the operation is performed/stored.
/// * `k`:
/// * `source`: random coin source.
/// * `sigma`: standard deviation of the discrete normal distribution.
/// * `bound`: rejection sampling bound.
pub trait VecZnxBigAddNormal<B: Backend> {
    fn vec_znx_big_add_normal<R: VecZnxBigToMut<B>>(
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

#[allow(clippy::too_many_arguments)]
pub trait VecZnxBigFillNormal<B: Backend> {
    fn vec_znx_big_fill_normal<R: VecZnxBigToMut<B>>(
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

#[allow(clippy::too_many_arguments)]
pub trait VecZnxBigFillDistF64<B: Backend> {
    fn vec_znx_big_fill_dist_f64<R: VecZnxBigToMut<B>, D: Distribution<f64>>(
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

#[allow(clippy::too_many_arguments)]
pub trait VecZnxBigAddDistF64<B: Backend> {
    fn vec_znx_big_add_dist_f64<R: VecZnxBigToMut<B>, D: Distribution<f64>>(
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

pub trait VecZnxBigAdd<B: Backend> {
    /// Adds `a` to `b` and stores the result on `c`.
    fn vec_znx_big_add<R, A, C>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>,
        C: VecZnxBigToRef<B>;
}

pub trait VecZnxBigAddInplace<B: Backend> {
    /// Adds `a` to `b` and stores the result on `b`.
    fn vec_znx_big_add_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>;
}

pub trait VecZnxBigAddSmall<B: Backend> {
    /// Adds `a` to `b` and stores the result on `c`.
    fn vec_znx_big_add_small<R, A, C>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>,
        C: VecZnxToRef;
}

pub trait VecZnxBigAddSmallInplace<B: Backend> {
    /// Adds `a` to `b` and stores the result on `b`.
    fn vec_znx_big_add_small_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxToRef;
}

pub trait VecZnxBigSub<B: Backend> {
    /// Subtracts `a` to `b` and stores the result on `c`.
    fn vec_znx_big_sub<R, A, C>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>,
        C: VecZnxBigToRef<B>;
}

pub trait VecZnxBigSubABInplace<B: Backend> {
    /// Subtracts `a` from `b` and stores the result on `b`.
    fn vec_znx_big_sub_ab_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>;
}

pub trait VecZnxBigSubBAInplace<B: Backend> {
    /// Subtracts `b` from `a` and stores the result on `b`.
    fn vec_znx_big_sub_ba_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>;
}

pub trait VecZnxBigSubSmallA<B: Backend> {
    /// Subtracts `b` from `a` and stores the result on `c`.
    fn vec_znx_big_sub_small_a<R, A, C>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxToRef,
        C: VecZnxBigToRef<B>;
}

pub trait VecZnxBigSubSmallAInplace<B: Backend> {
    /// Subtracts `a` from `res` and stores the result on `res`.
    fn vec_znx_big_sub_small_a_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxToRef;
}

pub trait VecZnxBigSubSmallB<B: Backend> {
    /// Subtracts `b` from `a` and stores the result on `c`.
    fn vec_znx_big_sub_small_b<R, A, C>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>,
        C: VecZnxToRef;
}

pub trait VecZnxBigSubSmallBInplace<B: Backend> {
    /// Subtracts `res` from `a` and stores the result on `res`.
    fn vec_znx_big_sub_small_b_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxToRef;
}

pub trait VecZnxBigNegateInplace<B: Backend> {
    fn vec_znx_big_negate_inplace<A>(&self, a: &mut A, a_col: usize)
    where
        A: VecZnxBigToMut<B>;
}

pub trait VecZnxBigNormalizeTmpBytes {
    fn vec_znx_big_normalize_tmp_bytes(&self, n: usize) -> usize;
}

pub trait VecZnxBigNormalize<B: Backend> {
    fn vec_znx_big_normalize<R, A>(
        &self,
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

pub trait VecZnxBigAutomorphism<B: Backend> {
    /// Applies the automorphism X^i -> X^ik on `a` and stores the result on `b`.
    fn vec_znx_big_automorphism<R, A>(&self, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>;
}

pub trait VecZnxBigAutomorphismInplace<B: Backend> {
    /// Applies the automorphism X^i -> X^ik on `a` and stores the result on `a`.
    fn vec_znx_big_automorphism_inplace<A>(&self, k: i64, a: &mut A, a_col: usize)
    where
        A: VecZnxBigToMut<B>;
}
