use rand_distr::Distribution;
use rug::Float;
use sampling::source::Source;

use crate::{ScalarZnxToRef, Scratch, VecZnxOwned, VecZnxToMut, VecZnxToRef};

pub trait VecZnxAlloc {
    /// Allocates a new [VecZnx].
    ///
    /// # Arguments
    ///
    /// * `cols`: the number of polynomials.
    /// * `size`: the number small polynomials per column.
    fn vec_znx_alloc(&self, cols: usize, size: usize) -> VecZnxOwned;
}

pub trait VecZnxFromBytes {
    /// Instantiates a new [VecZnx] from a slice of bytes.
    /// The returned [VecZnx] takes ownership of the slice of bytes.
    ///
    /// # Arguments
    ///
    /// * `cols`: the number of polynomials.
    /// * `size`: the number small polynomials per column.
    ///
    /// # Panic
    /// Requires the slice of bytes to be equal to [VecZnxOps::bytes_of_vec_znx].
    fn vec_znx_from_bytes(&self, cols: usize, size: usize, bytes: Vec<u8>) -> VecZnxOwned;
}

pub trait VecZnxAllocBytes {
    /// Returns the number of bytes necessary to allocate
    /// a new [VecZnx] through [VecZnxOps::new_vec_znx_from_bytes]
    /// or [VecZnxOps::new_vec_znx_from_bytes_borrow].
    fn vec_znx_alloc_bytes(&self, cols: usize, size: usize) -> usize;
}

pub trait VecZnxNormalizeTmpBytes {
    /// Returns the minimum number of bytes necessary for normalization.
    fn vec_znx_normalize_tmp_bytes(&self) -> usize;
}

pub trait VecZnxNormalize {
    /// Normalizes the selected column of `a` and stores the result into the selected column of `res`.
    fn vec_znx_normalize<R, A>(&self, basek: usize, res: &mut R, res_col: usize, a: &A, a_col: usize, scratch: &mut Scratch)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxNormalizeInplace {
    /// Normalizes the selected column of `a`.
    fn vec_znx_normalize_inplace<A>(&self, basek: usize, a: &mut A, a_col: usize, scratch: &mut Scratch)
    where
        A: VecZnxToMut;
}

pub trait VecZnxAdd {
    /// Adds the selected column of `a` to the selected column of `b` and writes the result on the selected column of `res`.
    fn vec_znx_add<R, A, B>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
        B: VecZnxToRef;
}

pub trait VecZnxAddInplace {
    /// Adds the selected column of `a` to the selected column of `res` and writes the result on the selected column of `res`.
    fn vec_znx_add_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxAddScalarInplace {
    /// Adds the selected column of `a` on the selected column and limb of `res`.
    fn vec_znx_add_scalar_inplace<R, A>(&self, res: &mut R, res_col: usize, res_limb: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: ScalarZnxToRef;
}

pub trait VecZnxSub {
    /// Subtracts the selected column of `b` from the selected column of `a` and writes the result on the selected column of `res`.
    fn vec_znx_sub<R, A, B>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
        B: VecZnxToRef;
}

pub trait VecZnxSubABInplace {
    /// Subtracts the selected column of `a` from the selected column of `res` inplace.
    ///
    /// res[res_col] -= a[a_col]
    fn vec_znx_sub_ab_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxSubBAInplace {
    /// Subtracts the selected column of `res` from the selected column of `a` and inplace mutates `res`
    ///
    /// res[res_col] = a[a_col] - res[res_col]
    fn vec_znx_sub_ba_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxSubScalarInplace {
    /// Subtracts the selected column of `a` on the selected column and limb of `res`.
    fn vec_znx_sub_scalar_inplace<R, A>(&self, res: &mut R, res_col: usize, res_limb: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: ScalarZnxToRef;
}

pub trait VecZnxNegate {
    // Negates the selected column of `a` and stores the result in `res_col` of `res`.
    fn vec_znx_negate<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxNegateInplace {
    /// Negates the selected column of `a`.
    fn vec_znx_negate_inplace<A>(&self, a: &mut A, a_col: usize)
    where
        A: VecZnxToMut;
}

pub trait VecZnxShiftInplace {
    /// Shifts by k bits all columns of `a`.
    /// A positive k applies a left shift, while a negative k applies a right shift.
    fn vec_znx_shift_inplace<A>(&self, basek: usize, k: i64, a: &mut A, scratch: &mut Scratch)
    where
        A: VecZnxToMut;
}

pub trait VecZnxRotate {
    /// Multiplies the selected column of `a` by X^k and stores the result in `res_col` of `res`.
    fn vec_znx_rotate<R, A>(&self, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxRotateInplace {
    /// Multiplies the selected column of `a` by X^k.
    fn vec_znx_rotate_inplace<A>(&self, k: i64, a: &mut A, a_col: usize)
    where
        A: VecZnxToMut;
}

pub trait VecZnxAutomorphism {
    /// Applies the automorphism X^i -> X^ik on the selected column of `a` and stores the result in `res_col` column of `res`.
    fn vec_znx_automorphism<R, A>(&self, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxAutomorphismInplace {
    /// Applies the automorphism X^i -> X^ik on the selected column of `a`.
    fn vec_znx_automorphism_inplace<A>(&self, k: i64, a: &mut A, a_col: usize)
    where
        A: VecZnxToMut;
}

pub trait VecZnxSplit {
    /// Splits the selected columns of `b` into subrings and copies them them into the selected column of `res`.
    ///
    /// # Panics
    ///
    /// This method requires that all [VecZnx] of b have the same ring degree
    /// and that b.n() * b.len() <= a.n()
    fn vec_znx_split<R, A>(&self, res: &mut Vec<R>, res_col: usize, a: &A, a_col: usize, scratch: &mut Scratch)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxMerge {
    /// Merges the subrings of the selected column of `a` into the selected column of `res`.
    ///
    /// # Panics
    ///
    /// This method requires that all [VecZnx] of a have the same ring degree
    /// and that a.n() * a.len() <= b.n()
    fn vec_znx_merge<R, A>(&self, res: &mut R, res_col: usize, a: Vec<A>, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxSwithcDegree {
    fn vec_znx_switch_degree<R, A>(&self, res: &mut R, res_col: usize, a: &A, col_a: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxCopy {
    fn vec_znx_copy<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxStd {
    /// Returns the standard devaition of the i-th polynomial.
    fn vec_znx_std<A>(&self, basek: usize, a: &A, a_col: usize) -> f64
    where
        A: VecZnxToRef;
}

pub trait VecZnxFillUniform {
    /// Fills the first `size` size with uniform values in \[-2^{basek-1}, 2^{basek-1}\]
    fn vec_znx_fill_uniform<R>(&self, basek: usize, res: &mut R, res_col: usize, k: usize, source: &mut Source)
    where
        R: VecZnxToMut;
}

pub trait VecZnxFillDistF64 {
    fn vec_znx_fill_dist_f64<R, D: Distribution<f64>>(
        &self,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        dist: D,
        bound: f64,
    ) where
        R: VecZnxToMut;
}

pub trait VecZnxAddDistF64 {
    /// Adds vector sampled according to the provided distribution, scaled by 2^{-k} and bounded to \[-bound, bound\].
    fn vec_znx_add_dist_f64<R, D: Distribution<f64>>(
        &self,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        dist: D,
        bound: f64,
    ) where
        R: VecZnxToMut;
}

pub trait VecZnxFillNormal {
    fn vec_znx_fill_normal<R>(
        &self,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        sigma: f64,
        bound: f64,
    ) where
        R: VecZnxToMut;
}

pub trait VecZnxAddNormal {
    /// Adds a discrete normal vector scaled by 2^{-k} with the provided standard deviation and bounded to \[-bound, bound\].
    fn vec_znx_add_normal<R>(
        &self,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        sigma: f64,
        bound: f64,
    ) where
        R: VecZnxToMut;
}

pub trait VecZnxEncodeVeci64 {
    /// encode a vector of i64 on the receiver.
    ///
    /// # Arguments
    ///
    /// * `col_i`: the index of the poly where to encode the data.
    /// * `basek`: base two negative logarithm decomposition of the receiver.
    /// * `k`: base two negative logarithm of the scaling of the data.
    /// * `data`: data to encode on the receiver.
    /// * `log_max`: base two logarithm of the infinity norm of the input data.
    fn encode_vec_i64<R>(&self, basek: usize, res: &mut R, res_col: usize, k: usize, data: &[i64], log_max: usize)
    where
        R: VecZnxToMut;
}

pub trait VecZnxEncodeCoeffsi64 {
    /// encodes a single i64 on the receiver at the given index.
    ///
    /// # Arguments
    ///
    /// * `res_col`: the index of the poly where to encode the data.
    /// * `basek`: base two negative logarithm decomposition of the receiver.
    /// * `k`: base two negative logarithm of the scaling of the data.
    /// * `i`: index of the coefficient on which to encode the data.
    /// * `data`: data to encode on the receiver.
    /// * `log_max`: base two logarithm of the infinity norm of the input data.
    fn encode_coeff_i64<R>(&self, basek: usize, res: &mut R, res_col: usize, k: usize, i: usize, data: i64, log_max: usize)
    where
        R: VecZnxToMut;
}

pub trait VecZnxDecodeVeci64 {
    /// decode a vector of i64 from the receiver.
    ///
    /// # Arguments
    ///
    /// * `res_col`: the index of the poly where to encode the data.
    /// * `basek`: base two negative logarithm decomposition of the receiver.
    /// * `k`: base two logarithm of the scaling of the data.
    /// * `data`: data to decode from the receiver.
    fn decode_vec_i64<R>(&self, basek: usize, res: &R, res_col: usize, k: usize, data: &mut [i64])
    where
        R: VecZnxToRef;
}

pub trait VecZnxDecodeCoeffsi64 {
    /// decode a single of i64 from the receiver at the given index.
    ///
    /// # Arguments
    ///
    /// * `res_col`: the index of the poly where to encode the data.
    /// * `basek`: base two negative logarithm decomposition of the receiver.
    /// * `k`: base two negative logarithm of the scaling of the data.
    /// * `i`: index of the coefficient to decode.
    /// * `data`: data to decode from the receiver.
    fn decode_coeff_i64<R>(&self, basek: usize, res: &R, res_col: usize, k: usize, i: usize) -> i64
    where
        R: VecZnxToRef;
}

pub trait VecZnxDecodeVecFloat {
    /// decode a vector of Float from the receiver.
    ///
    /// # Arguments
    /// * `col_i`: the index of the poly where to encode the data.
    /// * `basek`: base two negative logarithm decomposition of the receiver.
    /// * `data`: data to decode from the receiver.
    fn decode_vec_float<R>(&self, basek: usize, res: &R, col_i: usize, data: &mut [Float])
    where
        R: VecZnxToRef;
}
