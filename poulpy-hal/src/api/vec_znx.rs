use crate::{
    layouts::{Backend, ScalarZnxToRef, Scratch, VecZnxToMut, VecZnxToRef},
    source::Source,
};

pub trait VecZnxNormalizeTmpBytes {
    /// Returns the minimum number of bytes necessary for normalization.
    fn vec_znx_normalize_tmp_bytes(&self) -> usize;
}

pub trait VecZnxNormalize<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    /// Normalizes the selected column of `a` and stores the result into the selected column of `res`.
    fn vec_znx_normalize<R, A>(
        &self,
        res_basek: usize,
        res: &mut R,
        res_col: usize,
        a_basek: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxNormalizeInplace<B: Backend> {
    /// Normalizes the selected column of `a`.
    fn vec_znx_normalize_inplace<A>(&self, base2k: usize, a: &mut A, a_col: usize, scratch: &mut Scratch<B>)
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

pub trait VecZnxAddScalar {
    /// Adds the selected column of `a` on the selected column and limb of `b` and writes the result on the selected column of `res`.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_add_scalar<R, A, B>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize, b_limb: usize)
    where
        R: VecZnxToMut,
        A: ScalarZnxToRef,
        B: VecZnxToRef;
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

pub trait VecZnxSubInplace {
    /// Subtracts the selected column of `a` from the selected column of `res` inplace.
    ///
    /// res\[res_col\] -= a\[a_col\]
    fn vec_znx_sub_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxSubNegateInplace {
    /// Subtracts the selected column of `res` from the selected column of `a` and inplace mutates `res`
    ///
    /// res\[res_col\] = a\[a_col\] - res\[res_col\]
    fn vec_znx_sub_negate_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxSubScalar {
    /// Subtracts the selected column of `a` on the selected column and limb of `b` and writes the result on the selected column of `res`.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_sub_scalar<R, A, B>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize, b_limb: usize)
    where
        R: VecZnxToMut,
        A: ScalarZnxToRef,
        B: VecZnxToRef;
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

pub trait VecZnxLshTmpBytes {
    fn vec_znx_lsh_tmp_bytes(&self) -> usize;
}

pub trait VecZnxLsh<B: Backend> {
    /// Left shift by k bits all columns of `a`.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_lsh<R, A>(
        &self,
        base2k: usize,
        k: usize,
        r: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxRshTmpBytes {
    fn vec_znx_rsh_tmp_bytes(&self) -> usize;
}

pub trait VecZnxRsh<B: Backend> {
    /// Right shift by k bits all columns of `a`.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_rsh<R, A>(
        &self,
        base2k: usize,
        k: usize,
        r: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxLshInplace<B: Backend> {
    /// Left shift by k bits all columns of `a`.
    fn vec_znx_lsh_inplace<A>(&self, base2k: usize, k: usize, a: &mut A, a_col: usize, scratch: &mut Scratch<B>)
    where
        A: VecZnxToMut;
}

pub trait VecZnxRshInplace<B: Backend> {
    /// Right shift by k bits all columns of `a`.
    fn vec_znx_rsh_inplace<A>(&self, base2k: usize, k: usize, a: &mut A, a_col: usize, scratch: &mut Scratch<B>)
    where
        A: VecZnxToMut;
}

pub trait VecZnxRotate {
    /// Multiplies the selected column of `a` by X^k and stores the result in `res_col` of `res`.
    fn vec_znx_rotate<R, A>(&self, p: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxRotateInplaceTmpBytes {
    fn vec_znx_rotate_inplace_tmp_bytes(&self) -> usize;
}

pub trait VecZnxRotateInplace<B: Backend> {
    /// Multiplies the selected column of `a` by X^k.
    fn vec_znx_rotate_inplace<A>(&self, p: i64, a: &mut A, a_col: usize, scratch: &mut Scratch<B>)
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

pub trait VecZnxAutomorphismInplaceTmpBytes {
    fn vec_znx_automorphism_inplace_tmp_bytes(&self) -> usize;
}

pub trait VecZnxAutomorphismInplace<B: Backend> {
    /// Applies the automorphism X^i -> X^ik on the selected column of `a`.
    fn vec_znx_automorphism_inplace<R>(&self, k: i64, res: &mut R, res_col: usize, scratch: &mut Scratch<B>)
    where
        R: VecZnxToMut;
}

pub trait VecZnxMulXpMinusOne {
    fn vec_znx_mul_xp_minus_one<R, A>(&self, p: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxMulXpMinusOneInplaceTmpBytes {
    fn vec_znx_mul_xp_minus_one_inplace_tmp_bytes(&self) -> usize;
}

pub trait VecZnxMulXpMinusOneInplace<B: Backend> {
    fn vec_znx_mul_xp_minus_one_inplace<R>(&self, p: i64, res: &mut R, res_col: usize, scratch: &mut Scratch<B>)
    where
        R: VecZnxToMut;
}

pub trait VecZnxSplitRingTmpBytes {
    fn vec_znx_split_ring_tmp_bytes(&self) -> usize;
}

pub trait VecZnxSplitRing<B: Backend> {
    /// Splits the selected columns of `b` into subrings and copies them them into the selected column of `res`.
    ///
    /// # Panics
    ///
    /// This method requires that all [crate::layouts::VecZnx] of b have the same ring degree
    /// and that b.n() * b.len() <= a.n()
    fn vec_znx_split_ring<R, A>(&self, res: &mut [R], res_col: usize, a: &A, a_col: usize, scratch: &mut Scratch<B>)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxMergeRingsTmpBytes {
    fn vec_znx_merge_rings_tmp_bytes(&self) -> usize;
}

pub trait VecZnxMergeRings<B: Backend> {
    /// Merges the subrings of the selected column of `a` into the selected column of `res`.
    ///
    /// # Panics
    ///
    /// This method requires that all [crate::layouts::VecZnx] of a have the same ring degree
    /// and that a.n() * a.len() <= b.n()
    fn vec_znx_merge_rings<R, A>(&self, res: &mut R, res_col: usize, a: &[A], a_col: usize, scratch: &mut Scratch<B>)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxSwitchRing {
    fn vec_znx_switch_ring<R, A>(&self, res: &mut R, res_col: usize, a: &A, col_a: usize)
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

pub trait VecZnxFillUniform {
    /// Fills the first `size` size with uniform values in \[-2^{base2k-1}, 2^{base2k-1}\]
    fn vec_znx_fill_uniform<R>(&self, base2k: usize, res: &mut R, res_col: usize, source: &mut Source)
    where
        R: VecZnxToMut;
}

#[allow(clippy::too_many_arguments)]
pub trait VecZnxFillNormal {
    fn vec_znx_fill_normal<R>(
        &self,
        base2k: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        sigma: f64,
        bound: f64,
    ) where
        R: VecZnxToMut;
}

#[allow(clippy::too_many_arguments)]
pub trait VecZnxAddNormal {
    /// Adds a discrete normal vector scaled by 2^{-k} with the provided standard deviation and bounded to \[-bound, bound\].
    fn vec_znx_add_normal<R>(
        &self,
        base2k: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        sigma: f64,
        bound: f64,
    ) where
        R: VecZnxToMut;
}
