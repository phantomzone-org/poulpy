use crate::{
    Backend, Module, ScalarZnxToRef, Scratch,
    vec_znx::layout::{VecZnxOwned, VecZnxToMut, VecZnxToRef},
};

pub trait VecZnxAllocImpl<B: Backend> {
    /// Allocates a new [VecZnx].
    ///
    /// # Arguments
    ///
    /// * `cols`: the number of polynomials.
    /// * `size`: the number small polynomials per column.
    fn vec_znx_alloc_impl(module: &Module<B>, cols: usize, size: usize) -> VecZnxOwned;
}

pub trait VecZnxFromBytesImpl<B: Backend> {
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
    fn vec_znx_from_bytes_impl(module: &Module<B>, cols: usize, size: usize, bytes: Vec<u8>) -> VecZnxOwned;
}

pub trait VecZnxAllocBytesImpl<B: Backend> {
    /// Returns the number of bytes necessary to allocate
    /// a new [VecZnx] through [VecZnxOps::new_vec_znx_from_bytes]
    /// or [VecZnxOps::new_vec_znx_from_bytes_borrow].
    fn vec_znx_alloc_bytes_impl(module: &Module<B>, cols: usize, size: usize) -> usize;
}

pub trait VecZnxNormalizeTmpBytesImpl<B: Backend> {
    /// Returns the minimum number of bytes necessary for normalization.
    fn vec_znx_normalize_tmp_bytes_impl(module: &Module<B>) -> usize;
}

pub trait VecZnxNormalizeImpl<B: Backend> {
    /// Normalizes the selected column of `a` and stores the result into the selected column of `res`.
    fn vec_znx_normalize_impl<R, A>(
        module: &Module<B>,
        basek: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxNormalizeInplaceImpl<B: Backend> {
    /// Normalizes the selected column of `a`.
    fn vec_znx_normalize_inplace_impl<A>(module: &Module<B>, basek: usize, a: &mut A, a_col: usize, scratch: &mut Scratch)
    where
        A: VecZnxToMut;
}

pub trait VecZnxAddImpl<B: Backend> {
    fn vec_znx_add_impl<R, A, C>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
        C: VecZnxToRef;
}

pub trait VecZnxAddInplaceImpl<B: Backend> {
    /// Adds the selected column of `a` to the selected column of `res` and writes the result on the selected column of `res`.
    fn vec_znx_add_inplace_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxAddScalarInplaceImpl<B: Backend> {
    /// Adds the selected column of `a` on the selected column and limb of `res`.
    fn vec_znx_add_scalar_inplace_impl<R, A>(
        module: &Module<B>,
        res: &mut R,
        res_col: usize,
        res_limb: usize,
        a: &A,
        a_col: usize,
    ) where
        R: VecZnxToMut,
        A: ScalarZnxToRef;
}

pub trait VecZnxSubImpl<B: Backend> {
    /// Subtracts the selected column of `b` from the selected column of `a` and writes the result on the selected column of `res`.
    fn vec_znx_sub_impl<R, A, C>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
        C: VecZnxToRef;
}

pub trait VecZnxSubABInplaceImpl<B: Backend> {
    /// Subtracts the selected column of `a` from the selected column of `res` inplace.
    ///
    /// res[res_col] -= a[a_col]
    fn vec_znx_sub_ab_inplace_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxSubBAInplaceImpl<B: Backend> {
    /// Subtracts the selected column of `res` from the selected column of `a` and inplace mutates `res`
    ///
    /// res[res_col] = a[a_col] - res[res_col]
    fn vec_znx_sub_ba_inplace_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxSubScalarInplaceImpl<B: Backend> {
    /// Subtracts the selected column of `a` on the selected column and limb of `res`.
    fn vec_znx_sub_scalar_inplace_impl<R, A>(
        module: &Module<B>,
        res: &mut R,
        res_col: usize,
        res_limb: usize,
        a: &A,
        a_col: usize,
    ) where
        R: VecZnxToMut,
        A: ScalarZnxToRef;
}

pub trait VecZnxNegateImpl<B: Backend> {
    // Negates the selected column of `a` and stores the result in `res_col` of `res`.
    fn vec_znx_negate_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxNegateInplaceImpl<B: Backend> {
    /// Negates the selected column of `a`.
    fn vec_znx_negate_inplace_impl<A>(module: &Module<B>, a: &mut A, a_col: usize)
    where
        A: VecZnxToMut;
}

pub trait VecZnxShiftInplaceImpl<B: Backend> {
    /// Shifts by k bits all columns of `a`.
    /// A positive k applies a left shift, while a negative k applies a right shift.
    fn vec_znx_shift_inplace_impl<A>(module: &Module<B>, basek: usize, k: i64, a: &mut A, scratch: &mut Scratch)
    where
        A: VecZnxToMut;
}

pub trait VecZnxRotateImpl<B: Backend> {
    /// Multiplies the selected column of `a` by X^k and stores the result in `res_col` of `res`.
    fn vec_znx_rotate_impl<R, A>(module: &Module<B>, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxRotateInplaceImpl<B: Backend> {
    /// Multiplies the selected column of `a` by X^k.
    fn vec_znx_rotate_inplace_impl<A>(module: &Module<B>, k: i64, a: &mut A, a_col: usize)
    where
        A: VecZnxToMut;
}

pub trait VecZnxAutomorphismImpl<B: Backend> {
    /// Applies the automorphism X^i -> X^ik on the selected column of `a` and stores the result in `res_col` column of `res`.
    fn vec_znx_automorphism_impl<R, A>(module: &Module<B>, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxAutomorphismInplaceImpl<B: Backend> {
    /// Applies the automorphism X^i -> X^ik on the selected column of `a`.
    fn vec_znx_automorphism_inplace_impl<A>(module: &Module<B>, k: i64, a: &mut A, a_col: usize)
    where
        A: VecZnxToMut;
}

pub trait VecZnxSplitImpl<B: Backend> {
    /// Splits the selected columns of `b` into subrings and copies them them into the selected column of `res`.
    ///
    /// # Panics
    ///
    /// This method requires that all [VecZnx] of b have the same ring degree
    /// and that b.n() * b.len() <= a.n()
    fn vec_znx_split_impl<R, A>(module: &Module<B>, res: &mut Vec<R>, res_col: usize, a: &A, a_col: usize, scratch: &mut Scratch)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxMergeImpl<B: Backend> {
    /// Merges the subrings of the selected column of `a` into the selected column of `res`.
    ///
    /// # Panics
    ///
    /// This method requires that all [VecZnx] of a have the same ring degree
    /// and that a.n() * a.len() <= b.n()
    fn vec_znx_merge_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: Vec<A>, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxSwithcDegreeImpl<B: Backend> {
    fn vec_znx_switch_degree_impl<R: VecZnxToMut, A: VecZnxToRef>(
        module: &Module<B>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
    );
}

pub trait VecZnxCopyImpl<B: Backend> {
    fn vec_znx_copy_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}
