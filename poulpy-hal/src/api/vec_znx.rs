use crate::{
    layouts::{
        Backend, NoiseInfos, ScalarZnx, ScalarZnxBackendRef, ScratchArena, VecZnxBackendMut, VecZnxBackendRef, VecZnxToMut,
        VecZnxToRef,
    },
    source::Source,
};

pub trait VecZnxNormalizeTmpBytes {
    /// Returns the minimum number of bytes necessary for normalization.
    fn vec_znx_normalize_tmp_bytes(&self) -> usize;
}

pub trait VecZnxZeroBackend<B: Backend> {
    fn vec_znx_zero_backend<'r>(&self, res: &mut VecZnxBackendMut<'r, B>, res_col: usize);
}

pub trait VecZnxNormalize<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    /// Normalizes the selected column of `a` and stores the result into the selected column of `res`.
    fn vec_znx_normalize<'s, 'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    );
}

pub trait VecZnxNormalizeAssign<B: Backend> {
    /// Normalizes the selected column of `a`.
    fn vec_znx_normalize_inplace<'s, A>(&self, base2k: usize, a: &mut A, a_col: usize, scratch: &mut ScratchArena<'s, B>)
    where
        A: VecZnxToMut;
}

pub trait VecZnxNormalizeInplaceBackend<B: Backend> {
    fn vec_znx_normalize_inplace_backend<'s, 'r>(
        &self,
        base2k: usize,
        a: &mut VecZnxBackendMut<'r, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    );
}

pub trait VecZnxAddIntoBackend<B: Backend> {
    /// Adds the selected backend-native column of `a` to the selected backend-native column of `b`.
    fn vec_znx_add_into_backend<'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
        b: &VecZnxBackendRef<'a, B>,
        b_col: usize,
    );
}

pub trait VecZnxAddAssignBackend<B: Backend> {
    fn vec_znx_add_assign_backend<'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
    );
}

pub trait VecZnxAddScalarInto {
    /// Adds the selected column of `a` on the selected column and limb of `b` and writes the result on the selected column of `res`.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_add_scalar_into<R, B>(
        &self,
        res: &mut R,
        res_col: usize,
        a: &ScalarZnx<&[u8]>,
        a_col: usize,
        b: &B,
        b_col: usize,
        b_limb: usize,
    ) where
        R: VecZnxToMut,
        B: VecZnxToRef;
}

pub trait VecZnxAddScalarIntoBackend<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_add_scalar_into_backend<'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'a, B>,
        a_col: usize,
        b: &VecZnxBackendRef<'a, B>,
        b_col: usize,
        b_limb: usize,
    );
}

pub trait VecZnxAddScalarAssign {
    /// Adds the selected column of `a` on the selected column and limb of `res`.
    fn vec_znx_add_scalar_assign<R>(&self, res: &mut R, res_col: usize, res_limb: usize, a: &ScalarZnx<&[u8]>, a_col: usize)
    where
        R: VecZnxToMut;
}

pub trait VecZnxAddScalarAssignBackend<B: Backend> {
    fn vec_znx_add_scalar_assign_backend<'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        res_limb: usize,
        a: &ScalarZnxBackendRef<'a, B>,
        a_col: usize,
    );
}

pub trait VecZnxSubBackend<B: Backend> {
    fn vec_znx_sub_backend<'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
        b: &VecZnxBackendRef<'a, B>,
        b_col: usize,
    );
}

pub trait VecZnxSubInplaceBackend<B: Backend> {
    fn vec_znx_sub_inplace_backend<'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
    );
}

pub trait VecZnxSubNegateInplaceBackend<B: Backend> {
    fn vec_znx_sub_negate_inplace_backend<'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
    );
}

pub trait VecZnxSubScalar {
    /// Subtracts the selected column of `a` on the selected column and limb of `b` and writes the result on the selected column of `res`.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_sub_scalar<R, B>(
        &self,
        res: &mut R,
        res_col: usize,
        a: &ScalarZnx<&[u8]>,
        a_col: usize,
        b: &B,
        b_col: usize,
        b_limb: usize,
    ) where
        R: VecZnxToMut,
        B: VecZnxToRef;
}

pub trait VecZnxSubScalarBackend<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_sub_scalar_backend<'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'a, B>,
        a_col: usize,
        b: &VecZnxBackendRef<'a, B>,
        b_col: usize,
        b_limb: usize,
    );
}

pub trait VecZnxSubScalarInplace {
    /// Subtracts the selected column of `a` on the selected column and limb of `res`.
    fn vec_znx_sub_scalar_inplace<R>(&self, res: &mut R, res_col: usize, res_limb: usize, a: &ScalarZnx<&[u8]>, a_col: usize)
    where
        R: VecZnxToMut;
}

pub trait VecZnxSubScalarInplaceBackend<B: Backend> {
    fn vec_znx_sub_scalar_inplace_backend<'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        res_limb: usize,
        a: &ScalarZnxBackendRef<'a, B>,
        a_col: usize,
    );
}

pub trait VecZnxNegateBackend<B: Backend> {
    fn vec_znx_negate_backend(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
    );
}

pub trait VecZnxNegateInplaceBackend<B: Backend> {
    fn vec_znx_negate_inplace_backend(&self, a: &mut VecZnxBackendMut<'_, B>, a_col: usize);
}

/// Returns scratch bytes required for left-shift operations.
pub trait VecZnxLshTmpBytes {
    fn vec_znx_lsh_tmp_bytes(&self) -> usize;
}

pub trait VecZnxLsh<B: Backend> {
    /// Left shift by k bits all columns of `a`.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_lsh<'s, R, A>(
        &self,
        base2k: usize,
        k: usize,
        r: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxLshAddInto<B: Backend> {
    /// Left shift by k bits all columns of `a`.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_lsh_add_into<'s, R, A>(
        &self,
        base2k: usize,
        k: usize,
        r: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

/// Returns scratch bytes required for right-shift operations.
pub trait VecZnxRshTmpBytes {
    fn vec_znx_rsh_tmp_bytes(&self) -> usize;
}

pub trait VecZnxRsh<B: Backend> {
    /// Right shift by k bits all columns of `a`.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_rsh<'s, R, A>(
        &self,
        base2k: usize,
        k: usize,
        r: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxRshAddInto<B: Backend> {
    /// Right shift by k bits all columns of `a`.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_rsh_add_into<'s, R, A>(
        &self,
        base2k: usize,
        k: usize,
        r: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxLshSub<B: Backend> {
    /// Left shift by k bits and subtract from destination.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_lsh_sub<'s, R, A>(
        &self,
        base2k: usize,
        k: usize,
        r: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxRshSub<B: Backend> {
    /// Right shift by k bits and subtract from destination.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_rsh_sub<'s, R, A>(
        &self,
        base2k: usize,
        k: usize,
        r: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxLshAssign<B: Backend> {
    /// Left shift by k bits all columns of `a`.
    fn vec_znx_lsh_inplace<'s, A>(&self, base2k: usize, k: usize, a: &mut A, a_col: usize, scratch: &mut ScratchArena<'s, B>)
    where
        A: VecZnxToMut;
}

pub trait VecZnxRshAssign<B: Backend> {
    /// Right shift by k bits all columns of `a`.
    fn vec_znx_rsh_inplace<'s, A>(&self, base2k: usize, k: usize, a: &mut A, a_col: usize, scratch: &mut ScratchArena<'s, B>)
    where
        A: VecZnxToMut;
}

pub trait VecZnxRotate<B: Backend> {
    /// Multiplies the selected column of `a` by X^k and stores the result in `res_col` of `res`.
    fn vec_znx_rotate<'r, 'a>(
        &self,
        p: i64,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
    );
}

pub trait VecZnxRotateAssignTmpBytes {
    fn vec_znx_rotate_assign_tmp_bytes(&self) -> usize;
}

pub trait VecZnxRotateAssign<B: Backend> {
    /// Multiplies the selected column of `a` by X^k.
    fn vec_znx_rotate_inplace<'s, 'r>(
        &self,
        p: i64,
        a: &mut VecZnxBackendMut<'r, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    );
}

pub trait VecZnxAutomorphism {
    /// Applies the automorphism X^i -> X^ik on the selected column of `a` and stores the result in `res_col` column of `res`.
    fn vec_znx_automorphism<R, A>(&self, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxAutomorphismAssignTmpBytes {
    fn vec_znx_automorphism_assign_tmp_bytes(&self) -> usize;
}

pub trait VecZnxAutomorphismAssign<B: Backend> {
    /// Applies the automorphism X^i -> X^ik on the selected column of `a`.
    fn vec_znx_automorphism_inplace<'s, 'r>(
        &self,
        k: i64,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    );
}

/// Multiplies the selected column by `(X^p - 1)` in `Z[X]/(X^N + 1)`.
pub trait VecZnxMulXpMinusOne {
    fn vec_znx_mul_xp_minus_one<R, A>(&self, p: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxMulXpMinusOneAssignTmpBytes {
    fn vec_znx_mul_xp_minus_one_assign_tmp_bytes(&self) -> usize;
}

pub trait VecZnxMulXpMinusOneInplace<B: Backend> {
    fn vec_znx_mul_xp_minus_one_inplace<'s, R>(&self, p: i64, res: &mut R, res_col: usize, scratch: &mut ScratchArena<'s, B>)
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
    fn vec_znx_split_ring<'s, R, A>(&self, res: &mut [R], res_col: usize, a: &A, a_col: usize, scratch: &mut ScratchArena<'s, B>)
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
    fn vec_znx_merge_rings<'s, R, A>(
        &self,
        res: &mut R,
        res_col: usize,
        a: &[A],
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

/// Switches ring degree between `a` and `res` by truncation or zero-padding.
pub trait VecZnxSwitchRingBackend<B: Backend> {
    fn vec_znx_switch_ring_backend(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
    );
}

pub trait VecZnxCopyBackend<B: Backend> {
    fn vec_znx_copy_backend(&self, res: &mut VecZnxBackendMut<'_, B>, res_col: usize, a: &VecZnxBackendRef<'_, B>, a_col: usize);
}

pub trait VecZnxFillUniform {
    /// Fills the first `size` size with uniform values in \[-2^{base2k-1}, 2^{base2k-1}\]
    fn vec_znx_fill_uniform<R>(&self, base2k: usize, res: &mut R, res_col: usize, source: &mut Source)
    where
        R: VecZnxToMut;
}

pub trait VecZnxFillUniformBackend<B: Backend> {
    /// Fills the selected backend-native column from a backend-defined uniform sampler seeded by `seed`.
    fn vec_znx_fill_uniform_backend(&self, base2k: usize, res: &mut VecZnxBackendMut<'_, B>, res_col: usize, seed: [u8; 32]);
}

#[allow(clippy::too_many_arguments)]
/// Fills the selected column with a discrete Gaussian noise vector
/// scaled by `2^{-k}` with standard deviation `sigma`, bounded to `[-bound, bound]`.
pub trait VecZnxFillNormal {
    fn vec_znx_fill_normal<R>(&self, base2k: usize, res: &mut R, res_col: usize, noise_infos: NoiseInfos, source_xe: &mut Source)
    where
        R: VecZnxToMut;
}

#[allow(clippy::too_many_arguments)]
pub trait VecZnxFillNormalBackend<B: Backend> {
    /// Fills the selected backend-native column from a backend-defined normal sampler seeded by `seed`.
    fn vec_znx_fill_normal_backend(
        &self,
        base2k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        noise_infos: NoiseInfos,
        seed: [u8; 32],
    );
}

#[allow(clippy::too_many_arguments)]
pub trait VecZnxAddNormal {
    /// Adds a discrete normal vector scaled by 2^{-k} with the provided standard deviation and bounded to \[-bound, bound\].
    fn vec_znx_add_normal<R>(&self, base2k: usize, res: &mut R, res_col: usize, noise_infos: NoiseInfos, source_xe: &mut Source)
    where
        R: VecZnxToMut;
}

#[allow(clippy::too_many_arguments)]
pub trait VecZnxAddNormalBackend<B: Backend> {
    /// Adds backend-defined normal noise to the selected backend-native column using `seed`.
    fn vec_znx_add_normal_backend(
        &self,
        base2k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        noise_infos: NoiseInfos,
        seed: [u8; 32],
    );
}
