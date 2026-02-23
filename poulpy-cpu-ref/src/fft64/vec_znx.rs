//! Vector-of-ring-elements operations for [`FFT64Ref`](crate::FFT64Ref).
//!
//! Implements the `VecZnx*` OEP traits from `poulpy_hal::oep`, covering the full
//! surface of operations on `VecZnx` (vectors of polynomials in `Z[X]/(X^n+1)`,
//! stored in a base-2^k limb decomposition):
//!
//! - **Arithmetic**: add, sub, negate (with out-of-place, inplace, and scalar variants).
//! - **Bit shifts**: left-shift (LSH) and right-shift (RSH) across limbs.
//! - **Structural transforms**: rotation (`X^p * a`), automorphism (`X -> X^p`),
//!   multiplication by `X^p - 1`.
//! - **Ring operations**: split-ring, merge-rings, switch-ring, copy, zero.
//! - **Normalization**: carry propagation across the base-2^k decomposition.
//! - **Sampling**: uniform and normal (Gaussian) random fill.
//!
//! Operations that require temporary storage take a `&mut Scratch<Self>` parameter
//! and call `scratch.take_slice(...)` to carve out aligned workspace. The required
//! scratch size can be queried via the corresponding `*TmpBytes` trait methods.

use poulpy_hal::{
    api::{
        TakeSlice, VecZnxAutomorphismInplaceTmpBytes, VecZnxLshTmpBytes, VecZnxMergeRingsTmpBytes,
        VecZnxMulXpMinusOneInplaceTmpBytes, VecZnxNormalizeTmpBytes, VecZnxRotateInplaceTmpBytes, VecZnxRshTmpBytes,
        VecZnxSplitRingTmpBytes,
    },
    layouts::{Module, ScalarZnxToRef, Scratch, VecZnxToMut, VecZnxToRef},
    oep::{
        TakeSliceImpl, VecZnxAddImpl, VecZnxAddInplaceImpl, VecZnxAddNormalImpl, VecZnxAddScalarImpl, VecZnxAddScalarInplaceImpl,
        VecZnxAutomorphismImpl, VecZnxAutomorphismInplaceImpl, VecZnxAutomorphismInplaceTmpBytesImpl, VecZnxCopyImpl,
        VecZnxFillNormalImpl, VecZnxFillUniformImpl, VecZnxLshImpl, VecZnxLshInplaceImpl, VecZnxLshTmpBytesImpl,
        VecZnxMergeRingsImpl, VecZnxMergeRingsTmpBytesImpl, VecZnxMulXpMinusOneImpl, VecZnxMulXpMinusOneInplaceImpl,
        VecZnxMulXpMinusOneInplaceTmpBytesImpl, VecZnxNegateImpl, VecZnxNegateInplaceImpl, VecZnxNormalizeImpl,
        VecZnxNormalizeInplaceImpl, VecZnxNormalizeTmpBytesImpl, VecZnxRotateImpl, VecZnxRotateInplaceImpl,
        VecZnxRotateInplaceTmpBytesImpl, VecZnxRshImpl, VecZnxRshInplaceImpl, VecZnxRshTmpBytesImpl, VecZnxSplitRingImpl,
        VecZnxSplitRingTmpBytesImpl, VecZnxSubImpl, VecZnxSubInplaceImpl, VecZnxSubNegateInplaceImpl, VecZnxSubScalarImpl,
        VecZnxSubScalarInplaceImpl, VecZnxSwitchRingImpl, VecZnxZeroImpl,
    },
    reference::vec_znx::{
        vec_znx_add, vec_znx_add_inplace, vec_znx_add_normal_ref, vec_znx_add_scalar, vec_znx_add_scalar_inplace,
        vec_znx_automorphism, vec_znx_automorphism_inplace, vec_znx_automorphism_inplace_tmp_bytes, vec_znx_copy,
        vec_znx_fill_normal_ref, vec_znx_fill_uniform_ref, vec_znx_lsh, vec_znx_lsh_inplace, vec_znx_lsh_tmp_bytes,
        vec_znx_merge_rings, vec_znx_merge_rings_tmp_bytes, vec_znx_mul_xp_minus_one, vec_znx_mul_xp_minus_one_inplace,
        vec_znx_mul_xp_minus_one_inplace_tmp_bytes, vec_znx_negate, vec_znx_negate_inplace, vec_znx_normalize,
        vec_znx_normalize_inplace, vec_znx_normalize_tmp_bytes, vec_znx_rotate, vec_znx_rotate_inplace,
        vec_znx_rotate_inplace_tmp_bytes, vec_znx_rsh, vec_znx_rsh_inplace, vec_znx_rsh_tmp_bytes, vec_znx_split_ring,
        vec_znx_split_ring_tmp_bytes, vec_znx_sub, vec_znx_sub_inplace, vec_znx_sub_negate_inplace, vec_znx_sub_scalar,
        vec_znx_sub_scalar_inplace, vec_znx_switch_ring, vec_znx_zero,
    },
    source::Source,
};

use super::FFT64Ref;

unsafe impl VecZnxZeroImpl<Self> for FFT64Ref {
    fn vec_znx_zero_impl<R>(_module: &Module<Self>, res: &mut R, res_col: usize)
    where
        R: VecZnxToMut,
    {
        vec_znx_zero::<_, FFT64Ref>(res, res_col);
    }
}

unsafe impl VecZnxNormalizeTmpBytesImpl<Self> for FFT64Ref {
    fn vec_znx_normalize_tmp_bytes_impl(module: &Module<Self>) -> usize {
        vec_znx_normalize_tmp_bytes(module.n())
    }
}

unsafe impl VecZnxNormalizeImpl<Self> for FFT64Ref
where
    Self: TakeSliceImpl<Self> + VecZnxNormalizeTmpBytesImpl<Self>,
{
    fn vec_znx_normalize_impl<R, A>(
        module: &Module<Self>,
        res: &mut R,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &A,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut Scratch<Self>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let (carry, _) = scratch.take_slice(module.vec_znx_normalize_tmp_bytes() / size_of::<i64>());
        vec_znx_normalize::<R, A, Self>(res, res_base2k, res_offset, res_col, a, a_base2k, a_col, carry);
    }
}

unsafe impl VecZnxNormalizeInplaceImpl<Self> for FFT64Ref
where
    Self: TakeSliceImpl<Self> + VecZnxNormalizeTmpBytesImpl<Self>,
{
    fn vec_znx_normalize_inplace_impl<R>(
        module: &Module<Self>,
        base2k: usize,
        res: &mut R,
        res_col: usize,
        scratch: &mut Scratch<Self>,
    ) where
        R: VecZnxToMut,
    {
        let (carry, _) = scratch.take_slice(module.vec_znx_normalize_tmp_bytes() / size_of::<i64>());
        vec_znx_normalize_inplace::<R, Self>(base2k, res, res_col, carry);
    }
}

unsafe impl VecZnxAddImpl<Self> for FFT64Ref {
    fn vec_znx_add_impl<R, A, B>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
        B: VecZnxToRef,
    {
        vec_znx_add::<R, A, B, Self>(res, res_col, a, a_col, b, b_col);
    }
}

unsafe impl VecZnxAddInplaceImpl<Self> for FFT64Ref {
    fn vec_znx_add_inplace_impl<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        vec_znx_add_inplace::<R, A, Self>(res, res_col, a, a_col);
    }
}

unsafe impl VecZnxAddScalarInplaceImpl<Self> for FFT64Ref {
    fn vec_znx_add_scalar_inplace_impl<R, A>(
        _module: &Module<Self>,
        res: &mut R,
        res_col: usize,
        res_limb: usize,
        a: &A,
        a_col: usize,
    ) where
        R: VecZnxToMut,
        A: ScalarZnxToRef,
    {
        vec_znx_add_scalar_inplace::<R, A, Self>(res, res_col, res_limb, a, a_col);
    }
}

unsafe impl VecZnxAddScalarImpl<Self> for FFT64Ref {
    fn vec_znx_add_scalar_impl<R, A, B>(
        _module: &Module<Self>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &B,
        b_col: usize,
        b_limb: usize,
    ) where
        R: VecZnxToMut,
        A: ScalarZnxToRef,
        B: VecZnxToRef,
    {
        vec_znx_add_scalar::<R, A, B, Self>(res, res_col, a, a_col, b, b_col, b_limb);
    }
}

unsafe impl VecZnxSubImpl<Self> for FFT64Ref {
    fn vec_znx_sub_impl<R, A, B>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
        B: VecZnxToRef,
    {
        vec_znx_sub::<R, A, B, Self>(res, res_col, a, a_col, b, b_col);
    }
}

unsafe impl VecZnxSubInplaceImpl<Self> for FFT64Ref {
    fn vec_znx_sub_inplace_impl<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        vec_znx_sub_inplace::<R, A, Self>(res, res_col, a, a_col);
    }
}

unsafe impl VecZnxSubNegateInplaceImpl<Self> for FFT64Ref {
    fn vec_znx_sub_negate_inplace_impl<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        vec_znx_sub_negate_inplace::<R, A, Self>(res, res_col, a, a_col);
    }
}

unsafe impl VecZnxSubScalarImpl<Self> for FFT64Ref {
    fn vec_znx_sub_scalar_impl<R, A, B>(
        _module: &Module<Self>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &B,
        b_col: usize,
        b_limb: usize,
    ) where
        R: VecZnxToMut,
        A: ScalarZnxToRef,
        B: VecZnxToRef,
    {
        vec_znx_sub_scalar::<R, A, B, Self>(res, res_col, a, a_col, b, b_col, b_limb);
    }
}

unsafe impl VecZnxSubScalarInplaceImpl<Self> for FFT64Ref {
    fn vec_znx_sub_scalar_inplace_impl<R, A>(
        _module: &Module<Self>,
        res: &mut R,
        res_col: usize,
        res_limb: usize,
        a: &A,
        a_col: usize,
    ) where
        R: VecZnxToMut,
        A: ScalarZnxToRef,
    {
        vec_znx_sub_scalar_inplace::<R, A, Self>(res, res_col, res_limb, a, a_col);
    }
}

unsafe impl VecZnxNegateImpl<Self> for FFT64Ref {
    fn vec_znx_negate_impl<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        vec_znx_negate::<R, A, Self>(res, res_col, a, a_col);
    }
}

unsafe impl VecZnxNegateInplaceImpl<Self> for FFT64Ref {
    fn vec_znx_negate_inplace_impl<R>(_module: &Module<Self>, res: &mut R, res_col: usize)
    where
        R: VecZnxToMut,
    {
        vec_znx_negate_inplace::<R, Self>(res, res_col);
    }
}

unsafe impl VecZnxLshTmpBytesImpl<Self> for FFT64Ref {
    fn vec_znx_lsh_tmp_bytes_impl(module: &Module<Self>) -> usize {
        vec_znx_lsh_tmp_bytes(module.n())
    }
}

unsafe impl VecZnxRshTmpBytesImpl<Self> for FFT64Ref {
    fn vec_znx_rsh_tmp_bytes_impl(module: &Module<Self>) -> usize {
        vec_znx_rsh_tmp_bytes(module.n())
    }
}

unsafe impl VecZnxLshImpl<Self> for FFT64Ref
where
    Module<Self>: VecZnxNormalizeTmpBytes,
    Scratch<Self>: TakeSlice,
{
    fn vec_znx_lsh_impl<R, A>(
        module: &Module<Self>,
        base2k: usize,
        k: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<Self>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let (carry, _) = scratch.take_slice(module.vec_znx_lsh_tmp_bytes() / size_of::<i64>());
        vec_znx_lsh::<_, _, Self>(base2k, k, res, res_col, a, a_col, carry);
    }
}

unsafe impl VecZnxLshInplaceImpl<Self> for FFT64Ref
where
    Module<Self>: VecZnxNormalizeTmpBytes,
    Scratch<Self>: TakeSlice,
{
    fn vec_znx_lsh_inplace_impl<A>(
        module: &Module<Self>,
        base2k: usize,
        k: usize,
        a: &mut A,
        a_col: usize,
        scratch: &mut Scratch<Self>,
    ) where
        A: VecZnxToMut,
    {
        let (carry, _) = scratch.take_slice(module.vec_znx_lsh_tmp_bytes() / size_of::<i64>());
        vec_znx_lsh_inplace::<_, Self>(base2k, k, a, a_col, carry);
    }
}

unsafe impl VecZnxRshImpl<Self> for FFT64Ref
where
    Module<Self>: VecZnxNormalizeTmpBytes,
    Scratch<Self>: TakeSlice,
{
    fn vec_znx_rsh_impl<R, A>(
        module: &Module<Self>,
        base2k: usize,
        k: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<Self>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let (carry, _) = scratch.take_slice(module.vec_znx_rsh_tmp_bytes() / size_of::<i64>());
        vec_znx_rsh::<_, _, Self>(base2k, k, res, res_col, a, a_col, carry);
    }
}

unsafe impl VecZnxRshInplaceImpl<Self> for FFT64Ref
where
    Module<Self>: VecZnxNormalizeTmpBytes,
    Scratch<Self>: TakeSlice,
{
    fn vec_znx_rsh_inplace_impl<A>(
        module: &Module<Self>,
        base2k: usize,
        k: usize,
        a: &mut A,
        a_col: usize,
        scratch: &mut Scratch<Self>,
    ) where
        A: VecZnxToMut,
    {
        let (carry, _) = scratch.take_slice(module.vec_znx_rsh_tmp_bytes() / size_of::<i64>());
        vec_znx_rsh_inplace::<_, Self>(base2k, k, a, a_col, carry);
    }
}

unsafe impl VecZnxRotateImpl<Self> for FFT64Ref {
    fn vec_znx_rotate_impl<R, A>(_module: &Module<Self>, p: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        vec_znx_rotate::<R, A, Self>(p, res, res_col, a, a_col);
    }
}

unsafe impl VecZnxRotateInplaceTmpBytesImpl<Self> for FFT64Ref
where
    Scratch<Self>: TakeSlice,
{
    fn vec_znx_rotate_inplace_tmp_bytes_impl(module: &Module<Self>) -> usize {
        vec_znx_rotate_inplace_tmp_bytes(module.n())
    }
}

unsafe impl VecZnxRotateInplaceImpl<Self> for FFT64Ref
where
    Scratch<Self>: TakeSlice,
    Self: VecZnxRotateInplaceTmpBytesImpl<Self>,
{
    fn vec_znx_rotate_inplace_impl<R>(module: &Module<Self>, p: i64, res: &mut R, res_col: usize, scratch: &mut Scratch<Self>)
    where
        R: VecZnxToMut,
    {
        let (tmp, _) = scratch.take_slice(module.vec_znx_rotate_inplace_tmp_bytes() / size_of::<i64>());
        vec_znx_rotate_inplace::<R, Self>(p, res, res_col, tmp);
    }
}

unsafe impl VecZnxAutomorphismImpl<Self> for FFT64Ref {
    fn vec_znx_automorphism_impl<R, A>(_module: &Module<Self>, p: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        vec_znx_automorphism::<R, A, Self>(p, res, res_col, a, a_col);
    }
}

unsafe impl VecZnxAutomorphismInplaceTmpBytesImpl<Self> for FFT64Ref {
    fn vec_znx_automorphism_inplace_tmp_bytes_impl(module: &Module<Self>) -> usize {
        vec_znx_automorphism_inplace_tmp_bytes(module.n())
    }
}

unsafe impl VecZnxAutomorphismInplaceImpl<Self> for FFT64Ref
where
    Scratch<Self>: TakeSlice,
    Self: VecZnxAutomorphismInplaceTmpBytesImpl<Self>,
{
    fn vec_znx_automorphism_inplace_impl<R>(
        module: &Module<Self>,
        p: i64,
        res: &mut R,
        res_col: usize,
        scratch: &mut Scratch<Self>,
    ) where
        R: VecZnxToMut,
    {
        let (tmp, _) = scratch.take_slice(module.vec_znx_automorphism_inplace_tmp_bytes() / size_of::<i64>());
        vec_znx_automorphism_inplace::<R, Self>(p, res, res_col, tmp);
    }
}

unsafe impl VecZnxMulXpMinusOneImpl<Self> for FFT64Ref {
    fn vec_znx_mul_xp_minus_one_impl<R, A>(_module: &Module<Self>, p: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        vec_znx_mul_xp_minus_one::<R, A, Self>(p, res, res_col, a, a_col);
    }
}

unsafe impl VecZnxMulXpMinusOneInplaceTmpBytesImpl<Self> for FFT64Ref
where
    Scratch<Self>: TakeSlice,
    Self: VecZnxMulXpMinusOneImpl<Self>,
{
    fn vec_znx_mul_xp_minus_one_inplace_tmp_bytes_impl(module: &Module<Self>) -> usize {
        vec_znx_mul_xp_minus_one_inplace_tmp_bytes(module.n())
    }
}

unsafe impl VecZnxMulXpMinusOneInplaceImpl<Self> for FFT64Ref {
    fn vec_znx_mul_xp_minus_one_inplace_impl<R>(
        module: &Module<Self>,
        p: i64,
        res: &mut R,
        res_col: usize,
        scratch: &mut Scratch<Self>,
    ) where
        R: VecZnxToMut,
    {
        let (tmp, _) = scratch.take_slice(module.vec_znx_mul_xp_minus_one_inplace_tmp_bytes() / size_of::<i64>());
        vec_znx_mul_xp_minus_one_inplace::<R, Self>(p, res, res_col, tmp);
    }
}

unsafe impl VecZnxSplitRingTmpBytesImpl<Self> for FFT64Ref {
    fn vec_znx_split_ring_tmp_bytes_impl(module: &Module<Self>) -> usize {
        vec_znx_split_ring_tmp_bytes(module.n())
    }
}

unsafe impl VecZnxSplitRingImpl<Self> for FFT64Ref
where
    Module<Self>: VecZnxSplitRingTmpBytes,
    Scratch<Self>: TakeSlice,
{
    fn vec_znx_split_ring_impl<R, A>(
        module: &Module<Self>,
        res: &mut [R],
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<Self>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let (tmp, _) = scratch.take_slice(module.vec_znx_split_ring_tmp_bytes() / size_of::<i64>());
        vec_znx_split_ring::<R, A, Self>(res, res_col, a, a_col, tmp);
    }
}

unsafe impl VecZnxMergeRingsTmpBytesImpl<Self> for FFT64Ref {
    fn vec_znx_merge_rings_tmp_bytes_impl(module: &Module<Self>) -> usize {
        vec_znx_merge_rings_tmp_bytes(module.n())
    }
}

unsafe impl VecZnxMergeRingsImpl<Self> for FFT64Ref
where
    Module<Self>: VecZnxMergeRingsTmpBytes,
{
    fn vec_znx_merge_rings_impl<R, A>(
        module: &Module<Self>,
        res: &mut R,
        res_col: usize,
        a: &[A],
        a_col: usize,
        scratch: &mut Scratch<Self>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let (tmp, _) = scratch.take_slice(module.vec_znx_merge_rings_tmp_bytes() / size_of::<i64>());
        vec_znx_merge_rings::<R, A, Self>(res, res_col, a, a_col, tmp);
    }
}

unsafe impl VecZnxSwitchRingImpl<Self> for FFT64Ref
where
    Self: VecZnxCopyImpl<Self>,
{
    fn vec_znx_switch_ring_impl<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        vec_znx_switch_ring::<R, A, Self>(res, res_col, a, a_col);
    }
}

unsafe impl VecZnxCopyImpl<Self> for FFT64Ref {
    fn vec_znx_copy_impl<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        vec_znx_copy::<R, A, Self>(res, res_col, a, a_col)
    }
}

unsafe impl VecZnxFillUniformImpl<Self> for FFT64Ref {
    fn vec_znx_fill_uniform_impl<R>(_module: &Module<Self>, base2k: usize, res: &mut R, res_col: usize, source: &mut Source)
    where
        R: VecZnxToMut,
    {
        vec_znx_fill_uniform_ref(base2k, res, res_col, source)
    }
}

unsafe impl VecZnxFillNormalImpl<Self> for FFT64Ref {
    fn vec_znx_fill_normal_impl<R>(
        _module: &Module<Self>,
        base2k: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        sigma: f64,
        bound: f64,
    ) where
        R: VecZnxToMut,
    {
        vec_znx_fill_normal_ref(base2k, res, res_col, k, sigma, bound, source);
    }
}

unsafe impl VecZnxAddNormalImpl<Self> for FFT64Ref {
    fn vec_znx_add_normal_impl<R>(
        _module: &Module<Self>,
        base2k: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        sigma: f64,
        bound: f64,
    ) where
        R: VecZnxToMut,
    {
        vec_znx_add_normal_ref(base2k, res, res_col, k, sigma, bound, source);
    }
}
