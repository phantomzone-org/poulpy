//! Backend extension points for coefficient-domain [`VecZnx`](crate::layouts::VecZnx) operations.

use crate::{
    layouts::{Backend, Module, ScalarZnxToRef, Scratch, VecZnxToMut, VecZnxToRef},
    source::Source,
};

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See TODO for reference implementation.
/// * See [crate::api::VecZnxZero] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxZeroImpl<B: Backend> {
    fn vec_znx_zero_impl<R>(module: &Module<B>, res: &mut R, res_col: usize)
    where
        R: VecZnxToMut;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/vec_znx.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/vec_znx.rs) for reference implementation.
/// * See [crate::api::VecZnxNormalizeTmpBytes] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxNormalizeTmpBytesImpl<B: Backend> {
    fn vec_znx_normalize_tmp_bytes_impl(module: &Module<B>) -> usize;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/vec_znx.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/vec_znx.rs) for reference implementation.
/// * See [crate::api::VecZnxNormalize] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxNormalizeImpl<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_normalize_impl<R, A>(
        module: &Module<B>,
        res: &mut R,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &A,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/vec_znx.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/vec_znx.rs) for reference implementation.
/// * See [crate::api::VecZnxNormalizeInplace] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxNormalizeInplaceImpl<B: Backend> {
    fn vec_znx_normalize_inplace_impl<A>(module: &Module<B>, base2k: usize, a: &mut A, a_col: usize, scratch: &mut Scratch<B>)
    where
        A: VecZnxToMut;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/vec_znx.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/vec_znx.rs) for reference implementation.
/// * See [crate::api::VecZnxAdd] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxAddImpl<B: Backend> {
    fn vec_znx_add_impl<R, A, C>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
        C: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/vec_znx.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/vec_znx.rs) for reference implementation.
/// * See [crate::api::VecZnxAddInplace] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxAddInplaceImpl<B: Backend> {
    fn vec_znx_add_inplace_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/vec_znx.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/vec_znx.rs) for reference implementation.
/// * See [crate::api::VecZnxAddScalar] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxAddScalarImpl<D: Backend> {
    /// Adds the selected column of `a` on the selected column and limb of `b` and writes the result on the selected column of `res`.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_add_scalar_impl<R, A, B>(
        module: &Module<D>,
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
        B: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/vec_znx.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/vec_znx.rs) for reference implementation.
/// * See [crate::api::VecZnxAddScalarInplace] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxAddScalarInplaceImpl<B: Backend> {
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

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/vec_znx.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/vec_znx.rs) for reference implementation.
/// * See [crate::api::VecZnxSub] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxSubImpl<B: Backend> {
    fn vec_znx_sub_impl<R, A, C>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
        C: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/vec_znx.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/vec_znx.rs) for reference implementation.
/// * See [crate::api::VecZnxSubInplace] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxSubInplaceImpl<B: Backend> {
    fn vec_znx_sub_inplace_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/vec_znx.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/vec_znx.rs) for reference implementation.
/// * See [crate::api::VecZnxSubNegateInplace] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxSubNegateInplaceImpl<B: Backend> {
    fn vec_znx_sub_negate_inplace_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/vec_znx.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/vec_znx.rs) for reference implementation.
/// * See [crate::api::VecZnxAddScalar] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxSubScalarImpl<D: Backend> {
    /// Adds the selected column of `a` on the selected column and limb of `b` and writes the result on the selected column of `res`.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_sub_scalar_impl<R, A, B>(
        module: &Module<D>,
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
        B: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/vec_znx.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/vec_znx.rs) for reference implementation.
/// * See [crate::api::VecZnxSubScalarInplace] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxSubScalarInplaceImpl<B: Backend> {
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

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/vec_znx.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/vec_znx.rs) for reference implementation.
/// * See [crate::api::VecZnxNegate] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxNegateImpl<B: Backend> {
    fn vec_znx_negate_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/vec_znx.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/vec_znx.rs) for reference implementation.
/// * See [crate::api::VecZnxNegateInplace] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxNegateInplaceImpl<B: Backend> {
    fn vec_znx_negate_inplace_impl<A>(module: &Module<B>, a: &mut A, a_col: usize)
    where
        A: VecZnxToMut;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/vec_znx.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/vec_znx.rs) for reference implementation.
/// * See [crate::api::VecZnxRshTmpBytes] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxRshTmpBytesImpl<B: Backend> {
    fn vec_znx_rsh_tmp_bytes_impl(module: &Module<B>) -> usize;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/vec_znx.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/vec_znx.rs) for reference implementation.
/// * See [crate::api::VecZnxRsh] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxRshImpl<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_rsh_impl<R, A>(
        module: &Module<B>,
        base2k: usize,
        k: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/vec_znx.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/vec_znx.rs) for reference implementation.
/// * See [crate::api::VecZnxLshTmpBytes] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxLshTmpBytesImpl<B: Backend> {
    fn vec_znx_lsh_tmp_bytes_impl(module: &Module<B>) -> usize;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/vec_znx.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/vec_znx.rs) for reference implementation.
/// * See [crate::api::VecZnxLsh] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxLshImpl<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_lsh_impl<R, A>(
        module: &Module<B>,
        base2k: usize,
        k: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/vec_znx.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/vec_znx.rs) for reference implementation.
/// * See [crate::api::VecZnxRshInplace] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxRshInplaceImpl<B: Backend> {
    fn vec_znx_rsh_inplace_impl<R>(
        module: &Module<B>,
        base2k: usize,
        k: usize,
        res: &mut R,
        res_col: usize,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxToMut;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/vec_znx.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/vec_znx.rs) for reference implementation.
/// * See [crate::api::VecZnxLshInplace] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxLshInplaceImpl<B: Backend> {
    fn vec_znx_lsh_inplace_impl<R>(
        module: &Module<B>,
        base2k: usize,
        k: usize,
        res: &mut R,
        res_col: usize,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxToMut;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/vec_znx.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/vec_znx.rs) for reference implementation.
/// * See [crate::api::VecZnxRotate] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxRotateImpl<B: Backend> {
    fn vec_znx_rotate_impl<R, A>(module: &Module<B>, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/vec_znx.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/vec_znx.rs) for reference implementation.
/// * See [crate::api::VecZnxRotateInplaceTmpBytes] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxRotateInplaceTmpBytesImpl<B: Backend> {
    fn vec_znx_rotate_inplace_tmp_bytes_impl(module: &Module<B>) -> usize;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/vec_znx.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/vec_znx.rs) for reference implementation.
/// * See [crate::api::VecZnxRotateInplace] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxRotateInplaceImpl<B: Backend> {
    fn vec_znx_rotate_inplace_impl<A>(module: &Module<B>, k: i64, a: &mut A, a_col: usize, scratch: &mut Scratch<B>)
    where
        A: VecZnxToMut;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/vec_znx.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/vec_znx.rs) for reference implementation.
/// * See [crate::api::VecZnxAutomorphism] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxAutomorphismImpl<B: Backend> {
    fn vec_znx_automorphism_impl<R, A>(module: &Module<B>, p: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/vec_znx.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/vec_znx.rs) for reference implementation.
/// * See [crate::api::VecZnxAutomorphismInplaceTmpBytes] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxAutomorphismInplaceTmpBytesImpl<B: Backend> {
    fn vec_znx_automorphism_inplace_tmp_bytes_impl(module: &Module<B>) -> usize;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/vec_znx.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/vec_znx.rs) for reference implementation.
/// * See [crate::api::VecZnxAutomorphismInplace] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxAutomorphismInplaceImpl<B: Backend> {
    fn vec_znx_automorphism_inplace_impl<R>(module: &Module<B>, k: i64, res: &mut R, res_col: usize, scratch: &mut Scratch<B>)
    where
        R: VecZnxToMut;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/vec_znx.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/vec_znx.rs) for reference implementation.
/// * See [crate::api::VecZnxMulXpMinusOne] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxMulXpMinusOneImpl<B: Backend> {
    fn vec_znx_mul_xp_minus_one_impl<R, A>(module: &Module<B>, p: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/vec_znx.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/vec_znx.rs) for reference implementation.
/// * See [crate::api::VecZnxMulXpMinusOneInplaceTmpBytes] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxMulXpMinusOneInplaceTmpBytesImpl<B: Backend> {
    fn vec_znx_mul_xp_minus_one_inplace_tmp_bytes_impl(module: &Module<B>) -> usize;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/vec_znx.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/vec_znx.rs) for reference implementation.
/// * See [crate::api::VecZnxMulXpMinusOneInplace] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxMulXpMinusOneInplaceImpl<B: Backend> {
    fn vec_znx_mul_xp_minus_one_inplace_impl<R>(
        module: &Module<B>,
        p: i64,
        res: &mut R,
        res_col: usize,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxToMut;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/vec_znx.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/vec_znx.rs) for reference implementation.
/// * See [crate::api::VecZnxSplitRingTmpBytes] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxSplitRingTmpBytesImpl<B: Backend> {
    fn vec_znx_split_ring_tmp_bytes_impl(module: &Module<B>) -> usize;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See `poulpy-backend` `cpu_spqlios::vec_znx::vec_znx_split_ref` for reference code.
/// * See [crate::api::VecZnxSplitRing] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxSplitRingImpl<B: Backend> {
    fn vec_znx_split_ring_impl<R, A>(
        module: &Module<B>,
        res: &mut [R],
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/vec_znx.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/vec_znx.rs) for reference implementation.
/// * See [crate::api::VecZnxMergeRingsTmpBytes] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxMergeRingsTmpBytesImpl<B: Backend> {
    fn vec_znx_merge_rings_tmp_bytes_impl(module: &Module<B>) -> usize;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See `poulpy-backend` `cpu_spqlios::vec_znx::vec_znx_merge_ref` for reference code.
/// * See [crate::api::VecZnxMergeRings] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxMergeRingsImpl<B: Backend> {
    fn vec_znx_merge_rings_impl<R, A>(
        module: &Module<B>,
        res: &mut R,
        res_col: usize,
        a: &[A],
        a_col: usize,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/vec_znx.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/vec_znx.rs) for reference implementation.
/// * See [crate::api::VecZnxSwitchRing] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxSwitchRingImpl<B: Backend> {
    fn vec_znx_switch_ring_impl<R: VecZnxToMut, A: VecZnxToRef>(
        module: &Module<B>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
    );
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/vec_znx.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/vec_znx.rs) for reference implementation.
/// * See [crate::api::VecZnxCopy] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxCopyImpl<B: Backend> {
    fn vec_znx_copy_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/vec_znx.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/vec_znx.rs) for reference implementation.
/// * See [crate::api::VecZnxFillUniform] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxFillUniformImpl<B: Backend> {
    fn vec_znx_fill_uniform_impl<R>(module: &Module<B>, base2k: usize, res: &mut R, res_col: usize, source: &mut Source)
    where
        R: VecZnxToMut;
}

#[allow(clippy::too_many_arguments)]
/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/vec_znx.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/vec_znx.rs) for reference implementation.
/// * See [crate::api::VecZnxFillNormal] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxFillNormalImpl<B: Backend> {
    fn vec_znx_fill_normal_impl<R>(
        module: &Module<B>,
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
/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/vec_znx.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/vec_znx.rs) for reference implementation.
/// * See [crate::api::VecZnxAddNormal] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxAddNormalImpl<B: Backend> {
    fn vec_znx_add_normal_impl<R>(
        module: &Module<B>,
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
