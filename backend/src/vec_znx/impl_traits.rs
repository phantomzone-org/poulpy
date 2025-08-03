use rand_distr::Distribution;
use rug::Float;
use sampling::source::Source;

use crate::{
    Backend, Module, ScalarZnxToRef, Scratch,
    vec_znx::layout::{VecZnxOwned, VecZnxToMut, VecZnxToRef},
};

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See \[TODO\] for reference code.
/// * See [crate::vec_znx::traits::VecZnxAlloc] for behavioral contract.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxAllocImpl<B: Backend> {
    fn vec_znx_alloc_impl(n: usize, cols: usize, size: usize) -> VecZnxOwned;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See \[TODO\] for reference code.
/// * See [crate::vec_znx::traits::VecZnxFromBytes] for behavioral contract.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxFromBytesImpl<B: Backend> {
    fn vec_znx_from_bytes_impl(n: usize, cols: usize, size: usize, bytes: Vec<u8>) -> VecZnxOwned;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See \[TODO\] for reference code.
/// * See [crate::vec_znx::traits::VecZnxAllocBytes] for behavioral contract.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxAllocBytesImpl<B: Backend> {
    fn vec_znx_alloc_bytes_impl(n: usize, cols: usize, size: usize) -> usize;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See \[TODO\] for reference code.
/// * See [crate::vec_znx::traits::VecZnxNormalizeTmpBytes] for behavioral contract.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxNormalizeTmpBytesImpl<B: Backend> {
    fn vec_znx_normalize_tmp_bytes_impl(module: &Module<B>) -> usize;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See \[TODO\] for reference code.
/// * See [crate::vec_znx::traits::VecZnxNormalize] for behavioral contract.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxNormalizeImpl<B: Backend> {
    fn vec_znx_normalize_impl<R, A>(
        module: &Module<B>,
        basek: usize,
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
/// * See \[TODO\] for reference code.
/// * See [crate::vec_znx::traits::VecZnxNormalizeInplace] for behavioral contract.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxNormalizeInplaceImpl<B: Backend> {
    fn vec_znx_normalize_inplace_impl<A>(module: &Module<B>, basek: usize, a: &mut A, a_col: usize, scratch: &mut Scratch<B>)
    where
        A: VecZnxToMut;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See \[TODO\] for reference code.
/// * See [crate::vec_znx::traits::VecZnxAdd] for behavioral contract.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxAddImpl<B: Backend> {
    fn vec_znx_add_impl<R, A, C>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
        C: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See \[TODO\] for reference code.
/// * See [crate::vec_znx::traits::VecZnxAddInplace] for behavioral contract.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxAddInplaceImpl<B: Backend> {
    fn vec_znx_add_inplace_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See \[TODO\] for reference code.
/// * See [crate::vec_znx::traits::VecZnxAddScalarInplace] for behavioral contract.
/// * See [crate::doc::backend_safety] for safety contract.
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
/// * See \[TODO\] for reference code.
/// * See [crate::vec_znx::traits::VecZnxSub] for behavioral contract.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxSubImpl<B: Backend> {
    fn vec_znx_sub_impl<R, A, C>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
        C: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See \[TODO\] for reference code.
/// * See [crate::vec_znx::traits::VecZnxSubABInplace] for behavioral contract.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxSubABInplaceImpl<B: Backend> {
    fn vec_znx_sub_ab_inplace_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See \[TODO\] for reference code.
/// * See [crate::vec_znx::traits::VecZnxSubBAInplace] for behavioral contract.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxSubBAInplaceImpl<B: Backend> {
    fn vec_znx_sub_ba_inplace_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See \[TODO\] for reference code.
/// * See [crate::vec_znx::traits::VecZnxSubScalarInplace] for behavioral contract.
/// * See [crate::doc::backend_safety] for safety contract.
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
/// * See \[TODO\] for reference code.
/// * See [crate::vec_znx::traits::VecZnxNegate] for behavioral contract.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxNegateImpl<B: Backend> {
    fn vec_znx_negate_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See \[TODO\] for reference code.
/// * See [crate::vec_znx::traits::VecZnxNegateInplace] for behavioral contract.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxNegateInplaceImpl<B: Backend> {
    fn vec_znx_negate_inplace_impl<A>(module: &Module<B>, a: &mut A, a_col: usize)
    where
        A: VecZnxToMut;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See \[TODO\] for reference code.
/// * See [crate::vec_znx::traits::VecZnxRshInplace] for behavioral contract.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxRshInplaceImpl<B: Backend> {
    fn vec_znx_rsh_inplace_impl<A>(module: &Module<B>, basek: usize, k: usize, a: &mut A)
    where
        A: VecZnxToMut;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See \[TODO\] for reference code.
/// * See [crate::vec_znx::traits::VecZnxLshInplace] for behavioral contract.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxLshInplaceImpl<B: Backend> {
    fn vec_znx_lsh_inplace_impl<A>(module: &Module<B>, basek: usize, k: usize, a: &mut A)
    where
        A: VecZnxToMut;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See \[TODO\] for reference code.
/// * See [crate::vec_znx::traits::VecZnxRotate] for behavioral contract.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxRotateImpl<B: Backend> {
    fn vec_znx_rotate_impl<R, A>(module: &Module<B>, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See \[TODO\] for reference code.
/// * See [crate::vec_znx::traits::VecZnxRotateInplace] for behavioral contract.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxRotateInplaceImpl<B: Backend> {
    fn vec_znx_rotate_inplace_impl<A>(module: &Module<B>, k: i64, a: &mut A, a_col: usize)
    where
        A: VecZnxToMut;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See \[TODO\] for reference code.
/// * See [crate::vec_znx::traits::VecZnxAutomorphism] for behavioral contract.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxAutomorphismImpl<B: Backend> {
    fn vec_znx_automorphism_impl<R, A>(module: &Module<B>, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See \[TODO\] for reference code.
/// * See [crate::vec_znx::traits::VecZnxAutomorphismInplace] for behavioral contract.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxAutomorphismInplaceImpl<B: Backend> {
    fn vec_znx_automorphism_inplace_impl<A>(module: &Module<B>, k: i64, a: &mut A, a_col: usize)
    where
        A: VecZnxToMut;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See \[TODO\] for reference code.
/// * See [crate::vec_znx::traits::VecZnxSplit] for behavioral contract.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxSplitImpl<B: Backend> {
    fn vec_znx_split_impl<R, A>(
        module: &Module<B>,
        res: &mut Vec<R>,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See \[TODO\] for reference code.
/// * See [crate::vec_znx::traits::VecZnxMerge] for behavioral contract.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxMergeImpl<B: Backend> {
    fn vec_znx_merge_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: Vec<A>, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See \[TODO\] for reference code.
/// * See [crate::vec_znx::traits::VecZnxSwithcDegree] for behavioral contract.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxSwithcDegreeImpl<B: Backend> {
    fn vec_znx_switch_degree_impl<R: VecZnxToMut, A: VecZnxToRef>(
        module: &Module<B>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
    );
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See \[TODO\] for reference code.
/// * See [crate::vec_znx::traits::VecZnxCopy] for behavioral contract.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxCopyImpl<B: Backend> {
    fn vec_znx_copy_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See \[TODO\] for reference code.
/// * See [crate::vec_znx::traits::VecZnxStd] for behavioral contract.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxStdImpl<B: Backend> {
    fn vec_znx_std_impl<A>(module: &Module<B>, basek: usize, a: &A, a_col: usize) -> f64
    where
        A: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See \[TODO\] for reference code.
/// * See [crate::vec_znx::traits::VecZnxFillUniform] for behavioral contract.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxFillUniformImpl<B: Backend> {
    fn vec_znx_fill_uniform_impl<R>(module: &Module<B>, basek: usize, res: &mut R, res_col: usize, k: usize, source: &mut Source)
    where
        R: VecZnxToMut;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See \[TODO\] for reference code.
/// * See [crate::vec_znx::traits::VecZnxFillDistF64] for behavioral contract.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxFillDistF64Impl<B: Backend> {
    fn vec_znx_fill_dist_f64_impl<R, D: Distribution<f64>>(
        module: &Module<B>,
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

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See \[TODO\] for reference code.
/// * See [crate::vec_znx::traits::VecZnxAddDistF64] for behavioral contract.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxAddDistF64Impl<B: Backend> {
    fn vec_znx_add_dist_f64_impl<R, D: Distribution<f64>>(
        module: &Module<B>,
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

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See \[TODO\] for reference code.
/// * See [crate::vec_znx::traits::VecZnxFillNormal] for behavioral contract.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxFillNormalImpl<B: Backend> {
    fn vec_znx_fill_normal_impl<R>(
        module: &Module<B>,
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

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See \[TODO\] for reference code.
/// * See [crate::vec_znx::traits::VecZnxAddNormal] for behavioral contract.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxAddNormalImpl<B: Backend> {
    fn vec_znx_add_normal_impl<R>(
        module: &Module<B>,
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

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See \[TODO\] for reference code.
/// * See [crate::vec_znx::traits::VecZnxEncodeVeci64] for behavioral contract.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxEncodeVeci64Impl<B: Backend> {
    fn encode_vec_i64_impl<R>(
        module: &Module<B>,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        data: &[i64],
        log_max: usize,
    ) where
        R: VecZnxToMut;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See \[TODO\] for reference code.
/// * See [crate::vec_znx::traits::VecZnxEncodeCoeffsi64] for behavioral contract.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxEncodeCoeffsi64Impl<B: Backend> {
    fn encode_coeff_i64_impl<R>(
        module: &Module<B>,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        i: usize,
        data: i64,
        log_max: usize,
    ) where
        R: VecZnxToMut;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See \[TODO\] for reference code.
/// * See [crate::vec_znx::traits::VecZnxDecodeVeci64] for behavioral contract.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxDecodeVeci64Impl<B: Backend> {
    fn decode_vec_i64_impl<R>(module: &Module<B>, basek: usize, res: &R, res_col: usize, k: usize, data: &mut [i64])
    where
        R: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See \[TODO\] for reference code.
/// * See [crate::vec_znx::traits::VecZnxDecodeCoeffsi64] for behavioral contract.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxDecodeCoeffsi64Impl<B: Backend> {
    fn decode_coeff_i64_impl<R>(module: &Module<B>, basek: usize, res: &R, res_col: usize, k: usize, i: usize) -> i64
    where
        R: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See \[TODO\] for reference code.
/// * See [crate::vec_znx::traits::VecZnxDecodeVecFloat] for behavioral contract.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxDecodeVecFloatImpl<B: Backend> {
    fn decode_vec_float_impl<R>(module: &Module<B>, basek: usize, res: &R, res_col: usize, data: &mut [Float])
    where
        R: VecZnxToRef;
}
