use rand_distr::Distribution;
use rug::Float;
use sampling::source::Source;

use crate::hal::layouts::{Backend, Module, ScalarZnxToRef, Scratch, VecZnxToMut, VecZnxToRef};

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [vec_znx_normalize_base2k_tmp_bytes_ref](https://github.com/phantomzone-org/spqlios-arithmetic/blob/32a3f5fcce9863b58e949f2dfd5abc1bfbaa09b4/spqlios/arithmetic/vec_znx.c#L245C17-L245C55) for reference code.
/// * See [crate::hal::api::VecZnxNormalizeTmpBytes] for corresponding public API.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxNormalizeTmpBytesImpl<B: Backend> {
    fn vec_znx_normalize_tmp_bytes_impl(module: &Module<B>, n: usize) -> usize;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [vec_znx_normalize_base2k_ref](https://github.com/phantomzone-org/spqlios-arithmetic/blob/32a3f5fcce9863b58e949f2dfd5abc1bfbaa09b4/spqlios/arithmetic/vec_znx.c#L212) for reference code.
/// * See [crate::hal::api::VecZnxNormalize] for corresponding public API.
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
/// * See [vec_znx_normalize_base2k_ref](https://github.com/phantomzone-org/spqlios-arithmetic/blob/32a3f5fcce9863b58e949f2dfd5abc1bfbaa09b4/spqlios/arithmetic/vec_znx.c#L212) for reference code.
/// * See [crate::hal::api::VecZnxNormalizeInplace] for corresponding public API.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxNormalizeInplaceImpl<B: Backend> {
    fn vec_znx_normalize_inplace_impl<A>(module: &Module<B>, basek: usize, a: &mut A, a_col: usize, scratch: &mut Scratch<B>)
    where
        A: VecZnxToMut;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [vec_znx_add_ref](https://github.com/phantomzone-org/spqlios-arithmetic/blob/32a3f5fcce9863b58e949f2dfd5abc1bfbaa09b4/spqlios/arithmetic/vec_znx.c#L86) for reference code.
/// * See [crate::hal::api::VecZnxAdd] for corresponding public API.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxAddImpl<B: Backend> {
    fn vec_znx_add_impl<R, A, C>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
        C: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [vec_znx_add_ref](https://github.com/phantomzone-org/spqlios-arithmetic/blob/32a3f5fcce9863b58e949f2dfd5abc1bfbaa09b4/spqlios/arithmetic/vec_znx.c#L86) for reference code.
/// * See [crate::hal::api::VecZnxAddInplace] for corresponding public API.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxAddInplaceImpl<B: Backend> {
    fn vec_znx_add_inplace_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [vec_znx_add_ref](https://github.com/phantomzone-org/spqlios-arithmetic/blob/32a3f5fcce9863b58e949f2dfd5abc1bfbaa09b4/spqlios/arithmetic/vec_znx.c#L86) for reference code.
/// * See [crate::hal::api::VecZnxAddScalarInplace] for corresponding public API.
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
/// * See [vec_znx_sub_ref](https://github.com/phantomzone-org/spqlios-arithmetic/blob/32a3f5fcce9863b58e949f2dfd5abc1bfbaa09b4/spqlios/arithmetic/vec_znx.c#L125) for reference code.
/// * See [crate::hal::api::VecZnxSub] for corresponding public API.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxSubImpl<B: Backend> {
    fn vec_znx_sub_impl<R, A, C>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
        C: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [vec_znx_sub_ref](https://github.com/phantomzone-org/spqlios-arithmetic/blob/32a3f5fcce9863b58e949f2dfd5abc1bfbaa09b4/spqlios/arithmetic/vec_znx.c#L125) for reference code.
/// * See [crate::hal::api::VecZnxSubABInplace] for corresponding public API.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxSubABInplaceImpl<B: Backend> {
    fn vec_znx_sub_ab_inplace_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [vec_znx_sub_ref](https://github.com/phantomzone-org/spqlios-arithmetic/blob/32a3f5fcce9863b58e949f2dfd5abc1bfbaa09b4/spqlios/arithmetic/vec_znx.c#L125) for reference code.
/// * See [crate::hal::api::VecZnxSubBAInplace] for corresponding public API.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxSubBAInplaceImpl<B: Backend> {
    fn vec_znx_sub_ba_inplace_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [vec_znx_sub_ref](https://github.com/phantomzone-org/spqlios-arithmetic/blob/32a3f5fcce9863b58e949f2dfd5abc1bfbaa09b4/spqlios/arithmetic/vec_znx.c#L125) for reference code.
/// * See [crate::hal::api::VecZnxSubScalarInplace] for corresponding public API.
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
/// * See [vec_znx_negate_ref](https://github.com/phantomzone-org/spqlios-arithmetic/blob/32a3f5fcce9863b58e949f2dfd5abc1bfbaa09b4/spqlios/arithmetic/vec_znx.c#L322C13-L322C31) for reference code.
/// * See [crate::hal::api::VecZnxNegate] for corresponding public API.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxNegateImpl<B: Backend> {
    fn vec_znx_negate_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [vec_znx_negate_ref](https://github.com/phantomzone-org/spqlios-arithmetic/blob/32a3f5fcce9863b58e949f2dfd5abc1bfbaa09b4/spqlios/arithmetic/vec_znx.c#L322C13-L322C31) for reference code.
/// * See [crate::hal::api::VecZnxNegateInplace] for corresponding public API.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxNegateInplaceImpl<B: Backend> {
    fn vec_znx_negate_inplace_impl<A>(module: &Module<B>, a: &mut A, a_col: usize)
    where
        A: VecZnxToMut;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [crate::implementation::cpu_spqlios::vec_znx::vec_znx_rsh_inplace_ref] for reference code.
/// * See [crate::hal::api::VecZnxRshInplace] for corresponding public API.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxRshInplaceImpl<B: Backend> {
    fn vec_znx_rsh_inplace_impl<A>(module: &Module<B>, basek: usize, k: usize, a: &mut A)
    where
        A: VecZnxToMut;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [crate::implementation::cpu_spqlios::vec_znx::vec_znx_lsh_inplace_ref] for reference code.
/// * See [crate::hal::api::VecZnxLshInplace] for corresponding public API.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxLshInplaceImpl<B: Backend> {
    fn vec_znx_lsh_inplace_impl<A>(module: &Module<B>, basek: usize, k: usize, a: &mut A)
    where
        A: VecZnxToMut;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [vec_znx_rotate_ref](https://github.com/phantomzone-org/spqlios-arithmetic/blob/32a3f5fcce9863b58e949f2dfd5abc1bfbaa09b4/spqlios/arithmetic/vec_znx.c#L164) for reference code.
/// * See [crate::hal::api::VecZnxRotate] for corresponding public API.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxRotateImpl<B: Backend> {
    fn vec_znx_rotate_impl<R, A>(module: &Module<B>, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [vec_znx_rotate_ref](https://github.com/phantomzone-org/spqlios-arithmetic/blob/32a3f5fcce9863b58e949f2dfd5abc1bfbaa09b4/spqlios/arithmetic/vec_znx.c#L164) for reference code.
/// * See [crate::hal::api::VecZnxRotateInplace] for corresponding public API.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxRotateInplaceImpl<B: Backend> {
    fn vec_znx_rotate_inplace_impl<A>(module: &Module<B>, k: i64, a: &mut A, a_col: usize)
    where
        A: VecZnxToMut;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [vec_znx_automorphism_ref](https://github.com/phantomzone-org/spqlios-arithmetic/blob/32a3f5fcce9863b58e949f2dfd5abc1bfbaa09b4/spqlios/arithmetic/vec_znx.c#L188) for reference code.
/// * See [crate::hal::api::VecZnxAutomorphism] for corresponding public API.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxAutomorphismImpl<B: Backend> {
    fn vec_znx_automorphism_impl<R, A>(module: &Module<B>, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [vec_znx_automorphism_ref](https://github.com/phantomzone-org/spqlios-arithmetic/blob/32a3f5fcce9863b58e949f2dfd5abc1bfbaa09b4/spqlios/arithmetic/vec_znx.c#L188) for reference code.
/// * See [crate::hal::api::VecZnxAutomorphismInplace] for corresponding public API.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxAutomorphismInplaceImpl<B: Backend> {
    fn vec_znx_automorphism_inplace_impl<A>(module: &Module<B>, k: i64, a: &mut A, a_col: usize)
    where
        A: VecZnxToMut;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [vec_znx_mul_xp_minus_one_ref](https://github.com/phantomzone-org/spqlios-arithmetic/blob/7160f588da49712a042931ea247b4259b95cefcc/spqlios/arithmetic/vec_znx.c#L200C13-L200C41) for reference code.
/// * See [crate::hal::api::VecZnxMulXpMinusOne] for corresponding public API.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxMulXpMinusOneImpl<B: Backend> {
    fn vec_znx_mul_xp_minus_one_impl<R, A>(module: &Module<B>, p: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [vec_znx_mul_xp_minus_one_ref](https://github.com/phantomzone-org/spqlios-arithmetic/blob/7160f588da49712a042931ea247b4259b95cefcc/spqlios/arithmetic/vec_znx.c#L200C13-L200C41) for reference code.
/// * See [crate::hal::api::VecZnxMulXpMinusOneInplace] for corresponding public API.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxMulXpMinusOneInplaceImpl<B: Backend> {
    fn vec_znx_mul_xp_minus_one_inplace_impl<R>(module: &Module<B>, p: i64, res: &mut R, res_col: usize)
    where
        R: VecZnxToMut;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [crate::implementation::cpu_spqlios::vec_znx::vec_znx_split_ref] for reference code.
/// * See [crate::hal::api::VecZnxSplit] for corresponding public API.
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
/// * See [crate::implementation::cpu_spqlios::vec_znx::vec_znx_merge_ref] for reference code.
/// * See [crate::hal::api::VecZnxMerge] for corresponding public API.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxMergeImpl<B: Backend> {
    fn vec_znx_merge_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: Vec<A>, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [crate::implementation::cpu_spqlios::vec_znx::vec_znx_switch_degree_ref] for reference code.
/// * See [crate::hal::api::VecZnxSwithcDegree] for corresponding public API.
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
/// * See [crate::implementation::cpu_spqlios::vec_znx::vec_znx_copy_ref] for reference code.
/// * See [crate::hal::api::VecZnxCopy] for corresponding public API.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxCopyImpl<B: Backend> {
    fn vec_znx_copy_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [crate::hal::api::VecZnxStd] for corresponding public API.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxStdImpl<B: Backend> {
    fn vec_znx_std_impl<A>(module: &Module<B>, basek: usize, a: &A, a_col: usize) -> f64
    where
        A: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [crate::hal::api::VecZnxFillUniform] for corresponding public API.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxFillUniformImpl<B: Backend> {
    fn vec_znx_fill_uniform_impl<R>(module: &Module<B>, basek: usize, res: &mut R, res_col: usize, k: usize, source: &mut Source)
    where
        R: VecZnxToMut;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [crate::hal::api::VecZnxFillDistF64] for corresponding public API.
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
/// * See [crate::hal::api::VecZnxAddDistF64] for corresponding public API.
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
/// * See [crate::hal::api::VecZnxFillNormal] for corresponding public API.
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
/// * See [crate::hal::api::VecZnxAddNormal] for corresponding public API.
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
/// * See [crate::hal::api::VecZnxEncodeVeci64] for corresponding public API.
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
/// * See [crate::hal::api::VecZnxEncodeCoeffsi64] for corresponding public API.
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
/// * See [crate::hal::api::VecZnxDecodeVeci64] for corresponding public API.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxDecodeVeci64Impl<B: Backend> {
    fn decode_vec_i64_impl<R>(module: &Module<B>, basek: usize, res: &R, res_col: usize, k: usize, data: &mut [i64])
    where
        R: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See \[TODO\] for reference code.
/// * See [crate::hal::api::VecZnxDecodeCoeffsi64] for corresponding public API.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxDecodeCoeffsi64Impl<B: Backend> {
    fn decode_coeff_i64_impl<R>(module: &Module<B>, basek: usize, res: &R, res_col: usize, k: usize, i: usize) -> i64
    where
        R: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See \[TODO\] for reference code.
/// * See [crate::hal::api::VecZnxDecodeVecFloat] for corresponding public API.
/// * See [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxDecodeVecFloatImpl<B: Backend> {
    fn decode_vec_float_impl<R>(module: &Module<B>, basek: usize, res: &R, res_col: usize, data: &mut [Float])
    where
        R: VecZnxToRef;
}
