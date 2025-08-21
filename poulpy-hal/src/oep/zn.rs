use rand_distr::Distribution;

use crate::{
    layouts::{Backend, Scratch, ZnToMut},
    source::Source,
};

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [zn_normalize_base2k_ref](https://github.com/phantomzone-org/spqlios-arithmetic/blob/32a3f5fcce9863b58e949f2dfd5abc1bfbaa09b4/spqlios/arithmetic/zn64.c#L9) for reference code.
/// * See [crate::api::ZnxNormalizeInplace] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait ZnNormalizeInplaceImpl<B: Backend> {
    fn zn_normalize_inplace_impl<A>(n: usize, basek: usize, a: &mut A, a_col: usize, scratch: &mut Scratch<B>)
    where
        A: ZnToMut;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [crate::api::ZnFillUniform] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait ZnFillUniformImpl<B: Backend> {
    fn zn_fill_uniform_impl<R>(n: usize, basek: usize, res: &mut R, res_col: usize, k: usize, source: &mut Source)
    where
        R: ZnToMut;
}

#[allow(clippy::too_many_arguments)]
/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [crate::api::ZnFillDistF64] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait ZnFillDistF64Impl<B: Backend> {
    fn zn_fill_dist_f64_impl<R, D: Distribution<f64>>(
        n: usize,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        dist: D,
        bound: f64,
    ) where
        R: ZnToMut;
}

#[allow(clippy::too_many_arguments)]
/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [crate::api::ZnAddDistF64] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait ZnAddDistF64Impl<B: Backend> {
    fn zn_add_dist_f64_impl<R, D: Distribution<f64>>(
        n: usize,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        dist: D,
        bound: f64,
    ) where
        R: ZnToMut;
}

#[allow(clippy::too_many_arguments)]
/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [crate::api::ZnFillNormal] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait ZnFillNormalImpl<B: Backend> {
    fn zn_fill_normal_impl<R>(
        n: usize,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        sigma: f64,
        bound: f64,
    ) where
        R: ZnToMut;
}

#[allow(clippy::too_many_arguments)]
/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [crate::api::ZnAddNormal] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait ZnAddNormalImpl<B: Backend> {
    fn zn_add_normal_impl<R>(
        n: usize,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        sigma: f64,
        bound: f64,
    ) where
        R: ZnToMut;
}
