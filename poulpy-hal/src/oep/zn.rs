use crate::{
    layouts::{Backend, Scratch, ZnToMut},
    source::Source,
};

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/zn.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/zn.rs) for reference implementation.
/// * See [crate::api::ZnNormalizeTmpBytes] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait ZnNormalizeTmpBytesImpl<B: Backend> {
    fn zn_normalize_tmp_bytes_impl(n: usize) -> usize;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/zn.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/zn.rs) for reference implementation.
/// * See [crate::api::ZnNormalizeInplace] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait ZnNormalizeInplaceImpl<B: Backend> {
    fn zn_normalize_inplace_impl<R>(n: usize, base2k: usize, res: &mut R, res_col: usize, scratch: &mut Scratch<B>)
    where
        R: ZnToMut;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/zn.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/zn.rs) for reference implementation.
/// * See [crate::api::ZnFillUniform] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait ZnFillUniformImpl<B: Backend> {
    fn zn_fill_uniform_impl<R>(n: usize, base2k: usize, res: &mut R, res_col: usize, source: &mut Source)
    where
        R: ZnToMut;
}

#[allow(clippy::too_many_arguments)]
/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/zn.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/zn.rs) for reference implementation.
/// * See [crate::api::ZnFillNormal] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait ZnFillNormalImpl<B: Backend> {
    fn zn_fill_normal_impl<R>(
        n: usize,
        base2k: usize,
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
/// * See [poulpy-backend/src/cpu_fft64_ref/zn.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/zn.rs) for reference implementation.
/// * See [crate::api::ZnAddNormal] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait ZnAddNormalImpl<B: Backend> {
    fn zn_add_normal_impl<R>(
        n: usize,
        base2k: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        sigma: f64,
        bound: f64,
    ) where
        R: ZnToMut;
}
