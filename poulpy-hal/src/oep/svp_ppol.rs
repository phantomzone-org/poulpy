//! Backend extension points for scalar-vector product (SVP) operations
//! on [`SvpPPol`](crate::layouts::SvpPPol).

use crate::layouts::{Backend, Module, ScalarZnxToRef, SvpPPolToMut, SvpPPolToRef, VecZnxDftToMut, VecZnxDftToRef, VecZnxToRef};

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See the [poulpy-backend/src/cpu_fft64_ref/svp.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/svp.rs) reference implementation.
/// * See [crate::api::SvpPrepare] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait SvpPrepareImpl<B: Backend> {
    fn svp_prepare_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: SvpPPolToMut<B>,
        A: ScalarZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See the [poulpy-backend/src/cpu_fft64_ref/svp.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/svp.rs) reference implementation.
/// * See [crate::api::SvpApplyDft] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait SvpApplyDftImpl<B: Backend> {
    fn svp_apply_dft_impl<R, A, C>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: SvpPPolToRef<B>,
        C: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See the [poulpy-backend/src/cpu_fft64_ref/svp.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/svp.rs) reference implementation.
/// * See [crate::api::SvpApplyDftToDft] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait SvpApplyDftToDftImpl<B: Backend> {
    fn svp_apply_dft_to_dft_impl<R, A, C>(
        module: &Module<B>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &C,
        b_col: usize,
    ) where
        R: VecZnxDftToMut<B>,
        A: SvpPPolToRef<B>,
        C: VecZnxDftToRef<B>;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See the [poulpy-backend/src/cpu_fft64_ref/svp.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/svp.rs) reference implementation.
/// * See [crate::api::SvpApplyDftToDftAdd] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait SvpApplyDftToDftAddImpl<B: Backend> {
    fn svp_apply_dft_to_dft_add_impl<R, A, C>(
        module: &Module<B>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &C,
        b_col: usize,
    ) where
        R: VecZnxDftToMut<B>,
        A: SvpPPolToRef<B>,
        C: VecZnxDftToRef<B>;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See the [poulpy-backend/src/cpu_fft64_ref/svp.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/svp.rs) reference implementation.
/// * See [crate::api::SvpApplyDftToDftInplace] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait SvpApplyDftToDftInplaceImpl: Backend {
    fn svp_apply_dft_to_dft_inplace_impl<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<Self>,
        A: SvpPPolToRef<Self>;
}
