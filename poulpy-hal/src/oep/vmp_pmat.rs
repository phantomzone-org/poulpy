//! Backend extension points for vector-matrix product (VMP) operations
//! on [`VmpPMat`](crate::layouts::VmpPMat).

use crate::layouts::{
    Backend, MatZnxToRef, Module, Scratch, VecZnxDftToMut, VecZnxDftToRef, VecZnxToRef, VmpPMatToMut, VmpPMatToRef,
};

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/vmp.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/vmp.rs) for reference implementation.
/// * See [crate::api::VmpPrepareTmpBytes] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VmpPrepareTmpBytesImpl<B: Backend> {
    fn vmp_prepare_tmp_bytes_impl(module: &Module<B>, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/vmp.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/vmp.rs) for reference implementation.
/// * See [crate::api::VmpPrepare] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VmpPrepareImpl<B: Backend> {
    fn vmp_prepare_impl<R, A>(module: &Module<B>, res: &mut R, a: &A, scratch: &mut Scratch<B>)
    where
        R: VmpPMatToMut<B>,
        A: MatZnxToRef;
}

#[allow(clippy::too_many_arguments)]
/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/vmp.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/vmp.rs) for reference implementation.
/// * See [crate::api::VmpApplyDftTmpBytes] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VmpApplyDftTmpBytesImpl<B: Backend> {
    fn vmp_apply_dft_tmp_bytes_impl(
        module: &Module<B>,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/vmp.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/vmp.rs) for reference implementation.
/// * See [crate::api::VmpApplyDft] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VmpApplyDftImpl<B: Backend> {
    fn vmp_apply_dft_impl<R, A, C>(module: &Module<B>, res: &mut R, a: &A, b: &C, scratch: &mut Scratch<B>)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxToRef,
        C: VmpPMatToRef<B>;
}

#[allow(clippy::too_many_arguments)]
/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/vmp.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/vmp.rs) for reference implementation.
/// * See [crate::api::VmpApplyDftToDftTmpBytes] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VmpApplyDftToDftTmpBytesImpl<B: Backend> {
    fn vmp_apply_dft_to_dft_tmp_bytes_impl(
        module: &Module<B>,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/vmp.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/vmp.rs) for reference implementation.
/// * See [crate::api::VmpApplyDftToDft] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VmpApplyDftToDftImpl<B: Backend> {
    fn vmp_apply_dft_to_dft_impl<R, A, C>(module: &Module<B>, res: &mut R, a: &A, b: &C, scratch: &mut Scratch<B>)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
        C: VmpPMatToRef<B>;
}

#[allow(clippy::too_many_arguments)]
/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/vmp.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/vmp.rs) for reference implementation.
/// * See [crate::api::VmpApplyDftToDftAddTmpBytes] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VmpApplyDftToDftAddTmpBytesImpl<B: Backend> {
    fn vmp_apply_dft_to_dft_add_tmp_bytes_impl(
        module: &Module<B>,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See [poulpy-backend/src/cpu_fft64_ref/vmp.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/vmp.rs) for reference implementation.
/// * See [crate::api::VmpApplyDftToDftAdd] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VmpApplyDftToDftAddImpl<B: Backend> {
    // Same as [MatZnxDftOps::vmp_apply] except result is added on R instead of overwritting R.
    fn vmp_apply_dft_to_dft_add_impl<R, A, C>(
        module: &Module<B>,
        res: &mut R,
        a: &A,
        b: &C,
        scale: usize,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
        C: VmpPMatToRef<B>;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See TODO.
/// * See [crate::api::VmpZero] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VmpZeroImpl<B: Backend> {
    fn vmp_zero_impl<R>(module: &Module<B>, res: &mut R)
    where
        R: VmpPMatToMut<B>;
}
