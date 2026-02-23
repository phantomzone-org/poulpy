//! Scalar-vector product (SVP) operations for [`FFT64Ref`](crate::FFT64Ref).
//!
//! Implements the `SvpPPol*` and `SvpApply*` OEP traits. SVP multiplies a
//! single scalar polynomial (prepared into frequency domain as `SvpPPol`) against
//! a `VecZnxDft`, producing a `VecZnxDft` result. This is the frequency-domain
//! analogue of scaling each row of a ciphertext vector by a common polynomial.
//!
//! - **Prepare**: FFT a `ScalarZnx` into a prepared `SvpPPol`.
//! - **Apply DFT-to-DFT**: pointwise multiply a prepared scalar against each limb
//!   of a `VecZnxDft`. Available in out-of-place and inplace variants.

use poulpy_hal::{
    layouts::{Backend, Module, ScalarZnxToRef, SvpPPolOwned, SvpPPolToMut, SvpPPolToRef, VecZnxDftToMut, VecZnxDftToRef},
    oep::{
        SvpApplyDftToDftImpl, SvpApplyDftToDftInplaceImpl, SvpPPolAllocBytesImpl, SvpPPolAllocImpl, SvpPPolFromBytesImpl,
        SvpPrepareImpl,
    },
    reference::fft64::svp::{svp_apply_dft_to_dft, svp_apply_dft_to_dft_inplace, svp_prepare},
};

use super::{FFT64Ref, module::FFT64ModuleHandle};

unsafe impl SvpPPolFromBytesImpl<Self> for FFT64Ref {
    fn svp_ppol_from_bytes_impl(n: usize, cols: usize, bytes: Vec<u8>) -> SvpPPolOwned<Self> {
        SvpPPolOwned::from_bytes(n, cols, bytes)
    }
}

unsafe impl SvpPPolAllocImpl<Self> for FFT64Ref {
    fn svp_ppol_alloc_impl(n: usize, cols: usize) -> SvpPPolOwned<Self> {
        SvpPPolOwned::alloc(n, cols)
    }
}

unsafe impl SvpPPolAllocBytesImpl<Self> for FFT64Ref {
    fn svp_ppol_bytes_of_impl(n: usize, cols: usize) -> usize {
        Self::layout_prep_word_count() * n * cols * size_of::<f64>()
    }
}

unsafe impl SvpPrepareImpl<Self> for FFT64Ref {
    fn svp_prepare_impl<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: SvpPPolToMut<Self>,
        A: ScalarZnxToRef,
    {
        svp_prepare(module.get_fft_table(), res, res_col, a, a_col);
    }
}

unsafe impl SvpApplyDftToDftImpl<Self> for FFT64Ref {
    fn svp_apply_dft_to_dft_impl<R, A, B>(
        _module: &Module<Self>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &B,
        b_col: usize,
    ) where
        R: VecZnxDftToMut<Self>,
        A: SvpPPolToRef<Self>,
        B: VecZnxDftToRef<Self>,
    {
        svp_apply_dft_to_dft(res, res_col, a, a_col, b, b_col);
    }
}

unsafe impl SvpApplyDftToDftInplaceImpl for FFT64Ref {
    fn svp_apply_dft_to_dft_inplace_impl<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<Self>,
        A: SvpPPolToRef<Self>,
    {
        svp_apply_dft_to_dft_inplace(res, res_col, a, a_col);
    }
}
