//! Scalar-vector product (SVP) operations for [`NTTIfmaRef`](crate::NTTIfmaRef).
//!
//! Implements the `SvpPPol*` and `SvpApply*` OEP traits. SVP multiplies a
//! single scalar polynomial (prepared into NTT domain as `SvpPPol`, IFMA c format)
//! against a `VecZnxDft` (b format), producing a `VecZnxDft` result.
//!
//! - **Prepare**: NTT a `ScalarZnx` into a prepared `SvpPPol` (IFMA c format).
//! - **Apply DFT-to-DFT**: pointwise multiply a prepared scalar against each limb
//!   of a `VecZnxDft`. Available in overwrite and in-place variants.

use poulpy_hal::{
    layouts::{Module, ScalarZnxToRef, SvpPPolToMut, SvpPPolToRef, VecZnxDftToMut, VecZnxDftToRef},
    oep::{SvpApplyDftToDftImpl, SvpApplyDftToDftInplaceImpl, SvpPrepareImpl},
    reference::ntt_ifma::svp::{ntt_ifma_svp_apply_dft_to_dft, ntt_ifma_svp_apply_dft_to_dft_inplace, ntt_ifma_svp_prepare},
};

use crate::NTTIfmaRef;

unsafe impl SvpPrepareImpl<Self> for NTTIfmaRef {
    fn svp_prepare_impl<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: SvpPPolToMut<Self>,
        A: ScalarZnxToRef,
    {
        ntt_ifma_svp_prepare::<R, A, Self>(module, res, res_col, a, a_col);
    }
}

unsafe impl SvpApplyDftToDftImpl<Self> for NTTIfmaRef {
    fn svp_apply_dft_to_dft_impl<R, A, C>(
        module: &Module<Self>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &C,
        b_col: usize,
    ) where
        R: VecZnxDftToMut<Self>,
        A: SvpPPolToRef<Self>,
        C: VecZnxDftToRef<Self>,
    {
        ntt_ifma_svp_apply_dft_to_dft::<R, A, C, Self>(module, res, res_col, a, a_col, b, b_col);
    }
}

unsafe impl SvpApplyDftToDftInplaceImpl for NTTIfmaRef {
    fn svp_apply_dft_to_dft_inplace_impl<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<Self>,
        A: SvpPPolToRef<Self>,
    {
        ntt_ifma_svp_apply_dft_to_dft_inplace::<R, A, Self>(module, res, res_col, a, a_col);
    }
}
