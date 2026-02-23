//! Scalar-vector product (SVP) operations for [`NTT120Ref`](crate::NTT120Ref).
//!
//! Implements the `SvpPPol*` and `SvpApply*` OEP traits. SVP multiplies a
//! single scalar polynomial (prepared into NTT domain as `SvpPPol`, q120c format)
//! against a `VecZnxDft` (q120b format), producing a `VecZnxDft` result.
//!
//! - **Prepare**: NTT a `ScalarZnx` into a prepared `SvpPPol` (q120c).
//! - **Apply DFT-to-DFT**: pointwise multiply a prepared scalar against each limb
//!   of a `VecZnxDft`. Available in overwrite, accumulate, and in-place variants.

use std::mem::size_of;

use poulpy_hal::{
    layouts::{Backend, Module, ScalarZnxToRef, SvpPPolOwned, SvpPPolToMut, SvpPPolToRef, VecZnxDftToMut, VecZnxDftToRef},
    oep::{
        SvpApplyDftToDftAddImpl, SvpApplyDftToDftImpl, SvpApplyDftToDftInplaceImpl, SvpPPolAllocBytesImpl, SvpPPolAllocImpl,
        SvpPPolFromBytesImpl, SvpPrepareImpl,
    },
    reference::ntt120::{
        svp::{ntt120_svp_apply_dft_to_dft, ntt120_svp_apply_dft_to_dft_add, ntt120_svp_apply_dft_to_dft_inplace, ntt120_svp_prepare},
    },
};

use crate::NTT120Ref;

unsafe impl SvpPPolFromBytesImpl<Self> for NTT120Ref {
    fn svp_ppol_from_bytes_impl(n: usize, cols: usize, bytes: Vec<u8>) -> SvpPPolOwned<Self> {
        SvpPPolOwned::from_bytes(n, cols, bytes)
    }
}

unsafe impl SvpPPolAllocImpl<Self> for NTT120Ref {
    fn svp_ppol_alloc_impl(n: usize, cols: usize) -> SvpPPolOwned<Self> {
        SvpPPolOwned::alloc(n, cols)
    }
}

unsafe impl SvpPPolAllocBytesImpl<Self> for NTT120Ref {
    fn svp_ppol_bytes_of_impl(n: usize, cols: usize) -> usize {
        Self::layout_prep_word_count() * n * cols * size_of::<<NTT120Ref as Backend>::ScalarPrep>()
    }
}

unsafe impl SvpPrepareImpl<Self> for NTT120Ref {
    fn svp_prepare_impl<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: SvpPPolToMut<Self>,
        A: ScalarZnxToRef,
    {
        ntt120_svp_prepare::<R, A, Self>(module, res, res_col, a, a_col);
    }
}

unsafe impl SvpApplyDftToDftImpl<Self> for NTT120Ref {
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
        ntt120_svp_apply_dft_to_dft::<R, A, C, Self>(module, res, res_col, a, a_col, b, b_col);
    }
}

unsafe impl SvpApplyDftToDftAddImpl<Self> for NTT120Ref {
    fn svp_apply_dft_to_dft_add_impl<R, A, C>(
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
        ntt120_svp_apply_dft_to_dft_add::<R, A, C, Self>(module, res, res_col, a, a_col, b, b_col);
    }
}

unsafe impl SvpApplyDftToDftInplaceImpl for NTT120Ref {
    fn svp_apply_dft_to_dft_inplace_impl<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<Self>,
        A: SvpPPolToRef<Self>,
    {
        ntt120_svp_apply_dft_to_dft_inplace::<R, A, Self>(module, res, res_col, a, a_col);
    }
}
