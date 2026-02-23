//! Vector-matrix product (VMP) operations for [`NTT120Ref`](crate::NTT120Ref).
//!
//! Implements the `VmpPMat*` and `VmpApply*` OEP traits. VMP computes the product
//! of a `VecZnxDft` (q120b, row vector) with a `VmpPMat` (prepared matrix in q120c
//! NTT domain), yielding a `VecZnxDft` (q120b) result.
//!
//! - **Allocate / zero**: create and initialize prepared matrices.
//! - **Prepare**: NTT each row of an integer-domain `MatZnx` into a `VmpPMat`.
//! - **Apply DFT-to-DFT**: multiply a frequency-domain vector by the prepared
//!   matrix. The `_add` variant accumulates into an existing result.
//!
//! All apply and prepare operations require scratch space.

use poulpy_hal::{
    api::{TakeSlice, VmpPrepareTmpBytes},
    layouts::{
        MatZnx, MatZnxToRef, Module, Scratch, VecZnxDft, VecZnxDftToMut, VecZnxDftToRef, VmpPMat, VmpPMatOwned, VmpPMatToMut,
        VmpPMatToRef, ZnxInfos,
    },
    oep::{
        VmpApplyDftToDftAddImpl, VmpApplyDftToDftAddTmpBytesImpl, VmpApplyDftToDftImpl, VmpApplyDftToDftTmpBytesImpl,
        VmpPMatAllocBytesImpl, VmpPMatAllocImpl, VmpPrepareImpl, VmpPrepareTmpBytesImpl, VmpZeroImpl,
    },
    reference::ntt120::vmp::{
        ntt120_vmp_apply_dft_to_dft, ntt120_vmp_apply_dft_to_dft_add, ntt120_vmp_apply_dft_to_dft_tmp_bytes, ntt120_vmp_prepare,
        ntt120_vmp_prepare_tmp_bytes, ntt120_vmp_zero,
    },
};

use crate::NTT120Ref;

unsafe impl VmpPMatAllocBytesImpl<Self> for NTT120Ref {
    fn vmp_pmat_bytes_of_impl(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        // Each of the n * rows * cols_in * cols_out * size coefficients is stored as a
        // Q120bScalar (32 bytes = 4 × u64 = 8 × u32 in q120c view).
        n * rows * cols_in * cols_out * size * size_of::<<NTT120Ref as poulpy_hal::layouts::Backend>::ScalarPrep>()
    }
}

unsafe impl VmpPMatAllocImpl<Self> for NTT120Ref {
    fn vmp_pmat_alloc_impl(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> VmpPMatOwned<Self> {
        VmpPMatOwned::alloc(n, rows, cols_in, cols_out, size)
    }
}

unsafe impl VmpPrepareTmpBytesImpl<Self> for NTT120Ref {
    fn vmp_prepare_tmp_bytes_impl(module: &Module<Self>, _rows: usize, _cols_in: usize, _cols_out: usize, _size: usize) -> usize {
        ntt120_vmp_prepare_tmp_bytes(module.n())
    }
}

unsafe impl VmpPrepareImpl<Self> for NTT120Ref {
    fn vmp_prepare_impl<R, A>(module: &Module<Self>, res: &mut R, a: &A, scratch: &mut Scratch<Self>)
    where
        R: VmpPMatToMut<Self>,
        A: MatZnxToRef,
    {
        let a_ref: MatZnx<&[u8]> = a.to_ref();
        let (tmp, _) =
            scratch.take_slice::<u8>(module.vmp_prepare_tmp_bytes(a_ref.rows(), a_ref.cols_in(), a_ref.cols_out(), a_ref.size()));
        ntt120_vmp_prepare::<R, A, Self>(module, res, a, tmp);
    }
}

unsafe impl VmpApplyDftToDftTmpBytesImpl<Self> for NTT120Ref {
    fn vmp_apply_dft_to_dft_tmp_bytes_impl(
        _module: &Module<Self>,
        _res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        _b_cols_out: usize,
        _b_size: usize,
    ) -> usize {
        ntt120_vmp_apply_dft_to_dft_tmp_bytes(a_size, b_rows, b_cols_in)
    }
}

unsafe impl VmpApplyDftToDftAddTmpBytesImpl<Self> for NTT120Ref {
    fn vmp_apply_dft_to_dft_add_tmp_bytes_impl(
        _module: &Module<Self>,
        _res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        _b_cols_out: usize,
        _b_size: usize,
    ) -> usize {
        ntt120_vmp_apply_dft_to_dft_tmp_bytes(a_size, b_rows, b_cols_in)
    }
}

unsafe impl VmpApplyDftToDftImpl<Self> for NTT120Ref
where
    Scratch<Self>: TakeSlice,
    NTT120Ref: VmpApplyDftToDftTmpBytesImpl<Self>,
{
    fn vmp_apply_dft_to_dft_impl<R, A, C>(module: &Module<Self>, res: &mut R, a: &A, pmat: &C, scratch: &mut Scratch<Self>)
    where
        R: VecZnxDftToMut<Self>,
        A: VecZnxDftToRef<Self>,
        C: VmpPMatToRef<Self>,
    {
        let mut res_ref: VecZnxDft<&mut [u8], Self> = res.to_mut();
        let a_ref: VecZnxDft<&[u8], Self> = a.to_ref();
        let pmat_ref: VmpPMat<&[u8], Self> = pmat.to_ref();

        let (tmp, _) = scratch.take_slice::<u8>(Self::vmp_apply_dft_to_dft_tmp_bytes_impl(
            module,
            res_ref.size(),
            a_ref.size(),
            pmat_ref.rows(),
            pmat_ref.cols_in(),
            pmat_ref.cols_out(),
            pmat_ref.size(),
        ));
        ntt120_vmp_apply_dft_to_dft::<_, _, _, Self>(module, &mut res_ref, &a_ref, &pmat_ref, tmp);
    }
}

unsafe impl VmpApplyDftToDftAddImpl<Self> for NTT120Ref
where
    Scratch<Self>: TakeSlice,
    NTT120Ref: VmpApplyDftToDftTmpBytesImpl<Self>,
{
    fn vmp_apply_dft_to_dft_add_impl<R, A, C>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        pmat: &C,
        limb_offset: usize,
        scratch: &mut Scratch<Self>,
    ) where
        R: VecZnxDftToMut<Self>,
        A: VecZnxDftToRef<Self>,
        C: VmpPMatToRef<Self>,
    {
        let mut res_ref: VecZnxDft<&mut [u8], Self> = res.to_mut();
        let a_ref: VecZnxDft<&[u8], Self> = a.to_ref();
        let pmat_ref: VmpPMat<&[u8], Self> = pmat.to_ref();

        let (tmp, _) = scratch.take_slice::<u8>(Self::vmp_apply_dft_to_dft_tmp_bytes_impl(
            module,
            res_ref.size(),
            a_ref.size(),
            pmat_ref.rows(),
            pmat_ref.cols_in(),
            pmat_ref.cols_out(),
            pmat_ref.size(),
        ));
        ntt120_vmp_apply_dft_to_dft_add::<_, _, _, Self>(
            module,
            &mut res_ref,
            &a_ref,
            &pmat_ref,
            limb_offset * pmat_ref.cols_out(),
            tmp,
        );
    }
}

unsafe impl VmpZeroImpl<Self> for NTT120Ref {
    fn vmp_zero_impl<R>(_module: &Module<Self>, res: &mut R)
    where
        R: VmpPMatToMut<Self>,
    {
        ntt120_vmp_zero::<R, Self>(res);
    }
}
