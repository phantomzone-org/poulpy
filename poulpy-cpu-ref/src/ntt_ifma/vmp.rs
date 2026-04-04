//! Vector-matrix product (VMP) operations for [`NTTIfmaRef`](crate::NTTIfmaRef).
//!
//! Implements the `VmpPMat*` and `VmpApply*` OEP traits. VMP computes the product
//! of a `VecZnxDft` (b format, row vector) with a `VmpPMat` (prepared matrix in
//! IFMA c NTT domain), yielding a `VecZnxDft` (b format) result.
//!
//! - **Allocate / zero**: create and initialize prepared matrices.
//! - **Prepare**: NTT each row of an integer-domain `MatZnx` into a `VmpPMat`.
//! - **Apply DFT-to-DFT**: multiply a frequency-domain vector by the prepared
//!   matrix, with optional `limb_offset` to select which digit to accumulate.
//!
//! All apply and prepare operations require scratch space.

use poulpy_hal::{
    api::{TakeSlice, VmpPrepareTmpBytes},
    layouts::{
        MatZnx, MatZnxToRef, Module, Scratch, VecZnxDft, VecZnxDftToMut, VecZnxDftToRef, VmpPMat, VmpPMatToMut, VmpPMatToRef,
        ZnxInfos,
    },
    oep::{VmpApplyDftToDftImpl, VmpApplyDftToDftTmpBytesImpl, VmpPrepareImpl, VmpPrepareTmpBytesImpl, VmpZeroImpl},
    reference::ntt_ifma::vmp::{
        ntt_ifma_vmp_apply_dft_to_dft, ntt_ifma_vmp_apply_dft_to_dft_tmp_bytes, ntt_ifma_vmp_prepare,
        ntt_ifma_vmp_prepare_tmp_bytes, ntt_ifma_vmp_zero,
    },
};

use std::mem::size_of;

use crate::NTTIfmaRef;

unsafe impl VmpPrepareTmpBytesImpl<Self> for NTTIfmaRef {
    fn vmp_prepare_tmp_bytes_impl(module: &Module<Self>, _rows: usize, _cols_in: usize, _cols_out: usize, _size: usize) -> usize {
        ntt_ifma_vmp_prepare_tmp_bytes(module.n())
    }
}

unsafe impl VmpPrepareImpl<Self> for NTTIfmaRef {
    fn vmp_prepare_impl<R, A>(module: &Module<Self>, res: &mut R, a: &A, scratch: &mut Scratch<Self>)
    where
        R: VmpPMatToMut<Self>,
        A: MatZnxToRef,
    {
        let a_ref: MatZnx<&[u8]> = a.to_ref();
        let bytes = module.vmp_prepare_tmp_bytes(a_ref.rows(), a_ref.cols_in(), a_ref.cols_out(), a_ref.size());
        let (tmp, _) = scratch.take_slice::<u64>(bytes / size_of::<u64>());
        ntt_ifma_vmp_prepare::<R, A, Self>(module, res, a, tmp);
    }
}

unsafe impl VmpApplyDftToDftTmpBytesImpl<Self> for NTTIfmaRef {
    fn vmp_apply_dft_to_dft_tmp_bytes_impl(
        _module: &Module<Self>,
        _res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        _b_cols_out: usize,
        _b_size: usize,
    ) -> usize {
        ntt_ifma_vmp_apply_dft_to_dft_tmp_bytes(a_size, b_rows, b_cols_in)
    }
}

unsafe impl VmpApplyDftToDftImpl<Self> for NTTIfmaRef
where
    Scratch<Self>: TakeSlice,
    NTTIfmaRef: VmpApplyDftToDftTmpBytesImpl<Self>,
{
    fn vmp_apply_dft_to_dft_impl<R, A, C>(
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

        let bytes = Self::vmp_apply_dft_to_dft_tmp_bytes_impl(
            module,
            res_ref.size(),
            a_ref.size(),
            pmat_ref.rows(),
            pmat_ref.cols_in(),
            pmat_ref.cols_out(),
            pmat_ref.size(),
        );
        let (tmp, _) = scratch.take_slice::<u64>(bytes / size_of::<u64>());
        ntt_ifma_vmp_apply_dft_to_dft::<_, _, _, Self>(module, &mut res_ref, &a_ref, &pmat_ref, limb_offset, tmp);
    }
}

unsafe impl VmpZeroImpl<Self> for NTTIfmaRef {
    fn vmp_zero_impl<R>(_module: &Module<Self>, res: &mut R)
    where
        R: VmpPMatToMut<Self>,
    {
        ntt_ifma_vmp_zero::<R, Self>(res);
    }
}
