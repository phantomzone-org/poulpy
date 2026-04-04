//! Polynomial convolution operations for [`NTTIfmaRef`](crate::NTTIfmaRef).
//!
//! Delegates to the shared IFMA reference convolution pipeline in
//! `poulpy_hal::reference::ntt_ifma::convolution`.

use poulpy_hal::{
    api::TakeSlice,
    layouts::{CnvPVecLToMut, CnvPVecRToMut, Module, Scratch, VecZnxBigToMut, VecZnxDftToMut, VecZnxToRef},
    oep::ConvolutionImpl,
    reference::ntt_ifma::convolution::{
        ntt_ifma_cnv_apply_dft, ntt_ifma_cnv_apply_dft_tmp_bytes, ntt_ifma_cnv_by_const_apply,
        ntt_ifma_cnv_by_const_apply_tmp_bytes, ntt_ifma_cnv_pairwise_apply_dft, ntt_ifma_cnv_pairwise_apply_dft_tmp_bytes,
        ntt_ifma_cnv_prepare_left, ntt_ifma_cnv_prepare_left_tmp_bytes, ntt_ifma_cnv_prepare_right,
        ntt_ifma_cnv_prepare_right_tmp_bytes,
    },
};

use crate::NTTIfmaRef;
use std::mem::size_of;

unsafe impl ConvolutionImpl<Self> for NTTIfmaRef
where
    Scratch<Self>: TakeSlice,
{
    fn cnv_prepare_left_tmp_bytes_impl(module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
        ntt_ifma_cnv_prepare_left_tmp_bytes(module.n())
    }

    fn cnv_prepare_left_impl<R, A>(module: &Module<Self>, res: &mut R, a: &A, scratch: &mut Scratch<Self>)
    where
        R: CnvPVecLToMut<Self>,
        A: VecZnxToRef,
    {
        let bytes = Self::cnv_prepare_left_tmp_bytes_impl(module, 0, 0);
        let (tmp, _) = scratch.take_slice::<u8>(bytes);
        ntt_ifma_cnv_prepare_left::<_, _, Self>(module, res, a, tmp);
    }

    fn cnv_prepare_right_tmp_bytes_impl(module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
        ntt_ifma_cnv_prepare_right_tmp_bytes(module.n())
    }

    fn cnv_prepare_right_impl<R, A>(module: &Module<Self>, res: &mut R, a: &A, scratch: &mut Scratch<Self>)
    where
        R: CnvPVecRToMut<Self>,
        A: VecZnxToRef + poulpy_hal::layouts::ZnxInfos,
    {
        let bytes = Self::cnv_prepare_right_tmp_bytes_impl(module, 0, 0);
        let (tmp, _) = scratch.take_slice::<u64>(bytes / size_of::<u64>());
        ntt_ifma_cnv_prepare_right::<_, _, Self>(module, res, a, tmp);
    }

    fn cnv_apply_dft_tmp_bytes_impl(
        _module: &Module<Self>,
        res_size: usize,
        _res_offset: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        ntt_ifma_cnv_apply_dft_tmp_bytes(res_size, a_size, b_size)
    }

    fn cnv_by_const_apply_tmp_bytes_impl(
        _module: &Module<Self>,
        res_size: usize,
        _res_offset: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        ntt_ifma_cnv_by_const_apply_tmp_bytes(res_size, a_size, b_size)
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_by_const_apply_impl<R, A>(
        _module: &Module<Self>,
        res: &mut R,
        res_offset: usize,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &[i64],
        scratch: &mut Scratch<Self>,
    ) where
        R: VecZnxBigToMut<Self>,
        A: VecZnxToRef,
    {
        let bytes = ntt_ifma_cnv_by_const_apply_tmp_bytes(0, 0, 0);
        let (tmp, _) = scratch.take_slice::<u8>(bytes);
        ntt_ifma_cnv_by_const_apply::<_, _, Self>(res, res_offset, res_col, a, a_col, b, tmp);
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_apply_dft_impl<R, A, B>(
        module: &Module<Self>,
        res: &mut R,
        res_offset: usize,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &B,
        b_col: usize,
        scratch: &mut Scratch<Self>,
    ) where
        R: VecZnxDftToMut<Self>,
        A: poulpy_hal::layouts::CnvPVecLToRef<Self>,
        B: poulpy_hal::layouts::CnvPVecRToRef<Self>,
    {
        let bytes = ntt_ifma_cnv_apply_dft_tmp_bytes(0, 0, 0);
        let (tmp, _) = scratch.take_slice::<u8>(bytes);
        ntt_ifma_cnv_apply_dft::<_, _, _, Self>(module, res, res_offset, res_col, a, a_col, b, b_col, tmp);
    }

    fn cnv_pairwise_apply_dft_tmp_bytes(
        _module: &Module<Self>,
        res_size: usize,
        _res_offset: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        ntt_ifma_cnv_pairwise_apply_dft_tmp_bytes(res_size, a_size, b_size)
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_pairwise_apply_dft_impl<R, A, B>(
        module: &Module<Self>,
        res: &mut R,
        res_offset: usize,
        res_col: usize,
        a: &A,
        b: &B,
        col_0: usize,
        col_1: usize,
        scratch: &mut Scratch<Self>,
    ) where
        R: VecZnxDftToMut<Self>,
        A: poulpy_hal::layouts::CnvPVecLToRef<Self>,
        B: poulpy_hal::layouts::CnvPVecRToRef<Self>,
    {
        let bytes = ntt_ifma_cnv_pairwise_apply_dft_tmp_bytes(0, 0, 0);
        let (tmp, _) = scratch.take_slice::<u8>(bytes);
        ntt_ifma_cnv_pairwise_apply_dft::<_, _, _, Self>(module, res, res_offset, res_col, a, b, col_0, col_1, tmp);
    }
}
