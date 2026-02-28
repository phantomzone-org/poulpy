//! Polynomial convolution operations for [`NTT120Avx`](super::NTT120Avx).
//!
//! Delegates to the reference HAL functions in
//! `poulpy_hal::reference::ntt120::convolution`.  The pipeline:
//!
//! 1. **Prepare left** — encode a `VecZnx` into `CnvPVecL` (q120b NTT domain).
//! 2. **Prepare right** — encode a `VecZnx` into `CnvPVecR` (q120c NTT domain).
//! 3. **Apply DFT** — compute `res[k] = Σ left[j] ⊙ right[k−j]` (bbc product).
//! 4. **By-const apply** — coefficient-domain convolution into `VecZnxBig` (i128).
//! 5. **Pairwise apply DFT** — sum two parallel bbc convolutions.

use poulpy_hal::{
    api::TakeSlice,
    layouts::{CnvPVecLToMut, CnvPVecRToMut, Module, Scratch, VecZnxBigToMut, VecZnxDftToMut, VecZnxToRef},
    oep::ConvolutionImpl,
    reference::ntt120::convolution::{
        ntt120_cnv_apply_dft, ntt120_cnv_apply_dft_tmp_bytes, ntt120_cnv_by_const_apply, ntt120_cnv_by_const_apply_tmp_bytes,
        ntt120_cnv_pairwise_apply_dft, ntt120_cnv_pairwise_apply_dft_tmp_bytes, ntt120_cnv_prepare_left,
        ntt120_cnv_prepare_left_tmp_bytes, ntt120_cnv_prepare_right, ntt120_cnv_prepare_right_tmp_bytes,
    },
};

use std::mem::size_of;

use super::NTT120Avx;

unsafe impl ConvolutionImpl<Self> for NTT120Avx
where
    Scratch<Self>: TakeSlice,
{
    fn cnv_prepare_left_tmp_bytes_impl(module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
        ntt120_cnv_prepare_left_tmp_bytes(module.n())
    }

    fn cnv_prepare_left_impl<R, A>(module: &Module<Self>, res: &mut R, a: &A, scratch: &mut Scratch<Self>)
    where
        R: CnvPVecLToMut<Self>,
        A: VecZnxToRef,
    {
        let bytes = Self::cnv_prepare_left_tmp_bytes_impl(module, 0, 0);
        let (tmp, _) = scratch.take_slice::<u8>(bytes);
        ntt120_cnv_prepare_left::<_, _, Self>(module, res, a, tmp);
    }

    fn cnv_prepare_right_tmp_bytes_impl(module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
        ntt120_cnv_prepare_right_tmp_bytes(module.n())
    }

    fn cnv_prepare_right_impl<R, A>(module: &Module<Self>, res: &mut R, a: &A, scratch: &mut Scratch<Self>)
    where
        R: CnvPVecRToMut<Self>,
        A: VecZnxToRef + poulpy_hal::layouts::ZnxInfos,
    {
        let bytes = Self::cnv_prepare_right_tmp_bytes_impl(module, 0, 0);
        let (tmp, _) = scratch.take_slice::<u64>(bytes / size_of::<u64>());
        ntt120_cnv_prepare_right::<_, _, Self>(module, res, a, tmp);
    }

    fn cnv_apply_dft_tmp_bytes_impl(
        _module: &Module<Self>,
        res_size: usize,
        _res_offset: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        ntt120_cnv_apply_dft_tmp_bytes(res_size, a_size, b_size)
    }

    fn cnv_by_const_apply_tmp_bytes_impl(
        _module: &Module<Self>,
        res_size: usize,
        _res_offset: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        ntt120_cnv_by_const_apply_tmp_bytes(res_size, a_size, b_size)
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
        let bytes = ntt120_cnv_by_const_apply_tmp_bytes(0, 0, 0);
        let (tmp, _) = scratch.take_slice::<u8>(bytes);
        ntt120_cnv_by_const_apply::<_, _, Self>(res, res_offset, res_col, a, a_col, b, tmp);
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
        let bytes = ntt120_cnv_apply_dft_tmp_bytes(0, 0, 0);
        let (tmp, _) = scratch.take_slice::<u8>(bytes);
        ntt120_cnv_apply_dft::<_, _, _, Self>(module, res, res_offset, res_col, a, a_col, b, b_col, tmp);
    }

    fn cnv_pairwise_apply_dft_tmp_bytes(
        _module: &Module<Self>,
        res_size: usize,
        _res_offset: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        ntt120_cnv_pairwise_apply_dft_tmp_bytes(res_size, a_size, b_size)
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
        let bytes = ntt120_cnv_pairwise_apply_dft_tmp_bytes(0, 0, 0);
        let (tmp, _) = scratch.take_slice::<u8>(bytes);
        ntt120_cnv_pairwise_apply_dft::<_, _, _, Self>(module, res, res_offset, res_col, a, b, col_0, col_1, tmp);
    }
}
