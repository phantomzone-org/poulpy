use poulpy_hal::{
    api::{Convolution, ModuleN, ScratchTakeBasic, TakeSlice, VecZnxDftApply, VecZnxDftBytesOf},
    layouts::{
        Backend, CnvPVecL, CnvPVecLToMut, CnvPVecLToRef, CnvPVecR, CnvPVecRToMut, CnvPVecRToRef, Module, Scratch, VecZnx,
        VecZnxDft, VecZnxDftToMut, VecZnxToRef, ZnxInfos,
    },
    oep::{CnvPVecBytesOfImpl, CnvPVecLAllocImpl, ConvolutionImpl},
    reference::fft64::convolution::{
        convolution_apply_dft, convolution_apply_dft_tmp_bytes, convolution_pairwise_apply_dft,
        convolution_pairwise_apply_dft_tmp_bytes, convolution_prepare_left, convolution_prepare_right,
    },
};

use crate::{FFT64Avx, module::FFT64ModuleHandle};

unsafe impl CnvPVecLAllocImpl<Self> for FFT64Avx {
    fn cnv_pvec_left_alloc_impl(n: usize, cols: usize, size: usize) -> CnvPVecL<Vec<u8>, Self> {
        CnvPVecL::alloc(n, cols, size)
    }

    fn cnv_pvec_right_alloc_impl(n: usize, cols: usize, size: usize) -> CnvPVecR<Vec<u8>, Self> {
        CnvPVecR::alloc(n, cols, size)
    }
}

unsafe impl CnvPVecBytesOfImpl for FFT64Avx {
    fn bytes_of_cnv_pvec_left_impl(n: usize, cols: usize, size: usize) -> usize {
        Self::layout_prep_word_count() * n * cols * size * size_of::<<FFT64Avx as Backend>::ScalarPrep>()
    }

    fn bytes_of_cnv_pvec_right_impl(n: usize, cols: usize, size: usize) -> usize {
        Self::layout_prep_word_count() * n * cols * size * size_of::<<FFT64Avx as Backend>::ScalarPrep>()
    }
}

unsafe impl ConvolutionImpl<Self> for FFT64Avx
where
    Module<Self>: ModuleN + VecZnxDftBytesOf + VecZnxDftApply<Self>,
{
    fn cnv_prepare_left_tmp_bytes_impl(module: &Module<Self>, res_size: usize, a_size: usize) -> usize {
        module.bytes_of_vec_znx_dft(1, res_size.min(a_size))
    }

    fn cnv_prepare_left_impl<R, A>(module: &Module<Self>, res: &mut R, a: &A, scratch: &mut Scratch<Self>)
    where
        R: CnvPVecLToMut<Self>,
        A: VecZnxToRef,
    {
        let res: &mut CnvPVecL<&mut [u8], FFT64Avx> = &mut res.to_mut();
        let a: &VecZnx<&[u8]> = &a.to_ref();
        let (mut tmp, _) = scratch.take_vec_znx_dft(module, 1, res.size().min(a.size()));
        convolution_prepare_left(module.get_fft_table(), res, a, &mut tmp);
    }

    fn cnv_prepare_right_tmp_bytes_impl(module: &Module<Self>, res_size: usize, a_size: usize) -> usize {
        module.bytes_of_vec_znx_dft(1, res_size.min(a_size))
    }

    fn cnv_prepare_right_impl<R, A>(module: &Module<Self>, res: &mut R, a: &A, scratch: &mut Scratch<Self>)
    where
        R: CnvPVecRToMut<Self>,
        A: VecZnxToRef,
    {
        let res: &mut CnvPVecR<&mut [u8], FFT64Avx> = &mut res.to_mut();
        let a: &VecZnx<&[u8]> = &a.to_ref();
        let (mut tmp, _) = scratch.take_vec_znx_dft(module, 1, res.size().min(a.size()));
        convolution_prepare_right(module.get_fft_table(), res, a, &mut tmp);
    }
    fn cnv_apply_dft_tmp_bytes_impl(
        _module: &Module<Self>,
        res_size: usize,
        _res_offset: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        convolution_apply_dft_tmp_bytes(res_size, a_size, b_size)
    }

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
        A: CnvPVecLToRef<Self>,
        B: CnvPVecRToRef<Self>,
    {
        let res: &mut VecZnxDft<&mut [u8], FFT64Avx> = &mut res.to_mut();
        let a: &CnvPVecL<&[u8], FFT64Avx> = &a.to_ref();
        let b: &CnvPVecR<&[u8], FFT64Avx> = &b.to_ref();
        let (tmp, _) =
            scratch.take_slice(module.cnv_apply_dft_tmp_bytes(res.size(), res_offset, a.size(), b.size()) / size_of::<f64>());
        convolution_apply_dft(res, res_offset, res_col, a, a_col, b, b_col, tmp);
    }

    fn cnv_pairwise_apply_dft_tmp_bytes(
        _module: &Module<Self>,
        res_size: usize,
        _res_offset: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        convolution_pairwise_apply_dft_tmp_bytes(res_size, a_size, b_size)
    }

    fn cnv_pairwise_apply_dft_impl<R, A, B>(
        module: &Module<Self>,
        res: &mut R,
        res_offset: usize,
        res_col: usize,
        a: &A,
        b: &B,
        i: usize,
        j: usize,
        scratch: &mut Scratch<Self>,
    ) where
        R: VecZnxDftToMut<Self>,
        A: CnvPVecLToRef<Self>,
        B: CnvPVecRToRef<Self>,
    {
        let res: &mut VecZnxDft<&mut [u8], FFT64Avx> = &mut res.to_mut();
        let a: &CnvPVecL<&[u8], FFT64Avx> = &a.to_ref();
        let b: &CnvPVecR<&[u8], FFT64Avx> = &b.to_ref();
        let (tmp, _) = scratch
            .take_slice(module.cnv_pairwise_apply_dft_tmp_bytes(res.size(), res_offset, a.size(), b.size()) / size_of::<f64>());
        convolution_pairwise_apply_dft(res, res_offset, res_col, a, b, i, j, tmp);
    }
}
