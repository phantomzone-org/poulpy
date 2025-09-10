use poulpy_hal::{
    api::{TakeSlice, VmpPrepareTmpBytes},
    layouts::{
        Backend, MatZnx, MatZnxToRef, Module, Scratch, VecZnxDft, VecZnxDftToMut, VecZnxDftToRef, VmpPMat, VmpPMatOwned,
        VmpPMatToMut, VmpPMatToRef, ZnxInfos,
    },
    oep::{
        VmpApplyDftToDftImpl, VmpApplyDftToDftTmpBytesImpl, VmpPMatAllocBytesImpl, VmpPMatAllocImpl, VmpPrepareImpl,
        VmpPrepareTmpBytesImpl,
    },
    reference::{
        reim::{ReimArithmeticAvx, ReimArithmeticRef, ReimConvAvx, ReimConvRef, ReimFFTAvx, ReimFFTRef},
        reim4::{Reim4BlkAvx, Reim4BlkRef},
        vmp::fft64::{vmp_apply_dft_to_dft, vmp_apply_dft_to_dft_tmp_bytes, vmp_prepare, vmp_prepare_tmp_bytes},
    },
};

use crate::cpu_ref::{FFT64, fft64::module::FFT64ModuleHandle};

unsafe impl VmpPMatAllocBytesImpl<FFT64> for FFT64 {
    fn vmp_pmat_alloc_bytes_impl(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        FFT64::layout_prep_word_count() * n * rows * cols_in * cols_out * size * size_of::<f64>()
    }
}

unsafe impl VmpPMatAllocImpl<Self> for FFT64 {
    fn vmp_pmat_alloc_impl(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> VmpPMatOwned<Self> {
        VmpPMatOwned::alloc(n, rows, cols_in, cols_out, size)
    }
}

unsafe impl VmpApplyDftToDftImpl<Self> for FFT64
where
    Scratch<Self>: TakeSlice,
    FFT64: VmpApplyDftToDftTmpBytesImpl<Self>,
{
    fn vmp_apply_dft_to_dft_impl<R, A, C>(module: &Module<Self>, res: &mut R, a: &A, pmat: &C, scratch: &mut Scratch<Self>)
    where
        R: VecZnxDftToMut<Self>,
        A: VecZnxDftToRef<Self>,
        C: VmpPMatToRef<Self>,
    {
        let mut res: VecZnxDft<&mut [u8], FFT64> = res.to_mut();
        let a: VecZnxDft<&[u8], FFT64> = a.to_ref();
        let pmat: VmpPMat<&[u8], FFT64> = pmat.to_ref();

        let (tmp, _) = scratch.take_slice(
            FFT64::vmp_apply_dft_to_dft_tmp_bytes_impl(
                module,
                res.size(),
                a.size(),
                pmat.rows(),
                pmat.cols_in(),
                pmat.cols_out(),
                pmat.size(),
            ) / size_of::<f64>(),
        );
        if std::is_x86_feature_detected!("avx2") {
            vmp_apply_dft_to_dft::<_, _, _, _, ReimArithmeticAvx, Reim4BlkAvx>(&mut res, &a, &pmat, tmp);
        } else {
            vmp_apply_dft_to_dft::<_, _, _, _, ReimArithmeticRef, Reim4BlkRef>(&mut res, &a, &pmat, tmp);
        }
    }
}

unsafe impl VmpPrepareTmpBytesImpl<FFT64> for FFT64 {
    fn vmp_prepare_tmp_bytes_impl(
        module: &Module<FFT64>,
        _rows: usize,
        _cols_in: usize,
        _cols_out: usize,
        _size: usize,
    ) -> usize {
        vmp_prepare_tmp_bytes(module.n())
    }
}

unsafe impl VmpPrepareImpl<FFT64> for FFT64 {
    fn vmp_prepare_impl<R, A>(module: &Module<FFT64>, res: &mut R, a: &A, scratch: &mut Scratch<FFT64>)
    where
        R: VmpPMatToMut<FFT64>,
        A: MatZnxToRef,
    {
        let mut res: VmpPMat<&mut [u8], FFT64> = res.to_mut();
        let a: MatZnx<&[u8]> = a.to_ref();
        let (tmp, _) = scratch.take_slice(module.vmp_prepare_tmp_bytes(a.rows(), a.cols_in(), a.cols_out(), a.size()));
        if std::is_x86_feature_detected!("avx2") {
            vmp_prepare::<_, _, _, Reim4BlkAvx, ReimConvAvx, ReimFFTAvx>(module.get_fft_table(), &mut res, &a, tmp);
        } else {
            vmp_prepare::<_, _, _, Reim4BlkRef, ReimConvRef, ReimFFTRef>(module.get_fft_table(), &mut res, &a, tmp);
        }
    }
}

unsafe impl VmpApplyDftToDftTmpBytesImpl<FFT64> for FFT64 {
    fn vmp_apply_dft_to_dft_tmp_bytes_impl(
        _module: &Module<FFT64>,
        _res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        _b_cols_out: usize,
        _b_size: usize,
    ) -> usize {
        vmp_apply_dft_to_dft_tmp_bytes(a_size, b_rows, b_cols_in)
    }
}
