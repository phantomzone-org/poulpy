use poulpy_hal::{
    api::{TakeSlice, VmpPrepareTmpBytes},
    layouts::{
        MatZnx, MatZnxToRef, Module, Scratch, VecZnxDft, VecZnxDftToMut, VecZnxDftToRef, VmpPMat, VmpPMatToMut, VmpPMatToRef,
        ZnxInfos,
    },
    oep::{
        VmpApplyDftToDftAddImpl, VmpApplyDftToDftAddTmpBytesImpl, VmpApplyDftToDftImpl, VmpApplyDftToDftTmpBytesImpl,
        VmpPrepareImpl, VmpPrepareTmpBytesImpl, VmpZeroImpl,
    },
    reference::fft64::vmp::{
        vmp_apply_dft_to_dft, vmp_apply_dft_to_dft_add, vmp_apply_dft_to_dft_tmp_bytes, vmp_prepare, vmp_prepare_tmp_bytes,
        vmp_zero,
    },
};

use crate::{FFT64Avx, fft64::module::FFT64ModuleHandle};

unsafe impl VmpApplyDftToDftImpl<Self> for FFT64Avx
where
    Scratch<Self>: TakeSlice,
    FFT64Avx: VmpApplyDftToDftTmpBytesImpl<Self>,
{
    fn vmp_apply_dft_to_dft_impl<R, A, C>(module: &Module<Self>, res: &mut R, a: &A, pmat: &C, scratch: &mut Scratch<Self>)
    where
        R: VecZnxDftToMut<Self>,
        A: VecZnxDftToRef<Self>,
        C: VmpPMatToRef<Self>,
    {
        let mut res: VecZnxDft<&mut [u8], Self> = res.to_mut();
        let a: VecZnxDft<&[u8], Self> = a.to_ref();
        let pmat: VmpPMat<&[u8], Self> = pmat.to_ref();

        let (tmp, _) = scratch.take_slice(
            Self::vmp_apply_dft_to_dft_tmp_bytes_impl(
                module,
                res.size(),
                a.size(),
                pmat.rows(),
                pmat.cols_in(),
                pmat.cols_out(),
                pmat.size(),
            ) / size_of::<f64>(),
        );
        vmp_apply_dft_to_dft(&mut res, &a, &pmat, tmp);
    }
}

unsafe impl VmpApplyDftToDftAddImpl<Self> for FFT64Avx
where
    Scratch<Self>: TakeSlice,
    FFT64Avx: VmpApplyDftToDftTmpBytesImpl<Self>,
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
        let mut res: VecZnxDft<&mut [u8], Self> = res.to_mut();
        let a: VecZnxDft<&[u8], Self> = a.to_ref();
        let pmat: VmpPMat<&[u8], Self> = pmat.to_ref();

        let (tmp, _) = scratch.take_slice(
            Self::vmp_apply_dft_to_dft_tmp_bytes_impl(
                module,
                res.size(),
                a.size(),
                pmat.rows(),
                pmat.cols_in(),
                pmat.cols_out(),
                pmat.size(),
            ) / size_of::<f64>(),
        );
        vmp_apply_dft_to_dft_add(&mut res, &a, &pmat, limb_offset * pmat.cols_out(), tmp);
    }
}

unsafe impl VmpPrepareTmpBytesImpl<Self> for FFT64Avx {
    fn vmp_prepare_tmp_bytes_impl(module: &Module<Self>, _rows: usize, _cols_in: usize, _cols_out: usize, _size: usize) -> usize {
        vmp_prepare_tmp_bytes(module.n())
    }
}

unsafe impl VmpPrepareImpl<Self> for FFT64Avx {
    fn vmp_prepare_impl<R, A>(module: &Module<Self>, res: &mut R, a: &A, scratch: &mut Scratch<Self>)
    where
        R: VmpPMatToMut<Self>,
        A: MatZnxToRef,
    {
        {}
        let mut res: VmpPMat<&mut [u8], Self> = res.to_mut();
        let a: MatZnx<&[u8]> = a.to_ref();
        let (tmp, _) =
            scratch.take_slice(module.vmp_prepare_tmp_bytes(a.rows(), a.cols_in(), a.cols_out(), a.size()) / size_of::<f64>());
        vmp_prepare(module.get_fft_table(), &mut res, &a, tmp);
    }
}

unsafe impl VmpApplyDftToDftTmpBytesImpl<Self> for FFT64Avx {
    fn vmp_apply_dft_to_dft_tmp_bytes_impl(
        _module: &Module<Self>,
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

unsafe impl VmpApplyDftToDftAddTmpBytesImpl<Self> for FFT64Avx {
    fn vmp_apply_dft_to_dft_add_tmp_bytes_impl(
        _module: &Module<Self>,
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

unsafe impl VmpZeroImpl<Self> for FFT64Avx {
    fn vmp_zero_impl<R>(_module: &Module<Self>, res: &mut R)
    where
        R: VmpPMatToMut<Self>,
    {
        vmp_zero(res);
    }
}
