use poulpy_hal::{
    api::{TakeSlice, VmpPrepareTmpBytes},
    layouts::{
        Backend, MatZnx, MatZnxToRef, Module, Scratch, VecZnxDft, VecZnxDftToMut, VecZnxDftToRef, VmpPMat, VmpPMatOwned,
        VmpPMatToMut, VmpPMatToRef, ZnxInfos, ZnxView, ZnxViewMut,
    },
    oep::{
        VmpApplyDftToDftImpl, VmpApplyDftToDftTmpBytesImpl, VmpPMatAllocBytesImpl, VmpPMatAllocImpl, VmpPrepareImpl,
        VmpPrepareTmpBytesImpl,
    },
    reference::vmp::fft64::{vmp_apply_dft_to_dft_avx, vmp_apply_dft_to_dft_ref},
};

use crate::cpu_ref::{FFT64, ffi::vmp};

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
            unsafe {
                vmp_apply_dft_to_dft_avx(&mut res, &a, &pmat, tmp);
            }
        } else {
            vmp_apply_dft_to_dft_ref(&mut res, &a, &pmat, tmp);
        }
    }
}

unsafe impl VmpPrepareTmpBytesImpl<FFT64> for FFT64 {
    fn vmp_prepare_tmp_bytes_impl(module: &Module<FFT64>, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        unsafe {
            vmp::vmp_prepare_tmp_bytes(
                module.ptr(),
                (rows * cols_in) as u64,
                (cols_out * size) as u64,
            ) as usize
        }
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

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), res.n());
            assert_eq!(
                res.cols_in(),
                a.cols_in(),
                "res.cols_in: {} != a.cols_in: {}",
                res.cols_in(),
                a.cols_in()
            );
            assert_eq!(
                res.rows(),
                a.rows(),
                "res.rows: {} != a.rows: {}",
                res.rows(),
                a.rows()
            );
            assert_eq!(
                res.cols_out(),
                a.cols_out(),
                "res.cols_out: {} != a.cols_out: {}",
                res.cols_out(),
                a.cols_out()
            );
            assert_eq!(
                res.size(),
                a.size(),
                "res.size: {} != a.size: {}",
                res.size(),
                a.size()
            );
        }

        let (tmp_bytes, _) = scratch.take_slice(module.vmp_prepare_tmp_bytes(a.rows(), a.cols_in(), a.cols_out(), a.size()));

        unsafe {
            vmp::vmp_prepare_contiguous(
                module.ptr(),
                res.as_mut_ptr() as *mut vmp::vmp_pmat_t,
                a.as_ptr(),
                (a.rows() * a.cols_in()) as u64,
                (a.size() * a.cols_out()) as u64,
                tmp_bytes.as_mut_ptr(),
            );
        }
    }
}

unsafe impl VmpApplyDftToDftTmpBytesImpl<FFT64> for FFT64 {
    fn vmp_apply_dft_to_dft_tmp_bytes_impl(
        module: &Module<FFT64>,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize {
        unsafe {
            vmp::vmp_apply_dft_to_dft_tmp_bytes(
                module.ptr(),
                (res_size * b_cols_out) as u64,
                (a_size * b_cols_in) as u64,
                (b_rows * b_cols_in) as u64,
                (b_size * b_cols_out) as u64,
            ) as usize
        }
    }
}
