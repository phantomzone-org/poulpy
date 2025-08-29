use poulpy_hal::{
    api::{TakeSlice, VmpPrepareTmpBytes},
    layouts::{
        Backend, MatZnx, MatZnxToRef, Module, Scratch, VmpPMat, VmpPMatOwned, VmpPMatToMut, ZnxInfos, ZnxView, ZnxViewMut,
    },
    oep::{
        VmpApplyDftToDftImpl, VmpApplyDftToDftTmpBytesImpl, VmpPMatAllocBytesImpl, VmpPMatAllocImpl, VmpPMatPrepareImpl,
        VmpPrepareTmpBytesImpl,
    },
    reference::vmp::fft64_vmp_apply_dft_to_dft_avx,
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
    fn vmp_apply_dft_to_dft_impl<R, A, C>(
        module: &poulpy_hal::layouts::Module<Self>,
        res: &mut R,
        a: &A,
        b: &C,
        scratch: &mut poulpy_hal::layouts::Scratch<Self>,
    ) where
        R: poulpy_hal::layouts::VecZnxDftToMut<Self>,
        A: poulpy_hal::layouts::VecZnxDftToRef<Self>,
        C: poulpy_hal::layouts::VmpPMatToRef<Self>,
    {
        let mut res: poulpy_hal::layouts::VecZnxDft<&mut [u8], FFT64> = res.to_mut();
        let a: poulpy_hal::layouts::VecZnxDft<&[u8], FFT64> = a.to_ref();
        let b: poulpy_hal::layouts::VmpPMat<&[u8], FFT64> = b.to_ref();

        let (tmp_bytes, _) = scratch.take_slice(
            FFT64::vmp_apply_dft_to_dft_tmp_bytes_impl(
                module,
                res.size(),
                a.size(),
                b.rows(),
                b.cols_in(),
                b.cols_out(),
                b.size(),
            ) / size_of::<f64>(),
        );
        unsafe {
            fft64_vmp_apply_dft_to_dft_avx(
                module.n(),
                res.raw_mut(),
                a.raw(),
                b.raw(),
                b.rows() * b.cols_in(),
                b.size() * b.cols_out(),
                tmp_bytes,
            );
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

unsafe impl VmpPMatPrepareImpl<FFT64> for FFT64 {
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
