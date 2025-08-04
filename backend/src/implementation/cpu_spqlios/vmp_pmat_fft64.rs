use crate::{
    hal::{
        api::{TakeSlice, VmpApplyTmpBytes, VmpPrepareTmpBytes, ZnxInfos, ZnxView, ZnxViewMut},
        layouts::{
            MatZnx, MatZnxToRef, Module, Scratch, VecZnxDft, VecZnxDftToMut, VecZnxDftToRef, VmpPMat, VmpPMatBytesOf,
            VmpPMatOwned, VmpPMatToMut, VmpPMatToRef,
        },
        oep::{
            VmpApplyAddImpl, VmpApplyAddTmpBytesImpl, VmpApplyImpl, VmpApplyTmpBytesImpl, VmpPMatAllocBytesImpl,
            VmpPMatAllocImpl, VmpPMatFromBytesImpl, VmpPMatPrepareImpl, VmpPrepareTmpBytesImpl,
        },
    },
    implementation::cpu_spqlios::{
        ffi::{vec_znx_dft::vec_znx_dft_t, vmp},
        module_fft64::FFT64,
    },
};

const VMP_PMAT_FFT64_WORDSIZE: usize = 1;

impl<D: AsRef<[u8]>> ZnxView for VmpPMat<D, FFT64> {
    type Scalar = f64;
}

impl VmpPMatBytesOf for FFT64 {
    fn vmp_pmat_bytes_of(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        VMP_PMAT_FFT64_WORDSIZE * n * rows * cols_in * cols_out * size * size_of::<f64>()
    }
}

unsafe impl VmpPMatAllocBytesImpl<FFT64> for FFT64
where
    FFT64: VmpPMatBytesOf,
{
    fn vmp_pmat_alloc_bytes_impl(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        FFT64::vmp_pmat_bytes_of(n, rows, cols_in, cols_out, size)
    }
}

unsafe impl VmpPMatFromBytesImpl<FFT64> for FFT64 {
    fn vmp_pmat_from_bytes_impl(
        n: usize,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
        bytes: Vec<u8>,
    ) -> VmpPMatOwned<FFT64> {
        VmpPMatOwned::from_bytes(n, rows, cols_in, cols_out, size, bytes)
    }
}

unsafe impl VmpPMatAllocImpl<FFT64> for FFT64 {
    fn vmp_pmat_alloc_impl(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> VmpPMatOwned<FFT64> {
        VmpPMatOwned::alloc(n, rows, cols_in, cols_out, size)
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
            assert_eq!(res.n(), module.n());
            assert_eq!(a.n(), module.n());
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

unsafe impl VmpApplyTmpBytesImpl<FFT64> for FFT64 {
    fn vmp_apply_tmp_bytes_impl(
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

unsafe impl VmpApplyImpl<FFT64> for FFT64 {
    fn vmp_apply_impl<R, A, C>(module: &Module<FFT64>, res: &mut R, a: &A, b: &C, scratch: &mut Scratch<FFT64>)
    where
        R: VecZnxDftToMut<FFT64>,
        A: VecZnxDftToRef<FFT64>,
        C: VmpPMatToRef<FFT64>,
    {
        let mut res: VecZnxDft<&mut [u8], _> = res.to_mut();
        let a: VecZnxDft<&[u8], _> = a.to_ref();
        let b: VmpPMat<&[u8], _> = b.to_ref();

        #[cfg(debug_assertions)]
        {
            assert_eq!(res.n(), module.n());
            assert_eq!(b.n(), module.n());
            assert_eq!(a.n(), module.n());
            assert_eq!(
                res.cols(),
                b.cols_out(),
                "res.cols(): {} != b.cols_out: {}",
                res.cols(),
                b.cols_out()
            );
            assert_eq!(
                a.cols(),
                b.cols_in(),
                "a.cols(): {} != b.cols_in: {}",
                a.cols(),
                b.cols_in()
            );
        }

        let (tmp_bytes, _) = scratch.take_slice(module.vmp_apply_tmp_bytes(
            res.size(),
            a.size(),
            b.rows(),
            b.cols_in(),
            b.cols_out(),
            b.size(),
        ));
        unsafe {
            vmp::vmp_apply_dft_to_dft(
                module.ptr(),
                res.as_mut_ptr() as *mut vec_znx_dft_t,
                (res.size() * res.cols()) as u64,
                a.as_ptr() as *const vec_znx_dft_t,
                (a.size() * a.cols()) as u64,
                b.as_ptr() as *const vmp::vmp_pmat_t,
                (b.rows() * b.cols_in()) as u64,
                (b.size() * b.cols_out()) as u64,
                tmp_bytes.as_mut_ptr(),
            )
        }
    }
}

unsafe impl VmpApplyAddTmpBytesImpl<FFT64> for FFT64 {
    fn vmp_apply_add_tmp_bytes_impl(
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

unsafe impl VmpApplyAddImpl<FFT64> for FFT64 {
    fn vmp_apply_add_impl<R, A, C>(module: &Module<FFT64>, res: &mut R, a: &A, b: &C, scale: usize, scratch: &mut Scratch<FFT64>)
    where
        R: VecZnxDftToMut<FFT64>,
        A: VecZnxDftToRef<FFT64>,
        C: VmpPMatToRef<FFT64>,
    {
        let mut res: VecZnxDft<&mut [u8], _> = res.to_mut();
        let a: VecZnxDft<&[u8], _> = a.to_ref();
        let b: VmpPMat<&[u8], _> = b.to_ref();

        #[cfg(debug_assertions)]
        {
            use crate::hal::api::ZnxInfos;

            assert_eq!(res.n(), module.n());
            assert_eq!(b.n(), module.n());
            assert_eq!(a.n(), module.n());
            assert_eq!(
                res.cols(),
                b.cols_out(),
                "res.cols(): {} != b.cols_out: {}",
                res.cols(),
                b.cols_out()
            );
            assert_eq!(
                a.cols(),
                b.cols_in(),
                "a.cols(): {} != b.cols_in: {}",
                a.cols(),
                b.cols_in()
            );
        }

        let (tmp_bytes, _) = scratch.take_slice(module.vmp_apply_tmp_bytes(
            res.size(),
            a.size(),
            b.rows(),
            b.cols_in(),
            b.cols_out(),
            b.size(),
        ));
        unsafe {
            vmp::vmp_apply_dft_to_dft_add(
                module.ptr(),
                res.as_mut_ptr() as *mut vec_znx_dft_t,
                (res.size() * res.cols()) as u64,
                a.as_ptr() as *const vec_znx_dft_t,
                (a.size() * a.cols()) as u64,
                b.as_ptr() as *const vmp::vmp_pmat_t,
                (b.rows() * b.cols_in()) as u64,
                (b.size() * b.cols_out()) as u64,
                (scale * b.cols_out()) as u64,
                tmp_bytes.as_mut_ptr(),
            )
        }
    }
}
