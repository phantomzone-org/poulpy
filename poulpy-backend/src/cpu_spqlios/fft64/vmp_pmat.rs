use poulpy_hal::{
    api::{TakeSlice, VmpApplyDftToDftTmpBytes, VmpPrepareTmpBytes},
    layouts::{
        Backend, MatZnx, MatZnxToRef, Module, Scratch, VecZnxDft, VecZnxDftToMut, VecZnxDftToRef, VmpPMat, VmpPMatOwned,
        VmpPMatToMut, VmpPMatToRef, ZnxInfos, ZnxView, ZnxViewMut,
    },
    oep::{
        VmpApplyDftToDftAddImpl, VmpApplyDftToDftAddTmpBytesImpl, VmpApplyDftToDftImpl, VmpApplyDftToDftTmpBytesImpl,
        VmpPMatAllocBytesImpl, VmpPMatAllocImpl, VmpPMatFromBytesImpl, VmpPrepareImpl, VmpPrepareTmpBytesImpl,
    },
};

use crate::cpu_spqlios::{
    FFT64Spqlios,
    ffi::{vec_znx_dft::vec_znx_dft_t, vmp},
};

unsafe impl VmpPMatAllocBytesImpl<Self> for FFT64Spqlios {
    fn vmp_pmat_alloc_bytes_impl(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        Self::layout_prep_word_count() * n * rows * cols_in * cols_out * size * size_of::<f64>()
    }
}

unsafe impl VmpPMatFromBytesImpl<Self> for FFT64Spqlios {
    fn vmp_pmat_from_bytes_impl(
        n: usize,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
        bytes: Vec<u8>,
    ) -> VmpPMatOwned<Self> {
        VmpPMatOwned::from_bytes(n, rows, cols_in, cols_out, size, bytes)
    }
}

unsafe impl VmpPMatAllocImpl<Self> for FFT64Spqlios {
    fn vmp_pmat_alloc_impl(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> VmpPMatOwned<Self> {
        VmpPMatOwned::alloc(n, rows, cols_in, cols_out, size)
    }
}

unsafe impl VmpPrepareTmpBytesImpl<Self> for FFT64Spqlios {
    fn vmp_prepare_tmp_bytes_impl(module: &Module<Self>, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        unsafe {
            vmp::vmp_prepare_tmp_bytes(
                module.ptr(),
                (rows * cols_in) as u64,
                (cols_out * size) as u64,
            ) as usize
        }
    }
}

unsafe impl VmpPrepareImpl<Self> for FFT64Spqlios {
    fn vmp_prepare_impl<R, A>(module: &Module<Self>, res: &mut R, a: &A, scratch: &mut Scratch<Self>)
    where
        R: VmpPMatToMut<Self>,
        A: MatZnxToRef,
    {
        let mut res: VmpPMat<&mut [u8], Self> = res.to_mut();
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

unsafe impl VmpApplyDftToDftTmpBytesImpl<Self> for FFT64Spqlios {
    fn vmp_apply_dft_to_dft_tmp_bytes_impl(
        module: &Module<Self>,
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

unsafe impl VmpApplyDftToDftImpl<Self> for FFT64Spqlios {
    fn vmp_apply_dft_to_dft_impl<R, A, C>(module: &Module<Self>, res: &mut R, a: &A, b: &C, scratch: &mut Scratch<Self>)
    where
        R: VecZnxDftToMut<Self>,
        A: VecZnxDftToRef<Self>,
        C: VmpPMatToRef<Self>,
    {
        let mut res: VecZnxDft<&mut [u8], _> = res.to_mut();
        let a: VecZnxDft<&[u8], _> = a.to_ref();
        let b: VmpPMat<&[u8], _> = b.to_ref();

        #[cfg(debug_assertions)]
        {
            assert_eq!(b.n(), res.n());
            assert_eq!(a.n(), res.n());
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

        let (tmp_bytes, _) = scratch.take_slice(module.vmp_apply_dft_to_dft_tmp_bytes(
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

unsafe impl VmpApplyDftToDftAddTmpBytesImpl<Self> for FFT64Spqlios {
    fn vmp_apply_dft_to_dft_add_tmp_bytes_impl(
        module: &Module<Self>,
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

unsafe impl VmpApplyDftToDftAddImpl<Self> for FFT64Spqlios {
    fn vmp_apply_dft_to_dft_add_impl<R, A, C>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        b: &C,
        scale: usize,
        scratch: &mut Scratch<Self>,
    ) where
        R: VecZnxDftToMut<Self>,
        A: VecZnxDftToRef<Self>,
        C: VmpPMatToRef<Self>,
    {
        let mut res: VecZnxDft<&mut [u8], _> = res.to_mut();
        let a: VecZnxDft<&[u8], _> = a.to_ref();
        let b: VmpPMat<&[u8], _> = b.to_ref();

        #[cfg(debug_assertions)]
        {
            assert_eq!(b.n(), res.n());
            assert_eq!(a.n(), res.n());
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

        let (tmp_bytes, _) = scratch.take_slice(module.vmp_apply_dft_to_dft_tmp_bytes(
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
