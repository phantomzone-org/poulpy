use crate::{
    Backend, FFT64, MatZnx, MatZnxToRef, Module, Scratch, VecZnxDft, VecZnxDftToMut, VecZnxDftToRef, VmpApply, VmpApplyAdd,
    VmpApplyAddTmpBytes, VmpApplyTmpBytes, VmpPMat, VmpPMatAlloc, VmpPMatAllocBytes, VmpPMatBytesOf, VmpPMatFromBytes,
    VmpPMatOwned, VmpPMatPrepare, VmpPMatToMut, VmpPMatToRef, VmpPrepareTmpBytes, ZnxInfos, ZnxView, ZnxViewMut,
    ffi::{vec_znx_dft::vec_znx_dft_t, vmp},
};

const VMP_PMAT_FFT64_WORDSIZE: usize = 1;

impl<D: AsRef<[u8]>> ZnxView for VmpPMat<D, FFT64> {
    type Scalar = f64;
}

impl<D: AsRef<[u8]>, B: Backend> VmpPMatBytesOf for VmpPMat<D, B> {
    fn bytes_of(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        VMP_PMAT_FFT64_WORDSIZE * n * rows * cols_in * cols_out * size * size_of::<f64>()
    }
}

impl VmpPMatAllocBytes for Module<FFT64> {
    fn vmp_pmat_alloc_bytes(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        VmpPMat::<Vec<u8>, FFT64>::bytes_of(self.n(), rows, cols_in, cols_out, size)
    }
}

impl VmpPMatFromBytes<FFT64> for Module<FFT64> {
    fn vmp_pmat_from_bytes(
        &self,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
        bytes: Vec<u8>,
    ) -> VmpPMatOwned<FFT64> {
        VmpPMatOwned::from_bytes(self.n(), rows, cols_in, cols_out, size, bytes)
    }
}

impl VmpPMatAlloc<FFT64> for Module<FFT64> {
    fn vmp_pmat_alloc(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> VmpPMatOwned<FFT64> {
        VmpPMatOwned::alloc(self.n(), rows, cols_in, cols_out, size)
    }
}

impl VmpPrepareTmpBytes for Module<FFT64> {
    fn vmp_prepare_scratch_bytes(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        unsafe { vmp::vmp_prepare_tmp_bytes(self.ptr, (rows * cols_in) as u64, (cols_out * size) as u64) as usize }
    }
}

impl VmpPMatPrepare<FFT64> for Module<FFT64> {
    fn vmp_prepare<R, A>(&self, res: &mut R, a: &A, scratch: &mut Scratch)
    where
        R: VmpPMatToMut<FFT64>,
        A: MatZnxToRef,
    {
        let mut res: VmpPMat<&mut [u8], FFT64> = res.to_mut();
        let a: MatZnx<&[u8]> = a.to_ref();

        #[cfg(debug_assertions)]
        {
            assert_eq!(res.n(), self.n());
            assert_eq!(a.n(), self.n());
            assert_eq!(res.cols_in(), a.cols_in());
            assert_eq!(res.rows(), a.rows());
            assert_eq!(res.cols_out(), a.cols_out());
            assert_eq!(res.size(), a.size());
        }

        let (tmp_bytes, _) = scratch.tmp_slice(self.vmp_prepare_scratch_bytes(a.rows(), a.cols_in(), a.cols_out(), a.size()));

        unsafe {
            vmp::vmp_prepare_contiguous(
                self.ptr,
                res.as_mut_ptr() as *mut vmp::vmp_pmat_t,
                a.as_ptr(),
                (a.rows() * a.cols_in()) as u64,
                (a.size() * a.cols_out()) as u64,
                tmp_bytes.as_mut_ptr(),
            );
        }
    }
}

impl VmpApplyTmpBytes for Module<FFT64> {
    fn vmp_apply_tmp_bytes(
        &self,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize {
        unsafe {
            vmp::vmp_apply_dft_to_dft_tmp_bytes(
                self.ptr,
                (res_size * b_cols_out) as u64,
                (a_size * b_cols_in) as u64,
                (b_rows * b_cols_in) as u64,
                (b_size * b_cols_out) as u64,
            ) as usize
        }
    }
}

impl VmpApply<FFT64> for Module<FFT64> {
    fn vmp_apply<R, A, C>(&self, res: &mut R, a: &A, b: &C, scratch: &mut Scratch)
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
            assert_eq!(res.n(), self.n());
            assert_eq!(b.n(), self.n());
            assert_eq!(a.n(), self.n());
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

        let (tmp_bytes, _) = scratch.tmp_slice(self.vmp_apply_tmp_bytes(
            res.size(),
            a.size(),
            b.rows(),
            b.cols_in(),
            b.cols_out(),
            b.size(),
        ));
        unsafe {
            vmp::vmp_apply_dft_to_dft(
                self.ptr,
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

impl VmpApplyAddTmpBytes for Module<FFT64> {
    fn vmp_apply_add_tmp_bytes(
        &self,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize {
        unsafe {
            vmp::vmp_apply_dft_to_dft_tmp_bytes(
                self.ptr,
                (res_size * b_cols_out) as u64,
                (a_size * b_cols_in) as u64,
                (b_rows * b_cols_in) as u64,
                (b_size * b_cols_out) as u64,
            ) as usize
        }
    }
}

impl VmpApplyAdd<FFT64> for Module<FFT64> {
    fn vmp_apply_add<R, A, C>(&self, res: &mut R, a: &A, b: &C, scale: usize, scratch: &mut Scratch)
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
            assert_eq!(res.n(), self.n());
            assert_eq!(b.n(), self.n());
            assert_eq!(a.n(), self.n());
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

        let (tmp_bytes, _) = scratch.tmp_slice(self.vmp_apply_tmp_bytes(
            res.size(),
            a.size(),
            b.rows(),
            b.cols_in(),
            b.cols_out(),
            b.size(),
        ));
        unsafe {
            vmp::vmp_apply_dft_to_dft_add(
                self.ptr,
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
