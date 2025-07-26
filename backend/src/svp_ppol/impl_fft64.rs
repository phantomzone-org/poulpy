use crate::{
    FFT64, Module, ScalarZnxToRef, SvpApply, SvpApplyInplace, SvpPPol, SvpPPolAlloc, SvpPPolAllocBytes, SvpPPolBytesOf,
    SvpPPolOwned, SvpPPolToMut, SvpPPolToRef, SvpPPolyFromBytes, SvpPrepare, VecZnxDft, VecZnxDftToMut, VecZnxDftToRef, ZnxInfos,
    ZnxSliceSize, ZnxView, ZnxViewMut,
    ffi::{svp, vec_znx_dft::vec_znx_dft_t},
};

const SVP_PPOL_FFT64_WORD_SIZE: usize = 1;

impl<D: AsRef<[u8]>> SvpPPolBytesOf for SvpPPol<D, FFT64> {
    fn bytes_of(n: usize, cols: usize) -> usize {
        SVP_PPOL_FFT64_WORD_SIZE * n * cols * size_of::<f64>()
    }
}

impl<D> ZnxSliceSize for SvpPPol<D, FFT64> {
    fn sl(&self) -> usize {
        SVP_PPOL_FFT64_WORD_SIZE * self.n()
    }
}

impl<D: AsRef<[u8]>> ZnxView for SvpPPol<D, FFT64> {
    type Scalar = f64;
}

impl SvpPPolyFromBytes<FFT64> for Module<FFT64> {
    fn svp_ppol_from_bytes(&self, cols: usize, bytes: Vec<u8>) -> SvpPPolOwned<FFT64> {
        SvpPPolOwned::from_bytes(self.n(), cols, bytes)
    }
}

impl SvpPPolAlloc<FFT64> for Module<FFT64> {
    fn svp_ppol_alloc(&self, cols: usize) -> SvpPPolOwned<FFT64> {
        SvpPPolOwned::alloc(self.n(), cols)
    }
}

impl SvpPPolAllocBytes for Module<FFT64> {
    fn svp_ppol_alloc_bytes(&self, cols: usize) -> usize {
        SvpPPol::<Vec<u8>, FFT64>::bytes_of(self.n(), cols)
    }
}

impl SvpPrepare<FFT64> for Module<FFT64> {
    fn svp_prepare<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: SvpPPolToMut<FFT64>,
        A: ScalarZnxToRef,
    {
        unsafe {
            svp::svp_prepare(
                self.ptr,
                res.to_mut().at_mut_ptr(res_col, 0) as *mut svp::svp_ppol_t,
                a.to_ref().at_ptr(a_col, 0),
            )
        }
    }
}

impl SvpApply<FFT64> for Module<FFT64> {
    fn svp_apply<R, A, B>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
    where
        R: VecZnxDftToMut<FFT64>,
        A: SvpPPolToRef<FFT64>,
        B: VecZnxDftToRef<FFT64>,
    {
        let mut res: VecZnxDft<&mut [u8], FFT64> = res.to_mut();
        let a: SvpPPol<&[u8], FFT64> = a.to_ref();
        let b: VecZnxDft<&[u8], FFT64> = b.to_ref();
        unsafe {
            svp::svp_apply_dft_to_dft(
                self.ptr,
                res.at_mut_ptr(res_col, 0) as *mut vec_znx_dft_t,
                res.size() as u64,
                res.cols() as u64,
                a.at_ptr(a_col, 0) as *const svp::svp_ppol_t,
                b.at_ptr(b_col, 0) as *const vec_znx_dft_t,
                b.size() as u64,
                b.cols() as u64,
            )
        }
    }
}

impl SvpApplyInplace<FFT64> for Module<FFT64> {
    fn svp_apply_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<FFT64>,
        A: SvpPPolToRef<FFT64>,
    {
        let mut res: VecZnxDft<&mut [u8], FFT64> = res.to_mut();
        let a: SvpPPol<&[u8], FFT64> = a.to_ref();
        unsafe {
            svp::svp_apply_dft_to_dft(
                self.ptr,
                res.at_mut_ptr(res_col, 0) as *mut vec_znx_dft_t,
                res.size() as u64,
                res.cols() as u64,
                a.at_ptr(a_col, 0) as *const svp::svp_ppol_t,
                res.at_ptr(res_col, 0) as *const vec_znx_dft_t,
                res.size() as u64,
                res.cols() as u64,
            )
        }
    }
}
