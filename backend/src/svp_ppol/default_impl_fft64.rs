use crate::{
    FFT64, Module, ScalarZnxToRef, SvpApplyImpl, SvpApplyInplaceImpl, SvpPPol, SvpPPolAllocBytesImpl, SvpPPolAllocImpl,
    SvpPPolBytesOf, SvpPPolFromBytesImpl, SvpPPolOwned, SvpPPolToMut, SvpPPolToRef, SvpPrepareImpl, VecZnxDft, VecZnxDftToMut,
    VecZnxDftToRef, ZnxInfos, ZnxSliceSize, ZnxView, ZnxViewMut,
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

unsafe impl SvpPPolFromBytesImpl<Self> for FFT64 {
    fn svp_ppol_from_bytes_impl(module: &Module<Self>, cols: usize, bytes: Vec<u8>) -> SvpPPolOwned<Self> {
        SvpPPolOwned::from_bytes(module.n(), cols, bytes)
    }
}

unsafe impl SvpPPolAllocImpl<Self> for FFT64 {
    fn svp_ppol_alloc_impl(module: &Module<Self>, cols: usize) -> SvpPPolOwned<Self> {
        SvpPPolOwned::alloc(module.n(), cols)
    }
}

unsafe impl SvpPPolAllocBytesImpl<Self> for FFT64 {
    fn svp_ppol_alloc_bytes_impl(module: &Module<Self>, cols: usize) -> usize {
        SvpPPol::<Vec<u8>, Self>::bytes_of(module.n(), cols)
    }
}

unsafe impl SvpPrepareImpl<Self> for FFT64 {
    fn svp_prepare_impl<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: SvpPPolToMut<Self>,
        A: ScalarZnxToRef,
    {
        unsafe {
            svp::svp_prepare(
                module.ptr(),
                res.to_mut().at_mut_ptr(res_col, 0) as *mut svp::svp_ppol_t,
                a.to_ref().at_ptr(a_col, 0),
            )
        }
    }
}

unsafe impl SvpApplyImpl<Self> for FFT64 {
    fn svp_apply_impl<R, A, B>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
    where
        R: VecZnxDftToMut<Self>,
        A: SvpPPolToRef<Self>,
        B: VecZnxDftToRef<Self>,
    {
        let mut res: VecZnxDft<&mut [u8], Self> = res.to_mut();
        let a: SvpPPol<&[u8], Self> = a.to_ref();
        let b: VecZnxDft<&[u8], Self> = b.to_ref();
        unsafe {
            svp::svp_apply_dft_to_dft(
                module.ptr(),
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

unsafe impl SvpApplyInplaceImpl for FFT64 {
    fn svp_apply_inplace_impl<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<Self>,
        A: SvpPPolToRef<Self>,
    {
        let mut res: VecZnxDft<&mut [u8], Self> = res.to_mut();
        let a: SvpPPol<&[u8], Self> = a.to_ref();
        unsafe {
            svp::svp_apply_dft_to_dft(
                module.ptr(),
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
