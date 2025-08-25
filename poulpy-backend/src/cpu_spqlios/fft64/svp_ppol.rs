use poulpy_hal::{
    layouts::{
        Backend, Module, ScalarZnxToRef, SvpPPol, SvpPPolOwned, SvpPPolToMut, SvpPPolToRef, VecZnxDft, VecZnxDftToMut,
        VecZnxDftToRef, ZnxInfos, ZnxView, ZnxViewMut,
    },
    oep::{SvpApplyImpl, SvpApplyInplaceImpl, SvpPPolAllocBytesImpl, SvpPPolAllocImpl, SvpPPolFromBytesImpl, SvpPrepareImpl},
};

use crate::cpu_spqlios::{
    FFT64,
    ffi::{svp, vec_znx_dft::vec_znx_dft_t},
};

unsafe impl SvpPPolFromBytesImpl<Self> for FFT64 {
    fn svp_ppol_from_bytes_impl(n: usize, cols: usize, bytes: Vec<u8>) -> SvpPPolOwned<Self> {
        SvpPPolOwned::from_bytes(n, cols, bytes)
    }
}

unsafe impl SvpPPolAllocImpl<Self> for FFT64 {
    fn svp_ppol_alloc_impl(n: usize, cols: usize) -> SvpPPolOwned<Self> {
        SvpPPolOwned::alloc(n, cols)
    }
}

unsafe impl SvpPPolAllocBytesImpl<Self> for FFT64 {
    fn svp_ppol_alloc_bytes_impl(n: usize, cols: usize) -> usize {
        FFT64::layout_prep_word_count() * n * cols * size_of::<f64>()
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
