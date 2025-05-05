use crate::ffi::svp;
use crate::ffi::vec_znx_dft::vec_znx_dft_t;
use crate::znx_base::{ZnxInfos, ZnxView, ZnxViewMut};
use crate::{
    Backend, FFT64, Module, ScalarToRef, ScalarZnxDft, ScalarZnxDftOwned, ScalarZnxDftToMut, ScalarZnxDftToRef, VecZnx,
    VecZnxDft, VecZnxDftToMut, VecZnxToRef, ZnxSliceSize,
};

pub trait ScalarZnxDftAlloc<B: Backend> {
    fn new_scalar_znx_dft(&self, cols: usize) -> ScalarZnxDftOwned<B>;
    fn bytes_of_scalar_znx_dft(&self, cols: usize) -> usize;
    fn new_scalar_znx_dft_from_bytes(&self, cols: usize, bytes: Vec<u8>) -> ScalarZnxDftOwned<B>;
    // fn new_scalar_znx_dft_from_bytes_borrow(&self, cols: usize, bytes: &mut [u8]) -> ScalarZnxDft<B>;
}

pub trait ScalarZnxDftOps<BACKEND: Backend> {
    fn svp_prepare<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: ScalarZnxDftToMut<BACKEND>,
        A: ScalarToRef;
    fn svp_apply<R, A, B>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
    where
        R: VecZnxDftToMut<BACKEND>,
        A: ScalarZnxDftToRef<BACKEND>,
        B: VecZnxToRef;
}

impl<B: Backend> ScalarZnxDftAlloc<B> for Module<B> {
    fn new_scalar_znx_dft(&self, cols: usize) -> ScalarZnxDftOwned<B> {
        ScalarZnxDftOwned::new(self, cols)
    }

    fn bytes_of_scalar_znx_dft(&self, cols: usize) -> usize {
        ScalarZnxDftOwned::bytes_of(self, cols)
    }

    fn new_scalar_znx_dft_from_bytes(&self, cols: usize, bytes: Vec<u8>) -> ScalarZnxDftOwned<B> {
        ScalarZnxDftOwned::new_from_bytes(self, cols, bytes)
    }
}

impl ScalarZnxDftOps<FFT64> for Module<FFT64> {
    fn svp_prepare<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: ScalarZnxDftToMut<FFT64>,
        A: ScalarToRef,
    {
        unsafe {
            svp::svp_prepare(
                self.ptr,
                res.to_mut().at_mut_ptr(res_col, 0) as *mut svp::svp_ppol_t,
                a.to_ref().at_ptr(a_col, 0),
            )
        }
    }

    fn svp_apply<R, A, B>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
    where
        R: VecZnxDftToMut<FFT64>,
        A: ScalarZnxDftToRef<FFT64>,
        B: VecZnxToRef,
    {
        let mut res: VecZnxDft<&mut [u8], FFT64> = res.to_mut();
        let a: ScalarZnxDft<&[u8], FFT64> = a.to_ref();
        let b: VecZnx<&[u8]> = b.to_ref();
        unsafe {
            svp::svp_apply_dft(
                self.ptr,
                res.at_mut_ptr(res_col, 0) as *mut vec_znx_dft_t,
                res.size() as u64,
                a.at_ptr(a_col, 0) as *const svp::svp_ppol_t,
                b.at_ptr(b_col, 0),
                b.size() as u64,
                b.sl() as u64,
            )
        }
    }
}
