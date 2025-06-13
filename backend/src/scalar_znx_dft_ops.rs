use crate::ffi::svp;
use crate::ffi::vec_znx_dft::vec_znx_dft_t;
use crate::znx_base::{ZnxInfos, ZnxView, ZnxViewMut};
use crate::{
    Backend, FFT64, Module, ScalarZnx, ScalarZnxDft, ScalarZnxDftOwned, ScalarZnxDftToMut, ScalarZnxDftToRef, ScalarZnxToMut,
    ScalarZnxToRef, Scratch, VecZnxDft, VecZnxDftOps, VecZnxDftToMut, VecZnxDftToRef, VecZnxOps,
};

pub trait ScalarZnxDftAlloc<B: Backend> {
    fn new_scalar_znx_dft(&self, cols: usize) -> ScalarZnxDftOwned<B>;
    fn bytes_of_scalar_znx_dft(&self, cols: usize) -> usize;
    fn new_scalar_znx_dft_from_bytes(&self, cols: usize, bytes: Vec<u8>) -> ScalarZnxDftOwned<B>;
}

pub trait ScalarZnxDftOps<BACKEND: Backend> {
    fn svp_prepare<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: ScalarZnxDftToMut<BACKEND>,
        A: ScalarZnxToRef;

    fn svp_apply<R, A, B>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
    where
        R: VecZnxDftToMut<BACKEND>,
        A: ScalarZnxDftToRef<BACKEND>,
        B: VecZnxDftToRef<BACKEND>;

    fn svp_apply_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<BACKEND>,
        A: ScalarZnxDftToRef<BACKEND>;

    fn scalar_znx_idft<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, scratch: &mut Scratch)
    where
        R: ScalarZnxToMut,
        A: ScalarZnxDftToRef<BACKEND>;
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
    fn scalar_znx_idft<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, scratch: &mut Scratch)
    where
        R: ScalarZnxToMut,
        A: ScalarZnxDftToRef<FFT64>,
    {
        let res_mut: &mut ScalarZnx<&mut [u8]> = &mut res.to_mut();
        let a_ref: &ScalarZnxDft<&[u8], FFT64> = &a.to_ref();
        let (mut vec_znx_big, scratch1) = scratch.tmp_vec_znx_big(self, 1, 1);
        self.vec_znx_idft(&mut vec_znx_big, 0, a_ref, a_col, scratch1);
        self.vec_znx_copy(res_mut, res_col, &vec_znx_big.to_vec_znx_small(), 0);
    }

    fn svp_prepare<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: ScalarZnxDftToMut<FFT64>,
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

    fn svp_apply<R, A, B>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
    where
        R: VecZnxDftToMut<FFT64>,
        A: ScalarZnxDftToRef<FFT64>,
        B: VecZnxDftToRef<FFT64>,
    {
        let mut res: VecZnxDft<&mut [u8], FFT64> = res.to_mut();
        let a: ScalarZnxDft<&[u8], FFT64> = a.to_ref();
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

    fn svp_apply_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<FFT64>,
        A: ScalarZnxDftToRef<FFT64>,
    {
        let mut res: VecZnxDft<&mut [u8], FFT64> = res.to_mut();
        let a: ScalarZnxDft<&[u8], FFT64> = a.to_ref();
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
