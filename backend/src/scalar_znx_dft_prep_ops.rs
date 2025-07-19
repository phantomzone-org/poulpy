use crate::ffi::svp;
use crate::ffi::vec_znx_dft::vec_znx_dft_t;
use crate::znx_base::{ZnxInfos, ZnxView, ZnxViewMut};
use crate::{
    Backend, FFT64, Module, NTT120, ScalarZnxDftPrep, ScalarZnxDftPrepBytesOf, ScalarZnxDftPrepOwned, ScalarZnxDftPrepToMut,
    ScalarZnxDftPrepToRef, ScalarZnxToRef, VecZnxDft, VecZnxDftToMut, VecZnxDftToRef,
};

pub trait ScalarZnxDftPrepAlloc<B: Backend> {
    fn new_scalar_znx_dft_prep(&self, cols: usize) -> ScalarZnxDftPrepOwned<B>;
    fn bytes_of_scalar_znx_dft_prep(&self, cols: usize) -> usize;
    fn new_scalar_znx_dft_prep_from_bytes(&self, cols: usize, bytes: Vec<u8>) -> ScalarZnxDftPrepOwned<B>;
}

pub trait ScalarZnxDftPrepOps<BACKEND: Backend> {
    fn svp_prepare<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: ScalarZnxDftPrepToMut<BACKEND>,
        A: ScalarZnxToRef;

    fn svp_apply<R, A, B>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
    where
        R: VecZnxDftToMut<BACKEND>,
        A: ScalarZnxDftPrepToRef<BACKEND>,
        B: VecZnxDftToRef<BACKEND>;

    fn svp_apply_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<BACKEND>,
        A: ScalarZnxDftPrepToRef<BACKEND>;
}

impl<B: Backend> ScalarZnxDftPrepAlloc<B> for Module<B>
where
    ScalarZnxDftPrep<Vec<u8>, B>: ScalarZnxDftPrepBytesOf<B>,
{
    fn new_scalar_znx_dft_prep(&self, cols: usize) -> ScalarZnxDftPrepOwned<B> {
        ScalarZnxDftPrepOwned::new(self.n(), cols)
    }

    fn bytes_of_scalar_znx_dft_prep(&self, cols: usize) -> usize {
        ScalarZnxDftPrepOwned::bytes_of(self.n(), cols)
    }

    fn new_scalar_znx_dft_prep_from_bytes(&self, cols: usize, bytes: Vec<u8>) -> ScalarZnxDftPrepOwned<B> {
        ScalarZnxDftPrepOwned::new_from_bytes(self.n(), cols, bytes)
    }
}

impl ScalarZnxDftPrepOps<FFT64> for Module<FFT64> {
    fn svp_prepare<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: ScalarZnxDftPrepToMut<FFT64>,
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
        A: ScalarZnxDftPrepToRef<FFT64>,
        B: VecZnxDftToRef<FFT64>,
    {
        let mut res: VecZnxDft<&mut [u8], FFT64> = res.to_mut();
        let a: ScalarZnxDftPrep<&[u8], FFT64> = a.to_ref();
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
        A: ScalarZnxDftPrepToRef<FFT64>,
    {
        let mut res: VecZnxDft<&mut [u8], FFT64> = res.to_mut();
        let a: ScalarZnxDftPrep<&[u8], FFT64> = a.to_ref();
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

#[allow(unused_variables)]
impl ScalarZnxDftPrepOps<NTT120> for Module<NTT120> {
    fn svp_prepare<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: ScalarZnxDftPrepToMut<NTT120>,
        A: ScalarZnxToRef,
    {
        unimplemented!()
    }

    fn svp_apply<R, A, B>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
    where
        R: VecZnxDftToMut<NTT120>,
        A: ScalarZnxDftPrepToRef<NTT120>,
        B: VecZnxDftToRef<NTT120>,
    {
        unimplemented!()
    }

    fn svp_apply_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<NTT120>,
        A: ScalarZnxDftPrepToRef<NTT120>,
    {
        unimplemented!()
    }
}
