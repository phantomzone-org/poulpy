use crate::ffi::svp::{self, svp_ppol_t};
use crate::ffi::vec_znx_dft::vec_znx_dft_t;
use crate::znx_base::{ZnxAlloc, ZnxInfos, ZnxLayout, ZnxSliceSize};
use crate::{Backend, FFT64, Module, SCALAR_ZNX_DFT_ROWS, SCALAR_ZNX_DFT_SIZE, Scalar, ScalarZnxDft, VecZnx, VecZnxDft};

pub trait ScalarZnxDftOps<B: Backend> {
    fn new_scalar_znx_dft(&self, cols: usize) -> ScalarZnxDft<B>;
    fn bytes_of_scalar_znx_dft(&self, cols: usize) -> usize;
    fn new_scalar_znx_dft_from_bytes(&self, cols: usize, bytes: Vec<u8>) -> ScalarZnxDft<B>;
    fn new_scalar_znx_dft_from_bytes_borrow(&self, cols: usize, bytes: &mut [u8]) -> ScalarZnxDft<B>;
    fn svp_prepare(&self, res: &mut ScalarZnxDft<B>, res_col: usize, a: &Scalar, a_col: usize);
    fn svp_apply_dft(&self, res: &mut VecZnxDft<B>, res_col: usize, a: &ScalarZnxDft<B>, a_col: usize, b: &VecZnx, b_col: usize);
}

impl ScalarZnxDftOps<FFT64> for Module<FFT64> {
    fn new_scalar_znx_dft(&self, cols: usize) -> ScalarZnxDft<FFT64> {
        ScalarZnxDft::<FFT64>::new(&self, SCALAR_ZNX_DFT_ROWS, cols, SCALAR_ZNX_DFT_SIZE)
    }

    fn bytes_of_scalar_znx_dft(&self, cols: usize) -> usize {
        ScalarZnxDft::<FFT64>::bytes_of(self, SCALAR_ZNX_DFT_ROWS, cols, SCALAR_ZNX_DFT_SIZE)
    }

    fn new_scalar_znx_dft_from_bytes(&self, cols: usize, bytes: Vec<u8>) -> ScalarZnxDft<FFT64> {
        ScalarZnxDft::from_bytes(self, SCALAR_ZNX_DFT_ROWS, cols, SCALAR_ZNX_DFT_SIZE, bytes)
    }

    fn new_scalar_znx_dft_from_bytes_borrow(&self, cols: usize, bytes: &mut [u8]) -> ScalarZnxDft<FFT64> {
        ScalarZnxDft::from_bytes_borrow(self, SCALAR_ZNX_DFT_ROWS, cols, SCALAR_ZNX_DFT_SIZE, bytes)
    }

    fn svp_prepare(&self, res: &mut ScalarZnxDft<FFT64>, res_col: usize, a: &Scalar, a_col: usize) {
        unsafe {
            svp::svp_prepare(
                self.ptr,
                res.at_mut_ptr(res_col, 0) as *mut svp_ppol_t,
                a.at_ptr(a_col, 0),
            )
        }
    }

    fn svp_apply_dft(
        &self,
        res: &mut VecZnxDft<FFT64>,
        res_col: usize,
        a: &ScalarZnxDft<FFT64>,
        a_col: usize,
        b: &VecZnx,
        b_col: usize,
    ) {
        unsafe {
            svp::svp_apply_dft(
                self.ptr,
                res.at_mut_ptr(res_col, 0) as *mut vec_znx_dft_t,
                res.size() as u64,
                a.at_ptr(a_col, 0) as *const svp_ppol_t,
                b.at_ptr(b_col, 0),
                b.size() as u64,
                b.sl() as u64,
            )
        }
    }
}
