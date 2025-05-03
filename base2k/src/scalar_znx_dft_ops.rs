use crate::ffi::svp::{self, svp_ppol_t};
use crate::ffi::vec_znx_dft::vec_znx_dft_t;
use crate::znx_base::{ZnxInfos, ZnxView, ZnxViewMut};
use crate::{Backend, FFT64, Module, Scalar, ScalarZnxDft, ScalarZnxDftOwned, VecZnx, VecZnxDft};

pub trait ScalarZnxDftAlloc<B> {
    fn new_scalar_znx_dft(&self, cols: usize) -> ScalarZnxDftOwned<B>;
    fn bytes_of_scalar_znx_dft(&self, cols: usize) -> usize;
    fn new_scalar_znx_dft_from_bytes(&self, cols: usize, bytes: Vec<u8>) -> ScalarZnxDftOwned<B>;
    // fn new_scalar_znx_dft_from_bytes_borrow(&self, cols: usize, bytes: &mut [u8]) -> ScalarZnxDft<B>;
}

pub trait ScalarZnxDftOps<DataMut, Data, B: Backend> {
    fn svp_prepare(&self, res: &mut ScalarZnxDft<DataMut, B>, res_col: usize, a: &Scalar<Data>, a_col: usize);
    fn svp_apply_dft(
        &self,
        res: &mut VecZnxDft<DataMut, B>,
        res_col: usize,
        a: &ScalarZnxDft<Data, B>,
        a_col: usize,
        b: &VecZnx<Data>,
        b_col: usize,
    );
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

    // fn new_scalar_znx_dft_from_bytes_borrow(&self, cols: usize, bytes: &mut [u8]) -> ScalarZnxDft<FFT64> {
    //     ScalarZnxDft::from_bytes_borrow(self, SCALAR_ZNX_DFT_ROWS, cols, SCALAR_ZNX_DFT_SIZE, bytes)
    // }
}

impl<DataMut, Data> ScalarZnxDftOps<DataMut, Data, FFT64> for Module<FFT64>
where
    DataMut: AsMut<[u8]> + AsRef<[u8]>,
    Data: AsRef<[u8]>,
{
    fn svp_prepare(&self, res: &mut ScalarZnxDft<DataMut, FFT64>, res_col: usize, a: &Scalar<Data>, a_col: usize) {
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
        res: &mut VecZnxDft<DataMut, FFT64>,
        res_col: usize,
        a: &ScalarZnxDft<Data, FFT64>,
        a_col: usize,
        b: &VecZnx<Data>,
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
