use std::marker::PhantomData;

use crate::ffi::svp;
use crate::znx_base::{ZnxAlloc, ZnxBase, ZnxInfos, ZnxLayout, ZnxSliceSize};
use crate::{Backend, FFT64, GetZnxBase, Module};

pub const SCALAR_ZNX_DFT_ROWS: usize = 1;
pub const SCALAR_ZNX_DFT_SIZE: usize = 1;

pub struct ScalarZnxDft<B: Backend> {
    pub inner: ZnxBase,
    _marker: PhantomData<B>,
}

impl<B: Backend> GetZnxBase for ScalarZnxDft<B> {
    fn znx(&self) -> &ZnxBase {
        &self.inner
    }

    fn znx_mut(&mut self) -> &mut ZnxBase {
        &mut self.inner
    }
}

impl<B: Backend> ZnxInfos for ScalarZnxDft<B> {}

impl<B: Backend> ZnxAlloc<B> for ScalarZnxDft<B> {
    type Scalar = u8;

    fn from_bytes_borrow(module: &Module<B>, _rows: usize, cols: usize, _size: usize, bytes: &mut [u8]) -> Self {
        debug_assert_eq!(bytes.len(), Self::bytes_of(module, _rows, cols, _size));
        Self {
            inner: ZnxBase::from_bytes_borrow(
                module.n(),
                SCALAR_ZNX_DFT_ROWS,
                cols,
                SCALAR_ZNX_DFT_SIZE,
                bytes,
            ),
            _marker: PhantomData,
        }
    }

    fn bytes_of(module: &Module<B>, _rows: usize, cols: usize, _size: usize) -> usize {
        debug_assert_eq!(
            _rows, SCALAR_ZNX_DFT_ROWS,
            "rows != {} not supported for ScalarZnxDft",
            SCALAR_ZNX_DFT_ROWS
        );
        debug_assert_eq!(
            _size, SCALAR_ZNX_DFT_SIZE,
            "rows != {} not supported for ScalarZnxDft",
            SCALAR_ZNX_DFT_SIZE
        );
        unsafe { svp::bytes_of_svp_ppol(module.ptr) as usize * cols }
    }
}

impl ZnxLayout for ScalarZnxDft<FFT64> {
    type Scalar = f64;
}

impl ZnxSliceSize for ScalarZnxDft<FFT64> {
    fn sl(&self) -> usize {
        self.n() * self.cols()
    }
}
