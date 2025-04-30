use crate::ffi::vec_znx_dft;
use crate::znx_base::{GetZnxBase, ZnxAlloc, ZnxBase, ZnxInfos, ZnxLayout, ZnxSliceSize};
use crate::{Backend, FFT64, Module, VecZnxBig};
use std::marker::PhantomData;

const VEC_ZNX_DFT_ROWS: usize = 1;

pub struct VecZnxDft<B: Backend> {
    inner: ZnxBase,
    pub _marker: PhantomData<B>,
}

impl<B: Backend> GetZnxBase for VecZnxDft<B> {
    fn znx(&self) -> &ZnxBase {
        &self.inner
    }

    fn znx_mut(&mut self) -> &mut ZnxBase {
        &mut self.inner
    }
}

impl<B: Backend> ZnxInfos for VecZnxDft<B> {}

impl<B: Backend> ZnxAlloc<B> for VecZnxDft<B> {
    type Scalar = u8;

    fn from_bytes_borrow(module: &Module<B>, _rows: usize, cols: usize, size: usize, bytes: &mut [u8]) -> Self {
        Self {
            inner: ZnxBase::from_bytes_borrow(module.n(), VEC_ZNX_DFT_ROWS, cols, size, bytes),
            _marker: PhantomData,
        }
    }

    fn bytes_of(module: &Module<B>, _rows: usize, cols: usize, size: usize) -> usize {
        debug_assert_eq!(
            _rows, VEC_ZNX_DFT_ROWS,
            "rows != {} not supported for VecZnxDft",
            VEC_ZNX_DFT_ROWS
        );
        unsafe { vec_znx_dft::bytes_of_vec_znx_dft(module.ptr, size as u64) as usize * cols }
    }
}

impl ZnxLayout for VecZnxDft<FFT64> {
    type Scalar = f64;
}

impl ZnxSliceSize for VecZnxDft<FFT64> {
    fn sl(&self) -> usize {
        self.n()
    }
}

impl VecZnxDft<FFT64> {
    pub fn print(&self, n: usize) {
        (0..self.size()).for_each(|i| println!("{}: {:?}", i, &self.at_limb(i)[..n]));
    }
}

impl<B: Backend> VecZnxDft<B> {
    /// Cast a [VecZnxDft] into a [VecZnxBig].
    /// The returned [VecZnxBig] shares the backing array
    /// with the original [VecZnxDft].
    pub fn alias_as_vec_znx_big(&mut self) -> VecZnxBig<B> {
        VecZnxBig::<B> {
            inner: ZnxBase {
                data: Vec::new(),
                ptr: self.ptr(),
                n: self.n(),
                rows: self.rows(),
                cols: self.cols(),
                size: self.size(),
            },
            _marker: PhantomData,
        }
    }
}
