use crate::ffi::vec_znx_big;
use crate::znx_base::{GetZnxBase, ZnxAlloc, ZnxBase, ZnxBasics, ZnxInfos, ZnxLayout, ZnxSliceSize};
use crate::{Backend, FFT64, Module, NTT120};
use std::marker::PhantomData;

const VEC_ZNX_BIG_ROWS: usize = 1;

pub struct VecZnxBig<B: Backend> {
    pub inner: ZnxBase,
    pub _marker: PhantomData<B>,
}

impl<B: Backend> GetZnxBase for VecZnxBig<B> {
    fn znx(&self) -> &ZnxBase {
        &self.inner
    }

    fn znx_mut(&mut self) -> &mut ZnxBase {
        &mut self.inner
    }
}

impl<B: Backend> ZnxInfos for VecZnxBig<B> {}

impl<B: Backend> ZnxAlloc<B> for VecZnxBig<B> {
    type Scalar = u8;

    fn from_bytes_borrow(module: &Module<B>, _rows: usize, cols: usize, size: usize, bytes: &mut [u8]) -> Self {
        VecZnxBig {
            inner: ZnxBase::from_bytes_borrow(module.n(), VEC_ZNX_BIG_ROWS, cols, size, bytes),
            _marker: PhantomData,
        }
    }

    fn bytes_of(module: &Module<B>, _rows: usize, cols: usize, size: usize) -> usize {
        debug_assert_eq!(
            _rows, VEC_ZNX_BIG_ROWS,
            "rows != {} not supported for VecZnxBig",
            VEC_ZNX_BIG_ROWS
        );
        unsafe { vec_znx_big::bytes_of_vec_znx_big(module.ptr, size as u64) as usize * cols }
    }
}

impl ZnxLayout for VecZnxBig<FFT64> {
    type Scalar = i64;
}

impl ZnxLayout for VecZnxBig<NTT120> {
    type Scalar = i128;
}

impl ZnxBasics for VecZnxBig<FFT64> {}

impl ZnxSliceSize for VecZnxBig<FFT64> {
    fn sl(&self) -> usize {
        self.n()
    }
}

impl ZnxSliceSize for VecZnxBig<NTT120> {
    fn sl(&self) -> usize {
        self.n() * 4
    }
}

impl ZnxBasics for VecZnxBig<NTT120> {}

impl VecZnxBig<FFT64> {
    pub fn print(&self, n: usize) {
        (0..self.size()).for_each(|i| println!("{}: {:?}", i, &self.at_limb(i)[..n]));
    }
}
