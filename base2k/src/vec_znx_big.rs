use crate::ffi::vec_znx_big;
use crate::znx_base::{ZnxInfos, ZnxView};
use crate::{Backend, DataView, DataViewMut, FFT64, Module, alloc_aligned};
use std::marker::PhantomData;

const VEC_ZNX_BIG_ROWS: usize = 1;

/// VecZnxBig is `Backend` dependent, denoted with backend generic `B`
pub struct VecZnxBig<D, B> {
    data: D,
    n: usize,
    cols: usize,
    size: usize,
    _phantom: PhantomData<B>,
}

impl<D, B> ZnxInfos for VecZnxBig<D, B> {
    fn cols(&self) -> usize {
        self.cols
    }

    fn rows(&self) -> usize {
        1
    }

    fn n(&self) -> usize {
        self.n
    }

    fn size(&self) -> usize {
        self.size
    }

    fn sl(&self) -> usize {
        self.cols() * self.n()
    }
}

impl<D, B> DataView for VecZnxBig<D, B> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D, B> DataViewMut for VecZnxBig<D, B> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: AsRef<[u8]>> ZnxView for VecZnxBig<D, FFT64> {
    type Scalar = i64;
}

impl<D: From<Vec<u8>>, B: Backend> VecZnxBig<D, B> {
    pub(crate) fn bytes_of(module: &Module<B>, cols: usize, size: usize) -> usize {
        unsafe { vec_znx_big::bytes_of_vec_znx_big(module.ptr, size as u64) as usize * cols }
    }

    pub(crate) fn new(module: &Module<B>, cols: usize, size: usize) -> Self {
        let data = alloc_aligned::<u8>(Self::bytes_of(module, cols, size));
        Self {
            data: data.into(),
            n: module.n(),
            cols,
            size,
            _phantom: PhantomData,
        }
    }

    pub(crate) fn new_from_bytes(module: &Module<B>, cols: usize, size: usize, bytes: impl Into<Vec<u8>>) -> Self {
        let data: Vec<u8> = bytes.into();
        assert!(data.len() == Self::bytes_of(module, cols, size));
        Self {
            data: data.into(),
            n: module.n(),
            cols,
            size,
            _phantom: PhantomData,
        }
    }
}

pub type VecZnxBigOwned<B> = VecZnxBig<Vec<u8>, B>;

// impl VecZnxBig<FFT64> {
//     pub fn print(&self, n: usize, col: usize) {
//         (0..self.size()).for_each(|i| println!("{}: {:?}", i, &self.at(col, i)[..n]));
//     }
// }
