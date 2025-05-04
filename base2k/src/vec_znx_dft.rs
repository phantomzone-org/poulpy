use std::marker::PhantomData;

use crate::ffi::vec_znx_dft;
use crate::znx_base::ZnxInfos;
use crate::{Backend, DataView, DataViewMut, FFT64, Module, ZnxView, alloc_aligned};

// const VEC_ZNX_DFT_ROWS: usize = 1;

// VecZnxDft is `Backend` dependent denoted with generic `B`
pub struct VecZnxDft<D, B> {
    data: D,
    n: usize,
    cols: usize,
    size: usize,
    _phantom: PhantomData<B>,
}

impl<D, B> ZnxInfos for VecZnxDft<D, B> {
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

impl<D, B> DataView for VecZnxDft<D, B> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D, B> DataViewMut for VecZnxDft<D, B> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: AsRef<[u8]>> ZnxView for VecZnxDft<D, FFT64> {
    type Scalar = f64;
}

pub(crate) fn bytes_of_vec_znx_dft<B: Backend>(module: &Module<B>, cols: usize, size: usize) -> usize {
    unsafe { vec_znx_dft::bytes_of_vec_znx_dft(module.ptr, size as u64) as usize * cols }
}

impl<D: From<Vec<u8>>, B: Backend> VecZnxDft<D, B> {
    pub(crate) fn new(module: &Module<B>, cols: usize, size: usize) -> Self {
        let data = alloc_aligned::<u8>(bytes_of_vec_znx_dft(module, cols, size));
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
        assert!(data.len() == bytes_of_vec_znx_dft(module, cols, size));
        Self {
            data: data.into(),
            n: module.n(),
            cols,
            size,
            _phantom: PhantomData,
        }
    }
}

pub type VecZnxDftOwned<B> = VecZnxDft<Vec<u8>, B>;

impl<D, B> VecZnxDft<D, B> {
    pub(crate) fn from_data(data: D, n: usize, cols: usize, size: usize) -> Self {
        Self {
            data,
            n,
            cols,
            size,
            _phantom: PhantomData,
        }
    }
}

impl<B> VecZnxDft<Vec<u8>, B> {
    pub fn to_mut(&mut self) -> VecZnxDft<&mut [u8], B> {
        VecZnxDft {
            data: self.data.as_mut_slice(),
            n: self.n,
            cols: self.cols,
            size: self.size,
            _phantom: PhantomData,
        }
    }

    pub fn to_ref(&self) -> VecZnxDft<&[u8], B> {
        VecZnxDft {
            data: self.data.as_slice(),
            n: self.n,
            cols: self.cols,
            size: self.size,
            _phantom: PhantomData,
        }
    }
}

impl<B> VecZnxDft<&mut [u8], B> {
    pub fn to_ref(&self) -> VecZnxDft<&[u8], B> {
        VecZnxDft {
            data: &self.data,
            n: self.n,
            cols: self.cols,
            size: self.size,
            _phantom: PhantomData,
        }
    }
}
