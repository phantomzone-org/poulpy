use std::marker::PhantomData;

use crate::ffi::svp;
use crate::znx_base::ZnxInfos;
use crate::{Backend, DataView, DataViewMut, FFT64, Module, ZnxView, alloc_aligned};

pub const SCALAR_ZNX_DFT_ROWS: usize = 1;
pub const SCALAR_ZNX_DFT_SIZE: usize = 1;

pub struct ScalarZnxDft<D, B> {
    data: D,
    n: usize,
    cols: usize,
    _phantom: PhantomData<B>,
}

impl<D, B> ZnxInfos for ScalarZnxDft<D, B> {
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
        1
    }

    fn sl(&self) -> usize {
        self.n()
    }
}

impl<D, B> DataView for ScalarZnxDft<D, B> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D, B> DataViewMut for ScalarZnxDft<D, B> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: AsRef<[u8]>> ZnxView for ScalarZnxDft<D, FFT64> {
    type Scalar = f64;
}

impl<D: From<Vec<u8>>, B: Backend> ScalarZnxDft<D, B> {
    pub(crate) fn bytes_of(module: &Module<B>, cols: usize) -> usize {
        unsafe { svp::bytes_of_svp_ppol(module.ptr) as usize * cols }
    }

    pub(crate) fn new(module: &Module<B>, cols: usize) -> Self {
        let data = alloc_aligned::<u8>(Self::bytes_of(module, cols));
        Self {
            data: data.into(),
            n: module.n(),
            cols,
            _phantom: PhantomData,
        }
    }

    pub(crate) fn new_from_bytes(module: &Module<B>, cols: usize, bytes: impl Into<Vec<u8>>) -> Self {
        let data: Vec<u8> = bytes.into();
        assert!(data.len() == Self::bytes_of(module, cols));
        Self {
            data: data.into(),
            n: module.n(),
            cols,
            _phantom: PhantomData,
        }
    }

    // fn from_bytes_borrow(module: &Module<B>, _rows: usize, cols: usize, _size: usize, bytes: &mut [u8]) -> Self {
    //     debug_assert_eq!(bytes.len(), Self::bytes_of(module, _rows, cols, _size));
    //     Self {
    //         inner: ZnxBase::from_bytes_borrow(
    //             module.n(),
    //             SCALAR_ZNX_DFT_ROWS,
    //             cols,
    //             SCALAR_ZNX_DFT_SIZE,
    //             bytes,
    //         ),
    //         _phantom: PhantomData,
    //     }
    // }
}

pub type ScalarZnxDftOwned<B> = ScalarZnxDft<Vec<u8>, B>;
