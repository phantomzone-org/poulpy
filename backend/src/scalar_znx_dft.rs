use std::marker::PhantomData;

use crate::ffi::svp;
use crate::znx_base::ZnxInfos;
use crate::{
    Backend, DataView, DataViewMut, FFT64, Module, VecZnxDft, VecZnxDftToMut, VecZnxDftToRef, ZnxSliceSize, ZnxView,
    alloc_aligned,
};

pub struct ScalarZnxDft<D, B: Backend> {
    data: D,
    n: usize,
    cols: usize,
    _phantom: PhantomData<B>,
}

impl<D, B: Backend> ZnxInfos for ScalarZnxDft<D, B> {
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
}

impl<D> ZnxSliceSize for ScalarZnxDft<D, FFT64> {
    fn sl(&self) -> usize {
        self.n()
    }
}

impl<D, B: Backend> DataView for ScalarZnxDft<D, B> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D, B: Backend> DataViewMut for ScalarZnxDft<D, B> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: AsRef<[u8]>> ZnxView for ScalarZnxDft<D, FFT64> {
    type Scalar = f64;
}

pub(crate) fn bytes_of_scalar_znx_dft<B: Backend>(module: &Module<B>, cols: usize) -> usize {
    ScalarZnxDftOwned::bytes_of(module, cols)
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
}

impl<D, B: Backend> ScalarZnxDft<D, B> {
    pub(crate) fn from_data(data: D, n: usize, cols: usize) -> Self {
        Self {
            data,
            n,
            cols,
            _phantom: PhantomData,
        }
    }

    pub fn as_vec_znx_dft(self) -> VecZnxDft<D, B> {
        VecZnxDft {
            data: self.data,
            n: self.n,
            cols: self.cols,
            size: 1,
            _phantom: PhantomData,
        }
    }
}

pub type ScalarZnxDftOwned<B> = ScalarZnxDft<Vec<u8>, B>;

pub trait ScalarZnxDftToRef<B: Backend> {
    fn to_ref(&self) -> ScalarZnxDft<&[u8], B>;
}

impl<D, B: Backend> ScalarZnxDftToRef<B> for ScalarZnxDft<D, B>
where
    D: AsRef<[u8]>,
    B: Backend,
{
    fn to_ref(&self) -> ScalarZnxDft<&[u8], B> {
        ScalarZnxDft {
            data: self.data.as_ref(),
            n: self.n,
            cols: self.cols,
            _phantom: PhantomData,
        }
    }
}

pub trait ScalarZnxDftToMut<B: Backend> {
    fn to_mut(&mut self) -> ScalarZnxDft<&mut [u8], B>;
}

impl<D, B: Backend> ScalarZnxDftToMut<B> for ScalarZnxDft<D, B>
where
    D: AsMut<[u8]> + AsRef<[u8]>,
    B: Backend,
{
    fn to_mut(&mut self) -> ScalarZnxDft<&mut [u8], B> {
        ScalarZnxDft {
            data: self.data.as_mut(),
            n: self.n,
            cols: self.cols,
            _phantom: PhantomData,
        }
    }
}

impl<D, B: Backend> VecZnxDftToRef<B> for ScalarZnxDft<D, B>
where
    D: AsRef<[u8]>,
    B: Backend,
{
    fn to_ref(&self) -> VecZnxDft<&[u8], B> {
        VecZnxDft {
            data: self.data.as_ref(),
            n: self.n,
            cols: self.cols,
            size: 1,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<D, B: Backend> VecZnxDftToMut<B> for ScalarZnxDft<D, B>
where
    D: AsRef<[u8]> + AsMut<[u8]>,
    B: Backend,
{
    fn to_mut(&mut self) -> VecZnxDft<&mut [u8], B> {
        VecZnxDft {
            data: self.data.as_mut(),
            n: self.n,
            cols: self.cols,
            size: 1,
            _phantom: std::marker::PhantomData,
        }
    }
}
