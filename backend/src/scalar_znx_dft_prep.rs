use crate::znx_base::ZnxInfos;
use crate::{Backend, DataView, DataViewMut, FFT64, NTT120, ZnxSliceSize, ZnxView, alloc_aligned};
use std::marker::PhantomData;

pub struct ScalarZnxDftPrep<D, B: Backend> {
    data: D,
    n: usize,
    cols: usize,
    _phantom: PhantomData<B>,
}

impl<D, B: Backend> ZnxInfos for ScalarZnxDftPrep<D, B> {
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

impl<D> ZnxSliceSize for ScalarZnxDftPrep<D, FFT64> {
    fn sl(&self) -> usize {
        self.n()
    }
}

impl<D> ZnxSliceSize for ScalarZnxDftPrep<D, NTT120> {
    fn sl(&self) -> usize {
        4 * self.n()
    }
}

impl<D, B: Backend> DataView for ScalarZnxDftPrep<D, B> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D, B: Backend> DataViewMut for ScalarZnxDftPrep<D, B> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: AsRef<[u8]>> ZnxView for ScalarZnxDftPrep<D, FFT64> {
    type Scalar = f64;
}

impl<D: AsRef<[u8]>> ZnxView for ScalarZnxDftPrep<D, NTT120> {
    type Scalar = i64;
}

pub trait ScalarZnxDftPrepBytesOf<B: Backend> {
    fn bytes_of(n: usize, cols: usize) -> usize;
}

impl<D: AsRef<[u8]>> ScalarZnxDftPrepBytesOf<FFT64> for ScalarZnxDftPrep<D, FFT64> {
    fn bytes_of(n: usize, cols: usize) -> usize {
        n * cols * size_of::<f64>()
    }
}

impl<D: AsRef<[u8]>> ScalarZnxDftPrepBytesOf<NTT120> for ScalarZnxDftPrep<D, NTT120> {
    fn bytes_of(n: usize, cols: usize) -> usize {
        4 * n * cols * size_of::<i64>()
    }
}

impl<D: From<Vec<u8>> + AsRef<[u8]>, B: Backend> ScalarZnxDftPrep<D, B>
where
    ScalarZnxDftPrep<D, B>: ScalarZnxDftPrepBytesOf<B>,
{
    pub(crate) fn new(n: usize, cols: usize) -> Self {
        let data: Vec<u8> = alloc_aligned::<u8>(Self::bytes_of(n, cols));
        Self {
            data: data.into(),
            n,
            cols,
            _phantom: PhantomData,
        }
    }

    pub(crate) fn new_from_bytes(n: usize, cols: usize, bytes: impl Into<Vec<u8>>) -> Self {
        let data: Vec<u8> = bytes.into();
        assert!(data.len() == Self::bytes_of(n, cols));
        Self {
            data: data.into(),
            n,
            cols,
            _phantom: PhantomData,
        }
    }
}

pub type ScalarZnxDftPrepOwned<B> = ScalarZnxDftPrep<Vec<u8>, B>;

pub trait ScalarZnxDftPrepToRef<B: Backend> {
    fn to_ref(&self) -> ScalarZnxDftPrep<&[u8], B>;
}

impl<D: AsRef<[u8]>, B: Backend> ScalarZnxDftPrepToRef<B> for ScalarZnxDftPrep<D, B> {
    fn to_ref(&self) -> ScalarZnxDftPrep<&[u8], B> {
        ScalarZnxDftPrep {
            data: self.data.as_ref(),
            n: self.n,
            cols: self.cols,
            _phantom: PhantomData,
        }
    }
}

pub trait ScalarZnxDftPrepToMut<B: Backend> {
    fn to_mut(&mut self) -> ScalarZnxDftPrep<&mut [u8], B>;
}

impl<D: AsMut<[u8]> + AsRef<[u8]>, B: Backend> ScalarZnxDftPrepToMut<B> for ScalarZnxDftPrep<D, B> {
    fn to_mut(&mut self) -> ScalarZnxDftPrep<&mut [u8], B> {
        ScalarZnxDftPrep {
            data: self.data.as_mut(),
            n: self.n,
            cols: self.cols,
            _phantom: PhantomData,
        }
    }
}

impl<D, B: Backend> ScalarZnxDftPrep<D, B> {
    pub(crate) fn from_data(data: D, n: usize, cols: usize) -> Self {
        Self {
            data,
            n,
            cols,
            _phantom: PhantomData,
        }
    }
}
