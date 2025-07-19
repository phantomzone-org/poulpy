use crate::znx_base::ZnxInfos;
use crate::{Backend, DataView, DataViewMut, FFT64, Module, NTT120, ZnxSliceSize, ZnxView, alloc_aligned};
use std::marker::PhantomData;

/// An opaque version of [MatZnxDft], which is prepared for a specific backend, to be
/// given as right operand of [vmp_apply].
pub struct MatZnxDftPrep<D, B: Backend> {
    data: D,
    n: usize,
    size: usize,
    rows: usize,
    cols_in: usize,
    cols_out: usize,
    _phantom: PhantomData<B>,
}

impl<D, B: Backend> ZnxInfos for MatZnxDftPrep<D, B> {
    fn cols(&self) -> usize {
        self.cols_in
    }

    fn rows(&self) -> usize {
        self.rows
    }

    fn n(&self) -> usize {
        self.n
    }

    fn size(&self) -> usize {
        self.size
    }
}

impl<D> ZnxSliceSize for MatZnxDftPrep<D, FFT64> {
    fn sl(&self) -> usize {
        self.n() * self.cols_out()
    }
}

impl<D> ZnxSliceSize for MatZnxDftPrep<D, NTT120> {
    fn sl(&self) -> usize {
        4 * self.n() * self.cols_out()
    }
}

impl<D, B: Backend> DataView for MatZnxDftPrep<D, B> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D, B: Backend> DataViewMut for MatZnxDftPrep<D, B> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: AsRef<[u8]>> ZnxView for MatZnxDftPrep<D, FFT64> {
    type Scalar = f64;
}

impl<D: AsRef<[u8]>> ZnxView for MatZnxDftPrep<D, NTT120> {
    type Scalar = i64;
}

impl<D, B: Backend> MatZnxDftPrep<D, B> {
    pub fn cols_in(&self) -> usize {
        self.cols_in
    }

    pub fn cols_out(&self) -> usize {
        self.cols_out
    }
}

impl<D: From<Vec<u8>>, B: Backend> MatZnxDftPrep<D, B> {
    pub(crate) fn bytes_of(module: &Module<B>, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        unsafe {
            crate::ffi::vmp::bytes_of_vmp_pmat(
                module.ptr,
                (rows * cols_in) as u64,
                (size * cols_out) as u64,
            ) as usize
        }
    }

    pub(crate) fn new(module: &Module<B>, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> Self {
        let data: Vec<u8> = alloc_aligned(Self::bytes_of(module, rows, cols_in, cols_out, size));
        Self {
            data: data.into(),
            n: module.n(),
            size,
            rows,
            cols_in,
            cols_out,
            _phantom: PhantomData,
        }
    }

    pub(crate) fn new_from_bytes(
        module: &Module<B>,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
        bytes: impl Into<Vec<u8>>,
    ) -> Self {
        let data: Vec<u8> = bytes.into();
        assert!(data.len() == Self::bytes_of(module, rows, cols_in, cols_out, size));
        Self {
            data: data.into(),
            n: module.n(),
            size,
            rows,
            cols_in,
            cols_out,
            _phantom: PhantomData,
        }
    }
}

pub type MatZnxDftPrepOwned<B> = MatZnxDftPrep<Vec<u8>, B>;
pub type MatZnxDftPrepRef<'a, B> = MatZnxDftPrep<&'a [u8], B>;

pub trait MatZnxDftPrepToRef<B: Backend> {
    fn to_ref(&self) -> MatZnxDftPrep<&[u8], B>;
}

impl<D, B: Backend> MatZnxDftPrepToRef<B> for MatZnxDftPrep<D, B>
where
    D: AsRef<[u8]>,
    B: Backend,
{
    fn to_ref(&self) -> MatZnxDftPrep<&[u8], B> {
        MatZnxDftPrep {
            data: self.data.as_ref(),
            n: self.n,
            rows: self.rows,
            cols_in: self.cols_in,
            cols_out: self.cols_out,
            size: self.size,
            _phantom: std::marker::PhantomData,
        }
    }
}

pub trait MatZnxDftPrepToMut<B: Backend> {
    fn to_mut(&mut self) -> MatZnxDftPrep<&mut [u8], B>;
}

impl<D, B: Backend> MatZnxDftPrepToMut<B> for MatZnxDftPrep<D, B>
where
    D: AsRef<[u8]> + AsMut<[u8]>,
    B: Backend,
{
    fn to_mut(&mut self) -> MatZnxDftPrep<&mut [u8], B> {
        MatZnxDftPrep {
            data: self.data.as_mut(),
            n: self.n,
            rows: self.rows,
            cols_in: self.cols_in,
            cols_out: self.cols_out,
            size: self.size,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<D, B: Backend> MatZnxDftPrep<D, B> {
    pub(crate) fn from_data(data: D, n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> Self {
        Self {
            data,
            n,
            rows,
            cols_in,
            cols_out,
            size,
            _phantom: PhantomData,
        }
    }
}
