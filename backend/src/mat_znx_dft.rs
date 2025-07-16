use crate::znx_base::ZnxInfos;
use crate::{Backend, DataView, DataViewMut, FFT64, NTT120, VecZnxDft, VecZnxDftBytesOf, ZnxSliceSize, ZnxView, alloc_aligned};
use std::marker::PhantomData;

/// A matrix of [VecZnxDft].
pub struct MatZnxDft<D, B: Backend> {
    data: D,
    n: usize,
    size: usize,
    rows: usize,
    cols_in: usize,
    cols_out: usize,
    _phantom: PhantomData<B>,
}

impl<D, B: Backend> ZnxInfos for MatZnxDft<D, B> {
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

impl<D> ZnxSliceSize for MatZnxDft<D, FFT64> {
    fn sl(&self) -> usize {
        self.n() * self.cols_out()
    }
}

impl<D, B: Backend> DataView for MatZnxDft<D, B> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D, B: Backend> DataViewMut for MatZnxDft<D, B> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: AsRef<[u8]>> ZnxView for MatZnxDft<D, FFT64> {
    type Scalar = f64;
}

impl<D, B: Backend> MatZnxDft<D, B> {
    pub fn cols_in(&self) -> usize {
        self.cols_in
    }

    pub fn cols_out(&self) -> usize {
        self.cols_out
    }
}

pub trait MatZnxDftBytesOf<D, B: Backend> {
    fn bytes_of(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize;
}

impl<D: AsRef<[u8]>> MatZnxDftBytesOf<D, FFT64> for MatZnxDft<D, FFT64>
where
    VecZnxDft<D, FFT64>: VecZnxDftBytesOf<D, FFT64>,
{
    fn bytes_of(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        rows * cols_in * VecZnxDft::bytes_of(n, cols_out, size)
    }
}

impl<D: AsRef<[u8]>> MatZnxDftBytesOf<D, NTT120> for MatZnxDft<D, NTT120>
where
    VecZnxDft<D, FFT64>: VecZnxDftBytesOf<D, FFT64>,
{
    fn bytes_of(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        rows * cols_in * VecZnxDft::bytes_of(n, cols_out, size)
    }
}

impl<D: From<Vec<u8>> + AsRef<[u8]>, B: Backend> MatZnxDft<D, B>
where
    MatZnxDft<D, B>: MatZnxDftBytesOf<D, B>,
{
    pub(crate) fn new(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> Self {
        let data: Vec<u8> = alloc_aligned(Self::bytes_of(n, rows, cols_in, cols_out, size));
        Self {
            data: data.into(),
            n,
            size,
            rows,
            cols_in,
            cols_out,
            _phantom: PhantomData,
        }
    }

    pub(crate) fn new_from_bytes(
        n: usize,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
        bytes: impl Into<Vec<u8>>,
    ) -> Self {
        let data: Vec<u8> = bytes.into();
        assert!(data.len() == Self::bytes_of(n, rows, cols_in, cols_out, size));
        Self {
            data: data.into(),
            n,
            size,
            rows,
            cols_in,
            cols_out,
            _phantom: PhantomData,
        }
    }
}

impl<D: AsRef<[u8]>, B: Backend> MatZnxDft<D, B>
where
    MatZnxDft<D, B>: MatZnxDftToRef<B>,
    VecZnxDft<D, B>: VecZnxDftBytesOf<D, B>,
{
    pub fn at(&self, row: usize, col: usize) -> VecZnxDft<&[u8], B> {
        let self_ref: MatZnxDft<&[u8], B> = self.to_ref();
        let nb_bytes: usize = VecZnxDft::bytes_of(self.n, self.cols_out, self.size);
        let start: usize = nb_bytes * row * col;
        let end: usize = start + nb_bytes;

        VecZnxDft {
            data: &self_ref.data[start..end],
            n: self.n,
            cols: self.cols_out,
            size: self.size,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<D: AsRef<[u8]> + AsMut<[u8]>, B: Backend> MatZnxDft<D, B>
where
    MatZnxDft<D, B>: MatZnxDftToMut<B>,
    VecZnxDft<D, B>: VecZnxDftBytesOf<D, B>,
{
    pub fn at_mut(&mut self, row: usize, col: usize) -> VecZnxDft<&mut [u8], B> {
        let n: usize = self.n();
        let cols_out: usize = self.cols_out();
        let size: usize = self.size();

        let self_ref: MatZnxDft<&mut [u8], B> = self.to_mut();
        let nb_bytes: usize = VecZnxDft::bytes_of(n, cols_out, size);
        let start: usize = nb_bytes * row * col;
        let end: usize = start + nb_bytes;

        VecZnxDft {
            data: &mut self_ref.data[start..end],
            n,
            cols: cols_out,
            size,
            _phantom: std::marker::PhantomData,
        }
    }
}

pub type MatZnxDftOwned<B> = MatZnxDft<Vec<u8>, B>;
pub type MatZnxDftMut<'a, B> = MatZnxDft<&'a mut [u8], B>;
pub type MatZnxDftRef<'a, B> = MatZnxDft<&'a [u8], B>;

pub trait MatZnxDftToRef<B: Backend> {
    fn to_ref(&self) -> MatZnxDft<&[u8], B>;
}

impl<D, B: Backend> MatZnxDftToRef<B> for MatZnxDft<D, B>
where
    D: AsRef<[u8]>,
    B: Backend,
{
    fn to_ref(&self) -> MatZnxDft<&[u8], B> {
        MatZnxDft {
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

pub trait MatZnxDftToMut<B: Backend> {
    fn to_mut(&mut self) -> MatZnxDft<&mut [u8], B>;
}

impl<D, B: Backend> MatZnxDftToMut<B> for MatZnxDft<D, B>
where
    D: AsRef<[u8]> + AsMut<[u8]>,
    B: Backend,
{
    fn to_mut(&mut self) -> MatZnxDft<&mut [u8], B> {
        MatZnxDft {
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

impl<D, B: Backend> MatZnxDft<D, B> {
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
