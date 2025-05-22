use crate::znx_base::ZnxInfos;
use crate::{Backend, DataView, DataViewMut, FFT64, Module, ZnxSliceSize, ZnxView, alloc_aligned};
use std::marker::PhantomData;

/// Vector Matrix Product Prepared Matrix: a vector of [VecZnx],
/// stored as a 3D matrix in the DFT domain in a single contiguous array.
/// Each col of the [MatZnxDft] can be seen as a collection of [VecZnxDft].
///
/// [MatZnxDft] is used to permform a vector matrix product between a [VecZnx]/[VecZnxDft] and a [MatZnxDft].
/// See the trait [MatZnxDftOps] for additional information.
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

impl<D: From<Vec<u8>>, B: Backend> MatZnxDft<D, B> {
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

impl<D: AsRef<[u8]>> MatZnxDft<D, FFT64> {
    /// Returns a copy of the backend array at index (i, j) of the [MatZnxDft].
    ///
    /// # Arguments
    ///
    /// * `row`: row index (i).
    /// * `col`: col index (j).
    #[allow(dead_code)]
    fn at(&self, row: usize, col: usize) -> Vec<f64> {
        let n: usize = self.n();

        let mut res: Vec<f64> = alloc_aligned(n);

        if n < 8 {
            res.copy_from_slice(&self.raw()[(row + col * self.rows()) * n..(row + col * self.rows()) * (n + 1)]);
        } else {
            (0..n >> 3).for_each(|blk| {
                res[blk * 8..(blk + 1) * 8].copy_from_slice(&self.at_block(row, col, blk)[..8]);
            });
        }

        res
    }

    #[allow(dead_code)]
    fn at_block(&self, row: usize, col: usize, blk: usize) -> &[f64] {
        let nrows: usize = self.rows();
        let nsize: usize = self.size();
        if col == (nsize - 1) && (nsize & 1 == 1) {
            &self.raw()[blk * nrows * nsize * 8 + col * nrows * 8 + row * 8..]
        } else {
            &self.raw()[blk * nrows * nsize * 8 + (col / 2) * (2 * nrows) * 8 + row * 2 * 8 + (col % 2) * 8..]
        }
    }
}

pub type MatZnxDftOwned<B> = MatZnxDft<Vec<u8>, B>;

pub trait MatZnxDftToRef<B: Backend> {
    fn to_ref(&self) -> MatZnxDft<&[u8], B>;
}

pub trait MatZnxDftToMut<B: Backend>: MatZnxDftToRef<B> {
    fn to_mut(&mut self) -> MatZnxDft<&mut [u8], B>;
}

impl<B: Backend> MatZnxDftToMut<B> for MatZnxDft<Vec<u8>, B> {
    fn to_mut(&mut self) -> MatZnxDft<&mut [u8], B> {
        MatZnxDft {
            data: self.data.as_mut_slice(),
            n: self.n,
            rows: self.rows,
            cols_in: self.cols_in,
            cols_out: self.cols_out,
            size: self.size,
            _phantom: PhantomData,
        }
    }
}

impl<B: Backend> MatZnxDftToRef<B> for MatZnxDft<Vec<u8>, B> {
    fn to_ref(&self) -> MatZnxDft<&[u8], B> {
        MatZnxDft {
            data: self.data.as_slice(),
            n: self.n,
            rows: self.rows,
            cols_in: self.cols_in,
            cols_out: self.cols_out,
            size: self.size,
            _phantom: PhantomData,
        }
    }
}

impl<B: Backend> MatZnxDftToMut<B> for MatZnxDft<&mut [u8], B> {
    fn to_mut(&mut self) -> MatZnxDft<&mut [u8], B> {
        MatZnxDft {
            data: self.data,
            n: self.n,
            rows: self.rows,
            cols_in: self.cols_in,
            cols_out: self.cols_out,
            size: self.size,
            _phantom: PhantomData,
        }
    }
}

impl<B: Backend> MatZnxDftToRef<B> for MatZnxDft<&mut [u8], B> {
    fn to_ref(&self) -> MatZnxDft<&[u8], B> {
        MatZnxDft {
            data: self.data,
            n: self.n,
            rows: self.rows,
            cols_in: self.cols_in,
            cols_out: self.cols_out,
            size: self.size,
            _phantom: PhantomData,
        }
    }
}

impl<B: Backend> MatZnxDftToRef<B> for MatZnxDft<&[u8], B> {
    fn to_ref(&self) -> MatZnxDft<&[u8], B> {
        MatZnxDft {
            data: self.data,
            n: self.n,
            rows: self.rows,
            cols_in: self.cols_in,
            cols_out: self.cols_out,
            size: self.size,
            _phantom: PhantomData,
        }
    }
}
