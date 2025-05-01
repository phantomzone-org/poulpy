use crate::znx_base::{GetZnxBase, ZnxBase, ZnxInfos, ZnxLayout, ZnxSliceSize};
use crate::{Backend, FFT64, Module, alloc_aligned};
use std::marker::PhantomData;

/// Vector Matrix Product Prepared Matrix: a vector of [VecZnx],
/// stored as a 3D matrix in the DFT domain in a single contiguous array.
/// Each col of the [MatZnxDft] can be seen as a collection of [VecZnxDft].
///
/// [MatZnxDft] is used to permform a vector matrix product between a [VecZnx]/[VecZnxDft] and a [MatZnxDft].
/// See the trait [MatZnxDftOps] for additional information.
pub struct MatZnxDft<B: Backend> {
    pub inner: ZnxBase,
    pub cols_in: usize,
    pub cols_out: usize,
    _marker: PhantomData<B>,
}

impl<B: Backend> GetZnxBase for MatZnxDft<B> {
    fn znx(&self) -> &ZnxBase {
        &self.inner
    }

    fn znx_mut(&mut self) -> &mut ZnxBase {
        &mut self.inner
    }
}

impl<B: Backend> ZnxInfos for MatZnxDft<B> {}

impl ZnxSliceSize for MatZnxDft<FFT64> {
    fn sl(&self) -> usize {
        self.n()
    }
}

impl ZnxLayout for MatZnxDft<FFT64> {
    type Scalar = f64;
}

impl<B: Backend> MatZnxDft<B> {
    pub fn new(module: &Module<B>, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> Self {
        let bytes: Vec<u8> = alloc_aligned(Self::bytes_of(module, rows, cols_in, cols_out, size));
        Self::from_bytes(module, rows, cols_in, cols_out, size, bytes)
    }

    pub fn from_bytes(module: &Module<B>, rows: usize, cols_in: usize, cols_out: usize, size: usize, mut bytes: Vec<u8>) -> Self {
        let mut mat: MatZnxDft<B> = Self::from_bytes_borrow(module, rows, cols_in, cols_out, size, &mut bytes);
        mat.znx_mut().data = bytes;
        mat
    }

    pub fn from_bytes_borrow(
        module: &Module<B>,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
        bytes: &mut [u8],
    ) -> Self {
        debug_assert_eq!(
            bytes.len(),
            Self::bytes_of(module, rows, cols_in, cols_out, size)
        );
        Self {
            inner: ZnxBase::from_bytes_borrow(module.n(), rows, cols_out, size, bytes),
            cols_in: cols_in,
            cols_out: cols_out,
            _marker: PhantomData,
        }
    }

    pub fn bytes_of(module: &Module<B>, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        unsafe {
            crate::ffi::vmp::bytes_of_vmp_pmat(
                module.ptr,
                (rows * cols_in) as u64,
                (size * cols_out) as u64,
            ) as usize
        }
    }

    pub fn cols_in(&self) -> usize {
        self.cols_in
    }

    pub fn cols_out(&self) -> usize {
        self.cols_out
    }
}

impl MatZnxDft<FFT64> {
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
