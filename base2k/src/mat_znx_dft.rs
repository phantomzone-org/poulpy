use crate::znx_base::{GetZnxBase, ZnxAlloc, ZnxBase, ZnxInfos, ZnxLayout, ZnxSliceSize};
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

impl<B: Backend> ZnxAlloc<B> for MatZnxDft<B> {
    type Scalar = u8;

    fn from_bytes_borrow(module: &Module<B>, rows: usize, cols: usize, size: usize, bytes: &mut [u8]) -> Self {
        Self {
            inner: ZnxBase::from_bytes_borrow(module.n(), rows, cols, size, bytes),
            _marker: PhantomData,
        }
    }

    fn bytes_of(module: &Module<B>, rows: usize, cols: usize, size: usize) -> usize {
        unsafe { crate::ffi::vmp::bytes_of_vmp_pmat(module.ptr, rows as u64, size as u64) as usize * cols }
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
