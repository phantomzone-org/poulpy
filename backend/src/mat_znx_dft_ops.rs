use crate::{Backend, FFT64, MatZnxDft, MatZnxDftBytesOf, MatZnxDftOwned, Module};

pub trait MatZnxDftAlloc<B: Backend> {
    /// Allocates a new [MatZnxDft] with the given number of rows and columns.
    ///
    /// # Arguments
    ///
    /// * `rows`: number of rows (number of [VecZnxDft]).
    /// * `size`: number of size (number of size of each [VecZnxDft]).
    fn new_mat_znx_dft(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> MatZnxDftOwned<B>;

    fn bytes_of_mat_znx_dft(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize;

    fn new_mat_znx_dft_from_bytes(
        &self,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
        bytes: Vec<u8>,
    ) -> MatZnxDftOwned<B>;
}

impl<B: Backend> MatZnxDftAlloc<B> for Module<B>
where
    MatZnxDft<Vec<u8>, B>: MatZnxDftBytesOf<Vec<u8>, B>,
{
    fn bytes_of_mat_znx_dft(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        MatZnxDftOwned::bytes_of(self.n(), rows, cols_in, cols_out, size)
    }

    fn new_mat_znx_dft(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> MatZnxDftOwned<B> {
        MatZnxDftOwned::new(self.n(), rows, cols_in, cols_out, size)
    }

    fn new_mat_znx_dft_from_bytes(
        &self,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
        bytes: Vec<u8>,
    ) -> MatZnxDftOwned<B> {
        MatZnxDftOwned::new_from_bytes(self.n(), rows, cols_in, cols_out, size, bytes)
    }
}

pub trait MatZnxDftScratch {}

/// This trait implements methods for vector matrix product,
/// that is, multiplying a [VecZnx] with a [MatZnxDft].
pub trait MatZnxDftOps<BACKEND: Backend> {}

impl<BACKEND: Backend> MatZnxDftScratch for Module<BACKEND> {}

impl MatZnxDftOps<FFT64> for Module<FFT64> {}
