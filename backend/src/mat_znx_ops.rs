use crate::{Backend, MatZnxOwned, Module};

pub trait MatZnxAlloc {
    fn new_mat_znx(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> MatZnxOwned;

    fn bytes_of_mat_znx(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize;

    fn new_mat_znx_from_bytes(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize, bytes: Vec<u8>) -> MatZnxOwned;
}

impl<B: Backend> MatZnxAlloc for Module<B> {
    fn bytes_of_mat_znx(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        MatZnxOwned::bytes_of(self.n(), rows, cols_in, cols_out, size)
    }

    fn new_mat_znx(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> MatZnxOwned {
        MatZnxOwned::new(self.n(), rows, cols_in, cols_out, size)
    }

    fn new_mat_znx_from_bytes(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize, bytes: Vec<u8>) -> MatZnxOwned {
        MatZnxOwned::new_from_bytes(self.n(), rows, cols_in, cols_out, size, bytes)
    }
}

pub trait MatZnxScratch {}

/// This trait implements methods for vector matrix product,
/// that is, multiplying a [VecZnx] with a [MatZnx].
pub trait MatZnxOps {}

impl<BACKEND: Backend> MatZnxScratch for Module<BACKEND> {}

impl<BACKEND: Backend> MatZnxOps for Module<BACKEND> {}
