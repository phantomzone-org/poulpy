use crate::hal::layouts::{Backend, MatZnxOwned, Module};

pub unsafe trait MatZnxAllocImpl<B: Backend> {
    fn mat_znx_alloc_impl(module: &Module<B>, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> MatZnxOwned;
}

pub unsafe trait MatZnxAllocBytesImpl<B: Backend> {
    fn mat_znx_alloc_bytes_impl(module: &Module<B>, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize;
}

pub unsafe trait MatZnxFromBytesImpl<B: Backend> {
    fn mat_znx_from_bytes_impl(
        module: &Module<B>,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
        bytes: Vec<u8>,
    ) -> MatZnxOwned;
}
