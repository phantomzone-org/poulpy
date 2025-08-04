use crate::hal::{
    layouts::{Backend, MatZnxOwned, Module},
    oep::{MatZnxAllocBytesImpl, MatZnxAllocImpl, MatZnxFromBytesImpl},
};

unsafe impl<B: Backend> MatZnxAllocImpl<B> for B {
    fn mat_znx_alloc_impl(module: &Module<B>, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> MatZnxOwned {
        MatZnxOwned::new(module.n(), rows, cols_in, cols_out, size)
    }
}

unsafe impl<B: Backend> MatZnxAllocBytesImpl<B> for B {
    fn mat_znx_alloc_bytes_impl(module: &Module<B>, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        MatZnxOwned::bytes_of(module.n(), rows, cols_in, cols_out, size)
    }
}

unsafe impl<B: Backend> MatZnxFromBytesImpl<B> for B {
    fn mat_znx_from_bytes_impl(
        module: &Module<B>,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
        bytes: Vec<u8>,
    ) -> MatZnxOwned {
        MatZnxOwned::new_from_bytes(module.n(), rows, cols_in, cols_out, size, bytes)
    }
}
