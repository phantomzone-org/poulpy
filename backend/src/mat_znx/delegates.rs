use crate::{
    Backend, MatZnxAlloc, MatZnxAllocBytes, MatZnxAllocBytesImpl, MatZnxAllocImpl, MatZnxFromBytes, MatZnxFromBytesImpl, Module,
};

impl<B> MatZnxAlloc for Module<B>
where
    B: Backend + MatZnxAllocImpl<B>,
{
    fn mat_znx_alloc(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> super::MatZnxOwned {
        B::mat_znx_alloc_impl(self, rows, cols_in, cols_out, size)
    }
}

impl<B> MatZnxAllocBytes for Module<B>
where
    B: Backend + MatZnxAllocBytesImpl<B>,
{
    fn mat_znx_alloc_bytes(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        B::mat_znx_alloc_bytes_impl(self, rows, cols_in, cols_out, size)
    }
}

impl<B> MatZnxFromBytes for Module<B>
where
    B: Backend + MatZnxFromBytesImpl<B>,
{
    fn mat_znx_from_bytes(
        &self,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
        bytes: Vec<u8>,
    ) -> super::MatZnxOwned {
        B::mat_znx_from_bytes_impl(self, rows, cols_in, cols_out, size, bytes)
    }
}
