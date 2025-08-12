use crate::hal::layouts::{Backend, ScalarZnxOwned};

pub unsafe trait ScalarZnxFromBytesImpl<B: Backend> {
    fn scalar_znx_from_bytes_impl(n: usize, cols: usize, bytes: Vec<u8>) -> ScalarZnxOwned;
}

pub unsafe trait ScalarZnxAllocBytesImpl<B: Backend> {
    fn scalar_znx_alloc_bytes_impl(n: usize, cols: usize) -> usize;
}

pub unsafe trait ScalarZnxAllocImpl<B: Backend> {
    fn scalar_znx_alloc_impl(n: usize, cols: usize) -> ScalarZnxOwned;
}
