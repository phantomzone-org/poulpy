use crate::hal::layouts::MatZnxOwned;

pub trait MatZnxAlloc {
    fn mat_znx_alloc(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> MatZnxOwned;
}

pub trait MatZnxAllocBytes {
    fn mat_znx_alloc_bytes(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize;
}

pub trait MatZnxFromBytes {
    fn mat_znx_from_bytes(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize, bytes: Vec<u8>) -> MatZnxOwned;
}
