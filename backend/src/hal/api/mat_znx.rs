use crate::hal::layouts::MatZnxOwned;

/// Allocates as [crate::hal::layouts::MatZnx].
pub trait MatZnxAlloc {
    fn mat_znx_alloc(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> MatZnxOwned;
}

/// Returns the size in bytes to allocate a [crate::hal::layouts::MatZnx].
pub trait MatZnxAllocBytes {
    fn mat_znx_alloc_bytes(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize;
}

/// Consume a vector of bytes into a [crate::hal::layouts::MatZnx].
/// User must ensure that bytes is memory aligned and that it length is equal to [MatZnxAllocBytes].
pub trait MatZnxFromBytes {
    fn mat_znx_from_bytes(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize, bytes: Vec<u8>) -> MatZnxOwned;
}
