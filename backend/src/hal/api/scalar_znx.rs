use crate::hal::layouts::ScalarZnxOwned;

/// Allocates as [crate::hal::layouts::ScalarZnx].
pub trait ScalarZnxAlloc {
    fn scalar_znx_alloc(&self, cols: usize) -> ScalarZnxOwned;
}

/// Returns the size in bytes to allocate a [crate::hal::layouts::ScalarZnx].
pub trait ScalarZnxAllocBytes {
    fn scalar_znx_alloc_bytes(&self, cols: usize) -> usize;
}

/// Consume a vector of bytes into a [crate::hal::layouts::ScalarZnx].
/// User must ensure that bytes is memory aligned and that it length is equal to [ScalarZnxAllocBytes].
pub trait ScalarZnxFromBytes {
    fn scalar_znx_from_bytes(&self, cols: usize, bytes: Vec<u8>) -> ScalarZnxOwned;
}
