use crate::hal::{
    api::{ScalarZnxAlloc, ScalarZnxAllocBytes},
    layouts::{Backend, Module, ScalarZnxOwned},
    oep::{ScalarZnxAllocBytesImpl, ScalarZnxAllocImpl},
};

impl<B> ScalarZnxAllocBytes for Module<B>
where
    B: Backend + ScalarZnxAllocBytesImpl<B>,
{
    fn scalar_znx_alloc_bytes(&self, cols: usize) -> usize {
        B::scalar_znx_alloc_bytes_impl(self.n(), cols)
    }
}

impl<B> ScalarZnxAlloc for Module<B>
where
    B: Backend + ScalarZnxAllocImpl<B>,
{
    fn scalar_znx_alloc(&self, cols: usize) -> ScalarZnxOwned {
        B::scalar_znx_alloc_impl(self.n(), cols)
    }
}
