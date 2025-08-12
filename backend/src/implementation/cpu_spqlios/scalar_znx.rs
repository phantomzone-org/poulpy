use crate::{
    hal::{
        layouts::{Backend, ScalarZnxOwned},
        oep::{ScalarZnxAllocBytesImpl, ScalarZnxAllocImpl, ScalarZnxFromBytesImpl},
    },
    implementation::cpu_spqlios::CPUAVX,
};

unsafe impl<B: Backend> ScalarZnxAllocBytesImpl<B> for B
where
    B: CPUAVX,
{
    fn scalar_znx_alloc_bytes_impl(n: usize, cols: usize) -> usize {
        ScalarZnxOwned::bytes_of(n, cols)
    }
}

unsafe impl<B: Backend> ScalarZnxAllocImpl<B> for B
where
    B: CPUAVX,
{
    fn scalar_znx_alloc_impl(n: usize, cols: usize) -> ScalarZnxOwned {
        ScalarZnxOwned::alloc(n, cols)
    }
}

unsafe impl<B: Backend> ScalarZnxFromBytesImpl<B> for B
where
    B: CPUAVX,
{
    fn scalar_znx_from_bytes_impl(n: usize, cols: usize, bytes: Vec<u8>) -> ScalarZnxOwned {
        ScalarZnxOwned::from_bytes(n, cols, bytes)
    }
}
