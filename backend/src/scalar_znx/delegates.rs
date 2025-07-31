use crate::{
    Backend, Module, ScalarZnxAlloc, ScalarZnxAllocBytes, ScalarZnxAllocBytesImpl, ScalarZnxAllocImpl, ScalarZnxAutomorphism,
    ScalarZnxAutomorphismImpl, ScalarZnxAutomorphismInplace, ScalarZnxAutomorphismInplaceIml, ScalarZnxFromBytes,
    ScalarZnxFromBytesImpl, ScalarZnxOwned, ScalarZnxToMut, ScalarZnxToRef,
};

impl<B> ScalarZnxAllocBytes for Module<B>
where
    B: Backend + ScalarZnxAllocBytesImpl<B>,
{
    fn scalar_znx_alloc_bytes(&self, cols: usize) -> usize {
        B::scalar_znx_alloc_bytes_impl(self, cols)
    }
}

impl<B> ScalarZnxAlloc for Module<B>
where
    B: Backend + ScalarZnxAllocImpl<B>,
{
    fn scalar_znx_alloc(&self, cols: usize) -> ScalarZnxOwned {
        B::scalar_znx_alloc_impl(self, cols)
    }
}

impl<B> ScalarZnxFromBytes for Module<B>
where
    B: Backend + ScalarZnxFromBytesImpl<B>,
{
    fn scalar_znx_from_bytes(&self, cols: usize, bytes: Vec<u8>) -> ScalarZnxOwned {
        B::scalar_znx_from_bytes_impl(self, cols, bytes)
    }
}

impl<B> ScalarZnxAutomorphism for Module<B>
where
    B: Backend + ScalarZnxAutomorphismImpl<B>,
{
    fn scalar_znx_automorphism<R, A>(&self, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: ScalarZnxToMut,
        A: ScalarZnxToRef,
    {
        B::scalar_znx_automorphism_impl(self, k, res, res_col, a, a_col);
    }
}

impl<B> ScalarZnxAutomorphismInplace for Module<B>
where
    B: Backend + ScalarZnxAutomorphismInplaceIml<B>,
{
    fn scalar_znx_automorphism_inplace<A>(&self, k: i64, a: &mut A, a_col: usize)
    where
        A: ScalarZnxToMut,
    {
        B::scalar_znx_automorphism_inplace_impl(self, k, a, a_col);
    }
}
