use crate::hal::layouts::{Backend, Module, ScalarZnxOwned, ScalarZnxToMut, ScalarZnxToRef};

pub unsafe trait ScalarZnxFromBytesImpl<B: Backend> {
    fn scalar_znx_from_bytes_impl(n: usize, cols: usize, bytes: Vec<u8>) -> ScalarZnxOwned;
}

pub unsafe trait ScalarZnxAllocBytesImpl<B: Backend> {
    fn scalar_znx_alloc_bytes_impl(n: usize, cols: usize) -> usize;
}

pub unsafe trait ScalarZnxAllocImpl<B: Backend> {
    fn scalar_znx_alloc_impl(n: usize, cols: usize) -> ScalarZnxOwned;
}

pub unsafe trait ScalarZnxAutomorphismImpl<B: Backend> {
    fn scalar_znx_automorphism_impl<R, A>(module: &Module<B>, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: ScalarZnxToMut,
        A: ScalarZnxToRef;
}

pub unsafe trait ScalarZnxAutomorphismInplaceIml<B: Backend> {
    fn scalar_znx_automorphism_inplace_impl<A>(module: &Module<B>, k: i64, a: &mut A, a_col: usize)
    where
        A: ScalarZnxToMut;
}

pub unsafe trait ScalarZnxMulXpMinusOneImpl<B: Backend> {
    fn scalar_znx_mul_xp_minus_one_impl<R, A>(module: &Module<B>, p: i64, r: &mut R, r_col: usize, a: &A, a_col: usize)
    where
        R: ScalarZnxToMut,
        A: ScalarZnxToRef;
}

pub unsafe trait ScalarZnxMulXpMinusOneInplaceImpl<B: Backend> {
    fn scalar_znx_mul_xp_minus_one_inplace_impl<R>(module: &Module<B>, p: i64, r: &mut R, r_col: usize)
    where
        R: ScalarZnxToMut;
}
