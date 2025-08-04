use crate::hal::layouts::{ScalarZnxOwned, ScalarZnxToMut, ScalarZnxToRef};

pub trait ScalarZnxFromBytes {
    fn scalar_znx_from_bytes(&self, cols: usize, bytes: Vec<u8>) -> ScalarZnxOwned;
}

pub trait ScalarZnxAllocBytes {
    fn scalar_znx_alloc_bytes(&self, cols: usize) -> usize;
}

pub trait ScalarZnxAlloc {
    fn scalar_znx_alloc(&self, cols: usize) -> ScalarZnxOwned;
}

pub trait ScalarZnxAutomorphism {
    fn scalar_znx_automorphism<R, A>(&self, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: ScalarZnxToMut,
        A: ScalarZnxToRef;
}

pub trait ScalarZnxAutomorphismInplace {
    fn scalar_znx_automorphism_inplace<A>(&self, k: i64, a: &mut A, a_col: usize)
    where
        A: ScalarZnxToMut;
}
