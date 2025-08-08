use crate::hal::layouts::{ScalarZnxOwned, ScalarZnxToMut, ScalarZnxToRef};

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

/// Applies the mapping X -> X^k to a\[a_col\] and write the result on res\[res_col\].
pub trait ScalarZnxAutomorphism {
    fn scalar_znx_automorphism<R, A>(&self, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: ScalarZnxToMut,
        A: ScalarZnxToRef;
}

/// Applies the mapping X -> X^k on res\[res_col\].
pub trait ScalarZnxAutomorphismInplace {
    fn scalar_znx_automorphism_inplace<R>(&self, k: i64, res: &mut R, res_col: usize)
    where
        R: ScalarZnxToMut;
}

/// Multiply a\[a_col\] with (X^p - 1) and write the result on res\[res_col\].
pub trait ScalarZnxMulXpMinusOne {
    fn scalar_znx_mul_xp_minus_one<R, A>(&self, p: i64, r: &mut R, r_col: usize, a: &A, a_col: usize)
    where
        R: ScalarZnxToMut,
        A: ScalarZnxToRef;
}

/// Multiply res\[res_col\] with (X^p - 1).
pub trait ScalarZnxMulXpMinusOneInplace {
    fn scalar_znx_mul_xp_minus_one_inplace<R>(&self, p: i64, res: &mut R, res_col: usize)
    where
        R: ScalarZnxToMut;
}
