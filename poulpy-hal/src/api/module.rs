use crate::layouts::Backend;

/// Instantiate a new [crate::layouts::Module].
pub trait ModuleNew<B: Backend> {
    fn new(n: u64) -> Self;
}

/// Query the ring degree `N` of a [`Module`](crate::layouts::Module).
pub trait ModuleN {
    fn n(&self) -> usize;
}

/// Query `log2(N)` with a default implementation derived from [`ModuleN::n`].
pub trait ModuleLogN
where
    Self: ModuleN,
{
    fn log_n(&self) -> usize {
        (u64::BITS - (self.n() as u64 - 1).leading_zeros()) as usize
    }
}
