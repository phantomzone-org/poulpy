use crate::layouts::Backend;

/// Instantiate a new [crate::layouts::Module].
pub trait ModuleNew<B: Backend> {
    fn new(n: u64) -> Self;
}

pub trait ModuleN {
    fn n(&self) -> usize;
}

pub trait ModuleLogN where Self: ModuleN{
    fn log_n(&self) -> usize{
        (u64::BITS - (self.n() as u64-1).leading_zeros()) as usize
    }
}