use crate::layouts::Backend;

/// Instantiate a new [crate::layouts::Module].
pub trait ModuleNew<B: Backend> {
    fn new(n: u64) -> Self;
}
