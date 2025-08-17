use crate::hal::layouts::Backend;

/// Instantiate a new [crate::hal::layouts::Module].
pub trait ModuleNew<B: Backend> {
    fn new(n: u64) -> Self;
}
