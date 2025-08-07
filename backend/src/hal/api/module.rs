use crate::hal::layouts::{Backend, Module};

/// Instantiate a new [crate::hal::layouts::Module].
pub trait ModuleNew<B: Backend> {
    fn new(n: u64) -> Module<B>;
}
