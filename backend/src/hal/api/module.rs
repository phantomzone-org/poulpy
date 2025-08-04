use crate::hal::layouts::{Backend, Module};

pub trait ModuleNew<B: Backend> {
    fn new(n: u64) -> Module<B>;
}
