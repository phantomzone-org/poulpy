use crate::hal::layouts::{Backend, Module};

pub unsafe trait ModuleNewImpl<B: Backend> {
    fn new_impl(n: u64) -> Module<B>;
}
