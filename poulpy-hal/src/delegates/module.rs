use crate::{
    api::{ModuleN, ModuleNew},
    layouts::{Backend, Module},
    oep::HalModuleImpl,
};

impl<B> ModuleNew<B> for Module<B>
where
    B: Backend + HalModuleImpl<B>,
{
    fn new(n: u64) -> Self {
        <B as HalModuleImpl<B>>::new(n)
    }
}

impl<B> ModuleN for Module<B>
where
    B: Backend,
{
    fn n(&self) -> usize {
        self.n()
    }
}
