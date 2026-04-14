use crate::{
    api::{ModuleN, ModuleNew},
    layouts::{Backend, Module},
    oep::HalImpl,
};

impl<B> ModuleNew<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn new(n: u64) -> Self {
        <B as HalImpl<B>>::new(n)
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
