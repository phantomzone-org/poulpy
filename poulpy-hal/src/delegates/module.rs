use crate::{
    api::{ModuleN, ModuleNew},
    layouts::{Backend, Module},
    oep::ModuleNewImpl,
};

impl<B> ModuleNew<B> for Module<B>
where
    B: Backend + ModuleNewImpl<B>,
{
    fn new(n: u64) -> Self {
        B::new_impl(n)
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
