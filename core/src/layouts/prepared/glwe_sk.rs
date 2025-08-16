use backend::hal::{
    api::{SvpPPolAlloc, SvpPPolAllocBytes, SvpPrepare, ZnxInfos},
    layouts::{Backend, Data, DataMut, DataRef, Module, SvpPPol},
};

use crate::{
    dist::Distribution,
    layouts::{
        GLWESecret,
        prepared::{Prepare, PrepareAlloc},
    },
};

pub struct GLWESecretPrepared<D: Data, B: Backend> {
    pub(crate) data: SvpPPol<D, B>,
    pub(crate) dist: Distribution,
}

impl<B: Backend> GLWESecretPrepared<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, n: usize, rank: usize) -> Self
    where
        Module<B>: SvpPPolAlloc<B>,
    {
        Self {
            data: module.svp_ppol_alloc(n, rank),
            dist: Distribution::NONE,
        }
    }

    pub fn bytes_of(module: &Module<B>, n: usize, rank: usize) -> usize
    where
        Module<B>: SvpPPolAllocBytes,
    {
        module.svp_ppol_alloc_bytes(n, rank)
    }
}

impl<D: Data, B: Backend> GLWESecretPrepared<D, B> {
    pub fn n(&self) -> usize {
        self.data.n()
    }

    pub fn log_n(&self) -> usize {
        self.data.log_n()
    }

    pub fn rank(&self) -> usize {
        self.data.cols()
    }
}

impl<D: DataRef, B: Backend> PrepareAlloc<B, GLWESecretPrepared<Vec<u8>, B>> for GLWESecret<D>
where
    Module<B>: SvpPrepare<B> + SvpPPolAlloc<B>,
{
    fn prepare_alloc(
        &self,
        module: &Module<B>,
        scratch: &mut backend::hal::layouts::Scratch<B>,
    ) -> GLWESecretPrepared<Vec<u8>, B> {
        let mut sk_dft: GLWESecretPrepared<Vec<u8>, B> = GLWESecretPrepared::alloc(module, self.n(), self.rank());
        sk_dft.prepare(module, self, scratch);
        sk_dft
    }
}

impl<DM: DataMut, DR: DataRef, B: Backend> Prepare<B, GLWESecret<DR>> for GLWESecretPrepared<DM, B>
where
    Module<B>: SvpPrepare<B>,
{
    fn prepare(&mut self, module: &Module<B>, other: &GLWESecret<DR>, _scratch: &mut backend::hal::layouts::Scratch<B>) {
        (0..self.rank()).for_each(|i| {
            module.svp_prepare(&mut self.data, i, &other.data, i);
        });
        self.dist = other.dist
    }
}
