use backend::hal::{
    api::{SvpPPolAlloc, SvpPPolAllocBytes, SvpPrepare, ZnxInfos},
    layouts::{Backend, Data, DataMut, DataRef, Module, SvpPPol},
};

use crate::{dist::Distribution, layouts::GLWESecret, trait_families::GLWESecretExecModuleFamily};

pub struct GLWESecretExec<D: Data, B: Backend> {
    pub(crate) data: SvpPPol<D, B>,
    pub(crate) dist: Distribution,
}

impl<B: Backend> GLWESecretExec<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, n: usize, rank: usize) -> Self
    where
        Module<B>: GLWESecretExecModuleFamily<B>,
    {
        Self {
            data: module.svp_ppol_alloc(n, rank),
            dist: Distribution::NONE,
        }
    }

    pub fn bytes_of(module: &Module<B>, n: usize, rank: usize) -> usize
    where
        Module<B>: GLWESecretExecModuleFamily<B>,
    {
        module.svp_ppol_alloc_bytes(n, rank)
    }
}

impl<B: Backend> GLWESecretExec<Vec<u8>, B> {
    pub fn from<D>(module: &Module<B>, sk: &GLWESecret<D>) -> Self
    where
        D: DataRef,
        Module<B>: GLWESecretExecModuleFamily<B>,
    {
        let mut sk_dft: GLWESecretExec<Vec<u8>, B> = Self::alloc(module, sk.n(), sk.rank());
        sk_dft.prepare(module, sk);
        sk_dft
    }
}

impl<D: Data, B: Backend> GLWESecretExec<D, B> {
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

impl<D: DataMut, B: Backend> GLWESecretExec<D, B> {
    pub(crate) fn prepare<O>(&mut self, module: &Module<B>, sk: &GLWESecret<O>)
    where
        O: DataRef,
        Module<B>: GLWESecretExecModuleFamily<B>,
    {
        (0..self.rank()).for_each(|i| {
            module.svp_prepare(&mut self.data, i, &sk.data, i);
        });
        self.dist = sk.dist
    }
}
