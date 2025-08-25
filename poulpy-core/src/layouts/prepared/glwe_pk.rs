use poulpy_hal::{
    api::{DFT, VecZnxDftAlloc, VecZnxDftAllocBytes},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch, VecZnxDft},
};

use crate::{
    dist::Distribution,
    layouts::{
        GLWEPublicKey, Infos,
        prepared::{Prepare, PrepareAlloc},
    },
};

#[derive(PartialEq, Eq)]
pub struct GLWEPublicKeyPrepared<D: Data, B: Backend> {
    pub(crate) data: VecZnxDft<D, B>,
    pub(crate) basek: usize,
    pub(crate) k: usize,
    pub(crate) dist: Distribution,
}

impl<D: Data, B: Backend> Infos for GLWEPublicKeyPrepared<D, B> {
    type Inner = VecZnxDft<D, B>;

    fn inner(&self) -> &Self::Inner {
        &self.data
    }

    fn basek(&self) -> usize {
        self.basek
    }

    fn k(&self) -> usize {
        self.k
    }
}

impl<D: Data, B: Backend> GLWEPublicKeyPrepared<D, B> {
    pub fn rank(&self) -> usize {
        self.cols() - 1
    }
}

impl<B: Backend> GLWEPublicKeyPrepared<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, basek: usize, k: usize, rank: usize) -> Self
    where
        Module<B>: VecZnxDftAlloc<B>,
    {
        Self {
            data: module.vec_znx_dft_alloc(rank + 1, k.div_ceil(basek)),
            basek,
            k,
            dist: Distribution::NONE,
        }
    }

    pub fn bytes_of(module: &Module<B>, basek: usize, k: usize, rank: usize) -> usize
    where
        Module<B>: VecZnxDftAllocBytes,
    {
        module.vec_znx_dft_alloc_bytes(rank + 1, k.div_ceil(basek))
    }
}

impl<D: DataRef, B: Backend> PrepareAlloc<B, GLWEPublicKeyPrepared<Vec<u8>, B>> for GLWEPublicKey<D>
where
    Module<B>: VecZnxDftAlloc<B> + DFT<B>,
{
    fn prepare_alloc(&self, module: &Module<B>, scratch: &mut Scratch<B>) -> GLWEPublicKeyPrepared<Vec<u8>, B> {
        let mut pk_prepared: GLWEPublicKeyPrepared<Vec<u8>, B> =
            GLWEPublicKeyPrepared::alloc(module, self.basek(), self.k(), self.rank());
        pk_prepared.prepare(module, self, scratch);
        pk_prepared
    }
}

impl<DM: DataMut, DR: DataRef, B: Backend> Prepare<B, GLWEPublicKey<DR>> for GLWEPublicKeyPrepared<DM, B>
where
    Module<B>: DFT<B>,
{
    fn prepare(&mut self, module: &Module<B>, other: &GLWEPublicKey<DR>, _scratch: &mut Scratch<B>) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.n(), other.n());
            assert_eq!(self.size(), other.size());
        }

        (0..self.cols()).for_each(|i| {
            module.dft(1, 0, &mut self.data, i, &other.data, i);
        });
        self.k = other.k;
        self.basek = other.basek;
        self.dist = other.dist;
    }
}
