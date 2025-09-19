use poulpy_hal::{
    api::{VecZnxDftAlloc, VecZnxDftAllocBytes, VecZnxDftApply},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch, VecZnxDft},
};

use crate::{
    dist::Distribution,
    layouts::{
        prepared::{Prepare, PrepareAlloc}, GLWEMetadata, GLWEPublicKey, Infos
    },
};

#[derive(PartialEq, Eq)]
pub struct GLWEPublicKeyPrepared<D: Data, B: Backend> {
    pub(crate) data: VecZnxDft<D, B>,
    pub(crate) metadata: GLWEMetadata,
    pub(crate) dist: Distribution,
}

impl<D: Data, B: Backend> GLWEPublicKeyPrepared<D, B>{
    pub fn metadata(&self) -> GLWEMetadata{
        self.metadata
    }
}

impl<D: Data, B: Backend> Infos for GLWEPublicKeyPrepared<D, B> {
    type Inner = VecZnxDft<D, B>;

    fn inner(&self) -> &Self::Inner {
        &self.data
    }

    fn basek(&self) -> usize {
        self.metadata.basek
    }

    fn k(&self) -> usize {
        self.metadata.k
    }
}

impl<D: Data, B: Backend> GLWEPublicKeyPrepared<D, B> {
    pub fn rank(&self) -> usize {
        self.cols() - 1
    }
}

impl<B: Backend> GLWEPublicKeyPrepared<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, metadata: GLWEMetadata) -> Self
    where
        Module<B>: VecZnxDftAlloc<B>,
    {
        Self {
            data: module.vec_znx_dft_alloc(metadata.rank + 1, metadata.k.div_ceil(metadata.basek)),
            metadata,
            dist: Distribution::NONE,
        }
    }

    pub fn bytes_of(module: &Module<B>, metadata: GLWEMetadata) -> usize
    where
        Module<B>: VecZnxDftAllocBytes,
    {
        module.vec_znx_dft_alloc_bytes(metadata.rank + 1, metadata.k.div_ceil(metadata.basek))
    }
}

impl<D: DataRef, B: Backend> PrepareAlloc<B, GLWEPublicKeyPrepared<Vec<u8>, B>> for GLWEPublicKey<D>
where
    Module<B>: VecZnxDftAlloc<B> + VecZnxDftApply<B>,
{
    fn prepare_alloc(&self, module: &Module<B>, scratch: &mut Scratch<B>) -> GLWEPublicKeyPrepared<Vec<u8>, B> {
        let mut pk_prepared: GLWEPublicKeyPrepared<Vec<u8>, B> =
            GLWEPublicKeyPrepared::alloc(module, self.metadata());
        pk_prepared.prepare(module, self, scratch);
        pk_prepared
    }
}

impl<DM: DataMut, DR: DataRef, B: Backend> Prepare<B, GLWEPublicKey<DR>> for GLWEPublicKeyPrepared<DM, B>
where
    Module<B>: VecZnxDftApply<B>,
{
    fn prepare(&mut self, module: &Module<B>, other: &GLWEPublicKey<DR>, _scratch: &mut Scratch<B>) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.n(), other.n());
            assert_eq!(self.size(), other.size());
        }

        (0..self.cols()).for_each(|i| {
            module.vec_znx_dft_apply(1, 0, &mut self.data, i, &other.data, i);
        });
        self.metadata = other.metadata();
        self.dist = other.dist;
    }
}
