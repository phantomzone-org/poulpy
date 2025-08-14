use backend::hal::{
    api::{VecZnxDftAlloc, VecZnxDftAllocBytes, VecZnxDftFromVecZnx},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch, VecZnxDft},
};

use crate::{
    dist::Distribution,
    layouts::{GLWEPublicKey, Infos},
};

#[derive(PartialEq, Eq)]
pub struct GLWEPublicKeyExec<D: Data, B: Backend> {
    pub(crate) data: VecZnxDft<D, B>,
    pub(crate) basek: usize,
    pub(crate) k: usize,
    pub(crate) dist: Distribution,
}

impl<D: Data, B: Backend> Infos for GLWEPublicKeyExec<D, B> {
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

impl<D: Data, B: Backend> GLWEPublicKeyExec<D, B> {
    pub fn rank(&self) -> usize {
        self.cols() - 1
    }
}

impl<B: Backend> GLWEPublicKeyExec<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, n: usize, basek: usize, k: usize, rank: usize) -> Self
    where
        Module<B>: VecZnxDftAlloc<B>,
    {
        Self {
            data: module.vec_znx_dft_alloc(n, rank + 1, k.div_ceil(basek)),
            basek: basek,
            k: k,
            dist: Distribution::NONE,
        }
    }

    pub fn bytes_of(module: &Module<B>, n: usize, basek: usize, k: usize, rank: usize) -> usize
    where
        Module<B>: VecZnxDftAllocBytes,
    {
        module.vec_znx_dft_alloc_bytes(n, rank + 1, k.div_ceil(basek))
    }

    pub fn from<DataOther>(module: &Module<B>, other: &GLWEPublicKey<DataOther>, scratch: &mut Scratch<B>) -> Self
    where
        DataOther: DataRef,
        Module<B>: VecZnxDftAlloc<B> + VecZnxDftFromVecZnx<B>,
    {
        let mut pk_exec: GLWEPublicKeyExec<Vec<u8>, B> =
            GLWEPublicKeyExec::alloc(module, other.n(), other.basek(), other.k(), other.rank());
        pk_exec.prepare(module, other, scratch);
        pk_exec
    }
}

impl<D: DataMut, B: Backend> GLWEPublicKeyExec<D, B> {
    pub fn prepare<DataOther>(&mut self, module: &Module<B>, other: &GLWEPublicKey<DataOther>, _scratch: &mut Scratch<B>)
    where
        DataOther: DataRef,
        Module<B>: VecZnxDftFromVecZnx<B>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.n(), other.n());
            assert_eq!(self.size(), other.size());
        }

        (0..self.cols()).for_each(|i| {
            module.vec_znx_dft_from_vec_znx(1, 0, &mut self.data, i, &other.data, i);
        });
        self.k = other.k;
        self.basek = other.basek;
        self.dist = other.dist;
    }
}
