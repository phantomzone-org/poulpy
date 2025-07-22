use backend::{Backend, Module, VecZnxDft, VecZnxDftAlloc, VecZnxDftAllocBytes};

use crate::Infos;

pub struct FourierGLWECiphertext<C, B: Backend> {
    pub data: VecZnxDft<C, B>,
    pub basek: usize,
    pub k: usize,
}

impl<B: Backend> FourierGLWECiphertext<Vec<u8>, B>
where
    Module<B>: VecZnxDftAllocBytes + VecZnxDftAlloc<B>,
{
    pub fn alloc(module: &Module<B>, basek: usize, k: usize, rank: usize) -> Self {
        Self {
            data: module.vec_znx_dft_alloc(rank + 1, k.div_ceil(basek)),
            basek: basek,
            k: k,
        }
    }

    pub fn bytes_of(module: &Module<B>, basek: usize, k: usize, rank: usize) -> usize {
        module.vec_znx_dft_alloc_bytes(rank + 1, k.div_ceil(basek))
    }
}

impl<T, B: Backend> Infos for FourierGLWECiphertext<T, B> {
    type Inner = VecZnxDft<T, B>;

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

impl<T, B: Backend> FourierGLWECiphertext<T, B> {
    pub fn rank(&self) -> usize {
        self.cols() - 1
    }
}
