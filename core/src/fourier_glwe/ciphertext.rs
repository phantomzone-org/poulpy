use backend::{Backend, Module, VecZnxDft, VecZnxDftAlloc};

use crate::Infos;

pub struct FourierGLWECiphertext<C, B: Backend> {
    pub data: VecZnxDft<C, B>,
    pub basek: usize,
    pub k: usize,
}

impl<B: Backend> FourierGLWECiphertext<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, basek: usize, k: usize, rank: usize) -> Self {
        Self {
            data: module.new_vec_znx_dft(rank + 1, k.div_ceil(basek)),
            basek: basek,
            k: k,
        }
    }

    pub fn bytes_of(module: &Module<B>, basek: usize, k: usize, rank: usize) -> usize {
        module.bytes_of_vec_znx_dft(rank + 1, k.div_ceil(basek))
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
