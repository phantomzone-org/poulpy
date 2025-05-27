use backend::{Backend, Module, VecZnx, VecZnxAlloc};

use crate::{elem::Infos, utils::derive_size};

pub struct GLWEPlaintext<C> {
    pub data: VecZnx<C>,
    pub basek: usize,
    pub k: usize,
}

impl<T> Infos for GLWEPlaintext<T> {
    type Inner = VecZnx<T>;

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

impl GLWEPlaintext<Vec<u8>> {
    pub fn alloc<B: Backend>(module: &Module<B>, basek: usize, k: usize) -> Self {
        Self {
            data: module.new_vec_znx(1, derive_size(basek, k)),
            basek: basek,
            k,
        }
    }
}
