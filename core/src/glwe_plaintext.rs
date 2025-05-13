use base2k::{Backend, Module, VecZnx, VecZnxAlloc, VecZnxToMut, VecZnxToRef};

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

impl<C> VecZnxToMut for GLWEPlaintext<C>
where
    VecZnx<C>: VecZnxToMut,
{
    fn to_mut(&mut self) -> VecZnx<&mut [u8]> {
        self.data.to_mut()
    }
}

impl<C> VecZnxToRef for GLWEPlaintext<C>
where
    VecZnx<C>: VecZnxToRef,
{
    fn to_ref(&self) -> VecZnx<&[u8]> {
        self.data.to_ref()
    }
}

impl GLWEPlaintext<Vec<u8>> {
    pub fn new<B: Backend>(module: &Module<B>, base2k: usize, k: usize) -> Self {
        Self {
            data: module.new_vec_znx(1, derive_size(base2k, k)),
            basek: base2k,
            k,
        }
    }
}
