use backend::hal::{
    api::{VecZnxAlloc, VecZnxAllocBytes},
    layouts::{Backend, Data, DataMut, DataRef, Module, VecZnx, VecZnxToMut, VecZnxToRef},
};

use crate::{GLWECiphertext, GLWECiphertextToMut, GLWECiphertextToRef, GLWEOps, Infos, SetMetaData};

pub struct GLWEPlaintext<D: Data> {
    pub data: VecZnx<D>,
    pub basek: usize,
    pub k: usize,
}

impl<D: Data> Infos for GLWEPlaintext<D> {
    type Inner = VecZnx<D>;

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

impl<D: DataMut> SetMetaData for GLWEPlaintext<D> {
    fn set_k(&mut self, k: usize) {
        self.k = k
    }

    fn set_basek(&mut self, basek: usize) {
        self.basek = basek
    }
}

impl GLWEPlaintext<Vec<u8>> {
    pub fn alloc<B: Backend>(module: &Module<B>, basek: usize, k: usize) -> Self
    where
        Module<B>: VecZnxAlloc,
    {
        Self {
            data: module.vec_znx_alloc(1, k.div_ceil(basek)),
            basek: basek,
            k,
        }
    }

    pub fn byte_of<B: Backend>(module: &Module<B>, basek: usize, k: usize) -> usize
    where
        Module<B>: VecZnxAllocBytes,
    {
        module.vec_znx_alloc_bytes(1, k.div_ceil(basek))
    }
}

impl<D: DataRef> GLWECiphertextToRef for GLWEPlaintext<D> {
    fn to_ref(&self) -> GLWECiphertext<&[u8]> {
        GLWECiphertext {
            data: self.data.to_ref(),
            basek: self.basek,
            k: self.k,
        }
    }
}

impl<D: DataMut> GLWECiphertextToMut for GLWEPlaintext<D> {
    fn to_mut(&mut self) -> GLWECiphertext<&mut [u8]> {
        GLWECiphertext {
            data: self.data.to_mut(),
            basek: self.basek,
            k: self.k,
        }
    }
}

impl<D> GLWEOps for GLWEPlaintext<D>
where
    D: DataMut,
    GLWEPlaintext<D>: GLWECiphertextToMut + Infos + SetMetaData,
{
}
