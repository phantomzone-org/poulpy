use backend::hal::{
    api::{VecZnxAlloc, VecZnxAllocBytes},
    layouts::{Backend, Module, VecZnx, VecZnxToMut, VecZnxToRef},
};

use crate::{GLWEOps, Infos, SetMetaData};

pub struct GLWECiphertext<D> {
    pub data: VecZnx<D>,
    pub basek: usize,
    pub k: usize,
}

impl GLWECiphertext<Vec<u8>> {
    pub fn alloc<B: Backend>(module: &Module<B>, basek: usize, k: usize, rank: usize) -> Self {
        Self {
            data: module.vec_znx_alloc(rank + 1, k.div_ceil(basek)),
            basek,
            k,
        }
    }

    pub fn bytes_of<B: Backend>(module: &Module<B>, basek: usize, k: usize, rank: usize) -> usize {
        module.vec_znx_alloc_bytes(rank + 1, k.div_ceil(basek))
    }
}

impl<D> Infos for GLWECiphertext<D> {
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

impl<D> GLWECiphertext<D> {
    pub fn rank(&self) -> usize {
        self.cols() - 1
    }
}

impl<DataSelf: AsRef<[u8]>> GLWECiphertext<DataSelf> {
    pub fn clone(&self) -> GLWECiphertext<Vec<u8>> {
        GLWECiphertext {
            data: self.data.clone(),
            basek: self.basek(),
            k: self.k(),
        }
    }
}

impl<DataSelf: AsMut<[u8]> + AsRef<[u8]>> SetMetaData for GLWECiphertext<DataSelf> {
    fn set_k(&mut self, k: usize) {
        self.k = k
    }

    fn set_basek(&mut self, basek: usize) {
        self.basek = basek
    }
}

pub trait GLWECiphertextToRef: Infos {
    fn to_ref(&self) -> GLWECiphertext<&[u8]>;
}

impl<D: AsRef<[u8]>> GLWECiphertextToRef for GLWECiphertext<D> {
    fn to_ref(&self) -> GLWECiphertext<&[u8]> {
        GLWECiphertext {
            data: self.data.to_ref(),
            basek: self.basek,
            k: self.k,
        }
    }
}

pub trait GLWECiphertextToMut: Infos {
    fn to_mut(&mut self) -> GLWECiphertext<&mut [u8]>;
}

impl<D: AsMut<[u8]> + AsRef<[u8]>> GLWECiphertextToMut for GLWECiphertext<D> {
    fn to_mut(&mut self) -> GLWECiphertext<&mut [u8]> {
        GLWECiphertext {
            data: self.data.to_mut(),
            basek: self.basek,
            k: self.k,
        }
    }
}

impl<D> GLWEOps for GLWECiphertext<D>
where
    D: AsRef<[u8]> + AsMut<[u8]>,
    GLWECiphertext<D>: GLWECiphertextToMut + Infos + SetMetaData,
{
}
