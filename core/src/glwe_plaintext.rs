use backend::{Backend, FFT64, Module, VecZnx, VecZnxAlloc, VecZnxToMut, VecZnxToRef};

use crate::{GLWECiphertext, GLWECiphertextToMut, GLWECiphertextToRef, GLWEOps, Infos, SetMetaData, div_ceil};

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

impl<DataSelf: AsMut<[u8]> + AsRef<[u8]>> SetMetaData for GLWEPlaintext<DataSelf> {
    fn set_k(&mut self, k: usize) {
        self.k = k
    }

    fn set_basek(&mut self, basek: usize) {
        self.basek = basek
    }
}

impl GLWEPlaintext<Vec<u8>> {
    pub fn alloc<B: Backend>(module: &Module<B>, basek: usize, k: usize) -> Self {
        Self {
            data: module.new_vec_znx(1, div_ceil(basek, k)),
            basek: basek,
            k,
        }
    }

    pub fn byte_of(module: &Module<FFT64>, basek: usize, k: usize) -> usize {
        module.bytes_of_vec_znx(1, div_ceil(basek, k))
    }
}

impl<D: AsRef<[u8]>> GLWECiphertextToRef for GLWEPlaintext<D> {
    fn to_ref(&self) -> GLWECiphertext<&[u8]> {
        GLWECiphertext {
            data: self.data.to_ref(),
            basek: self.basek,
            k: self.k,
        }
    }
}

impl<D: AsMut<[u8]> + AsRef<[u8]>> GLWECiphertextToMut for GLWEPlaintext<D> {
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
    D: AsRef<[u8]> + AsMut<[u8]>,
    GLWEPlaintext<D>: GLWECiphertextToMut + Infos + SetMetaData,
{
}
