use std::fmt;

use poulpy_backend::hal::layouts::{Data, DataMut, DataRef, VecZnx, VecZnxToMut, VecZnxToRef};

use crate::layouts::{GLWECiphertext, GLWECiphertextToMut, GLWECiphertextToRef, Infos, SetMetaData};

pub struct GLWEPlaintext<D: Data> {
    pub data: VecZnx<D>,
    pub basek: usize,
    pub k: usize,
}

impl<D: DataRef> fmt::Display for GLWEPlaintext<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GLWEPlaintext: basek={} k={}: {}",
            self.basek(),
            self.k(),
            self.data
        )
    }
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
    pub fn alloc(n: usize, basek: usize, k: usize) -> Self {
        Self {
            data: VecZnx::alloc(n, 1, k.div_ceil(basek)),
            basek,
            k,
        }
    }

    pub fn byte_of(n: usize, basek: usize, k: usize) -> usize {
        VecZnx::alloc_bytes(n, 1, k.div_ceil(basek))
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
