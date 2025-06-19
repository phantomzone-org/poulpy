use backend::{VecZnx, VecZnxToMut, VecZnxToRef};

use crate::{Infos, SetMetaData};

pub struct LWECiphertext<D> {
    pub(crate) data: VecZnx<D>,
    pub(crate) k: usize,
    pub(crate) basek: usize,
}

impl LWECiphertext<Vec<u8>> {
    pub fn alloc(n: usize, basek: usize, k: usize) -> Self {
        Self {
            data: VecZnx::new::<i64>(n + 1, 1, k.div_ceil(basek)),
            k: k,
            basek: basek,
        }
    }
}

impl<T> Infos for LWECiphertext<T> {
    type Inner = VecZnx<T>;

    fn n(&self) -> usize {
        &self.inner().n - 1
    }

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

impl<DataSelf: AsMut<[u8]> + AsRef<[u8]>> SetMetaData for LWECiphertext<DataSelf> {
    fn set_k(&mut self, k: usize) {
        self.k = k
    }

    fn set_basek(&mut self, basek: usize) {
        self.basek = basek
    }
}

pub trait LWECiphertextToRef {
    fn to_ref(&self) -> LWECiphertext<&[u8]>;
}

impl<D: AsRef<[u8]>> LWECiphertextToRef for LWECiphertext<D> {
    fn to_ref(&self) -> LWECiphertext<&[u8]> {
        LWECiphertext {
            data: self.data.to_ref(),
            basek: self.basek,
            k: self.k,
        }
    }
}

pub trait LWECiphertextToMut {
    fn to_mut(&mut self) -> LWECiphertext<&mut [u8]>;
}

impl<D: AsMut<[u8]> + AsRef<[u8]>> LWECiphertextToMut for LWECiphertext<D> {
    fn to_mut(&mut self) -> LWECiphertext<&mut [u8]> {
        LWECiphertext {
            data: self.data.to_mut(),
            basek: self.basek,
            k: self.k,
        }
    }
}
