use backend::{VecZnx, VecZnxToMut, VecZnxToRef};

use crate::{Infos, SetMetaData};

pub struct LWEPlaintext<D> {
    pub(crate) data: VecZnx<D>,
    pub(crate) k: usize,
    pub(crate) basek: usize,
}

impl LWEPlaintext<Vec<u8>> {
    pub fn alloc(basek: usize, k: usize) -> Self {
        Self {
            data: VecZnx::new::<i64>(1, 1, k.div_ceil(basek)),
            k: k,
            basek: basek,
        }
    }
}

impl<T> Infos for LWEPlaintext<T> {
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

impl<DataSelf: AsMut<[u8]> + AsRef<[u8]>> SetMetaData for LWEPlaintext<DataSelf> {
    fn set_k(&mut self, k: usize) {
        self.k = k
    }

    fn set_basek(&mut self, basek: usize) {
        self.basek = basek
    }
}

pub trait LWEPlaintextToRef {
    fn to_ref(&self) -> LWEPlaintext<&[u8]>;
}

impl<D: AsRef<[u8]>> LWEPlaintextToRef for LWEPlaintext<D> {
    fn to_ref(&self) -> LWEPlaintext<&[u8]> {
        LWEPlaintext {
            data: self.data.to_ref(),
            basek: self.basek,
            k: self.k,
        }
    }
}

pub trait LWEPlaintextToMut {
    fn to_mut(&mut self) -> LWEPlaintext<&mut [u8]>;
}

impl<D: AsMut<[u8]> + AsRef<[u8]>> LWEPlaintextToMut for LWEPlaintext<D> {
    fn to_mut(&mut self) -> LWEPlaintext<&mut [u8]> {
        LWEPlaintext {
            data: self.data.to_mut(),
            basek: self.basek,
            k: self.k,
        }
    }
}
