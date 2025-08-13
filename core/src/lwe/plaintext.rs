use backend::hal::layouts::{Data, DataMut, DataRef, VecZnx, VecZnxToMut, VecZnxToRef};

use crate::{Infos, SetMetaData};

pub struct LWEPlaintext<D: Data> {
    pub(crate) data: VecZnx<D>,
    pub(crate) k: usize,
    pub(crate) basek: usize,
}

impl LWEPlaintext<Vec<u8>> {
    pub fn alloc(basek: usize, k: usize) -> Self {
        Self {
            data: VecZnx::alloc(1, 1, k.div_ceil(basek)),
            k: k,
            basek: basek,
        }
    }
}

impl<D: Data> Infos for LWEPlaintext<D> {
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

impl<D: DataMut> SetMetaData for LWEPlaintext<D> {
    fn set_k(&mut self, k: usize) {
        self.k = k
    }

    fn set_basek(&mut self, basek: usize) {
        self.basek = basek
    }
}

pub trait LWEPlaintextToRef {
    #[allow(dead_code)]
    fn to_ref(&self) -> LWEPlaintext<&[u8]>;
}

impl<D: DataRef> LWEPlaintextToRef for LWEPlaintext<D> {
    fn to_ref(&self) -> LWEPlaintext<&[u8]> {
        LWEPlaintext {
            data: self.data.to_ref(),
            basek: self.basek,
            k: self.k,
        }
    }
}

pub trait LWEPlaintextToMut {
    #[allow(dead_code)]
    fn to_mut(&mut self) -> LWEPlaintext<&mut [u8]>;
}

impl<D: DataMut> LWEPlaintextToMut for LWEPlaintext<D> {
    fn to_mut(&mut self) -> LWEPlaintext<&mut [u8]> {
        LWEPlaintext {
            data: self.data.to_mut(),
            basek: self.basek,
            k: self.k,
        }
    }
}
