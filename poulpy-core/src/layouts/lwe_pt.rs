use std::fmt;

use poulpy_hal::layouts::{Backend, Data, DataMut, DataRef, Module, Zn, ZnToMut, ZnToRef, ZnxInfos};

use crate::layouts::{Base2K, Degree, LWEInfos, TorusPrecision};

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct LWEPlaintextLayout {
    k: TorusPrecision,
    base2k: Base2K,
}

impl LWEInfos for LWEPlaintextLayout {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }

    fn n(&self) -> Degree {
        Degree(0)
    }

    fn size(&self) -> usize {
        self.k.0.div_ceil(self.base2k.0) as usize
    }
}

pub struct LWEPlaintext<D: Data> {
    pub(crate) data: Zn<D>,
    pub(crate) k: TorusPrecision,
    pub(crate) base2k: Base2K,
}

impl<D: Data> LWEInfos for LWEPlaintext<D> {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }

    fn n(&self) -> Degree {
        Degree(self.data.n() as u32 - 1)
    }

    fn size(&self) -> usize {
        self.data.size()
    }
}

pub trait LWEPlaintextAlloc {
    fn alloc_lwe_plaintext(&self, base2k: Base2K, k: TorusPrecision) -> LWEPlaintext<Vec<u8>> {
        LWEPlaintext {
            data: Zn::alloc(1, 1, k.0.div_ceil(base2k.0) as usize),
            k,
            base2k,
        }
    }

    fn alloc_lwe_plaintext_from_infos<A>(&self, infos: &A) -> LWEPlaintext<Vec<u8>>
    where
        A: LWEInfos,
    {
        self.alloc_lwe_plaintext(infos.base2k(), infos.k())
    }
}

impl<B: Backend> LWEPlaintextAlloc for Module<B> {}

impl LWEPlaintext<Vec<u8>> {
    pub fn alloc_from_infos<A, M>(module: &M, infos: &A) -> Self
    where
        A: LWEInfos,
        M: LWEPlaintextAlloc,
    {
        module.alloc_lwe_plaintext_from_infos(infos)
    }

    pub fn alloc<M>(module: &M, base2k: Base2K, k: TorusPrecision) -> Self
    where
        M: LWEPlaintextAlloc,
    {
        module.alloc_lwe_plaintext(base2k, k)
    }
}

impl<D: DataRef> fmt::Display for LWEPlaintext<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LWEPlaintext: base2k={} k={}: {}",
            self.base2k().0,
            self.k().0,
            self.data
        )
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
            base2k: self.base2k,
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
            base2k: self.base2k,
            k: self.k,
        }
    }
}

impl<D: DataMut> LWEPlaintext<D> {
    pub fn data_mut(&mut self) -> &mut Zn<D> {
        &mut self.data
    }
}
