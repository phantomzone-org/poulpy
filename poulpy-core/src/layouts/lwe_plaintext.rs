use std::fmt;

use poulpy_hal::layouts::{Data, DataMut, DataRef, VecZnx, VecZnxToMut, VecZnxToRef, ZnxInfos};

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
    pub(crate) data: VecZnx<D>,
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

impl LWEPlaintext<Vec<u8>> {
    pub fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: LWEInfos,
    {
        Self::alloc(infos.base2k(), infos.k())
    }

    pub fn alloc(base2k: Base2K, k: TorusPrecision) -> Self {
        LWEPlaintext {
            data: VecZnx::alloc(1, 1, k.0.div_ceil(base2k.0) as usize),
            k,
            base2k,
        }
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

impl<D: DataRef> LWEPlaintext<D> {
    pub fn data(&self) -> &VecZnx<D> {
        &self.data
    }
}

impl<D: DataMut> LWEPlaintext<D> {
    pub fn data_mut(&mut self) -> &mut VecZnx<D> {
        &mut self.data
    }
}
