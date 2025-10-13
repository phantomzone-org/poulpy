use std::fmt;

use poulpy_hal::layouts::{Data, DataMut, DataRef, VecZnx, VecZnxToMut, VecZnxToRef, ZnxInfos};

use crate::layouts::{
    Base2K, Degree, GLWECiphertext, GLWECiphertextToMut, GLWECiphertextToRef, GLWEInfos, GLWELayoutSet, LWEInfos, Rank,
    TorusPrecision,
};

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct GLWEPlaintextLayout {
    pub n: Degree,
    pub base2k: Base2K,
    pub k: TorusPrecision,
}

impl LWEInfos for GLWEPlaintextLayout {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }

    fn n(&self) -> Degree {
        self.n
    }
}

impl GLWEInfos for GLWEPlaintextLayout {
    fn rank(&self) -> Rank {
        Rank(0)
    }
}

pub struct GLWEPlaintext<D: Data> {
    pub data: VecZnx<D>,
    pub base2k: Base2K,
    pub k: TorusPrecision,
}

impl<D: DataMut> GLWELayoutSet for GLWEPlaintext<D> {
    fn set_basek(&mut self, base2k: Base2K) {
        self.base2k = base2k
    }

    fn set_k(&mut self, k: TorusPrecision) {
        self.k = k
    }
}

impl<D: Data> LWEInfos for GLWEPlaintext<D> {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }

    fn size(&self) -> usize {
        self.data.size()
    }

    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }
}

impl<D: Data> GLWEInfos for GLWEPlaintext<D> {
    fn rank(&self) -> Rank {
        Rank(self.data.cols() as u32 - 1)
    }
}

impl<D: DataRef> fmt::Display for GLWEPlaintext<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GLWEPlaintext: base2k={} k={}: {}",
            self.base2k().0,
            self.k().0,
            self.data
        )
    }
}

impl GLWEPlaintext<Vec<u8>> {
    pub fn alloc<A>(infos: &A) -> Self
    where
        A: GLWEInfos,
    {
        Self::alloc_with(infos.n(), infos.base2k(), infos.k(), Rank(0))
    }

    pub fn alloc_with(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank) -> Self {
        debug_assert!(rank.0 == 0);
        Self {
            data: VecZnx::alloc(n.into(), (rank + 1).into(), k.0.div_ceil(base2k.0) as usize),
            base2k,
            k,
        }
    }

    pub fn alloc_bytes<A>(infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        Self::alloc_bytes_with(infos.n(), infos.base2k(), infos.k(), Rank(0))
    }

    pub fn alloc_bytes_with(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank) -> usize {
        debug_assert!(rank.0 == 0);
        VecZnx::alloc_bytes(n.into(), (rank + 1).into(), k.0.div_ceil(base2k.0) as usize)
    }
}

impl<D: DataRef> GLWECiphertextToRef for GLWEPlaintext<D> {
    fn to_ref(&self) -> GLWECiphertext<&[u8]> {
        GLWECiphertext {
            k: self.k,
            base2k: self.base2k,
            data: self.data.to_ref(),
        }
    }
}

impl<D: DataMut> GLWECiphertextToMut for GLWEPlaintext<D> {
    fn to_mut(&mut self) -> GLWECiphertext<&mut [u8]> {
        GLWECiphertext {
            k: self.k,
            base2k: self.base2k,
            data: self.data.to_mut(),
        }
    }
}

pub trait GLWEPlaintextToRef {
    fn to_ref(&self) -> GLWEPlaintext<&[u8]>;
}

impl<D: DataRef> GLWEPlaintextToRef for GLWEPlaintext<D> {
    fn to_ref(&self) -> GLWEPlaintext<&[u8]> {
        GLWEPlaintext {
            data: self.data.to_ref(),
            base2k: self.base2k,
            k: self.k,
        }
    }
}

pub trait GLWEPlaintextToMut {
    fn to_ref(&mut self) -> GLWEPlaintext<&mut [u8]>;
}

impl<D: DataMut> GLWEPlaintextToMut for GLWEPlaintext<D> {
    fn to_ref(&mut self) -> GLWEPlaintext<&mut [u8]> {
        GLWEPlaintext {
            base2k: self.base2k,
            k: self.k,
            data: self.data.to_mut(),
        }
    }
}
