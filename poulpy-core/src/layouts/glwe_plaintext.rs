use std::fmt;

use poulpy_hal::layouts::{Data, DataMut, DataRef, VecZnx, VecZnxToMut, VecZnxToRef, ZnxInfos};

use crate::layouts::{Base2K, Degree, GLWE, GLWEInfos, GLWEToMut, GLWEToRef, LWEInfos, Rank, SetLWEInfos, TorusPrecision};

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

    fn n(&self) -> Degree {
        self.n
    }

    fn size(&self) -> usize {
        self.k.as_usize().div_ceil(self.base2k.as_usize())
    }
}

impl GLWEInfos for GLWEPlaintextLayout {
    fn rank(&self) -> Rank {
        Rank(0)
    }
}

pub struct GLWEPlaintext<D: Data, M = ()> {
    pub data: VecZnx<D>,
    pub base2k: Base2K,
    pub meta: M,
}

impl<D: DataMut, M> SetLWEInfos for GLWEPlaintext<D, M> {
    fn set_base2k(&mut self, base2k: Base2K) {
        self.base2k = base2k
    }
}

impl<D: Data, M> LWEInfos for GLWEPlaintext<D, M> {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn size(&self) -> usize {
        self.data.size()
    }

    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }
}

impl<D: Data, M> GLWEInfos for GLWEPlaintext<D, M> {
    fn rank(&self) -> Rank {
        Rank(self.data.cols() as u32 - 1)
    }
}

impl<D: DataRef, M> fmt::Display for GLWEPlaintext<D, M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GLWEPlaintext: base2k={} k={}: {}",
            self.base2k().0,
            self.max_k().0,
            self.data
        )
    }
}

impl<M: Default> GLWEPlaintext<Vec<u8>, M> {
    pub fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GLWEInfos,
    {
        Self::alloc(infos.n(), infos.base2k(), infos.max_k())
    }

    pub fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision) -> Self {
        GLWEPlaintext {
            data: VecZnx::alloc(n.into(), 1, k.0.div_ceil(base2k.0) as usize),
            base2k,
            meta: M::default(),
        }
    }
}

impl<M> GLWEPlaintext<Vec<u8>, M> {
    pub fn alloc_with_meta(n: Degree, base2k: Base2K, k: TorusPrecision, meta: M) -> Self {
        GLWEPlaintext {
            data: VecZnx::alloc(n.into(), 1, k.0.div_ceil(base2k.0) as usize),
            base2k,
            meta,
        }
    }

    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        Self::bytes_of(infos.n(), infos.base2k(), infos.max_k())
    }

    pub fn bytes_of(n: Degree, base2k: Base2K, k: TorusPrecision) -> usize {
        VecZnx::bytes_of(n.into(), 1, k.0.div_ceil(base2k.0) as usize)
    }
}

impl<D: DataRef, M> GLWEToRef for GLWEPlaintext<D, M> {
    fn to_ref(&self) -> GLWE<&[u8]> {
        GLWE {
            base2k: self.base2k,
            data: self.data.to_ref(),
            meta: (),
        }
    }
}

impl<D: DataMut, M> GLWEToMut for GLWEPlaintext<D, M> {
    fn to_mut(&mut self) -> GLWE<&mut [u8]> {
        GLWE {
            base2k: self.base2k,
            data: self.data.to_mut(),
            meta: (),
        }
    }
}

pub trait GLWEPlaintextToRef {
    fn to_ref(&self) -> GLWEPlaintext<&[u8]>;
}

impl<D: DataRef, M> GLWEPlaintextToRef for GLWEPlaintext<D, M> {
    fn to_ref(&self) -> GLWEPlaintext<&[u8]> {
        GLWEPlaintext {
            data: self.data.to_ref(),
            base2k: self.base2k,
            meta: (),
        }
    }
}

pub trait GLWEPlaintextToMut {
    fn to_mut(&mut self) -> GLWEPlaintext<&mut [u8]>;
}

impl<D: DataMut, M> GLWEPlaintextToMut for GLWEPlaintext<D, M> {
    fn to_mut(&mut self) -> GLWEPlaintext<&mut [u8]> {
        GLWEPlaintext {
            base2k: self.base2k,
            data: self.data.to_mut(),
            meta: (),
        }
    }
}

impl<D: DataMut, M> GLWEPlaintext<D, M> {
    pub fn data_mut(&mut self) -> &mut VecZnx<D> {
        &mut self.data
    }
}

impl<D: DataRef, M> GLWEPlaintext<D, M> {
    pub fn data(&self) -> &VecZnx<D> {
        &self.data
    }
}
