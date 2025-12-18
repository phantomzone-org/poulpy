use poulpy_hal::{
    layouts::{Data, DataMut, DataRef, FillUniform, VecZnx, VecZnxToMut, VecZnxToRef, ZnxInfos},
    source::Source,
};

use crate::layouts::{Base2K, Degree, GLWE, GLWEInfos, GLWEToMut, GLWEToRef, LWEInfos, Rank, SetGLWEInfos, TorusPrecision};
use std::fmt;

#[derive(PartialEq, Eq, Clone)]
pub struct GLWETensor<D: Data> {
    pub(crate) data: VecZnx<D>,
    pub(crate) base2k: Base2K,
    pub(crate) rank: Rank,
    pub(crate) k: TorusPrecision,
}

impl<D: DataMut> SetGLWEInfos for GLWETensor<D> {
    fn set_base2k(&mut self, base2k: Base2K) {
        self.base2k = base2k
    }

    fn set_k(&mut self, k: TorusPrecision) {
        self.k = k
    }
}

impl<D: DataRef> GLWETensor<D> {
    pub fn data(&self) -> &VecZnx<D> {
        &self.data
    }
}

impl<D: DataMut> GLWETensor<D> {
    pub fn data_mut(&mut self) -> &mut VecZnx<D> {
        &mut self.data
    }
}

impl<D: Data> LWEInfos for GLWETensor<D> {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }

    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    fn limbs(&self) -> usize {
        self.data.size()
    }
}

impl<D: Data> GLWEInfos for GLWETensor<D> {
    fn rank(&self) -> Rank {
        self.rank
    }
}

impl<D: DataRef> fmt::Debug for GLWETensor<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: DataRef> fmt::Display for GLWETensor<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GLWETensor: base2k={} k={}: {}", self.base2k().0, self.k().0, self.data)
    }
}

impl<D: DataMut> FillUniform for GLWETensor<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.data.fill_uniform(log_bound, source);
    }
}

impl GLWETensor<Vec<u8>> {
    pub fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GLWEInfos,
    {
        Self::alloc(infos.n(), infos.base2k(), infos.k(), infos.rank())
    }

    pub fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank) -> Self {
        let cols: usize = rank.as_usize() + 1;
        let pairs: usize = (((cols + 1) * cols) >> 1).max(1);
        GLWETensor {
            data: VecZnx::alloc(n.into(), pairs, k.0.div_ceil(base2k.0) as usize),
            base2k,
            k,
            rank,
        }
    }

    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        Self::bytes_of(infos.n(), infos.base2k(), infos.k(), infos.rank())
    }

    pub fn bytes_of(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank) -> usize {
        let cols: usize = rank.as_usize() + 1;
        let pairs: usize = (((cols + 1) * cols) >> 1).max(1);
        VecZnx::bytes_of(n.into(), pairs, k.0.div_ceil(base2k.0) as usize)
    }
}

impl<D: DataRef> GLWEToRef for GLWETensor<D> {
    fn to_ref(&self) -> GLWE<&[u8]> {
        GLWE {
            k: self.k,
            base2k: self.base2k,
            data: self.data.to_ref(),
        }
    }
}

impl<D: DataMut> GLWEToMut for GLWETensor<D> {
    fn to_mut(&mut self) -> GLWE<&mut [u8]> {
        GLWE {
            k: self.k,
            base2k: self.base2k,
            data: self.data.to_mut(),
        }
    }
}
