use poulpy_hal::layouts::{Data, DataMut, DataRef, ReaderFrom, VecZnx, WriterTo};

use crate::{
    GetDistribution, GetDistributionMut,
    dist::Distribution,
    layouts::{Base2K, Degree, GLWE, GLWEInfos, GLWEToMut, GLWEToRef, LWEInfos, Rank, TorusPrecision},
};

#[derive(PartialEq, Eq)]
pub struct GLWEPublicKey<D: Data> {
    pub(crate) key: GLWE<D>,
    pub(crate) dist: Distribution,
}

impl<D: DataMut> GetDistributionMut for GLWEPublicKey<D> {
    fn dist_mut(&mut self) -> &mut Distribution {
        &mut self.dist
    }
}

impl<D: DataRef> GetDistribution for GLWEPublicKey<D> {
    fn dist(&self) -> &Distribution {
        &self.dist
    }
}

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct GLWEPublicKeyLayout {
    pub n: Degree,
    pub base2k: Base2K,
    pub k: TorusPrecision,
    pub rank: Rank,
}

impl<D: Data> LWEInfos for GLWEPublicKey<D> {
    fn base2k(&self) -> Base2K {
        self.key.base2k()
    }

    fn k(&self) -> TorusPrecision {
        self.key.k()
    }

    fn n(&self) -> Degree {
        self.key.n()
    }

    fn size(&self) -> usize {
        self.key.size()
    }
}

impl<D: Data> GLWEInfos for GLWEPublicKey<D> {
    fn rank(&self) -> Rank {
        self.key.rank()
    }
}

impl LWEInfos for GLWEPublicKeyLayout {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }

    fn n(&self) -> Degree {
        self.n
    }

    fn size(&self) -> usize {
        self.k.0.div_ceil(self.base2k.0) as usize
    }
}

impl GLWEInfos for GLWEPublicKeyLayout {
    fn rank(&self) -> Rank {
        self.rank
    }
}

impl GLWEPublicKey<Vec<u8>> {
    pub fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GLWEInfos,
    {
        Self::alloc(infos.n(), infos.base2k(), infos.k(), infos.rank())
    }

    pub fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank) -> Self {
        GLWEPublicKey {
            key: GLWE::alloc(n, base2k, k, rank),
            dist: Distribution::NONE,
        }
    }

    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        Self::bytes_of(infos.n(), infos.base2k(), infos.k(), infos.rank())
    }

    pub fn bytes_of(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank) -> usize {
        VecZnx::bytes_of(n.into(), (rank + 1).into(), k.0.div_ceil(base2k.0) as usize)
    }
}

impl<D: DataMut> ReaderFrom for GLWEPublicKey<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        match Distribution::read_from(reader) {
            Ok(dist) => self.dist = dist,
            Err(e) => return Err(e),
        }
        self.key.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GLWEPublicKey<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        match self.dist.write_to(writer) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        self.key.write_to(writer)
    }
}

impl<D: DataRef> GLWEToRef for GLWEPublicKey<D> {
    fn to_ref(&self) -> GLWE<&[u8]> {
        self.key.to_ref()
    }
}

impl<D: DataMut> GLWEToMut for GLWEPublicKey<D> {
    fn to_mut(&mut self) -> GLWE<&mut [u8]> {
        self.key.to_mut()
    }
}
