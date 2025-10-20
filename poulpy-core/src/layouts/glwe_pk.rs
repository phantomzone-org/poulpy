use poulpy_hal::layouts::{Data, DataMut, DataRef, ReaderFrom, VecZnx, VecZnxToMut, VecZnxToRef, WriterTo, ZnxInfos};

use crate::{
    GetDistribution, GetDistributionMut,
    dist::Distribution,
    layouts::{Base2K, Degree, GLWEInfos, LWEInfos, Rank, TorusPrecision},
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

#[derive(PartialEq, Eq)]
pub struct GLWEPublicKey<D: Data> {
    pub(crate) data: VecZnx<D>,
    pub(crate) base2k: Base2K,
    pub(crate) k: TorusPrecision,
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
        self.base2k
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }

    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    fn size(&self) -> usize {
        self.data.size()
    }
}

impl<D: Data> GLWEInfos for GLWEPublicKey<D> {
    fn rank(&self) -> Rank {
        Rank(self.data.cols() as u32 - 1)
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
            data: VecZnx::alloc(n.into(), (rank + 1).into(), k.0.div_ceil(base2k.0) as usize),
            base2k,
            k,
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
        self.k = TorusPrecision(reader.read_u32::<LittleEndian>()?);
        self.base2k = Base2K(reader.read_u32::<LittleEndian>()?);
        match Distribution::read_from(reader) {
            Ok(dist) => self.dist = dist,
            Err(e) => return Err(e),
        }
        self.data.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GLWEPublicKey<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u32::<LittleEndian>(self.k.0)?;
        writer.write_u32::<LittleEndian>(self.base2k.0)?;
        match self.dist.write_to(writer) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        self.data.write_to(writer)
    }
}

pub trait GLWEPublicKeyToRef {
    fn to_ref(&self) -> GLWEPublicKey<&[u8]>;
}

impl<D: DataRef> GLWEPublicKeyToRef for GLWEPublicKey<D> {
    fn to_ref(&self) -> GLWEPublicKey<&[u8]> {
        GLWEPublicKey {
            data: self.data.to_ref(),
            base2k: self.base2k,
            k: self.k,
            dist: self.dist,
        }
    }
}

pub trait GLWEPublicKeyToMut {
    fn to_mut(&mut self) -> GLWEPublicKey<&mut [u8]>;
}

impl<D: DataMut> GLWEPublicKeyToMut for GLWEPublicKey<D> {
    fn to_mut(&mut self) -> GLWEPublicKey<&mut [u8]> {
        GLWEPublicKey {
            base2k: self.base2k,
            k: self.k,
            dist: self.dist,
            data: self.data.to_mut(),
        }
    }
}
