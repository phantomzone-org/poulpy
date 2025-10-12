use poulpy_hal::{
    layouts::{
        Backend, Data, DataMut, DataRef, FillUniform, Module, ReaderFrom, ToOwnedDeep, VecZnx, VecZnxToMut, VecZnxToRef,
        WriterTo, ZnxInfos,
    },
    source::Source,
};

use crate::layouts::{Base2K, GetRingDegree, LWEInfos, Rank, RingDegree, TorusPrecision};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fmt;

pub trait GLWEInfos
where
    Self: LWEInfos,
{
    fn rank(&self) -> Rank;
    fn glwe_layout(&self) -> GLWELayout {
        GLWELayout {
            n: self.n(),
            base2k: self.base2k(),
            k: self.k(),
            rank: self.rank(),
        }
    }
}

pub trait SetGLWEInfos {
    fn set_k(&mut self, k: TorusPrecision);
    fn set_base2k(&mut self, base2k: Base2K);
}

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct GLWELayout {
    pub n: RingDegree,
    pub base2k: Base2K,
    pub k: TorusPrecision,
    pub rank: Rank,
}

impl LWEInfos for GLWELayout {
    fn n(&self) -> RingDegree {
        self.n
    }

    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }
}

impl GLWEInfos for GLWELayout {
    fn rank(&self) -> Rank {
        self.rank
    }
}

#[derive(PartialEq, Eq, Clone)]
pub struct GLWE<D: Data> {
    pub(crate) data: VecZnx<D>,
    pub(crate) base2k: Base2K,
    pub(crate) k: TorusPrecision,
}

impl<D: DataMut> SetGLWEInfos for GLWE<D> {
    fn set_base2k(&mut self, base2k: Base2K) {
        self.base2k = base2k
    }

    fn set_k(&mut self, k: TorusPrecision) {
        self.k = k
    }
}

impl<D: DataRef> GLWE<D> {
    pub fn data(&self) -> &VecZnx<D> {
        &self.data
    }
}

impl<D: DataMut> GLWE<D> {
    pub fn data_mut(&mut self) -> &mut VecZnx<D> {
        &mut self.data
    }
}

impl<D: Data> LWEInfos for GLWE<D> {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }

    fn n(&self) -> RingDegree {
        RingDegree(self.data.n() as u32)
    }

    fn size(&self) -> usize {
        self.data.size()
    }
}

impl<D: Data> GLWEInfos for GLWE<D> {
    fn rank(&self) -> Rank {
        Rank(self.data.cols() as u32 - 1)
    }
}

impl<D: DataRef> ToOwnedDeep for GLWE<D> {
    type Owned = GLWE<Vec<u8>>;
    fn to_owned_deep(&self) -> Self::Owned {
        GLWE {
            data: self.data.to_owned_deep(),
            k: self.k,
            base2k: self.base2k,
        }
    }
}

impl<D: DataRef> fmt::Debug for GLWE<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: DataRef> fmt::Display for GLWE<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GLWE: base2k={} k={}: {}",
            self.base2k().0,
            self.k().0,
            self.data
        )
    }
}

impl<D: DataMut> FillUniform for GLWE<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.data.fill_uniform(log_bound, source);
    }
}

pub trait GLWEAlloc
where
    Self: GetRingDegree,
{
    fn alloc_glwe(&self, base2k: Base2K, k: TorusPrecision, rank: Rank) -> GLWE<Vec<u8>> {
        GLWE {
            data: VecZnx::alloc(
                self.ring_degree().into(),
                (rank + 1).into(),
                k.0.div_ceil(base2k.0) as usize,
            ),
            base2k,
            k,
        }
    }

    fn alloc_glwe_from_infos<A>(&self, infos: &A) -> GLWE<Vec<u8>>
    where
        A: GLWEInfos,
    {
        self.alloc_glwe(infos.base2k(), infos.k(), infos.rank())
    }

    fn bytes_of_glwe(&self, base2k: Base2K, k: TorusPrecision, rank: Rank) -> usize {
        VecZnx::bytes_of(
            self.ring_degree().into(),
            (rank + 1).into(),
            k.0.div_ceil(base2k.0) as usize,
        )
    }

    fn bytes_of_glwe_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        self.bytes_of_glwe(infos.base2k(), infos.k(), infos.rank())
    }
}

impl<B: Backend> GLWEAlloc for Module<B> where Self: GetRingDegree {}

impl GLWE<Vec<u8>> {
    pub fn alloc_from_infos<A, M>(module: &M, infos: &A) -> Self
    where
        A: GLWEInfos,
        M: GLWEAlloc,
    {
        module.alloc_glwe_from_infos(infos)
    }

    pub fn alloc<M>(module: &M, base2k: Base2K, k: TorusPrecision, rank: Rank) -> Self
    where
        M: GLWEAlloc,
    {
        module.alloc_glwe(base2k, k, rank)
    }

    pub fn bytes_of_from_infos<A, M>(module: &M, infos: &A) -> usize
    where
        A: GLWEInfos,
        M: GLWEAlloc,
    {
        module.bytes_of_glwe_from_infos(infos)
    }

    pub fn bytes_of<M>(module: &M, base2k: Base2K, k: TorusPrecision, rank: Rank) -> usize
    where
        M: GLWEAlloc,
    {
        module.bytes_of_glwe(base2k, k, rank)
    }
}

impl<D: DataMut> ReaderFrom for GLWE<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.k = TorusPrecision(reader.read_u32::<LittleEndian>()?);
        self.base2k = Base2K(reader.read_u32::<LittleEndian>()?);
        self.data.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GLWE<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u32::<LittleEndian>(self.k.0)?;
        writer.write_u32::<LittleEndian>(self.base2k.0)?;
        self.data.write_to(writer)
    }
}

pub trait GLWEToRef {
    fn to_ref(&self) -> GLWE<&[u8]>;
}

impl<D: DataRef> GLWEToRef for GLWE<D> {
    fn to_ref(&self) -> GLWE<&[u8]> {
        GLWE {
            k: self.k,
            base2k: self.base2k,
            data: self.data.to_ref(),
        }
    }
}

pub trait GLWEToMut {
    fn to_mut(&mut self) -> GLWE<&mut [u8]>;
}

impl<D: DataMut> GLWEToMut for GLWE<D> {
    fn to_mut(&mut self) -> GLWE<&mut [u8]> {
        GLWE {
            k: self.k,
            base2k: self.base2k,
            data: self.data.to_mut(),
        }
    }
}
