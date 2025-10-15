use poulpy_hal::{
    layouts::{Backend, Data, DataMut, DataRef, FillUniform, Module, ReaderFrom, WriterTo},
    source::Source,
};

use crate::layouts::{
    Base2K, Dnum, Dsize, GGLWE, GGLWEAlloc, GGLWEInfos, GGLWEToMut, GGLWEToRef, GLWE, GLWEInfos, LWEInfos, Rank, RingDegree,
    TorusPrecision,
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use std::fmt;

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct GLWESwitchingKeyLayout {
    pub n: RingDegree,
    pub base2k: Base2K,
    pub k: TorusPrecision,
    pub rank_in: Rank,
    pub rank_out: Rank,
    pub dnum: Dnum,
    pub dsize: Dsize,
}

impl LWEInfos for GLWESwitchingKeyLayout {
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

impl GLWEInfos for GLWESwitchingKeyLayout {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl GGLWEInfos for GLWESwitchingKeyLayout {
    fn rank_in(&self) -> Rank {
        self.rank_in
    }

    fn rank_out(&self) -> Rank {
        self.rank_out
    }

    fn dsize(&self) -> Dsize {
        self.dsize
    }

    fn dnum(&self) -> Dnum {
        self.dnum
    }
}

#[derive(PartialEq, Eq, Clone)]
pub struct GLWESwitchingKey<D: Data> {
    pub(crate) key: GGLWE<D>,
    pub(crate) sk_in_n: usize,  // Degree of sk_in
    pub(crate) sk_out_n: usize, // Degree of sk_out
}

pub(crate) trait GLWESwitchingKeySetMetaData {
    fn set_sk_in_n(&mut self, sk_in_n: usize);
    fn set_sk_out_n(&mut self, sk_out_n: usize);
}

impl<D: DataMut> GLWESwitchingKeySetMetaData for GLWESwitchingKey<D> {
    fn set_sk_in_n(&mut self, sk_in_n: usize) {
        self.sk_in_n = sk_in_n
    }

    fn set_sk_out_n(&mut self, sk_out_n: usize) {
        self.sk_out_n = sk_out_n
    }
}

pub(crate) trait GLWESwtichingKeyGetMetaData {
    fn sk_in_n(&self) -> usize;
    fn sk_out_n(&self) -> usize;
}

impl<D: DataRef> GLWESwtichingKeyGetMetaData for GLWESwitchingKey<D> {
    fn sk_in_n(&self) -> usize {
        self.sk_in_n
    }

    fn sk_out_n(&self) -> usize {
        self.sk_out_n
    }
}

impl<D: Data> LWEInfos for GLWESwitchingKey<D> {
    fn n(&self) -> RingDegree {
        self.key.n()
    }

    fn base2k(&self) -> Base2K {
        self.key.base2k()
    }

    fn k(&self) -> TorusPrecision {
        self.key.k()
    }

    fn size(&self) -> usize {
        self.key.size()
    }
}

impl<D: Data> GLWEInfos for GLWESwitchingKey<D> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data> GGLWEInfos for GLWESwitchingKey<D> {
    fn rank_in(&self) -> Rank {
        self.key.rank_in()
    }

    fn rank_out(&self) -> Rank {
        self.key.rank_out()
    }

    fn dsize(&self) -> Dsize {
        self.key.dsize()
    }

    fn dnum(&self) -> Dnum {
        self.key.dnum()
    }
}

impl<D: DataRef> fmt::Debug for GLWESwitchingKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: DataRef> fmt::Display for GLWESwitchingKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "(GLWESwitchingKey: sk_in_n={} sk_out_n={}) {}",
            self.sk_in_n,
            self.sk_out_n,
            self.key.data()
        )
    }
}

impl<D: DataMut> FillUniform for GLWESwitchingKey<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.key.fill_uniform(log_bound, source);
    }
}

pub trait GLWESwitchingKeyAlloc
where
    Self: GGLWEAlloc,
{
    fn alloc_glwe_switching_key(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> GLWESwitchingKey<Vec<u8>> {
        GLWESwitchingKey {
            key: self.alloc_gglwe(base2k, k, rank_in, rank_out, dnum, dsize),
            sk_in_n: 0,
            sk_out_n: 0,
        }
    }

    fn alloc_glwe_switching_key_from_infos<A>(&self, infos: &A) -> GLWESwitchingKey<Vec<u8>>
    where
        A: GGLWEInfos,
    {
        self.alloc_glwe_switching_key(
            infos.base2k(),
            infos.k(),
            infos.rank_in(),
            infos.rank_out(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    fn bytes_of_glwe_switching_key(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> usize {
        self.bytes_of_gglwe(base2k, k, rank_in, rank_out, dnum, dsize)
    }

    fn bytes_of_glwe_switching_key_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        self.bytes_of_glwe_switching_key(
            infos.base2k(),
            infos.k(),
            infos.rank_in(),
            infos.rank_out(),
            infos.dnum(),
            infos.dsize(),
        )
    }
}

impl<B: Backend> GLWESwitchingKeyAlloc for Module<B> where Self: GGLWEAlloc {}

impl GLWESwitchingKey<Vec<u8>> {
    pub fn alloc_from_infos<A, M>(module: &M, infos: &A) -> Self
    where
        A: GGLWEInfos,
        M: GLWESwitchingKeyAlloc,
    {
        module.alloc_glwe_switching_key_from_infos(infos)
    }

    pub fn alloc<M>(
        module: &M,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> Self
    where
        M: GLWESwitchingKeyAlloc,
    {
        module.alloc_glwe_switching_key(base2k, k, rank_in, rank_out, dnum, dsize)
    }

    pub fn bytes_of_from_infos<A, M>(module: &M, infos: &A) -> usize
    where
        A: GGLWEInfos,
        M: GLWESwitchingKeyAlloc,
    {
        module.bytes_of_glwe_switching_key_from_infos(infos)
    }

    pub fn bytes_of<M>(
        module: &M,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> usize
    where
        M: GLWESwitchingKeyAlloc,
    {
        module.bytes_of_glwe_switching_key(base2k, k, rank_in, rank_out, dnum, dsize)
    }
}

pub trait GLWESwitchingKeyToMut {
    fn to_mut(&mut self) -> GLWESwitchingKey<&mut [u8]>;
}

impl<D: DataMut> GLWESwitchingKeyToMut for GLWESwitchingKey<D>
where
    GGLWE<D>: GGLWEToMut,
{
    fn to_mut(&mut self) -> GLWESwitchingKey<&mut [u8]> {
        GLWESwitchingKey {
            key: self.key.to_mut(),
            sk_in_n: self.sk_in_n,
            sk_out_n: self.sk_out_n,
        }
    }
}

pub trait GLWESwitchingKeyToRef {
    fn to_ref(&self) -> GLWESwitchingKey<&[u8]>;
}

impl<D: DataRef> GLWESwitchingKeyToRef for GLWESwitchingKey<D>
where
    GGLWE<D>: GGLWEToRef,
{
    fn to_ref(&self) -> GLWESwitchingKey<&[u8]> {
        GLWESwitchingKey {
            key: self.key.to_ref(),
            sk_in_n: self.sk_in_n,
            sk_out_n: self.sk_out_n,
        }
    }
}

impl<D: DataRef> GLWESwitchingKey<D> {
    pub fn at(&self, row: usize, col: usize) -> GLWE<&[u8]> {
        self.key.at(row, col)
    }
}

impl<D: DataMut> GLWESwitchingKey<D> {
    pub fn at_mut(&mut self, row: usize, col: usize) -> GLWE<&mut [u8]> {
        self.key.at_mut(row, col)
    }
}

impl<D: DataMut> ReaderFrom for GLWESwitchingKey<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.sk_in_n = reader.read_u64::<LittleEndian>()? as usize;
        self.sk_out_n = reader.read_u64::<LittleEndian>()? as usize;
        self.key.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GLWESwitchingKey<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.sk_in_n as u64)?;
        writer.write_u64::<LittleEndian>(self.sk_out_n as u64)?;
        self.key.write_to(writer)
    }
}
