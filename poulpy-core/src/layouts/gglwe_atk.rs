use poulpy_hal::{
    layouts::{Data, DataMut, DataRef, FillUniform, ReaderFrom, WriterTo},
    source::Source,
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GGLWESwitchingKey, GLWECiphertext, GLWEInfos, LWEInfos, Rank, TorusPrecision,
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use std::fmt;

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct GGLWEAutomorphismKeyLayout {
    pub n: Degree,
    pub base2k: Base2K,
    pub k: TorusPrecision,
    pub rank: Rank,
    pub dnum: Dnum,
    pub dsize: Dsize,
}

#[derive(PartialEq, Eq, Clone)]
pub struct GGLWEAutomorphismKey<D: Data> {
    pub(crate) key: GGLWESwitchingKey<D>,
    pub(crate) p: i64,
}

impl<D: Data> GGLWEAutomorphismKey<D> {
    pub fn p(&self) -> i64 {
        self.p
    }
}

impl<D: Data> LWEInfos for GGLWEAutomorphismKey<D> {
    fn n(&self) -> Degree {
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

impl<D: Data> GLWEInfos for GGLWEAutomorphismKey<D> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data> GGLWEInfos for GGLWEAutomorphismKey<D> {
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

impl LWEInfos for GGLWEAutomorphismKeyLayout {
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

impl GLWEInfos for GGLWEAutomorphismKeyLayout {
    fn rank(&self) -> Rank {
        self.rank
    }
}

impl GGLWEInfos for GGLWEAutomorphismKeyLayout {
    fn rank_in(&self) -> Rank {
        self.rank
    }

    fn dsize(&self) -> Dsize {
        self.dsize
    }

    fn rank_out(&self) -> Rank {
        self.rank
    }

    fn dnum(&self) -> Dnum {
        self.dnum
    }
}

impl<D: DataRef> fmt::Debug for GGLWEAutomorphismKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: DataMut> FillUniform for GGLWEAutomorphismKey<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.key.fill_uniform(log_bound, source);
    }
}

impl<D: DataRef> fmt::Display for GGLWEAutomorphismKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(AutomorphismKey: p={}) {}", self.p, self.key)
    }
}

impl GGLWEAutomorphismKey<Vec<u8>> {
    pub fn alloc<A>(infos: &A) -> Self
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWEAutomorphismKey"
        );
        GGLWEAutomorphismKey {
            key: GGLWESwitchingKey::alloc(infos),
            p: 0,
        }
    }

    pub fn alloc_with(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> Self {
        GGLWEAutomorphismKey {
            key: GGLWESwitchingKey::alloc_with(n, base2k, k, rank, rank, dnum, dsize),
            p: 0,
        }
    }

    pub fn alloc_bytes<A>(infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWEAutomorphismKey"
        );
        GGLWESwitchingKey::alloc_bytes(infos)
    }

    pub fn bytes_of(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> usize {
        GGLWESwitchingKey::alloc_bytes_with(n, base2k, k, rank, rank, dnum, dsize)
    }
}

impl<D: DataRef> GGLWEAutomorphismKey<D> {
    pub fn at(&self, row: usize, col: usize) -> GLWECiphertext<&[u8]> {
        self.key.at(row, col)
    }
}

impl<D: DataMut> GGLWEAutomorphismKey<D> {
    pub fn at_mut(&mut self, row: usize, col: usize) -> GLWECiphertext<&mut [u8]> {
        self.key.at_mut(row, col)
    }
}

impl<D: DataMut> ReaderFrom for GGLWEAutomorphismKey<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.p = reader.read_u64::<LittleEndian>()? as i64;
        self.key.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GGLWEAutomorphismKey<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.p as u64)?;
        self.key.write_to(writer)
    }
}
