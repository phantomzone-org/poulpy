use poulpy_hal::{
    layouts::{Data, DataMut, DataRef, FillUniform, ReaderFrom, WriterTo},
    source::Source,
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWE, GGLWEInfos, GGLWEToMut, GGLWEToRef, GLWE, GLWEInfos, LWEInfos, Rank, TorusPrecision,
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use std::fmt;

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct GLWEAutomorphismKeyLayout {
    pub n: Degree,
    pub base2k: Base2K,
    pub k: TorusPrecision,
    pub rank: Rank,
    pub dnum: Dnum,
    pub dsize: Dsize,
}

#[derive(PartialEq, Eq, Clone)]
pub struct GLWEAutomorphismKey<D: Data> {
    pub(crate) key: GGLWE<D>,
    pub(crate) p: i64,
}

pub trait GetGaloisElement {
    fn p(&self) -> i64;
}

pub trait SetGaloisElement {
    fn set_p(&mut self, p: i64);
}

impl<D: DataMut> SetGaloisElement for GLWEAutomorphismKey<D> {
    fn set_p(&mut self, p: i64) {
        self.p = p
    }
}

impl<D: DataRef> GetGaloisElement for GLWEAutomorphismKey<D> {
    fn p(&self) -> i64 {
        self.p
    }
}

impl<D: Data> GLWEAutomorphismKey<D> {
    pub fn p(&self) -> i64 {
        self.p
    }
}

impl<D: Data> LWEInfos for GLWEAutomorphismKey<D> {
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

impl<D: Data> GLWEInfos for GLWEAutomorphismKey<D> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data> GGLWEInfos for GLWEAutomorphismKey<D> {
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

impl LWEInfos for GLWEAutomorphismKeyLayout {
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

impl GLWEInfos for GLWEAutomorphismKeyLayout {
    fn rank(&self) -> Rank {
        self.rank
    }
}

impl GGLWEInfos for GLWEAutomorphismKeyLayout {
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

impl<D: DataRef> fmt::Debug for GLWEAutomorphismKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: DataMut> FillUniform for GLWEAutomorphismKey<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.key.fill_uniform(log_bound, source);
    }
}

impl<D: DataRef> fmt::Display for GLWEAutomorphismKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(AutomorphismKey: p={}) {}", self.p, self.key)
    }
}

impl GLWEAutomorphismKey<Vec<u8>> {
    pub fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GGLWEInfos,
    {
        Self::alloc(
            infos.n(),
            infos.base2k(),
            infos.k(),
            infos.rank(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    pub fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> Self {
        GLWEAutomorphismKey {
            key: GGLWE::alloc(n, base2k, k, rank, rank, dnum, dsize),
            p: 0,
        }
    }

    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for AutomorphismKey"
        );
        Self::bytes_of(
            infos.n(),
            infos.base2k(),
            infos.k(),
            infos.rank(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    pub fn bytes_of(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> usize {
        GGLWE::bytes_of(n, base2k, k, rank, rank, dnum, dsize)
    }
}

impl<D: DataMut> GGLWEToMut for GLWEAutomorphismKey<D> {
    fn to_mut(&mut self) -> GGLWE<&mut [u8]> {
        self.key.to_mut()
    }
}

impl<D: DataMut> GGLWEToRef for GLWEAutomorphismKey<D> {
    fn to_ref(&self) -> GGLWE<&[u8]> {
        self.key.to_ref()
    }
}

impl<D: DataRef> GLWEAutomorphismKey<D> {
    pub fn at(&self, row: usize, col: usize) -> GLWE<&[u8]> {
        self.key.at(row, col)
    }
}

impl<D: DataMut> GLWEAutomorphismKey<D> {
    pub fn at_mut(&mut self, row: usize, col: usize) -> GLWE<&mut [u8]> {
        self.key.at_mut(row, col)
    }
}

impl<D: DataMut> ReaderFrom for GLWEAutomorphismKey<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.p = reader.read_u64::<LittleEndian>()? as i64;
        self.key.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GLWEAutomorphismKey<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.p as u64)?;
        self.key.write_to(writer)
    }
}
