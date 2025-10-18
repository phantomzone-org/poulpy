use poulpy_hal::{
    layouts::{Backend, Data, DataMut, DataRef, FillUniform, Module, ReaderFrom, WriterTo},
    source::Source,
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GLWE, GLWEInfos, GLWESwitchingKey, GLWESwitchingKeyAlloc, GLWESwitchingKeyToMut,
    GLWESwitchingKeyToRef, LWEInfos, Rank, TorusPrecision,
    prepared::{GetAutomorphismGaloisElement, SetAutomorphismGaloisElement},
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use std::fmt;

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct AutomorphismKeyLayout {
    pub n: Degree,
    pub base2k: Base2K,
    pub k: TorusPrecision,
    pub rank: Rank,
    pub dnum: Dnum,
    pub dsize: Dsize,
}

#[derive(PartialEq, Eq, Clone)]
pub struct AutomorphismKey<D: Data> {
    pub(crate) key: GLWESwitchingKey<D>,
    pub(crate) p: i64,
}

impl<D: DataMut> SetAutomorphismGaloisElement for AutomorphismKey<D> {
    fn set_p(&mut self, p: i64) {
        self.p = p
    }
}

impl<D: DataRef> GetAutomorphismGaloisElement for AutomorphismKey<D> {
    fn p(&self) -> i64 {
        self.p
    }
}

impl<D: Data> AutomorphismKey<D> {
    pub fn p(&self) -> i64 {
        self.p
    }
}

impl<D: Data> LWEInfos for AutomorphismKey<D> {
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

impl<D: Data> GLWEInfos for AutomorphismKey<D> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data> GGLWEInfos for AutomorphismKey<D> {
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

impl LWEInfos for AutomorphismKeyLayout {
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

impl GLWEInfos for AutomorphismKeyLayout {
    fn rank(&self) -> Rank {
        self.rank
    }
}

impl GGLWEInfos for AutomorphismKeyLayout {
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

impl<D: DataRef> fmt::Debug for AutomorphismKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: DataMut> FillUniform for AutomorphismKey<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.key.fill_uniform(log_bound, source);
    }
}

impl<D: DataRef> fmt::Display for AutomorphismKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(AutomorphismKey: p={}) {}", self.p, self.key)
    }
}

impl<B: Backend> AutomorphismKeyAlloc for Module<B> where Self: GLWESwitchingKeyAlloc {}

pub trait AutomorphismKeyAlloc
where
    Self: GLWESwitchingKeyAlloc,
{
    fn alloc_automorphism_key(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> AutomorphismKey<Vec<u8>> {
        AutomorphismKey {
            key: self.alloc_glwe_switching_key(base2k, k, rank, rank, dnum, dsize),
            p: 0,
        }
    }

    fn alloc_automorphism_key_from_infos<A>(&self, infos: &A) -> AutomorphismKey<Vec<u8>>
    where
        A: GGLWEInfos,
    {
        self.alloc_automorphism_key(
            infos.base2k(),
            infos.k(),
            infos.rank(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    fn bytes_of_automorphism_key(&self, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> usize {
        self.bytes_of_glwe_switching_key(base2k, k, rank, rank, dnum, dsize)
    }

    fn bytes_of_automorphism_key_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for AutomorphismKey"
        );
        self.bytes_of_automorphism_key(
            infos.base2k(),
            infos.k(),
            infos.rank(),
            infos.dnum(),
            infos.dsize(),
        )
    }
}

impl AutomorphismKey<Vec<u8>> {
    pub fn alloc_from_infos<A, M>(module: &M, infos: &A) -> Self
    where
        A: GGLWEInfos,
        M: AutomorphismKeyAlloc,
    {
        module.alloc_automorphism_key_from_infos(infos)
    }

    pub fn alloc_with<M>(module: &M, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> Self
    where
        M: AutomorphismKeyAlloc,
    {
        module.alloc_automorphism_key(base2k, k, rank, dnum, dsize)
    }

    pub fn bytes_of_from_infos<A, M>(module: &M, infos: &A) -> usize
    where
        A: GGLWEInfos,
        M: AutomorphismKeyAlloc,
    {
        module.bytes_of_automorphism_key_from_infos(infos)
    }

    pub fn bytes_of<M>(module: &M, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> usize
    where
        M: AutomorphismKeyAlloc,
    {
        module.bytes_of_automorphism_key(base2k, k, rank, dnum, dsize)
    }
}

pub trait AutomorphismKeyToMut {
    fn to_mut(&mut self) -> AutomorphismKey<&mut [u8]>;
}

impl<D: DataMut> AutomorphismKeyToMut for AutomorphismKey<D>
where
    GLWESwitchingKey<D>: GLWESwitchingKeyToMut,
{
    fn to_mut(&mut self) -> AutomorphismKey<&mut [u8]> {
        AutomorphismKey {
            key: self.key.to_mut(),
            p: self.p,
        }
    }
}

pub trait AutomorphismKeyToRef {
    fn to_ref(&self) -> AutomorphismKey<&[u8]>;
}

impl<D: DataRef> AutomorphismKeyToRef for AutomorphismKey<D>
where
    GLWESwitchingKey<D>: GLWESwitchingKeyToRef,
{
    fn to_ref(&self) -> AutomorphismKey<&[u8]> {
        AutomorphismKey {
            p: self.p,
            key: self.key.to_ref(),
        }
    }
}

impl<D: DataRef> AutomorphismKey<D> {
    pub fn at(&self, row: usize, col: usize) -> GLWE<&[u8]> {
        self.key.at(row, col)
    }
}

impl<D: DataMut> AutomorphismKey<D> {
    pub fn at_mut(&mut self, row: usize, col: usize) -> GLWE<&mut [u8]> {
        self.key.at_mut(row, col)
    }
}

impl<D: DataMut> ReaderFrom for AutomorphismKey<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.p = reader.read_u64::<LittleEndian>()? as i64;
        self.key.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for AutomorphismKey<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.p as u64)?;
        self.key.write_to(writer)
    }
}
