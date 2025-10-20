use poulpy_hal::{
    layouts::{Backend, Data, DataMut, DataRef, FillUniform, Module, ReaderFrom, WriterTo},
    source::Source,
};

use crate::layouts::{
    AutomorphismKey, Base2K, Degree, Dnum, Dsize, GGLWECompressed, GGLWECompressedSeedMut, GGLWECompressedToMut,
    GGLWECompressedToRef, GGLWEDecompress, GGLWEInfos, GGLWEToMut, GLWECompressed, GLWECompressedToMut, GLWECompressedToRef,
    GLWEDecompress, GLWEInfos, LWEInfos, Rank, TorusPrecision,
    prepared::{GetAutomorphismGaloisElement, SetAutomorphismGaloisElement},
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fmt;

#[derive(PartialEq, Eq, Clone)]
pub struct AutomorphismKeyCompressed<D: Data> {
    pub(crate) key: GGLWECompressed<D>,
    pub(crate) p: i64,
}

impl<D: Data> GetAutomorphismGaloisElement for AutomorphismKeyCompressed<D> {
    fn p(&self) -> i64 {
        self.p
    }
}

impl<D: Data> LWEInfos for AutomorphismKeyCompressed<D> {
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

impl<D: Data> GLWEInfos for AutomorphismKeyCompressed<D> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data> GGLWEInfos for AutomorphismKeyCompressed<D> {
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

impl<D: DataRef> fmt::Debug for AutomorphismKeyCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: DataMut> FillUniform for AutomorphismKeyCompressed<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.key.fill_uniform(log_bound, source);
    }
}

impl<D: DataRef> fmt::Display for AutomorphismKeyCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(AutomorphismKeyCompressed: p={}) {}", self.p, self.key)
    }
}

impl AutomorphismKeyCompressed<Vec<u8>> {
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
        AutomorphismKeyCompressed {
            key: GGLWECompressed::alloc(n, base2k, k, rank, rank, dnum, dsize),
            p: 0,
        }
    }

    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
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
        GGLWECompressed::bytes_of(n, base2k, k, rank, dnum, dsize)
    }
}

impl<D: DataMut> ReaderFrom for AutomorphismKeyCompressed<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.p = reader.read_u64::<LittleEndian>()? as i64;
        self.key.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for AutomorphismKeyCompressed<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.p as u64)?;
        self.key.write_to(writer)
    }
}

pub trait AutomorphismKeyDecompress
where
    Self: GGLWEDecompress,
{
    fn decompress_automorphism_key<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToMut + SetAutomorphismGaloisElement,
        O: GGLWECompressedToRef + GetAutomorphismGaloisElement,
    {
        self.decompress_gglwe(res, other);
        res.set_p(other.p());
    }
}

impl<B: Backend> AutomorphismKeyDecompress for Module<B> where Self: GLWEDecompress {}

impl<D: DataMut> AutomorphismKey<D>
where
    Self: SetAutomorphismGaloisElement,
{
    pub fn decompress<O, M>(&mut self, module: &M, other: &O)
    where
        O: GGLWECompressedToRef + GetAutomorphismGaloisElement,
        M: AutomorphismKeyDecompress,
    {
        module.decompress_automorphism_key(self, other);
    }
}

impl<D: DataRef> GGLWECompressedToRef for AutomorphismKeyCompressed<D> {
    fn to_ref(&self) -> GGLWECompressed<&[u8]> {
        self.key.to_ref()
    }
}

impl<D: DataMut> GGLWECompressedToMut for AutomorphismKeyCompressed<D> {
    fn to_mut(&mut self) -> GGLWECompressed<&mut [u8]> {
        self.key.to_mut()
    }
}

pub trait AutomorphismKeyCompressedToRef {
    fn to_ref(&self) -> AutomorphismKeyCompressed<&[u8]>;
}

impl<D: DataRef> AutomorphismKeyCompressedToRef for AutomorphismKeyCompressed<D>
where
    GLWECompressed<D>: GLWECompressedToRef,
{
    fn to_ref(&self) -> AutomorphismKeyCompressed<&[u8]> {
        AutomorphismKeyCompressed {
            key: self.key.to_ref(),
            p: self.p,
        }
    }
}

pub trait AutomorphismKeyCompressedToMut {
    fn to_mut(&mut self) -> AutomorphismKeyCompressed<&mut [u8]>;
}

impl<D: DataMut> AutomorphismKeyCompressedToMut for AutomorphismKeyCompressed<D>
where
    GLWECompressed<D>: GLWECompressedToMut,
{
    fn to_mut(&mut self) -> AutomorphismKeyCompressed<&mut [u8]> {
        AutomorphismKeyCompressed {
            p: self.p,
            key: self.key.to_mut(),
        }
    }
}

impl<D: DataMut> GGLWECompressedSeedMut for AutomorphismKeyCompressed<D> {
    fn seed_mut(&mut self) -> &mut Vec<[u8; 32]> {
        &mut self.key.seed
    }
}

impl<D: DataMut> SetAutomorphismGaloisElement for AutomorphismKeyCompressed<D> {
    fn set_p(&mut self, p: i64) {
        self.p = p
    }
}
