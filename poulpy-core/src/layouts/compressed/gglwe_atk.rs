use poulpy_hal::{
    layouts::{Backend, Data, DataMut, DataRef, FillUniform, Module, ReaderFrom, WriterTo},
    source::Source,
};

use crate::layouts::{
    AutomorphismKey, AutomorphismKeyToMut, Base2K, Degree, Dnum, Dsize, GGLWEInfos, GLWEInfos, LWEInfos, Rank, TorusPrecision,
    compressed::{
        GLWESwitchingKeyCompressed, GLWESwitchingKeyCompressedAlloc, GLWESwitchingKeyCompressedToMut,
        GLWESwitchingKeyCompressedToRef, GLWESwitchingKeyDecompress,
    },
    prepared::{GetAutomorphismGaloisElement, SetAutomorphismGaloisElement},
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fmt;

#[derive(PartialEq, Eq, Clone)]
pub struct AutomorphismKeyCompressed<D: Data> {
    pub(crate) key: GLWESwitchingKeyCompressed<D>,
    pub(crate) p: i64,
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

pub trait AutomorphismKeyCompressedAlloc
where
    Self: GLWESwitchingKeyCompressedAlloc,
{
    fn alloc_automorphism_key_compressed(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> AutomorphismKeyCompressed<Vec<u8>> {
        AutomorphismKeyCompressed {
            key: self.alloc_glwe_switching_key_compressed(base2k, k, rank, rank, dnum, dsize),
            p: 0,
        }
    }

    fn alloc_automorphism_key_compressed_from_infos<A>(&self, infos: &A) -> AutomorphismKeyCompressed<Vec<u8>>
    where
        A: GGLWEInfos,
    {
        assert_eq!(infos.rank_in(), infos.rank_out());
        self.alloc_automorphism_key_compressed(
            infos.base2k(),
            infos.k(),
            infos.rank(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    fn bytes_of_automorphism_key_compressed(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> usize {
        self.bytes_of_glwe_switching_key_compressed(base2k, k, rank, dnum, dsize)
    }

    fn bytes_of_automorphism_key_compressed_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(infos.rank_in(), infos.rank_out());
        self.bytes_of_automorphism_key_compressed(
            infos.base2k(),
            infos.k(),
            infos.rank(),
            infos.dnum(),
            infos.dsize(),
        )
    }
}

impl AutomorphismKeyCompressed<Vec<u8>> {
    pub fn alloc_from_infos<A, B: Backend>(module: Module<B>, infos: &A) -> Self
    where
        A: GGLWEInfos,
        Module<B>: AutomorphismKeyCompressedAlloc,
    {
        module.alloc_automorphism_key_compressed_from_infos(infos)
    }

    pub fn alloc<B: Backend>(module: Module<B>, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> Self
    where
        Module<B>: AutomorphismKeyCompressedAlloc,
    {
        module.alloc_automorphism_key_compressed(base2k, k, rank, dnum, dsize)
    }

    pub fn bytes_of_from_infos<A, B: Backend>(module: Module<B>, infos: &A) -> usize
    where
        A: GGLWEInfos,
        Module<B>: AutomorphismKeyCompressedAlloc,
    {
        module.bytes_of_automorphism_key_compressed_from_infos(infos)
    }

    pub fn bytes_of<B: Backend>(
        module: Module<B>,
        base2k: Base2K,
        k: TorusPrecision,
        rank: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> usize
    where
        Module<B>: AutomorphismKeyCompressedAlloc,
    {
        module.bytes_of_automorphism_key_compressed(base2k, k, rank, dnum, dsize)
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
    Self: GLWESwitchingKeyDecompress,
{
    fn decompress_automorphism_key<R, O>(&self, res: &mut R, other: &O)
    where
        R: AutomorphismKeyToMut + SetAutomorphismGaloisElement,
        O: AutomorphismKeyCompressedToRef + GetAutomorphismGaloisElement,
    {
        self.decompress_glwe_switching_key(&mut res.to_mut().key, &other.to_ref().key);
        res.set_p(other.p());
    }
}

impl<B: Backend> AutomorphismKeyDecompress for Module<B> where Self: AutomorphismKeyDecompress {}

impl<D: DataMut> AutomorphismKey<D>
where
    Self: SetAutomorphismGaloisElement,
{
    pub fn decompressed<O, B: Backend>(&mut self, module: &Module<B>, other: &O)
    where
        O: AutomorphismKeyCompressedToRef + GetAutomorphismGaloisElement,
        Module<B>: AutomorphismKeyDecompress,
    {
        module.decompress_automorphism_key(self, other);
    }
}

pub trait AutomorphismKeyCompressedToRef {
    fn to_ref(&self) -> AutomorphismKeyCompressed<&[u8]>;
}

impl<D: DataRef> AutomorphismKeyCompressedToRef for AutomorphismKeyCompressed<D>
where
    GLWESwitchingKeyCompressed<D>: GLWESwitchingKeyCompressedToRef,
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
    GLWESwitchingKeyCompressed<D>: GLWESwitchingKeyCompressedToMut,
{
    fn to_mut(&mut self) -> AutomorphismKeyCompressed<&mut [u8]> {
        AutomorphismKeyCompressed {
            p: self.p,
            key: self.key.to_mut(),
        }
    }
}
