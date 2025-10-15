use poulpy_hal::{
    layouts::{Backend, Data, DataMut, DataRef, FillUniform, Module, ReaderFrom, WriterTo},
    source::Source,
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GLWEInfos, GLWESwitchingKey, GLWESwitchingKeySetMetaData, GLWESwitchingKeyToMut,
    LWEInfos, Rank, TorusPrecision,
    compressed::{GGLWECompressed, GGLWECompressedAlloc, GGLWECompressedToMut, GGLWECompressedToRef, GGLWEDecompress},
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fmt;

#[derive(PartialEq, Eq, Clone)]
pub struct GLWESwitchingKeyCompressed<D: Data> {
    pub(crate) key: GGLWECompressed<D>,
    pub(crate) sk_in_n: usize,  // Degree of sk_in
    pub(crate) sk_out_n: usize, // Degree of sk_out
}

impl<D: Data> LWEInfos for GLWESwitchingKeyCompressed<D> {
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
impl<D: Data> GLWEInfos for GLWESwitchingKeyCompressed<D> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data> GGLWEInfos for GLWESwitchingKeyCompressed<D> {
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

impl<D: DataRef> fmt::Debug for GLWESwitchingKeyCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: DataMut> FillUniform for GLWESwitchingKeyCompressed<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.key.fill_uniform(log_bound, source);
    }
}

impl<D: DataRef> fmt::Display for GLWESwitchingKeyCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "(GLWESwitchingKeyCompressed: sk_in_n={} sk_out_n={}) {}",
            self.sk_in_n, self.sk_out_n, self.key.data
        )
    }
}

pub trait GLWESwitchingKeyCompressedAlloc
where
    Self: GGLWECompressedAlloc,
{
    fn alloc_glwe_switching_key_compressed(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> GLWESwitchingKeyCompressed<Vec<u8>> {
        GLWESwitchingKeyCompressed {
            key: self.alloc_gglwe_compressed(base2k, k, rank_in, rank_out, dnum, dsize),
            sk_in_n: 0,
            sk_out_n: 0,
        }
    }

    fn alloc_glwe_switching_key_compressed_from_infos<A>(&self, infos: &A) -> GLWESwitchingKeyCompressed<Vec<u8>>
    where
        A: GGLWEInfos,
    {
        self.alloc_glwe_switching_key_compressed(
            infos.base2k(),
            infos.k(),
            infos.rank_in(),
            infos.rank_out(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    fn bytes_of_glwe_switching_key_compressed(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> usize {
        self.bytes_of_gglwe_compressed(base2k, k, rank_in, dnum, dsize)
    }

    fn bytes_of_glwe_switching_key_compressed_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        self.bytes_of_gglwe_compressed_from_infos(infos)
    }
}

impl GLWESwitchingKeyCompressed<Vec<u8>> {
    pub fn alloc_from_infos<A, M>(module: &M, infos: &A) -> Self
    where
        A: GGLWEInfos,
        M: GLWESwitchingKeyCompressedAlloc,
    {
        module.alloc_glwe_switching_key_compressed_from_infos(infos)
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
        M: GLWESwitchingKeyCompressedAlloc,
    {
        module.alloc_glwe_switching_key_compressed(base2k, k, rank_in, rank_out, dnum, dsize)
    }

    pub fn bytes_of_from_infos<A, M>(module: &M, infos: &A) -> usize
    where
        A: GGLWEInfos,
        M: GLWESwitchingKeyCompressedAlloc,
    {
        module.bytes_of_glwe_switching_key_compressed_from_infos(infos)
    }

    pub fn bytes_of<M>(module: &M, base2k: Base2K, k: TorusPrecision, rank_in: Rank, dnum: Dnum, dsize: Dsize) -> usize
    where
        M: GLWESwitchingKeyCompressedAlloc,
    {
        module.bytes_of_glwe_switching_key_compressed(base2k, k, rank_in, dnum, dsize)
    }
}

impl<D: DataMut> ReaderFrom for GLWESwitchingKeyCompressed<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.sk_in_n = reader.read_u64::<LittleEndian>()? as usize;
        self.sk_out_n = reader.read_u64::<LittleEndian>()? as usize;
        self.key.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GLWESwitchingKeyCompressed<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.sk_in_n as u64)?;
        writer.write_u64::<LittleEndian>(self.sk_out_n as u64)?;
        self.key.write_to(writer)
    }
}

pub trait GLWESwitchingKeyDecompress
where
    Self: GGLWEDecompress,
{
    fn decompress_glwe_switching_key<R, O>(&self, res: &mut R, other: &O)
    where
        R: GLWESwitchingKeyToMut + GLWESwitchingKeySetMetaData,
        O: GLWESwitchingKeyCompressedToRef,
    {
        let other: &GLWESwitchingKeyCompressed<&[u8]> = &other.to_ref();
        self.decompress_gglwe(&mut res.to_mut().key, &other.key);
        res.set_sk_in_n(other.sk_in_n);
        res.set_sk_out_n(other.sk_out_n);
    }
}

impl<B: Backend> GLWESwitchingKeyDecompress for Module<B> where Self: GGLWEDecompress {}

impl<D: DataMut> GLWESwitchingKey<D> {
    pub fn decompress<O, M>(&mut self, module: &M, other: &O)
    where
        O: GLWESwitchingKeyCompressedToRef,
        M: GLWESwitchingKeyDecompress,
    {
        module.decompress_glwe_switching_key(self, other);
    }
}

pub trait GLWESwitchingKeyCompressedToMut {
    fn to_mut(&mut self) -> GLWESwitchingKeyCompressed<&mut [u8]>;
}

impl<D: DataMut> GLWESwitchingKeyCompressedToMut for GLWESwitchingKeyCompressed<D>
where
    GGLWECompressed<D>: GGLWECompressedToMut,
{
    fn to_mut(&mut self) -> GLWESwitchingKeyCompressed<&mut [u8]> {
        GLWESwitchingKeyCompressed {
            sk_in_n: self.sk_in_n,
            sk_out_n: self.sk_out_n,
            key: self.key.to_mut(),
        }
    }
}

pub trait GLWESwitchingKeyCompressedToRef {
    fn to_ref(&self) -> GLWESwitchingKeyCompressed<&[u8]>;
}

impl<D: DataRef> GLWESwitchingKeyCompressedToRef for GLWESwitchingKeyCompressed<D>
where
    GGLWECompressed<D>: GGLWECompressedToRef,
{
    fn to_ref(&self) -> GLWESwitchingKeyCompressed<&[u8]> {
        GLWESwitchingKeyCompressed {
            sk_in_n: self.sk_in_n,
            sk_out_n: self.sk_out_n,
            key: self.key.to_ref(),
        }
    }
}
