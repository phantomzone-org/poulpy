use poulpy_hal::{
    layouts::{Backend, Data, DataMut, DataRef, FillUniform, Module, ReaderFrom, WriterTo},
    source::Source,
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GLWEInfos, LWEInfos, Rank, TensorKey, TensorKeyToMut, TorusPrecision,
    compressed::{
        GLWESwitchingKeyCompressed, GLWESwitchingKeyCompressedAlloc, GLWESwitchingKeyCompressedToMut,
        GLWESwitchingKeyCompressedToRef, GLWESwitchingKeyDecompress,
    },
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fmt;

#[derive(PartialEq, Eq, Clone)]
pub struct TensorKeyCompressed<D: Data> {
    pub(crate) keys: Vec<GLWESwitchingKeyCompressed<D>>,
}

impl<D: Data> LWEInfos for TensorKeyCompressed<D> {
    fn n(&self) -> Degree {
        self.keys[0].n()
    }

    fn base2k(&self) -> Base2K {
        self.keys[0].base2k()
    }

    fn k(&self) -> TorusPrecision {
        self.keys[0].k()
    }
    fn size(&self) -> usize {
        self.keys[0].size()
    }
}
impl<D: Data> GLWEInfos for TensorKeyCompressed<D> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data> GGLWEInfos for TensorKeyCompressed<D> {
    fn rank_in(&self) -> Rank {
        self.rank_out()
    }

    fn rank_out(&self) -> Rank {
        self.keys[0].rank_out()
    }

    fn dsize(&self) -> Dsize {
        self.keys[0].dsize()
    }

    fn dnum(&self) -> Dnum {
        self.keys[0].dnum()
    }
}

impl<D: DataRef> fmt::Debug for TensorKeyCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: DataMut> FillUniform for TensorKeyCompressed<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.keys
            .iter_mut()
            .for_each(|key: &mut GLWESwitchingKeyCompressed<D>| key.fill_uniform(log_bound, source))
    }
}

impl<D: DataRef> fmt::Display for TensorKeyCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "(GLWETensorKeyCompressed)",)?;
        for (i, key) in self.keys.iter().enumerate() {
            write!(f, "{i}: {key}")?;
        }
        Ok(())
    }
}

pub trait TensorKeyCompressedAlloc
where
    Self: GLWESwitchingKeyCompressedAlloc,
{
    fn tensor_key_compressed_alloc(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> TensorKeyCompressed<Vec<u8>> {
        let pairs: u32 = (((rank.as_u32() + 1) * rank.as_u32()) >> 1).max(1);
        TensorKeyCompressed {
            keys: (0..pairs)
                .map(|_| self.alloc_glwe_switching_key_compressed(base2k, k, Rank(1), rank, dnum, dsize))
                .collect(),
        }
    }

    fn tensor_key_compressed_alloc_from_infos<A>(&self, infos: &A) -> TensorKeyCompressed<Vec<u8>>
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWETensorKeyCompressed"
        );
        self.tensor_key_compressed_alloc(
            infos.base2k(),
            infos.k(),
            infos.rank(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    fn tensor_key_compressed_bytes_of(&self, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> usize {
        let pairs: usize = (((rank.0 + 1) * rank.0) >> 1).max(1) as usize;
        pairs * self.bytes_of_glwe_switching_key_compressed(base2k, k, Rank(1), dnum, dsize)
    }

    fn tensor_key_compressed_bytes_of_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        self.tensor_key_compressed_bytes_of(
            infos.base2k(),
            infos.k(),
            infos.rank(),
            infos.dnum(),
            infos.dsize(),
        )
    }
}

impl TensorKeyCompressed<Vec<u8>> {
    pub fn alloc_from_infos<A, B: Backend>(module: Module<B>, infos: &A) -> Self
    where
        A: GGLWEInfos,
        Module<B>: TensorKeyCompressedAlloc,
    {
        module.tensor_key_compressed_alloc_from_infos(infos)
    }

    pub fn alloc<B: Backend>(module: Module<B>, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> Self
    where
        Module<B>: TensorKeyCompressedAlloc,
    {
        module.tensor_key_compressed_alloc(base2k, k, rank, dnum, dsize)
    }

    pub fn bytes_of_from_infos<A, B: Backend>(module: Module<B>, infos: &A) -> usize
    where
        A: GGLWEInfos,
        Module<B>: TensorKeyCompressedAlloc,
    {
        module.tensor_key_compressed_bytes_of_from_infos(infos)
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
        Module<B>: TensorKeyCompressedAlloc,
    {
        module.tensor_key_compressed_bytes_of(base2k, k, rank, dnum, dsize)
    }
}

impl<D: DataMut> ReaderFrom for TensorKeyCompressed<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        let len: usize = reader.read_u64::<LittleEndian>()? as usize;
        if self.keys.len() != len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("self.keys.len()={} != read len={}", self.keys.len(), len),
            ));
        }
        for key in &mut self.keys {
            key.read_from(reader)?;
        }
        Ok(())
    }
}

impl<D: DataRef> WriterTo for TensorKeyCompressed<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.keys.len() as u64)?;
        for key in &self.keys {
            key.write_to(writer)?;
        }
        Ok(())
    }
}

impl<D: DataMut> TensorKeyCompressed<D> {
    pub(crate) fn at_mut(&mut self, mut i: usize, mut j: usize) -> &mut GLWESwitchingKeyCompressed<D> {
        if i > j {
            std::mem::swap(&mut i, &mut j);
        };
        let rank: usize = self.rank_out().into();
        &mut self.keys[i * rank + j - (i * (i + 1) / 2)]
    }
}

pub trait TensorKeyDecompress
where
    Self: GLWESwitchingKeyDecompress,
{
    fn decompress_tensor_key<R, O>(&self, res: &mut R, other: &O)
    where
        R: TensorKeyToMut,
        O: TensorKeyCompressedToRef,
    {
        let res: &mut TensorKey<&mut [u8]> = &mut res.to_mut();
        let other: &TensorKeyCompressed<&[u8]> = &other.to_ref();

        assert_eq!(
            res.keys.len(),
            other.keys.len(),
            "invalid receiver: res.keys.len()={} != other.keys.len()={}",
            res.keys.len(),
            other.keys.len()
        );

        for (a, b) in res.keys.iter_mut().zip(other.keys.iter()) {
            self.decompress_glwe_switching_key(a, b);
        }
    }
}

impl<B: Backend> TensorKeyDecompress for Module<B> where Self: GLWESwitchingKeyDecompress {}

impl<D: DataMut> TensorKey<D> {
    pub fn decompress<O, B: Backend>(&mut self, module: &Module<B>, other: &O)
    where
        O: TensorKeyCompressedToRef,
        Module<B>: GLWESwitchingKeyDecompress,
    {
        module.decompress_tensor_key(self, other);
    }
}

pub trait TensorKeyCompressedToMut {
    fn to_mut(&mut self) -> TensorKeyCompressed<&mut [u8]>;
}

impl<D: DataMut> TensorKeyCompressedToMut for TensorKeyCompressed<D>
where
    GLWESwitchingKeyCompressed<D>: GLWESwitchingKeyCompressedToMut,
{
    fn to_mut(&mut self) -> TensorKeyCompressed<&mut [u8]> {
        TensorKeyCompressed {
            keys: self.keys.iter_mut().map(|c| c.to_mut()).collect(),
        }
    }
}

pub trait TensorKeyCompressedToRef {
    fn to_ref(&self) -> TensorKeyCompressed<&[u8]>;
}

impl<D: DataRef> TensorKeyCompressedToRef for TensorKeyCompressed<D>
where
    GLWESwitchingKeyCompressed<D>: GLWESwitchingKeyCompressedToRef,
{
    fn to_ref(&self) -> TensorKeyCompressed<&[u8]> {
        TensorKeyCompressed {
            keys: self.keys.iter().map(|c| c.to_ref()).collect(),
        }
    }
}
