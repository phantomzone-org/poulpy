use poulpy_hal::{
    layouts::{Backend, Data, DataMut, DataRef, FillUniform, Module, ReaderFrom, WriterTo},
    source::Source,
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWECompressed, GGLWECompressedToMut, GGLWECompressedToRef, GGLWEDecompress, GGLWEInfos,
    GLWEInfos, LWEInfos, Rank, TensorKey, TensorKeyToMut, TorusPrecision,
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fmt;

#[derive(PartialEq, Eq, Clone)]
pub struct GLWETensorKeyCompressed<D: Data> {
    pub(crate) keys: Vec<GGLWECompressed<D>>,
}

impl<D: Data> LWEInfos for GLWETensorKeyCompressed<D> {
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
impl<D: Data> GLWEInfos for GLWETensorKeyCompressed<D> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data> GGLWEInfos for GLWETensorKeyCompressed<D> {
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

impl<D: DataRef> fmt::Debug for GLWETensorKeyCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: DataMut> FillUniform for GLWETensorKeyCompressed<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.keys
            .iter_mut()
            .for_each(|key: &mut GGLWECompressed<D>| key.fill_uniform(log_bound, source))
    }
}

impl<D: DataRef> fmt::Display for GLWETensorKeyCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "(GLWETensorKeyCompressed)",)?;
        for (i, key) in self.keys.iter().enumerate() {
            write!(f, "{i}: {key}")?;
        }
        Ok(())
    }
}

impl GLWETensorKeyCompressed<Vec<u8>> {
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
        let pairs: u32 = (((rank.as_u32() + 1) * rank.as_u32()) >> 1).max(1);
        GLWETensorKeyCompressed {
            keys: (0..pairs)
                .map(|_| GGLWECompressed::alloc(n, base2k, k, Rank(1), rank, dnum, dsize))
                .collect(),
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
        let pairs: usize = (((rank.0 + 1) * rank.0) >> 1).max(1) as usize;
        pairs * GGLWECompressed::bytes_of(n, base2k, k, Rank(1), dnum, dsize)
    }
}

impl<D: DataMut> ReaderFrom for GLWETensorKeyCompressed<D> {
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

impl<D: DataRef> WriterTo for GLWETensorKeyCompressed<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.keys.len() as u64)?;
        for key in &self.keys {
            key.write_to(writer)?;
        }
        Ok(())
    }
}

pub trait TensorKeyCompressedAtRef<D: DataRef> {
    fn at(&self, i: usize, j: usize) -> &GGLWECompressed<D>;
}

impl<D: DataRef> TensorKeyCompressedAtRef<D> for GLWETensorKeyCompressed<D> {
    fn at(&self, mut i: usize, mut j: usize) -> &GGLWECompressed<D> {
        if i > j {
            std::mem::swap(&mut i, &mut j);
        };
        let rank: usize = self.rank_out().into();
        &self.keys[i * rank + j - (i * (i + 1) / 2)]
    }
}

pub trait TensorKeyCompressedAtMut<D: DataMut> {
    fn at_mut(&mut self, i: usize, j: usize) -> &mut GGLWECompressed<D>;
}

impl<D: DataMut> TensorKeyCompressedAtMut<D> for GLWETensorKeyCompressed<D> {
    fn at_mut(&mut self, mut i: usize, mut j: usize) -> &mut GGLWECompressed<D> {
        if i > j {
            std::mem::swap(&mut i, &mut j);
        };
        let rank: usize = self.rank_out().into();
        &mut self.keys[i * rank + j - (i * (i + 1) / 2)]
    }
}

pub trait TensorKeyDecompress
where
    Self: GGLWEDecompress,
{
    fn decompress_tensor_key<R, O>(&self, res: &mut R, other: &O)
    where
        R: TensorKeyToMut,
        O: TensorKeyCompressedToRef,
    {
        let res: &mut TensorKey<&mut [u8]> = &mut res.to_mut();
        let other: &GLWETensorKeyCompressed<&[u8]> = &other.to_ref();

        assert_eq!(
            res.keys.len(),
            other.keys.len(),
            "invalid receiver: res.keys.len()={} != other.keys.len()={}",
            res.keys.len(),
            other.keys.len()
        );

        for (a, b) in res.keys.iter_mut().zip(other.keys.iter()) {
            self.decompress_gglwe(a, b);
        }
    }
}

impl<B: Backend> TensorKeyDecompress for Module<B> where Self: GGLWEDecompress {}

impl<D: DataMut> TensorKey<D> {
    pub fn decompress<O, M>(&mut self, module: &M, other: &O)
    where
        O: TensorKeyCompressedToRef,
        M: TensorKeyDecompress,
    {
        module.decompress_tensor_key(self, other);
    }
}

pub trait TensorKeyCompressedToMut {
    fn to_mut(&mut self) -> GLWETensorKeyCompressed<&mut [u8]>;
}

impl<D: DataMut> TensorKeyCompressedToMut for GLWETensorKeyCompressed<D>
where
    GGLWECompressed<D>: GGLWECompressedToMut,
{
    fn to_mut(&mut self) -> GLWETensorKeyCompressed<&mut [u8]> {
        GLWETensorKeyCompressed {
            keys: self.keys.iter_mut().map(|c| c.to_mut()).collect(),
        }
    }
}

pub trait TensorKeyCompressedToRef {
    fn to_ref(&self) -> GLWETensorKeyCompressed<&[u8]>;
}

impl<D: DataRef> TensorKeyCompressedToRef for GLWETensorKeyCompressed<D>
where
    GGLWECompressed<D>: GGLWECompressedToRef,
{
    fn to_ref(&self) -> GLWETensorKeyCompressed<&[u8]> {
        GLWETensorKeyCompressed {
            keys: self.keys.iter().map(|c| c.to_ref()).collect(),
        }
    }
}
