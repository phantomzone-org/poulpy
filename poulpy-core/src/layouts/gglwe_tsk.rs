use poulpy_hal::{
    layouts::{Backend, Data, DataMut, DataRef, FillUniform, Module, ReaderFrom, WriterTo},
    source::Source,
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GLWEInfos, GLWESwitchingKey, GLWESwitchingKeyAlloc, GLWESwitchingKeyToMut,
    GLWESwitchingKeyToRef, LWEInfos, Rank, TorusPrecision,
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use std::fmt;

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct TensorKeyLayout {
    pub n: Degree,
    pub base2k: Base2K,
    pub k: TorusPrecision,
    pub rank: Rank,
    pub dnum: Dnum,
    pub dsize: Dsize,
}

#[derive(PartialEq, Eq, Clone)]
pub struct TensorKey<D: Data> {
    pub(crate) keys: Vec<GLWESwitchingKey<D>>,
}

impl<D: Data> LWEInfos for TensorKey<D> {
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

impl<D: Data> GLWEInfos for TensorKey<D> {
    fn rank(&self) -> Rank {
        self.keys[0].rank_out()
    }
}

impl<D: Data> GGLWEInfos for TensorKey<D> {
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

impl LWEInfos for TensorKeyLayout {
    fn n(&self) -> Degree {
        self.n
    }

    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }
}

impl GLWEInfos for TensorKeyLayout {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl GGLWEInfos for TensorKeyLayout {
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

impl<D: DataRef> fmt::Debug for TensorKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: DataMut> FillUniform for TensorKey<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.keys
            .iter_mut()
            .for_each(|key: &mut GLWESwitchingKey<D>| key.fill_uniform(log_bound, source))
    }
}

impl<D: DataRef> fmt::Display for TensorKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "(GLWETensorKey)",)?;
        for (i, key) in self.keys.iter().enumerate() {
            write!(f, "{i}: {key}")?;
        }
        Ok(())
    }
}

pub trait TensorKeyAlloc
where
    Self: GLWESwitchingKeyAlloc,
{
    fn alloc_tensor_key(&self, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> TensorKey<Vec<u8>> {
        let pairs: u32 = (((rank.0 + 1) * rank.0) >> 1).max(1);
        TensorKey {
            keys: (0..pairs)
                .map(|_| self.alloc_glwe_switching_key(base2k, k, Rank(1), rank, dnum, dsize))
                .collect(),
        }
    }

    fn alloc_tensor_key_from_infos<A>(&self, infos: &A) -> TensorKey<Vec<u8>>
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWETensorKey"
        );
        self.alloc_tensor_key(
            infos.base2k(),
            infos.k(),
            infos.rank(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    fn bytes_of_tensor_key(&self, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> usize {
        let pairs: usize = (((rank.0 + 1) * rank.0) >> 1).max(1) as usize;
        pairs * self.bytes_of_glwe_switching_key(base2k, k, Rank(1), rank, dnum, dsize)
    }

    fn bytes_of_tensor_key_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWETensorKey"
        );
        self.bytes_of_tensor_key(
            infos.base2k(),
            infos.k(),
            infos.rank(),
            infos.dnum(),
            infos.dsize(),
        )
    }
}

impl<B: Backend> TensorKeyAlloc for Module<B> where Self: GLWESwitchingKeyAlloc {}

impl TensorKey<Vec<u8>> {
    pub fn alloc_from_infos<A, M>(module: &M, infos: &A) -> Self
    where
        A: GGLWEInfos,
        M: TensorKeyAlloc,
    {
        module.alloc_tensor_key_from_infos(infos)
    }

    pub fn alloc<M>(module: &M, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> Self
    where
        M: TensorKeyAlloc,
    {
        module.alloc_tensor_key(base2k, k, rank, dnum, dsize)
    }

    pub fn bytes_of_from_infos<A, M>(module: &M, infos: &A) -> usize
    where
        A: GGLWEInfos,
        M: TensorKeyAlloc,
    {
        module.bytes_of_tensor_key_from_infos(infos)
    }

    pub fn bytes_of<M>(module: &M, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> usize
    where
        M: TensorKeyAlloc,
    {
        module.bytes_of_tensor_key(base2k, k, rank, dnum, dsize)
    }
}

impl<D: DataMut> TensorKey<D> {
    // Returns a mutable reference to GLWESwitchingKey_{s}(s[i] * s[j])
    pub fn at_mut(&mut self, mut i: usize, mut j: usize) -> &mut GLWESwitchingKey<D> {
        if i > j {
            std::mem::swap(&mut i, &mut j);
        };
        let rank: usize = self.rank_out().into();
        &mut self.keys[i * rank + j - (i * (i + 1) / 2)]
    }
}

impl<D: DataRef> TensorKey<D> {
    // Returns a reference to GLWESwitchingKey_{s}(s[i] * s[j])
    pub fn at(&self, mut i: usize, mut j: usize) -> &GLWESwitchingKey<D> {
        if i > j {
            std::mem::swap(&mut i, &mut j);
        };
        let rank: usize = self.rank_out().into();
        &self.keys[i * rank + j - (i * (i + 1) / 2)]
    }
}

impl<D: DataMut> ReaderFrom for TensorKey<D> {
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

impl<D: DataRef> WriterTo for TensorKey<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.keys.len() as u64)?;
        for key in &self.keys {
            key.write_to(writer)?;
        }
        Ok(())
    }
}

pub trait TensorKeyToRef {
    fn to_ref(&self) -> TensorKey<&[u8]>;
}

impl<D: DataRef> TensorKeyToRef for TensorKey<D>
where
    GLWESwitchingKey<D>: GLWESwitchingKeyToRef,
{
    fn to_ref(&self) -> TensorKey<&[u8]> {
        TensorKey {
            keys: self.keys.iter().map(|c| c.to_ref()).collect(),
        }
    }
}

pub trait TensorKeyToMut {
    fn to_mut(&mut self) -> TensorKey<&mut [u8]>;
}

impl<D: DataMut> TensorKeyToMut for TensorKey<D>
where
    GLWESwitchingKey<D>: GLWESwitchingKeyToMut,
{
    fn to_mut(&mut self) -> TensorKey<&mut [u8]> {
        TensorKey {
            keys: self.keys.iter_mut().map(|c| c.to_mut()).collect(),
        }
    }
}
