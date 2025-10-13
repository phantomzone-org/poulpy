use poulpy_hal::{
    api::{VecZnxCopy, VecZnxFillUniform},
    layouts::{Backend, Data, DataMut, DataRef, FillUniform, Module, ReaderFrom, WriterTo},
    source::Source,
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GLWEInfos, LWEInfos, Rank, TensorKey, TorusPrecision,
    compressed::{Decompress, GLWESwitchingKeyCompressed, GLWESwitchingKeyCompressedToMut, GLWESwitchingKeyCompressedToRef},
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

impl TensorKeyCompressed<Vec<u8>> {
    pub fn alloc<A>(infos: &A) -> Self
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWETensorKeyCompressed"
        );
        Self::alloc_with(
            infos.n(),
            infos.base2k(),
            infos.k(),
            infos.rank_out(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    pub fn alloc_with(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> Self {
        let mut keys: Vec<GLWESwitchingKeyCompressed<Vec<u8>>> = Vec::new();
        let pairs: u32 = (((rank.0 + 1) * rank.0) >> 1).max(1);
        (0..pairs).for_each(|_| {
            keys.push(GLWESwitchingKeyCompressed::alloc_with(
                n,
                base2k,
                k,
                Rank(1),
                rank,
                dnum,
                dsize,
            ));
        });
        Self { keys }
    }

    pub fn alloc_bytes<A>(infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWETensorKeyCompressed"
        );
        let rank_out: usize = infos.rank_out().into();
        let pairs: usize = (((rank_out + 1) * rank_out) >> 1).max(1);
        pairs
            * GLWESwitchingKeyCompressed::alloc_bytes_with(
                infos.n(),
                infos.base2k(),
                infos.k(),
                Rank(1),
                infos.dnum(),
                infos.dsize(),
            )
    }

    pub fn alloc_bytes_with(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> usize {
        let pairs: usize = (((rank.0 + 1) * rank.0) >> 1).max(1) as usize;
        pairs * GLWESwitchingKeyCompressed::alloc_bytes_with(n, base2k, k, Rank(1), dnum, dsize)
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

impl<D: DataMut, DR: DataRef, B: Backend> Decompress<B, TensorKeyCompressed<DR>> for TensorKey<D>
where
    Module<B>: VecZnxFillUniform + VecZnxCopy,
{
    fn decompress(&mut self, module: &Module<B>, other: &TensorKeyCompressed<DR>) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(
                self.keys.len(),
                other.keys.len(),
                "invalid receiver: self.keys.len()={} != other.keys.len()={}",
                self.keys.len(),
                other.keys.len()
            );
        }

        self.keys
            .iter_mut()
            .zip(other.keys.iter())
            .for_each(|(a, b)| {
                a.decompress(module, b);
            });
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
