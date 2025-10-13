use poulpy_hal::{
    api::{VecZnxCopy, VecZnxFillUniform},
    layouts::{Backend, Data, DataMut, DataRef, FillUniform, Module, ReaderFrom, WriterTo},
    source::Source,
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GLWEInfos, GLWESwitchingKey, LWEInfos, Rank, TorusPrecision,
    compressed::{Decompress, GGLWECiphertextCompressed, GGLWECiphertextCompressedToMut, GGLWECiphertextCompressedToRef},
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fmt;

#[derive(PartialEq, Eq, Clone)]
pub struct GLWESwitchingKeyCompressed<D: Data> {
    pub(crate) key: GGLWECiphertextCompressed<D>,
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

impl GLWESwitchingKeyCompressed<Vec<u8>> {
    pub fn alloc<A>(infos: &A) -> Self
    where
        A: GGLWEInfos,
    {
        GLWESwitchingKeyCompressed {
            key: GGLWECiphertextCompressed::alloc(infos),
            sk_in_n: 0,
            sk_out_n: 0,
        }
    }

    pub fn alloc_with(
        n: Degree,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> Self {
        GLWESwitchingKeyCompressed {
            key: GGLWECiphertextCompressed::alloc_with(n, base2k, k, rank_in, rank_out, dnum, dsize),
            sk_in_n: 0,
            sk_out_n: 0,
        }
    }

    pub fn alloc_bytes<A>(infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        GGLWECiphertextCompressed::alloc_bytes(infos)
    }

    pub fn alloc_bytes_with(n: Degree, base2k: Base2K, k: TorusPrecision, rank_in: Rank, dnum: Dnum, dsize: Dsize) -> usize {
        GGLWECiphertextCompressed::alloc_bytes_with(n, base2k, k, rank_in, dnum, dsize)
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

impl<D: DataMut, DR: DataRef, B: Backend> Decompress<B, GLWESwitchingKeyCompressed<DR>> for GLWESwitchingKey<D>
where
    Module<B>: VecZnxFillUniform + VecZnxCopy,
{
    fn decompress(&mut self, module: &Module<B>, other: &GLWESwitchingKeyCompressed<DR>) {
        self.key.decompress(module, &other.key);
        self.sk_in_n = other.sk_in_n;
        self.sk_out_n = other.sk_out_n;
    }
}

pub trait GLWESwitchingKeyCompressedToMut {
    fn to_mut(&mut self) -> GLWESwitchingKeyCompressed<&mut [u8]>;
}

impl<D: DataMut> GLWESwitchingKeyCompressedToMut for GLWESwitchingKeyCompressed<D>
where
    GGLWECiphertextCompressed<D>: GGLWECiphertextCompressedToMut,
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
    GGLWECiphertextCompressed<D>: GGLWECiphertextCompressedToRef,
{
    fn to_ref(&self) -> GLWESwitchingKeyCompressed<&[u8]> {
        GLWESwitchingKeyCompressed {
            sk_in_n: self.sk_in_n,
            sk_out_n: self.sk_out_n,
            key: self.key.to_ref(),
        }
    }
}
