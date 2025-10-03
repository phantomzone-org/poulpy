use poulpy_hal::{
    layouts::{Data, DataMut, DataRef, FillUniform, ReaderFrom, WriterTo},
    source::Source,
};

use crate::layouts::{
    Base2K, Degree, Digits, GGLWECiphertext, GGLWEInfos, GLWECiphertext, GLWEInfos, LWEInfos, Rank, Rows, TorusPrecision,
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use std::fmt;

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct GGLWESwitchingKeyLayout {
    pub n: Degree,
    pub base2k: Base2K,
    pub k: TorusPrecision,
    pub rows: Rows,
    pub digits: Digits,
    pub rank_in: Rank,
    pub rank_out: Rank,
}

impl LWEInfos for GGLWESwitchingKeyLayout {
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

impl GLWEInfos for GGLWESwitchingKeyLayout {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl GGLWEInfos for GGLWESwitchingKeyLayout {
    fn rank_in(&self) -> Rank {
        self.rank_in
    }

    fn rank_out(&self) -> Rank {
        self.rank_out
    }

    fn digits(&self) -> Digits {
        self.digits
    }

    fn rows(&self) -> Rows {
        self.rows
    }
}

#[derive(PartialEq, Eq, Clone)]
pub struct GGLWESwitchingKey<D: Data> {
    pub(crate) key: GGLWECiphertext<D>,
    pub(crate) sk_in_n: usize,  // Degree of sk_in
    pub(crate) sk_out_n: usize, // Degree of sk_out
}

impl<D: Data> LWEInfos for GGLWESwitchingKey<D> {
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

impl<D: Data> GLWEInfos for GGLWESwitchingKey<D> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data> GGLWEInfos for GGLWESwitchingKey<D> {
    fn rank_in(&self) -> Rank {
        self.key.rank_in()
    }

    fn rank_out(&self) -> Rank {
        self.key.rank_out()
    }

    fn digits(&self) -> Digits {
        self.key.digits()
    }

    fn rows(&self) -> Rows {
        self.key.rows()
    }
}

impl<D: DataRef> fmt::Debug for GGLWESwitchingKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: DataRef> fmt::Display for GGLWESwitchingKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "(GLWESwitchingKey: sk_in_n={} sk_out_n={}) {}",
            self.sk_in_n,
            self.sk_out_n,
            self.key.data()
        )
    }
}

impl<D: DataMut> FillUniform for GGLWESwitchingKey<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.key.fill_uniform(log_bound, source);
    }
}

impl GGLWESwitchingKey<Vec<u8>> {
    pub fn alloc<A>(infos: &A) -> Self
    where
        A: GGLWEInfos,
    {
        GGLWESwitchingKey {
            key: GGLWECiphertext::alloc(infos),
            sk_in_n: 0,
            sk_out_n: 0,
        }
    }

    pub fn alloc_with(
        n: Degree,
        base2k: Base2K,
        k: TorusPrecision,
        rows: Rows,
        digits: Digits,
        rank_in: Rank,
        rank_out: Rank,
    ) -> Self {
        GGLWESwitchingKey {
            key: GGLWECiphertext::alloc_with(n, base2k, k, rows, digits, rank_in, rank_out),
            sk_in_n: 0,
            sk_out_n: 0,
        }
    }

    pub fn alloc_bytes<A>(infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        GGLWECiphertext::alloc_bytes(infos)
    }

    pub fn alloc_bytes_with(
        n: Degree,
        base2k: Base2K,
        k: TorusPrecision,
        rows: Rows,
        digits: Digits,
        rank_in: Rank,
        rank_out: Rank,
    ) -> usize {
        GGLWECiphertext::alloc_bytes_with(n, base2k, k, rows, digits, rank_in, rank_out)
    }
}

impl<D: DataRef> GGLWESwitchingKey<D> {
    pub fn at(&self, row: usize, col: usize) -> GLWECiphertext<&[u8]> {
        self.key.at(row, col)
    }
}

impl<D: DataMut> GGLWESwitchingKey<D> {
    pub fn at_mut(&mut self, row: usize, col: usize) -> GLWECiphertext<&mut [u8]> {
        self.key.at_mut(row, col)
    }
}

impl<D: DataMut> ReaderFrom for GGLWESwitchingKey<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.sk_in_n = reader.read_u64::<LittleEndian>()? as usize;
        self.sk_out_n = reader.read_u64::<LittleEndian>()? as usize;
        self.key.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GGLWESwitchingKey<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.sk_in_n as u64)?;
        writer.write_u64::<LittleEndian>(self.sk_out_n as u64)?;
        self.key.write_to(writer)
    }
}
