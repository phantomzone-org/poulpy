use poulpy_hal::{
    layouts::{Data, DataMut, DataRef, FillUniform, MatZnx, ReaderFrom, Reset, WriterTo},
    source::Source,
};

use crate::layouts::{GGLWESwitchingKey, GLWECiphertext, Infos};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use std::fmt;

#[derive(PartialEq, Eq, Clone)]
pub struct GGLWEAutomorphismKey<D: Data> {
    pub(crate) key: GGLWESwitchingKey<D>,
    pub(crate) p: i64,
}

impl<D: DataRef> fmt::Debug for GGLWEAutomorphismKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl<D: DataMut> FillUniform for GGLWEAutomorphismKey<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.key.fill_uniform(log_bound, source);
    }
}

impl<D: DataMut> Reset for GGLWEAutomorphismKey<D>
where
    MatZnx<D>: Reset,
{
    fn reset(&mut self) {
        self.key.reset();
        self.p = 0;
    }
}

impl<D: DataRef> fmt::Display for GGLWEAutomorphismKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(AutomorphismKey: p={}) {}", self.p, self.key)
    }
}

impl GGLWEAutomorphismKey<Vec<u8>> {
    pub fn alloc(n: usize, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> Self {
        GGLWEAutomorphismKey {
            key: GGLWESwitchingKey::alloc(n, basek, k, rows, digits, rank, rank),
            p: 0,
        }
    }

    pub fn bytes_of(n: usize, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> usize {
        GGLWESwitchingKey::bytes_of(n, basek, k, rows, digits, rank, rank)
    }
}

impl<D: Data> Infos for GGLWEAutomorphismKey<D> {
    type Inner = MatZnx<D>;

    fn inner(&self) -> &Self::Inner {
        self.key.inner()
    }

    fn basek(&self) -> usize {
        self.key.basek()
    }

    fn k(&self) -> usize {
        self.key.k()
    }
}

impl<D: Data> GGLWEAutomorphismKey<D> {
    pub fn p(&self) -> i64 {
        self.p
    }

    pub fn digits(&self) -> usize {
        self.key.digits()
    }

    pub fn rank(&self) -> usize {
        self.key.rank()
    }

    pub fn rank_in(&self) -> usize {
        self.key.rank_in()
    }

    pub fn rank_out(&self) -> usize {
        self.key.rank_out()
    }
}

impl<D: DataRef> GGLWEAutomorphismKey<D> {
    pub fn at(&self, row: usize, col: usize) -> GLWECiphertext<&[u8]> {
        self.key.at(row, col)
    }
}

impl<D: DataMut> GGLWEAutomorphismKey<D> {
    pub fn at_mut(&mut self, row: usize, col: usize) -> GLWECiphertext<&mut [u8]> {
        self.key.at_mut(row, col)
    }
}

impl<D: DataMut> ReaderFrom for GGLWEAutomorphismKey<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.p = reader.read_u64::<LittleEndian>()? as i64;
        self.key.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GGLWEAutomorphismKey<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.p as u64)?;
        self.key.write_to(writer)
    }
}
