use poulpy_hal::{
    api::{VecZnxCopy, VecZnxFillUniform},
    layouts::{Backend, Data, DataMut, DataRef, FillUniform, MatZnx, Module, ReaderFrom, Reset, WriterTo},
    source::Source,
};

use crate::layouts::{
    GGLWEAutomorphismKey, Infos,
    compressed::{Decompress, GGLWESwitchingKeyCompressed},
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fmt;

#[derive(PartialEq, Eq, Clone)]
pub struct GGLWEAutomorphismKeyCompressed<D: Data> {
    pub(crate) key: GGLWESwitchingKeyCompressed<D>,
    pub(crate) p: i64,
}

impl<D: DataRef> fmt::Debug for GGLWEAutomorphismKeyCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl<D: DataMut> FillUniform for GGLWEAutomorphismKeyCompressed<D> {
    fn fill_uniform(&mut self, source: &mut Source) {
        self.key.fill_uniform(source);
    }
}

impl<D: DataMut> Reset for GGLWEAutomorphismKeyCompressed<D>
where
    MatZnx<D>: Reset,
{
    fn reset(&mut self) {
        self.key.reset();
        self.p = 0;
    }
}

impl<D: DataRef> fmt::Display for GGLWEAutomorphismKeyCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(AutomorphismKeyCompressed: p={}) {}", self.p, self.key)
    }
}

impl GGLWEAutomorphismKeyCompressed<Vec<u8>> {
    pub fn alloc(n: usize, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> Self {
        GGLWEAutomorphismKeyCompressed {
            key: GGLWESwitchingKeyCompressed::alloc(n, basek, k, rows, digits, rank, rank),
            p: 0,
        }
    }

    pub fn bytes_of(n: usize, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> usize {
        GGLWESwitchingKeyCompressed::<Vec<u8>>::bytes_of(n, basek, k, rows, digits, rank)
    }
}

impl<D: Data> Infos for GGLWEAutomorphismKeyCompressed<D> {
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

impl<D: Data> GGLWEAutomorphismKeyCompressed<D> {
    pub fn rank(&self) -> usize {
        self.key.rank()
    }

    pub fn digits(&self) -> usize {
        self.key.digits()
    }

    pub fn rank_in(&self) -> usize {
        self.key.rank_in()
    }

    pub fn rank_out(&self) -> usize {
        self.key.rank_out()
    }
}

impl<D: DataMut> ReaderFrom for GGLWEAutomorphismKeyCompressed<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.p = reader.read_u64::<LittleEndian>()? as i64;
        self.key.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GGLWEAutomorphismKeyCompressed<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.p as u64)?;
        self.key.write_to(writer)
    }
}

impl<D: DataMut, DR: DataRef, B: Backend> Decompress<B, GGLWEAutomorphismKeyCompressed<DR>> for GGLWEAutomorphismKey<D>
where
    Module<B>: VecZnxFillUniform + VecZnxCopy,
{
    fn decompress(&mut self, module: &Module<B>, other: &GGLWEAutomorphismKeyCompressed<DR>) {
        self.key.decompress(module, &other.key);
        self.p = other.p;
    }
}
