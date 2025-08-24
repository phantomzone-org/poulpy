use poulpy_hal::{
    api::{VecZnxCopy, VecZnxFillUniform},
    layouts::{Backend, Data, DataMut, DataRef, FillUniform, MatZnx, Module, ReaderFrom, Reset, WriterTo},
    source::Source,
};

use crate::layouts::{
    GGLWESwitchingKey, Infos,
    compressed::{Decompress, GGLWECiphertextCompressed},
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fmt;

#[derive(PartialEq, Eq, Clone)]
pub struct GGLWESwitchingKeyCompressed<D: Data> {
    pub(crate) key: GGLWECiphertextCompressed<D>,
    pub(crate) sk_in_n: usize,  // Degree of sk_in
    pub(crate) sk_out_n: usize, // Degree of sk_out
}

impl<D: DataRef> fmt::Debug for GGLWESwitchingKeyCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl<D: DataMut> FillUniform for GGLWESwitchingKeyCompressed<D> {
    fn fill_uniform(&mut self, source: &mut Source) {
        self.key.fill_uniform(source);
    }
}

impl<D: DataMut> Reset for GGLWESwitchingKeyCompressed<D>
where
    MatZnx<D>: Reset,
{
    fn reset(&mut self) {
        self.key.reset();
        self.sk_in_n = 0;
        self.sk_out_n = 0;
    }
}

impl<D: DataRef> fmt::Display for GGLWESwitchingKeyCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "(GLWESwitchingKeyCompressed: sk_in_n={} sk_out_n={}) {}",
            self.sk_in_n, self.sk_out_n, self.key.data
        )
    }
}

impl<D: Data> Infos for GGLWESwitchingKeyCompressed<D> {
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

impl<D: Data> GGLWESwitchingKeyCompressed<D> {
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

impl GGLWESwitchingKeyCompressed<Vec<u8>> {
    pub fn alloc(n: usize, basek: usize, k: usize, rows: usize, digits: usize, rank_in: usize, rank_out: usize) -> Self {
        GGLWESwitchingKeyCompressed {
            key: GGLWECiphertextCompressed::alloc(n, basek, k, rows, digits, rank_in, rank_out),
            sk_in_n: 0,
            sk_out_n: 0,
        }
    }

    pub fn bytes_of(n: usize, basek: usize, k: usize, rows: usize, digits: usize, rank_in: usize) -> usize {
        GGLWECiphertextCompressed::bytes_of(n, basek, k, rows, digits, rank_in)
    }
}

impl<D: DataMut> ReaderFrom for GGLWESwitchingKeyCompressed<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.sk_in_n = reader.read_u64::<LittleEndian>()? as usize;
        self.sk_out_n = reader.read_u64::<LittleEndian>()? as usize;
        self.key.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GGLWESwitchingKeyCompressed<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.sk_in_n as u64)?;
        writer.write_u64::<LittleEndian>(self.sk_out_n as u64)?;
        self.key.write_to(writer)
    }
}

impl<D: DataMut, DR: DataRef, B: Backend> Decompress<B, GGLWESwitchingKeyCompressed<DR>> for GGLWESwitchingKey<D>
where
    Module<B>: VecZnxFillUniform + VecZnxCopy,
{
    fn decompress(&mut self, module: &Module<B>, other: &GGLWESwitchingKeyCompressed<DR>) {
        self.key.decompress(module, &other.key);
        self.sk_in_n = other.sk_in_n;
        self.sk_out_n = other.sk_out_n;
    }
}
