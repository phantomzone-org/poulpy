use poulpy_backend::hal::layouts::{Data, DataMut, DataRef, ReaderFrom, VecZnx, WriterTo};

use crate::{dist::Distribution, layouts::Infos};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

#[derive(PartialEq, Eq)]
pub struct GLWEPublicKey<D: Data> {
    pub(crate) data: VecZnx<D>,
    pub(crate) basek: usize,
    pub(crate) k: usize,
    pub(crate) dist: Distribution,
}

impl GLWEPublicKey<Vec<u8>> {
    pub fn alloc(n: usize, basek: usize, k: usize, rank: usize) -> Self {
        Self {
            data: VecZnx::alloc(n, rank + 1, k.div_ceil(basek)),
            basek,
            k,
            dist: Distribution::NONE,
        }
    }

    pub fn bytes_of(n: usize, basek: usize, k: usize, rank: usize) -> usize {
        VecZnx::alloc_bytes(n, rank + 1, k.div_ceil(basek))
    }
}

impl<D: Data> Infos for GLWEPublicKey<D> {
    type Inner = VecZnx<D>;

    fn inner(&self) -> &Self::Inner {
        &self.data
    }

    fn basek(&self) -> usize {
        self.basek
    }

    fn k(&self) -> usize {
        self.k
    }
}

impl<D: Data> GLWEPublicKey<D> {
    pub fn rank(&self) -> usize {
        self.cols() - 1
    }
}

impl<D: DataMut> ReaderFrom for GLWEPublicKey<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.k = reader.read_u64::<LittleEndian>()? as usize;
        self.basek = reader.read_u64::<LittleEndian>()? as usize;
        match Distribution::read_from(reader) {
            Ok(dist) => self.dist = dist,
            Err(e) => return Err(e),
        }
        self.data.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GLWEPublicKey<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.k as u64)?;
        writer.write_u64::<LittleEndian>(self.basek as u64)?;
        match self.dist.write_to(writer) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        self.data.write_to(writer)
    }
}
