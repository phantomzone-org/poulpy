use poulpy_hal::{
    layouts::{Data, DataMut, DataRef, FillUniform, MatZnx, ReaderFrom, Reset, WriterTo},
    source::Source,
};
use std::fmt;

use crate::layouts::{GLWECiphertext, GLWEMetadata, Infos};

#[derive(PartialEq, Eq, Clone)]
pub struct GGSWCiphertext<D: Data> {
    pub(crate) data: MatZnx<D>,
    pub(crate) metadata: GGSWMetadata,
}
impl<D: Data> GGSWCiphertext<D> {
    pub fn metadata(&self) -> GGSWMetadata {
        self.metadata
    }
}

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct GGSWMetadata {
    pub basek: usize,
    pub rows: usize,
    pub k: usize,
    pub rank: usize,
    pub digits: usize,
}

impl GGSWMetadata {
    pub fn as_glwe(&self) -> GLWEMetadata {
        GLWEMetadata {
            basek: self.basek,
            k: self.k,
            rank: self.rank,
        }
    }
}

impl<D: DataRef> fmt::Debug for GGSWCiphertext<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.data)
    }
}

impl<D: DataRef> fmt::Display for GGSWCiphertext<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "(GGSWCiphertext: metadata: {:?}) {}",
            self.metadata, self.data
        )
    }
}

impl<D: DataMut> Reset for GGSWCiphertext<D> {
    fn reset(&mut self) {
        self.data.reset();
        self.metadata.basek = 0;
        self.metadata.k = 0;
        self.metadata.digits = 0;
    }
}

impl<D: DataMut> FillUniform for GGSWCiphertext<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.data.fill_uniform(log_bound, source);
    }
}

impl<D: DataRef> GGSWCiphertext<D> {
    pub fn at(&self, row: usize, col: usize) -> GLWECiphertext<&[u8]> {
        GLWECiphertext {
            data: self.data.at(row, col),
            metadata: self.metadata().as_glwe(),
        }
    }
}

impl<D: DataMut> GGSWCiphertext<D> {
    pub fn at_mut(&mut self, row: usize, col: usize) -> GLWECiphertext<&mut [u8]> {
        GLWECiphertext {
            metadata: self.metadata().as_glwe(),
            data: self.data.at_mut(row, col),
        }
    }
}

impl GGSWCiphertext<Vec<u8>> {
    pub fn alloc(n: usize, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> Self {
        let size: usize = k.div_ceil(basek);
        debug_assert!(digits > 0, "invalid ggsw: `digits` == 0");

        debug_assert!(
            size > digits,
            "invalid ggsw: ceil(k/basek): {size} <= digits: {digits}"
        );

        assert!(
            rows * digits <= size,
            "invalid ggsw: rows: {rows} * digits:{digits} > ceil(k/basek): {size}"
        );

        Self {
            data: MatZnx::alloc(n, rows, rank + 1, rank + 1, k.div_ceil(basek)),
            metadata: GGSWMetadata {
                basek,
                rows,
                k,
                digits,
                rank,
            },
        }
    }

    pub fn bytes_of(n: usize, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> usize {
        let size: usize = k.div_ceil(basek);
        debug_assert!(
            size > digits,
            "invalid ggsw: ceil(k/basek): {size} <= digits: {digits}"
        );

        assert!(
            rows * digits <= size,
            "invalid ggsw: rows: {rows} * digits:{digits} > ceil(k/basek): {size}"
        );

        MatZnx::alloc_bytes(n, rows, rank + 1, rank + 1, size)
    }
}

impl<D: Data> Infos for GGSWCiphertext<D> {
    type Inner = MatZnx<D>;

    fn inner(&self) -> &Self::Inner {
        &self.data
    }

    fn basek(&self) -> usize {
        self.metadata.basek
    }

    fn k(&self) -> usize {
        self.metadata.k
    }
}

impl<D: Data> GGSWCiphertext<D> {
    pub fn rank(&self) -> usize {
        self.data.cols_out() - 1
    }

    pub fn digits(&self) -> usize {
        self.metadata.digits
    }
}

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

impl<D: DataMut> ReaderFrom for GGSWCiphertext<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.metadata.k = reader.read_u64::<LittleEndian>()? as usize;
        self.metadata.basek = reader.read_u64::<LittleEndian>()? as usize;
        self.metadata.digits = reader.read_u64::<LittleEndian>()? as usize;
        self.data.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GGSWCiphertext<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.metadata.k as u64)?;
        writer.write_u64::<LittleEndian>(self.metadata.basek as u64)?;
        writer.write_u64::<LittleEndian>(self.metadata.digits as u64)?;
        self.data.write_to(writer)
    }
}
