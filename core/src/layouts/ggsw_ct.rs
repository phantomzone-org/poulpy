use backend::hal::{
    api::{FillUniform, Reset},
    layouts::{Data, DataMut, DataRef, MatZnx, ReaderFrom, WriterTo},
};
use std::fmt;

use crate::layouts::{GLWECiphertext, Infos};

#[derive(PartialEq, Eq, Clone)]
pub struct GGSWCiphertext<D: Data> {
    pub(crate) data: MatZnx<D>,
    pub(crate) basek: usize,
    pub(crate) k: usize,
    pub(crate) digits: usize,
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
            "(GGSWCiphertext: basek={} k={} digits={}) {}",
            self.basek, self.k, self.digits, self.data
        )
    }
}

impl<D: DataMut> Reset for GGSWCiphertext<D> {
    fn reset(&mut self) {
        self.data.reset();
        self.basek = 0;
        self.k = 0;
        self.digits = 0;
    }
}

impl<D: DataMut> FillUniform for GGSWCiphertext<D> {
    fn fill_uniform(&mut self, source: &mut sampling::source::Source) {
        self.data.fill_uniform(source);
    }
}

impl<D: DataRef> GGSWCiphertext<D> {
    pub fn at(&self, row: usize, col: usize) -> GLWECiphertext<&[u8]> {
        GLWECiphertext {
            data: self.data.at(row, col),
            basek: self.basek,
            k: self.k,
        }
    }
}

impl<D: DataMut> GGSWCiphertext<D> {
    pub fn at_mut(&mut self, row: usize, col: usize) -> GLWECiphertext<&mut [u8]> {
        GLWECiphertext {
            data: self.data.at_mut(row, col),
            basek: self.basek,
            k: self.k,
        }
    }
}

impl GGSWCiphertext<Vec<u8>> {
    pub fn alloc(n: usize, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> Self {
        let size: usize = k.div_ceil(basek);
        debug_assert!(digits > 0, "invalid ggsw: `digits` == 0");

        debug_assert!(
            size > digits,
            "invalid ggsw: ceil(k/basek): {} <= digits: {}",
            size,
            digits
        );

        assert!(
            rows * digits <= size,
            "invalid ggsw: rows: {} * digits:{} > ceil(k/basek): {}",
            rows,
            digits,
            size
        );

        Self {
            data: MatZnx::alloc(n, rows, rank + 1, rank + 1, k.div_ceil(basek)),
            basek,
            k,
            digits,
        }
    }

    pub fn bytes_of(n: usize, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> usize {
        let size: usize = k.div_ceil(basek);
        debug_assert!(
            size > digits,
            "invalid ggsw: ceil(k/basek): {} <= digits: {}",
            size,
            digits
        );

        assert!(
            rows * digits <= size,
            "invalid ggsw: rows: {} * digits:{} > ceil(k/basek): {}",
            rows,
            digits,
            size
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
        self.basek
    }

    fn k(&self) -> usize {
        self.k
    }
}

impl<D: Data> GGSWCiphertext<D> {
    pub fn rank(&self) -> usize {
        self.data.cols_out() - 1
    }

    pub fn digits(&self) -> usize {
        self.digits
    }
}

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

impl<D: DataMut> ReaderFrom for GGSWCiphertext<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.k = reader.read_u64::<LittleEndian>()? as usize;
        self.basek = reader.read_u64::<LittleEndian>()? as usize;
        self.digits = reader.read_u64::<LittleEndian>()? as usize;
        self.data.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GGSWCiphertext<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.k as u64)?;
        writer.write_u64::<LittleEndian>(self.basek as u64)?;
        writer.write_u64::<LittleEndian>(self.digits as u64)?;
        self.data.write_to(writer)
    }
}
