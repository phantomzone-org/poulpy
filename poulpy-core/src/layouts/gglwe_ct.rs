use poulpy_hal::{
    layouts::{Data, DataMut, DataRef, FillUniform, MatZnx, ReaderFrom, Reset, WriterTo},
    source::Source,
};

use crate::layouts::{GLWECiphertext, Infos};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use std::fmt;

#[derive(PartialEq, Eq, Clone)]
pub struct GGLWECiphertext<D: Data> {
    pub(crate) data: MatZnx<D>,
    pub(crate) basek: usize,
    pub(crate) k: usize,
    pub(crate) digits: usize,
}

impl<D: DataRef> fmt::Debug for GGLWECiphertext<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl<D: DataMut> FillUniform for GGLWECiphertext<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.data.fill_uniform(log_bound, source);
    }
}

impl<D: DataMut> Reset for GGLWECiphertext<D> {
    fn reset(&mut self) {
        self.data.reset();
        self.basek = 0;
        self.k = 0;
        self.digits = 0;
    }
}

impl<D: DataRef> fmt::Display for GGLWECiphertext<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "(GGLWECiphertext: basek={} k={} digits={}) {}",
            self.basek, self.k, self.digits, self.data
        )
    }
}

impl<D: DataRef> GGLWECiphertext<D> {
    pub fn at(&self, row: usize, col: usize) -> GLWECiphertext<&[u8]> {
        GLWECiphertext {
            data: self.data.at(row, col),
            basek: self.basek,
            k: self.k,
        }
    }
}

impl<D: DataMut> GGLWECiphertext<D> {
    pub fn at_mut(&mut self, row: usize, col: usize) -> GLWECiphertext<&mut [u8]> {
        GLWECiphertext {
            data: self.data.at_mut(row, col),
            basek: self.basek,
            k: self.k,
        }
    }
}

impl GGLWECiphertext<Vec<u8>> {
    pub fn alloc(n: usize, basek: usize, k: usize, rows: usize, digits: usize, rank_in: usize, rank_out: usize) -> Self {
        let size: usize = k.div_ceil(basek);
        debug_assert!(
            size > digits,
            "invalid gglwe: ceil(k/basek): {} <= digits: {}",
            size,
            digits
        );

        assert!(
            rows * digits <= size,
            "invalid gglwe: rows: {} * digits:{} > ceil(k/basek): {}",
            rows,
            digits,
            size
        );

        Self {
            data: MatZnx::alloc(n, rows, rank_in, rank_out + 1, size),
            basek,
            k,
            digits,
        }
    }

    pub fn bytes_of(n: usize, basek: usize, k: usize, rows: usize, digits: usize, rank_in: usize, rank_out: usize) -> usize {
        let size: usize = k.div_ceil(basek);
        debug_assert!(
            size > digits,
            "invalid gglwe: ceil(k/basek): {} <= digits: {}",
            size,
            digits
        );

        assert!(
            rows * digits <= size,
            "invalid gglwe: rows: {} * digits:{} > ceil(k/basek): {}",
            rows,
            digits,
            size
        );

        MatZnx::alloc_bytes(n, rows, rank_in, rank_out + 1, rows)
    }
}

impl<D: Data> Infos for GGLWECiphertext<D> {
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

impl<D: Data> GGLWECiphertext<D> {
    pub fn rank(&self) -> usize {
        self.data.cols_out() - 1
    }

    pub fn digits(&self) -> usize {
        self.digits
    }

    pub fn rank_in(&self) -> usize {
        self.data.cols_in()
    }

    pub fn rank_out(&self) -> usize {
        self.data.cols_out() - 1
    }
}

impl<D: DataMut> ReaderFrom for GGLWECiphertext<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.k = reader.read_u64::<LittleEndian>()? as usize;
        self.basek = reader.read_u64::<LittleEndian>()? as usize;
        self.digits = reader.read_u64::<LittleEndian>()? as usize;
        self.data.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GGLWECiphertext<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.k as u64)?;
        writer.write_u64::<LittleEndian>(self.basek as u64)?;
        writer.write_u64::<LittleEndian>(self.digits as u64)?;
        self.data.write_to(writer)
    }
}
