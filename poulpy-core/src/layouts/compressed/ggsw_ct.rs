use poulpy_hal::{
    api::{VecZnxCopy, VecZnxFillUniform},
    layouts::{Backend, Data, DataMut, DataRef, FillUniform, MatZnx, Module, ReaderFrom, Reset, WriterTo},
    source::Source,
};

use crate::layouts::{
    GGSWCiphertext, Infos,
    compressed::{Decompress, GLWECiphertextCompressed},
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fmt;

#[derive(PartialEq, Eq, Clone)]
pub struct GGSWCiphertextCompressed<D: Data> {
    pub(crate) data: MatZnx<D>,
    pub(crate) basek: usize,
    pub(crate) k: usize,
    pub(crate) digits: usize,
    pub(crate) rank: usize,
    pub(crate) seed: Vec<[u8; 32]>,
}

impl<D: DataRef> fmt::Debug for GGSWCiphertextCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.data)
    }
}

impl<D: DataRef> fmt::Display for GGSWCiphertextCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "(GGSWCiphertextCompressed: basek={} k={} digits={}) {}",
            self.basek, self.k, self.digits, self.data
        )
    }
}

impl<D: DataMut> Reset for GGSWCiphertextCompressed<D> {
    fn reset(&mut self) {
        self.data.reset();
        self.basek = 0;
        self.k = 0;
        self.digits = 0;
        self.rank = 0;
        self.seed = Vec::new();
    }
}

impl<D: DataMut> FillUniform for GGSWCiphertextCompressed<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.data.fill_uniform(log_bound, source);
    }
}

impl GGSWCiphertextCompressed<Vec<u8>> {
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
            data: MatZnx::alloc(n, rows, rank + 1, 1, k.div_ceil(basek)),
            basek,
            k,
            digits,
            rank,
            seed: Vec::new(),
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

        MatZnx::alloc_bytes(n, rows, rank + 1, 1, size)
    }
}

impl<D: DataRef> GGSWCiphertextCompressed<D> {
    pub fn at(&self, row: usize, col: usize) -> GLWECiphertextCompressed<&[u8]> {
        GLWECiphertextCompressed {
            data: self.data.at(row, col),
            basek: self.basek,
            k: self.k,
            rank: self.rank(),
            seed: self.seed[row * (self.rank() + 1) + col],
        }
    }
}

impl<D: DataMut> GGSWCiphertextCompressed<D> {
    pub fn at_mut(&mut self, row: usize, col: usize) -> GLWECiphertextCompressed<&mut [u8]> {
        let rank: usize = self.rank();
        GLWECiphertextCompressed {
            data: self.data.at_mut(row, col),
            basek: self.basek,
            k: self.k,
            rank,
            seed: self.seed[row * (rank + 1) + col],
        }
    }
}

impl<D: Data> Infos for GGSWCiphertextCompressed<D> {
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

impl<D: Data> GGSWCiphertextCompressed<D> {
    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn digits(&self) -> usize {
        self.digits
    }
}

impl<D: DataMut> ReaderFrom for GGSWCiphertextCompressed<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.k = reader.read_u64::<LittleEndian>()? as usize;
        self.basek = reader.read_u64::<LittleEndian>()? as usize;
        self.digits = reader.read_u64::<LittleEndian>()? as usize;
        self.rank = reader.read_u64::<LittleEndian>()? as usize;
        let seed_len = reader.read_u64::<LittleEndian>()? as usize;
        self.seed = vec![[0u8; 32]; seed_len];
        for s in &mut self.seed {
            reader.read_exact(s)?;
        }
        self.data.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GGSWCiphertextCompressed<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.k as u64)?;
        writer.write_u64::<LittleEndian>(self.basek as u64)?;
        writer.write_u64::<LittleEndian>(self.digits as u64)?;
        writer.write_u64::<LittleEndian>(self.rank as u64)?;
        writer.write_u64::<LittleEndian>(self.seed.len() as u64)?;
        for s in &self.seed {
            writer.write_all(s)?;
        }
        self.data.write_to(writer)
    }
}

impl<D: DataMut, B: Backend, DR: DataRef> Decompress<B, GGSWCiphertextCompressed<DR>> for GGSWCiphertext<D>
where
    Module<B>: VecZnxFillUniform + VecZnxCopy,
{
    fn decompress(&mut self, module: &Module<B>, other: &GGSWCiphertextCompressed<DR>) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.rank(), other.rank())
        }

        let rows: usize = self.rows();
        let rank: usize = self.rank();
        (0..rows).for_each(|row_i| {
            (0..rank + 1).for_each(|col_j| {
                self.at_mut(row_i, col_j)
                    .decompress(module, &other.at(row_i, col_j));
            });
        });
    }
}
