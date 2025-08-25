use poulpy_hal::{
    api::{VecZnxCopy, VecZnxFillUniform},
    layouts::{Backend, Data, DataMut, DataRef, FillUniform, MatZnx, Module, ReaderFrom, Reset, WriterTo},
    source::Source,
};

use crate::layouts::{
    GGLWECiphertext, Infos,
    compressed::{Decompress, GLWECiphertextCompressed},
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fmt;

#[derive(PartialEq, Eq, Clone)]
pub struct GGLWECiphertextCompressed<D: Data> {
    pub(crate) data: MatZnx<D>,
    pub(crate) basek: usize,
    pub(crate) k: usize,
    pub(crate) rank_out: usize,
    pub(crate) digits: usize,
    pub(crate) seed: Vec<[u8; 32]>,
}

impl<D: DataRef> fmt::Debug for GGLWECiphertextCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl<D: DataMut> FillUniform for GGLWECiphertextCompressed<D> {
    fn fill_uniform(&mut self, source: &mut Source) {
        self.data.fill_uniform(source);
    }
}

impl<D: DataMut> Reset for GGLWECiphertextCompressed<D>
where
    MatZnx<D>: Reset,
{
    fn reset(&mut self) {
        self.data.reset();
        self.basek = 0;
        self.k = 0;
        self.digits = 0;
        self.rank_out = 0;
        self.seed = Vec::new();
    }
}

impl<D: DataRef> fmt::Display for GGLWECiphertextCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "(GGLWECiphertextCompressed: basek={} k={} digits={}) {}",
            self.basek, self.k, self.digits, self.data
        )
    }
}

impl GGLWECiphertextCompressed<Vec<u8>> {
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
            data: MatZnx::alloc(n, rows, rank_in, 1, size),
            basek,
            k,
            rank_out,
            digits,
            seed: vec![[0u8; 32]; rows * rank_in],
        }
    }

    pub fn bytes_of(n: usize, basek: usize, k: usize, rows: usize, digits: usize, rank_in: usize) -> usize {
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

        MatZnx::alloc_bytes(n, rows, rank_in, 1, rows)
    }
}

impl<D: Data> Infos for GGLWECiphertextCompressed<D> {
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

impl<D: Data> GGLWECiphertextCompressed<D> {
    pub fn rank(&self) -> usize {
        self.rank_out
    }

    pub fn digits(&self) -> usize {
        self.digits
    }

    pub fn rank_in(&self) -> usize {
        self.data.cols_in()
    }

    pub fn rank_out(&self) -> usize {
        self.rank_out
    }
}

impl<D: DataRef> GGLWECiphertextCompressed<D> {
    pub(crate) fn at(&self, row: usize, col: usize) -> GLWECiphertextCompressed<&[u8]> {
        GLWECiphertextCompressed {
            data: self.data.at(row, col),
            basek: self.basek,
            k: self.k,
            rank: self.rank_out,
            seed: self.seed[self.rank_in() * row + col],
        }
    }
}

impl<D: DataMut> GGLWECiphertextCompressed<D> {
    pub(crate) fn at_mut(&mut self, row: usize, col: usize) -> GLWECiphertextCompressed<&mut [u8]> {
        let rank_in: usize = self.rank_in();
        GLWECiphertextCompressed {
            data: self.data.at_mut(row, col),
            basek: self.basek,
            k: self.k,
            rank: self.rank_out,
            seed: self.seed[rank_in * row + col], // Warning: value is copied and not borrow mut
        }
    }
}

impl<D: DataMut> ReaderFrom for GGLWECiphertextCompressed<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.k = reader.read_u64::<LittleEndian>()? as usize;
        self.basek = reader.read_u64::<LittleEndian>()? as usize;
        self.digits = reader.read_u64::<LittleEndian>()? as usize;
        self.rank_out = reader.read_u64::<LittleEndian>()? as usize;
        let seed_len = reader.read_u64::<LittleEndian>()? as usize;
        self.seed = vec![[0u8; 32]; seed_len];
        for s in &mut self.seed {
            reader.read_exact(s)?;
        }
        self.data.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GGLWECiphertextCompressed<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.k as u64)?;
        writer.write_u64::<LittleEndian>(self.basek as u64)?;
        writer.write_u64::<LittleEndian>(self.digits as u64)?;
        writer.write_u64::<LittleEndian>(self.rank_out as u64)?;
        writer.write_u64::<LittleEndian>(self.seed.len() as u64)?;
        for s in &self.seed {
            writer.write_all(s)?;
        }
        self.data.write_to(writer)
    }
}

impl<D: DataMut, B: Backend, DR: DataRef> Decompress<B, GGLWECiphertextCompressed<DR>> for GGLWECiphertext<D>
where
    Module<B>: VecZnxFillUniform + VecZnxCopy,
{
    fn decompress(&mut self, module: &Module<B>, other: &GGLWECiphertextCompressed<DR>) {
        #[cfg(debug_assertions)]
        {
            use poulpy_hal::layouts::ZnxInfos;

            assert_eq!(
                self.n(),
                other.data.n(),
                "invalid receiver: self.n()={} != other.n()={}",
                self.n(),
                other.data.n()
            );
            assert_eq!(
                self.size(),
                other.size(),
                "invalid receiver: self.size()={} != other.size()={}",
                self.size(),
                other.size()
            );
            assert_eq!(
                self.rank_in(),
                other.rank_in(),
                "invalid receiver: self.rank_in()={} != other.rank_in()={}",
                self.rank_in(),
                other.rank_in()
            );
            assert_eq!(
                self.rank_out(),
                other.rank_out(),
                "invalid receiver: self.rank_out()={} != other.rank_out()={}",
                self.rank_out(),
                other.rank_out()
            );

            assert_eq!(
                self.rows(),
                other.rows(),
                "invalid receiver: self.rows()={} != other.rows()={}",
                self.rows(),
                other.rows()
            );
        }

        let rank_in: usize = self.rank_in();
        let rows: usize = self.rows();

        (0..rank_in).for_each(|col_i| {
            (0..rows).for_each(|row_i| {
                self.at_mut(row_i, col_i)
                    .decompress(module, &other.at(row_i, col_i));
            });
        });
    }
}
