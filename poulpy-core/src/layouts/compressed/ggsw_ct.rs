use poulpy_hal::{
    api::{VecZnxCopy, VecZnxFillUniform},
    layouts::{Backend, Data, DataMut, DataRef, FillUniform, MatZnx, Module, ReaderFrom, WriterTo, ZnxInfos},
    source::Source,
};

use crate::layouts::{
    Base2K, Degree, Digits, GGSWCiphertext, GGSWInfos, GLWEInfos, LWEInfos, Rank, Rows, TorusPrecision,
    compressed::{Decompress, GLWECiphertextCompressed},
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fmt;

#[derive(PartialEq, Eq, Clone)]
pub struct GGSWCiphertextCompressed<D: Data> {
    pub(crate) data: MatZnx<D>,
    pub(crate) k: TorusPrecision,
    pub(crate) base2k: Base2K,
    pub(crate) digits: Digits,
    pub(crate) rank: Rank,
    pub(crate) seed: Vec<[u8; 32]>,
}

impl<D: Data> LWEInfos for GGSWCiphertextCompressed<D> {
    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }
    fn size(&self) -> usize {
        self.data.size()
    }
}
impl<D: Data> GLWEInfos for GGSWCiphertextCompressed<D> {
    fn rank(&self) -> Rank {
        self.rank
    }
}

impl<D: Data> GGSWInfos for GGSWCiphertextCompressed<D> {
    fn digits(&self) -> Digits {
        self.digits
    }

    fn rows(&self) -> Rows {
        Rows(self.data.rows() as u32)
    }
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
            "(GGSWCiphertextCompressed: base2k={} k={} digits={}) {}",
            self.base2k, self.k, self.digits, self.data
        )
    }
}

impl<D: DataMut> FillUniform for GGSWCiphertextCompressed<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.data.fill_uniform(log_bound, source);
    }
}

impl GGSWCiphertextCompressed<Vec<u8>> {
    pub fn alloc<A>(infos: &A) -> Self
    where
        A: GGSWInfos,
    {
        Self::alloc_with(
            infos.n(),
            infos.base2k(),
            infos.k(),
            infos.rows(),
            infos.digits(),
            infos.rank(),
        )
    }

    pub fn alloc_with(n: Degree, base2k: Base2K, k: TorusPrecision, rows: Rows, digits: Digits, rank: Rank) -> Self {
        let size: usize = k.0.div_ceil(base2k.0) as usize;
        debug_assert!(
            size as u32 > digits.0,
            "invalid ggsw: ceil(k/base2k): {size} <= digits: {}",
            digits.0
        );

        assert!(
            rows.0 * digits.0 <= size as u32,
            "invalid ggsw: rows: {} * digits:{} > ceil(k/base2k): {size}",
            rows.0,
            digits.0,
        );

        Self {
            data: MatZnx::alloc(
                n.into(),
                rows.into(),
                (rank + 1).into(),
                1,
                k.0.div_ceil(base2k.0) as usize,
            ),
            k,
            base2k,
            digits,
            rank,
            seed: Vec::new(),
        }
    }

    pub fn alloc_bytes<A>(infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        Self::alloc_bytes_with(
            infos.n(),
            infos.base2k(),
            infos.k(),
            infos.rows(),
            infos.digits(),
            infos.rank(),
        )
    }

    pub fn alloc_bytes_with(n: Degree, base2k: Base2K, k: TorusPrecision, rows: Rows, digits: Digits, rank: Rank) -> usize {
        let size: usize = k.0.div_ceil(base2k.0) as usize;
        debug_assert!(
            size as u32 > digits.0,
            "invalid ggsw: ceil(k/base2k): {size} <= digits: {}",
            digits.0
        );

        assert!(
            rows.0 * digits.0 <= size as u32,
            "invalid ggsw: rows: {} * digits:{} > ceil(k/base2k): {size}",
            rows.0,
            digits.0,
        );

        MatZnx::alloc_bytes(
            n.into(),
            rows.into(),
            (rank + 1).into(),
            1,
            k.0.div_ceil(base2k.0) as usize,
        )
    }
}

impl<D: DataRef> GGSWCiphertextCompressed<D> {
    pub fn at(&self, row: usize, col: usize) -> GLWECiphertextCompressed<&[u8]> {
        let rank: usize = self.rank().into();
        GLWECiphertextCompressed {
            data: self.data.at(row, col),
            k: self.k,
            base2k: self.base2k,
            rank: self.rank,
            seed: self.seed[row * (rank + 1) + col],
        }
    }
}

impl<D: DataMut> GGSWCiphertextCompressed<D> {
    pub fn at_mut(&mut self, row: usize, col: usize) -> GLWECiphertextCompressed<&mut [u8]> {
        let rank: usize = self.rank().into();
        GLWECiphertextCompressed {
            data: self.data.at_mut(row, col),
            k: self.k,
            base2k: self.base2k,
            rank: self.rank,
            seed: self.seed[row * (rank + 1) + col],
        }
    }
}

impl<D: DataMut> ReaderFrom for GGSWCiphertextCompressed<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.k = TorusPrecision(reader.read_u32::<LittleEndian>()?);
        self.base2k = Base2K(reader.read_u32::<LittleEndian>()?);
        self.digits = Digits(reader.read_u32::<LittleEndian>()?);
        self.rank = Rank(reader.read_u32::<LittleEndian>()?);
        let seed_len: usize = reader.read_u32::<LittleEndian>()? as usize;
        self.seed = vec![[0u8; 32]; seed_len];
        for s in &mut self.seed {
            reader.read_exact(s)?;
        }
        self.data.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GGSWCiphertextCompressed<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u32::<LittleEndian>(self.k.into())?;
        writer.write_u32::<LittleEndian>(self.base2k.into())?;
        writer.write_u32::<LittleEndian>(self.digits.into())?;
        writer.write_u32::<LittleEndian>(self.rank.into())?;
        writer.write_u32::<LittleEndian>(self.seed.len() as u32)?;
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

        let rows: usize = self.rows().into();
        let rank: usize = self.rank().into();
        (0..rows).for_each(|row_i| {
            (0..rank + 1).for_each(|col_j| {
                self.at_mut(row_i, col_j)
                    .decompress(module, &other.at(row_i, col_j));
            });
        });
    }
}
