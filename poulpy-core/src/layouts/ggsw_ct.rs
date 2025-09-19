use poulpy_hal::{
    layouts::{Data, DataMut, DataRef, FillUniform, MatZnx, ReaderFrom, WriterTo, ZnxInfos},
    source::Source,
};
use std::fmt;

use crate::layouts::{Base2K, BuildError, Degree, Digits, GLWECiphertext, GLWEInfos, LWEInfos, Rank, Rows, TorusPrecision};

pub trait GGSWInfos
where
    Self: GLWEInfos,
{
    fn rows(&self) -> Rows;
    fn digits(&self) -> Digits;
    fn layout(&self) -> GGSWCiphertextLayout {
        GGSWCiphertextLayout {
            n: self.n(),
            base2k: self.base2k(),
            k: self.k(),
            rank: self.rank(),
            rows: self.rows(),
            digits: self.digits(),
        }
    }
}

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct GGSWCiphertextLayout {
    pub n: Degree,
    pub base2k: Base2K,
    pub k: TorusPrecision,
    pub rows: Rows,
    pub digits: Digits,
    pub rank: Rank,
}

impl LWEInfos for GGSWCiphertextLayout {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }

    fn n(&self) -> Degree {
        self.n
    }
}
impl GLWEInfos for GGSWCiphertextLayout {
    fn rank(&self) -> Rank {
        self.rank
    }
}

impl GGSWInfos for GGSWCiphertextLayout {
    fn digits(&self) -> Digits {
        self.digits
    }

    fn rows(&self) -> Rows {
        self.rows
    }
}

#[derive(PartialEq, Eq, Clone)]
pub struct GGSWCiphertext<D: Data> {
    pub(crate) data: MatZnx<D>,
    pub(crate) k: TorusPrecision,
    pub(crate) base2k: Base2K,
    pub(crate) digits: Digits,
}

impl<D: Data> LWEInfos for GGSWCiphertext<D> {
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

impl<D: Data> GLWEInfos for GGSWCiphertext<D> {
    fn rank(&self) -> Rank {
        Rank(self.data.cols_out() as u32 - 1)
    }
}

impl<D: Data> GGSWInfos for GGSWCiphertext<D> {
    fn digits(&self) -> Digits {
        self.digits
    }

    fn rows(&self) -> Rows {
        Rows(self.data.rows() as u32)
    }
}

pub struct GGSWCiphertextBuilder<D: Data> {
    data: Option<MatZnx<D>>,
    base2k: Option<Base2K>,
    k: Option<TorusPrecision>,
    digits: Option<Digits>,
}

impl<D: Data> GGSWCiphertext<D> {
    #[inline]
    pub fn builder() -> GGSWCiphertextBuilder<D> {
        GGSWCiphertextBuilder {
            data: None,
            base2k: None,
            k: None,
            digits: None,
        }
    }
}

impl GGSWCiphertextBuilder<Vec<u8>> {
    #[inline]
    pub fn layout<A>(mut self, infos: &A) -> Self
    where
        A: GGSWInfos,
    {
        debug_assert!(
            infos.size() as u32 > infos.digits().0,
            "invalid ggsw: ceil(k/base2k): {} <= digits: {}",
            infos.size(),
            infos.digits()
        );

        assert!(
            infos.rows().0 * infos.digits().0 <= infos.size() as u32,
            "invalid ggsw: rows: {} * digits:{} > ceil(k/base2k): {}",
            infos.rows(),
            infos.digits(),
            infos.size(),
        );

        self.data = Some(MatZnx::alloc(
            infos.n().into(),
            infos.rows().into(),
            (infos.rank() + 1).into(),
            (infos.rank() + 1).into(),
            infos.size(),
        ));
        self.base2k = Some(infos.base2k());
        self.k = Some(infos.k());
        self.digits = Some(infos.digits());
        self
    }
}

impl<D: Data> GGSWCiphertextBuilder<D> {
    #[inline]
    pub fn data(mut self, data: MatZnx<D>) -> Self {
        self.data = Some(data);
        self
    }
    #[inline]
    pub fn base2k(mut self, base2k: Base2K) -> Self {
        self.base2k = Some(base2k);
        self
    }
    #[inline]
    pub fn k(mut self, k: TorusPrecision) -> Self {
        self.k = Some(k);
        self
    }

    #[inline]
    pub fn digits(mut self, digits: Digits) -> Self {
        self.digits = Some(digits);
        self
    }

    pub fn build(self) -> Result<GGSWCiphertext<D>, BuildError> {
        let data: MatZnx<D> = self.data.ok_or(BuildError::MissingData)?;
        let base2k: Base2K = self.base2k.ok_or(BuildError::MissingBase2K)?;
        let k: TorusPrecision = self.k.ok_or(BuildError::MissingK)?;
        let digits: Digits = self.digits.ok_or(BuildError::MissingDigits)?;

        if base2k == 0_u32 {
            return Err(BuildError::ZeroBase2K);
        }

        if digits == 0_u32 {
            return Err(BuildError::ZeroBase2K);
        }

        if k == 0_u32 {
            return Err(BuildError::ZeroTorusPrecision);
        }

        if data.n() == 0 {
            return Err(BuildError::ZeroDegree);
        }

        if data.cols() == 0 {
            return Err(BuildError::ZeroCols);
        }

        if data.size() == 0 {
            return Err(BuildError::ZeroLimbs);
        }

        Ok(GGSWCiphertext {
            data,
            base2k,
            k,
            digits,
        })
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
            "(GGSWCiphertext: k: {} base2k: {} digits: {}) {}",
            self.k().0,
            self.base2k().0,
            self.digits().0,
            self.data
        )
    }
}

impl<D: DataMut> FillUniform for GGSWCiphertext<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.data.fill_uniform(log_bound, source);
    }
}

impl<D: DataRef> GGSWCiphertext<D> {
    pub fn at(&self, row: usize, col: usize) -> GLWECiphertext<&[u8]> {
        GLWECiphertext::builder()
            .data(self.data.at(row, col))
            .base2k(self.base2k())
            .k(self.k())
            .build()
            .unwrap()
    }
}

impl<D: DataMut> GGSWCiphertext<D> {
    pub fn at_mut(&mut self, row: usize, col: usize) -> GLWECiphertext<&mut [u8]> {
        GLWECiphertext::builder()
            .base2k(self.base2k())
            .k(self.k())
            .data(self.data.at_mut(row, col))
            .build()
            .unwrap()
    }
}

impl GGSWCiphertext<Vec<u8>> {
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
                (rank + 1).into(),
                k.0.div_ceil(base2k.0) as usize,
            ),
            k,
            base2k,
            digits,
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
            (rank + 1).into(),
            k.0.div_ceil(base2k.0) as usize,
        )
    }
}

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

impl<D: DataMut> ReaderFrom for GGSWCiphertext<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.k = TorusPrecision(reader.read_u32::<LittleEndian>()?);
        self.base2k = Base2K(reader.read_u32::<LittleEndian>()?);
        self.digits = Digits(reader.read_u32::<LittleEndian>()?);
        self.data.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GGSWCiphertext<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u32::<LittleEndian>(self.k.into())?;
        writer.write_u32::<LittleEndian>(self.base2k.into())?;
        writer.write_u32::<LittleEndian>(self.digits.into())?;
        self.data.write_to(writer)
    }
}
