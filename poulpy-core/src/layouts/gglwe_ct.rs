use poulpy_hal::{
    layouts::{Data, DataMut, DataRef, FillUniform, MatZnx, ReaderFrom, WriterTo, ZnxInfos},
    source::Source,
};

use crate::layouts::{Base2K, BuildError, Degree, Digits, GLWECiphertext, GLWEInfos, LWEInfos, Rank, Rows, TorusPrecision};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use std::fmt;

pub trait GGLWEInfos
where
    Self: GLWEInfos,
{
    fn rows(&self) -> Rows;
    fn digits(&self) -> Digits;
    fn rank_in(&self) -> Rank;
    fn rank_out(&self) -> Rank;
    fn layout(&self) -> GGLWECiphertextLayout {
        GGLWECiphertextLayout {
            n: self.n(),
            base2k: self.base2k(),
            k: self.k(),
            rank_in: self.rank_in(),
            rank_out: self.rank_out(),
            digits: self.digits(),
            rows: self.rows(),
        }
    }
}

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct GGLWECiphertextLayout {
    pub n: Degree,
    pub base2k: Base2K,
    pub k: TorusPrecision,
    pub rows: Rows,
    pub digits: Digits,
    pub rank_in: Rank,
    pub rank_out: Rank,
}

impl LWEInfos for GGLWECiphertextLayout {
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

impl GLWEInfos for GGLWECiphertextLayout {
    fn rank(&self) -> Rank {
        self.rank_out
    }
}

impl GGLWEInfos for GGLWECiphertextLayout {
    fn rank_in(&self) -> Rank {
        self.rank_in
    }

    fn digits(&self) -> Digits {
        self.digits
    }

    fn rank_out(&self) -> Rank {
        self.rank_out
    }

    fn rows(&self) -> Rows {
        self.rows
    }
}

#[derive(PartialEq, Eq, Clone)]
pub struct GGLWECiphertext<D: Data> {
    pub(crate) data: MatZnx<D>,
    pub(crate) k: TorusPrecision,
    pub(crate) base2k: Base2K,
    pub(crate) digits: Digits,
}

impl<D: Data> LWEInfos for GGLWECiphertext<D> {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }

    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    fn size(&self) -> usize {
        self.data.size()
    }
}

impl<D: Data> GLWEInfos for GGLWECiphertext<D> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data> GGLWEInfos for GGLWECiphertext<D> {
    fn rank_in(&self) -> Rank {
        Rank(self.data.cols_in() as u32)
    }

    fn rank_out(&self) -> Rank {
        Rank(self.data.cols_out() as u32 - 1)
    }

    fn digits(&self) -> Digits {
        self.digits
    }

    fn rows(&self) -> Rows {
        Rows(self.data.rows() as u32)
    }
}

pub struct GGLWECiphertextBuilder<D: Data> {
    data: Option<MatZnx<D>>,
    base2k: Option<Base2K>,
    k: Option<TorusPrecision>,
    digits: Option<Digits>,
}

impl<D: Data> GGLWECiphertext<D> {
    #[inline]
    pub fn builder() -> GGLWECiphertextBuilder<D> {
        GGLWECiphertextBuilder {
            data: None,
            base2k: None,
            k: None,
            digits: None,
        }
    }
}

impl GGLWECiphertextBuilder<Vec<u8>> {
    #[inline]
    pub fn layout<A>(mut self, infos: &A) -> Self
    where
        A: GGLWEInfos,
    {
        self.data = Some(MatZnx::alloc(
            infos.n().into(),
            infos.rows().into(),
            infos.rank_in().into(),
            (infos.rank_out() + 1).into(),
            infos.size(),
        ));
        self.base2k = Some(infos.base2k());
        self.k = Some(infos.k());
        self.digits = Some(infos.digits());
        self
    }
}

impl<D: Data> GGLWECiphertextBuilder<D> {
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

    pub fn build(self) -> Result<GGLWECiphertext<D>, BuildError> {
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

        Ok(GGLWECiphertext {
            data,
            base2k,
            k,
            digits,
        })
    }
}

impl<D: DataRef> GGLWECiphertext<D> {
    pub fn data(&self) -> &MatZnx<D> {
        &self.data
    }
}

impl<D: DataMut> GGLWECiphertext<D> {
    pub fn data_mut(&mut self) -> &mut MatZnx<D> {
        &mut self.data
    }
}

impl<D: DataRef> fmt::Debug for GGLWECiphertext<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: DataMut> FillUniform for GGLWECiphertext<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.data.fill_uniform(log_bound, source);
    }
}

impl<D: DataRef> fmt::Display for GGLWECiphertext<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "(GGLWECiphertext: k={} base2k={} digits={}) {}",
            self.k().0,
            self.base2k().0,
            self.digits().0,
            self.data
        )
    }
}

impl<D: DataRef> GGLWECiphertext<D> {
    pub fn at(&self, row: usize, col: usize) -> GLWECiphertext<&[u8]> {
        GLWECiphertext::builder()
            .data(self.data.at(row, col))
            .base2k(self.base2k())
            .k(self.k())
            .build()
            .unwrap()
    }
}

impl<D: DataMut> GGLWECiphertext<D> {
    pub fn at_mut(&mut self, row: usize, col: usize) -> GLWECiphertext<&mut [u8]> {
        GLWECiphertext::builder()
            .base2k(self.base2k())
            .k(self.k())
            .data(self.data.at_mut(row, col))
            .build()
            .unwrap()
    }
}

impl GGLWECiphertext<Vec<u8>> {
    pub fn alloc<A>(infos: &A) -> Self
    where
        A: GGLWEInfos,
    {
        Self::alloc_with(
            infos.n(),
            infos.base2k(),
            infos.k(),
            infos.rows(),
            infos.digits(),
            infos.rank_in(),
            infos.rank_out(),
        )
    }

    pub fn alloc_with(
        n: Degree,
        base2k: Base2K,
        k: TorusPrecision,
        rows: Rows,
        digits: Digits,
        rank_in: Rank,
        rank_out: Rank,
    ) -> Self {
        let size: usize = k.0.div_ceil(base2k.0) as usize;
        debug_assert!(
            size as u32 > digits.0,
            "invalid gglwe: ceil(k/base2k): {size} <= digits: {}",
            digits.0
        );

        assert!(
            rows.0 * digits.0 <= size as u32,
            "invalid gglwe: rows: {} * digits:{} > ceil(k/base2k): {size}",
            rows.0,
            digits.0,
        );

        Self {
            data: MatZnx::alloc(
                n.into(),
                rows.into(),
                rank_in.into(),
                (rank_out + 1).into(),
                k.0.div_ceil(base2k.0) as usize,
            ),
            k,
            base2k,
            digits,
        }
    }

    pub fn alloc_bytes<A>(infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        Self::alloc_bytes_with(
            infos.n(),
            infos.base2k(),
            infos.k(),
            infos.rows(),
            infos.digits(),
            infos.rank_in(),
            infos.rank_out(),
        )
    }

    pub fn alloc_bytes_with(
        n: Degree,
        base2k: Base2K,
        k: TorusPrecision,
        rows: Rows,
        digits: Digits,
        rank_in: Rank,
        rank_out: Rank,
    ) -> usize {
        let size: usize = k.0.div_ceil(base2k.0) as usize;
        debug_assert!(
            size as u32 > digits.0,
            "invalid gglwe: ceil(k/base2k): {size} <= digits: {}",
            digits.0
        );

        assert!(
            rows.0 * digits.0 <= size as u32,
            "invalid gglwe: rows: {} * digits:{} > ceil(k/base2k): {size}",
            rows.0,
            digits.0,
        );

        MatZnx::alloc_bytes(
            n.into(),
            rows.into(),
            rank_in.into(),
            (rank_out + 1).into(),
            k.0.div_ceil(base2k.0) as usize,
        )
    }
}

impl<D: DataMut> ReaderFrom for GGLWECiphertext<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.k = TorusPrecision(reader.read_u32::<LittleEndian>()?);
        self.base2k = Base2K(reader.read_u32::<LittleEndian>()?);
        self.digits = Digits(reader.read_u32::<LittleEndian>()?);
        self.data.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GGLWECiphertext<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u32::<LittleEndian>(self.k.0)?;
        writer.write_u32::<LittleEndian>(self.base2k.0)?;
        writer.write_u32::<LittleEndian>(self.digits.0)?;
        self.data.write_to(writer)
    }
}
