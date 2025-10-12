use poulpy_hal::{
    layouts::{Data, DataMut, DataRef, FillUniform, MatZnx, MatZnxToMut, MatZnxToRef, ReaderFrom, WriterTo, ZnxInfos},
    source::Source,
};

use crate::layouts::{Base2K, BuildError, Degree, Dnum, Dsize, GLWECiphertext, GLWEInfos, LWEInfos, Rank, TorusPrecision};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use std::fmt;

pub trait GGLWEInfos
where
    Self: GLWEInfos,
{
    fn dnum(&self) -> Dnum;
    fn dsize(&self) -> Dsize;
    fn rank_in(&self) -> Rank;
    fn rank_out(&self) -> Rank;
    fn layout(&self) -> GGLWECiphertextLayout {
        GGLWECiphertextLayout {
            n: self.n(),
            base2k: self.base2k(),
            k: self.k(),
            rank_in: self.rank_in(),
            rank_out: self.rank_out(),
            dsize: self.dsize(),
            dnum: self.dnum(),
        }
    }
}

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct GGLWECiphertextLayout {
    pub n: Degree,
    pub base2k: Base2K,
    pub k: TorusPrecision,
    pub rank_in: Rank,
    pub rank_out: Rank,
    pub dnum: Dnum,
    pub dsize: Dsize,
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

    fn dsize(&self) -> Dsize {
        self.dsize
    }

    fn rank_out(&self) -> Rank {
        self.rank_out
    }

    fn dnum(&self) -> Dnum {
        self.dnum
    }
}

#[derive(PartialEq, Eq, Clone)]
pub struct GGLWECiphertext<D: Data> {
    pub(crate) data: MatZnx<D>,
    pub(crate) k: TorusPrecision,
    pub(crate) base2k: Base2K,
    pub(crate) dsize: Dsize,
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

    fn dsize(&self) -> Dsize {
        self.dsize
    }

    fn dnum(&self) -> Dnum {
        Dnum(self.data.rows() as u32)
    }
}

pub struct GGLWECiphertextBuilder<D: Data> {
    data: Option<MatZnx<D>>,
    base2k: Option<Base2K>,
    k: Option<TorusPrecision>,
    dsize: Option<Dsize>,
}

impl<D: Data> GGLWECiphertext<D> {
    #[inline]
    pub fn builder() -> GGLWECiphertextBuilder<D> {
        GGLWECiphertextBuilder {
            data: None,
            base2k: None,
            k: None,
            dsize: None,
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
            infos.dnum().into(),
            infos.rank_in().into(),
            (infos.rank_out() + 1).into(),
            infos.size(),
        ));
        self.base2k = Some(infos.base2k());
        self.k = Some(infos.k());
        self.dsize = Some(infos.dsize());
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
    pub fn dsize(mut self, dsize: Dsize) -> Self {
        self.dsize = Some(dsize);
        self
    }

    pub fn build(self) -> Result<GGLWECiphertext<D>, BuildError> {
        let data: MatZnx<D> = self.data.ok_or(BuildError::MissingData)?;
        let base2k: Base2K = self.base2k.ok_or(BuildError::MissingBase2K)?;
        let k: TorusPrecision = self.k.ok_or(BuildError::MissingK)?;
        let dsize: Dsize = self.dsize.ok_or(BuildError::MissingDigits)?;

        if base2k == 0_u32 {
            return Err(BuildError::ZeroBase2K);
        }

        if dsize == 0_u32 {
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
            dsize,
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
            "(GGLWECiphertext: k={} base2k={} dsize={}) {}",
            self.k().0,
            self.base2k().0,
            self.dsize().0,
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
            infos.rank_in(),
            infos.rank_out(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    pub fn alloc_with(
        n: Degree,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> Self {
        let size: usize = k.0.div_ceil(base2k.0) as usize;
        debug_assert!(
            size as u32 > dsize.0,
            "invalid gglwe: ceil(k/base2k): {size} <= dsize: {}",
            dsize.0
        );

        assert!(
            dnum.0 * dsize.0 <= size as u32,
            "invalid gglwe: dnum: {} * dsize:{} > ceil(k/base2k): {size}",
            dnum.0,
            dsize.0,
        );

        Self {
            data: MatZnx::alloc(
                n.into(),
                dnum.into(),
                rank_in.into(),
                (rank_out + 1).into(),
                k.0.div_ceil(base2k.0) as usize,
            ),
            k,
            base2k,
            dsize,
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
            infos.rank_in(),
            infos.rank_out(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    pub fn alloc_bytes_with(
        n: Degree,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> usize {
        let size: usize = k.0.div_ceil(base2k.0) as usize;
        debug_assert!(
            size as u32 > dsize.0,
            "invalid gglwe: ceil(k/base2k): {size} <= dsize: {}",
            dsize.0
        );

        assert!(
            dnum.0 * dsize.0 <= size as u32,
            "invalid gglwe: dnum: {} * dsize:{} > ceil(k/base2k): {size}",
            dnum.0,
            dsize.0,
        );

        MatZnx::alloc_bytes(
            n.into(),
            dnum.into(),
            rank_in.into(),
            (rank_out + 1).into(),
            k.0.div_ceil(base2k.0) as usize,
        )
    }
}

pub trait GGLWECiphertextToMut {
    fn to_mut(&mut self) -> GGLWECiphertext<&mut [u8]>;
}

impl<D: DataMut> GGLWECiphertextToMut for GGLWECiphertext<D> {
    fn to_mut(&mut self) -> GGLWECiphertext<&mut [u8]> {
        GGLWECiphertext {
            k: self.k(),
            base2k: self.base2k(),
            dsize: self.dsize(),
            data: self.data.to_mut(),
        }
    }
}

pub trait GGLWECiphertextToRef {
    fn to_ref(&self) -> GGLWECiphertext<&[u8]>;
}

impl<D: DataMut> GGLWECiphertextToRef for GGLWECiphertext<D> {
    fn to_ref(&self) -> GGLWECiphertext<&[u8]> {
        GGLWECiphertext {
            k: self.k(),
            base2k: self.base2k(),
            dsize: self.dsize(),
            data: self.data.to_ref(),
        }
    }
}

impl<D: DataMut> ReaderFrom for GGLWECiphertext<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.k = TorusPrecision(reader.read_u32::<LittleEndian>()?);
        self.base2k = Base2K(reader.read_u32::<LittleEndian>()?);
        self.dsize = Dsize(reader.read_u32::<LittleEndian>()?);
        self.data.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GGLWECiphertext<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u32::<LittleEndian>(self.k.0)?;
        writer.write_u32::<LittleEndian>(self.base2k.0)?;
        writer.write_u32::<LittleEndian>(self.dsize.0)?;
        self.data.write_to(writer)
    }
}
