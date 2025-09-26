use std::fmt;

use poulpy_hal::{
    layouts::{Data, DataMut, DataRef, FillUniform, ReaderFrom, WriterTo, Zn, ZnToMut, ZnToRef, ZnxInfos},
    source::Source,
};

use crate::layouts::{Base2K, BuildError, Degree, TorusPrecision};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

pub trait LWEInfos {
    fn n(&self) -> Degree;
    fn k(&self) -> TorusPrecision;
    fn max_k(&self) -> TorusPrecision {
        TorusPrecision(self.k().0 * self.size() as u32)
    }
    fn base2k(&self) -> Base2K;
    fn size(&self) -> usize {
        self.k().0.div_ceil(self.base2k().0) as usize
    }
    fn lwe_layout(&self) -> LWECiphertextLayout {
        LWECiphertextLayout {
            n: self.n(),
            k: self.k(),
            base2k: self.base2k(),
        }
    }
}

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct LWECiphertextLayout {
    pub n: Degree,
    pub k: TorusPrecision,
    pub base2k: Base2K,
}

impl LWEInfos for LWECiphertextLayout {
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

#[derive(PartialEq, Eq, Clone)]
pub struct LWECiphertext<D: Data> {
    pub(crate) data: Zn<D>,
    pub(crate) k: TorusPrecision,
    pub(crate) base2k: Base2K,
}

impl<D: Data> LWEInfos for LWECiphertext<D> {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }
    fn n(&self) -> Degree {
        Degree(self.data.n() as u32 - 1)
    }

    fn size(&self) -> usize {
        self.data.size()
    }
}

impl<D: DataRef> LWECiphertext<D> {
    pub fn data(&self) -> &Zn<D> {
        &self.data
    }
}

impl<D: DataMut> LWECiphertext<D> {
    pub fn data_mut(&mut self) -> &Zn<D> {
        &mut self.data
    }
}

impl<D: DataRef> fmt::Debug for LWECiphertext<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: DataRef> fmt::Display for LWECiphertext<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LWECiphertext: base2k={} k={}: {}",
            self.base2k().0,
            self.k().0,
            self.data
        )
    }
}

impl<D: DataMut> FillUniform for LWECiphertext<D>
where
    Zn<D>: FillUniform,
{
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.data.fill_uniform(log_bound, source);
    }
}

impl LWECiphertext<Vec<u8>> {
    pub fn alloc<A>(infos: &A) -> Self
    where
        A: LWEInfos,
    {
        Self::alloc_with(infos.n(), infos.base2k(), infos.k())
    }

    pub fn alloc_with(n: Degree, base2k: Base2K, k: TorusPrecision) -> Self {
        Self {
            data: Zn::alloc((n + 1).into(), 1, k.0.div_ceil(base2k.0) as usize),
            k,
            base2k,
        }
    }

    pub fn alloc_bytes<A>(infos: &A) -> usize
    where
        A: LWEInfos,
    {
        Self::alloc_bytes_with(infos.n(), infos.base2k(), infos.k())
    }

    pub fn alloc_bytes_with(n: Degree, base2k: Base2K, k: TorusPrecision) -> usize {
        Zn::alloc_bytes((n + 1).into(), 1, k.0.div_ceil(base2k.0) as usize)
    }
}

impl LWECiphertextBuilder<Vec<u8>> {
    #[inline]
    pub fn layout<A>(mut self, layout: A) -> Self
    where
        A: LWEInfos,
    {
        self.data = Some(Zn::alloc((layout.n() + 1).into(), 1, layout.size()));
        self.base2k = Some(layout.base2k());
        self.k = Some(layout.k());
        self
    }
}

pub struct LWECiphertextBuilder<D: Data> {
    data: Option<Zn<D>>,
    base2k: Option<Base2K>,
    k: Option<TorusPrecision>,
}

impl<D: Data> LWECiphertext<D> {
    #[inline]
    pub fn builder() -> LWECiphertextBuilder<D> {
        LWECiphertextBuilder {
            data: None,
            base2k: None,
            k: None,
        }
    }
}

impl<D: Data> LWECiphertextBuilder<D> {
    #[inline]
    pub fn data(mut self, data: Zn<D>) -> Self {
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

    pub fn build(self) -> Result<LWECiphertext<D>, BuildError> {
        let data: Zn<D> = self.data.ok_or(BuildError::MissingData)?;
        let base2k: Base2K = self.base2k.ok_or(BuildError::MissingBase2K)?;
        let k: TorusPrecision = self.k.ok_or(BuildError::MissingK)?;

        if base2k.0 == 0 {
            return Err(BuildError::ZeroBase2K);
        }

        if k.0 == 0 {
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

        Ok(LWECiphertext { data, base2k, k })
    }
}

pub trait LWECiphertextToRef {
    fn to_ref(&self) -> LWECiphertext<&[u8]>;
}

impl<D: DataRef> LWECiphertextToRef for LWECiphertext<D> {
    fn to_ref(&self) -> LWECiphertext<&[u8]> {
        LWECiphertext::builder()
            .base2k(self.base2k())
            .k(self.k())
            .data(self.data.to_ref())
            .build()
            .unwrap()
    }
}

pub trait LWECiphertextToMut {
    #[allow(dead_code)]
    fn to_mut(&mut self) -> LWECiphertext<&mut [u8]>;
}

impl<D: DataMut> LWECiphertextToMut for LWECiphertext<D> {
    fn to_mut(&mut self) -> LWECiphertext<&mut [u8]> {
        LWECiphertext::builder()
            .base2k(self.base2k())
            .k(self.k())
            .data(self.data.to_mut())
            .build()
            .unwrap()
    }
}

impl<D: DataMut> ReaderFrom for LWECiphertext<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.k = TorusPrecision(reader.read_u32::<LittleEndian>()?);
        self.base2k = Base2K(reader.read_u32::<LittleEndian>()?);
        self.data.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for LWECiphertext<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u32::<LittleEndian>(self.k.into())?;
        writer.write_u32::<LittleEndian>(self.base2k.into())?;
        self.data.write_to(writer)
    }
}
