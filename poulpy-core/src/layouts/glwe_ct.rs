use poulpy_hal::{
    layouts::{
        Data, DataMut, DataRef, FillUniform, ReaderFrom, ToOwnedDeep, VecZnx, VecZnxToMut, VecZnxToRef, WriterTo, ZnxInfos,
    },
    source::Source,
};

use crate::layouts::{Base2K, BuildError, Degree, LWEInfos, Rank, TorusPrecision};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fmt;

pub trait GLWEInfos
where
    Self: LWEInfos,
{
    fn rank(&self) -> Rank;
    fn glwe_layout(&self) -> GLWECiphertextLayout {
        GLWECiphertextLayout {
            n: self.n(),
            base2k: self.base2k(),
            k: self.k(),
            rank: self.rank(),
        }
    }
}

pub trait GLWELayoutSet {
    fn set_k(&mut self, k: TorusPrecision);
    fn set_basek(&mut self, base2k: Base2K);
}

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct GLWECiphertextLayout {
    pub n: Degree,
    pub base2k: Base2K,
    pub k: TorusPrecision,
    pub rank: Rank,
}

impl LWEInfos for GLWECiphertextLayout {
    fn n(&self) -> Degree {
        self.n
    }

    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }
}

impl GLWEInfos for GLWECiphertextLayout {
    fn rank(&self) -> Rank {
        self.rank
    }
}

#[derive(PartialEq, Eq, Clone)]
pub struct GLWECiphertext<D: Data> {
    pub(crate) data: VecZnx<D>,
    pub(crate) base2k: Base2K,
    pub(crate) k: TorusPrecision,
}

impl<D: DataMut> GLWELayoutSet for GLWECiphertext<D> {
    fn set_basek(&mut self, base2k: Base2K) {
        self.base2k = base2k
    }

    fn set_k(&mut self, k: TorusPrecision) {
        self.k = k
    }
}

impl<D: DataRef> GLWECiphertext<D> {
    pub fn data(&self) -> &VecZnx<D> {
        &self.data
    }
}

impl<D: DataMut> GLWECiphertext<D> {
    pub fn data_mut(&mut self) -> &mut VecZnx<D> {
        &mut self.data
    }
}

pub struct GLWECiphertextBuilder<D: Data> {
    data: Option<VecZnx<D>>,
    base2k: Option<Base2K>,
    k: Option<TorusPrecision>,
}

impl<D: Data> GLWECiphertext<D> {
    #[inline]
    pub fn builder() -> GLWECiphertextBuilder<D> {
        GLWECiphertextBuilder {
            data: None,
            base2k: None,
            k: None,
        }
    }
}

impl GLWECiphertextBuilder<Vec<u8>> {
    #[inline]
    pub fn layout<A>(mut self, layout: &A) -> Self
    where
        A: GLWEInfos,
    {
        self.data = Some(VecZnx::alloc(
            layout.n().into(),
            (layout.rank() + 1).into(),
            layout.size(),
        ));
        self.base2k = Some(layout.base2k());
        self.k = Some(layout.k());
        self
    }
}

impl<D: Data> GLWECiphertextBuilder<D> {
    #[inline]
    pub fn data(mut self, data: VecZnx<D>) -> Self {
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

    pub fn build(self) -> Result<GLWECiphertext<D>, BuildError> {
        let data: VecZnx<D> = self.data.ok_or(BuildError::MissingData)?;
        let base2k: Base2K = self.base2k.ok_or(BuildError::MissingBase2K)?;
        let k: TorusPrecision = self.k.ok_or(BuildError::MissingK)?;

        if base2k == 0_u32 {
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

        Ok(GLWECiphertext { data, base2k, k })
    }
}

impl<D: Data> LWEInfos for GLWECiphertext<D> {
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

impl<D: Data> GLWEInfos for GLWECiphertext<D> {
    fn rank(&self) -> Rank {
        Rank(self.data.cols() as u32 - 1)
    }
}

impl<D: DataRef> ToOwnedDeep for GLWECiphertext<D> {
    type Owned = GLWECiphertext<Vec<u8>>;
    fn to_owned_deep(&self) -> Self::Owned {
        GLWECiphertext {
            data: self.data.to_owned_deep(),
            k: self.k,
            base2k: self.base2k,
        }
    }
}

impl<D: DataRef> fmt::Debug for GLWECiphertext<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: DataRef> fmt::Display for GLWECiphertext<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GLWECiphertext: base2k={} k={}: {}",
            self.base2k().0,
            self.k().0,
            self.data
        )
    }
}

impl<D: DataMut> FillUniform for GLWECiphertext<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.data.fill_uniform(log_bound, source);
    }
}

impl GLWECiphertext<Vec<u8>> {
    pub fn alloc<A>(infos: &A) -> Self
    where
        A: GLWEInfos,
    {
        Self::alloc_with(infos.n(), infos.base2k(), infos.k(), infos.rank())
    }

    pub fn alloc_with(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank) -> Self {
        Self {
            data: VecZnx::alloc(n.into(), (rank + 1).into(), k.0.div_ceil(base2k.0) as usize),
            base2k,
            k,
        }
    }

    pub fn alloc_bytes<A>(infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        Self::alloc_bytes_with(infos.n(), infos.base2k(), infos.k(), infos.rank())
    }

    pub fn alloc_bytes_with(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank) -> usize {
        VecZnx::alloc_bytes(n.into(), (rank + 1).into(), k.0.div_ceil(base2k.0) as usize)
    }
}

pub trait GLWECiphertextToRef {
    fn to_ref(&self) -> GLWECiphertext<&[u8]>;
}

impl<D: DataRef> GLWECiphertextToRef for GLWECiphertext<D> {
    fn to_ref(&self) -> GLWECiphertext<&[u8]> {
        GLWECiphertext::builder()
            .k(self.k())
            .base2k(self.base2k())
            .data(self.data.to_ref())
            .build()
            .unwrap()
    }
}

pub trait GLWECiphertextToMut {
    fn to_mut(&mut self) -> GLWECiphertext<&mut [u8]>;
}

impl<D: DataMut> GLWECiphertextToMut for GLWECiphertext<D> {
    fn to_mut(&mut self) -> GLWECiphertext<&mut [u8]> {
        GLWECiphertext::builder()
            .k(self.k())
            .base2k(self.base2k())
            .data(self.data.to_mut())
            .build()
            .unwrap()
    }
}

impl<D: DataMut> ReaderFrom for GLWECiphertext<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.k = TorusPrecision(reader.read_u32::<LittleEndian>()?);
        self.base2k = Base2K(reader.read_u32::<LittleEndian>()?);
        self.data.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GLWECiphertext<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u32::<LittleEndian>(self.k.0)?;
        writer.write_u32::<LittleEndian>(self.base2k.0)?;
        self.data.write_to(writer)
    }
}
