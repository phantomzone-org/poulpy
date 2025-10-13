use poulpy_hal::{
    api::{VecZnxCopy, VecZnxFillUniform},
    layouts::{
        Backend, Data, DataMut, DataRef, FillUniform, Module, ReaderFrom, VecZnx, VecZnxToMut, VecZnxToRef, WriterTo, ZnxInfos,
    },
    source::Source,
};

use crate::layouts::{
    Base2K, BuildError, Degree, GLWECiphertext, GLWEInfos, LWEInfos, Rank, TorusPrecision, compressed::Decompress,
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fmt;

#[derive(PartialEq, Eq, Clone)]
pub struct GLWECiphertextCompressed<D: Data> {
    pub(crate) data: VecZnx<D>,
    pub(crate) base2k: Base2K,
    pub(crate) k: TorusPrecision,
    pub(crate) rank: Rank,
    pub(crate) seed: [u8; 32],
}

impl<D: Data> LWEInfos for GLWECiphertextCompressed<D> {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }

    fn size(&self) -> usize {
        self.data.size()
    }

    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }
}
impl<D: Data> GLWEInfos for GLWECiphertextCompressed<D> {
    fn rank(&self) -> Rank {
        self.rank
    }
}

impl<D: DataRef> fmt::Debug for GLWECiphertextCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: DataRef> fmt::Display for GLWECiphertextCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GLWECiphertextCompressed: base2k={} k={} rank={} seed={:?}: {}",
            self.base2k(),
            self.k(),
            self.rank(),
            self.seed,
            self.data
        )
    }
}

impl<D: DataMut> FillUniform for GLWECiphertextCompressed<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.data.fill_uniform(log_bound, source);
    }
}

impl GLWECiphertextCompressed<Vec<u8>> {
    pub fn alloc<A>(infos: &A) -> Self
    where
        A: GLWEInfos,
    {
        Self::alloc_with(infos.n(), infos.base2k(), infos.k(), infos.rank())
    }

    pub fn alloc_with(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank) -> Self {
        Self {
            data: VecZnx::alloc(n.into(), 1, k.0.div_ceil(base2k.0) as usize),
            base2k,
            k,
            rank,
            seed: [0u8; 32],
        }
    }

    pub fn alloc_bytes<A>(infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        Self::alloc_bytes_with(infos.n(), infos.base2k(), infos.k())
    }

    pub fn alloc_bytes_with(n: Degree, base2k: Base2K, k: TorusPrecision) -> usize {
        VecZnx::alloc_bytes(n.into(), 1, k.0.div_ceil(base2k.0) as usize)
    }
}

pub struct GLWECiphertextCompressedBuilder<D: Data> {
    data: Option<VecZnx<D>>,
    base2k: Option<Base2K>,
    k: Option<TorusPrecision>,
    rank: Option<Rank>,
}

impl<D: Data> GLWECiphertextCompressed<D> {
    #[inline]
    pub fn builder() -> GLWECiphertextCompressedBuilder<D> {
        GLWECiphertextCompressedBuilder {
            data: None,
            base2k: None,
            k: None,
            rank: None,
        }
    }
}

impl GLWECiphertextCompressedBuilder<Vec<u8>> {
    #[inline]
    pub fn layout<A>(mut self, layout: &A) -> Self
    where
        A: GLWEInfos,
    {
        self.data = Some(VecZnx::alloc(layout.n().into(), 1, layout.size()));
        self.base2k = Some(layout.base2k());
        self.k = Some(layout.k());
        self.rank = Some(layout.rank());
        self
    }
}

impl<D: Data> GLWECiphertextCompressedBuilder<D> {
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
    #[inline]
    pub fn rank(mut self, rank: Rank) -> Self {
        self.rank = Some(rank);
        self
    }

    pub fn build(self) -> Result<GLWECiphertextCompressed<D>, BuildError> {
        let data: VecZnx<D> = self.data.ok_or(BuildError::MissingData)?;
        let base2k: Base2K = self.base2k.ok_or(BuildError::MissingBase2K)?;
        let k: TorusPrecision = self.k.ok_or(BuildError::MissingK)?;
        let rank: Rank = self.rank.ok_or(BuildError::MissingK)?;

        if base2k == 0_u32 {
            return Err(BuildError::ZeroBase2K);
        }

        if k == 0_u32 {
            return Err(BuildError::ZeroTorusPrecision);
        }

        if rank == 0_u32 {
            return Err(BuildError::ZeroRank);
        }

        if data.n() == 0 {
            return Err(BuildError::ZeroDegree);
        }

        if data.cols() != 1 {
            return Err(BuildError::VecZnxColsNotOne);
        }

        if data.size() == 0 {
            return Err(BuildError::ZeroLimbs);
        }

        Ok(GLWECiphertextCompressed {
            data,
            base2k,
            k,
            rank,
            seed: [0u8; 32],
        })
    }
}

impl<D: DataMut> ReaderFrom for GLWECiphertextCompressed<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.k = TorusPrecision(reader.read_u32::<LittleEndian>()?);
        self.base2k = Base2K(reader.read_u32::<LittleEndian>()?);
        self.rank = Rank(reader.read_u32::<LittleEndian>()?);
        reader.read_exact(&mut self.seed)?;
        self.data.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GLWECiphertextCompressed<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u32::<LittleEndian>(self.k.into())?;
        writer.write_u32::<LittleEndian>(self.base2k.into())?;
        writer.write_u32::<LittleEndian>(self.rank.into())?;
        writer.write_all(&self.seed)?;
        self.data.write_to(writer)
    }
}

impl<D: DataMut, B: Backend, DR: DataRef> Decompress<B, GLWECiphertextCompressed<DR>> for GLWECiphertext<D>
where
    Module<B>: VecZnxFillUniform + VecZnxCopy,
{
    fn decompress(&mut self, module: &Module<B>, other: &GLWECiphertextCompressed<DR>) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(
                self.n(),
                other.n(),
                "invalid receiver: self.n()={} != other.n()={}",
                self.n(),
                other.n()
            );
            assert_eq!(
                self.size(),
                other.size(),
                "invalid receiver: self.size()={} != other.size()={}",
                self.size(),
                other.size()
            );
            assert_eq!(
                self.rank(),
                other.rank(),
                "invalid receiver: self.rank()={} != other.rank()={}",
                self.rank(),
                other.rank()
            );
        }

        let mut source: Source = Source::new(other.seed);
        self.decompress_internal(module, other, &mut source);
    }
}

impl<D: DataMut> GLWECiphertext<D> {
    pub(crate) fn decompress_internal<DataOther, B: Backend>(
        &mut self,
        module: &Module<B>,
        other: &GLWECiphertextCompressed<DataOther>,
        source: &mut Source,
    ) where
        DataOther: DataRef,
        Module<B>: VecZnxCopy + VecZnxFillUniform,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.rank(), other.rank());
            debug_assert_eq!(self.size(), other.size());
        }

        module.vec_znx_copy(&mut self.data, 0, &other.data, 0);
        (1..(other.rank() + 1).into()).for_each(|i| {
            module.vec_znx_fill_uniform(other.base2k.into(), &mut self.data, i, source);
        });

        self.base2k = other.base2k;
        self.k = other.k;
    }
}

pub trait GLWECiphertextCompressedToRef {
    fn to_ref(&self) -> GLWECiphertextCompressed<&[u8]>;
}

impl<D: DataRef> GLWECiphertextCompressedToRef for GLWECiphertextCompressed<D> {
    fn to_ref(&self) -> GLWECiphertextCompressed<&[u8]> {
        GLWECiphertextCompressed::builder()
            .k(self.k())
            .base2k(self.base2k())
            .rank(self.rank())
            .data(self.data.to_ref())
            .build()
            .unwrap()
    }
}

pub trait GLWECiphertextCompressedToMut {
    fn to_mut(&mut self) -> GLWECiphertextCompressed<&mut [u8]>;
}

impl<D: DataMut> GLWECiphertextCompressedToMut for GLWECiphertextCompressed<D> {
    fn to_mut(&mut self) -> GLWECiphertextCompressed<&mut [u8]> {
        GLWECiphertextCompressed::builder()
            .k(self.k())
            .base2k(self.base2k())
            .rank(self.rank())
            .data(self.data.to_mut())
            .build()
            .unwrap()
    }
}
