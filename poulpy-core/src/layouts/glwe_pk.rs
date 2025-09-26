use poulpy_hal::layouts::{Data, DataMut, DataRef, ReaderFrom, VecZnx, WriterTo, ZnxInfos};

use crate::{
    dist::Distribution,
    layouts::{Base2K, BuildError, Degree, GLWEInfos, LWEInfos, Rank, TorusPrecision},
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

#[derive(PartialEq, Eq)]
pub struct GLWEPublicKey<D: Data> {
    pub(crate) data: VecZnx<D>,
    pub(crate) base2k: Base2K,
    pub(crate) k: TorusPrecision,
    pub(crate) dist: Distribution,
}

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct GLWEPublicKeyLayout {
    pub n: Degree,
    pub base2k: Base2K,
    pub k: TorusPrecision,
    pub rank: Rank,
}

impl<D: Data> LWEInfos for GLWEPublicKey<D> {
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

impl<D: Data> GLWEInfos for GLWEPublicKey<D> {
    fn rank(&self) -> Rank {
        Rank(self.data.cols() as u32 - 1)
    }
}

impl LWEInfos for GLWEPublicKeyLayout {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }

    fn n(&self) -> Degree {
        self.n
    }

    fn size(&self) -> usize {
        self.k.0.div_ceil(self.base2k.0) as usize
    }
}

impl GLWEInfos for GLWEPublicKeyLayout {
    fn rank(&self) -> Rank {
        self.rank
    }
}

pub struct GLWEPublicKeyBuilder<D: Data> {
    data: Option<VecZnx<D>>,
    base2k: Option<Base2K>,
    k: Option<TorusPrecision>,
}

impl<D: Data> GLWEPublicKey<D> {
    #[inline]
    pub fn builder() -> GLWEPublicKeyBuilder<D> {
        GLWEPublicKeyBuilder {
            data: None,
            base2k: None,
            k: None,
        }
    }
}

impl GLWEPublicKeyBuilder<Vec<u8>> {
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

impl<D: Data> GLWEPublicKeyBuilder<D> {
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

    pub fn build(self) -> Result<GLWEPublicKey<D>, BuildError> {
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

        Ok(GLWEPublicKey {
            data,
            base2k,
            k,
            dist: Distribution::NONE,
        })
    }
}

impl GLWEPublicKey<Vec<u8>> {
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
            dist: Distribution::NONE,
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

impl<D: DataMut> ReaderFrom for GLWEPublicKey<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.k = TorusPrecision(reader.read_u32::<LittleEndian>()?);
        self.base2k = Base2K(reader.read_u32::<LittleEndian>()?);
        match Distribution::read_from(reader) {
            Ok(dist) => self.dist = dist,
            Err(e) => return Err(e),
        }
        self.data.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GLWEPublicKey<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u32::<LittleEndian>(self.k.0)?;
        writer.write_u32::<LittleEndian>(self.base2k.0)?;
        match self.dist.write_to(writer) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        self.data.write_to(writer)
    }
}
