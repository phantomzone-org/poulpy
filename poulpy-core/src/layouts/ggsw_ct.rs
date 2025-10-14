use poulpy_hal::{
    layouts::{
        Backend, Data, DataMut, DataRef, FillUniform, MatZnx, MatZnxToMut, MatZnxToRef, Module, ReaderFrom, WriterTo, ZnxInfos,
    },
    source::Source,
};
use std::fmt;

use crate::layouts::{Base2K, Degree, Dnum, Dsize, GLWE, GLWEInfos, GetDegree, LWEInfos, Rank, TorusPrecision};

pub trait GGSWInfos
where
    Self: GLWEInfos,
{
    fn dnum(&self) -> Dnum;
    fn dsize(&self) -> Dsize;
    fn ggsw_layout(&self) -> GGSWCiphertextLayout {
        GGSWCiphertextLayout {
            n: self.n(),
            base2k: self.base2k(),
            k: self.k(),
            rank: self.rank(),
            dnum: self.dnum(),
            dsize: self.dsize(),
        }
    }
}

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct GGSWCiphertextLayout {
    pub n: Degree,
    pub base2k: Base2K,
    pub k: TorusPrecision,
    pub rank: Rank,
    pub dnum: Dnum,
    pub dsize: Dsize,
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
    fn dsize(&self) -> Dsize {
        self.dsize
    }

    fn dnum(&self) -> Dnum {
        self.dnum
    }
}

#[derive(PartialEq, Eq, Clone)]
pub struct GGSW<D: Data> {
    pub(crate) data: MatZnx<D>,
    pub(crate) k: TorusPrecision,
    pub(crate) base2k: Base2K,
    pub(crate) dsize: Dsize,
}

impl<D: Data> LWEInfos for GGSW<D> {
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

impl<D: Data> GLWEInfos for GGSW<D> {
    fn rank(&self) -> Rank {
        Rank(self.data.cols_out() as u32 - 1)
    }
}

impl<D: Data> GGSWInfos for GGSW<D> {
    fn dsize(&self) -> Dsize {
        self.dsize
    }

    fn dnum(&self) -> Dnum {
        Dnum(self.data.rows() as u32)
    }
}

impl<D: DataRef> fmt::Debug for GGSW<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.data)
    }
}

impl<D: DataRef> fmt::Display for GGSW<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "(GGSWCiphertext: k: {} base2k: {} dsize: {}) {}",
            self.k().0,
            self.base2k().0,
            self.dsize().0,
            self.data
        )
    }
}

impl<D: DataMut> FillUniform for GGSW<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.data.fill_uniform(log_bound, source);
    }
}

impl<D: DataRef> GGSW<D> {
    pub fn at(&self, row: usize, col: usize) -> GLWE<&[u8]> {
        GLWE {
            k: self.k,
            base2k: self.base2k,
            data: self.data.at(row, col),
        }
    }
}

impl<D: DataMut> GGSW<D> {
    pub fn at_mut(&mut self, row: usize, col: usize) -> GLWE<&mut [u8]> {
        GLWE {
            k: self.k,
            base2k: self.base2k,
            data: self.data.at_mut(row, col),
        }
    }
}

impl<B: Backend> GGSWAlloc for Module<B> where Self: GetDegree {}

pub trait GGSWAlloc
where
    Self: GetDegree,
{
    fn alloc_ggsw(&self, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> GGSW<Vec<u8>> {
        let size: usize = k.0.div_ceil(base2k.0) as usize;
        debug_assert!(
            size as u32 > dsize.0,
            "invalid ggsw: ceil(k/base2k): {size} <= dsize: {}",
            dsize.0
        );

        assert!(
            dnum.0 * dsize.0 <= size as u32,
            "invalid ggsw: dnum: {} * dsize:{} > ceil(k/base2k): {size}",
            dnum.0,
            dsize.0,
        );

        GGSW {
            data: MatZnx::alloc(
                self.n().into(),
                dnum.into(),
                (rank + 1).into(),
                (rank + 1).into(),
                k.0.div_ceil(base2k.0) as usize,
            ),
            k,
            base2k,
            dsize,
        }
    }

    fn alloc_ggsw_from_infos<A>(&self, infos: &A) -> GGSW<Vec<u8>>
    where
        A: GGSWInfos,
    {
        self.alloc_ggsw(
            infos.base2k(),
            infos.k(),
            infos.rank(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    fn bytes_of_ggsw(&self, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> usize {
        let size: usize = k.0.div_ceil(base2k.0) as usize;
        debug_assert!(
            size as u32 > dsize.0,
            "invalid ggsw: ceil(k/base2k): {size} <= dsize: {}",
            dsize.0
        );

        assert!(
            dnum.0 * dsize.0 <= size as u32,
            "invalid ggsw: dnum: {} * dsize:{} > ceil(k/base2k): {size}",
            dnum.0,
            dsize.0,
        );

        MatZnx::bytes_of(
            self.n().into(),
            dnum.into(),
            (rank + 1).into(),
            (rank + 1).into(),
            k.0.div_ceil(base2k.0) as usize,
        )
    }

    fn bytes_of_ggsw_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        self.bytes_of_ggsw(
            infos.base2k(),
            infos.k(),
            infos.rank(),
            infos.dnum(),
            infos.dsize(),
        )
    }
}

impl GGSW<Vec<u8>> {
    pub fn alloc_from_infos<A, B: Backend>(module: Module<B>, infos: &A) -> Self
    where
        A: GGSWInfos,
        Module<B>: GGSWAlloc,
    {
        module.alloc_ggsw_from_infos(infos)
    }

    pub fn alloc<B: Backend>(module: Module<B>, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> Self
    where
        Module<B>: GGSWAlloc,
    {
        module.alloc_ggsw(base2k, k, rank, dnum, dsize)
    }

    pub fn bytes_of_from_infos<A, B: Backend>(module: Module<B>, infos: &A) -> usize
    where
        A: GGSWInfos,
        Module<B>: GGSWAlloc,
    {
        module.bytes_of_ggsw_from_infos(infos)
    }

    pub fn bytes_of<B: Backend>(
        module: Module<B>,
        base2k: Base2K,
        k: TorusPrecision,
        rank: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> usize
    where
        Module<B>: GGSWAlloc,
    {
        module.bytes_of_ggsw(base2k, k, rank, dnum, dsize)
    }
}

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

impl<D: DataMut> ReaderFrom for GGSW<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.k = TorusPrecision(reader.read_u32::<LittleEndian>()?);
        self.base2k = Base2K(reader.read_u32::<LittleEndian>()?);
        self.dsize = Dsize(reader.read_u32::<LittleEndian>()?);
        self.data.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GGSW<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u32::<LittleEndian>(self.k.into())?;
        writer.write_u32::<LittleEndian>(self.base2k.into())?;
        writer.write_u32::<LittleEndian>(self.dsize.into())?;
        self.data.write_to(writer)
    }
}

pub trait GGSWToMut {
    fn to_mut(&mut self) -> GGSW<&mut [u8]>;
}

impl<D: DataMut> GGSWToMut for GGSW<D> {
    fn to_mut(&mut self) -> GGSW<&mut [u8]> {
        GGSW {
            dsize: self.dsize,
            k: self.k,
            base2k: self.base2k,
            data: self.data.to_mut(),
        }
    }
}

pub trait GGSWToRef {
    fn to_ref(&self) -> GGSW<&[u8]>;
}

impl<D: DataRef> GGSWToRef for GGSW<D> {
    fn to_ref(&self) -> GGSW<&[u8]> {
        GGSW {
            dsize: self.dsize,
            k: self.k,
            base2k: self.base2k,
            data: self.data.to_ref(),
        }
    }
}
