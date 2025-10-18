use poulpy_hal::{
    layouts::{
        Backend, Data, DataMut, DataRef, FillUniform, MatZnx, MatZnxToMut, MatZnxToRef, Module, ReaderFrom, WriterTo, ZnxInfos,
    },
    source::Source,
};

use crate::layouts::{Base2K, Degree, Dnum, Dsize, GLWE, GLWEInfos, GetDegree, LWEInfos, Rank, TorusPrecision};
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
    fn gglwe_layout(&self) -> GGLWELayout {
        GGLWELayout {
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

pub trait SetGGLWEInfos {
    fn set_dsize(&mut self, dsize: usize);
}

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct GGLWELayout {
    pub n: Degree,
    pub base2k: Base2K,
    pub k: TorusPrecision,
    pub rank_in: Rank,
    pub rank_out: Rank,
    pub dnum: Dnum,
    pub dsize: Dsize,
}

impl LWEInfos for GGLWELayout {
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

impl GLWEInfos for GGLWELayout {
    fn rank(&self) -> Rank {
        self.rank_out
    }
}

impl GGLWEInfos for GGLWELayout {
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
pub struct GGLWE<D: Data> {
    pub(crate) data: MatZnx<D>,
    pub(crate) k: TorusPrecision,
    pub(crate) base2k: Base2K,
    pub(crate) dsize: Dsize,
}

impl<D: Data> LWEInfos for GGLWE<D> {
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

impl<D: Data> GLWEInfos for GGLWE<D> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data> GGLWEInfos for GGLWE<D> {
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

impl<D: DataRef> GGLWE<D> {
    pub fn data(&self) -> &MatZnx<D> {
        &self.data
    }
}

impl<D: DataMut> GGLWE<D> {
    pub fn data_mut(&mut self) -> &mut MatZnx<D> {
        &mut self.data
    }
}

impl<D: DataRef> fmt::Debug for GGLWE<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: DataMut> FillUniform for GGLWE<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.data.fill_uniform(log_bound, source);
    }
}

impl<D: DataRef> fmt::Display for GGLWE<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "(GGLWE: k={} base2k={} dsize={}) {}",
            self.k().0,
            self.base2k().0,
            self.dsize().0,
            self.data
        )
    }
}

impl<D: DataRef> GGLWE<D> {
    pub fn at(&self, row: usize, col: usize) -> GLWE<&[u8]> {
        GLWE {
            k: self.k,
            base2k: self.base2k,
            data: self.data.at(row, col),
        }
    }
}

impl<D: DataMut> GGLWE<D> {
    pub fn at_mut(&mut self, row: usize, col: usize) -> GLWE<&mut [u8]> {
        GLWE {
            k: self.k,
            base2k: self.base2k,
            data: self.data.at_mut(row, col),
        }
    }
}

pub trait GGLWEAlloc
where
    Self: GetDegree,
{
    fn alloc_gglwe(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> GGLWE<Vec<u8>> {
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

        GGLWE {
            data: MatZnx::alloc(
                self.ring_degree().into(),
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

    fn alloc_glwe_from_infos<A>(&self, infos: &A) -> GGLWE<Vec<u8>>
    where
        A: GGLWEInfos,
    {
        self.alloc_gglwe(
            infos.base2k(),
            infos.k(),
            infos.rank_in(),
            infos.rank_out(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    fn bytes_of_gglwe(
        &self,
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

        MatZnx::bytes_of(
            self.ring_degree().into(),
            dnum.into(),
            rank_in.into(),
            (rank_out + 1).into(),
            k.0.div_ceil(base2k.0) as usize,
        )
    }

    fn bytes_of_gglwe_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        self.bytes_of_gglwe(
            infos.base2k(),
            infos.k(),
            infos.rank_in(),
            infos.rank_out(),
            infos.dnum(),
            infos.dsize(),
        )
    }
}

impl<B: Backend> GGLWEAlloc for Module<B> where Self: GetDegree {}

impl GGLWE<Vec<u8>> {
    pub fn alloc_from_infos<A, M>(module: &M, infos: &A) -> Self
    where
        A: GGLWEInfos,
        M: GGLWEAlloc,
    {
        module.alloc_glwe_from_infos(infos)
    }

    pub fn alloc<M>(
        module: &M,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> Self
    where
        M: GGLWEAlloc,
    {
        module.alloc_gglwe(base2k, k, rank_in, rank_out, dnum, dsize)
    }

    pub fn bytes_of_from_infos<A, M>(module: &M, infos: &A) -> usize
    where
        A: GGLWEInfos,
        M: GGLWEAlloc,
    {
        module.bytes_of_gglwe_from_infos(infos)
    }

    pub fn bytes_of<M>(
        module: &M,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> usize
    where
        M: GGLWEAlloc,
    {
        module.bytes_of_gglwe(base2k, k, rank_in, rank_out, dnum, dsize)
    }
}

pub trait GGLWEToMut {
    fn to_mut(&mut self) -> GGLWE<&mut [u8]>;
}

impl<D: DataMut> GGLWEToMut for GGLWE<D> {
    fn to_mut(&mut self) -> GGLWE<&mut [u8]> {
        GGLWE {
            k: self.k(),
            base2k: self.base2k(),
            dsize: self.dsize(),
            data: self.data.to_mut(),
        }
    }
}

pub trait GGLWEToRef {
    fn to_ref(&self) -> GGLWE<&[u8]>;
}

impl<D: DataRef> GGLWEToRef for GGLWE<D> {
    fn to_ref(&self) -> GGLWE<&[u8]> {
        GGLWE {
            k: self.k(),
            base2k: self.base2k(),
            dsize: self.dsize(),
            data: self.data.to_ref(),
        }
    }
}

impl<D: DataMut> ReaderFrom for GGLWE<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.k = TorusPrecision(reader.read_u32::<LittleEndian>()?);
        self.base2k = Base2K(reader.read_u32::<LittleEndian>()?);
        self.dsize = Dsize(reader.read_u32::<LittleEndian>()?);
        self.data.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GGLWE<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u32::<LittleEndian>(self.k.0)?;
        writer.write_u32::<LittleEndian>(self.base2k.0)?;
        writer.write_u32::<LittleEndian>(self.dsize.0)?;
        self.data.write_to(writer)
    }
}
