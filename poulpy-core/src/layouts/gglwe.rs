use poulpy_hal::{
    layouts::{
        Backend, Data, FillUniform, HostDataMut, HostDataRef, MatZnx, MatZnxAtBackendMut, MatZnxAtBackendRef, MatZnxToBackendMut,
        MatZnxToBackendRef, MatZnxToMut, MatZnxToRef, Module, ReaderFrom, TransferFrom, WriterTo, ZnxInfos,
    },
    source::Source,
};

use crate::api::ModuleTransfer;
use crate::layouts::{Base2K, Degree, Dnum, Dsize, GLWE, GLWEInfos, LWEInfos, Rank, TorusPrecision};
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
            k: self.max_k(),
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

    fn n(&self) -> Degree {
        self.n
    }

    fn size(&self) -> usize {
        self.k.as_usize().div_ceil(self.base2k.as_usize())
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
    pub(crate) base2k: Base2K,
    pub(crate) dsize: Dsize,
}

pub type GGLWEBackendRef<'a, BE> = GGLWE<<BE as Backend>::BufRef<'a>>;
pub type GGLWEBackendMut<'a, BE> = GGLWE<<BE as Backend>::BufMut<'a>>;

impl<D: Data> LWEInfos for GGLWE<D> {
    fn base2k(&self) -> Base2K {
        self.base2k
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

impl<D: HostDataRef> GGLWE<D> {
    pub fn data(&self) -> &MatZnx<D> {
        &self.data
    }
}

pub trait GGLWEAtBackendRef<BE: Backend> {
    fn at_backend(&self, row: usize, col: usize) -> GLWE<BE::BufRef<'_>>;
}

impl<BE: Backend> GGLWEAtBackendRef<BE> for GGLWE<BE::OwnedBuf> {
    fn at_backend(&self, row: usize, col: usize) -> GLWE<BE::BufRef<'_>> {
        let data = <MatZnx<BE::OwnedBuf> as MatZnxAtBackendRef<BE>>::at_backend(&self.data, row, col);
        GLWE {
            base2k: self.base2k,
            data,
        }
    }
}

pub fn gglwe_at_backend_ref_from_ref<'a, 'b, BE: Backend>(
    gglwe: &'a GGLWE<BE::BufRef<'b>>,
    row: usize,
    col: usize,
) -> GLWE<BE::BufRef<'a>> {
    let data = poulpy_hal::layouts::mat_znx_at_backend_ref_from_ref::<BE>(&gglwe.data, row, col);
    GLWE {
        base2k: gglwe.base2k,
        data,
    }
}

pub fn gglwe_at_backend_ref_from_mut<'a, 'b, BE: Backend>(
    gglwe: &'a GGLWE<BE::BufMut<'b>>,
    row: usize,
    col: usize,
) -> GLWE<BE::BufRef<'a>> {
    let data = poulpy_hal::layouts::mat_znx_at_backend_ref_from_mut::<BE>(&gglwe.data, row, col);
    GLWE {
        base2k: gglwe.base2k,
        data,
    }
}

impl<D: HostDataMut> GGLWE<D> {
    pub fn data_mut(&mut self) -> &mut MatZnx<D> {
        &mut self.data
    }
}

pub trait GGLWEAtBackendMut<BE: Backend> {
    fn at_backend_mut(&mut self, row: usize, col: usize) -> GLWE<BE::BufMut<'_>>;
}

impl<BE: Backend> GGLWEAtBackendMut<BE> for GGLWE<BE::OwnedBuf> {
    fn at_backend_mut(&mut self, row: usize, col: usize) -> GLWE<BE::BufMut<'_>> {
        let base2k = self.base2k;
        let data = <MatZnx<BE::OwnedBuf> as MatZnxAtBackendMut<BE>>::at_backend_mut(&mut self.data, row, col);
        GLWE { base2k, data }
    }
}

pub fn gglwe_at_backend_mut_from_mut<'a, 'b, BE: Backend>(
    gglwe: &'a mut GGLWE<BE::BufMut<'b>>,
    row: usize,
    col: usize,
) -> GLWE<BE::BufMut<'a>> {
    let base2k = gglwe.base2k;
    let data = poulpy_hal::layouts::mat_znx_at_backend_mut_from_mut::<BE>(&mut gglwe.data, row, col);
    GLWE { base2k, data }
}

impl<D: HostDataRef> fmt::Debug for GGLWE<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: HostDataMut> FillUniform for GGLWE<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.data.fill_uniform(log_bound, source);
    }
}

impl<D: HostDataRef> fmt::Display for GGLWE<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "(GGLWE: k={} base2k={} dsize={}) {}",
            self.max_k().0,
            self.base2k().0,
            self.dsize().0,
            self.data
        )
    }
}

impl<D: HostDataRef> GGLWE<D> {
    pub fn at(&self, row: usize, col: usize) -> GLWE<&[u8]> {
        let data = self.data.at(row, col);
        GLWE {
            base2k: self.base2k,
            data,
        }
    }
}

impl<D: HostDataMut> GGLWE<D> {
    pub fn at_mut(&mut self, row: usize, col: usize) -> GLWE<&mut [u8]> {
        let base2k = self.base2k;
        let data = self.data.at_mut(row, col);
        GLWE { base2k, data }
    }
}

impl<D: HostDataRef> GGLWE<D> {
    /// Copies this ciphertext's backing bytes into an owned buffer of
    /// backend `To`, routing via host bytes.
    pub fn to_backend<BE, To>(&self, dst: &Module<To>) -> GGLWE<To::OwnedBuf>
    where
        BE: Backend<OwnedBuf = D>,
        To: Backend,
        To: TransferFrom<BE>,
    {
        dst.upload_gglwe(self)
    }
}

impl<D: Data> GGLWE<D> {
    /// Zero-cost rename when both backends share the same `OwnedBuf`.
    pub fn reinterpret<To>(self) -> GGLWE<To::OwnedBuf>
    where
        To: Backend<OwnedBuf = D>,
    {
        let (n, rows, cols_in, cols_out, size) = (
            self.data.n(),
            self.data.rows(),
            self.data.cols_in(),
            self.data.cols_out(),
            self.data.size(),
        );
        GGLWE {
            data: MatZnx::from_data(self.data.into_data(), n, rows, cols_in, cols_out, size),
            base2k: self.base2k,
            dsize: self.dsize,
        }
    }
}

impl GGLWE<Vec<u8>> {
    pub fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GGLWEInfos,
    {
        Self::alloc(
            infos.n(),
            infos.base2k(),
            infos.max_k(),
            infos.rank_in(),
            infos.rank_out(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    pub fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision, rank_in: Rank, rank_out: Rank, dnum: Dnum, dsize: Dsize) -> Self {
        let size: usize = k.0.div_ceil(base2k.0) as usize;
        assert!(
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
                n.into(),
                dnum.into(),
                rank_in.into(),
                (rank_out + 1).into(),
                k.0.div_ceil(base2k.0) as usize,
            ),
            base2k,
            dsize,
        }
    }

    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        Self::bytes_of(
            infos.n(),
            infos.base2k(),
            infos.max_k(),
            infos.rank_in(),
            infos.rank_out(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    pub fn bytes_of(
        n: Degree,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> usize {
        let size: usize = k.0.div_ceil(base2k.0) as usize;
        assert!(
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
            n.into(),
            dnum.into(),
            rank_in.into(),
            (rank_out + 1).into(),
            k.0.div_ceil(base2k.0) as usize,
        )
    }
}

pub trait GGLWEToMut {
    fn to_mut(&mut self) -> GGLWE<&mut [u8]>;
}

pub trait GGLWEToBackendMut<BE: Backend>: GGLWEToBackendRef<BE> {
    fn to_backend_mut(&mut self) -> GGLWEBackendMut<'_, BE>;
}

impl<BE: Backend> GGLWEToBackendMut<BE> for GGLWE<BE::OwnedBuf> {
    fn to_backend_mut(&mut self) -> GGLWEBackendMut<'_, BE> {
        GGLWE {
            base2k: self.base2k(),
            dsize: self.dsize(),
            data: <MatZnx<BE::OwnedBuf> as MatZnxToBackendMut<BE>>::to_backend_mut(&mut self.data),
        }
    }
}

impl<D: HostDataMut> GGLWEToMut for GGLWE<D> {
    fn to_mut(&mut self) -> GGLWE<&mut [u8]> {
        GGLWE {
            base2k: self.base2k(),
            dsize: self.dsize(),
            data: self.data.to_mut(),
        }
    }
}

pub trait GGLWEToRef {
    fn to_ref(&self) -> GGLWE<&[u8]>;
}

pub trait GGLWEToBackendRef<BE: Backend> {
    fn to_backend_ref(&self) -> GGLWEBackendRef<'_, BE>;
}

impl<BE: Backend> GGLWEToBackendRef<BE> for GGLWE<BE::OwnedBuf> {
    fn to_backend_ref(&self) -> GGLWEBackendRef<'_, BE> {
        GGLWE {
            base2k: self.base2k(),
            dsize: self.dsize(),
            data: <MatZnx<BE::OwnedBuf> as MatZnxToBackendRef<BE>>::to_backend_ref(&self.data),
        }
    }
}

impl<D: HostDataRef> GGLWEToRef for GGLWE<D> {
    fn to_ref(&self) -> GGLWE<&[u8]> {
        GGLWE {
            base2k: self.base2k(),
            dsize: self.dsize(),
            data: self.data.to_ref(),
        }
    }
}

impl<D: HostDataMut> ReaderFrom for GGLWE<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.base2k = Base2K(reader.read_u32::<LittleEndian>()?);
        self.dsize = Dsize(reader.read_u32::<LittleEndian>()?);
        self.data.read_from(reader)
    }
}

impl<D: HostDataRef> WriterTo for GGLWE<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u32::<LittleEndian>(self.base2k.0)?;
        writer.write_u32::<LittleEndian>(self.dsize.0)?;
        self.data.write_to(writer)
    }
}
