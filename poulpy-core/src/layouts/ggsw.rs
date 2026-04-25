use poulpy_hal::{
    layouts::{
        Backend, Data, FillUniform, HostDataMut, HostDataRef, MatZnx, MatZnxAtBackendMut, MatZnxAtBackendRef, MatZnxToBackendMut,
        MatZnxToBackendRef, MatZnxToMut, MatZnxToRef, Module, ReaderFrom, TransferFrom, WriterTo, ZnxInfos,
    },
    source::Source,
};
use std::fmt;

use crate::api::ModuleTransfer;
use crate::layouts::{Base2K, Degree, Dnum, Dsize, GLWE, GLWEInfos, LWEInfos, Rank, TorusPrecision};

/// Trait providing the parameter accessors for a GGSW (Gadget GSW) ciphertext.
///
/// A GGSW ciphertext is a matrix of [`GLWE`] ciphertexts with `rank_in = rank + 1`
/// input columns and `rank_out = rank + 1` output columns. It is used as the
/// left operand of external products.
/// Extends [`GLWEInfos`] with gadget decomposition parameters.
pub trait GGSWInfos
where
    Self: GLWEInfos,
{
    /// Returns the number of decomposition rows.
    fn dnum(&self) -> Dnum;
    /// Returns the decomposition digit size.
    fn dsize(&self) -> Dsize;
    /// Returns a plain-data [`GGSWLayout`] snapshot of the current parameters.
    fn ggsw_layout(&self) -> GGSWLayout {
        GGSWLayout {
            n: self.n(),
            base2k: self.base2k(),
            k: self.max_k(),
            rank: self.rank(),
            dnum: self.dnum(),
            dsize: self.dsize(),
        }
    }
}

/// Plain-data snapshot of the parameters that describe a [`GGSW`] ciphertext.
#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct GGSWLayout {
    /// Ring degree.
    pub n: Degree,
    /// Base-2-log of the limb width.
    pub base2k: Base2K,
    /// Torus precision.
    pub k: TorusPrecision,
    /// GLWE rank (number of mask polynomials per row).
    pub rank: Rank,
    /// Number of decomposition rows.
    pub dnum: Dnum,
    /// Decomposition digit size.
    pub dsize: Dsize,
}

impl LWEInfos for GGSWLayout {
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
impl GLWEInfos for GGSWLayout {
    fn rank(&self) -> Rank {
        self.rank
    }
}

impl GGSWInfos for GGSWLayout {
    fn dsize(&self) -> Dsize {
        self.dsize
    }

    fn dnum(&self) -> Dnum {
        self.dnum
    }
}

/// A GGSW (Gadget GSW) ciphertext.
///
/// Stored as a [`MatZnx`] matrix of [`GLWE`] ciphertexts with
/// `rank_in = rank + 1` input columns and `rank_out = rank + 1` output columns.
/// Used as the left operand of external products.
///
/// `D: Data` is the storage backend (e.g. `Vec<u8>`, `&[u8]`, `&mut [u8]`).
#[derive(PartialEq, Eq, Clone)]
pub struct GGSW<D: Data> {
    pub(crate) data: MatZnx<D>,
    pub(crate) base2k: Base2K,
    pub(crate) dsize: Dsize,
}

pub type GGSWBackendRef<'a, BE> = GGSW<<BE as Backend>::BufRef<'a>>;
pub type GGSWBackendMut<'a, BE> = GGSW<<BE as Backend>::BufMut<'a>>;

impl<D: Data> LWEInfos for GGSW<D> {
    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    fn base2k(&self) -> Base2K {
        self.base2k
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

impl<D: HostDataRef> fmt::Debug for GGSW<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.data)
    }
}

impl<D: HostDataRef> fmt::Display for GGSW<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "(GGSW: k: {} base2k: {} dsize: {}) {}",
            self.max_k().0,
            self.base2k().0,
            self.dsize().0,
            self.data
        )
    }
}

impl<D: HostDataMut> FillUniform for GGSW<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.data.fill_uniform(log_bound, source);
    }
}

impl<D: HostDataRef> GGSW<D> {
    pub fn at(&self, row: usize, col: usize) -> GLWE<&[u8]> {
        let data = self.data.at(row, col);
        GLWE {
            base2k: self.base2k,
            data,
        }
    }
}

pub trait GGSWAtBackendRef<BE: Backend> {
    fn at_backend(&self, row: usize, col: usize) -> GLWE<BE::BufRef<'_>>;
}

impl<BE: Backend> GGSWAtBackendRef<BE> for GGSW<BE::OwnedBuf> {
    fn at_backend(&self, row: usize, col: usize) -> GLWE<BE::BufRef<'_>> {
        let data = <MatZnx<BE::OwnedBuf> as MatZnxAtBackendRef<BE>>::at_backend(&self.data, row, col);
        GLWE {
            base2k: self.base2k,
            data,
        }
    }
}

pub fn ggsw_at_backend_ref_from_ref<'a, 'b, BE: Backend>(
    ggsw: &'a GGSW<BE::BufRef<'b>>,
    row: usize,
    col: usize,
) -> GLWE<BE::BufRef<'a>> {
    let data = poulpy_hal::layouts::mat_znx_at_backend_ref_from_ref::<BE>(&ggsw.data, row, col);
    GLWE {
        base2k: ggsw.base2k,
        data,
    }
}

pub fn ggsw_at_backend_ref_from_mut<'a, 'b, BE: Backend>(
    ggsw: &'a GGSW<BE::BufMut<'b>>,
    row: usize,
    col: usize,
) -> GLWE<BE::BufRef<'a>> {
    let data = poulpy_hal::layouts::mat_znx_at_backend_ref_from_mut::<BE>(&ggsw.data, row, col);
    GLWE {
        base2k: ggsw.base2k,
        data,
    }
}

impl<D: HostDataMut> GGSW<D> {
    pub fn at_mut(&mut self, row: usize, col: usize) -> GLWE<&mut [u8]> {
        let base2k = self.base2k;
        let data = self.data.at_mut(row, col);
        GLWE { base2k, data }
    }
}

pub trait GGSWAtBackendMut<BE: Backend> {
    fn at_backend_mut(&mut self, row: usize, col: usize) -> GLWE<BE::BufMut<'_>>;
}

impl<BE: Backend> GGSWAtBackendMut<BE> for GGSW<BE::OwnedBuf> {
    fn at_backend_mut(&mut self, row: usize, col: usize) -> GLWE<BE::BufMut<'_>> {
        let base2k = self.base2k;
        let data = <MatZnx<BE::OwnedBuf> as MatZnxAtBackendMut<BE>>::at_backend_mut(&mut self.data, row, col);
        GLWE { base2k, data }
    }
}

pub fn ggsw_at_backend_mut_from_mut<'a, 'b, BE: Backend>(
    ggsw: &'a mut GGSW<BE::BufMut<'b>>,
    row: usize,
    col: usize,
) -> GLWE<BE::BufMut<'a>> {
    let base2k = ggsw.base2k;
    let data = poulpy_hal::layouts::mat_znx_at_backend_mut_from_mut::<BE>(&mut ggsw.data, row, col);
    GLWE { base2k, data }
}

impl<D: HostDataRef> GGSW<D> {
    /// Copies this ciphertext's backing bytes into an owned buffer of
    /// backend `To`, routing via host bytes.
    pub fn to_backend<BE, To>(&self, dst: &Module<To>) -> GGSW<To::OwnedBuf>
    where
        BE: Backend<OwnedBuf = D>,
        To: Backend,
        To: TransferFrom<BE>,
    {
        dst.upload_ggsw(self)
    }
}

impl<D: Data> GGSW<D> {
    /// Zero-cost rename when both backends share the same `OwnedBuf`.
    pub fn reinterpret<To>(self) -> GGSW<To::OwnedBuf>
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
        GGSW {
            data: MatZnx::from_data(self.data.into_data(), n, rows, cols_in, cols_out, size),
            base2k: self.base2k,
            dsize: self.dsize,
        }
    }
}

impl GGSW<Vec<u8>> {
    pub fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GGSWInfos,
    {
        Self::alloc(
            infos.n(),
            infos.base2k(),
            infos.max_k(),
            infos.rank(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    pub fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> Self {
        let size: usize = k.0.div_ceil(base2k.0) as usize;
        assert!(
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
                n.into(),
                dnum.into(),
                (rank + 1).into(),
                (rank + 1).into(),
                k.0.div_ceil(base2k.0) as usize,
            ),
            base2k,
            dsize,
        }
    }

    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        Self::bytes_of(
            infos.n(),
            infos.base2k(),
            infos.max_k(),
            infos.rank(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    pub fn bytes_of(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> usize {
        let size: usize = k.0.div_ceil(base2k.0) as usize;
        assert!(
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
            n.into(),
            dnum.into(),
            (rank + 1).into(),
            (rank + 1).into(),
            k.0.div_ceil(base2k.0) as usize,
        )
    }
}

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

impl<D: HostDataMut> ReaderFrom for GGSW<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.base2k = Base2K(reader.read_u32::<LittleEndian>()?);
        self.dsize = Dsize(reader.read_u32::<LittleEndian>()?);
        self.data.read_from(reader)
    }
}

impl<D: HostDataRef> WriterTo for GGSW<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u32::<LittleEndian>(self.base2k.into())?;
        writer.write_u32::<LittleEndian>(self.dsize.into())?;
        self.data.write_to(writer)
    }
}

pub trait GGSWToMut {
    fn to_mut(&mut self) -> GGSW<&mut [u8]>;
}

pub trait GGSWToBackendMut<BE: Backend>: GGSWToBackendRef<BE> {
    fn to_backend_mut(&mut self) -> GGSWBackendMut<'_, BE>;
}

impl<BE: Backend> GGSWToBackendMut<BE> for GGSW<BE::OwnedBuf> {
    fn to_backend_mut(&mut self) -> GGSWBackendMut<'_, BE> {
        GGSW {
            dsize: self.dsize,
            base2k: self.base2k,
            data: <MatZnx<BE::OwnedBuf> as MatZnxToBackendMut<BE>>::to_backend_mut(&mut self.data),
        }
    }
}


pub fn ggsw_backend_mut_from_mut<'a, 'b, BE: Backend>(ggsw: &'a mut GGSW<BE::BufMut<'b>>) -> GGSWBackendMut<'a, BE> {
    GGSW {
        dsize: ggsw.dsize,
        base2k: ggsw.base2k,
        data: poulpy_hal::layouts::mat_znx_backend_mut_from_mut::<BE>(&mut ggsw.data),
    }
}

impl<D: HostDataMut> GGSWToMut for GGSW<D> {
    fn to_mut(&mut self) -> GGSW<&mut [u8]> {
        GGSW {
            dsize: self.dsize,
            base2k: self.base2k,
            data: self.data.to_mut(),
        }
    }
}

pub trait GGSWToRef {
    fn to_ref(&self) -> GGSW<&[u8]>;
}

pub trait GGSWToBackendRef<BE: Backend> {
    fn to_backend_ref(&self) -> GGSWBackendRef<'_, BE>;
}

impl<BE: Backend> GGSWToBackendRef<BE> for GGSW<BE::OwnedBuf> {
    fn to_backend_ref(&self) -> GGSWBackendRef<'_, BE> {
        GGSW {
            dsize: self.dsize,
            base2k: self.base2k,
            data: <MatZnx<BE::OwnedBuf> as MatZnxToBackendRef<BE>>::to_backend_ref(&self.data),
        }
    }
}

pub fn ggsw_backend_ref_from_ref<'a, 'b, BE: Backend>(ggsw: &'a GGSW<BE::BufRef<'b>>) -> GGSWBackendRef<'a, BE> {
    GGSW {
        dsize: ggsw.dsize,
        base2k: ggsw.base2k,
        data: poulpy_hal::layouts::mat_znx_backend_ref_from_ref::<BE>(&ggsw.data),
    }
}

pub fn ggsw_backend_ref_from_mut<'a, 'b, BE: Backend>(ggsw: &'a GGSW<BE::BufMut<'b>>) -> GGSWBackendRef<'a, BE> {
    GGSW {
        dsize: ggsw.dsize,
        base2k: ggsw.base2k,
        data: poulpy_hal::layouts::mat_znx_backend_ref_from_mut::<BE>(&ggsw.data),
    }
}

impl<D: HostDataRef> GGSWToRef for GGSW<D> {
    fn to_ref(&self) -> GGSW<&[u8]> {
        GGSW {
            dsize: self.dsize,
            base2k: self.base2k,
            data: self.data.to_ref(),
        }
    }
}
