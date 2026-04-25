use poulpy_hal::{
    layouts::{Backend, Data, FillUniform, HostDataMut, HostDataRef, ReaderFrom, WriterTo},
    source::Source,
};

use crate::{
    DeclaredK,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GGLWE, GGLWEBackendMut, GGLWEBackendRef, GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef,
        GGLWEToMut, GGLWEToRef, GLWEInfos, LWEInfos, Rank, TorusPrecision,
    },
};

use std::fmt;

/// Plain-data descriptor for a [`GLWETensorKey`] carrying only the
/// layout parameters (no backing buffer).
///
/// Implements [`LWEInfos`], [`GLWEInfos`] and [`GGLWEInfos`] so it can
/// be passed to any generic constructor that needs layout information.
/// The `rank_in` is derived from `rank` as `max(1, rank*(rank+1)/2)`.
#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct GLWETensorKeyLayout {
    pub n: Degree,
    pub base2k: Base2K,
    pub k: TorusPrecision,
    pub rank: Rank,
    pub dnum: Dnum,
    pub dsize: Dsize,
}

impl DeclaredK for GLWETensorKeyLayout {
    fn k(&self) -> TorusPrecision {
        self.k
    }
}

/// GLWE tensor key used for relinearisation after a tensor product.
///
/// Wraps a [`GGLWE`] whose `rank_in` equals the number of unique
/// pairs `max(1, rank*(rank+1)/2)` produced by the tensor product.
///
/// `D: Data` is the backing storage type (e.g. `Vec<u8>`, `&[u8]`,
/// `&mut [u8]`).
#[derive(PartialEq, Eq, Clone)]
pub struct GLWETensorKey<D: Data>(pub(crate) GGLWE<D>);

impl<D: Data> LWEInfos for GLWETensorKey<D> {
    fn n(&self) -> Degree {
        self.0.n()
    }

    fn base2k(&self) -> Base2K {
        self.0.base2k()
    }

    fn size(&self) -> usize {
        self.0.size()
    }
}

impl<D: Data> GLWEInfos for GLWETensorKey<D> {
    fn rank(&self) -> Rank {
        self.0.rank_out()
    }
}

impl<D: Data> GGLWEInfos for GLWETensorKey<D> {
    fn rank_in(&self) -> Rank {
        let rank_out: usize = self.rank_out().as_usize();
        let pairs: usize = (((rank_out + 1) * rank_out) >> 1).max(1);
        pairs.into()
    }

    fn rank_out(&self) -> Rank {
        self.0.rank_out()
    }

    fn dsize(&self) -> Dsize {
        self.0.dsize()
    }

    fn dnum(&self) -> Dnum {
        self.0.dnum()
    }
}

impl LWEInfos for GLWETensorKeyLayout {
    fn n(&self) -> Degree {
        self.n
    }

    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn size(&self) -> usize {
        self.k.as_usize().div_ceil(self.base2k.as_usize())
    }
}

impl GLWEInfos for GLWETensorKeyLayout {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl GGLWEInfos for GLWETensorKeyLayout {
    fn rank_in(&self) -> Rank {
        let rank_out: usize = self.rank_out().as_usize();
        let pairs: usize = (((rank_out + 1) * rank_out) >> 1).max(1);
        pairs.into()
    }

    fn dsize(&self) -> Dsize {
        self.dsize
    }

    fn rank_out(&self) -> Rank {
        self.rank
    }

    fn dnum(&self) -> Dnum {
        self.dnum
    }
}

impl<D: HostDataRef> fmt::Debug for GLWETensorKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: HostDataMut> FillUniform for GLWETensorKey<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.0.fill_uniform(log_bound, source)
    }
}

impl<D: HostDataRef> fmt::Display for GLWETensorKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "(GLWETensorKey)",)?;
        write!(f, "{}", self.0)?;
        Ok(())
    }
}

impl GLWETensorKey<Vec<u8>> {
    /// Allocates a new [`GLWETensorKey`] with the given parameters.
    pub fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GGLWEInfos,
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

    /// Allocates a new [`GLWETensorKey`] with the given parameters.
    pub fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> Self {
        let pairs: u32 = (((rank.0 + 1) * rank.0) >> 1).max(1);
        GLWETensorKey(GGLWE::alloc(n, base2k, k, Rank(pairs), rank, dnum, dsize))
    }

    /// Returns the byte count required for a [`GLWETensorKey`] with the given parameters.
    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GGLWEInfos,
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

    /// Returns the byte count required for a [`GLWETensorKey`] with the given parameters.
    pub fn bytes_of(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> usize {
        let pairs: u32 = (((rank.0 + 1) * rank.0) >> 1).max(1);
        GGLWE::bytes_of(n, base2k, k, Rank(pairs), rank, dnum, dsize)
    }
}

impl<D: HostDataMut> ReaderFrom for GLWETensorKey<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.0.read_from(reader)?;
        Ok(())
    }
}

impl<D: HostDataRef> WriterTo for GLWETensorKey<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        self.0.write_to(writer)?;
        Ok(())
    }
}

impl<D: HostDataRef> GGLWEToRef for GLWETensorKey<D>
where
    GGLWE<D>: GGLWEToRef,
{
    fn to_ref(&self) -> GGLWE<&[u8]> {
        self.0.to_ref()
    }
}

impl<D: HostDataMut> GGLWEToMut for GLWETensorKey<D>
where
    GGLWE<D>: GGLWEToMut,
{
    fn to_mut(&mut self) -> GGLWE<&mut [u8]> {
        self.0.to_mut()
    }
}

impl<BE: Backend> GGLWEToBackendRef<BE> for GLWETensorKey<BE::OwnedBuf> {
    fn to_backend_ref(&self) -> GGLWEBackendRef<'_, BE> {
        <GGLWE<BE::OwnedBuf> as GGLWEToBackendRef<BE>>::to_backend_ref(&self.0)
    }
}

impl<BE: Backend> GGLWEToBackendMut<BE> for GLWETensorKey<BE::OwnedBuf> {
    fn to_backend_mut(&mut self) -> GGLWEBackendMut<'_, BE> {
        <GGLWE<BE::OwnedBuf> as GGLWEToBackendMut<BE>>::to_backend_mut(&mut self.0)
    }
}

impl<BE: Backend> GGLWEToBackendRef<BE> for &mut GLWETensorKey<BE::OwnedBuf> {
    fn to_backend_ref(&self) -> GGLWEBackendRef<'_, BE> {
        <GLWETensorKey<BE::OwnedBuf> as GGLWEToBackendRef<BE>>::to_backend_ref(self)
    }
}

impl<BE: Backend> GGLWEToBackendMut<BE> for &mut GLWETensorKey<BE::OwnedBuf> {
    fn to_backend_mut(&mut self) -> GGLWEBackendMut<'_, BE> {
        <GLWETensorKey<BE::OwnedBuf> as GGLWEToBackendMut<BE>>::to_backend_mut(self)
    }
}
