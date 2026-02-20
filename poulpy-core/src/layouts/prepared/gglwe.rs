use poulpy_hal::{
    api::{VmpPMatAlloc, VmpPMatBytesOf, VmpPrepare, VmpPrepareTmpBytes},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch, VmpPMat, VmpPMatToMut, VmpPMatToRef, ZnxInfos},
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWE, GGLWEInfos, GGLWEToRef, GLWEInfos, GetDegree, LWEInfos, Rank, TorusPrecision,
};

/// DFT-domain (prepared) variant of [`GGLWE`].
///
/// Stores the gadget GLWE matrix with polynomials in the frequency domain
/// of the backend's DFT/NTT transform, enabling O(N log N) polynomial
/// multiplication. The underlying data is held as a [`VmpPMat`], which
/// represents a prepared matrix suitable for vector-matrix products.
///
/// Tied to a specific backend via `B: Backend`.
#[derive(PartialEq, Eq)]
pub struct GGLWEPrepared<D: Data, B: Backend> {
    pub(crate) data: VmpPMat<D, B>,
    pub(crate) k: TorusPrecision,
    pub(crate) base2k: Base2K,
    pub(crate) dsize: Dsize,
}

/// Provides LWE-level parameter accessors (degree, base2k, precision, size).
impl<D: Data, B: Backend> LWEInfos for GGLWEPrepared<D, B> {
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

/// Provides the GLWE rank, derived from the output rank.
impl<D: Data, B: Backend> GLWEInfos for GGLWEPrepared<D, B> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

/// Provides GGLWE-specific parameter accessors (input/output rank, dsize, dnum).
impl<D: Data, B: Backend> GGLWEInfos for GGLWEPrepared<D, B> {
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

/// Factory trait for allocating and preparing [`GGLWEPrepared`] instances.
///
/// Requires the backend module to support VMP prepared-matrix allocation,
/// byte-size queries, and the prepare transform.
pub trait GGLWEPreparedFactory<BE: Backend>
where
    Self: GetDegree + VmpPMatAlloc<BE> + VmpPMatBytesOf + VmpPrepare<BE> + VmpPrepareTmpBytes,
{
    /// Allocates a new [`GGLWEPrepared`] with the given parameters.
    ///
    /// Panics if `dnum * dsize > ceil(k / base2k)`.
    fn alloc_gglwe_prepared(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> GGLWEPrepared<Vec<u8>, BE> {
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

        GGLWEPrepared {
            data: self.vmp_pmat_alloc(dnum.into(), rank_in.into(), (rank_out + 1).into(), size),
            k,
            base2k,
            dsize,
        }
    }

    /// Allocates a new [`GGLWEPrepared`] matching the parameters of `infos`.
    fn alloc_gglwe_prepared_from_infos<A>(&self, infos: &A) -> GGLWEPrepared<Vec<u8>, BE>
    where
        A: GGLWEInfos,
    {
        assert_eq!(self.ring_degree(), infos.n());
        self.alloc_gglwe_prepared(
            infos.base2k(),
            infos.k(),
            infos.rank_in(),
            infos.rank_out(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    /// Returns the byte size required to store a [`GGLWEPrepared`] with the given parameters.
    fn bytes_of_gglwe_prepared(
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

        self.bytes_of_vmp_pmat(dnum.into(), rank_in.into(), (rank_out + 1).into(), size)
    }

    /// Returns the byte size required to store a [`GGLWEPrepared`] matching `infos`.
    fn bytes_of_gglwe_prepared_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(self.ring_degree(), infos.n());
        self.bytes_of_gglwe_prepared(
            infos.base2k(),
            infos.k(),
            infos.rank_in(),
            infos.rank_out(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    /// Returns the scratch-space bytes needed by [`prepare_gglwe`](Self::prepare_gglwe).
    fn prepare_gglwe_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        self.vmp_prepare_tmp_bytes(
            infos.dnum().into(),
            infos.rank_in().into(),
            (infos.rank() + 1).into(),
            infos.size(),
        )
    }

    /// Transforms a standard [`GGLWE`] into the DFT domain, writing the result into `res`.
    ///
    /// Both `res` and `other` must share the same ring degree, base2k, precision, and dsize.
    fn prepare_gglwe<R, O>(&self, res: &mut R, other: &O, scratch: &mut Scratch<BE>)
    where
        R: GGLWEPreparedToMut<BE>,
        O: GGLWEToRef,
    {
        let mut res: GGLWEPrepared<&mut [u8], BE> = res.to_mut();
        let other: GGLWE<&[u8]> = other.to_ref();

        assert_eq!(res.n(), self.ring_degree());
        assert_eq!(other.n(), self.ring_degree());
        assert_eq!(res.base2k, other.base2k);
        assert_eq!(res.k, other.k);
        assert_eq!(res.dsize, other.dsize);

        self.vmp_prepare(&mut res.data, &other.data, scratch);
    }
}

impl<BE: Backend> GGLWEPreparedFactory<BE> for Module<BE> where
    Module<BE>: GetDegree + VmpPMatAlloc<BE> + VmpPMatBytesOf + VmpPrepare<BE> + VmpPrepareTmpBytes
{
}

/// Convenience associated functions for owned (`Vec<u8>`) allocation and byte-size queries.
impl<B: Backend> GGLWEPrepared<Vec<u8>, B> {
    /// Allocates a new [`GGLWEPrepared`] matching the parameters of `infos`.
    pub fn alloc_from_infos<A, M>(module: &M, infos: &A) -> Self
    where
        A: GGLWEInfos,
        M: GGLWEPreparedFactory<B>,
    {
        module.alloc_gglwe_prepared_from_infos(infos)
    }

    /// Allocates a new [`GGLWEPrepared`] with explicit parameters.
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
        M: GGLWEPreparedFactory<B>,
    {
        module.alloc_gglwe_prepared(base2k, k, rank_in, rank_out, dnum, dsize)
    }

    /// Returns the byte size for a [`GGLWEPrepared`] matching `infos`.
    pub fn bytes_of_from_infos<A, M>(module: &M, infos: &A) -> usize
    where
        A: GGLWEInfos,
        M: GGLWEPreparedFactory<B>,
    {
        module.bytes_of_gglwe_prepared_from_infos(infos)
    }

    /// Returns the byte size for a [`GGLWEPrepared`] with explicit parameters.
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
        M: GGLWEPreparedFactory<B>,
    {
        module.bytes_of_gglwe_prepared(base2k, k, rank_in, rank_out, dnum, dsize)
    }
}

impl<D: DataMut, B: Backend> GGLWEPrepared<D, B> {
    /// Transforms a standard [`GGLWE`] (`other`) into the DFT domain, writing into `self`.
    pub fn prepare<O, M>(&mut self, module: &M, other: &O, scratch: &mut Scratch<B>)
    where
        O: GGLWEToRef,
        M: GGLWEPreparedFactory<B>,
    {
        module.prepare_gglwe(self, other, scratch);
    }
}

impl<B: Backend> GGLWEPrepared<Vec<u8>, B> {
    /// Returns the scratch-space bytes needed by [`prepare`](Self::prepare).
    pub fn prepare_tmp_bytes<M>(&self, module: &M) -> usize
    where
        M: GGLWEPreparedFactory<B>,
    {
        module.prepare_gglwe_tmp_bytes(self)
    }
}

/// Conversion trait for obtaining a mutable borrowed [`GGLWEPrepared`].
pub trait GGLWEPreparedToMut<B: Backend> {
    /// Returns a [`GGLWEPrepared`] with a mutable borrow of the underlying data.
    fn to_mut(&mut self) -> GGLWEPrepared<&mut [u8], B>;
}

impl<D: DataMut, B: Backend> GGLWEPreparedToMut<B> for GGLWEPrepared<D, B> {
    fn to_mut(&mut self) -> GGLWEPrepared<&mut [u8], B> {
        GGLWEPrepared {
            k: self.k,
            base2k: self.base2k,
            dsize: self.dsize,
            data: self.data.to_mut(),
        }
    }
}

/// Conversion trait for obtaining an immutably borrowed [`GGLWEPrepared`].
pub trait GGLWEPreparedToRef<B: Backend> {
    /// Returns a [`GGLWEPrepared`] with an immutable borrow of the underlying data.
    fn to_ref(&self) -> GGLWEPrepared<&[u8], B>;
}

impl<D: DataRef, B: Backend> GGLWEPreparedToRef<B> for GGLWEPrepared<D, B> {
    fn to_ref(&self) -> GGLWEPrepared<&[u8], B> {
        GGLWEPrepared {
            k: self.k,
            base2k: self.base2k,
            dsize: self.dsize,
            data: self.data.to_ref(),
        }
    }
}
