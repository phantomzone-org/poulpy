use poulpy_hal::layouts::{Backend, Data, DataMut, DataRef, Module, Scratch};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GGLWEPrepared, GGLWEPreparedFactory, GGLWEPreparedToMut, GGLWEPreparedToRef,
    GGLWEToGGSWKey, GGLWEToGGSWKeyToRef, GLWEInfos, LWEInfos, Rank, TorusPrecision,
};

/// DFT-domain (prepared) variant of [`GGLWEToGGSWKey`].
///
/// Stores a collection of [`GGLWEPrepared`] matrices (one per rank element)
/// with polynomials in the frequency domain of the backend's DFT/NTT transform,
/// enabling O(N log N) polynomial multiplication. Used for GGLWE-to-GGSW
/// key-switching operations.
///
/// Requires `rank_in == rank_out`. Tied to a specific backend via `BE: Backend`.
pub struct GGLWEToGGSWKeyPrepared<D: Data, BE: Backend> {
    pub(crate) keys: Vec<GGLWEPrepared<D, BE>>,
}

/// Provides LWE-level parameter accessors, delegating to the first key element.
impl<D: Data, BE: Backend> LWEInfos for GGLWEToGGSWKeyPrepared<D, BE> {
    fn n(&self) -> Degree {
        self.keys[0].n()
    }

    fn base2k(&self) -> Base2K {
        self.keys[0].base2k()
    }

    fn k(&self) -> TorusPrecision {
        self.keys[0].k()
    }

    fn size(&self) -> usize {
        self.keys[0].size()
    }
}

/// Provides the GLWE rank, derived from the output rank.
impl<D: Data, BE: Backend> GLWEInfos for GGLWEToGGSWKeyPrepared<D, BE> {
    fn rank(&self) -> Rank {
        self.keys[0].rank_out()
    }
}

/// Provides GGLWE-specific parameter accessors. Note that `rank_in == rank_out` for this type.
impl<D: Data, BE: Backend> GGLWEInfos for GGLWEToGGSWKeyPrepared<D, BE> {
    fn rank_in(&self) -> Rank {
        self.rank_out()
    }

    fn rank_out(&self) -> Rank {
        self.keys[0].rank_out()
    }

    fn dsize(&self) -> Dsize {
        self.keys[0].dsize()
    }

    fn dnum(&self) -> Dnum {
        self.keys[0].dnum()
    }
}

/// Factory trait for allocating and preparing [`GGLWEToGGSWKeyPrepared`] instances.
pub trait GGLWEToGGSWKeyPreparedFactory<BE: Backend> {
    /// Allocates a new [`GGLWEToGGSWKeyPrepared`] matching the parameters of `infos`.
    ///
    /// Panics if `rank_in != rank_out`.
    fn alloc_gglwe_to_ggsw_key_prepared_from_infos<A>(&self, infos: &A) -> GGLWEToGGSWKeyPrepared<Vec<u8>, BE>
    where
        A: GGLWEInfos;

    /// Allocates a new [`GGLWEToGGSWKeyPrepared`] with explicit parameters.
    ///
    /// Creates `rank` prepared GGLWE matrices, one per secret-key component.
    fn alloc_gglwe_to_ggsw_key_prepared(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> GGLWEToGGSWKeyPrepared<Vec<u8>, BE>;

    /// Returns the byte size required to store a [`GGLWEToGGSWKeyPrepared`] matching `infos`.
    fn bytes_of_gglwe_to_ggsw_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    /// Returns the byte size required to store a [`GGLWEToGGSWKeyPrepared`] with explicit parameters.
    fn bytes_of_gglwe_to_ggsw(&self, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> usize;

    /// Returns the scratch-space bytes needed by [`prepare_gglwe_to_ggsw_key`](Self::prepare_gglwe_to_ggsw_key).
    fn prepare_gglwe_to_ggsw_key_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    /// Transforms a standard [`GGLWEToGGSWKey`] into the DFT domain, writing into `res`.
    ///
    /// Iterates over each key element and prepares it individually.
    fn prepare_gglwe_to_ggsw_key<R, O>(&self, res: &mut R, other: &O, scratch: &mut Scratch<BE>)
    where
        R: GGLWEToGGSWKeyPreparedToMut<BE>,
        O: GGLWEToGGSWKeyToRef;
}

impl<BE: Backend> GGLWEToGGSWKeyPreparedFactory<BE> for Module<BE>
where
    Self: GGLWEPreparedFactory<BE>,
{
    fn alloc_gglwe_to_ggsw_key_prepared_from_infos<A>(&self, infos: &A) -> GGLWEToGGSWKeyPrepared<Vec<u8>, BE>
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWEToGGSWKeyPrepared"
        );
        self.alloc_gglwe_to_ggsw_key_prepared(infos.base2k(), infos.k(), infos.rank(), infos.dnum(), infos.dsize())
    }

    fn alloc_gglwe_to_ggsw_key_prepared(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> GGLWEToGGSWKeyPrepared<Vec<u8>, BE> {
        GGLWEToGGSWKeyPrepared {
            keys: (0..rank.as_usize())
                .map(|_| self.alloc_gglwe_prepared(base2k, k, rank, rank, dnum, dsize))
                .collect(),
        }
    }

    fn bytes_of_gglwe_to_ggsw_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWEToGGSWKeyPrepared"
        );
        self.bytes_of_gglwe_to_ggsw(infos.base2k(), infos.k(), infos.rank(), infos.dnum(), infos.dsize())
    }

    fn bytes_of_gglwe_to_ggsw(&self, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> usize {
        rank.as_usize() * self.bytes_of_gglwe_prepared(base2k, k, rank, rank, dnum, dsize)
    }

    fn prepare_gglwe_to_ggsw_key_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        self.prepare_gglwe_tmp_bytes(infos)
    }

    fn prepare_gglwe_to_ggsw_key<R, O>(&self, res: &mut R, other: &O, scratch: &mut Scratch<BE>)
    where
        R: GGLWEToGGSWKeyPreparedToMut<BE>,
        O: GGLWEToGGSWKeyToRef,
    {
        let res: &mut GGLWEToGGSWKeyPrepared<&mut [u8], BE> = &mut res.to_mut();
        let other: &GGLWEToGGSWKey<&[u8]> = &other.to_ref();

        assert_eq!(res.keys.len(), other.keys.len());

        for (a, b) in res.keys.iter_mut().zip(other.keys.iter()) {
            self.prepare_gglwe(a, b, scratch);
        }
    }
}

/// Convenience associated functions for owned (`Vec<u8>`) allocation and byte-size queries.
impl<BE: Backend> GGLWEToGGSWKeyPrepared<Vec<u8>, BE> {
    /// Allocates a new [`GGLWEToGGSWKeyPrepared`] matching the parameters of `infos`.
    pub fn alloc_from_infos<A, M>(module: &M, infos: &A) -> Self
    where
        A: GGLWEInfos,
        M: GGLWEToGGSWKeyPreparedFactory<BE>,
    {
        module.alloc_gglwe_to_ggsw_key_prepared_from_infos(infos)
    }

    /// Allocates a new [`GGLWEToGGSWKeyPrepared`] with explicit parameters.
    pub fn alloc<M>(module: &M, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> Self
    where
        M: GGLWEToGGSWKeyPreparedFactory<BE>,
    {
        module.alloc_gglwe_to_ggsw_key_prepared(base2k, k, rank, dnum, dsize)
    }

    /// Returns the byte size for a [`GGLWEToGGSWKeyPrepared`] matching `infos`.
    pub fn bytes_of_from_infos<A, M>(module: &M, infos: &A) -> usize
    where
        A: GGLWEInfos,
        M: GGLWEToGGSWKeyPreparedFactory<BE>,
    {
        module.bytes_of_gglwe_to_ggsw_from_infos(infos)
    }

    /// Returns the byte size for a [`GGLWEToGGSWKeyPrepared`] with explicit parameters.
    pub fn bytes_of<M>(module: &M, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> usize
    where
        M: GGLWEToGGSWKeyPreparedFactory<BE>,
    {
        module.bytes_of_gglwe_to_ggsw(base2k, k, rank, dnum, dsize)
    }
}

impl<D: DataMut, BE: Backend> GGLWEToGGSWKeyPrepared<D, BE> {
    /// Transforms a standard [`GGLWEToGGSWKey`] (`other`) into the DFT domain, writing into `self`.
    pub fn prepare<M, O>(&mut self, module: &M, other: &O, scratch: &mut Scratch<BE>)
    where
        M: GGLWEToGGSWKeyPreparedFactory<BE>,
        O: GGLWEToGGSWKeyToRef,
    {
        module.prepare_gglwe_to_ggsw_key(self, other, scratch);
    }
}

impl<D: DataMut, BE: Backend> GGLWEToGGSWKeyPrepared<D, BE> {
    /// Returns a mutable reference to the `i`-th prepared GGLWE key element.
    ///
    /// The `i`-th element corresponds to `GGLWEPrepared_s([s[i]*s[0], s[i]*s[1], ..., s[i]*s[rank]])`.
    pub fn at_mut(&mut self, i: usize) -> &mut GGLWEPrepared<D, BE> {
        assert!((i as u32) < self.rank());
        &mut self.keys[i]
    }
}

impl<D: DataRef, BE: Backend> GGLWEToGGSWKeyPrepared<D, BE> {
    /// Returns a reference to the `i`-th prepared GGLWE key element.
    ///
    /// The `i`-th element corresponds to `GGLWEPrepared_s([s[i]*s[0], s[i]*s[1], ..., s[i]*s[rank]])`.
    pub fn at(&self, i: usize) -> &GGLWEPrepared<D, BE> {
        assert!((i as u32) < self.rank());
        &self.keys[i]
    }
}

/// Conversion trait for obtaining an immutably borrowed [`GGLWEToGGSWKeyPrepared`].
pub trait GGLWEToGGSWKeyPreparedToRef<BE: Backend> {
    /// Returns a [`GGLWEToGGSWKeyPrepared`] with immutable borrows of the underlying data.
    fn to_ref(&self) -> GGLWEToGGSWKeyPrepared<&[u8], BE>;
}

impl<D: DataRef, BE: Backend> GGLWEToGGSWKeyPreparedToRef<BE> for GGLWEToGGSWKeyPrepared<D, BE>
where
    GGLWEPrepared<D, BE>: GGLWEPreparedToRef<BE>,
{
    fn to_ref(&self) -> GGLWEToGGSWKeyPrepared<&[u8], BE> {
        GGLWEToGGSWKeyPrepared {
            keys: self.keys.iter().map(|c| c.to_ref()).collect(),
        }
    }
}

/// Conversion trait for obtaining a mutably borrowed [`GGLWEToGGSWKeyPrepared`].
pub trait GGLWEToGGSWKeyPreparedToMut<BE: Backend> {
    /// Returns a [`GGLWEToGGSWKeyPrepared`] with mutable borrows of the underlying data.
    fn to_mut(&mut self) -> GGLWEToGGSWKeyPrepared<&mut [u8], BE>;
}

impl<D: DataMut, BE: Backend> GGLWEToGGSWKeyPreparedToMut<BE> for GGLWEToGGSWKeyPrepared<D, BE>
where
    GGLWEPrepared<D, BE>: GGLWEPreparedToMut<BE>,
{
    fn to_mut(&mut self) -> GGLWEToGGSWKeyPrepared<&mut [u8], BE> {
        GGLWEToGGSWKeyPrepared {
            keys: self.keys.iter_mut().map(|c| c.to_mut()).collect(),
        }
    }
}
