use poulpy_hal::{
    api::{VmpPMatAlloc, VmpPMatBytesOf, VmpPrepare, VmpPrepareTmpBytes, VmpZero},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch, VmpPMat, VmpPMatToMut, VmpPMatToRef, ZnxInfos},
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGSW, GGSWInfos, GGSWToRef, GLWEInfos, GetDegree, LWEInfos, Rank, TorusPrecision,
};

/// DFT-domain (prepared) variant of [`GGSW`].
///
/// Stores the GGSW gadget matrix with polynomials in the frequency domain
/// of the backend's DFT/NTT transform, enabling O(N log N) polynomial
/// operations. Tied to a specific backend via `B: Backend`.
#[derive(PartialEq, Eq)]
pub struct GGSWPrepared<D: Data, B: Backend> {
    pub(crate) data: VmpPMat<D, B>,
    pub(crate) k: TorusPrecision,
    pub(crate) base2k: Base2K,
    pub(crate) dsize: Dsize,
}

impl<D: Data, B: Backend> LWEInfos for GGSWPrepared<D, B> {
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

impl<D: Data, B: Backend> GLWEInfos for GGSWPrepared<D, B> {
    fn rank(&self) -> Rank {
        Rank(self.data.cols_out() as u32 - 1)
    }
}

impl<D: Data, B: Backend> GGSWInfos for GGSWPrepared<D, B> {
    fn dsize(&self) -> Dsize {
        self.dsize
    }

    fn dnum(&self) -> Dnum {
        Dnum(self.data.rows() as u32)
    }
}

/// Trait for allocating and preparing DFT-domain GGSW ciphertexts.
pub trait GGSWPreparedFactory<B: Backend>
where
    Self: GetDegree + VmpPMatAlloc<B> + VmpPMatBytesOf + VmpPrepareTmpBytes + VmpPrepare<B> + VmpZero<B>,
{
    /// Allocates a new prepared GGSW with the given parameters.
    fn alloc_ggsw_prepared(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        dnum: Dnum,
        dsize: Dsize,
        rank: Rank,
    ) -> GGSWPrepared<Vec<u8>, B> {
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

        GGSWPrepared {
            data: self.vmp_pmat_alloc(
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

    fn alloc_ggsw_prepared_from_infos<A>(&self, infos: &A) -> GGSWPrepared<Vec<u8>, B>
    where
        A: GGSWInfos,
    {
        assert_eq!(self.ring_degree(), infos.n());
        self.alloc_ggsw_prepared(infos.base2k(), infos.k(), infos.dnum(), infos.dsize(), infos.rank())
    }

    fn bytes_of_ggsw_prepared(&self, base2k: Base2K, k: TorusPrecision, dnum: Dnum, dsize: Dsize, rank: Rank) -> usize {
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

        self.bytes_of_vmp_pmat(dnum.into(), (rank + 1).into(), (rank + 1).into(), size)
    }

    fn bytes_of_ggsw_prepared_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        assert_eq!(self.ring_degree(), infos.n());
        self.bytes_of_ggsw_prepared(infos.base2k(), infos.k(), infos.dnum(), infos.dsize(), infos.rank())
    }

    fn ggsw_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        assert_eq!(self.ring_degree(), infos.n());
        self.vmp_prepare_tmp_bytes(
            infos.dnum().into(),
            (infos.rank() + 1).into(),
            (infos.rank() + 1).into(),
            infos.size(),
        )
    }
    fn ggsw_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut Scratch<B>)
    where
        R: GGSWPreparedToMut<B>,
        O: GGSWToRef,
    {
        let mut res: GGSWPrepared<&mut [u8], B> = res.to_mut();
        let other: GGSW<&[u8]> = other.to_ref();
        assert_eq!(res.n(), self.ring_degree());
        assert_eq!(other.n(), self.ring_degree());
        assert_eq!(res.k, other.k);
        assert_eq!(res.base2k, other.base2k);
        assert_eq!(res.dsize, other.dsize);
        self.vmp_prepare(&mut res.data, &other.data, scratch);
    }
}

impl<B: Backend> GGSWPreparedFactory<B> for Module<B> where
    Self: GetDegree + VmpPMatAlloc<B> + VmpPMatBytesOf + VmpPrepareTmpBytes + VmpPrepare<B> + VmpZero<B>
{
}

impl<B: Backend> GGSWPrepared<Vec<u8>, B> {
    pub fn alloc_from_infos<A, M>(module: &M, infos: &A) -> Self
    where
        A: GGSWInfos,
        M: GGSWPreparedFactory<B>,
    {
        module.alloc_ggsw_prepared_from_infos(infos)
    }

    pub fn alloc<M>(module: &M, base2k: Base2K, k: TorusPrecision, dnum: Dnum, dsize: Dsize, rank: Rank) -> Self
    where
        M: GGSWPreparedFactory<B>,
    {
        module.alloc_ggsw_prepared(base2k, k, dnum, dsize, rank)
    }

    pub fn bytes_of_from_infos<A, M>(module: &M, infos: &A) -> usize
    where
        A: GGSWInfos,
        M: GGSWPreparedFactory<B>,
    {
        module.bytes_of_ggsw_prepared_from_infos(infos)
    }

    pub fn bytes_of<M>(module: &M, base2k: Base2K, k: TorusPrecision, dnum: Dnum, dsize: Dsize, rank: Rank) -> usize
    where
        M: GGSWPreparedFactory<B>,
    {
        module.bytes_of_ggsw_prepared(base2k, k, dnum, dsize, rank)
    }
}

impl<D: DataRef, B: Backend> GGSWPrepared<D, B> {
    pub fn data(&self) -> &VmpPMat<D, B> {
        &self.data
    }
}

impl<B: Backend> GGSWPrepared<Vec<u8>, B> {
    pub fn prepare_tmp_bytes<A, M>(&self, module: &M, infos: &A) -> usize
    where
        A: GGSWInfos,
        M: GGSWPreparedFactory<B>,
    {
        module.ggsw_prepare_tmp_bytes(infos)
    }
}

impl<D: DataMut, B: Backend> GGSWPrepared<D, B> {
    pub fn prepare<O, M>(&mut self, module: &M, other: &O, scratch: &mut Scratch<B>)
    where
        O: GGSWToRef,
        M: GGSWPreparedFactory<B>,
    {
        module.ggsw_prepare(self, other, scratch);
    }

    pub fn zero<M>(&mut self, module: &M)
    where
        M: GGSWPreparedFactory<B>,
    {
        module.vmp_zero(&mut self.data);
    }
}

pub trait GGSWPreparedToMut<B: Backend> {
    fn to_mut(&mut self) -> GGSWPrepared<&mut [u8], B>;
}

impl<D: DataMut, B: Backend> GGSWPreparedToMut<B> for GGSWPrepared<D, B> {
    fn to_mut(&mut self) -> GGSWPrepared<&mut [u8], B> {
        GGSWPrepared {
            base2k: self.base2k,
            k: self.k,
            dsize: self.dsize,
            data: self.data.to_mut(),
        }
    }
}

pub trait GGSWPreparedToRef<B: Backend> {
    fn to_ref(&self) -> GGSWPrepared<&[u8], B>;
}

impl<D: DataRef, B: Backend> GGSWPreparedToRef<B> for GGSWPrepared<D, B> {
    fn to_ref(&self) -> GGSWPrepared<&[u8], B> {
        GGSWPrepared {
            base2k: self.base2k,
            k: self.k,
            dsize: self.dsize,
            data: self.data.to_ref(),
        }
    }
}
