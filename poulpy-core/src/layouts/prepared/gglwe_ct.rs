use poulpy_hal::{
    api::{VmpPMatAlloc, VmpPMatBytesOf, VmpPrepare, VmpPrepareTmpBytes},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch, VmpPMat, VmpPMatToMut, VmpPMatToRef, ZnxInfos},
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWE, GGLWEInfos, GGLWEToRef, GLWEInfos, GetDegree, LWEInfos, Rank, TorusPrecision,
};

#[derive(PartialEq, Eq)]
pub struct GGLWEPrepared<D: Data, B: Backend> {
    pub(crate) data: VmpPMat<D, B>,
    pub(crate) k: TorusPrecision,
    pub(crate) base2k: Base2K,
    pub(crate) dsize: Dsize,
}

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

impl<D: Data, B: Backend> GLWEInfos for GGLWEPrepared<D, B> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

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

pub trait GGLWEPreparedAlloc<B: Backend>
where
    Self: GetDegree + VmpPMatAlloc<B> + VmpPMatBytesOf,
{
    fn alloc_gglwe_prepared(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> GGLWEPrepared<Vec<u8>, B> {
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

    fn alloc_gglwe_prepared_from_infos<A>(&self, infos: &A) -> GGLWEPrepared<Vec<u8>, B>
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
}

impl<B: Backend> GGLWEPreparedAlloc<B> for Module<B> where Module<B>: GetDegree + VmpPMatAlloc<B> + VmpPMatBytesOf {}

impl<B: Backend> GGLWEPrepared<Vec<u8>, B> {
    pub fn alloc_from_infos<A, M>(module: &M, infos: &A) -> Self
    where
        A: GGLWEInfos,
        M: GGLWEPreparedAlloc<B>,
    {
        module.alloc_gglwe_prepared_from_infos(infos)
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
        M: GGLWEPreparedAlloc<B>,
    {
        module.alloc_gglwe_prepared(base2k, k, rank_in, rank_out, dnum, dsize)
    }

    pub fn bytes_of_from_infos<A, M>(module: &M, infos: &A) -> usize
    where
        A: GGLWEInfos,
        M: GGLWEPreparedAlloc<B>,
    {
        module.bytes_of_gglwe_prepared_from_infos(infos)
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
        M: GGLWEPreparedAlloc<B>,
    {
        module.bytes_of_gglwe_prepared(base2k, k, rank_in, rank_out, dnum, dsize)
    }
}

pub trait GGLWEPrepare<B: Backend>
where
    Self: GetDegree + VmpPrepareTmpBytes + VmpPrepare<B>,
{
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

    fn prepare_gglwe<R, O>(&self, res: &mut R, other: &O, scratch: &mut Scratch<B>)
    where
        R: GGLWEPreparedToMut<B>,
        O: GGLWEToRef,
    {
        let mut res: GGLWEPrepared<&mut [u8], B> = res.to_mut();
        let other: GGLWE<&[u8]> = other.to_ref();

        assert_eq!(res.n(), self.ring_degree());
        assert_eq!(other.n(), self.ring_degree());
        assert_eq!(res.base2k, other.base2k);
        assert_eq!(res.k, other.k);
        assert_eq!(res.dsize, other.dsize);

        self.vmp_prepare(&mut res.data, &other.data, scratch);
    }
}

impl<B: Backend> GGLWEPrepare<B> for Module<B> where Self: GetDegree + VmpPrepareTmpBytes + VmpPrepare<B> {}

impl<D: DataMut, B: Backend> GGLWEPrepared<D, B> {
    pub fn prepare<O, M>(&mut self, module: &M, other: &O, scratch: &mut Scratch<B>)
    where
        O: GGLWEToRef,
        M: GGLWEPrepare<B>,
    {
        module.prepare_gglwe(self, other, scratch);
    }
}

impl<B: Backend> GGLWEPrepared<Vec<u8>, B> {
    pub fn prepare_tmp_bytes<M>(&self, module: &M) -> usize
    where
        M: GGLWEPrepare<B>,
    {
        module.prepare_gglwe_tmp_bytes(self)
    }
}

pub trait GGLWEPreparedToMut<B: Backend> {
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

pub trait GGLWEPreparedToRef<B: Backend> {
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
