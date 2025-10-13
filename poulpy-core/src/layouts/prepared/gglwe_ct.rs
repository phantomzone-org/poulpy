use poulpy_hal::{
    api::{VmpPMatAlloc, VmpPMatAllocBytes, VmpPrepare, VmpPrepareTmpBytes},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch, VmpPMat, VmpPMatToMut, VmpPMatToRef, ZnxInfos},
    oep::VmpPMatAllocBytesImpl,
};

use crate::layouts::{
    Base2K, BuildError, Degree, Dnum, Dsize, GGLWE, GGLWEInfos, GGLWEToRef, GLWEInfos, LWEInfos, Rank, TorusPrecision,
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

pub struct GGLWEPreparedBuilder<D: Data, B: Backend> {
    data: Option<VmpPMat<D, B>>,
    base2k: Option<Base2K>,
    k: Option<TorusPrecision>,
    dsize: Option<Dsize>,
}

impl<D: Data, B: Backend> GGLWEPrepared<D, B> {
    #[inline]
    pub fn builder() -> GGLWEPreparedBuilder<D, B> {
        GGLWEPreparedBuilder {
            data: None,
            base2k: None,
            k: None,
            dsize: None,
        }
    }
}

impl<B: Backend> GGLWEPreparedBuilder<Vec<u8>, B> {
    #[inline]
    pub fn layout<A>(mut self, infos: &A) -> Self
    where
        A: GGLWEInfos,
        B: VmpPMatAllocBytesImpl<B>,
    {
        self.data = Some(VmpPMat::alloc(
            infos.n().into(),
            infos.dnum().into(),
            infos.rank_in().into(),
            (infos.rank_out() + 1).into(),
            infos.size(),
        ));
        self.base2k = Some(infos.base2k());
        self.k = Some(infos.k());
        self.dsize = Some(infos.dsize());
        self
    }
}

impl<D: Data, B: Backend> GGLWEPreparedBuilder<D, B> {
    #[inline]
    pub fn data(mut self, data: VmpPMat<D, B>) -> Self {
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

    #[inline]
    pub fn dsize(mut self, dsize: Dsize) -> Self {
        self.dsize = Some(dsize);
        self
    }

    pub fn build(self) -> Result<GGLWEPrepared<D, B>, BuildError> {
        let data: VmpPMat<D, B> = self.data.ok_or(BuildError::MissingData)?;
        let base2k: Base2K = self.base2k.ok_or(BuildError::MissingBase2K)?;
        let k: TorusPrecision = self.k.ok_or(BuildError::MissingK)?;
        let dsize: Dsize = self.dsize.ok_or(BuildError::MissingDigits)?;

        if base2k == 0_u32 {
            return Err(BuildError::ZeroBase2K);
        }

        if dsize == 0_u32 {
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

        Ok(GGLWEPrepared {
            data,
            base2k,
            k,
            dsize,
        })
    }
}

impl<B: Backend> GGLWEPrepared<Vec<u8>, B> {
    pub fn alloc<A>(module: &Module<B>, infos: &A) -> Self
    where
        A: GGLWEInfos,
        Module<B>: VmpPMatAlloc<B>,
    {
        debug_assert_eq!(module.n(), infos.n().0 as usize, "module.n() != infos.n()");
        Self::alloc_with(
            module,
            infos.base2k(),
            infos.k(),
            infos.rank_in(),
            infos.rank_out(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    pub fn alloc_with(
        module: &Module<B>,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> Self
    where
        Module<B>: VmpPMatAlloc<B>,
    {
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

        Self {
            data: module.vmp_pmat_alloc(dnum.into(), rank_in.into(), (rank_out + 1).into(), size),
            k,
            base2k,
            dsize,
        }
    }

    pub fn alloc_bytes<A>(module: &Module<B>, infos: &A) -> usize
    where
        A: GGLWEInfos,
        Module<B>: VmpPMatAllocBytes,
    {
        debug_assert_eq!(module.n(), infos.n().0 as usize, "module.n() != infos.n()");
        Self::alloc_bytes_with(
            module,
            infos.base2k(),
            infos.k(),
            infos.rank_in(),
            infos.rank_out(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    pub fn alloc_bytes_with(
        module: &Module<B>,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> usize
    where
        Module<B>: VmpPMatAllocBytes,
    {
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

        module.vmp_pmat_alloc_bytes(dnum.into(), rank_in.into(), (rank_out + 1).into(), size)
    }
}

pub trait GGLWEPrepareTmpBytes {
    fn gglwe_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;
}

impl<B: Backend> GGLWEPrepareTmpBytes for Module<B>
where
    Module<B>: VmpPrepareTmpBytes,
{
    fn gglwe_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
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
}

impl<B: Backend> GGLWEPrepared<Vec<u8>, B> {
    pub fn prepare_tmp_bytes(&self, module: &Module<B>)
    where
        Module<B>: GGLWEPrepareTmpBytes,
    {
        module.gglwe_prepare_tmp_bytes(self)
    }
}

pub trait GGLWEPrepare<B: Backend> {
    fn gglwe_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut Scratch<B>)
    where
        R: GGLWEPreparedToMut<B>,
        O: GGLWEToRef;
}

impl<B: Backend> GGLWEPrepare<B> for Module<B>
where
    Module<B>: VmpPrepare<B>,
{
    fn gglwe_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut Scratch<B>)
    where
        R: GGLWEPreparedToMut<B>,
        O: GGLWEToRef,
    {
        let mut res: GGLWEPrepared<&mut [u8], B> = self.to_mut();
        let other: GGLWE<&[u8]> = other.to_ref();

        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(other.n(), self.n() as u32);
        assert_eq!(res.base2k, other.base2k);
        assert_eq!(res.k, other.k);
        assert_eq!(res.dsize, other.dsize);

        self.vmp_prepare(&mut res.data, &other.data, scratch);
    }
}

impl<D: DataMut, B: Backend> GGLWEPrepared<D, B>
where
    Module<B>: GGLWEPrepare<B>,
{
    fn prepare<O>(&mut self, module: &Module<B>, other: &O, scratch: &mut Scratch<B>)
    where
        O: GGLWEToRef,
    {
        module.gglwe_prepare(self, other, scratch);
    }
}

pub trait GGLWEPrepareAlloc<B: Backend> {
    fn gglwe_prepare_alloc<O>(&self, other: &O, scratch: &mut Scratch<B>) -> GGLWEPrepared<Vec<u8>, B>;
}

impl<B: Backend> GGLWEPrepareAlloc<B> for Module<B>
where
    Module<B>: VmpPMatAlloc<B> + VmpPrepare<B>,
{
    fn gglwe_prepare_alloc<O>(&self, other: &O, scratch: &mut Scratch<B>) -> GGLWEPrepared<Vec<u8>, B> {
        let mut ct_prepared: GGLWEPrepared<Vec<u8>, B> = GGLWEPrepared::alloc(self, &other.to_ref());
        ct_prepared.prepare(self, &other.to_ref(), scratch);
        ct_prepared
    }
}

impl<D: DataRef> GGLWE<D> {
    fn prepare_alloc<B: Backend>(&self, module: &Module<B>, scratch: &Scratch<B>) -> GGLWEPrepared<Vec<u8>, B>
    where
        Module<B>: GGLWEPrepareAlloc<B>,
    {
        module.gglwe_prepare_alloc(self, scratch)
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
