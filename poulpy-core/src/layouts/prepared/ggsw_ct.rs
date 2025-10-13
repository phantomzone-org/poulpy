use poulpy_hal::{
    api::{VmpPMatAlloc, VmpPMatAllocBytes, VmpPrepare, VmpPrepareTmpBytes},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch, VmpPMat, VmpPMatToMut, VmpPMatToRef, ZnxInfos},
    oep::VmpPMatAllocBytesImpl,
};

use crate::layouts::{
    Base2K, BuildError, Degree, Dnum, Dsize, GGSW, GGSWInfos, GGSWToMut, GGSWToRef, GLWEInfos, LWEInfos, Rank, TorusPrecision,
};

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

pub struct GGSWCiphertextPreparedBuilder<D: Data, B: Backend> {
    data: Option<VmpPMat<D, B>>,
    base2k: Option<Base2K>,
    k: Option<TorusPrecision>,
    dsize: Option<Dsize>,
}

impl<D: Data, B: Backend> GGSWPrepared<D, B> {
    #[inline]
    pub fn builder() -> GGSWCiphertextPreparedBuilder<D, B> {
        GGSWCiphertextPreparedBuilder {
            data: None,
            base2k: None,
            k: None,
            dsize: None,
        }
    }
}

impl<B: Backend> GGSWCiphertextPreparedBuilder<Vec<u8>, B> {
    #[inline]
    pub fn layout<A>(mut self, infos: &A) -> Self
    where
        A: GGSWInfos,
        B: VmpPMatAllocBytesImpl<B>,
    {
        debug_assert!(
            infos.size() as u32 > infos.dsize().0,
            "invalid ggsw: ceil(k/base2k): {} <= dsize: {}",
            infos.size(),
            infos.dsize()
        );

        assert!(
            infos.dnum().0 * infos.dsize().0 <= infos.size() as u32,
            "invalid ggsw: dnum: {} * dsize:{} > ceil(k/base2k): {}",
            infos.dnum(),
            infos.dsize(),
            infos.size(),
        );

        self.data = Some(VmpPMat::alloc(
            infos.n().into(),
            infos.dnum().into(),
            (infos.rank() + 1).into(),
            (infos.rank() + 1).into(),
            infos.size(),
        ));
        self.base2k = Some(infos.base2k());
        self.k = Some(infos.k());
        self.dsize = Some(infos.dsize());
        self
    }
}

impl<D: Data, B: Backend> GGSWCiphertextPreparedBuilder<D, B> {
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

    pub fn build(self) -> Result<GGSWPrepared<D, B>, BuildError> {
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

        Ok(GGSWPrepared {
            data,
            base2k,
            k,
            dsize,
        })
    }
}

impl<B: Backend> GGSWPrepared<Vec<u8>, B> {
    pub fn alloc<A>(module: &Module<B>, infos: &A) -> Self
    where
        A: GGSWInfos,
        Module<B>: VmpPMatAlloc<B>,
    {
        Self::alloc_with(
            module,
            infos.base2k(),
            infos.k(),
            infos.dnum(),
            infos.dsize(),
            infos.rank(),
        )
    }

    pub fn alloc_with(module: &Module<B>, base2k: Base2K, k: TorusPrecision, dnum: Dnum, dsize: Dsize, rank: Rank) -> Self
    where
        Module<B>: VmpPMatAlloc<B>,
    {
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

        Self {
            data: module.vmp_pmat_alloc(
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

    pub fn alloc_bytes<A>(module: &Module<B>, infos: &A) -> usize
    where
        A: GGSWInfos,
        Module<B>: VmpPMatAllocBytes,
    {
        Self::alloc_bytes_with(
            module,
            infos.base2k(),
            infos.k(),
            infos.dnum(),
            infos.dsize(),
            infos.rank(),
        )
    }

    pub fn alloc_bytes_with(module: &Module<B>, base2k: Base2K, k: TorusPrecision, dnum: Dnum, dsize: Dsize, rank: Rank) -> usize
    where
        Module<B>: VmpPMatAllocBytes,
    {
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

        module.vmp_pmat_alloc_bytes(dnum.into(), (rank + 1).into(), (rank + 1).into(), size)
    }
}

impl<D: DataRef, B: Backend> GGSWPrepared<D, B> {
    pub fn data(&self) -> &VmpPMat<D, B> {
        &self.data
    }
}

pub trait GGSWPrepareTmpBytes {
    fn ggsw_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos;
}

impl<B: Backend> GGSWPrepareTmpBytes for Module<B>
where
    Module<B>: VmpPrepareTmpBytes,
{
    fn ggsw_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        self.vmp_prepare_tmp_bytes(
            infos.dnum().into(),
            (infos.rank() + 1).into(),
            (infos.rank() + 1).into(),
            infos.size(),
        )
    }
}

impl<B: Backend> GGSWPrepared<Vec<u8>, B>
where
    Module<B>: GGSWPrepareTmpBytes,
{
    pub fn prepare_tmp_bytes<A>(&self, module: Module<B>, infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        module.ggsw_prepare_tmp_bytes(self)
    }
}

pub trait GGSWPrepare<B: Backend> {
    fn ggsw_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut Scratch<B>)
    where
        R: GGSWPreparedToMut<B>,
        O: GGSWToRef;
}

impl<B: Backend> GGSWPrepare<B> for Module<B>
where
    Module<B>: VmpPrepare<B>,
{
    fn ggsw_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut Scratch<B>)
    where
        R: GGSWPreparedToMut<B>,
        O: GGSWToRef,
    {
        let mut res: GGSWPrepared<&mut [u8], B> = res.to_mut();
        let other: GGSW<&[u8]> = other.to_ref();
        assert_eq!(res.k, other.k);
        assert_eq!(res.base2k, other.base2k);
        assert_eq!(res.dsize, other.dsize);
        self.vmp_prepare(&mut res.data, &other.data, scratch);
    }
}

impl<D: DataMut, B: Backend> GGSWPrepared<D, B>
where
    Module<B>: GGSWPrepare<B>,
{
    pub fn prepare<O>(&mut self, module: &Module<B>, other: &O, scratch: &mut Scratch<B>)
    where
        O: GGSWToRef,
    {
        module.ggsw_prepare(self, other, scratch);
    }
}

pub trait GGSWPrepareAlloc<B: Backend> {
    fn ggsw_prepare_alloc<O>(&self, other: &O, scratch: &mut Scratch<B>)
    where
        O: GGSWToRef;
}

impl<B: Backend> GGSWPrepareAlloc<B> for Module<B>
where
    Module<B>: GGSWPrepare<B>,
{
    fn ggsw_prepare_alloc<O>(&self, other: &O, scratch: &mut Scratch<B>)
    where
        O: GGSWToRef,
    {
        let mut ct_prepared: GGSWPrepared<Vec<u8>, B> = GGSWPrepared::alloc(self, other);
        self.ggsw_prepare(&mut ct_prepared, other, scratch);
        ct_prepared
    }
}

impl<D: DataRef> GGSW<D> {
    fn prepare_alloc<B: Backend>(&self, module: &Module<B>, scratch: &mut Scratch<B>)
    where
        Module<B>: GGSWPrepareAlloc<B>,
    {
        module.ggsw_prepare_alloc(self, scratch);
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

pub trait GGSWCiphertextPreparedToRef<B: Backend> {
    fn to_ref(&self) -> GGSWPrepared<&[u8], B>;
}

impl<D: DataRef, B: Backend> GGSWCiphertextPreparedToRef<B> for GGSWPrepared<D, B> {
    fn to_ref(&self) -> GGSWPrepared<&[u8], B> {
        GGSWPrepared {
            base2k: self.base2k,
            k: self.k,
            dsize: self.dsize,
            data: self.data.to_mut(),
        }
    }
}
