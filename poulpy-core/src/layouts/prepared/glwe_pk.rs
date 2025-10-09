use poulpy_hal::{
    api::{VecZnxDftAlloc, VecZnxDftAllocBytes, VecZnxDftApply},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch, VecZnxDft, ZnxInfos},
    oep::VecZnxDftAllocBytesImpl,
};

use crate::{
    dist::Distribution,
    layouts::{
        Base2K, BuildError, Degree, GLWEInfos, GLWEPublicKey, LWEInfos, Rank, TorusPrecision,
        prepared::{Prepare, PrepareAlloc, PrepareScratchSpace},
    },
};

#[derive(PartialEq, Eq)]
pub struct GLWEPublicKeyPrepared<D: Data, B: Backend> {
    pub(crate) data: VecZnxDft<D, B>,
    pub(crate) base2k: Base2K,
    pub(crate) k: TorusPrecision,
    pub(crate) dist: Distribution,
}

impl<D: Data, B: Backend> LWEInfos for GLWEPublicKeyPrepared<D, B> {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }

    fn size(&self) -> usize {
        self.data.size()
    }

    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }
}

impl<D: Data, B: Backend> GLWEInfos for GLWEPublicKeyPrepared<D, B> {
    fn rank(&self) -> Rank {
        Rank(self.data.cols() as u32 - 1)
    }
}

pub struct GLWEPublicKeyPreparedBuilder<D: Data, B: Backend> {
    data: Option<VecZnxDft<D, B>>,
    base2k: Option<Base2K>,
    k: Option<TorusPrecision>,
}

impl<D: Data, B: Backend> GLWEPublicKeyPrepared<D, B> {
    #[inline]
    pub fn builder() -> GLWEPublicKeyPreparedBuilder<D, B> {
        GLWEPublicKeyPreparedBuilder {
            data: None,
            base2k: None,
            k: None,
        }
    }
}

impl<B: Backend> GLWEPublicKeyPreparedBuilder<Vec<u8>, B> {
    #[inline]
    pub fn layout<A>(mut self, layout: &A) -> Self
    where
        A: GLWEInfos,
        B: VecZnxDftAllocBytesImpl<B>,
    {
        self.data = Some(VecZnxDft::alloc(
            layout.n().into(),
            (layout.rank() + 1).into(),
            layout.size(),
        ));
        self.base2k = Some(layout.base2k());
        self.k = Some(layout.k());
        self
    }
}

impl<D: Data, B: Backend> GLWEPublicKeyPreparedBuilder<D, B> {
    #[inline]
    pub fn data(mut self, data: VecZnxDft<D, B>) -> Self {
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

    pub fn build(self) -> Result<GLWEPublicKeyPrepared<D, B>, BuildError> {
        let data: VecZnxDft<D, B> = self.data.ok_or(BuildError::MissingData)?;
        let base2k: Base2K = self.base2k.ok_or(BuildError::MissingBase2K)?;
        let k: TorusPrecision = self.k.ok_or(BuildError::MissingK)?;

        if base2k == 0_u32 {
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

        Ok(GLWEPublicKeyPrepared {
            data,
            base2k,
            k,
            dist: Distribution::NONE,
        })
    }
}

impl<B: Backend> GLWEPublicKeyPrepared<Vec<u8>, B> {
    pub fn alloc<A>(module: &Module<B>, infos: &A) -> Self
    where
        A: GLWEInfos,
        Module<B>: VecZnxDftAlloc<B>,
    {
        debug_assert_eq!(module.n(), infos.n().0 as usize, "module.n() != infos.n()");
        Self::alloc_with(module, infos.base2k(), infos.k(), infos.rank())
    }

    pub fn alloc_with(module: &Module<B>, base2k: Base2K, k: TorusPrecision, rank: Rank) -> Self
    where
        Module<B>: VecZnxDftAlloc<B>,
    {
        Self {
            data: module.vec_znx_dft_alloc((rank + 1).into(), k.0.div_ceil(base2k.0) as usize),
            base2k,
            k,
            dist: Distribution::NONE,
        }
    }

    pub fn alloc_bytes<A>(module: &Module<B>, infos: &A) -> usize
    where
        A: GLWEInfos,
        Module<B>: VecZnxDftAllocBytes,
    {
        debug_assert_eq!(module.n(), infos.n().0 as usize, "module.n() != infos.n()");
        Self::alloc_bytes_with(module, infos.base2k(), infos.k(), infos.rank())
    }

    pub fn alloc_bytes_with(module: &Module<B>, base2k: Base2K, k: TorusPrecision, rank: Rank) -> usize
    where
        Module<B>: VecZnxDftAllocBytes,
    {
        module.vec_znx_dft_alloc_bytes((rank + 1).into(), k.0.div_ceil(base2k.0) as usize)
    }
}

impl<D: DataRef, B: Backend> PrepareAlloc<B, GLWEPublicKeyPrepared<Vec<u8>, B>> for GLWEPublicKey<D>
where
    Module<B>: VecZnxDftAlloc<B> + VecZnxDftApply<B>,
{
    fn prepare_alloc(&self, module: &Module<B>, scratch: &mut Scratch<B>) -> GLWEPublicKeyPrepared<Vec<u8>, B> {
        let mut pk_prepared: GLWEPublicKeyPrepared<Vec<u8>, B> = GLWEPublicKeyPrepared::alloc(module, self);
        pk_prepared.prepare(module, self, scratch);
        pk_prepared
    }
}

impl<DR: DataRef, B: Backend, A: GLWEInfos> PrepareScratchSpace<B, A> for GLWEPublicKeyPrepared<DR, B> {
    fn prepare_scratch_space(_module: &Module<B>, _infos: &A) -> usize {
        0
    }
}

impl<DM: DataMut, DR: DataRef, B: Backend> Prepare<B, GLWEPublicKey<DR>> for GLWEPublicKeyPrepared<DM, B>
where
    Module<B>: VecZnxDftApply<B>,
{
    fn prepare(&mut self, module: &Module<B>, other: &GLWEPublicKey<DR>, _scratch: &mut Scratch<B>) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.n(), other.n());
            assert_eq!(self.size(), other.size());
        }

        (0..(self.rank() + 1).into()).for_each(|i| {
            module.vec_znx_dft_apply(1, 0, &mut self.data, i, &other.data, i);
        });
        self.k = other.k();
        self.base2k = other.base2k();
        self.dist = other.dist;
    }
}
