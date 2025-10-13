use poulpy_hal::{
    api::{VecZnxDftAlloc, VecZnxDftAllocBytes, VecZnxDftApply},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch, VecZnxDft, VecZnxDftToMut, VecZnxDftToRef, ZnxInfos},
    oep::VecZnxDftAllocBytesImpl,
};

use crate::{
    dist::Distribution,
    layouts::{Base2K, BuildError, Degree, GLWEInfos, GLWEPublicKey, GLWEPublicKeyToRef, LWEInfos, Rank, TorusPrecision},
};

#[derive(PartialEq, Eq)]
pub struct GLWEPublicKeyPrepared<D: Data, B: Backend> {
    pub(crate) data: VecZnxDft<D, B>,
    pub(crate) base2k: Base2K,
    pub(crate) k: TorusPrecision,
    pub(crate) dist: Distribution,
}

pub(crate) trait SetDist {
    fn set_dist(&mut self, dist: Distribution);
}

impl<D: Data, B: Backend> SetDist for GLWEPublicKeyPrepared<D, B> {
    fn set_dist(&mut self, dist: Distribution) {
        self.dist = dist
    }
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

pub trait GLWEPublicKeyPrepareTmpBytes {
    fn glwe_public_key_prepare_tmp_bytes<A>(&self, infos: &A)
    where
        A: GLWEInfos;
}

impl<B: Backend> GLWEPublicKeyPrepareTmpBytes for Module<B> {
    fn glwe_public_key_prepare_tmp_bytes<A>(&self, infos: &A)
    where
        A: GLWEInfos,
    {
        0
    }
}

impl<B: Backend> GLWEPublicKeyPrepared<Vec<u8>, B> {
    pub fn prepare_tmp_bytes<A>(&self, module: &Module<B>, infos: &A)
    where
        A: GLWEInfos,
        Module<B>: GLWEPublicKeyPrepareTmpBytes,
    {
        module.glwe_public_key_prepare_tmp_bytes(infos);
    }
}

pub trait GLWEPublicKeyPrepare<B: Backend> {
    fn glwe_public_key_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &Scratch<B>)
    where
        R: GLWEPublicKeyPreparedToMut<B> + SetDist,
        O: GLWEPublicKeyToRef;
}

impl<B: Backend> GLWEPublicKeyPrepare<B> for Module<B>
where
    Module<B>: VecZnxDftAlloc<B> + VecZnxDftApply<B>,
{
    fn glwe_public_key_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &Scratch<B>)
    where
        R: GLWEPublicKeyPreparedToMut<B> + SetDist,
        O: GLWEPublicKeyToRef,
    {
        {
            let res: GLWEPublicKeyPrepared<&mut [u8], B> = res.to_mut();
            let other: GLWEPublicKey<&[u8]> = other.to_ref();

            assert_eq!(res.n(), self.n() as u32);
            assert_eq!(other.n(), self.n() as u32);
            assert_eq!(res.size(), other.size());
            assert_eq!(res.k(), other.k());
            assert_eq!(res.base2k(), other.base2k());

            for i in 0..(self.rank() + 1).into() {
                self.vec_znx_dft_apply(1, 0, &mut self.data, i, &other.data, i);
            }
        }

        res.set_dist(other.dist);
    }
}

impl<D: DataMut, B: Backend> GLWEPublicKeyPrepared<D, B>
where
    Module<B>: GLWEPublicKeyPrepare<B>,
{
    pub fn prepare<O>(&mut self, module: &Module<B>, other: &O, scratch: &mut Scratch<B>)
    where
        O: GLWEPublicKeyToRef,
    {
        module.glwe_public_key_prepare(self, other, scratch);
    }
}

pub trait GLWEPublicKeyPrepareAlloc<B: Backend> {
    fn glwe_public_key_prepare_alloc<O>(&self, other: &O, scratch: &mut Scratch<B>)
    where
        O: GLWEPublicKeyToRef;
}

impl<B: Backend> GLWEPublicKeyPrepareAlloc<B> for Module<B>
where
    Module<B>: GLWEPublicKeyPrepare<B>,
{
    fn glwe_public_key_prepare_alloc<O>(&self, other: &O, scratch: &mut Scratch<B>)
    where
        O: GLWEPublicKeyToRef,
    {
        let mut ct_prepared: GLWEPublicKeyPrepared<Vec<u8>, B> = GLWEPublicKeyPrepared::alloc(self, other);
        self.glwe_public_key_prepare(&mut ct_prepared, ct_prepared, scratch);
        ct_prepared
    }
}

impl<D: DataRef> GLWEPublicKey<D> {
    pub fn prepare_alloc<B: Backend>(&self, module: &Module<B>, scratch: &mut Scratch<B>)
    where
        Module<B>: GLWEPublicKeyPrepareAlloc<B>,
    {
        module.glwe_public_key_prepare_alloc(self, scratch);
    }
}

pub trait GLWEPublicKeyPreparedToMut<B: Backend> {
    fn to_mut(&mut self) -> GLWEPublicKeyPrepared<&mut [u8], B>;
}

impl<D: DataMut, B: Backend> GLWEPublicKeyPreparedToMut<B> for GLWEPublicKeyPrepared<D, B> {
    fn to_mut(&mut self) -> GLWEPublicKeyPrepared<&mut [u8], B> {
        GLWEPublicKeyPrepared {
            dist: self.dist,
            k: self.k,
            base2k: self.base2k,
            data: self.data.to_mut(),
        }
    }
}

pub trait GLWEPublicKeyPreparedToRef<B: Backend> {
    fn to_ref(&self) -> GLWEPublicKeyPrepared<&[u8], B>;
}

impl<D: DataRef, B: Backend> GLWEPublicKeyPreparedToRef<B> for GLWEPublicKeyPrepared<D, B> {
    fn to_ref(&self) -> GLWEPublicKeyPrepared<&[u8], B> {
        GLWEPublicKeyPrepared {
            data: self.data.to_ref(),
            dist: self.dist,
            k: self.k,
            base2k: self.base2k,
        }
    }
}
