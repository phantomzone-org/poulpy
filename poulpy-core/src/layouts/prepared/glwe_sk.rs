use poulpy_hal::{
    api::{SvpPPolAlloc, SvpPPolAllocBytes, SvpPrepare},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch, SvpPPol, SvpPPolToMut, SvpPPolToRef, ZnxInfos},
};

use crate::{
    dist::Distribution,
    layouts::{Base2K, Degree, GLWEInfos, GLWESecret, LWEInfos, Rank, TorusPrecision},
};

pub struct GLWESecretPrepared<D: Data, B: Backend> {
    pub(crate) data: SvpPPol<D, B>,
    pub(crate) dist: Distribution,
}

impl<D: Data, B: Backend> LWEInfos for GLWESecretPrepared<D, B> {
    fn base2k(&self) -> Base2K {
        Base2K(0)
    }

    fn k(&self) -> TorusPrecision {
        TorusPrecision(0)
    }

    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    fn size(&self) -> usize {
        self.data.size()
    }
}
impl<D: Data, B: Backend> GLWEInfos for GLWESecretPrepared<D, B> {
    fn rank(&self) -> Rank {
        Rank(self.data.cols() as u32)
    }
}
impl<B: Backend> GLWESecretPrepared<Vec<u8>, B> {
    pub fn alloc<A>(module: &Module<B>, infos: &A) -> Self
    where
        A: GLWEInfos,
        Module<B>: SvpPPolAlloc<B>,
    {
        assert_eq!(module.n() as u32, infos.n());
        Self::alloc_with(module, infos.rank())
    }

    pub fn alloc_with(module: &Module<B>, rank: Rank) -> Self
    where
        Module<B>: SvpPPolAlloc<B>,
    {
        Self {
            data: module.svp_ppol_alloc(rank.into()),
            dist: Distribution::NONE,
        }
    }

    pub fn alloc_bytes<A>(module: &Module<B>, infos: &A) -> usize
    where
        A: GLWEInfos,
        Module<B>: SvpPPolAllocBytes,
    {
        assert_eq!(module.n() as u32, infos.n());
        Self::alloc_bytes_with(module, infos.rank())
    }

    pub fn alloc_bytes_with(module: &Module<B>, rank: Rank) -> usize
    where
        Module<B>: SvpPPolAllocBytes,
    {
        module.svp_ppol_alloc_bytes(rank.into())
    }
}

impl<D: Data, B: Backend> GLWESecretPrepared<D, B> {
    pub fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    pub fn rank(&self) -> Rank {
        Rank(self.data.cols() as u32)
    }
}

impl<B: Backend, A: GLWEInfos> PrepareScratchSpace<B, A> for GLWESecretPrepared<Vec<u8>, B> {
    fn prepare_scratch_space(_module: &Module<B>, _infos: &A) -> usize {
        0
    }
}

impl<D: DataRef, B: Backend> PrepareAlloc<B, GLWESecretPrepared<Vec<u8>, B>> for GLWESecret<D>
where
    Module<B>: SvpPrepare<B> + SvpPPolAlloc<B>,
{
    fn prepare_alloc(&self, module: &Module<B>, _scratch: &mut Scratch<B>) -> GLWESecretPrepared<Vec<u8>, B> {
        let mut sk_dft: GLWESecretPrepared<Vec<u8>, B> = GLWESecretPrepared::alloc(module, self);
        sk_dft.prepare(module, self, _scratch);
        sk_dft
    }
}

impl<DM: DataMut, DR: DataRef, B: Backend> Prepare<B, GLWESecret<DR>> for GLWESecretPrepared<DM, B>
where
    Module<B>: SvpPrepare<B>,
{
    fn prepare(&mut self, module: &Module<B>, other: &GLWESecret<DR>, _scratch: &mut Scratch<B>) {
        (0..self.rank().into()).for_each(|i| {
            module.svp_prepare(&mut self.data, i, &other.data, i);
        });
        self.dist = other.dist
    }
}

pub trait GLWESecretPreparedToRef<B: Backend> {
    fn to_ref(&self) -> GLWESecretPrepared<&[u8], B>;
}

impl<D: DataRef, B: Backend> GLWESecretPreparedToRef<B> for GLWESecretPrepared<D, B> {
    fn to_ref(&self) -> GLWESecretPrepared<&[u8], B> {
        GLWESecretPrepared {
            data: self.data.to_ref(),
            dist: self.dist,
        }
    }
}

pub trait GLWESecretPreparedToMut<B: Backend> {
    fn to_mut(&mut self) -> GLWESecretPrepared<&mut [u8], B>;
}

impl<D: DataMut, B: Backend> GLWESecretPreparedToMut<B> for GLWESecretPrepared<D, B> {
    fn to_mut(&mut self) -> GLWESecretPrepared<&mut [u8], B> {
        GLWESecretPrepared {
            dist: self.dist,
            data: self.data.to_mut(),
        }
    }
}
