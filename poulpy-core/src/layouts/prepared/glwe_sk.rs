use poulpy_hal::{
    api::{SvpPPolAlloc, SvpPPolAllocBytes, SvpPrepare},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch, SvpPPol, SvpPPolToMut, SvpPPolToRef, ZnxInfos},
};

use crate::{
    dist::Distribution,
    layouts::{
        Base2K, Degree, GLWEInfos, GLWESecret, GLWESecretToMut, GLWESecretToRef, LWEInfos, Rank, TorusPrecision,
        prepared::SetDist,
    },
};

pub struct GLWESecretPrepared<D: Data, B: Backend> {
    pub(crate) data: SvpPPol<D, B>,
    pub(crate) dist: Distribution,
}

impl<D: DataRef, B: Backend> SetDist for GLWESecretPrepared<D, B> {
    fn set_dist(&mut self, dist: Distribution) {
        self.dist = dist
    }
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

pub trait GLWESecretPrepareTmpBytes {
    fn glwe_secret_prepare_tmp_bytes<A>(&self, infos: &A)
    where
        A: GLWEInfos;
}

impl<B: Backend> GLWESecretPrepareTmpBytes for Module<B> {
    fn glwe_secret_prepare_tmp_bytes<A>(&self, infos: &A)
    where
        A: GLWEInfos,
    {
        0
    }
}

impl<B: Backend> GLWESecretPrepared<Vec<u8>, B>
where
    Module<B>: GLWESecretPrepareTmpBytes,
{
    fn prepare_tmp_bytes<A>(&self, module: &Module<B>, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        0
    }
}

pub trait GLWESecretPrepare<B: Backend> {
    fn glwe_secret_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut Scratch<B>)
    where
        R: GLWESecretPreparedToMut<B> + SetDist,
        O: GLWESecretToRef;
}

impl<B: Backend> GLWESecretPrepare<B> for Module<B>
where
    Module<B>: SvpPrepare<B>,
{
    fn glwe_secret_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut Scratch<B>)
    where
        R: GLWESecretPreparedToMut<B> + SetDist,
        O: GLWESecretToRef,
    {
        {
            let res: GLWESecretPrepared<&mut [u8], _> = res.to_mut();
            let other: GLWESecret<&[u8]> = other.to_ref();

            for i in 0..self.rank().into() {
                self.svp_prepare(&mut res.data, i, &other.data, i);
            }
        }

        res.set_dist(other.dist);
    }
}

pub trait GLWESecretPrepareAlloc<B: Backend> {
    fn glwe_secret_prepare_alloc<O>(&self, other: &O, scratch: &mut Scratch<B>)
    where
        O: GLWESecretToMut;
}

impl<B: Backend> GLWESecretPrepareAlloc<B> for Module<B>
where
    Module<B>: GLWESecretPrepare<B>,
{
    fn glwe_secret_prepare_alloc<O>(&self, other: &O, scratch: &mut Scratch<B>)
    where
        O: GLWESecretToMut,
    {
        let mut ct_prep: GLWESecretPrepared<Vec<u8>, B> = GLWESecretPrepared::alloc(self, self);
        self.glwe_secret_prepare(&mut ct_prep, other, scratch);
        ct_prep
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
