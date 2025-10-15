use poulpy_hal::{
    api::{SvpPPolAlloc, SvpPPolBytesOf, SvpPrepare},
    layouts::{Backend, Data, DataMut, DataRef, Module, SvpPPol, SvpPPolToMut, SvpPPolToRef, ZnxInfos},
};

use crate::{
    dist::Distribution,
    layouts::{
        Base2K, GLWEInfos, GLWESecret, GLWESecretToRef, GetDist, GetRingDegree, LWEInfos, Rank, RingDegree, TorusPrecision,
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

    fn n(&self) -> RingDegree {
        RingDegree(self.data.n() as u32)
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

pub trait GLWESecretPreparedAlloc<B: Backend>
where
    Self: GetRingDegree + SvpPPolBytesOf + SvpPPolAlloc<B>,
{
    fn alloc_glwe_secret_prepared(&self, rank: Rank) -> GLWESecretPrepared<Vec<u8>, B> {
        GLWESecretPrepared {
            data: self.svp_ppol_alloc(rank.into()),
            dist: Distribution::NONE,
        }
    }
    fn alloc_glwe_secret_prepared_from_infos<A>(&self, infos: &A) -> GLWESecretPrepared<Vec<u8>, B>
    where
        A: GLWEInfos,
    {
        assert_eq!(self.ring_degree(), infos.n());
        self.alloc_glwe_secret_prepared(infos.rank())
    }

    fn bytes_of_glwe_secret(&self, rank: Rank) -> usize {
        self.bytes_of_svp_ppol(rank.into())
    }
    fn bytes_of_glwe_secret_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        assert_eq!(self.ring_degree(), infos.n());
        self.bytes_of_glwe_secret(infos.rank())
    }
}

impl<B: Backend> GLWESecretPreparedAlloc<B> for Module<B> where Self: GetRingDegree + SvpPPolBytesOf + SvpPPolAlloc<B> {}

impl<B: Backend> GLWESecretPrepared<Vec<u8>, B> {
    pub fn alloc_from_infos<A, M>(module: &M, infos: &A) -> Self
    where
        A: GLWEInfos,
        M: GLWESecretPreparedAlloc<B>,
    {
        module.alloc_glwe_secret_prepared_from_infos(infos)
    }

    pub fn alloc<M>(module: &M, rank: Rank) -> Self
    where
        M: GLWESecretPreparedAlloc<B>,
    {
        module.alloc_glwe_secret_prepared(rank)
    }

    pub fn bytes_of_from_infos<A, M>(module: &M, infos: &A) -> usize
    where
        A: GLWEInfos,
        M: GLWESecretPreparedAlloc<B>,
    {
        module.bytes_of_glwe_secret_from_infos(infos)
    }

    pub fn bytes_of<M>(module: &M, rank: Rank) -> usize
    where
        M: GLWESecretPreparedAlloc<B>,
    {
        module.bytes_of_glwe_secret(rank)
    }
}

impl<D: Data, B: Backend> GLWESecretPrepared<D, B> {
    pub fn n(&self) -> RingDegree {
        RingDegree(self.data.n() as u32)
    }

    pub fn rank(&self) -> Rank {
        Rank(self.data.cols() as u32)
    }
}

pub trait GLWESecretPrepare<B: Backend>
where
    Self: SvpPrepare<B>,
{
    fn prepare_glwe_secret<R, O>(&self, res: &mut R, other: &O)
    where
        R: GLWESecretPreparedToMut<B> + SetDist,
        O: GLWESecretToRef + GetDist,
    {
        {
            let mut res: GLWESecretPrepared<&mut [u8], _> = res.to_mut();
            let other: GLWESecret<&[u8]> = other.to_ref();

            for i in 0..res.rank().into() {
                self.svp_prepare(&mut res.data, i, &other.data, i);
            }
        }

        res.set_dist(other.get_dist());
    }
}

impl<B: Backend> GLWESecretPrepare<B> for Module<B> where Self: SvpPrepare<B> {}

impl<D: DataMut, B: Backend> GLWESecretPrepared<D, B> {
    pub fn prepare<M, O>(&mut self, module: &M, other: &O)
    where
        M: GLWESecretPrepare<B>,
        O: GLWESecretToRef + GetDist,
    {
        module.prepare_glwe_secret(self, other);
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
