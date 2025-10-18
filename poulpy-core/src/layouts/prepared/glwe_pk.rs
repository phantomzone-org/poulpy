use poulpy_hal::{
    api::{VecZnxDftAlloc, VecZnxDftApply, VecZnxDftBytesOf},
    layouts::{Backend, Data, DataMut, DataRef, Module, VecZnxDft, VecZnxDftToMut, VecZnxDftToRef, ZnxInfos},
};

use crate::{
    dist::Distribution,
    layouts::{Base2K, Degree, GLWEInfos, GLWEPublicKey, GLWEPublicKeyToRef, GetDegree, GetDist, LWEInfos, Rank, TorusPrecision},
};

#[derive(PartialEq, Eq)]
pub struct GLWEPublicKeyPrepared<D: Data, B: Backend> {
    pub(crate) data: VecZnxDft<D, B>,
    pub(crate) base2k: Base2K,
    pub(crate) k: TorusPrecision,
    pub(crate) dist: Distribution,
}

pub trait SetDist {
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

pub trait GLWEPublicKeyPreparedAlloc<B: Backend>
where
    Self: GetDegree + VecZnxDftAlloc<B> + VecZnxDftBytesOf,
{
    fn alloc_glwe_public_key_prepared(&self, base2k: Base2K, k: TorusPrecision, rank: Rank) -> GLWEPublicKeyPrepared<Vec<u8>, B> {
        GLWEPublicKeyPrepared {
            data: self.vec_znx_dft_alloc((rank + 1).into(), k.0.div_ceil(base2k.0) as usize),
            base2k,
            k,
            dist: Distribution::NONE,
        }
    }

    fn alloc_glwe_public_key_prepared_from_infos<A>(&self, infos: &A) -> GLWEPublicKeyPrepared<Vec<u8>, B>
    where
        A: GLWEInfos,
    {
        self.alloc_glwe_public_key_prepared(infos.base2k(), infos.k(), infos.rank())
    }

    fn bytes_of_glwe_public_key_prepared(&self, base2k: Base2K, k: TorusPrecision, rank: Rank) -> usize {
        self.bytes_of_vec_znx_dft((rank + 1).into(), k.0.div_ceil(base2k.0) as usize)
    }

    fn bytes_of_glwe_public_key_prepared_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        self.bytes_of_glwe_public_key_prepared(infos.base2k(), infos.k(), infos.rank())
    }
}

impl<B: Backend> GLWEPublicKeyPreparedAlloc<B> for Module<B> where Self: VecZnxDftAlloc<B> + VecZnxDftBytesOf {}

impl<B: Backend> GLWEPublicKeyPrepared<Vec<u8>, B> {
    pub fn alloc_from_infos<A, M>(module: &M, infos: &A) -> Self
    where
        A: GLWEInfos,
        M: GLWEPublicKeyPreparedAlloc<B>,
    {
        module.alloc_glwe_public_key_prepared_from_infos(infos)
    }

    pub fn alloc<M>(module: &M, base2k: Base2K, k: TorusPrecision, rank: Rank) -> Self
    where
        M: GLWEPublicKeyPreparedAlloc<B>,
    {
        module.alloc_glwe_public_key_prepared(base2k, k, rank)
    }

    pub fn bytes_of_from_infos<A, M>(module: &M, infos: &A) -> usize
    where
        A: GLWEInfos,
        M: GLWEPublicKeyPreparedAlloc<B>,
    {
        module.bytes_of_glwe_public_key_prepared_from_infos(infos)
    }

    pub fn bytes_of<M>(module: &M, base2k: Base2K, k: TorusPrecision, rank: Rank) -> usize
    where
        M: GLWEPublicKeyPreparedAlloc<B>,
    {
        module.bytes_of_glwe_public_key_prepared(base2k, k, rank)
    }
}

pub trait GLWEPublicKeyPrepare<B: Backend>
where
    Self: GetDegree + VecZnxDftApply<B>,
{
    fn prepare_glwe_public_key<R, O>(&self, res: &mut R, other: &O)
    where
        R: GLWEPublicKeyPreparedToMut<B> + SetDist,
        O: GLWEPublicKeyToRef + GetDist,
    {
        {
            let mut res: GLWEPublicKeyPrepared<&mut [u8], B> = res.to_mut();
            let other: GLWEPublicKey<&[u8]> = other.to_ref();

            assert_eq!(res.n(), self.ring_degree());
            assert_eq!(other.n(), self.ring_degree());
            assert_eq!(res.size(), other.size());
            assert_eq!(res.k(), other.k());
            assert_eq!(res.base2k(), other.base2k());

            for i in 0..(res.rank() + 1).into() {
                self.vec_znx_dft_apply(1, 0, &mut res.data, i, &other.data, i);
            }
        }

        res.set_dist(other.get_dist());
    }
}

impl<B: Backend> GLWEPublicKeyPrepare<B> for Module<B> where Self: GetDegree + VecZnxDftApply<B> {}

impl<D: DataMut, B: Backend> GLWEPublicKeyPrepared<D, B> {
    pub fn prepare<O, M>(&mut self, module: &M, other: &O)
    where
        O: GLWEPublicKeyToRef + GetDist,
        M: GLWEPublicKeyPrepare<B>,
    {
        module.prepare_glwe_public_key(self, other);
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
