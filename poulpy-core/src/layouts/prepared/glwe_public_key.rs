use poulpy_hal::{
    api::{VecZnxDftAlloc, VecZnxDftApply, VecZnxDftBytesOf},
    layouts::{Backend, Data, DataMut, DataRef, DeviceBuf, Module},
};

use crate::{
    GetDistribution, GetDistributionMut,
    dist::Distribution,
    layouts::{
        Base2K, Degree, GLWEInfos, GLWEPrepared, GLWEPreparedFactory, GLWEPreparedToMut, GLWEPreparedToRef, GLWEToRef, GetDegree,
        LWEInfos, Rank, TorusPrecision,
    },
};

/// DFT-domain (prepared) variant of a GLWE public key.
///
/// Wraps a [`GLWEPrepared`] with distribution metadata for public-key
/// encryption. Tied to a specific backend via `B: Backend`.
#[derive(PartialEq, Eq)]
pub struct GLWEPublicKeyPrepared<D: Data, B: Backend> {
    pub(crate) key: GLWEPrepared<D, B>,
    pub(crate) dist: Distribution,
}

impl<D: DataRef, BE: Backend> GetDistribution for GLWEPublicKeyPrepared<D, BE> {
    fn dist(&self) -> &Distribution {
        &self.dist
    }
}

impl<D: DataMut, BE: Backend> GetDistributionMut for GLWEPublicKeyPrepared<D, BE> {
    fn dist_mut(&mut self) -> &mut Distribution {
        &mut self.dist
    }
}

impl<D: Data, B: Backend> LWEInfos for GLWEPublicKeyPrepared<D, B> {
    fn base2k(&self) -> Base2K {
        self.key.base2k()
    }

    fn size(&self) -> usize {
        self.key.size()
    }

    fn n(&self) -> Degree {
        self.key.n()
    }
}

impl<D: Data, B: Backend> GLWEInfos for GLWEPublicKeyPrepared<D, B> {
    fn rank(&self) -> Rank {
        self.key.rank()
    }
}

pub trait GLWEPublicKeyPreparedFactory<B: Backend>
where
    Self: GetDegree + GLWEPreparedFactory<B>,
{
    fn alloc_glwe_public_key_prepared(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank: Rank,
    ) -> GLWEPublicKeyPrepared<DeviceBuf<B>, B> {
        GLWEPublicKeyPrepared {
            key: self.alloc_glwe_prepared(base2k, k, rank),
            dist: Distribution::NONE,
        }
    }

    fn alloc_glwe_public_key_prepared_from_infos<A>(&self, infos: &A) -> GLWEPublicKeyPrepared<DeviceBuf<B>, B>
    where
        A: GLWEInfos,
    {
        self.alloc_glwe_public_key_prepared(infos.base2k(), infos.max_k(), infos.rank())
    }

    fn bytes_of_glwe_public_key_prepared(&self, base2k: Base2K, k: TorusPrecision, rank: Rank) -> usize {
        self.bytes_of_glwe_prepared(base2k, k, rank)
    }

    fn bytes_of_glwe_public_key_prepared_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        self.bytes_of_glwe_public_key_prepared(infos.base2k(), infos.max_k(), infos.rank())
    }

    fn prepare_glwe_public_key<R, O>(&self, res: &mut R, other: &O)
    where
        R: GLWEPreparedToMut<B> + GetDistributionMut,
        O: GLWEToRef + GetDistribution,
    {
        self.prepare_glwe(res, other);
        *res.dist_mut() = *other.dist();
    }
}

impl<B: Backend> GLWEPublicKeyPreparedFactory<B> for Module<B> where Self: VecZnxDftAlloc<B> + VecZnxDftBytesOf + VecZnxDftApply<B>
{}

// module-only API: allocation, sizing, and preparation are provided by
// `GLWEPublicKeyPreparedFactory` on `Module`.

impl<D: DataMut, B: Backend> GLWEPreparedToMut<B> for GLWEPublicKeyPrepared<D, B>
where
    GLWEPrepared<D, B>: GLWEPreparedToMut<B>,
{
    fn to_mut(&mut self) -> GLWEPrepared<&mut [u8], B> {
        self.key.to_mut()
    }
}

impl<D: DataRef, B: Backend> GLWEPreparedToRef<B> for GLWEPublicKeyPrepared<D, B>
where
    GLWEPrepared<D, B>: GLWEPreparedToRef<B>,
{
    fn to_ref(&self) -> GLWEPrepared<&[u8], B> {
        self.key.to_ref()
    }
}
