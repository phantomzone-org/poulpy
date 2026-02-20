use poulpy_hal::{
    api::{SvpPPolAlloc, SvpPPolBytesOf},
    layouts::{Backend, Data, DataMut, DataRef, Module, SvpPPol, SvpPPolToMut, SvpPPolToRef, ZnxInfos},
};

use crate::{
    GetDistribution, GetDistributionMut,
    dist::Distribution,
    layouts::{
        Base2K, Degree, GLWEInfos, GLWESecretPrepared, GLWESecretPreparedFactory, GLWESecretPreparedToMut,
        GLWESecretPreparedToRef, GLWESecretTensor, GLWESecretToRef, GetDegree, LWEInfos, Rank, TorusPrecision,
    },
};

/// DFT-domain (prepared) variant of [`GLWESecretTensor`].
///
/// Stores the GLWE secret tensor with polynomials in the frequency domain
/// for fast tensor operations. Tied to a specific backend via `B: Backend`.
pub struct GLWESecretTensorPrepared<D: Data, B: Backend> {
    pub(crate) data: SvpPPol<D, B>,
    pub(crate) rank: Rank,
    pub(crate) dist: Distribution,
}

impl<D: DataRef, BE: Backend> GetDistribution for GLWESecretTensorPrepared<D, BE> {
    fn dist(&self) -> &Distribution {
        &self.dist
    }
}

impl<D: DataMut, BE: Backend> GetDistributionMut for GLWESecretTensorPrepared<D, BE> {
    fn dist_mut(&mut self) -> &mut Distribution {
        &mut self.dist
    }
}

impl<D: Data, B: Backend> LWEInfos for GLWESecretTensorPrepared<D, B> {
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
impl<D: Data, B: Backend> GLWEInfos for GLWESecretTensorPrepared<D, B> {
    fn rank(&self) -> Rank {
        self.rank
    }
}

pub trait GLWESecretTensorPreparedFactory<B: Backend> {
    fn alloc_glwe_secret_tensor_prepared(&self, rank: Rank) -> GLWESecretTensorPrepared<Vec<u8>, B>;
    fn alloc_glwe_secret_tensor_prepared_from_infos<A>(&self, infos: &A) -> GLWESecretTensorPrepared<Vec<u8>, B>
    where
        A: GLWEInfos;

    fn bytes_of_glwe_secret_tensor_prepared(&self, rank: Rank) -> usize;
    fn bytes_of_glwe_secret_tensor_prepared_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn prepare_glwe_secret_tensor<R, O>(&self, res: &mut R, other: &O)
    where
        R: GLWESecretPreparedToMut<B> + GetDistributionMut,
        O: GLWESecretToRef + GetDistribution;
}

impl<B: Backend> GLWESecretTensorPreparedFactory<B> for Module<B>
where
    Self: GLWESecretPreparedFactory<B>,
{
    fn alloc_glwe_secret_tensor_prepared(&self, rank: Rank) -> GLWESecretTensorPrepared<Vec<u8>, B> {
        GLWESecretTensorPrepared {
            data: self.svp_ppol_alloc(GLWESecretTensor::pairs(rank.into())),
            rank,
            dist: Distribution::NONE,
        }
    }
    fn alloc_glwe_secret_tensor_prepared_from_infos<A>(&self, infos: &A) -> GLWESecretTensorPrepared<Vec<u8>, B>
    where
        A: GLWEInfos,
    {
        assert_eq!(self.ring_degree(), infos.n());
        self.alloc_glwe_secret_tensor_prepared(infos.rank())
    }

    fn bytes_of_glwe_secret_tensor_prepared(&self, rank: Rank) -> usize {
        self.bytes_of_svp_ppol(GLWESecretTensor::pairs(rank.into()))
    }
    fn bytes_of_glwe_secret_tensor_prepared_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        assert_eq!(self.ring_degree(), infos.n());
        self.bytes_of_glwe_secret_prepared(infos.rank())
    }

    fn prepare_glwe_secret_tensor<R, O>(&self, res: &mut R, other: &O)
    where
        R: GLWESecretPreparedToMut<B> + GetDistributionMut,
        O: GLWESecretToRef + GetDistribution,
    {
        self.prepare_glwe_secret(res, other);
    }
}

impl<B: Backend> GLWESecretTensorPrepared<Vec<u8>, B> {
    pub fn alloc_from_infos<A, M>(module: &M, infos: &A) -> Self
    where
        A: GLWEInfos,
        M: GLWESecretTensorPreparedFactory<B>,
    {
        module.alloc_glwe_secret_tensor_prepared_from_infos(infos)
    }

    pub fn alloc<M>(module: &M, rank: Rank) -> Self
    where
        M: GLWESecretTensorPreparedFactory<B>,
    {
        module.alloc_glwe_secret_tensor_prepared(rank)
    }

    pub fn bytes_of_from_infos<A, M>(module: &M, infos: &A) -> usize
    where
        A: GLWEInfos,
        M: GLWESecretTensorPreparedFactory<B>,
    {
        module.bytes_of_glwe_secret_tensor_prepared_from_infos(infos)
    }

    pub fn bytes_of<M>(module: &M, rank: Rank) -> usize
    where
        M: GLWESecretTensorPreparedFactory<B>,
    {
        module.bytes_of_glwe_secret_tensor_prepared(rank)
    }
}

impl<D: Data, B: Backend> GLWESecretTensorPrepared<D, B> {
    pub fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    pub fn rank(&self) -> Rank {
        Rank(self.data.cols() as u32)
    }
}

impl<D: DataMut, B: Backend> GLWESecretTensorPrepared<D, B> {
    pub fn prepare<M, O>(&mut self, module: &M, other: &O)
    where
        M: GLWESecretTensorPreparedFactory<B>,
        O: GLWESecretToRef + GetDistribution,
    {
        module.prepare_glwe_secret_tensor(self, other);
    }
}

impl<D: DataRef, B: Backend> GLWESecretPreparedToRef<B> for GLWESecretTensorPrepared<D, B> {
    fn to_ref(&self) -> GLWESecretPrepared<&[u8], B> {
        GLWESecretPrepared {
            data: self.data.to_ref(),
            dist: self.dist,
        }
    }
}

impl<D: DataMut, B: Backend> GLWESecretPreparedToMut<B> for GLWESecretTensorPrepared<D, B> {
    fn to_mut(&mut self) -> GLWESecretPrepared<&mut [u8], B> {
        GLWESecretPrepared {
            dist: self.dist,
            data: self.data.to_mut(),
        }
    }
}
