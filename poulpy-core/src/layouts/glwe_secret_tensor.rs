use poulpy_hal::{
    api::{
        ModuleN, ScratchTakeBasic, SvpApplyDftToDft, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxDftApply, VecZnxDftBytesOf, VecZnxIdftApplyTmpA,
    },
    layouts::{
        Backend, Data, DataMut, DataRef, Module, ScalarZnx, ScalarZnxToMut, ScalarZnxToRef, Scratch, ZnxInfos, ZnxView,
        ZnxViewMut,
    },
};

use crate::{
    GetDistribution, ScratchTakeCore,
    dist::Distribution,
    layouts::{
        Base2K, Degree, GLWEInfos, GLWESecret, GLWESecretPreparedFactory, GLWESecretToMut, GLWESecretToRef, LWEInfos, Rank,
        TorusPrecision,
    },
};

pub struct GLWESecretTensor<D: Data> {
    pub(crate) data: ScalarZnx<D>,
    pub(crate) rank: Rank,
    pub(crate) dist: Distribution,
}

impl GLWESecretTensor<Vec<u8>> {
    pub(crate) fn pairs(rank: usize) -> usize {
        (((rank + 1) * rank) >> 1).max(1)
    }
}

impl<D: Data> GetDistribution for GLWESecretTensor<D> {
    fn dist(&self) -> &Distribution {
        &self.dist
    }
}

impl<D: Data> LWEInfos for GLWESecretTensor<D> {
    fn base2k(&self) -> Base2K {
        Base2K(0)
    }

    fn k(&self) -> TorusPrecision {
        TorusPrecision(0)
    }

    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    fn limbs(&self) -> usize {
        1
    }
}

impl<D: DataRef> GLWESecretTensor<D> {
    pub fn at(&self, mut i: usize, mut j: usize) -> ScalarZnx<&[u8]> {
        if i > j {
            std::mem::swap(&mut i, &mut j);
        };
        let rank: usize = self.rank().into();
        ScalarZnx {
            data: bytemuck::cast_slice(self.data.at(i * rank + j - (i * (i + 1) / 2), 0)),
            n: self.n().into(),
            cols: 1,
        }
    }
}

impl<D: DataMut> GLWESecretTensor<D> {
    pub fn at_mut(&mut self, mut i: usize, mut j: usize) -> ScalarZnx<&mut [u8]> {
        if i > j {
            std::mem::swap(&mut i, &mut j);
        };
        let rank: usize = self.rank().into();
        ScalarZnx {
            n: self.n().into(),
            data: bytemuck::cast_slice_mut(self.data.at_mut(i * rank + j - (i * (i + 1) / 2), 0)),
            cols: 1,
        }
    }
}

impl<D: Data> GLWEInfos for GLWESecretTensor<D> {
    fn rank(&self) -> Rank {
        self.rank
    }
}

impl<D: DataRef> GLWESecretToRef for GLWESecretTensor<D> {
    fn to_ref(&self) -> GLWESecret<&[u8]> {
        GLWESecret {
            data: self.data.to_ref(),
            dist: self.dist,
        }
    }
}

impl<D: DataMut> GLWESecretToMut for GLWESecretTensor<D> {
    fn to_mut(&mut self) -> GLWESecret<&mut [u8]> {
        GLWESecret {
            dist: self.dist,
            data: self.data.to_mut(),
        }
    }
}

impl GLWESecretTensor<Vec<u8>> {
    pub fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GLWEInfos,
    {
        Self::alloc(infos.n(), infos.rank())
    }

    pub fn alloc(n: Degree, rank: Rank) -> Self {
        GLWESecretTensor {
            data: ScalarZnx::alloc(n.into(), Self::pairs(rank.into())),
            rank,
            dist: Distribution::NONE,
        }
    }

    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        Self::bytes_of(infos.n(), Self::pairs(infos.rank().into()).into())
    }

    pub fn bytes_of(n: Degree, rank: Rank) -> usize {
        ScalarZnx::bytes_of(n.into(), Self::pairs(rank.into()))
    }
}

impl<D: DataMut> GLWESecretTensor<D> {
    pub fn prepare<M, S, BE: Backend>(&mut self, module: &M, other: &S, scratch: &mut Scratch<BE>)
    where
        M: GLWESecretTensorFactory<BE>,
        S: GLWESecretToRef + GLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.glwe_secret_tensor_prepare(self, other, scratch);
    }
}

pub trait GLWESecretTensorFactory<BE: Backend> {
    fn glwe_secret_tensor_prepare_tmp_bytes(&self, rank: Rank) -> usize;

    fn glwe_secret_tensor_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut Scratch<BE>)
    where
        R: GLWESecretToMut + GLWEInfos,
        O: GLWESecretToRef + GLWEInfos;
}

impl<BE: Backend> GLWESecretTensorFactory<BE> for Module<BE>
where
    Self: ModuleN
        + GLWESecretPreparedFactory<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxDftApply<BE>
        + SvpApplyDftToDft<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VecZnxBigNormalizeTmpBytes,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn glwe_secret_tensor_prepare_tmp_bytes(&self, rank: Rank) -> usize {
        self.bytes_of_glwe_secret_prepared(rank)
            + self.bytes_of_vec_znx_dft(rank.into(), 1)
            + self.bytes_of_vec_znx_dft(1, 1)
            + self.bytes_of_vec_znx_big(1, 1)
            + self.vec_znx_big_normalize_tmp_bytes()
    }

    fn glwe_secret_tensor_prepare<R, A>(&self, res: &mut R, a: &A, scratch: &mut Scratch<BE>)
    where
        R: GLWESecretToMut + GLWEInfos,
        A: GLWESecretToRef + GLWEInfos,
    {
        let res: &mut GLWESecret<&mut [u8]> = &mut res.to_mut();
        let a: &GLWESecret<&[u8]> = &a.to_ref();

        assert_eq!(res.rank(), GLWESecretTensor::pairs(a.rank().into()) as u32);
        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);

        let rank: usize = a.rank().into();

        let (mut a_prepared, scratch_1) = scratch.take_glwe_secret_prepared(self, rank.into());
        a_prepared.prepare(self, a);

        let base2k: usize = 17;

        let (mut a_dft, scratch_2) = scratch_1.take_vec_znx_dft(self, rank, 1);
        for i in 0..rank {
            self.vec_znx_dft_apply(1, 0, &mut a_dft, i, &a.data.as_vec_znx(), i);
        }

        let (mut a_ij_big, scratch_3) = scratch_2.take_vec_znx_big(self, 1, 1);
        let (mut a_ij_dft, scratch_4) = scratch_3.take_vec_znx_dft(self, 1, 1);

        // sk_tensor = sk (x) sk
        // For example: (s0, s1) (x) (s0, s1) = (s0^2, s0s1, s1^2)
        for i in 0..rank {
            for j in i..rank {
                let idx: usize = i * rank + j - (i * (i + 1) / 2);
                self.svp_apply_dft_to_dft(&mut a_ij_dft, 0, &a_prepared.data, j, &a_dft, i);
                self.vec_znx_idft_apply_tmpa(&mut a_ij_big, 0, &mut a_ij_dft, 0);

                self.vec_znx_big_normalize(
                    &mut res.data.as_vec_znx_mut(),
                    base2k,
                    0,
                    idx,
                    &a_ij_big,
                    base2k,
                    0,
                    scratch_4,
                );
            }
        }
    }
}
