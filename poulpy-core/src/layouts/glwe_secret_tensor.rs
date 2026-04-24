use poulpy_hal::{
    api::{
        ModuleN, ScratchAvailable, ScratchTakeBasic, SvpApplyDftToDft, SvpPrepare, VecZnxBigBytesOf, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxDftApply, VecZnxDftBytesOf, VecZnxIdftApplyTmpA,
    },
    layouts::{
        Backend, Data, DataMut, DataRef, Module, ScalarZnx, ScalarZnxAsVecZnxBackendRef, ScalarZnxToMut, ScalarZnxToRef, Scratch,
        ScratchOwned, SvpPPolToBackendMut, SvpPPolToBackendRef, VecZnx, VecZnxBig, VecZnxBigToBackendMut, VecZnxBigToBackendRef,
        VecZnxDft, VecZnxDftToBackendMut, VecZnxToBackendMut, ZnxInfos, ZnxView, ZnxViewMut,
    },
};

use crate::{
    GetDistribution, ScratchTakeCore,
    dist::Distribution,
    layouts::{
        Base2K, Degree, GLWEInfos, GLWESecret, GLWESecretPreparedFactory, GLWESecretToMut, GLWESecretToRef, LWEInfos, Rank,
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

    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    fn size(&self) -> usize {
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

// module-only API: secret tensor preparation is provided by `GLWESecretTensorFactory` on `Module`.

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
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    fn glwe_secret_tensor_prepare_tmp_bytes(&self, rank: Rank) -> usize {
        let lvl_0: usize = self.glwe_secret_prepared_bytes_of(rank);
        let lvl_1: usize = self.bytes_of_vec_znx_dft(rank.into(), 1);
        let lvl_2: usize = self.bytes_of_vec_znx_big(1, 1);
        let lvl_3: usize = self.bytes_of_vec_znx_dft(1, 1);
        let lvl_4: usize = self.vec_znx_big_normalize_tmp_bytes();

        lvl_0 + lvl_1 + lvl_2 + lvl_3 + lvl_4
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
        assert!(
            scratch.available() >= self.glwe_secret_tensor_prepare_tmp_bytes(a.rank()),
            "scratch.available(): {} < GLWESecretTensorFactory::glwe_secret_tensor_prepare_tmp_bytes: {}",
            scratch.available(),
            self.glwe_secret_tensor_prepare_tmp_bytes(a.rank())
        );

        let rank: usize = a.rank().into();

        let mut a_prepared = self.glwe_secret_prepared_alloc(rank.into());
        {
            let mut a_prepared_data = a_prepared.data.to_backend_mut();
            for i in 0..rank {
                self.svp_prepare(&mut a_prepared_data, i, &a.data, i);
            }
        }
        a_prepared.dist = *a.dist();

        let base2k: usize = 17;

        let (mut a_dft, scratch_2) = scratch.take_vec_znx_dft(self, rank, 1);
        let a_backend = GLWESecret {
            data: ScalarZnx::from_data(BE::from_host_bytes(a.data.data), a.data.n, a.data.cols),
            dist: *a.dist(),
        };
        let a_backend_vec = <ScalarZnx<BE::OwnedBuf> as ScalarZnxAsVecZnxBackendRef<BE>>::as_vec_znx_backend(&a_backend.data);
        let mut a_i_dft_backend = VecZnxDft::<BE::OwnedBuf, BE>::alloc(self.n(), 1, 1);
        for i in 0..rank {
            {
                let mut a_i_dft = a_i_dft_backend.to_backend_mut();
                self.vec_znx_dft_apply(1, 0, &mut a_i_dft, 0, &a_backend_vec, i);
            }
            BE::copy_to_host(&a_i_dft_backend.data, bytemuck::cast_slice_mut(a_dft.at_mut(i, 0)));
        }

        let (mut a_ij_dft, _) = scratch_2.take_vec_znx_dft(self, 1, 1);
        let a_prepared_ref = a_prepared.data.to_backend_ref();
        let mut a_ij_dft_backend = VecZnxDft::<BE::OwnedBuf, BE>::alloc(self.n(), 1, 1);
        let mut a_ij_big_backend = VecZnxBig::<BE::OwnedBuf, BE>::alloc(self.n(), 1, 1);
        let mut a_ij_small_backend: VecZnx<BE::OwnedBuf> =
            VecZnx::from_data(BE::alloc_bytes(VecZnx::<Vec<u8>>::bytes_of(self.n(), 1, 1)), self.n(), 1, 1);
        let mut norm_scratch = ScratchOwned {
            data: BE::alloc_bytes(self.vec_znx_big_normalize_tmp_bytes()),
            _phantom: std::marker::PhantomData,
        };

        // sk_tensor = sk (x) sk
        // For example: (s0, s1) (x) (s0, s1) = (s0^2, s0s1, s1^2)
        for i in 0..rank {
            for j in i..rank {
                let idx: usize = i * rank + j - (i * (i + 1) / 2);
                self.svp_apply_dft_to_dft(&mut a_ij_dft, 0, &a_prepared_ref, j, &a_dft, i);
                BE::copy_from_host(&mut a_ij_dft_backend.data, a_ij_dft.data);
                {
                    let mut a_ij_big = a_ij_big_backend.to_backend_mut();
                    let mut a_ij_dft = a_ij_dft_backend.to_backend_mut();
                    self.vec_znx_idft_apply_tmpa(&mut a_ij_big, 0, &mut a_ij_dft, 0);
                }
                {
                    let a_ij_big = a_ij_big_backend.to_backend_ref();
                    let mut a_ij_small =
                        <VecZnx<BE::OwnedBuf> as VecZnxToBackendMut<BE>>::to_backend_mut(&mut a_ij_small_backend);
                    self.vec_znx_big_normalize(
                        &mut a_ij_small,
                        base2k,
                        0,
                        0,
                        &a_ij_big,
                        base2k,
                        0,
                        &mut norm_scratch.arena(),
                    );
                }
                BE::copy_to_host(&a_ij_small_backend.data, bytemuck::cast_slice_mut(res.data.at_mut(idx, 0)));
            }
        }
    }
}
