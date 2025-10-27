use poulpy_hal::{
    api::{ModuleN, ScratchTakeBasic, VecZnxCopy},
    layouts::{Backend, DataMut, Module, Scratch},
    source::Source,
};

use crate::{
    GGLWEEncryptSk, GetDistribution, ScratchTakeCore,
    layouts::{
        GGLWEInfos, GGLWEToGGSWKey, GGLWEToGGSWKeyToMut, GLWEInfos, GLWESecret, GLWESecretTensor, GLWESecretTensorFactory,
        GLWESecretToRef,
        prepared::{GLWESecretPrepared, GLWESecretPreparedFactory},
    },
};

impl GGLWEToGGSWKey<Vec<u8>> {
    pub fn encrypt_sk_tmp_bytes<M, A, BE: Backend>(module: &M, infos: &A) -> usize
    where
        A: GGLWEInfos,
        M: GGLWEToGGSWKeyEncryptSk<BE>,
    {
        module.gglwe_to_ggsw_key_encrypt_sk_tmp_bytes(infos)
    }
}

impl<DataSelf: DataMut> GGLWEToGGSWKey<DataSelf> {
    pub fn encrypt_sk<M, S, BE: Backend>(
        &mut self,
        module: &M,
        sk: &S,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        M: GGLWEToGGSWKeyEncryptSk<BE>,
        S: GLWESecretToRef + GetDistribution + GLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.gglwe_to_ggsw_key_encrypt_sk(self, sk, source_xa, source_xe, scratch);
    }
}

pub trait GGLWEToGGSWKeyEncryptSk<BE: Backend> {
    fn gglwe_to_ggsw_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn gglwe_to_ggsw_key_encrypt_sk<R, S>(
        &self,
        res: &mut R,
        sk: &S,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWEToGGSWKeyToMut,
        S: GLWESecretToRef + GetDistribution + GLWEInfos;
}

impl<BE: Backend> GGLWEToGGSWKeyEncryptSk<BE> for Module<BE>
where
    Self: ModuleN + GGLWEEncryptSk<BE> + GLWESecretTensorFactory<BE> + GLWESecretPreparedFactory<BE> + VecZnxCopy,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn gglwe_to_ggsw_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        let sk_prepared: usize = GLWESecretPrepared::bytes_of(self, infos.rank());
        let sk_tensor: usize = GLWESecretTensor::bytes_of_from_infos(infos);
        let gglwe_encrypt: usize = self.gglwe_encrypt_sk_tmp_bytes(infos);
        let sk_ij = GLWESecret::bytes_of(self.n().into(), infos.rank());
        (sk_prepared + sk_tensor + sk_ij) + gglwe_encrypt.max(self.glwe_secret_tensor_prepare_tmp_bytes(infos.rank()))
    }

    fn gglwe_to_ggsw_key_encrypt_sk<R, S>(
        &self,
        res: &mut R,
        sk: &S,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWEToGGSWKeyToMut,
        S: GLWESecretToRef + GetDistribution + GLWEInfos,
    {
        let res: &mut GGLWEToGGSWKey<&mut [u8]> = &mut res.to_mut();

        let rank: usize = res.rank_out().as_usize();

        let (mut sk_prepared, scratch_1) = scratch.take_glwe_secret_prepared(self, res.rank());
        let (mut sk_tensor, scratch_2) = scratch_1.take_glwe_secret_tensor(self.n().into(), res.rank());
        sk_prepared.prepare(self, sk);
        sk_tensor.prepare(self, sk, scratch_2);

        let (mut sk_ij, scratch_3) = scratch_2.take_scalar_znx(self.n(), rank);

        for i in 0..rank {
            for j in 0..rank {
                self.vec_znx_copy(
                    &mut sk_ij.as_vec_znx_mut(),
                    j,
                    &sk_tensor.at(i, j).as_vec_znx(),
                    0,
                );
            }

            res.at_mut(i)
                .encrypt_sk(self, &sk_ij, &sk_prepared, source_xa, source_xe, scratch_3);
        }
    }
}
