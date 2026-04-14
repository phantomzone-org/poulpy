use poulpy_hal::{
    api::{ModuleN, ScratchAvailable, ScratchTakeBasic, VecZnxCopy},
    layouts::{Backend, Module, Scratch},
    source::Source,
};

pub use crate::api::GGLWEToGGSWKeyEncryptSk;
use crate::{
    EncryptionInfos, GGLWEEncryptSk, GetDistribution, ScratchTakeCore,
    layouts::{
        GGLWEInfos, GGLWEToGGSWKey, GGLWEToGGSWKeyToMut, GLWEInfos, GLWESecret, GLWESecretTensor, GLWESecretTensorFactory,
        GLWESecretToRef, prepared::GLWESecretPreparedFactory,
    },
};

#[doc(hidden)]
pub trait GGLWEToGGSWKeyEncryptSkDefault<BE: Backend> {
    fn gglwe_to_ggsw_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn gglwe_to_ggsw_key_encrypt_sk<R, S, E>(
        &self,
        res: &mut R,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWEToGGSWKeyToMut,
        E: EncryptionInfos,
        S: GLWESecretToRef + GetDistribution + GLWEInfos;
}

impl<BE: Backend> GGLWEToGGSWKeyEncryptSkDefault<BE> for Module<BE>
where
    Self: ModuleN + GGLWEEncryptSk<BE> + GLWESecretTensorFactory<BE> + GLWESecretPreparedFactory<BE> + VecZnxCopy,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn gglwe_to_ggsw_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(self.n() as u32, infos.n());

        let sk_prepared: usize = self.glwe_secret_prepared_bytes_of(infos.rank());
        let sk_tensor: usize = GLWESecretTensor::bytes_of_from_infos(infos);
        let sk_ij: usize = GLWESecret::bytes_of(self.n().into(), infos.rank());
        let lvl_0: usize = sk_prepared;
        let lvl_1: usize = sk_tensor;
        let lvl_2: usize = sk_ij;
        let lvl_3_encrypt: usize = self.gglwe_encrypt_sk_tmp_bytes(infos);
        let lvl_3_prepare: usize = self.glwe_secret_tensor_prepare_tmp_bytes(infos.rank());
        let lvl_3: usize = lvl_3_encrypt.max(lvl_3_prepare);

        lvl_0 + lvl_1 + lvl_2 + lvl_3
    }

    fn gglwe_to_ggsw_key_encrypt_sk<R, S, E>(
        &self,
        res: &mut R,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,

        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWEToGGSWKeyToMut,
        E: EncryptionInfos,
        S: GLWESecretToRef + GetDistribution + GLWEInfos,
    {
        let res: &mut GGLWEToGGSWKey<&mut [u8]> = &mut res.to_mut();

        let rank: usize = res.rank_out().as_usize();
        assert!(
            scratch.available() >= self.gglwe_to_ggsw_key_encrypt_sk_tmp_bytes(res),
            "scratch.available(): {} < GGLWEToGGSWKeyEncryptSk::gglwe_to_ggsw_key_encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            self.gglwe_to_ggsw_key_encrypt_sk_tmp_bytes(res)
        );

        let (mut sk_prepared, scratch_1) = scratch.take_glwe_secret_prepared(self, res.rank());
        let (mut sk_tensor, scratch_2) = scratch_1.take_glwe_secret_tensor(self.n().into(), res.rank());
        self.glwe_secret_prepare(&mut sk_prepared, sk);
        self.glwe_secret_tensor_prepare(&mut sk_tensor, sk, scratch_2);

        let (mut sk_ij, scratch_3) = scratch_2.take_scalar_znx(self.n(), rank);

        for i in 0..rank {
            for j in 0..rank {
                self.vec_znx_copy(&mut sk_ij.as_vec_znx_mut(), j, &sk_tensor.at(i, j).as_vec_znx(), 0);
            }

            self.gglwe_encrypt_sk(
                res.at_mut(i),
                &sk_ij,
                &sk_prepared,
                enc_infos,
                source_xe,
                source_xa,
                scratch_3,
            );
        }
    }
}
