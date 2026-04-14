use poulpy_hal::{
    api::{ScratchAvailable, ScratchTakeBasic},
    layouts::{Backend, Module, Scratch},
    source::Source,
};

pub use crate::api::GLWETensorKeyCompressedEncryptSk;
use crate::{
    EncryptionInfos, GGLWECompressedEncryptSk, GetDistribution, ScratchTakeCore,
    layouts::{
        GGLWECompressedSeedMut, GGLWECompressedToMut, GGLWEInfos, GGLWELayout, GLWEInfos, GLWESecretPreparedFactory,
        GLWESecretTensor, GLWESecretTensorFactory, GLWESecretToRef,
    },
};

#[doc(hidden)]
pub trait GLWETensorKeyCompressedEncryptSkDefault<BE: Backend> {
    fn glwe_tensor_key_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_tensor_key_compressed_encrypt_sk<R, S, E>(
        &self,
        res: &mut R,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWECompressedToMut + GGLWEInfos + GGLWECompressedSeedMut,
        E: EncryptionInfos,
        S: GLWESecretToRef + GetDistribution + GLWEInfos;
}

impl<BE: Backend> GLWETensorKeyCompressedEncryptSkDefault<BE> for Module<BE>
where
    Self: GGLWECompressedEncryptSk<BE> + GLWESecretPreparedFactory<BE> + GLWESecretTensorFactory<BE>,
    Scratch<BE>: ScratchTakeBasic + ScratchTakeCore<BE>,
{
    fn glwe_tensor_key_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(self.n() as u32, infos.n());

        let sk_prepared: usize = self.bytes_of_glwe_secret_prepared(infos.rank_out());
        let sk_tensor: usize = GLWESecretTensor::bytes_of_from_infos(infos);

        let tensor_infos: GGLWELayout = GGLWELayout {
            n: infos.n(),
            base2k: infos.base2k(),
            k: infos.max_k(),
            rank_in: GLWESecretTensor::pairs(infos.rank().into()).into(),
            rank_out: infos.rank_out(),
            dnum: infos.dnum(),
            dsize: infos.dsize(),
        };

        let lvl_0: usize = sk_prepared;
        let lvl_1: usize = sk_tensor;
        let lvl_2_encrypt: usize = self.gglwe_compressed_encrypt_sk_tmp_bytes(&tensor_infos);
        let lvl_2_prepare: usize = self.glwe_secret_tensor_prepare_tmp_bytes(infos.rank());
        let lvl_2: usize = lvl_2_encrypt.max(lvl_2_prepare);

        lvl_0 + lvl_1 + lvl_2
    }

    fn glwe_tensor_key_compressed_encrypt_sk<R, S, E>(
        &self,
        res: &mut R,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWEInfos + GGLWECompressedToMut + GGLWECompressedSeedMut,
        E: EncryptionInfos,
        S: GLWESecretToRef + GetDistribution + GLWEInfos,
    {
        assert_eq!(res.rank_out(), sk.rank());
        assert_eq!(res.n(), sk.n());
        assert!(
            scratch.available() >= self.glwe_tensor_key_compressed_encrypt_sk_tmp_bytes(res),
            "scratch.available(): {} < GLWETensorKeyCompressedEncryptSk::glwe_tensor_key_compressed_encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            self.glwe_tensor_key_compressed_encrypt_sk_tmp_bytes(res)
        );

        let (mut sk_prepared, scratch_1) = scratch.take_glwe_secret_prepared(self, res.rank());
        let (mut sk_tensor, scratch_2) = scratch_1.take_glwe_secret_tensor(self.n().into(), res.rank());
        self.prepare_glwe_secret(&mut sk_prepared, sk);
        self.glwe_secret_tensor_prepare(&mut sk_tensor, sk, scratch_2);

        self.gglwe_compressed_encrypt_sk(res, &sk_tensor.data, &sk_prepared, seed_xa, enc_infos, source_xe, scratch_2);
    }
}
