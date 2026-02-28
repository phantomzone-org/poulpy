use poulpy_hal::{
    api::{ScratchAvailable, ScratchTakeBasic},
    layouts::{Backend, DataMut, Module, Scratch},
    source::Source,
};

use crate::{
    GGLWECompressedEncryptSk, GetDistribution, ScratchTakeCore,
    layouts::{
        GGLWECompressedSeedMut, GGLWECompressedToMut, GGLWEInfos, GGLWELayout, GLWEInfos, GLWESecretPrepared,
        GLWESecretPreparedFactory, GLWESecretTensor, GLWESecretTensorFactory, GLWESecretToRef,
        compressed::GLWETensorKeyCompressed,
    },
};

impl GLWETensorKeyCompressed<Vec<u8>> {
    pub fn encrypt_sk_tmp_bytes<M, A, BE: Backend>(module: &M, infos: &A) -> usize
    where
        A: GGLWEInfos,
        M: GLWETensorKeyCompressedEncryptSk<BE>,
    {
        module.glwe_tensor_key_compressed_encrypt_sk_tmp_bytes(infos)
    }
}

impl<DataSelf: DataMut> GLWETensorKeyCompressed<DataSelf> {
    pub fn encrypt_sk<S, M, BE: Backend>(
        &mut self,
        module: &M,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        S: GLWESecretToRef + GetDistribution + GLWEInfos,
        M: GLWETensorKeyCompressedEncryptSk<BE>,
    {
        module.glwe_tensor_key_compressed_encrypt_sk(self, sk, seed_xa, source_xe, scratch);
    }
}

pub trait GLWETensorKeyCompressedEncryptSk<BE: Backend> {
    fn glwe_tensor_key_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_tensor_key_compressed_encrypt_sk<R, S>(
        &self,
        res: &mut R,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWECompressedToMut + GGLWEInfos + GGLWECompressedSeedMut,
        S: GLWESecretToRef + GetDistribution + GLWEInfos;
}

impl<BE: Backend> GLWETensorKeyCompressedEncryptSk<BE> for Module<BE>
where
    Self: GGLWECompressedEncryptSk<BE> + GLWESecretPreparedFactory<BE> + GLWESecretTensorFactory<BE>,
    Scratch<BE>: ScratchTakeBasic + ScratchTakeCore<BE>,
{
    fn glwe_tensor_key_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(self.n() as u32, infos.n());

        let sk_prepared: usize = GLWESecretPrepared::bytes_of(self, infos.rank_out());
        let sk_tensor: usize = GLWESecretTensor::bytes_of_from_infos(infos);

        let tensor_infos: GGLWELayout = GGLWELayout {
            n: infos.n(),
            base2k: infos.base2k(),
            k: infos.k(),
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

    fn glwe_tensor_key_compressed_encrypt_sk<R, S>(
        &self,
        res: &mut R,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWEInfos + GGLWECompressedToMut + GGLWECompressedSeedMut,
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
        sk_prepared.prepare(self, sk);
        sk_tensor.prepare(self, sk, scratch_2);

        self.gglwe_compressed_encrypt_sk(res, &sk_tensor.data, &sk_prepared, seed_xa, source_xe, scratch_2);
    }
}
