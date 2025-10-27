use poulpy_hal::{
    api::ModuleN,
    layouts::{Backend, DataMut, Module, Scratch},
    source::Source,
};

use crate::{
    GGLWEEncryptSk, GetDistribution, ScratchTakeCore,
    layouts::{
        GGLWEInfos, GGLWELayout, GGLWEToMut, GLWEInfos, GLWESecretTensor, GLWESecretTensorFactory, GLWESecretToRef,
        GLWETensorKey,
        prepared::{GLWESecretPrepared, GLWESecretPreparedFactory},
    },
};

impl GLWETensorKey<Vec<u8>> {
    pub fn encrypt_sk_tmp_bytes<M, A, BE: Backend>(module: &M, infos: &A) -> usize
    where
        A: GGLWEInfos,
        M: GLWETensorKeyEncryptSk<BE>,
    {
        module.glwe_tensor_key_encrypt_sk_tmp_bytes(infos)
    }
}

impl<DataSelf: DataMut> GLWETensorKey<DataSelf> {
    pub fn encrypt_sk<M, S, BE: Backend>(
        &mut self,
        module: &M,
        sk: &S,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        M: GLWETensorKeyEncryptSk<BE>,
        S: GLWESecretToRef + GetDistribution + GLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.glwe_tensor_key_encrypt_sk(self, sk, source_xa, source_xe, scratch);
    }
}

pub trait GLWETensorKeyEncryptSk<BE: Backend> {
    fn glwe_tensor_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_tensor_key_encrypt_sk<R, S>(
        &self,
        res: &mut R,
        sk: &S,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWEToMut + GGLWEInfos,
        S: GLWESecretToRef + GetDistribution + GLWEInfos;
}

impl<BE: Backend> GLWETensorKeyEncryptSk<BE> for Module<BE>
where
    Self: ModuleN + GGLWEEncryptSk<BE> + GLWESecretPreparedFactory<BE> + GLWESecretTensorFactory<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn glwe_tensor_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
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

        let gglwe_encrypt: usize = self.gglwe_encrypt_sk_tmp_bytes(&tensor_infos);

        (sk_prepared + sk_tensor) + gglwe_encrypt.max(self.glwe_secret_tensor_prepare_tmp_bytes(infos.rank()))
    }

    fn glwe_tensor_key_encrypt_sk<R, S>(
        &self,
        res: &mut R,
        sk: &S,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWEToMut + GGLWEInfos,
        S: GLWESecretToRef + GetDistribution + GLWEInfos,
    {
        assert_eq!(res.rank_out(), sk.rank());
        assert_eq!(res.n(), sk.n());

        let (mut sk_prepared, scratch_1) = scratch.take_glwe_secret_prepared(self, res.rank());
        let (mut sk_tensor, scratch_2) = scratch_1.take_glwe_secret_tensor(self.n().into(), res.rank());
        sk_prepared.prepare(self, sk);
        sk_tensor.prepare(self, sk, scratch_2);

        self.gglwe_encrypt_sk(
            res,
            &sk_tensor.data,
            &sk_prepared,
            source_xa,
            source_xe,
            scratch_2,
        );
    }
}
