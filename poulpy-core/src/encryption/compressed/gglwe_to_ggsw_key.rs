use poulpy_hal::{
    api::{ModuleN, ScratchTakeBasic, VecZnxCopy},
    layouts::{Backend, DataMut, Module, Scratch},
    source::Source,
};

use crate::{
    GGLWECompressedEncryptSk, GetDistribution, ScratchTakeCore,
    layouts::{
        GGLWEInfos, GGLWEToGGSWKeyCompressed, GGLWEToGGSWKeyCompressedToMut, GLWEInfos, GLWESecret, GLWESecretTensor,
        GLWESecretTensorFactory, GLWESecretToRef,
        prepared::{GLWESecretPrepared, GLWESecretPreparedFactory},
    },
};

impl GGLWEToGGSWKeyCompressed<Vec<u8>> {
    pub fn encrypt_sk_tmp_bytes<M, A, BE: Backend>(module: &M, infos: &A) -> usize
    where
        A: GGLWEInfos,
        M: GGLWEToGGSWKeyCompressedEncryptSk<BE>,
    {
        module.gglwe_to_ggsw_key_encrypt_sk_tmp_bytes(infos)
    }
}

impl<DataSelf: DataMut> GGLWEToGGSWKeyCompressed<DataSelf> {
    pub fn encrypt_sk<M, S, BE: Backend>(
        &mut self,
        module: &M,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        M: GGLWEToGGSWKeyCompressedEncryptSk<BE>,
        S: GLWESecretToRef + GetDistribution + GLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.gglwe_to_ggsw_key_encrypt_sk(self, sk, seed_xa, source_xe, scratch);
    }
}

pub trait GGLWEToGGSWKeyCompressedEncryptSk<BE: Backend> {
    fn gglwe_to_ggsw_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn gglwe_to_ggsw_key_encrypt_sk<R, S>(
        &self,
        res: &mut R,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWEToGGSWKeyCompressedToMut + GGLWEInfos,
        S: GLWESecretToRef + GetDistribution + GLWEInfos;
}

impl<BE: Backend> GGLWEToGGSWKeyCompressedEncryptSk<BE> for Module<BE>
where
    Self: ModuleN + GGLWECompressedEncryptSk<BE> + GLWESecretTensorFactory<BE> + GLWESecretPreparedFactory<BE> + VecZnxCopy,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn gglwe_to_ggsw_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        let sk_prepared: usize = GLWESecretPrepared::bytes_of(self, infos.rank());
        let sk_tensor: usize = GLWESecretTensor::bytes_of_from_infos(infos);
        let gglwe_encrypt: usize = self.gglwe_compressed_encrypt_sk_tmp_bytes(infos);
        let sk_ij = GLWESecret::bytes_of(self.n().into(), infos.rank());
        (sk_prepared + sk_tensor + sk_ij) + gglwe_encrypt.max(self.glwe_secret_tensor_prepare_tmp_bytes(infos.rank()))
    }

    fn gglwe_to_ggsw_key_encrypt_sk<R, S>(
        &self,
        res: &mut R,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWEToGGSWKeyCompressedToMut + GGLWEInfos,
        S: GLWESecretToRef + GetDistribution + GLWEInfos,
    {
        assert_eq!(res.rank(), sk.rank());
        assert_eq!(res.n(), sk.n());

        let res: &mut GGLWEToGGSWKeyCompressed<&mut [u8]> = &mut res.to_mut();
        let rank: usize = res.rank_out().as_usize();

        let (mut sk_prepared, scratch_1) = scratch.take_glwe_secret_prepared(self, res.rank());
        let (mut sk_tensor, scratch_2) = scratch_1.take_glwe_secret_tensor(self.n().into(), res.rank());
        sk_prepared.prepare(self, sk);
        sk_tensor.prepare(self, sk, scratch_2);

        let (mut sk_ij, scratch_3) = scratch_2.take_scalar_znx(self.n(), rank);

        let mut source_xa = Source::new(seed_xa);

        for i in 0..rank {
            for j in 0..rank {
                self.vec_znx_copy(
                    &mut sk_ij.as_vec_znx_mut(),
                    j,
                    &sk_tensor.at(i, j).as_vec_znx(),
                    0,
                );
            }

            let (seed_xa_tmp, _) = source_xa.branch();

            res.at_mut(i).encrypt_sk(
                self,
                &sk_ij,
                &sk_prepared,
                seed_xa_tmp,
                source_xe,
                scratch_3,
            );
        }
    }
}
