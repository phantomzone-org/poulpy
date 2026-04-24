use poulpy_hal::{
    api::{ModuleN, ScratchOwnedAlloc, SvpPrepare, VecZnxCopy},
    layouts::{Backend, HostDataMut, Module, ScalarZnx, ScratchArena, ScratchOwned, SvpPPolToBackendMut},
    source::Source,
};

use crate::{
    EncryptionInfos, GGLWECompressedEncryptSk, GetDistribution, ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GGLWEToGGSWKeyCompressed, GGLWEToGGSWKeyCompressedToMut, GLWEInfos, GLWESecret, GLWESecretTensor,
        GLWESecretTensorFactory, GLWESecretToRef, prepared::GLWESecretPreparedFactory,
    },
};

#[doc(hidden)]
pub trait GGLWEToGGSWKeyCompressedEncryptSkDefault<BE: Backend> {
    fn gglwe_to_ggsw_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn gglwe_to_ggsw_key_encrypt_sk<R, S, E>(
        &self,
        res: &mut R,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToGGSWKeyCompressedToMut + GGLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretToRef + GetDistribution + GLWEInfos;
}

impl<BE: Backend> GGLWEToGGSWKeyCompressedEncryptSkDefault<BE> for Module<BE>
where
    Self: ModuleN + GGLWECompressedEncryptSk<BE> + GLWESecretTensorFactory<BE> + GLWESecretPreparedFactory<BE> + VecZnxCopy,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
    for<'s> BE::BufMut<'s>: HostDataMut,
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
        let lvl_3_encrypt: usize = self.gglwe_compressed_encrypt_sk_tmp_bytes(infos);
        let lvl_3_prepare: usize = self.glwe_secret_tensor_prepare_tmp_bytes(infos.rank());
        let lvl_3: usize = lvl_3_encrypt.max(lvl_3_prepare);

        lvl_0 + lvl_1 + lvl_2 + lvl_3
    }

    fn gglwe_to_ggsw_key_encrypt_sk<R, S, E>(
        &self,
        res: &mut R,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToGGSWKeyCompressedToMut + GGLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretToRef + GetDistribution + GLWEInfos,
    {
        assert_eq!(res.rank(), sk.rank());
        assert_eq!(res.n(), sk.n());
        assert!(
            scratch.available()
                >= <Module<BE> as GGLWEToGGSWKeyCompressedEncryptSkDefault<BE>>::gglwe_to_ggsw_key_encrypt_sk_tmp_bytes(
                    self, res
                ),
            "scratch.available(): {} < GGLWEToGGSWKeyCompressedEncryptSk::gglwe_to_ggsw_key_encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            <Module<BE> as GGLWEToGGSWKeyCompressedEncryptSkDefault<BE>>::gglwe_to_ggsw_key_encrypt_sk_tmp_bytes(self, res)
        );

        let res: &mut GGLWEToGGSWKeyCompressed<&mut [u8]> = &mut res.to_mut();
        let rank: usize = res.rank_out().as_usize();

        let mut sk_prepared = self.glwe_secret_prepared_alloc(res.rank());
        let mut sk_tensor = GLWESecretTensor::alloc(self.n().into(), res.rank());
        {
            let sk_ref = sk.to_ref();
            let mut sk_prepared_data = sk_prepared.data.to_backend_mut();
            for i in 0..sk_ref.rank().into() {
                self.svp_prepare(&mut sk_prepared_data, i, &sk_ref.data, i);
            }
            sk_prepared.dist = *sk.dist();
        }
        self.glwe_secret_tensor_prepare(&mut sk_tensor, sk, scratch);

        let mut sk_ij = ScalarZnx::alloc(self.n(), rank);
        let mut enc_scratch: ScratchOwned<BE> = ScratchOwned::alloc(self.gglwe_compressed_encrypt_sk_tmp_bytes(res));

        let mut source_xa = Source::new(seed_xa);

        for i in 0..rank {
            for j in 0..rank {
                self.vec_znx_copy(&mut sk_ij.as_vec_znx_mut(), j, &sk_tensor.at(i, j).as_vec_znx(), 0);
            }

            let (seed_xa_tmp, _) = source_xa.branch();

            self.gglwe_compressed_encrypt_sk(
                res.at_mut(i),
                &sk_ij,
                &sk_prepared,
                seed_xa_tmp,
                enc_infos,
                source_xe,
                &mut enc_scratch.arena(),
            );
        }
    }
}
