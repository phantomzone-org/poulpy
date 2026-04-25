use poulpy_hal::{
    api::{ModuleN, ScratchOwnedAlloc, SvpPrepare},
    layouts::{Backend, HostDataMut, Module, ScalarZnx, ScalarZnxToBackendRef, ScratchArena, ScratchOwned, SvpPPolToBackendMut},
    source::Source,
};

use crate::{
    EncryptionInfos, GGLWEEncryptSk, GetDistribution, ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GGLWEToGGSWKeyToBackendMut, GLWEInfos, GLWESecret, GLWESecretTensor, GLWESecretTensorFactory,
        GLWESecretToRef, gglwe_to_ggsw_key_at_backend_mut_from_mut, prepared::GLWESecretPreparedFactory,
    },
    vec_znx_host_ops::vec_znx_copy,
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
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToGGSWKeyToBackendMut<BE>,
        E: EncryptionInfos,
        S: GLWESecretToRef + GetDistribution + GLWEInfos;
}

impl<BE: Backend> GGLWEToGGSWKeyEncryptSkDefault<BE> for Module<BE>
where
    Self: ModuleN + GGLWEEncryptSk<BE> + GLWESecretTensorFactory<BE> + GLWESecretPreparedFactory<BE>,
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

        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToGGSWKeyToBackendMut<BE>,
        E: EncryptionInfos,
        S: GLWESecretToRef + GetDistribution + GLWEInfos,
    {
        let mut res = res.to_backend_mut();

        let rank: usize = res.rank_out().as_usize();
        assert!(
            scratch.available()
                >= <Module<BE> as GGLWEToGGSWKeyEncryptSkDefault<BE>>::gglwe_to_ggsw_key_encrypt_sk_tmp_bytes(self, &res),
            "scratch.available(): {} < GGLWEToGGSWKeyEncryptSk::gglwe_to_ggsw_key_encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            <Module<BE> as GGLWEToGGSWKeyEncryptSkDefault<BE>>::gglwe_to_ggsw_key_encrypt_sk_tmp_bytes(self, &res)
        );

        let mut sk_prepared = self.glwe_secret_prepared_alloc(res.rank());
        let mut sk_tensor = GLWESecretTensor::alloc(self.n().into(), res.rank());
        {
            let sk_ref = sk.to_ref();
            let sk_backend = ScalarZnx::from_data(BE::from_host_bytes(sk_ref.data.data), sk_ref.data.n, sk_ref.data.cols);
            let sk_backend_ref = <ScalarZnx<BE::OwnedBuf> as ScalarZnxToBackendRef<BE>>::to_backend_ref(&sk_backend);
            let mut sk_prepared_data = sk_prepared.data.to_backend_mut();
            for i in 0..sk_ref.rank().into() {
                self.svp_prepare(&mut sk_prepared_data, i, &sk_backend_ref, i);
            }
            sk_prepared.dist = *sk.dist();
        }
        self.glwe_secret_tensor_prepare(&mut sk_tensor, sk, scratch);

        let mut sk_ij = ScalarZnx::alloc(self.n(), rank);
        let mut enc_scratch: ScratchOwned<BE> = ScratchOwned::alloc(self.gglwe_encrypt_sk_tmp_bytes(&res));

        for i in 0..rank {
            for j in 0..rank {
                vec_znx_copy(&mut sk_ij.as_vec_znx_mut(), j, &sk_tensor.at(i, j).as_vec_znx(), 0);
            }
            let sk_ij_ref = sk_ij.to_ref();
            let sk_ij_backend = ScalarZnx::from_data(BE::from_host_bytes(sk_ij_ref.data), sk_ij_ref.n, sk_ij_ref.cols);

            let mut ct = gglwe_to_ggsw_key_at_backend_mut_from_mut::<BE>(&mut res, i);
            let mut ct_ref = &mut ct;

            self.gglwe_encrypt_sk(
                &mut ct_ref,
                &sk_ij_backend,
                &sk_prepared,
                enc_infos,
                source_xe,
                source_xa,
                &mut enc_scratch.arena(),
            );
        }
    }
}
