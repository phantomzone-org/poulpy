use poulpy_hal::{
    api::{ModuleN, ScratchArenaTakeBasic, ScratchOwnedAlloc, VecZnxCopyBackend},
    layouts::{
        Backend, Module, ScratchArena, ScratchOwned, scalar_znx_as_vec_znx_backend_mut_from_mut,
        scalar_znx_as_vec_znx_backend_ref_from_mut,
    },
    source::Source,
};

use crate::{
    EncryptionInfos, GGLWECompressedEncryptSk, GetDistribution, ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GGLWEToGGSWKeyCompressedToBackendMut, GLWEInfos, GLWESecret, GLWESecretTensor, GLWESecretTensorFactory,
        GLWESecretToBackendRef, gglwe_to_ggsw_key_compressed_at_backend_mut_from_mut, prepared::GLWESecretPreparedFactory,
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
        R: GGLWEToGGSWKeyCompressedToBackendMut<BE> + GGLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos;
}

impl<BE: Backend> GGLWEToGGSWKeyCompressedEncryptSkDefault<BE> for Module<BE>
where
    Self: ModuleN
        + GGLWECompressedEncryptSk<BE>
        + GLWESecretTensorFactory<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxCopyBackend<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
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
        let lvl_3_prepare: usize = self.glwe_secret_tensor_prepare_tmp_bytes(infos.rank());
        let lvl_3: usize = lvl_3_prepare;

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
        R: GGLWEToGGSWKeyCompressedToBackendMut<BE> + GGLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos,
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

        let mut res = res.to_backend_mut();
        let rank: usize = res.rank_out().as_usize();

        let scratch = scratch.borrow();
        let (mut sk_prepared, scratch_1) = scratch.take_glwe_secret_prepared(self, res.rank());
        let (mut sk_tensor, scratch_2) = scratch_1.take_glwe_secret_tensor(self.n().into(), res.rank());
        let (mut sk_ij, mut tensor_scratch) = scratch_2.take_scalar_znx(self.n(), rank);
        let mut sk_prepared_ref = &mut sk_prepared;
        let mut sk_tensor_ref = &mut sk_tensor;
        self.glwe_secret_prepare(&mut sk_prepared_ref, sk);
        self.glwe_secret_tensor_prepare(&mut sk_tensor_ref, sk, &mut tensor_scratch);

        let mut enc_scratch: ScratchOwned<BE> = ScratchOwned::alloc(self.gglwe_compressed_encrypt_sk_tmp_bytes(&res));
        let sk_tensor_backend = scalar_znx_as_vec_znx_backend_ref_from_mut::<BE>(&sk_tensor.data);

        let mut source_xa = Source::new(seed_xa);

        for i in 0..rank {
            {
                let mut sk_ij_backend = scalar_znx_as_vec_znx_backend_mut_from_mut::<BE>(&mut sk_ij);
                for j in 0..rank {
                    let (lo, hi) = if i <= j { (i, j) } else { (j, i) };
                    let idx: usize = lo * rank + hi - (lo * (lo + 1) / 2);
                    self.vec_znx_copy_backend(&mut sk_ij_backend, j, &sk_tensor_backend, idx);
                }
            }

            let (seed_xa_tmp, _) = source_xa.branch();

            let mut ct = gglwe_to_ggsw_key_compressed_at_backend_mut_from_mut::<BE>(&mut res, i);
            let mut ct_ref = &mut ct;
            let sk_ij_ref = &mut sk_ij;

            self.gglwe_compressed_encrypt_sk(
                &mut ct_ref,
                &sk_ij_ref,
                &sk_prepared_ref,
                seed_xa_tmp,
                enc_infos,
                source_xe,
                &mut enc_scratch.arena(),
            );
        }
    }
}
