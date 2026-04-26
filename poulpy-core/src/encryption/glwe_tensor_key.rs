use poulpy_hal::{
    api::{ModuleN, ScratchOwnedAlloc},
    layouts::{Backend, Module, ScratchArena, ScratchOwned},
    source::Source,
};

use crate::{
    EncryptionInfos, GGLWEEncryptSk, GetDistribution, ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GGLWELayout, GGLWEToBackendMut, GLWEInfos, GLWESecretTensor, GLWESecretTensorFactory, GLWESecretToBackendRef,
        prepared::GLWESecretPreparedFactory,
    },
};

#[doc(hidden)]
pub trait GLWETensorKeyEncryptSkDefault<BE: Backend> {
    fn glwe_tensor_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_tensor_key_encrypt_sk<R, S, E>(
        &self,
        res: &mut R,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos;
}

impl<BE: Backend> GLWETensorKeyEncryptSkDefault<BE> for Module<BE>
where
    Self: ModuleN + GGLWEEncryptSk<BE> + GLWESecretPreparedFactory<BE> + GLWESecretTensorFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
    fn glwe_tensor_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(self.n() as u32, infos.n());

        let sk_prepared: usize = self.glwe_secret_prepared_bytes_of(infos.rank_out());
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
        let lvl_2_encrypt: usize = self.gglwe_encrypt_sk_tmp_bytes(&tensor_infos);
        let lvl_2_prepare: usize = self.glwe_secret_tensor_prepare_tmp_bytes(infos.rank());
        let lvl_2: usize = lvl_2_encrypt.max(lvl_2_prepare);

        lvl_0 + lvl_1 + lvl_2
    }

    fn glwe_tensor_key_encrypt_sk<R, S, E>(
        &self,
        res: &mut R,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos,
    {
        assert_eq!(res.rank_out(), sk.rank());
        assert_eq!(res.n(), sk.n());
        assert!(
            scratch.available()
                >= <Module<BE> as GLWETensorKeyEncryptSkDefault<BE>>::glwe_tensor_key_encrypt_sk_tmp_bytes(self, res),
            "scratch.available(): {} < GLWETensorKeyEncryptSk::glwe_tensor_key_encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            <Module<BE> as GLWETensorKeyEncryptSkDefault<BE>>::glwe_tensor_key_encrypt_sk_tmp_bytes(self, res)
        );

        let scratch = scratch.borrow();
        let (mut sk_prepared, scratch_1) = scratch.take_glwe_secret_prepared(self, res.rank());
        let (mut sk_tensor, mut tensor_scratch) = scratch_1.take_glwe_secret_tensor(self.n().into(), res.rank());
        let mut sk_prepared_ref = &mut sk_prepared;
        let mut sk_tensor_ref = &mut sk_tensor;
        self.glwe_secret_prepare(&mut sk_prepared_ref, sk);
        self.glwe_secret_tensor_prepare(&mut sk_tensor_ref, sk, &mut tensor_scratch);

        let mut enc_scratch: ScratchOwned<BE> = ScratchOwned::alloc(self.gglwe_encrypt_sk_tmp_bytes(res));
        let sk_tensor_data = &mut sk_tensor.data;
        self.gglwe_encrypt_sk(
            res,
            &sk_tensor_data,
            &sk_prepared_ref,
            enc_infos,
            source_xe,
            source_xa,
            &mut enc_scratch.arena(),
        );
    }
}
