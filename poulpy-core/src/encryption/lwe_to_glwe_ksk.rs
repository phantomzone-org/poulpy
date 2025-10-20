use poulpy_hal::{
    api::{ModuleN, VecZnxAutomorphismInplace},
    layouts::{Backend, DataMut, Module, Scratch, ZnxView, ZnxViewMut},
    source::Source,
};

use crate::{
    ScratchTakeCore,
    encryption::gglwe_ksk::GLWESwitchingKeyEncryptSk,
    layouts::{
        GGLWEInfos, GLWESecret, GLWESecretToRef, GLWESwitchingKey, LWEInfos, LWESecret, LWESecretToRef, LWEToGLWESwitchingKey,
        LWEToGLWESwitchingKeyToMut, Rank,
    },
};

impl LWEToGLWESwitchingKey<Vec<u8>> {
    pub fn encrypt_sk_tmp_bytes<M, A, BE: Backend>(module: &M, infos: &A) -> usize
    where
        A: GGLWEInfos,
        M: LWEToGLWESwitchingKeyEncrypt<BE>,
    {
        module.lwe_to_glwe_switching_key_encrypt_sk_tmp_bytes(infos)
    }
}

impl<D: DataMut> LWEToGLWESwitchingKey<D> {
    pub fn encrypt_sk<S1, S2, M, BE: Backend>(
        &mut self,
        module: &M,
        sk_lwe: &S1,
        sk_glwe: &S2,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        S1: LWESecretToRef,
        S2: GLWESecretToRef,
        M: LWEToGLWESwitchingKeyEncrypt<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.lwe_to_glwe_switching_key_encrypt_sk(self, sk_lwe, sk_glwe, source_xa, source_xe, scratch);
    }
}

pub trait LWEToGLWESwitchingKeyEncrypt<BE: Backend> {
    fn lwe_to_glwe_switching_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn lwe_to_glwe_switching_key_encrypt_sk<R, S1, S2>(
        &self,
        res: &mut R,
        sk_lwe: &S1,
        sk_glwe: &S2,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        S1: LWESecretToRef,
        S2: GLWESecretToRef,
        R: LWEToGLWESwitchingKeyToMut;
}

impl<BE: Backend> LWEToGLWESwitchingKeyEncrypt<BE> for Module<BE>
where
    Self: ModuleN + GLWESwitchingKeyEncryptSk<BE> + VecZnxAutomorphismInplace<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn lwe_to_glwe_switching_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        debug_assert_eq!(
            infos.rank_in(),
            Rank(1),
            "rank_in != 1 is not supported for LWEToGLWESwitchingKey"
        );
        GLWESwitchingKey::encrypt_sk_tmp_bytes(self, infos) + GLWESecret::bytes_of(self.n().into(), infos.rank_in())
    }

    fn lwe_to_glwe_switching_key_encrypt_sk<R, S1, S2>(
        &self,
        res: &mut R,
        sk_lwe: &S1,
        sk_glwe: &S2,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        S1: LWESecretToRef,
        S2: GLWESecretToRef,
        R: LWEToGLWESwitchingKeyToMut,
    {
        let res: &mut LWEToGLWESwitchingKey<&mut [u8]> = &mut res.to_mut();
        let sk_lwe: &LWESecret<&[u8]> = &sk_lwe.to_ref();
        let sk_glwe: &GLWESecret<&[u8]> = &sk_glwe.to_ref();

        assert!(sk_lwe.n().0 <= self.n() as u32);

        let (mut sk_lwe_as_glwe, scratch_1) = scratch.take_glwe_secret(self, Rank(1));
        sk_lwe_as_glwe.data.at_mut(0, 0)[..sk_lwe.n().into()].copy_from_slice(sk_lwe.data.at(0, 0));
        sk_lwe_as_glwe.data.at_mut(0, 0)[sk_lwe.n().into()..].fill(0);
        self.vec_znx_automorphism_inplace(-1, &mut sk_lwe_as_glwe.data.as_vec_znx_mut(), 0, scratch_1);

        res.0.encrypt_sk(
            self,
            &sk_lwe_as_glwe,
            sk_glwe,
            source_xa,
            source_xe,
            scratch_1,
        );
    }
}
