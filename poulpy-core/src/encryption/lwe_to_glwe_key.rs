use poulpy_hal::{
    api::{ModuleN, VecZnxAutomorphismInplace, VecZnxAutomorphismInplaceTmpBytes},
    layouts::{Backend, DataMut, Module, Scratch, ZnxView, ZnxViewMut},
    source::Source,
};

use crate::{
    GGLWEEncryptSk, ScratchTakeCore,
    layouts::{
        GGLWE, GGLWEInfos, GGLWEToMut, GLWESecret, GLWESecretPreparedFactory, GLWESecretPreparedToRef, LWEInfos, LWESecret,
        LWESecretToRef, LWEToGLWEKey, Rank,
    },
};

impl LWEToGLWEKey<Vec<u8>> {
    pub fn encrypt_sk_tmp_bytes<M, A, BE: Backend>(module: &M, infos: &A) -> usize
    where
        A: GGLWEInfos,
        M: LWEToGLWESwitchingKeyEncryptSk<BE>,
    {
        module.lwe_to_glwe_key_encrypt_sk_tmp_bytes(infos)
    }
}

impl<D: DataMut> LWEToGLWEKey<D> {
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
        S2: GLWESecretPreparedToRef<BE>,
        M: LWEToGLWESwitchingKeyEncryptSk<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.lwe_to_glwe_key_encrypt_sk(self, sk_lwe, sk_glwe, source_xa, source_xe, scratch);
    }
}

pub trait LWEToGLWESwitchingKeyEncryptSk<BE: Backend> {
    fn lwe_to_glwe_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn lwe_to_glwe_key_encrypt_sk<R, S1, S2>(
        &self,
        res: &mut R,
        sk_lwe: &S1,
        sk_glwe: &S2,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        S1: LWESecretToRef,
        S2: GLWESecretPreparedToRef<BE>,
        R: GGLWEToMut;
}

impl<BE: Backend> LWEToGLWESwitchingKeyEncryptSk<BE> for Module<BE>
where
    Self: ModuleN
        + GGLWEEncryptSk<BE>
        + VecZnxAutomorphismInplace<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxAutomorphismInplaceTmpBytes,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn lwe_to_glwe_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        debug_assert_eq!(
            infos.rank_in(),
            Rank(1),
            "rank_in != 1 is not supported for LWEToGLWEKeyPrepared"
        );
        GLWESecret::bytes_of(self.n().into(), infos.rank_in())
            + GGLWE::encrypt_sk_tmp_bytes(self, infos).max(self.vec_znx_automorphism_inplace_tmp_bytes())
    }

    fn lwe_to_glwe_key_encrypt_sk<R, S1, S2>(
        &self,
        res: &mut R,
        sk_lwe: &S1,
        sk_glwe: &S2,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        S1: LWESecretToRef,
        S2: GLWESecretPreparedToRef<BE>,
        R: GGLWEToMut,
    {
        let sk_lwe: &LWESecret<&[u8]> = &sk_lwe.to_ref();

        assert!(sk_lwe.n().0 <= self.n() as u32);

        let (mut sk_lwe_as_glwe, scratch_1) = scratch.take_glwe_secret(self.n().into(), Rank(1));
        sk_lwe_as_glwe.dist = sk_lwe.dist;

        sk_lwe_as_glwe.data.at_mut(0, 0)[..sk_lwe.n().into()].copy_from_slice(sk_lwe.data.at(0, 0));
        sk_lwe_as_glwe.data.at_mut(0, 0)[sk_lwe.n().into()..].fill(0);
        self.vec_znx_automorphism_inplace(-1, &mut sk_lwe_as_glwe.data.as_vec_znx_mut(), 0, scratch_1);

        self.gglwe_encrypt_sk(res, &sk_lwe_as_glwe.data, sk_glwe, source_xa, source_xe, scratch_1);
    }
}
