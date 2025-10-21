use poulpy_hal::{
    api::{ModuleN, VecZnxAutomorphismInplace, VecZnxAutomorphismInplaceTmpBytes},
    layouts::{Backend, DataMut, Module, Scratch, ZnxView, ZnxViewMut, ZnxZero},
    source::Source,
};

use crate::{
    GGLWEEncryptSk, ScratchTakeCore,
    layouts::{
        GGLWE, GGLWEInfos, GGLWEToMut, GLWESecret, GLWESecretToRef, GLWEToLWESwitchingKey, LWEInfos, LWESecret, LWESecretToRef,
        Rank,
        prepared::{GLWESecretPrepared, GLWESecretPreparedFactory},
    },
};

impl GLWEToLWESwitchingKey<Vec<u8>> {
    pub fn encrypt_sk_tmp_bytes<M, A, BE: Backend>(module: &M, infos: &A) -> usize
    where
        A: GGLWEInfos,
        M: GLWEToLWESwitchingKeyEncryptSk<BE>,
    {
        module.glwe_to_lwe_switching_key_encrypt_sk_tmp_bytes(infos)
    }
}

impl<D: DataMut> GLWEToLWESwitchingKey<D> {
    pub fn encrypt_sk<M, S1, S2, BE: Backend>(
        &mut self,
        module: &M,
        sk_lwe: &S1,
        sk_glwe: &S2,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        M: GLWEToLWESwitchingKeyEncryptSk<BE>,
        S1: LWESecretToRef,
        S2: GLWESecretToRef,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.glwe_to_lwe_switching_key_encrypt_sk(self, sk_lwe, sk_glwe, source_xa, source_xe, scratch);
    }
}

pub trait GLWEToLWESwitchingKeyEncryptSk<BE: Backend> {
    fn glwe_to_lwe_switching_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_to_lwe_switching_key_encrypt_sk<R, S1, S2>(
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
        R: GGLWEToMut;
}

impl<BE: Backend> GLWEToLWESwitchingKeyEncryptSk<BE> for Module<BE>
where
    Self: ModuleN
        + GGLWEEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxAutomorphismInplace<BE>
        + VecZnxAutomorphismInplaceTmpBytes,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn glwe_to_lwe_switching_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        GLWESecretPrepared::bytes_of(self, infos.rank_in())
            + GGLWE::encrypt_sk_tmp_bytes(self, infos)
                .max(GLWESecret::bytes_of(self.n().into(), infos.rank_in()) + self.vec_znx_automorphism_inplace_tmp_bytes())
    }

    fn glwe_to_lwe_switching_key_encrypt_sk<R, S1, S2>(
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
        R: GGLWEToMut,
    {
        let sk_lwe: &LWESecret<&[u8]> = &sk_lwe.to_ref();
        let sk_glwe: &GLWESecret<&[u8]> = &sk_glwe.to_ref();

        assert!(sk_lwe.n().0 <= self.n() as u32);

        let (mut sk_lwe_as_glwe_prep, scratch_1) = scratch.take_glwe_secret_prepared(self, Rank(1));

        {
            let (mut sk_lwe_as_glwe, scratch_2) = scratch_1.take_glwe_secret(self, Rank(1));
            sk_lwe_as_glwe.data.zero();
            sk_lwe_as_glwe.data.at_mut(0, 0)[..sk_lwe.n().into()].copy_from_slice(sk_lwe.data.at(0, 0));
            self.vec_znx_automorphism_inplace(-1, &mut sk_lwe_as_glwe.data.as_vec_znx_mut(), 0, scratch_2);
            sk_lwe_as_glwe_prep.prepare(self, &sk_lwe_as_glwe);
        }

        self.gglwe_encrypt_sk(
            res,
            &sk_glwe.data,
            &sk_lwe_as_glwe_prep,
            source_xa,
            source_xe,
            scratch_1,
        );
    }
}
