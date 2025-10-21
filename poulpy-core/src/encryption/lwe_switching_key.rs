use poulpy_hal::{
    api::{ModuleN, VecZnxAutomorphismInplace},
    layouts::{Backend, DataMut, Module, Scratch, ZnxView, ZnxViewMut},
    source::Source,
};

use crate::{
    ScratchTakeCore,
    encryption::glwe_switching_key::GLWESwitchingKeyEncryptSk,
    layouts::{
        GGLWEInfos, GGLWEToMut, GLWESecret, GLWESwitchingKey, GLWESwitchingKeyDegreesMut, LWEInfos, LWESecret, LWESecretToRef,
        LWESwitchingKey, Rank,
        prepared::{GLWESecretPrepared, GLWESecretPreparedFactory},
    },
};

impl LWESwitchingKey<Vec<u8>> {
    pub fn encrypt_sk_tmp_bytes<M, A, BE: Backend>(module: &M, infos: &A) -> usize
    where
        A: GGLWEInfos,
        M: LWESwitchingKeyEncrypt<BE>,
    {
        module.lwe_switching_key_encrypt_sk_tmp_bytes(infos)
    }
}

impl<D: DataMut> LWESwitchingKey<D> {
    pub fn encrypt_sk<S1, S2, M, BE: Backend>(
        &mut self,
        module: &M,
        sk_lwe_in: &S1,
        sk_lwe_out: &S2,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        S1: LWESecretToRef,
        S2: LWESecretToRef,
        M: LWESwitchingKeyEncrypt<BE>,
    {
        module.lwe_switching_key_encrypt_sk(self, sk_lwe_in, sk_lwe_out, source_xa, source_xe, scratch);
    }
}

pub trait LWESwitchingKeyEncrypt<BE: Backend> {
    fn lwe_switching_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn lwe_switching_key_encrypt_sk<R, S1, S2>(
        &self,
        res: &mut R,
        sk_lwe_in: &S1,
        sk_lwe_out: &S2,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWEToMut + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        S1: LWESecretToRef,
        S2: LWESecretToRef;
}

impl<BE: Backend> LWESwitchingKeyEncrypt<BE> for Module<BE>
where
    Self: ModuleN + GLWESwitchingKeyEncryptSk<BE> + GLWESecretPreparedFactory<BE> + VecZnxAutomorphismInplace<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn lwe_switching_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.dsize().0,
            1,
            "dsize > 1 is not supported for LWESwitchingKey"
        );
        assert_eq!(
            infos.rank_in().0,
            1,
            "rank_in > 1 is not supported for LWESwitchingKey"
        );
        assert_eq!(
            infos.rank_out().0,
            1,
            "rank_out > 1 is not supported for LWESwitchingKey"
        );
        GLWESecret::bytes_of(self.n().into(), Rank(1))
            + GLWESecretPrepared::bytes_of(self, Rank(1))
            + GLWESwitchingKey::encrypt_sk_tmp_bytes(self, infos)
    }

    #[allow(clippy::too_many_arguments)]
    fn lwe_switching_key_encrypt_sk<R, S1, S2>(
        &self,
        res: &mut R,
        sk_lwe_in: &S1,
        sk_lwe_out: &S2,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWEToMut + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        S1: LWESecretToRef,
        S2: LWESecretToRef,
    {
        let sk_lwe_in: &LWESecret<&[u8]> = &sk_lwe_in.to_ref();
        let sk_lwe_out: &LWESecret<&[u8]> = &sk_lwe_out.to_ref();

        assert!(sk_lwe_in.n().0 <= res.n().0);
        assert!(sk_lwe_out.n().0 <= res.n().0);
        assert!(res.n() <= self.n() as u32);

        let (mut sk_in_glwe, scratch_1) = scratch.take_glwe_secret(self.n().into(), Rank(1));
        let (mut sk_out_glwe, scratch_2) = scratch_1.take_glwe_secret(self.n().into(), Rank(1));

        sk_out_glwe.data.at_mut(0, 0)[..sk_lwe_out.n().into()].copy_from_slice(sk_lwe_out.data.at(0, 0));
        sk_out_glwe.data.at_mut(0, 0)[sk_lwe_out.n().into()..].fill(0);
        self.vec_znx_automorphism_inplace(-1, &mut sk_out_glwe.data.as_vec_znx_mut(), 0, scratch_2);

        sk_in_glwe.data.at_mut(0, 0)[..sk_lwe_in.n().into()].copy_from_slice(sk_lwe_in.data.at(0, 0));
        sk_in_glwe.data.at_mut(0, 0)[sk_lwe_in.n().into()..].fill(0);
        self.vec_znx_automorphism_inplace(-1, &mut sk_in_glwe.data.as_vec_znx_mut(), 0, scratch_2);

        self.glwe_switching_key_encrypt_sk(
            res,
            &sk_in_glwe,
            &sk_out_glwe,
            source_xa,
            source_xe,
            scratch_2,
        );
    }
}
