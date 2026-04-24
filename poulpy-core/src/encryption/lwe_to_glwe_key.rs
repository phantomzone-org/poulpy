use poulpy_hal::{
    api::{ModuleN, ScratchAvailable, ScratchOwnedAlloc, VecZnxAutomorphism},
    layouts::{Backend, HostDataMut, Module, Scratch, ScratchArena, ScratchOwned, ZnxView, ZnxViewMut},
    source::Source,
};

use crate::{
    EncryptionInfos, GGLWEEncryptSk, ScratchArenaTakeCore, ScratchTakeCore,
    layouts::{
        GGLWEInfos, GGLWEToMut, GLWESecret, GLWESecretPreparedFactory, GLWESecretPreparedToBackendRef, LWEInfos, LWESecret,
        LWESecretToRef, Rank,
    },
};

#[doc(hidden)]
pub trait LWEToGLWESwitchingKeyEncryptSkDefault<BE: Backend> {
    fn lwe_to_glwe_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn lwe_to_glwe_key_encrypt_sk<R, S1, S2, E>(
        &self,
        res: &mut R,
        sk_lwe: &S1,
        sk_glwe: &S2,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        S1: LWESecretToRef,
        S2: GLWESecretPreparedToBackendRef<BE>,
        E: EncryptionInfos,
        R: GGLWEToMut + GGLWEInfos;
}

impl<BE: Backend> LWEToGLWESwitchingKeyEncryptSkDefault<BE> for Module<BE>
where
    Self: ModuleN + GGLWEEncryptSk<BE> + GLWESecretPreparedFactory<BE> + VecZnxAutomorphism,
    Scratch<BE>: ScratchTakeCore<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
    for<'s> BE::BufMut<'s>: HostDataMut,
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
        assert_eq!(self.n() as u32, infos.n());

        let lvl_0: usize = GLWESecret::bytes_of(self.n().into(), Rank(1));
        let lvl_1: usize = GLWESecret::bytes_of(self.n().into(), Rank(1));

        lvl_0 + lvl_1
    }

    #[allow(clippy::too_many_arguments)]
    fn lwe_to_glwe_key_encrypt_sk<R, S1, S2, E>(
        &self,
        res: &mut R,
        sk_lwe: &S1,
        sk_glwe: &S2,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        S1: LWESecretToRef,
        S2: GLWESecretPreparedToBackendRef<BE>,
        E: EncryptionInfos,
        R: GGLWEToMut + GGLWEInfos,
    {
        let sk_lwe: &LWESecret<&[u8]> = &sk_lwe.to_ref();

        assert!(sk_lwe.n().0 <= self.n() as u32);
        assert!(
            scratch.available()
                >= <Module<BE> as LWEToGLWESwitchingKeyEncryptSkDefault<BE>>::lwe_to_glwe_key_encrypt_sk_tmp_bytes(self, res),
            "scratch.available(): {} < LWEToGLWESwitchingKeyEncryptSk::lwe_to_glwe_key_encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            <Module<BE> as LWEToGLWESwitchingKeyEncryptSkDefault<BE>>::lwe_to_glwe_key_encrypt_sk_tmp_bytes(self, res)
        );

        let (mut sk_lwe_as_glwe_src, scratch_1) = scratch.take_glwe_secret(self.n().into(), Rank(1));
        sk_lwe_as_glwe_src.dist = sk_lwe.dist;
        sk_lwe_as_glwe_src.data.at_mut(0, 0)[..sk_lwe.n().into()].copy_from_slice(sk_lwe.data.at(0, 0));
        sk_lwe_as_glwe_src.data.at_mut(0, 0)[sk_lwe.n().into()..].fill(0);

        let (mut sk_lwe_as_glwe, _) = scratch_1.take_glwe_secret(self.n().into(), Rank(1));
        sk_lwe_as_glwe.dist = sk_lwe.dist;
        {
            let sk_lwe_as_glwe_src_data = sk_lwe_as_glwe_src.data.as_vec_znx();
            self.vec_znx_automorphism(-1, &mut sk_lwe_as_glwe.data.as_vec_znx_mut(), 0, &sk_lwe_as_glwe_src_data, 0);
        }

        let mut enc_scratch: ScratchOwned<BE> = ScratchOwned::alloc(self.gglwe_encrypt_sk_tmp_bytes(res));
        self.gglwe_encrypt_sk(
            res,
            &sk_lwe_as_glwe.data,
            sk_glwe,
            enc_infos,
            source_xe,
            source_xa,
            &mut enc_scratch.arena(),
        );
    }
}
