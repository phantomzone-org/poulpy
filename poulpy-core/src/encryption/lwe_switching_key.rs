use poulpy_hal::{
    api::{ModuleN, ScratchOwnedAlloc, VecZnxAutomorphism},
    layouts::{Backend, Module, ScratchArena, ScratchOwned, ZnxView, ZnxViewMut},
    source::Source,
};

use crate::{
    EncryptionInfos, ScratchArenaTakeCore,
    encryption::glwe_switching_key::GLWESwitchingKeyEncryptSk,
    layouts::{GGLWEInfos, GGLWEToMut, GLWESecret, GLWESwitchingKeyDegreesMut, LWEInfos, LWESecret, LWESecretToRef, Rank},
};

#[doc(hidden)]
pub trait LWESwitchingKeyEncryptDefault<BE: Backend> {
    fn lwe_switching_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn lwe_switching_key_encrypt_sk<R, S1, S2, E>(
        &self,
        res: &mut R,
        sk_lwe_in: &S1,
        sk_lwe_out: &S2,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToMut + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        E: EncryptionInfos,
        S1: LWESecretToRef,
        S2: LWESecretToRef;
}

impl<BE: Backend> LWESwitchingKeyEncryptDefault<BE> for Module<BE>
where
    Self: ModuleN + GLWESwitchingKeyEncryptSk<BE> + VecZnxAutomorphism,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    fn lwe_switching_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(infos.dsize().0, 1, "dsize > 1 is not supported for LWESwitchingKey");
        assert_eq!(infos.rank_in().0, 1, "rank_in > 1 is not supported for LWESwitchingKey");
        assert_eq!(infos.rank_out().0, 1, "rank_out > 1 is not supported for LWESwitchingKey");
        assert_eq!(self.n() as u32, infos.n());

        let lvl_0: usize = GLWESecret::bytes_of(self.n().into(), Rank(1));
        let lvl_1: usize = GLWESecret::bytes_of(self.n().into(), Rank(1));
        let lvl_2: usize = GLWESecret::bytes_of(self.n().into(), Rank(1));
        let lvl_3: usize = self.glwe_switching_key_encrypt_sk_tmp_bytes(infos);

        lvl_0 + lvl_1 + lvl_2 + lvl_3
    }

    #[allow(clippy::too_many_arguments)]
    fn lwe_switching_key_encrypt_sk<R, S1, S2, E>(
        &self,
        res: &mut R,
        sk_lwe_in: &S1,
        sk_lwe_out: &S2,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToMut + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        E: EncryptionInfos,
        S1: LWESecretToRef,
        S2: LWESecretToRef,
    {
        let sk_lwe_in: &LWESecret<&[u8]> = &sk_lwe_in.to_ref();
        let sk_lwe_out: &LWESecret<&[u8]> = &sk_lwe_out.to_ref();

        assert!(sk_lwe_in.n().0 <= res.n().0);
        assert!(sk_lwe_out.n().0 <= res.n().0);
        assert!(res.n() <= self.n() as u32);
        assert!(
            scratch.available()
                >= <Module<BE> as LWESwitchingKeyEncryptDefault<BE>>::lwe_switching_key_encrypt_sk_tmp_bytes(self, res),
            "scratch.available(): {} < LWESwitchingKeyEncrypt::lwe_switching_key_encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            <Module<BE> as LWESwitchingKeyEncryptDefault<BE>>::lwe_switching_key_encrypt_sk_tmp_bytes(self, res)
        );

        let mut sk_glwe_src = GLWESecret::alloc(self.n().into(), Rank(1));
        let mut sk_glwe_out = GLWESecret::alloc(self.n().into(), Rank(1));
        let mut sk_glwe_in = GLWESecret::alloc(self.n().into(), Rank(1));

        sk_glwe_out.dist = sk_lwe_out.dist;
        sk_glwe_src.dist = sk_lwe_out.dist;

        sk_glwe_src.data.at_mut(0, 0)[..sk_lwe_out.n().into()].copy_from_slice(sk_lwe_out.data.at(0, 0));
        sk_glwe_src.data.at_mut(0, 0)[sk_lwe_out.n().into()..].fill(0);
        {
            let sk_glwe_src_data = sk_glwe_src.data.as_vec_znx();
            self.vec_znx_automorphism(-1, &mut sk_glwe_out.data.as_vec_znx_mut(), 0, &sk_glwe_src_data, 0);
        }

        sk_glwe_src.dist = sk_lwe_in.dist;
        sk_glwe_in.dist = sk_lwe_in.dist;
        sk_glwe_src.data.at_mut(0, 0)[..sk_lwe_in.n().into()].copy_from_slice(sk_lwe_in.data.at(0, 0));
        sk_glwe_src.data.at_mut(0, 0)[sk_lwe_in.n().into()..].fill(0);
        {
            let sk_glwe_src_data = sk_glwe_src.data.as_vec_znx();
            self.vec_znx_automorphism(-1, &mut sk_glwe_in.data.as_vec_znx_mut(), 0, &sk_glwe_src_data, 0);
        }

        // TODO(device): LWESwitchingKey still stages its offline keygen through a
        // dedicated local arena borrow; fold that seam into the surrounding path.
        let mut enc_scratch: ScratchOwned<BE> = ScratchOwned::alloc(self.glwe_switching_key_encrypt_sk_tmp_bytes(res));
        self.glwe_switching_key_encrypt_sk(
            res,
            &sk_glwe_in,
            &sk_glwe_out,
            enc_infos,
            source_xe,
            source_xa,
            &mut enc_scratch.arena(),
        );
    }
}
