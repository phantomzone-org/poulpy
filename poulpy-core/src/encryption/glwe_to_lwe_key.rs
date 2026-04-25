use poulpy_hal::{
    api::{ModuleN, ScratchArenaTakeBasic, ScratchOwnedAlloc, SvpPrepare, VecZnxAutomorphism},
    layouts::{
        Backend, HostDataMut, Module, ScalarZnx, ScalarZnxToBackendRef, ScratchArena, ScratchOwned, ZnxView, ZnxViewMut, ZnxZero,
    },
    source::Source,
};

use crate::{
    EncryptionInfos, GGLWEEncryptSk, ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GGLWEToMut, GLWEInfos, GLWESecret, GLWESecretToRef, LWEInfos, LWESecret, LWESecretToRef, Rank,
        prepared::{GLWESecretPreparedFactory, GLWESecretPreparedToBackendMut},
    },
};

#[doc(hidden)]
pub trait GLWEToLWESwitchingKeyEncryptSkDefault<BE: Backend> {
    fn glwe_to_lwe_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_to_lwe_key_encrypt_sk<R, S1, S2, E>(
        &self,
        res: &mut R,
        sk_lwe: &S1,
        sk_glwe: &S2,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S1: LWESecretToRef,
        S2: GLWESecretToRef,
        E: EncryptionInfos,
        R: GGLWEToMut + GGLWEInfos;
}

impl<BE: Backend> GLWEToLWESwitchingKeyEncryptSkDefault<BE> for Module<BE>
where
    Self: ModuleN + GGLWEEncryptSk<BE> + GLWESecretPreparedFactory<BE> + VecZnxAutomorphism + SvpPrepare<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
    for<'s> BE::BufMut<'s>: HostDataMut,
{
    fn glwe_to_lwe_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(self.n() as u32, infos.n());

        let lvl_0: usize = self.glwe_secret_prepared_bytes_of(infos.rank_in());
        let lvl_1_sk_lwe_as_glwe_src: usize = GLWESecret::bytes_of(self.n().into(), Rank(1));
        let lvl_2_sk_lwe_as_glwe: usize = GLWESecret::bytes_of(self.n().into(), Rank(1));

        lvl_0 + lvl_1_sk_lwe_as_glwe_src + lvl_2_sk_lwe_as_glwe
    }

    #[allow(clippy::too_many_arguments)]
    fn glwe_to_lwe_key_encrypt_sk<R, S1, S2, E>(
        &self,
        res: &mut R,
        sk_lwe: &S1,
        sk_glwe: &S2,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S1: LWESecretToRef,
        S2: GLWESecretToRef,
        E: EncryptionInfos,
        R: GGLWEToMut + GGLWEInfos,
    {
        let sk_lwe: &LWESecret<&[u8]> = &sk_lwe.to_ref();
        let sk_glwe: &GLWESecret<&[u8]> = &sk_glwe.to_ref();

        assert!(sk_lwe.n().0 <= self.n() as u32);
        assert!(
            scratch.available()
                >= <Module<BE> as GLWEToLWESwitchingKeyEncryptSkDefault<BE>>::glwe_to_lwe_key_encrypt_sk_tmp_bytes(self, res),
            "scratch.available(): {} < GLWEToLWESwitchingKeyEncryptSk::glwe_to_lwe_key_encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            <Module<BE> as GLWEToLWESwitchingKeyEncryptSkDefault<BE>>::glwe_to_lwe_key_encrypt_sk_tmp_bytes(self, res)
        );

        let mut sk_lwe_as_glwe_prep = self.glwe_secret_prepared_alloc(Rank(1));
        let (data_src, scratch_1) = scratch.borrow().take_scalar_znx(self.n(), Rank(1).into());
        let mut sk_lwe_as_glwe_src = GLWESecret {
            data: data_src,
            dist: sk_lwe.dist,
        };
        sk_lwe_as_glwe_src.dist = sk_lwe.dist;
        sk_lwe_as_glwe_src.data.zero();
        sk_lwe_as_glwe_src.data.at_mut(0, 0)[..sk_lwe.n().into()].copy_from_slice(sk_lwe.data.at(0, 0));

        let (data_dst, _) = scratch_1.take_scalar_znx(self.n(), Rank(1).into());
        let mut sk_lwe_as_glwe = GLWESecret {
            data: data_dst,
            dist: sk_lwe.dist,
        };
        sk_lwe_as_glwe.dist = sk_lwe.dist;
        {
            let sk_lwe_as_glwe_src_data = sk_lwe_as_glwe_src.data.as_vec_znx();
            self.vec_znx_automorphism(-1, &mut sk_lwe_as_glwe.data.as_vec_znx_mut(), 0, &sk_lwe_as_glwe_src_data, 0);
        }
        {
            let sk_ref = sk_lwe_as_glwe.to_ref();
            let sk_backend = ScalarZnx::from_data(BE::from_host_bytes(sk_ref.data.data), sk_ref.data.n, sk_ref.data.cols);
            let sk_backend_ref = <ScalarZnx<BE::OwnedBuf> as ScalarZnxToBackendRef<BE>>::to_backend_ref(&sk_backend);
            let mut sk_lwe_as_glwe_prep_backend = sk_lwe_as_glwe_prep.to_backend_mut();
            for i in 0..sk_ref.rank().into() {
                self.svp_prepare(&mut sk_lwe_as_glwe_prep_backend.data, i, &sk_backend_ref, i);
            }
        }
        sk_lwe_as_glwe_prep.dist = sk_lwe_as_glwe.dist;

        let mut enc_scratch: ScratchOwned<BE> = ScratchOwned::alloc(self.gglwe_encrypt_sk_tmp_bytes(res));
        self.gglwe_encrypt_sk(
            res,
            &ScalarZnx::from_data(BE::from_host_bytes(sk_glwe.data.data), sk_glwe.data.n, sk_glwe.data.cols),
            &sk_lwe_as_glwe_prep,
            enc_infos,
            source_xe,
            source_xa,
            &mut enc_scratch.arena(),
        );
    }
}
