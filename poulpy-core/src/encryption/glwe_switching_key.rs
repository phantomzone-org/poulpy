use poulpy_hal::{
    api::{ModuleN, ScratchArenaTakeBasic, ScratchOwnedAlloc, SvpPrepare},
    layouts::{Backend, HostDataMut, Module, ScalarZnx, ScalarZnxToBackendRef, ScratchArena, ScratchOwned, SvpPPolToBackendMut},
    source::Source,
};

pub use crate::api::GLWESwitchingKeyEncryptSk;
use crate::{
    EncryptionInfos, ScratchArenaTakeCore,
    encryption::gglwe::GGLWEEncryptSk,
    layouts::{
        GGLWEInfos, GGLWEToMut, GLWEInfos, GLWESecret, GLWESecretToRef, GLWESwitchingKeyDegreesMut, LWEInfos,
        prepared::GLWESecretPreparedFactory,
    },
    vec_znx_host_ops::vec_znx_switch_ring,
};

#[doc(hidden)]
pub trait GLWESwitchingKeyEncryptSkDefault<BE: Backend> {
    fn glwe_switching_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_switching_key_encrypt_sk<R, S1, S2, E>(
        &self,
        res: &mut R,
        sk_in: &S1,
        sk_out: &S2,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToMut + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        E: EncryptionInfos,
        S1: GLWESecretToRef,
        S2: GLWESecretToRef;
}

impl<BE: Backend> GLWESwitchingKeyEncryptSkDefault<BE> for Module<BE>
where
    Self: ModuleN + GGLWEEncryptSk<BE> + GLWESecretPreparedFactory<BE> + SvpPrepare<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
    for<'s> BE::BufMut<'s>: HostDataMut,
{
    fn glwe_switching_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(self.n() as u32, infos.n());

        let lvl_0: usize = ScalarZnx::bytes_of(self.n(), infos.rank_in().into());
        let lvl_1: usize = self.glwe_secret_prepared_bytes_of_from_infos(infos);
        let lvl_2: usize = ScalarZnx::bytes_of(self.n(), 1).max(self.gglwe_encrypt_sk_tmp_bytes(infos));

        lvl_0 + lvl_1 + lvl_2
    }

    #[allow(clippy::too_many_arguments)]
    fn glwe_switching_key_encrypt_sk<R, S1, S2, E>(
        &self,
        res: &mut R,
        sk_in: &S1,
        sk_out: &S2,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToMut + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        E: EncryptionInfos,
        S1: GLWESecretToRef,
        S2: GLWESecretToRef,
    {
        let sk_in: &GLWESecret<&[u8]> = &sk_in.to_ref();
        let sk_out: &GLWESecret<&[u8]> = &sk_out.to_ref();

        assert!(sk_in.n().0 <= self.n() as u32);
        assert!(sk_out.n().0 <= self.n() as u32);
        assert!(
            scratch.available()
                >= <Module<BE> as GLWESwitchingKeyEncryptSkDefault<BE>>::glwe_switching_key_encrypt_sk_tmp_bytes(self, res),
            "scratch.available(): {} < GLWESwitchingKeyEncryptSk::glwe_switching_key_encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            <Module<BE> as GLWESwitchingKeyEncryptSkDefault<BE>>::glwe_switching_key_encrypt_sk_tmp_bytes(self, res)
        );

        let (mut sk_in_tmp, scratch_1) = scratch.borrow().take_scalar_znx(self.n(), sk_in.rank().into());
        for i in 0..sk_in.rank().into() {
            vec_znx_switch_ring(&mut sk_in_tmp.as_vec_znx_mut(), i, &sk_in.data.as_vec_znx(), i);
        }

        let mut sk_out_tmp = self.glwe_secret_prepared_alloc(sk_out.rank());
        {
            let (mut tmp, _) = scratch_1.take_scalar_znx(self.n(), 1);
            let mut sk_out_tmp_data = sk_out_tmp.data.to_backend_mut();
            for i in 0..sk_out.rank().into() {
                vec_znx_switch_ring(&mut tmp.as_vec_znx_mut(), 0, &sk_out.data.as_vec_znx(), i);
                let tmp_ref = tmp.to_ref();
                let tmp_backend = ScalarZnx::from_data(BE::from_host_bytes(tmp_ref.data), tmp_ref.n, tmp_ref.cols);
                self.svp_prepare(
                    &mut sk_out_tmp_data,
                    i,
                    &<ScalarZnx<BE::OwnedBuf> as ScalarZnxToBackendRef<BE>>::to_backend_ref(&tmp_backend),
                    0,
                );
            }
        }

        sk_out_tmp.dist = sk_out.dist;

        let sk_in_tmp_ref = sk_in_tmp.to_ref();
        let sk_in_tmp_backend =
            ScalarZnx::from_data(BE::from_host_bytes(sk_in_tmp_ref.data), sk_in_tmp_ref.n, sk_in_tmp_ref.cols);
        let mut enc_scratch: ScratchOwned<BE> = ScratchOwned::alloc(self.gglwe_encrypt_sk_tmp_bytes(res));
        self.gglwe_encrypt_sk(
            res,
            &sk_in_tmp_backend,
            &sk_out_tmp,
            enc_infos,
            source_xe,
            source_xa,
            &mut enc_scratch.arena(),
        );

        *res.input_degree() = sk_in.n();
        *res.output_degree() = sk_out.n();
    }
}

#[doc(hidden)]
pub trait GLWESwitchingKeyEncryptPkDefault<BE: Backend> {
    fn glwe_switching_key_encrypt_pk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;
}

impl<BE: Backend> GLWESwitchingKeyEncryptPkDefault<BE> for Module<BE> {
    fn glwe_switching_key_encrypt_pk_tmp_bytes<A>(&self, _infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        unimplemented!()
    }
}
