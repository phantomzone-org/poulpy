use poulpy_hal::{
    api::{ModuleN, ScratchArenaTakeBasic, ScratchOwnedAlloc, VecZnxSwitchRingBackend},
    layouts::{
        scalar_znx_as_vec_znx_backend_mut_from_mut, scalar_znx_as_vec_znx_backend_ref_from_ref, Backend, Module,
        ScalarZnx, ScratchArena, ScratchOwned,
    },
    source::Source,
};

pub use crate::api::GLWESwitchingKeyEncryptSk;
use crate::{
    EncryptionInfos, GetDistribution, ScratchArenaTakeCore,
    encryption::gglwe::GGLWEEncryptSk,
    layouts::{
        GGLWEInfos, GGLWEToBackendMut, GLWEInfos, GLWESecretToBackendRef, GLWESwitchingKeyDegreesMut, LWEInfos,
        prepared::GLWESecretPreparedFactory,
    },
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
        R: GGLWEToBackendMut<BE> + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        E: EncryptionInfos,
        S1: GLWESecretToBackendRef<BE> + GLWEInfos,
        S2: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos;
}

impl<BE: Backend> GLWESwitchingKeyEncryptSkDefault<BE> for Module<BE>
where
    Self: ModuleN + GGLWEEncryptSk<BE> + GLWESecretPreparedFactory<BE> + VecZnxSwitchRingBackend<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
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
        R: GGLWEToBackendMut<BE> + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        E: EncryptionInfos,
        S1: GLWESecretToBackendRef<BE> + GLWEInfos,
        S2: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos,
    {
        let sk_in = sk_in.to_backend_ref();
        let sk_out_ref = sk_out.to_backend_ref();

        assert!(sk_in.n().0 <= self.n() as u32);
        assert!(sk_out_ref.n().0 <= self.n() as u32);
        assert!(
            scratch.available()
                >= <Module<BE> as GLWESwitchingKeyEncryptSkDefault<BE>>::glwe_switching_key_encrypt_sk_tmp_bytes(self, res),
            "scratch.available(): {} < GLWESwitchingKeyEncryptSk::glwe_switching_key_encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            <Module<BE> as GLWESwitchingKeyEncryptSkDefault<BE>>::glwe_switching_key_encrypt_sk_tmp_bytes(self, res)
        );

        let (mut sk_in_tmp, scratch_1) = scratch.borrow().take_scalar_znx(self.n(), sk_in.rank().into());
        let sk_in_backend_vec = scalar_znx_as_vec_znx_backend_ref_from_ref::<BE>(&sk_in.data);
        for i in 0..sk_in.rank().into() {
            let mut sk_in_tmp_backend_vec = scalar_znx_as_vec_znx_backend_mut_from_mut::<BE>(&mut sk_in_tmp);
            self.vec_znx_switch_ring_backend(&mut sk_in_tmp_backend_vec, i, &sk_in_backend_vec, i);
        }

        let (mut sk_out_tmp, _scratch_2) = scratch_1.take_glwe_secret_prepared(self, sk_out_ref.rank());
        let mut sk_out_tmp_ref = &mut sk_out_tmp;
        self.glwe_secret_prepare(&mut sk_out_tmp_ref, sk_out);

        let mut enc_scratch: ScratchOwned<BE> = ScratchOwned::alloc(self.gglwe_encrypt_sk_tmp_bytes(res));
        let sk_in_tmp_ref = &mut sk_in_tmp;
        self.gglwe_encrypt_sk(
            res,
            &sk_in_tmp_ref,
            &sk_out_tmp_ref,
            enc_infos,
            source_xe,
            source_xa,
            &mut enc_scratch.arena(),
        );

        *res.input_degree() = sk_in.n();
        *res.output_degree() = sk_out_ref.n();
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
