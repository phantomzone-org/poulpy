use poulpy_hal::{
    api::{ScratchAvailable, ScratchOwnedAlloc, SvpPPolBytesOf, SvpPrepare, VecZnxAutomorphism},
    layouts::{Backend, GaloisElement, HostDataMut, Module, Scratch, ScratchArena, ScratchOwned, SvpPPolToBackendMut},
    source::Source,
};

use crate::{
    EncryptionInfos, GGLWEEncryptSk, ScratchArenaTakeCore, ScratchTakeCore,
    layouts::{
        GGLWEInfos, GGLWEToMut, GLWEInfos, GLWESecret, GLWESecretPreparedFactory, GLWESecretToRef, LWEInfos, SetGaloisElement,
    },
};

#[doc(hidden)]
pub trait GLWEAutomorphismKeyEncryptSkDefault<BE: Backend> {
    fn glwe_automorphism_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_automorphism_key_encrypt_sk<R, S, E>(
        &self,
        res: &mut R,
        p: i64,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWEToMut + SetGaloisElement + GGLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretToRef;
}

impl<BE: Backend> GLWEAutomorphismKeyEncryptSkDefault<BE> for Module<BE>
where
    Self: GGLWEEncryptSk<BE> + VecZnxAutomorphism + GaloisElement + SvpPPolBytesOf + GLWESecretPreparedFactory<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
    for<'s> BE::BufMut<'s>: HostDataMut,
{
    fn glwe_automorphism_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWEAutomorphismKey"
        );
        assert_eq!(self.n() as u32, infos.n());

        let lvl_0: usize = self.glwe_secret_prepared_bytes_of_from_infos(infos);
        let lvl_1_encrypt: usize = self.gglwe_encrypt_sk_tmp_bytes(infos);
        let lvl_1_sk: usize = GLWESecret::bytes_of_from_infos(infos);
        let lvl_1: usize = lvl_1_encrypt.max(lvl_1_sk);

        lvl_0 + lvl_1
    }

    #[allow(clippy::too_many_arguments)]
    fn glwe_automorphism_key_encrypt_sk<R, S, E>(
        &self,
        res: &mut R,
        p: i64,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,

        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWEToMut + SetGaloisElement + GGLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretToRef,
    {
        let sk: &GLWESecret<&[u8]> = &sk.to_ref();

        assert_eq!(res.n(), sk.n());
        assert_eq!(res.rank_out(), res.rank_in());
        assert_eq!(sk.rank(), res.rank_out());
        assert!(
            scratch.available()
                >= <Module<BE> as GLWEAutomorphismKeyEncryptSkDefault<BE>>::glwe_automorphism_key_encrypt_sk_tmp_bytes(self, res),
            "scratch.available(): {} < GLWEAutomorphismKeyEncryptSk::glwe_automorphism_key_encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            <Module<BE> as GLWEAutomorphismKeyEncryptSkDefault<BE>>::glwe_automorphism_key_encrypt_sk_tmp_bytes(self, res)
        );

        let mut sk_out_prepared = self.glwe_secret_prepared_alloc(sk.rank());

        {
            let (mut sk_out, _) = scratch.take_glwe_secret(sk.n(), sk.rank());
            sk_out.dist = sk.dist;

            for i in 0..sk.rank().into() {
                self.vec_znx_automorphism(
                    self.galois_element_inv(p),
                    &mut sk_out.data.as_vec_znx_mut(),
                    i,
                    &sk.data.as_vec_znx(),
                    i,
                );
            }
            let mut sk_out_prepared_data = sk_out_prepared.data.to_backend_mut();
            for i in 0..sk_out.rank().into() {
                self.svp_prepare(&mut sk_out_prepared_data, i, &sk_out.data, i);
            }
            sk_out_prepared.dist = sk_out.dist;
        }

        let mut enc_scratch: ScratchOwned<BE> = ScratchOwned::alloc(self.gglwe_encrypt_sk_tmp_bytes(res));
        self.gglwe_encrypt_sk(
            res,
            &sk.data,
            &sk_out_prepared,
            enc_infos,
            source_xe,
            source_xa,
            &mut enc_scratch.arena(),
        );

        res.set_p(p);
    }
}

#[doc(hidden)]
pub trait GLWEAutomorphismKeyEncryptPkDefault<BE: Backend> {
    fn glwe_automorphism_key_encrypt_pk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;
}

impl<BE: Backend> GLWEAutomorphismKeyEncryptPkDefault<BE> for Module<BE>
where
    Self:,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn glwe_automorphism_key_encrypt_pk_tmp_bytes<A>(&self, _infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        unimplemented!()
    }
}
