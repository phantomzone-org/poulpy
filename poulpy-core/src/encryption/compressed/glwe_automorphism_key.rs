#![allow(clippy::too_many_arguments)]

use poulpy_hal::{
    api::{ModuleN, ScratchOwnedAlloc, SvpPrepare, VecZnxAutomorphismBackend},
    layouts::{
        Backend, GaloisElement, HostDataMut, Module, ScalarZnx, ScalarZnxAsVecZnxBackendMut, ScalarZnxAsVecZnxBackendRef,
        ScalarZnxToBackendRef, ScratchArena, ScratchOwned, SvpPPolToBackendMut,
    },
    source::Source,
};

use crate::{
    EncryptionInfos, GGLWECompressedEncryptSk, ScratchArenaTakeCore,
    layouts::{
        GGLWECompressedSeedMut, GGLWECompressedToBackendMut, GGLWEInfos, GLWEInfos, GLWESecret, GLWESecretPreparedFactory,
        GLWESecretToRef, LWEInfos, SetGaloisElement,
    },
};

#[doc(hidden)]
pub trait GLWEAutomorphismKeyCompressedEncryptSkDefault<BE: Backend> {
    fn glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_automorphism_key_compressed_encrypt_sk<R, S, E>(
        &self,
        res: &mut R,
        p: i64,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWECompressedToBackendMut<BE> + GGLWECompressedSeedMut + SetGaloisElement + GGLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretToRef + GLWEInfos;
}

impl<BE: Backend> GLWEAutomorphismKeyCompressedEncryptSkDefault<BE> for Module<BE>
where
    Self: ModuleN + GaloisElement + VecZnxAutomorphismBackend<BE> + GGLWECompressedEncryptSk<BE> + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
    for<'s> BE::BufMut<'s>: HostDataMut,
{
    fn glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(self.n() as u32, infos.n());
        let lvl_0: usize = self.glwe_secret_prepared_bytes_of_from_infos(infos);
        let lvl_1: usize = self
            .gglwe_compressed_encrypt_sk_tmp_bytes(infos)
            .max(GLWESecret::bytes_of_from_infos(infos));

        lvl_0 + lvl_1
    }

    #[allow(clippy::too_many_arguments)]
    fn glwe_automorphism_key_compressed_encrypt_sk<R, S, E>(
        &self,
        res: &mut R,
        p: i64,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWECompressedToBackendMut<BE> + GGLWECompressedSeedMut + SetGaloisElement + GGLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretToRef + GLWEInfos,
    {
        let sk: &GLWESecret<&[u8]> = &sk.to_ref();
        assert_eq!(res.n(), sk.n());
        assert_eq!(res.rank_out(), res.rank_in());
        assert_eq!(sk.rank(), res.rank_out());
        assert!(
            scratch.available() >= <Module<BE> as GLWEAutomorphismKeyCompressedEncryptSkDefault<BE>>::glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes(self, res),
            "scratch.available(): {} < GLWEAutomorphismKeyCompressedEncryptSk::glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            <Module<BE> as GLWEAutomorphismKeyCompressedEncryptSkDefault<BE>>::glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes(self, res)
        );

        let mut sk_out_prepared = self.glwe_secret_prepared_alloc(sk.rank());
        {
            let mut sk_out = GLWESecret::alloc(self.n().into(), sk.rank());
            sk_out.dist = sk.dist;
            let sk_backend = ScalarZnx::from_data(BE::from_host_bytes(sk.data.data), sk.data.n, sk.data.cols);
            let mut sk_out_backend = ScalarZnx::from_data(
                BE::from_host_bytes(sk_out.data.data.as_ref()),
                sk_out.data.n,
                sk_out.data.cols,
            );
            for i in 0..sk.rank().into() {
                self.vec_znx_automorphism_backend(
                    self.galois_element_inv(p),
                    &mut <ScalarZnx<BE::OwnedBuf> as ScalarZnxAsVecZnxBackendMut<BE>>::as_vec_znx_backend_mut(
                        &mut sk_out_backend,
                    ),
                    i,
                    &<ScalarZnx<BE::OwnedBuf> as ScalarZnxAsVecZnxBackendRef<BE>>::as_vec_znx_backend(&sk_backend),
                    i,
                );
            }
            BE::copy_to_host(&sk_out_backend.data, sk_out.data.data.as_mut());
            let sk_out_backend = ScalarZnx::from_data(
                BE::from_host_bytes(sk_out.data.to_ref().data),
                sk_out.data.n,
                sk_out.data.cols,
            );
            let sk_out_backend_ref = <ScalarZnx<BE::OwnedBuf> as ScalarZnxToBackendRef<BE>>::to_backend_ref(&sk_out_backend);
            let mut sk_out_prepared_data = sk_out_prepared.data.to_backend_mut();
            for i in 0..sk_out.rank().into() {
                self.svp_prepare(&mut sk_out_prepared_data, i, &sk_out_backend_ref, i);
            }
            sk_out_prepared.dist = sk_out.dist;
        }

        let sk_backend = ScalarZnx::from_data(BE::from_host_bytes(sk.data.data), sk.data.n, sk.data.cols);
        let mut enc_scratch: ScratchOwned<BE> = ScratchOwned::alloc(self.gglwe_compressed_encrypt_sk_tmp_bytes(res));
        self.gglwe_compressed_encrypt_sk(
            res,
            &sk_backend,
            &sk_out_prepared,
            seed_xa,
            enc_infos,
            source_xe,
            &mut enc_scratch.arena(),
        );

        res.set_p(p);
    }
}
