#![allow(clippy::too_many_arguments)]

use poulpy_hal::{
    api::{ModuleN, ScratchAvailable, VecZnxAutomorphism},
    layouts::{Backend, GaloisElement, Module, Scratch},
    source::Source,
};

pub use crate::api::GLWEAutomorphismKeyCompressedEncryptSk;
use crate::{
    EncryptionInfos, GGLWECompressedEncryptSk, ScratchTakeCore,
    layouts::{
        GGLWECompressedSeedMut, GGLWECompressedToMut, GGLWEInfos, GLWEInfos, GLWESecret, GLWESecretPreparedFactory,
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
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWECompressedToMut + GGLWECompressedSeedMut + SetGaloisElement + GGLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretToRef + GLWEInfos;
}

impl<BE: Backend> GLWEAutomorphismKeyCompressedEncryptSkDefault<BE> for Module<BE>
where
    Self: ModuleN + GaloisElement + VecZnxAutomorphism + GGLWECompressedEncryptSk<BE> + GLWESecretPreparedFactory<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(self.n() as u32, infos.n());
        let lvl_0: usize = self.bytes_of_glwe_secret_prepared_from_infos(infos);
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
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWECompressedToMut + GGLWECompressedSeedMut + SetGaloisElement + GGLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretToRef + GLWEInfos,
    {
        let sk: &GLWESecret<&[u8]> = &sk.to_ref();
        assert_eq!(res.n(), sk.n());
        assert_eq!(res.rank_out(), res.rank_in());
        assert_eq!(sk.rank(), res.rank_out());
        assert!(
            scratch.available() >= self.glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes(res),
            "scratch.available(): {} < GLWEAutomorphismKeyCompressedEncryptSk::glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            self.glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes(res)
        );

        let (mut sk_out_prepared, scratch_1) = scratch.take_glwe_secret_prepared(self, sk.rank());
        {
            let (mut sk_out, _) = scratch_1.take_glwe_secret(self.n().into(), sk.rank());
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
            self.prepare_glwe_secret(&mut sk_out_prepared, &sk_out);
        }

        self.gglwe_compressed_encrypt_sk(res, &sk.data, &sk_out_prepared, seed_xa, enc_infos, source_xe, scratch_1);

        res.set_p(p);
    }
}
