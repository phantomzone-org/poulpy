use poulpy_hal::{
    api::{ModuleN, ScratchAvailable, VecZnxAutomorphism},
    layouts::{Backend, DataMut, GaloisElement, Module, Scratch},
    source::Source,
};

use crate::{
    GGLWECompressedEncryptSk, ScratchTakeCore,
    layouts::{
        GGLWECompressedSeedMut, GGLWECompressedToMut, GGLWEInfos, GLWEInfos, GLWESecret, GLWESecretPrepare, GLWESecretPrepared,
        GLWESecretPreparedAlloc, GLWESecretToRef, LWEInfos, SetAutomorphismGaloisElement, compressed::AutomorphismKeyCompressed,
    },
};

impl AutomorphismKeyCompressed<Vec<u8>> {
    pub fn encrypt_sk_tmp_bytes<M, BE: Backend, A>(module: &M, infos: &A) -> usize
    where
        A: GGLWEInfos,
        M: AutomorphismKeyCompressedEncryptSk<BE>,
    {
        module.automorphism_key_compressed_encrypt_sk_tmp_bytes(infos)
    }
}

impl<DataSelf: DataMut> AutomorphismKeyCompressed<DataSelf> {
    #[allow(clippy::too_many_arguments)]
    pub fn encrypt_sk<M, S, BE: Backend>(
        &mut self,
        module: &M,
        p: i64,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        S: GLWESecretToRef + GLWEInfos,
        M: AutomorphismKeyCompressedEncryptSk<BE>,
    {
        module.automorphism_key_compressed_encrypt_sk(self, p, sk, seed_xa, source_xe, scratch);
    }
}

pub trait AutomorphismKeyCompressedEncryptSk<BE: Backend> {
    fn automorphism_key_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn automorphism_key_compressed_encrypt_sk<R, S>(
        &self,
        res: &mut R,
        p: i64,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWECompressedToMut + GGLWECompressedSeedMut + SetAutomorphismGaloisElement + GGLWEInfos,
        S: GLWESecretToRef + GLWEInfos;
}

impl<BE: Backend> AutomorphismKeyCompressedEncryptSk<BE> for Module<BE>
where
    Self: ModuleN
        + GaloisElement
        + VecZnxAutomorphism
        + GGLWECompressedEncryptSk<BE>
        + GLWESecretPreparedAlloc<BE>
        + GLWESecretPrepare<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn automorphism_key_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(self.n() as u32, infos.n());
        self.gglwe_compressed_encrypt_sk_tmp_bytes(infos)
            .max(GLWESecret::bytes_of_from_infos(infos))
            + GLWESecretPrepared::bytes_of_from_infos(self, infos)
    }

    fn automorphism_key_compressed_encrypt_sk<R, S>(
        &self,
        res: &mut R,
        p: i64,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWECompressedToMut + GGLWECompressedSeedMut + SetAutomorphismGaloisElement + GGLWEInfos,
        S: GLWESecretToRef + GLWEInfos,
    {
        let sk: &GLWESecret<&[u8]> = &sk.to_ref();
        assert_eq!(res.n(), sk.n());
        assert_eq!(res.rank_out(), res.rank_in());
        assert_eq!(sk.rank(), res.rank_out());
        assert!(
            scratch.available() >= AutomorphismKeyCompressed::encrypt_sk_tmp_bytes(self, res),
            "scratch.available(): {} < AutomorphismKey::encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            AutomorphismKeyCompressed::encrypt_sk_tmp_bytes(self, res)
        );

        let (mut sk_out_prepared, scratch_1) = scratch.take_glwe_secret_prepared(self, sk.rank());
        {
            let (mut sk_out, _) = scratch_1.take_glwe_secret(self, sk.rank());
            for i in 0..res.rank_out().into() {
                self.vec_znx_automorphism(
                    self.galois_element_inv(p),
                    &mut sk_out.data.as_vec_znx_mut(),
                    i,
                    &sk.data.as_vec_znx(),
                    i,
                );
            }
            sk_out_prepared.prepare(self, &sk_out);
        }

        self.gglwe_compressed_encrypt_sk(
            res,
            &sk.data,
            &sk_out_prepared,
            seed_xa,
            source_xe,
            scratch_1,
        );

        res.set_p(p);
    }
}
