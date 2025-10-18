use poulpy_hal::{
    api::{ModuleN, ScratchAvailable, VecZnxAutomorphism},
    layouts::{Backend, DataMut, GaloisElement, Module, Scratch},
    source::Source,
};

use crate::{
    ScratchTakeCore,
    encryption::compressed::gglwe_ksk::GLWESwitchingKeyCompressedEncryptSk,
    layouts::{
        GGLWEInfos, GLWEInfos, GLWESecret, GLWESecretAlloc, GLWESecretToRef, LWEInfos,
        compressed::{AutomorphismKeyCompressed, AutomorphismKeyCompressedToMut},
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
        S: GLWESecretToRef,
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
        R: AutomorphismKeyCompressedToMut,
        S: GLWESecretToRef;
}

impl<BE: Backend> AutomorphismKeyCompressedEncryptSk<BE> for Module<BE>
where
    Self: ModuleN + GaloisElement + VecZnxAutomorphism + GLWESwitchingKeyCompressedEncryptSk<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn automorphism_key_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        self.glwe_switching_key_compressed_encrypt_sk_tmp_bytes(infos) + self.bytes_of_glwe_secret(infos.rank())
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
        R: AutomorphismKeyCompressedToMut,
        S: GLWESecretToRef,
    {
        let res: &mut AutomorphismKeyCompressed<&mut [u8]> = &mut res.to_mut();
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

        let (mut sk_out, scratch_1) = scratch.take_glwe_secret(self, sk.rank());

        {
            for i in 0..res.rank_out().into() {
                self.vec_znx_automorphism(
                    self.galois_element_inv(p),
                    &mut sk_out.data.as_vec_znx_mut(),
                    i,
                    &sk.data.as_vec_znx(),
                    i,
                );
            }
        }

        self.glwe_switching_key_compressed_encrypt_sk(&mut res.key, sk, &sk_out, seed_xa, source_xe, scratch_1);

        res.p = p;
    }
}
