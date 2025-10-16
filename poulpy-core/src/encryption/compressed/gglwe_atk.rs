use poulpy_hal::{
    api::{ModuleN, ScratchAvailable, ScratchTakeBasic, SvpPPolAlloc, SvpPPolBytesOf, VecZnxAutomorphism, VecZnxDftBytesOf, VecZnxNormalizeTmpBytes},
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
    source::Source,
};

use crate::{
    ScratchTakeCore,
    encryption::compressed::gglwe_ksk::GGLWEKeyCompressedEncryptSk,
    layouts::{
        GGLWEInfos, GLWEInfos, GLWESecret, GLWESecretToRef, LWEInfos,
        compressed::{AutomorphismKeyCompressed, AutomorphismKeyCompressedToMut, GLWESwitchingKeyCompressed},
    },
};

impl AutomorphismKeyCompressed<Vec<u8>> {
    pub fn encrypt_sk_tmp_bytes<B: Backend, A>(module: &Module<B>, infos: &A) -> usize
    where
        A: GGLWEInfos,
        Module<B>: ModuleN + SvpPPolAlloc<B> + VecZnxNormalizeTmpBytes + VecZnxDftBytesOf + VecZnxNormalizeTmpBytes + SvpPPolBytesOf,
    {
        assert_eq!(module.n() as u32, infos.n());
        GLWESwitchingKeyCompressed::encrypt_sk_tmp_bytes(module, infos) + GLWESecret::bytes_of(module, infos.rank_out())
    }
}

pub trait GGLWEAutomorphismKeyCompressedEncryptSk<B: Backend> {
    fn gglwe_automorphism_key_compressed_encrypt_sk<R, S>(
        &self,
        res: &mut R,
        p: i64,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        R: AutomorphismKeyCompressedToMut,
        S: GLWESecretToRef;
}

impl<B: Backend> GGLWEAutomorphismKeyCompressedEncryptSk<B> for Module<B>
where
    Module<B>: ModuleN
        + GGLWEKeyCompressedEncryptSk<B>
        + VecZnxNormalizeTmpBytes
        + VecZnxDftBytesOf
        + SvpPPolBytesOf
        + VecZnxAutomorphism
        + SvpPPolAlloc<B>,
    Scratch<B>: ScratchAvailable + ScratchTakeBasic + ScratchTakeCore<B>,
{
    fn gglwe_automorphism_key_compressed_encrypt_sk<R, S>(
        &self,
        res: &mut R,
        p: i64,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        R: AutomorphismKeyCompressedToMut,
        S: GLWESecretToRef,
    {
        let res: &mut AutomorphismKeyCompressed<&mut [u8]> = &mut res.to_mut();
        let sk: &GLWESecret<&[u8]> = &sk.to_ref();

        #[cfg(debug_assertions)]
        {
            assert_eq!(res.n(), sk.n());
            assert_eq!(res.rank_out(), res.rank_in());
            assert_eq!(sk.rank(), res.rank_out());
            assert!(
                scratch.available() >= AutomorphismKeyCompressed::encrypt_sk_tmp_bytes(self, res),
                "scratch.available(): {} < AutomorphismKey::encrypt_sk_tmp_bytes: {}",
                scratch.available(),
                AutomorphismKeyCompressed::encrypt_sk_tmp_bytes(self, res)
            )
        }

        let (mut sk_out, scratch_1) = scratch.take_glwe_secret(self, sk.rank());

        {
            (0..res.rank_out().into()).for_each(|i| {
                self.vec_znx_automorphism(
                    self.galois_element_inv(p),
                    &mut sk_out.data.as_vec_znx_mut(),
                    i,
                    &sk.data.as_vec_znx(),
                    i,
                );
            });
        }

        self.gglwe_key_compressed_encrypt_sk(&mut res.key, sk, &sk_out, seed_xa, source_xe, scratch_1);

        res.p = p;
    }
}

impl<DataSelf: DataMut> AutomorphismKeyCompressed<DataSelf> {
    #[allow(clippy::too_many_arguments)]
    pub fn encrypt_sk<DataSk: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        p: i64,
        sk: &GLWESecret<DataSk>,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GGLWEAutomorphismKeyCompressedEncryptSk<B>,
    {
        module.gglwe_automorphism_key_compressed_encrypt_sk(self, p, sk, seed_xa, source_xe, scratch);
    }
}
