use poulpy_hal::{
    api::{ScratchAvailable, SvpPPolAllocBytes, VecZnxAutomorphism, VecZnxDftAllocBytes, VecZnxNormalizeTmpBytes},
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
    source::Source,
};

use crate::{
    TakeGLWESecret,
    encryption::compressed::gglwe_ksk::GGLWEKeyCompressedEncryptSk,
    layouts::{
        GGLWEInfos, GLWEInfos, GLWESecret, GLWESecretToRef, LWEInfos,
        compressed::{GGLWEAutomorphismKeyCompressed, GGLWEAutomorphismKeyCompressedToMut, GGLWEKeyCompressed},
    },
};

impl GGLWEAutomorphismKeyCompressed<Vec<u8>> {
    pub fn encrypt_sk_scratch_space<B: Backend, A>(module: &Module<B>, infos: &A) -> usize
    where
        A: GGLWEInfos,
        Module<B>: VecZnxNormalizeTmpBytes + VecZnxDftAllocBytes + VecZnxNormalizeTmpBytes + SvpPPolAllocBytes,
    {
        assert_eq!(module.n() as u32, infos.n());
        GGLWEKeyCompressed::encrypt_sk_scratch_space(module, infos) + GLWESecret::alloc_bytes_with(infos.n(), infos.rank_out())
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
        R: GGLWEAutomorphismKeyCompressedToMut,
        S: GLWESecretToRef;
}

impl<B: Backend> GGLWEAutomorphismKeyCompressedEncryptSk<B> for Module<B>
where
    Module<B>:
        GGLWEKeyCompressedEncryptSk<B> + VecZnxNormalizeTmpBytes + VecZnxDftAllocBytes + SvpPPolAllocBytes + VecZnxAutomorphism,
    Scratch<B>: TakeGLWESecret + ScratchAvailable,
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
        R: GGLWEAutomorphismKeyCompressedToMut,
        S: GLWESecretToRef,
    {
        let res: &mut GGLWEAutomorphismKeyCompressed<&mut [u8]> = &mut res.to_mut();
        let sk: &GLWESecret<&[u8]> = &sk.to_ref();

        #[cfg(debug_assertions)]
        {
            assert_eq!(res.n(), sk.n());
            assert_eq!(res.rank_out(), res.rank_in());
            assert_eq!(sk.rank(), res.rank_out());
            assert!(
                scratch.available() >= GGLWEAutomorphismKeyCompressed::encrypt_sk_scratch_space(self, res),
                "scratch.available(): {} < AutomorphismKey::encrypt_sk_scratch_space: {}",
                scratch.available(),
                GGLWEAutomorphismKeyCompressed::encrypt_sk_scratch_space(self, res)
            )
        }

        let (mut sk_out, scratch_1) = scratch.take_glwe_secret(sk.n(), sk.rank());

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

impl<DataSelf: DataMut> GGLWEAutomorphismKeyCompressed<DataSelf> {
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
