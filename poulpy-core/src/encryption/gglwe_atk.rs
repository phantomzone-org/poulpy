use poulpy_hal::{
    api::{ScratchAvailable, VecZnxAutomorphism},
    layouts::{Backend, DataMut, GaloisElement, Module, Scratch},
    source::Source,
};

use crate::{
    ScratchTakeCore,
    encryption::gglwe_ksk::GLWESwitchingKeyEncryptSk,
    layouts::{AutomorphismKey, AutomorphismKeyToMut, GGLWEInfos, GLWEInfos, GLWESecret, GLWESecretToRef, LWEInfos},
};

impl AutomorphismKey<Vec<u8>> {
    pub fn encrypt_sk_tmp_bytes<M, A, BE: Backend>(module: &M, infos: &A) -> usize
    where
        A: GGLWEInfos,
        M: AutomorphismKeyEncryptSk<BE>,
    {
        module.automorphism_key_encrypt_sk_tmp_bytes(infos)
    }

    pub fn encrypt_pk_tmp_bytes<M, A, BE: Backend>(module: &M, infos: &A) -> usize
    where
        A: GGLWEInfos,
        M: GGLWEAutomorphismKeyEncryptPk<BE>,
    {
        module.automorphism_key_encrypt_pk_tmp_bytes(infos)
    }
}

impl<DM: DataMut> AutomorphismKey<DM>
where
    Self: AutomorphismKeyToMut,
{
    pub fn encrypt_sk<S, M, BE: Backend>(
        &mut self,
        module: &M,
        p: i64,
        sk: &S,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        S: GLWESecretToRef,
        M: AutomorphismKeyEncryptSk<BE>,
    {
        module.automorphism_key_encrypt_sk(self, p, sk, source_xa, source_xe, scratch);
    }
}

pub trait AutomorphismKeyEncryptSk<BE: Backend> {
    fn automorphism_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn automorphism_key_encrypt_sk<R, S>(
        &self,
        res: &mut R,
        p: i64,
        sk: &S,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: AutomorphismKeyToMut,
        S: GLWESecretToRef;
}

impl<BE: Backend> AutomorphismKeyEncryptSk<BE> for Module<BE>
where
    Self: GLWESwitchingKeyEncryptSk<BE> + VecZnxAutomorphism + GaloisElement,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn automorphism_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWEAutomorphismKey"
        );
        self.glwe_switching_key_encrypt_sk_tmp_bytes(infos) + GLWESecret::bytes_of_from_infos(self, infos)
    }

    fn automorphism_key_encrypt_sk<R, S>(
        &self,
        res: &mut R,
        p: i64,
        sk: &S,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: AutomorphismKeyToMut,
        S: GLWESecretToRef,
    {
        let res: &mut AutomorphismKey<&mut [u8]> = &mut res.to_mut();
        let sk: &GLWESecret<&[u8]> = &sk.to_ref();

        assert_eq!(res.n(), sk.n());
        assert_eq!(res.rank_out(), res.rank_in());
        assert_eq!(sk.rank(), res.rank_out());
        assert!(
            scratch.available() >= self.automorphism_key_encrypt_sk_tmp_bytes(res),
            "scratch.available(): {} < AutomorphismKey::encrypt_sk_tmp_bytes: {:?}",
            scratch.available(),
            self.automorphism_key_encrypt_sk_tmp_bytes(res)
        );

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

        res.key
            .encrypt_sk(self, sk, &sk_out, source_xa, source_xe, scratch_1);

        res.p = p;
    }
}

pub trait GGLWEAutomorphismKeyEncryptPk<BE: Backend> {
    fn automorphism_key_encrypt_pk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;
}

impl<BE: Backend> GGLWEAutomorphismKeyEncryptPk<BE> for Module<BE>
where
    Self:,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn automorphism_key_encrypt_pk_tmp_bytes<A>(&self, _infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        unimplemented!()
    }
}
