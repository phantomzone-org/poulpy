use poulpy_hal::{
    api::{
        ModuleN, ScratchAvailable, SvpApplyDftToDftInplace, SvpPPolAlloc, SvpPPolBytesOf, SvpPrepare, VecZnxAddInplace, VecZnxAddNormal,
        VecZnxAddScalarInplace, VecZnxAutomorphism, VecZnxBigNormalize, VecZnxDftApply, VecZnxDftBytesOf, VecZnxFillUniform,
        VecZnxIdftApplyConsume, VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxSub, VecZnxSubInplace,
        VecZnxSwitchRing,
    },
    layouts::{Backend, DataMut, GaloisElement, Module, Scratch},
    source::Source,
};

use crate::{
    ScratchTakeCore,
    layouts::{
        AutomorphismKey, AutomorphismKeyToMut, GGLWEInfos, GLWEInfos, GLWESecret, GLWESecretToRef, GLWESwitchingKey,
    },
};

impl AutomorphismKey<Vec<u8>> {

    pub fn encrypt_sk_tmp_bytes<M, A, BE: Backend>(module: &M, infos: &A) -> usize
    where
        A: GGLWEInfos,
        M: GGLWEAutomorphismKeyEncryptSk<BE>
    {
        module.gglwe_automorphism_key_encrypt_sk_tmp_bytes(infos)
    }

    pub fn encrypt_pk_tmp_bytes<M, A, BE: Backend>(module: &M, infos: &A) -> usize
    where
        A: GGLWEInfos,
        M: GGLWEAutomorphismKeyEncryptPk<BE>
    {
        module.gglwe_automorphism_key_encrypt_pk_tmp_bytes(infos)
    }
}

impl<DM: DataMut> AutomorphismKey<DM>
where
    Self: AutomorphismKeyToMut,
{
    pub fn encrypt_sk<S, BE: Backend>(
        &mut self,
        module: &Module<BE>,
        p: i64,
        sk: &S,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        S: GLWESecretToRef,
        Module<BE>: GGLWEAutomorphismKeyEncryptSk<BE>,
    {
        module.gglwe_automorphism_key_encrypt_sk(self, p, sk, source_xa, source_xe, scratch);
    }
}

pub trait GGLWEAutomorphismKeyEncryptSk<BE: Backend> {
    fn gglwe_automorphism_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn gglwe_automorphism_key_encrypt_sk<A, B>(
        &self,
        res: &mut A,
        p: i64,
        sk: &B,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        A: AutomorphismKeyToMut,
        B: GLWESecretToRef;
}

impl<BE: Backend> GGLWEAutomorphismKeyEncryptSk<BE> for Module<BE>
where
    Module<BE>: ModuleN 
        + VecZnxAddScalarInplace
        + VecZnxDftBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxDftApply<BE>
        + SvpApplyDftToDftInplace<BE>
        + VecZnxIdftApplyConsume<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxFillUniform
        + VecZnxSubInplace
        + VecZnxAddInplace
        + VecZnxNormalizeInplace<BE>
        + VecZnxAddNormal
        + VecZnxNormalize<BE>
        + VecZnxSub
        + SvpPrepare<BE>
        + VecZnxSwitchRing
        + SvpPPolBytesOf
        + VecZnxAutomorphism
        + SvpPPolAlloc<BE>
        + GaloisElement,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{

    fn gglwe_automorphism_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWEAutomorphismKey"
        );
        GLWESwitchingKey::encrypt_sk_tmp_bytes(self, infos) + GLWESecret::bytes_of_from_infos(self, &infos.glwe_layout())
    }

    fn gglwe_automorphism_key_encrypt_sk<A, B>(
        &self,
        res: &mut A,
        p: i64,
        sk: &B,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        A: AutomorphismKeyToMut,
        B: GLWESecretToRef,
    {
        let res: &mut AutomorphismKey<&mut [u8]> = &mut res.to_mut();
        let sk: &GLWESecret<&[u8]> = &sk.to_ref();

        #[cfg(debug_assertions)]
        {
            use crate::layouts::{GLWEInfos, LWEInfos};

            assert_eq!(res.n(), sk.n());
            assert_eq!(res.rank_out(), res.rank_in());
            assert_eq!(sk.rank(), res.rank_out());
            assert!(
                scratch.available() >= AutomorphismKey::encrypt_sk_tmp_bytes(self, res),
                "scratch.available(): {} < AutomorphismKey::encrypt_sk_tmp_bytes: {:?}",
                scratch.available(),
                AutomorphismKey::encrypt_sk_tmp_bytes(self, res)
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

        res.key
            .encrypt_sk(self, sk, &sk_out, source_xa, source_xe, scratch_1);

        res.p = p;
    }
}


pub trait GGLWEAutomorphismKeyEncryptPk<BE: Backend> {
    fn gglwe_automorphism_key_encrypt_pk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

}

impl<BE: Backend> GGLWEAutomorphismKeyEncryptPk<BE> for Module<BE>
where
    Module<BE>: ModuleN 
        + VecZnxAddScalarInplace
        + VecZnxDftBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxDftApply<BE>
        + SvpApplyDftToDftInplace<BE>
        + VecZnxIdftApplyConsume<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxFillUniform
        + VecZnxSubInplace
        + VecZnxAddInplace
        + VecZnxNormalizeInplace<BE>
        + VecZnxAddNormal
        + VecZnxNormalize<BE>
        + VecZnxSub
        + SvpPPolBytesOf
        + SvpPPolAlloc<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{

    fn gglwe_automorphism_key_encrypt_pk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWEAutomorphismKey"
        );
        GLWESwitchingKey::encrypt_pk_tmp_bytes(self, infos)
    }
}