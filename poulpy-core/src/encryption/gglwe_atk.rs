use poulpy_hal::{
    api::{
        ModuleN, ScratchAvailable, SvpApplyDftToDftInplace, SvpPPolAlloc, SvpPPolBytesOf, SvpPrepare, VecZnxAddInplace, VecZnxAddNormal,
        VecZnxAddScalarInplace, VecZnxAutomorphism, VecZnxBigNormalize, VecZnxDftApply, VecZnxDftBytesOf, VecZnxFillUniform,
        VecZnxIdftApplyConsume, VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxSub, VecZnxSubInplace,
        VecZnxSwitchRing,
    },
    layouts::{Backend, DataMut, Module, Scratch},
    source::Source,
};

use crate::{
    ScratchTakeCore,
    layouts::{
        AutomorphismKey, AutomorphismKeyToMut, GGLWEInfos, GLWEInfos, GLWESecret, GLWESecretToRef, GLWESwitchingKey, LWEInfos,
    },
};

impl AutomorphismKey<Vec<u8>> {
    pub fn encrypt_sk_tmp_bytes<BE: Backend, A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos,
        Module<BE>: ModuleN + SvpPPolBytesOf + VecZnxNormalizeTmpBytes + VecZnxDftBytesOf + VecZnxNormalizeTmpBytes + SvpPPolAlloc<BE>,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWEAutomorphismKey"
        );
        GLWESwitchingKey::encrypt_sk_tmp_bytes(module, infos) + GLWESecret::bytes_of_from_infos(module, &infos.glwe_layout())
    }

    pub fn encrypt_pk_tmp_bytes<BE: Backend, A>(module: &Module<BE>, _infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            _infos.rank_in(),
            _infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWEAutomorphismKey"
        );
        GLWESwitchingKey::encrypt_pk_tmp_bytes(module, _infos)
    }
}

pub trait GGLWEAutomorphismKeyEncryptSk<BE: Backend> {
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
        + SvpPPolBytesOf,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
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