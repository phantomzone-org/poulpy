use poulpy_hal::{
    api::{
        ScratchAvailable, SvpApplyDftToDftInplace, SvpPPolAllocBytes, SvpPrepare, TakeScalarZnx, TakeVecZnx, TakeVecZnxDft,
        VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxAutomorphism, VecZnxBigNormalize, VecZnxDftAllocBytes,
        VecZnxDftApply, VecZnxFillUniform, VecZnxIdftApplyConsume, VecZnxNormalize, VecZnxNormalizeInplace,
        VecZnxNormalizeTmpBytes, VecZnxSub, VecZnxSubInplace, VecZnxSwitchRing,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
    source::Source,
};

use crate::{
    TakeGLWESecret, TakeGLWESecretPrepared,
    layouts::{GGLWEAutomorphismKey, GGLWEInfos, GGLWESwitchingKey, GLWEInfos, GLWESecret, LWEInfos},
};

impl GGLWEAutomorphismKey<Vec<u8>> {
    pub fn encrypt_sk_scratch_space<B: Backend, A>(module: &Module<B>, infos: &A) -> usize
    where
        A: GGLWEInfos,
        Module<B>: SvpPPolAllocBytes + VecZnxNormalizeTmpBytes + VecZnxDftAllocBytes + VecZnxNormalizeTmpBytes,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWEAutomorphismKey"
        );
        GGLWESwitchingKey::encrypt_sk_scratch_space(module, infos) + GLWESecret::alloc_bytes(&infos.glwe_layout())
    }

    pub fn encrypt_pk_scratch_space<B: Backend, A>(module: &Module<B>, _infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            _infos.rank_in(),
            _infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWEAutomorphismKey"
        );
        GGLWESwitchingKey::encrypt_pk_scratch_space(module, _infos)
    }
}

impl<DataSelf: DataMut> GGLWEAutomorphismKey<DataSelf> {
    #[allow(clippy::too_many_arguments)]
    pub fn encrypt_sk<DataSk: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        p: i64,
        sk: &GLWESecret<DataSk>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: VecZnxAddScalarInplace
            + VecZnxDftAllocBytes
            + VecZnxBigNormalize<B>
            + VecZnxDftApply<B>
            + SvpApplyDftToDftInplace<B>
            + VecZnxIdftApplyConsume<B>
            + VecZnxNormalizeTmpBytes
            + VecZnxFillUniform
            + VecZnxSubInplace
            + VecZnxAddInplace
            + VecZnxNormalizeInplace<B>
            + VecZnxAddNormal
            + VecZnxNormalize<B>
            + VecZnxSub
            + SvpPrepare<B>
            + VecZnxSwitchRing
            + SvpPPolAllocBytes
            + VecZnxAutomorphism,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx + TakeScalarZnx + TakeGLWESecretPrepared<B>,
    {
        #[cfg(debug_assertions)]
        {
            use crate::layouts::{GLWEInfos, LWEInfos};

            assert_eq!(self.n(), sk.n());
            assert_eq!(self.rank_out(), self.rank_in());
            assert_eq!(sk.rank(), self.rank_out());
            assert!(
                scratch.available() >= GGLWEAutomorphismKey::encrypt_sk_scratch_space(module, self),
                "scratch.available(): {} < AutomorphismKey::encrypt_sk_scratch_space: {:?}",
                scratch.available(),
                GGLWEAutomorphismKey::encrypt_sk_scratch_space(module, self)
            )
        }

        let (mut sk_out, scratch_1) = scratch.take_glwe_secret(sk.n(), sk.rank());

        {
            (0..self.rank_out().into()).for_each(|i| {
                module.vec_znx_automorphism(
                    module.galois_element_inv(p),
                    &mut sk_out.data.as_vec_znx_mut(),
                    i,
                    &sk.data.as_vec_znx(),
                    i,
                );
            });
        }

        self.key
            .encrypt_sk(module, sk, &sk_out, source_xa, source_xe, scratch_1);

        self.p = p;
    }
}
