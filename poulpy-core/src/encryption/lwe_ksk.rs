use poulpy_hal::{
    api::{
        ScratchAvailable, SvpApplyDftToDftInplace, SvpPPolAllocBytes, SvpPrepare, TakeScalarZnx, TakeVecZnx, TakeVecZnxDft,
        VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxAutomorphismInplace, VecZnxBigNormalize,
        VecZnxDftAllocBytes, VecZnxDftApply, VecZnxFillUniform, VecZnxIdftApplyConsume, VecZnxNormalize, VecZnxNormalizeInplace,
        VecZnxNormalizeTmpBytes, VecZnxSub, VecZnxSubInplace, VecZnxSwitchRing,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch, ZnxView, ZnxViewMut},
    source::Source,
};

use crate::{
    TakeGLWESecret, TakeGLWESecretPrepared,
    layouts::{
        Degree, GGLWEInfos, GLWESecret, GLWESwitchingKey, LWEInfos, LWESecret, LWESwitchingKey, Rank,
        prepared::GLWESecretPrepared,
    },
};

impl LWESwitchingKey<Vec<u8>> {
    pub fn encrypt_sk_scratch_space<B: Backend, A>(module: &Module<B>, infos: &A) -> usize
    where
        A: GGLWEInfos,
        Module<B>: SvpPPolAllocBytes + VecZnxNormalizeTmpBytes + VecZnxDftAllocBytes + VecZnxNormalizeTmpBytes,
    {
        debug_assert_eq!(
            infos.dsize().0,
            1,
            "dsize > 1 is not supported for LWESwitchingKey"
        );
        debug_assert_eq!(
            infos.rank_in().0,
            1,
            "rank_in > 1 is not supported for LWESwitchingKey"
        );
        debug_assert_eq!(
            infos.rank_out().0,
            1,
            "rank_out > 1 is not supported for LWESwitchingKey"
        );
        GLWESecret::bytes_of(Degree(module.n() as u32), Rank(1))
            + GLWESecretPrepared::bytes_of(module, Rank(1))
            + GLWESwitchingKey::encrypt_sk_scratch_space(module, infos)
    }
}

impl<D: DataMut> LWESwitchingKey<D> {
    #[allow(clippy::too_many_arguments)]
    pub fn encrypt_sk<DIn, DOut, B: Backend>(
        &mut self,
        module: &Module<B>,
        sk_lwe_in: &LWESecret<DIn>,
        sk_lwe_out: &LWESecret<DOut>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        DIn: DataRef,
        DOut: DataRef,
        Module<B>: VecZnxAutomorphismInplace<B>
            + VecZnxAddScalarInplace
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
            + SvpPPolAllocBytes,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx + TakeScalarZnx + TakeGLWESecretPrepared<B>,
    {
        #[cfg(debug_assertions)]
        {
            assert!(sk_lwe_in.n().0 <= self.n().0);
            assert!(sk_lwe_out.n().0 <= self.n().0);
            assert!(self.n().0 <= module.n() as u32);
        }

        let (mut sk_in_glwe, scratch_1) = scratch.take_glwe_secret(self.n(), Rank(1));
        let (mut sk_out_glwe, scratch_2) = scratch_1.take_glwe_secret(self.n(), Rank(1));

        sk_out_glwe.data.at_mut(0, 0)[..sk_lwe_out.n().into()].copy_from_slice(sk_lwe_out.data.at(0, 0));
        sk_out_glwe.data.at_mut(0, 0)[sk_lwe_out.n().into()..].fill(0);
        module.vec_znx_automorphism_inplace(-1, &mut sk_out_glwe.data.as_vec_znx_mut(), 0, scratch_2);

        sk_in_glwe.data.at_mut(0, 0)[..sk_lwe_in.n().into()].copy_from_slice(sk_lwe_in.data.at(0, 0));
        sk_in_glwe.data.at_mut(0, 0)[sk_lwe_in.n().into()..].fill(0);
        module.vec_znx_automorphism_inplace(-1, &mut sk_in_glwe.data.as_vec_znx_mut(), 0, scratch_2);

        self.0.encrypt_sk(
            module,
            &sk_in_glwe,
            &sk_out_glwe,
            source_xa,
            source_xe,
            scratch_2,
        );
    }
}
