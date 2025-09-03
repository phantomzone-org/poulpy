use poulpy_hal::{
    api::{
        DFT, IDFTConsume, ScratchAvailable, SvpApplyInplace, SvpPPolAllocBytes, SvpPrepare, TakeScalarZnx, TakeVecZnx,
        TakeVecZnxDft, VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxAutomorphismInplace, VecZnxBigNormalize,
        VecZnxDftAllocBytes, VecZnxFillUniform, VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxSub,
        VecZnxSubABInplace, VecZnxSwithcDegree,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch, ZnxView, ZnxViewMut},
    source::Source,
};

use crate::{
    TakeGLWESecret, TakeGLWESecretPrepared,
    layouts::{GGLWESwitchingKey, GLWESecret, Infos, LWESecret, LWESwitchingKey, prepared::GLWESecretPrepared},
};

impl LWESwitchingKey<Vec<u8>> {
    pub fn encrypt_sk_scratch_space<B: Backend>(module: &Module<B>, basek: usize, k: usize) -> usize
    where
        Module<B>: SvpPPolAllocBytes + VecZnxNormalizeTmpBytes + VecZnxDftAllocBytes + VecZnxNormalizeTmpBytes,
    {
        GLWESecret::bytes_of(module.n(), 1)
            + GLWESecretPrepared::bytes_of(module, 1)
            + GGLWESwitchingKey::encrypt_sk_scratch_space(module, basek, k, 1, 1)
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
        Module<B>: VecZnxAutomorphismInplace
            + VecZnxAddScalarInplace
            + VecZnxDftAllocBytes
            + VecZnxBigNormalize<B>
            + DFT<B>
            + SvpApplyInplace<B>
            + IDFTConsume<B>
            + VecZnxNormalizeTmpBytes
            + VecZnxFillUniform
            + VecZnxSubABInplace
            + VecZnxAddInplace
            + VecZnxNormalizeInplace<B>
            + VecZnxAddNormal
            + VecZnxNormalize<B>
            + VecZnxSub
            + SvpPrepare<B>
            + VecZnxSwithcDegree<B>
            + SvpPPolAllocBytes,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx + TakeScalarZnx + TakeGLWESecretPrepared<B>,
    {
        #[cfg(debug_assertions)]
        {
            assert!(sk_lwe_in.n() <= self.n());
            assert!(sk_lwe_out.n() <= self.n());
            assert!(self.n() <= module.n());
        }

        let (mut sk_in_glwe, scratch1) = scratch.take_glwe_secret(self.n(), 1);
        let (mut sk_out_glwe, scratch2) = scratch1.take_glwe_secret(self.n(), 1);

        sk_out_glwe.data.at_mut(0, 0)[..sk_lwe_out.n()].copy_from_slice(sk_lwe_out.data.at(0, 0));
        sk_out_glwe.data.at_mut(0, 0)[sk_lwe_out.n()..].fill(0);
        module.vec_znx_automorphism_inplace(-1, &mut sk_out_glwe.data.as_vec_znx_mut(), 0);

        sk_in_glwe.data.at_mut(0, 0)[..sk_lwe_in.n()].copy_from_slice(sk_lwe_in.data.at(0, 0));
        sk_in_glwe.data.at_mut(0, 0)[sk_lwe_in.n()..].fill(0);
        module.vec_znx_automorphism_inplace(-1, &mut sk_in_glwe.data.as_vec_znx_mut(), 0);

        self.0.encrypt_sk(
            module,
            &sk_in_glwe,
            &sk_out_glwe,
            source_xa,
            source_xe,
            scratch2,
        );
    }
}
