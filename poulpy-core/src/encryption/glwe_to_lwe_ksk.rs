use poulpy_hal::{
    api::{
        ScratchAvailable, SvpApplyDftToDftInplace, SvpPPolBytesOf, SvpPrepare, VecZnxAddInplace, VecZnxAddNormal,
        VecZnxAddScalarInplace, VecZnxAutomorphismInplace, VecZnxBigNormalize, VecZnxDftApply, VecZnxDftBytesOf,
        VecZnxFillUniform, VecZnxIdftApplyConsume, VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxSub,
        VecZnxSubInplace, VecZnxSwitchRing,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch, ZnxView, ZnxViewMut, ZnxZero},
    source::Source,
};

use crate::layouts::{
    GGLWEInfos, GLWESecret, GLWESwitchingKey, GLWEToLWESwitchingKey, LWEInfos, LWESecret, Rank, prepared::GLWESecretPrepared,
};

impl GLWEToLWESwitchingKey<Vec<u8>> {
    pub fn encrypt_sk_tmp_bytes<B: Backend, A>(module: &Module<B>, infos: &A) -> usize
    where
        A: GGLWEInfos,
        Module<B>: SvpPPolBytesOf + VecZnxNormalizeTmpBytes + VecZnxDftBytesOf + VecZnxNormalizeTmpBytes,
    {
        GLWESecretPrepared::bytes_of(module, infos.rank_in())
            + (GLWESwitchingKey::encrypt_sk_tmp_bytes(module, infos) | GLWESecret::bytes_of(infos.n(), infos.rank_in()))
    }
}

impl<D: DataMut> GLWEToLWESwitchingKey<D> {
    #[allow(clippy::too_many_arguments)]
    pub fn encrypt_sk<DLwe, DGlwe, B: Backend>(
        &mut self,
        module: &Module<B>,
        sk_lwe: &LWESecret<DLwe>,
        sk_glwe: &GLWESecret<DGlwe>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        DLwe: DataRef,
        DGlwe: DataRef,
        Module<B>: VecZnxAutomorphismInplace<B>
            + VecZnxAddScalarInplace
            + VecZnxDftBytesOf
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
            + SvpPPolBytesOf,
        Scratch<B>: ScratchAvailable,
    {
        #[cfg(debug_assertions)]
        {
            assert!(sk_lwe.n().0 <= module.n() as u32);
        }

        let (mut sk_lwe_as_glwe, scratch_1) = scratch.take_glwe_secret(sk_glwe.n(), Rank(1));
        sk_lwe_as_glwe.data.zero();
        sk_lwe_as_glwe.data.at_mut(0, 0)[..sk_lwe.n().into()].copy_from_slice(sk_lwe.data.at(0, 0));
        module.vec_znx_automorphism_inplace(-1, &mut sk_lwe_as_glwe.data.as_vec_znx_mut(), 0, scratch_1);

        self.0.encrypt_sk(
            module,
            sk_glwe,
            &sk_lwe_as_glwe,
            source_xa,
            source_xe,
            scratch_1,
        );
    }
}
