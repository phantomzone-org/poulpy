use poulpy_hal::{
    api::{
        ModuleN, ScratchAvailable, SvpApplyDftToDftInplace, SvpPPolAlloc, SvpPPolBytesOf, SvpPrepare, VecZnxAddInplace, VecZnxAddNormal,
        VecZnxAddScalarInplace, VecZnxAutomorphismInplace, VecZnxBigNormalize, VecZnxDftApply, VecZnxDftBytesOf,
        VecZnxFillUniform, VecZnxIdftApplyConsume, VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxSub,
        VecZnxSubInplace, VecZnxSwitchRing,
        ScratchTakeBasic
    }, 
    layouts::{Backend, DataMut, DataRef, Module, Scratch, ZnxView, ZnxViewMut, ZnxZero},
    oep::{ScratchAvailableImpl, VecZnxFillUniformImpl},
    source::Source
};

use crate::{
    ScratchTakeCore,
    layouts::{
        GGLWEInfos, GLWESecret, GLWESwitchingKey, GLWEToLWESwitchingKey, LWEInfos, LWESecret, Rank, GLWEToLWESwitchingKeyToMut,
        prepared::GLWESecretPrepared,
    },
    encryption::gglwe_ksk::GLWESwitchingKeyEncryptSk,
};

impl GLWEToLWESwitchingKey<Vec<u8>> {
    pub fn encrypt_sk_tmp_bytes<M, A, BE: Backend>(module: &M, infos: &A) -> usize
    where
        A: GGLWEInfos,
        M: GLWEToLWESwitchingKeyEncrypt<BE>,
    {
        module.glwe_to_lwe_switching_key_encrypt_sk_tmp_bytes(infos)
    }
    // pub fn encrypt_sk_tmp_bytes<B: Backend, A>(module: &Module<B>, infos: &A) -> usize
    // where
    //     A: GGLWEInfos,
    //     Module<B>: ModuleN + SvpPPolBytesOf + SvpPPolAlloc<B> + VecZnxNormalizeTmpBytes + VecZnxDftBytesOf + VecZnxNormalizeTmpBytes,
    // {
    //     GLWESecretPrepared::bytes_of(module, infos.rank_in())
    //         + (GLWESwitchingKey::encrypt_sk_tmp_bytes(module, infos) | GLWESecret::bytes_of(module, infos.rank_in()))
    // }
}

impl<D: DataMut> GLWEToLWESwitchingKey<D> {
    pub fn encrypt_sk<M, DLwe, DGlwe, BE: Backend>(
        &mut self,
        module: &M,
        sk_lwe: &LWESecret<DLwe>,
        sk_glwe: &GLWESecret<DGlwe>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        M: GLWEToLWESwitchingKeyEncrypt<BE>,
        DLwe: DataRef,
        DGlwe: DataRef,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        module.glwe_to_lwe_switching_key_encrypt_sk(self, sk_lwe, sk_glwe, source_xa, source_xe, scratch);
    }
    // #[allow(clippy::too_many_arguments)]
    // pub fn encrypt_sk<DLwe, DGlwe, B: Backend>(
    //     &mut self,
    //     module: &Module<B>,
    //     sk_lwe: &LWESecret<DLwe>,
    //     sk_glwe: &GLWESecret<DGlwe>,
    //     source_xa: &mut Source,
    //     source_xe: &mut Source,
    //     scratch: &mut Scratch<B>,
    // ) where
    //     DLwe: DataRef,
    //     DGlwe: DataRef,
    //     Module<B>: ModuleN
    //         + VecZnxAutomorphismInplace<B>
    //         + VecZnxAddScalarInplace
    //         + VecZnxDftBytesOf
    //         + VecZnxBigNormalize<B>
    //         + VecZnxDftApply<B>
    //         + SvpApplyDftToDftInplace<B>
    //         + VecZnxIdftApplyConsume<B>
    //         + VecZnxNormalizeTmpBytes
    //         + VecZnxFillUniform
    //         + VecZnxSubInplace
    //         + VecZnxAddInplace
    //         + VecZnxNormalizeInplace<B>
    //         + VecZnxAddNormal
    //         + VecZnxNormalize<B>
    //         + VecZnxSub
    //         + SvpPrepare<B>
    //         + VecZnxSwitchRing
    //         + SvpPPolBytesOf
    //         + SvpPPolAlloc<B>,
    //     Scratch<B>: ScratchAvailable + ScratchTakeCore<B>,
    // {
    //     #[cfg(debug_assertions)]
    //     {
    //         assert!(sk_lwe.n().0 <= module.n() as u32);
    //     }

    //     let (mut sk_lwe_as_glwe, scratch_1) = scratch.take_glwe_secret(module, Rank(1));
    //     sk_lwe_as_glwe.data.zero();
    //     sk_lwe_as_glwe.data.at_mut(0, 0)[..sk_lwe.n().into()].copy_from_slice(sk_lwe.data.at(0, 0));
    //     module.vec_znx_automorphism_inplace(-1, &mut sk_lwe_as_glwe.data.as_vec_znx_mut(), 0, scratch_1);

    //     self.0.encrypt_sk(
    //         module,
    //         sk_glwe,
    //         &sk_lwe_as_glwe,
    //         source_xa,
    //         source_xe,
    //         scratch_1,
    //     );
    // }
}

pub trait GLWEToLWESwitchingKeyEncrypt<BE: Backend>
where
    Self: Sized
        + ModuleN
        + SvpPPolBytesOf
        + SvpPPolAlloc<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxDftBytesOf
        + VecZnxAutomorphismInplace<BE>
        + VecZnxAddScalarInplace
        + VecZnxBigNormalize<BE>
        + VecZnxDftApply<BE>
        + SvpApplyDftToDftInplace<BE>
        + VecZnxIdftApplyConsume<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeInplace<BE>
        + VecZnxAddNormal
        + VecZnxSub
        + VecZnxSubInplace
        + VecZnxAddInplace
        + SvpPrepare<BE>
        + VecZnxSwitchRing
        + ScratchAvailable
        + ScratchAvailableImpl<BE>
        + VecZnxFillUniform
        + VecZnxFillUniformImpl<BE>
{
    fn glwe_to_lwe_switching_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_to_lwe_switching_key_encrypt_sk<R, DLwe: DataRef, DataGlwe: DataRef>(
        &self,
        res: &mut R,
        sk_lwe: &LWESecret<DLwe>,
        sk_glwe: &GLWESecret<DataGlwe>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GLWEToLWESwitchingKeyToMut,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;
}

impl<BE: Backend> GLWEToLWESwitchingKeyEncrypt<BE> for Module<BE> where
    Module<BE>: ModuleN
        + SvpPPolBytesOf
        + SvpPPolAlloc<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxDftBytesOf
        + VecZnxAutomorphismInplace<BE>
        + VecZnxAddScalarInplace
        + VecZnxBigNormalize<BE>
        + VecZnxDftApply<BE>
        + SvpApplyDftToDftInplace<BE>
        + VecZnxIdftApplyConsume<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeInplace<BE>
        + VecZnxAddNormal
        + VecZnxSub
        + VecZnxSubInplace
        + VecZnxAddInplace
        + SvpPrepare<BE>
        + VecZnxSwitchRing
        + ScratchAvailable
        + ScratchAvailableImpl<BE>
        + ScratchTakeBasic
        + ScratchTakeCore<BE>
        + VecZnxFillUniform
        + VecZnxFillUniformImpl<BE>
        + GLWESwitchingKeyEncryptSk<BE>
{
    fn glwe_to_lwe_switching_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize 
    where
        A: GGLWEInfos
    {
        GLWESecretPrepared::bytes_of(self, infos.rank_in())
        + (GLWESwitchingKey::encrypt_sk_tmp_bytes(self, infos) | GLWESecret::bytes_of(self, infos.rank_in()))
    }

    fn glwe_to_lwe_switching_key_encrypt_sk<R, DLwe: DataRef, DataGlwe: DataRef>(
        &self,
        res: &mut R,
        sk_lwe: &LWESecret<DLwe>,
        sk_glwe: &GLWESecret<DataGlwe>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GLWEToLWESwitchingKeyToMut,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {

        let res: &mut GLWEToLWESwitchingKey<&mut [u8]> = &mut res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert!(sk_lwe.n().0 <= self.n() as u32);
        }

        let (mut sk_lwe_as_glwe, scratch_1) = scratch.take_glwe_secret(self, Rank(1));
        sk_lwe_as_glwe.data.zero();
        sk_lwe_as_glwe.data.at_mut(0, 0)[..sk_lwe.n().into()].copy_from_slice(sk_lwe.data.at(0, 0));
        self.vec_znx_automorphism_inplace(-1, &mut sk_lwe_as_glwe.data.as_vec_znx_mut(), 0, scratch_1);

        res.0.encrypt_sk(
            self,
            sk_glwe,
            &sk_lwe_as_glwe,
            source_xa,
            source_xe,
            scratch_1,
        );        
    }
}