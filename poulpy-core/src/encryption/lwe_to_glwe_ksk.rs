use poulpy_hal::{
    api::{
        ModuleN, ScratchAvailable, SvpApplyDftToDftInplace, SvpPPolAlloc, SvpPPolBytesOf, SvpPrepare, VecZnxAddInplace, VecZnxAddNormal,
        VecZnxAddScalarInplace, VecZnxAutomorphismInplace, VecZnxBigNormalize, VecZnxDftApply, VecZnxDftBytesOf,
        VecZnxFillUniform, VecZnxIdftApplyConsume, VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxSub,
        VecZnxSubInplace, VecZnxSwitchRing,
        ScratchTakeBasic,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch, ZnxView, ZnxViewMut},
    oep::{ScratchAvailableImpl},
    source::Source,
};

use crate::{
    layouts::{GGLWEInfos, GLWESecret, GLWESwitchingKey, LWEInfos, LWESecret, LWEToGLWESwitchingKey, LWEToGLWESwitchingKeyToMut, Rank},
    ScratchTakeCore,
    encryption::gglwe_ksk::GLWESwitchingKeyEncryptSk,
};

impl LWEToGLWESwitchingKey<Vec<u8>> {
    pub fn encrypt_sk_tmp_bytes<M, A, BE: Backend>(module: &M, infos: &A) -> usize
    where
        A: GGLWEInfos,
        M: LWEToGLWESwitchingKeyEncrypt<BE>,
    {
        module.lwe_to_glwe_switching_key_encrypt_sk_tmp_bytes(infos)
    }

    // pub fn encrypt_sk_tmp_bytes<B: Backend, A>(module: &Module<B>, infos: &A) -> usize
    // where
    //     A: GGLWEInfos,
    //     Module<B>: ModuleN + SvpPPolBytesOf + VecZnxNormalizeTmpBytes + VecZnxDftBytesOf + VecZnxNormalizeTmpBytes + SvpPPolAlloc<B>,
    // {
    //     debug_assert_eq!(
    //         infos.rank_in(),
    //         Rank(1),
    //         "rank_in != 1 is not supported for LWEToGLWESwitchingKey"
    //     );
    //     GLWESwitchingKey::encrypt_sk_tmp_bytes(module, infos)
    //         + GLWESecret::bytes_of(module, infos.rank_in())
    // }
}

impl<D: DataMut> LWEToGLWESwitchingKey<D> {
    pub fn encrypt_sk<M, DLwe, DGlwe, BE: Backend>(
        &mut self,
        module: &M,
        sk_lwe: &LWESecret<DLwe>,
        sk_glwe: &GLWESecret<DGlwe>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        M: LWEToGLWESwitchingKeyEncrypt<BE>,
        DLwe: DataRef,
        DGlwe: DataRef,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        module.lwe_to_glwe_switching_key_encrypt_sk(self, sk_lwe, sk_glwe, source_xa, source_xe, scratch);
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
    //         use crate::layouts::LWEInfos;

    //         assert!(sk_lwe.n().0 <= module.n() as u32);
    //     }

    //     let (mut sk_lwe_as_glwe, scratch_1) = scratch.take_glwe_secret(module, Rank(1));
    //     sk_lwe_as_glwe.data.at_mut(0, 0)[..sk_lwe.n().into()].copy_from_slice(sk_lwe.data.at(0, 0));
    //     sk_lwe_as_glwe.data.at_mut(0, 0)[sk_lwe.n().into()..].fill(0);
    //     module.vec_znx_automorphism_inplace(-1, &mut sk_lwe_as_glwe.data.as_vec_znx_mut(), 0, scratch_1);

    //     self.0.encrypt_sk(
    //         module,
    //         &sk_lwe_as_glwe,
    //         sk_glwe,
    //         source_xa,
    //         source_xe,
    //         scratch_1,
    //     );
    // }
}

pub trait LWEToGLWESwitchingKeyEncrypt<BE: Backend>
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
        + ScratchTakeBasic
        + ScratchAvailableImpl<BE>
{
    fn lwe_to_glwe_switching_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn lwe_to_glwe_switching_key_encrypt_sk<R, DLwe: DataRef, DataGlwe: DataRef>(
        &self,
        res: &mut R,
        sk_lwe: &LWESecret<DLwe>,
        sk_glwe: &GLWESecret<DataGlwe>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: LWEToGLWESwitchingKeyToMut,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;
}

impl<BE: Backend> LWEToGLWESwitchingKeyEncrypt<BE> for Module<BE> where
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
        + GLWESwitchingKeyEncryptSk<BE>
{
    fn lwe_to_glwe_switching_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos
    {
        debug_assert_eq!(
            infos.rank_in(),
            Rank(1),
            "rank_in != 1 is not supported for LWEToGLWESwitchingKey"
        );
        GLWESwitchingKey::encrypt_sk_tmp_bytes(self, infos)
            + GLWESecret::bytes_of(self, infos.rank_in())
    }

    fn lwe_to_glwe_switching_key_encrypt_sk<R, DLwe: DataRef, DataGlwe: DataRef>(
        &self,
        res: &mut R,
        sk_lwe: &LWESecret<DLwe>,
        sk_glwe: &GLWESecret<DataGlwe>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where 
        R: LWEToGLWESwitchingKeyToMut,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        let res: &mut LWEToGLWESwitchingKey<&mut [u8]> = &mut res.to_mut();
        
        #[cfg(debug_assertions)]
        {
            use crate::layouts::LWEInfos;

            assert!(sk_lwe.n().0 <= self.n() as u32);
        }

        let (mut sk_lwe_as_glwe, scratch_1) = scratch.take_glwe_secret(self, Rank(1));
        sk_lwe_as_glwe.data.at_mut(0, 0)[..sk_lwe.n().into()].copy_from_slice(sk_lwe.data.at(0, 0));
        sk_lwe_as_glwe.data.at_mut(0, 0)[sk_lwe.n().into()..].fill(0);
        self.vec_znx_automorphism_inplace(-1, &mut sk_lwe_as_glwe.data.as_vec_znx_mut(), 0, scratch_1);

        res.0.encrypt_sk(
            self,
            &sk_lwe_as_glwe,
            sk_glwe,
            source_xa,
            source_xe,
            scratch_1,
        );
    }
}