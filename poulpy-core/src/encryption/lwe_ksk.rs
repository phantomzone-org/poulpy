use poulpy_hal::{
    api::{
        ModuleN, SvpApplyDftToDftInplace, SvpPPolAlloc, SvpPPolBytesOf, SvpPrepare, VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace,
        VecZnxAutomorphismInplace, VecZnxBigNormalize, VecZnxDftApply, VecZnxDftBytesOf, VecZnxFillUniform,
        VecZnxIdftApplyConsume, VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxSub, VecZnxSubInplace,
        VecZnxSwitchRing,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch, ZnxView, ZnxViewMut},
    source::Source,
};

use crate::{
    layouts::{
        GLWESecretAlloc,
        GGLWEInfos, GLWESecret, GLWESwitchingKey, LWEInfos, LWESecret, LWESwitchingKey, Rank,
        prepared::{GLWESecretPrepared, GLWESecretPreparedAlloc},
    },
    encryption::gglwe_ksk::GLWESwitchingKeyEncryptSk,
    ScratchTakeCore, 
};

impl LWESwitchingKey<Vec<u8>> {
    pub fn encrypt_sk_tmp_bytes<M, A, BE: Backend>(module: &M, infos: &A) -> usize
    where
        A: GGLWEInfos,
        M: LWESwitchingKeyEncrypt<BE>,
    {
        module.lwe_switching_key_encrypt_sk_tmp_bytes(infos)
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
        Module<B>: ModuleN
            + VecZnxAutomorphismInplace<B>
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
            + SvpPPolBytesOf
            + SvpPPolAlloc<B>,
        Scratch<B>: ScratchTakeCore<B>,
    {
        #[cfg(debug_assertions)]
        {
            assert!(sk_lwe_in.n().0 <= self.n().0);
            assert!(sk_lwe_out.n().0 <= self.n().0);
            assert!(self.n().0 <= module.n() as u32);
        }

        let (mut sk_in_glwe, scratch_1) = scratch.take_glwe_secret(module, Rank(1));
        let (mut sk_out_glwe, scratch_2) = scratch_1.take_glwe_secret(module, Rank(1));

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

pub trait LWESwitchingKeyEncrypt<BE: Backend>
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
        + VecZnxFillUniform
        + VecZnxSubInplace
        + VecZnxAddInplace
        + VecZnxNormalizeInplace<BE>
        + VecZnxAddNormal
        + VecZnxNormalize<BE>
        + VecZnxSub
        + SvpPrepare<BE>
        + VecZnxSwitchRing
        + GLWESecretAlloc
        + GLWESecretPreparedAlloc<BE>
{
    fn lwe_switching_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;
}

impl<BE:Backend> LWESwitchingKeyEncrypt<BE> for Module<BE> where 
    Self: ModuleN
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
        + VecZnxFillUniform
        + VecZnxSubInplace
        + VecZnxAddInplace
        + VecZnxNormalizeInplace<BE>
        + VecZnxAddNormal
        + VecZnxNormalize<BE>
        + VecZnxSub
        + SvpPrepare<BE>
        + VecZnxSwitchRing
        + GLWESecretAlloc
        + GLWESecretPreparedAlloc<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        ,
{
    fn lwe_switching_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos
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
        GLWESecret::bytes_of(self, Rank(1))
            + GLWESecretPrepared::bytes_of(self, Rank(1))
            + GLWESwitchingKey::encrypt_sk_tmp_bytes(self, infos)
    }
}