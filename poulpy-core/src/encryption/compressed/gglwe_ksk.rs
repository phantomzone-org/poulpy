use poulpy_hal::{
    api::{
        ScratchAvailable, SvpApplyDftToDftInplace, SvpPPolAllocBytes, SvpPrepare, TakeScalarZnx, TakeVecZnx, TakeVecZnxDft,
        VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxBigNormalize, VecZnxDftAllocBytes, VecZnxDftApply,
        VecZnxFillUniform, VecZnxIdftApplyConsume, VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxSub,
        VecZnxSubABInplace, VecZnxSwitchRing,
    },
    layouts::{Backend, DataMut, DataRef, Module, ScalarZnx, Scratch},
    source::Source,
};

use crate::{
    TakeGLWESecretPrepared,
    layouts::{
        Degree, GGLWECiphertext, GGLWELayoutInfos, GLWEInfos, GLWESecret, LWEInfos, compressed::GGLWESwitchingKeyCompressed,
        prepared::GLWESecretPrepared,
    },
};

impl GGLWESwitchingKeyCompressed<Vec<u8>> {
    pub fn encrypt_sk_scratch_space<B: Backend, A>(module: &Module<B>, infos: &A) -> usize
    where
        A: GGLWELayoutInfos,
        Module<B>: VecZnxNormalizeTmpBytes + VecZnxDftAllocBytes + VecZnxNormalizeTmpBytes + SvpPPolAllocBytes,
    {
        (GGLWECiphertext::encrypt_sk_scratch_space(module, infos) | ScalarZnx::alloc_bytes(module.n(), 1))
            + ScalarZnx::alloc_bytes(module.n(), infos.rank_in().into())
            + GLWESecretPrepared::alloc_bytes_with(module, infos.rank_out())
    }
}

impl<DataSelf: DataMut> GGLWESwitchingKeyCompressed<DataSelf> {
    #[allow(clippy::too_many_arguments)]
    pub fn encrypt_sk<DataSkIn: DataRef, DataSkOut: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        sk_in: &GLWESecret<DataSkIn>,
        sk_out: &GLWESecret<DataSkOut>,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: SvpPrepare<B>
            + SvpPPolAllocBytes
            + VecZnxSwitchRing
            + VecZnxDftAllocBytes
            + VecZnxBigNormalize<B>
            + VecZnxDftApply<B>
            + SvpApplyDftToDftInplace<B>
            + VecZnxIdftApplyConsume<B>
            + VecZnxNormalizeTmpBytes
            + VecZnxFillUniform
            + VecZnxSubABInplace
            + VecZnxAddInplace
            + VecZnxNormalizeInplace<B>
            + VecZnxAddNormal
            + VecZnxNormalize<B>
            + VecZnxSub
            + VecZnxAddScalarInplace,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx + TakeScalarZnx + TakeGLWESecretPrepared<B>,
    {
        #[cfg(debug_assertions)]
        {
            use crate::layouts::GGLWESwitchingKey;

            assert!(sk_in.n().0 <= module.n() as u32);
            assert!(sk_out.n().0 <= module.n() as u32);
            assert!(
                scratch.available() >= GGLWESwitchingKey::encrypt_sk_scratch_space(module, self),
                "scratch.available()={} < GLWESwitchingKey::encrypt_sk_scratch_space={}",
                scratch.available(),
                GGLWESwitchingKey::encrypt_sk_scratch_space(module, self)
            )
        }

        let n: usize = sk_in.n().max(sk_out.n()).into();

        let (mut sk_in_tmp, scratch_1) = scratch.take_scalar_znx(n, sk_in.rank().into());
        (0..sk_in.rank().into()).for_each(|i| {
            module.vec_znx_switch_ring(
                &mut sk_in_tmp.as_vec_znx_mut(),
                i,
                &sk_in.data.as_vec_znx(),
                i,
            );
        });

        let (mut sk_out_tmp, scratch_2) = scratch_1.take_glwe_secret_prepared(Degree(n as u32), sk_out.rank());
        {
            let (mut tmp, _) = scratch_2.take_scalar_znx(n, 1);
            (0..sk_out.rank().into()).for_each(|i| {
                module.vec_znx_switch_ring(&mut tmp.as_vec_znx_mut(), 0, &sk_out.data.as_vec_znx(), i);
                module.svp_prepare(&mut sk_out_tmp.data, i, &tmp, 0);
            });
        }

        self.key.encrypt_sk(
            module,
            &sk_in_tmp,
            &sk_out_tmp,
            seed_xa,
            source_xe,
            scratch_2,
        );
        self.sk_in_n = sk_in.n().into();
        self.sk_out_n = sk_out.n().into();
    }
}
