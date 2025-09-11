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
    layouts::{GGLWECiphertext, GGLWESwitchingKey, GLWESecret, prepared::GLWESecretPrepared},
};

impl GGLWESwitchingKey<Vec<u8>> {
    pub fn encrypt_sk_scratch_space<B: Backend>(
        module: &Module<B>,
        basek: usize,
        k: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> usize
    where
        Module<B>: SvpPPolAllocBytes + VecZnxNormalizeTmpBytes + VecZnxDftAllocBytes + VecZnxNormalizeTmpBytes,
    {
        (GGLWECiphertext::encrypt_sk_scratch_space(module, basek, k) | ScalarZnx::alloc_bytes(module.n(), 1))
            + ScalarZnx::alloc_bytes(module.n(), rank_in)
            + GLWESecretPrepared::bytes_of(module, rank_out)
    }

    pub fn encrypt_pk_scratch_space<B: Backend>(
        module: &Module<B>,
        _basek: usize,
        _k: usize,
        _rank_in: usize,
        _rank_out: usize,
    ) -> usize {
        GGLWECiphertext::encrypt_pk_scratch_space(module, _basek, _k, _rank_out)
    }
}

impl<DataSelf: DataMut> GGLWESwitchingKey<DataSelf> {
    #[allow(clippy::too_many_arguments)]
    pub fn encrypt_sk<DataSkIn: DataRef, DataSkOut: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        sk_in: &GLWESecret<DataSkIn>,
        sk_out: &GLWESecret<DataSkOut>,
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
            + VecZnxSubABInplace
            + VecZnxAddInplace
            + VecZnxNormalizeInplace<B>
            + VecZnxAddNormal
            + VecZnxNormalize<B>
            + VecZnxSub
            + SvpPrepare<B>
            + VecZnxSwitchRing<B>
            + SvpPPolAllocBytes,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx + TakeScalarZnx + TakeGLWESecretPrepared<B>,
    {
        #[cfg(debug_assertions)]
        {
            use crate::layouts::Infos;

            assert!(sk_in.n() <= module.n());
            assert!(sk_out.n() <= module.n());
            assert!(
                scratch.available()
                    >= GGLWESwitchingKey::encrypt_sk_scratch_space(
                        module,
                        self.basek(),
                        self.k(),
                        self.rank_in(),
                        self.rank_out()
                    ),
                "scratch.available()={} < GLWESwitchingKey::encrypt_sk_scratch_space={}",
                scratch.available(),
                GGLWESwitchingKey::encrypt_sk_scratch_space(
                    module,
                    self.basek(),
                    self.k(),
                    self.rank_in(),
                    self.rank_out()
                )
            )
        }

        let n: usize = sk_in.n().max(sk_out.n());

        let (mut sk_in_tmp, scratch_1) = scratch.take_scalar_znx(n, sk_in.rank());
        (0..sk_in.rank()).for_each(|i| {
            module.vec_znx_switch_ring(
                &mut sk_in_tmp.as_vec_znx_mut(),
                i,
                &sk_in.data.as_vec_znx(),
                i,
            );
        });

        let (mut sk_out_tmp, scratch_2) = scratch_1.take_glwe_secret_prepared(n, sk_out.rank());
        {
            let (mut tmp, _) = scratch_2.take_scalar_znx(n, 1);
            (0..sk_out.rank()).for_each(|i| {
                module.vec_znx_switch_ring(&mut tmp.as_vec_znx_mut(), 0, &sk_out.data.as_vec_znx(), i);
                module.svp_prepare(&mut sk_out_tmp.data, i, &tmp, 0);
            });
        }

        self.key.encrypt_sk(
            module,
            &sk_in_tmp,
            &sk_out_tmp,
            source_xa,
            source_xe,
            scratch_2,
        );
        self.sk_in_n = sk_in.n();
        self.sk_out_n = sk_out.n();
    }
}
