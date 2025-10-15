use poulpy_hal::{
    api::{
        ScratchAvailable, SvpApplyDftToDftInplace, SvpPPolBytesOf, SvpPrepare, VecZnxAddInplace, VecZnxAddNormal,
        VecZnxAddScalarInplace, VecZnxBigNormalize, VecZnxDftApply, VecZnxDftBytesOf, VecZnxFillUniform, VecZnxIdftApplyConsume,
        VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxSub, VecZnxSubInplace, VecZnxSwitchRing,
    },
    layouts::{Backend, DataMut, DataRef, Module, ScalarZnx, Scratch},
    source::Source,
};

use crate::layouts::{
    GGLWE, GGLWEInfos, GLWEInfos, GLWESecret, GLWESwitchingKey, LWEInfos, RingDegree, prepared::GLWESecretPrepared,
};

impl GLWESwitchingKey<Vec<u8>> {
    pub fn encrypt_sk_tmp_bytes<B: Backend, A>(module: &Module<B>, infos: &A) -> usize
    where
        A: GGLWEInfos,
        Module<B>: SvpPPolBytesOf + VecZnxNormalizeTmpBytes + VecZnxDftBytesOf + VecZnxNormalizeTmpBytes,
    {
        (GGLWE::encrypt_sk_tmp_bytes(module, infos) | ScalarZnx::bytes_of(module.n(), 1))
            + ScalarZnx::bytes_of(module.n(), infos.rank_in().into())
            + GLWESecretPrepared::bytes_of_from_infos(module, &infos.glwe_layout())
    }

    pub fn encrypt_pk_tmp_bytes<B: Backend, A>(module: &Module<B>, _infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        GGLWE::encrypt_pk_tmp_bytes(module, _infos)
    }
}

impl<DataSelf: DataMut> GLWESwitchingKey<DataSelf> {
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
            assert!(sk_in.n().0 <= module.n() as u32);
            assert!(sk_out.n().0 <= module.n() as u32);
            assert!(
                scratch.available() >= GLWESwitchingKey::encrypt_sk_tmp_bytes(module, self),
                "scratch.available()={} < GLWESwitchingKey::encrypt_sk_tmp_bytes={}",
                scratch.available(),
                GLWESwitchingKey::encrypt_sk_tmp_bytes(module, self)
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

        let (mut sk_out_tmp, scratch_2) = scratch_1.take_glwe_secret_prepared(RingDegree(n as u32), sk_out.rank());
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
            source_xa,
            source_xe,
            scratch_2,
        );
        self.sk_in_n = sk_in.n().into();
        self.sk_out_n = sk_out.n().into();
    }
}
