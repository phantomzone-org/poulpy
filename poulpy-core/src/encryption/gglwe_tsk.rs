use poulpy_hal::{
    api::{
        SvpApplyDftToDft, SvpApplyDftToDftInplace, SvpPPolBytesOf, SvpPrepare, VecZnxAddInplace, VecZnxAddNormal,
        VecZnxAddScalarInplace, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxDftApply, VecZnxDftBytesOf, VecZnxFillUniform,
        VecZnxIdftApplyConsume, VecZnxIdftApplyTmpA, VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxSub,
        VecZnxSubInplace, VecZnxSwitchRing,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
    source::Source,
};

use crate::layouts::{
    Degree, GGLWEInfos, GLWEInfos, GLWESecret, GLWESwitchingKey, LWEInfos, Rank, TensorKey, prepared::GLWESecretPrepared,
};

impl TensorKey<Vec<u8>> {
    pub fn encrypt_sk_tmp_bytes<B: Backend, A>(module: &Module<B>, infos: &A) -> usize
    where
        A: GGLWEInfos,
        Module<B>: SvpPPolBytesOf + VecZnxNormalizeTmpBytes + VecZnxDftBytesOf + VecZnxNormalizeTmpBytes + VecZnxBigBytesOf,
    {
        GLWESecretPrepared::bytes_of(module, infos.rank_out())
            + module.bytes_of_vec_znx_dft(infos.rank_out().into(), 1)
            + module.bytes_of_vec_znx_big(1, 1)
            + module.bytes_of_vec_znx_dft(1, 1)
            + GLWESecret::bytes_of(Degree(module.n() as u32), Rank(1))
            + GLWESwitchingKey::encrypt_sk_tmp_bytes(module, infos)
    }
}

impl<DataSelf: DataMut> TensorKey<DataSelf> {
    pub fn encrypt_sk<DataSk: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        sk: &GLWESecret<DataSk>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: SvpApplyDftToDft<B>
            + VecZnxIdftApplyTmpA<B>
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
        Scratch<B>:,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.rank_out(), sk.rank());
            assert_eq!(self.n(), sk.n());
        }

        let n: Degree = sk.n();
        let rank: Rank = self.rank_out();

        let (mut sk_dft_prep, scratch_1) = scratch.take_glwe_secret_prepared(n, rank);
        sk_dft_prep.prepare(module, sk, scratch_1);

        let (mut sk_dft, scratch_2) = scratch_1.take_vec_znx_dft(n.into(), rank.into(), 1);

        (0..rank.into()).for_each(|i| {
            module.vec_znx_dft_apply(1, 0, &mut sk_dft, i, &sk.data.as_vec_znx(), i);
        });

        let (mut sk_ij_big, scratch_3) = scratch_2.take_vec_znx_big(n.into(), 1, 1);
        let (mut sk_ij, scratch_4) = scratch_3.take_glwe_secret(n, Rank(1));
        let (mut sk_ij_dft, scratch_5) = scratch_4.take_vec_znx_dft(n.into(), 1, 1);

        (0..rank.into()).for_each(|i| {
            (i..rank.into()).for_each(|j| {
                module.svp_apply_dft_to_dft(&mut sk_ij_dft, 0, &sk_dft_prep.data, j, &sk_dft, i);

                module.vec_znx_idft_apply_tmpa(&mut sk_ij_big, 0, &mut sk_ij_dft, 0);
                module.vec_znx_big_normalize(
                    self.base2k().into(),
                    &mut sk_ij.data.as_vec_znx_mut(),
                    0,
                    self.base2k().into(),
                    &sk_ij_big,
                    0,
                    scratch_5,
                );

                self.at_mut(i, j)
                    .encrypt_sk(module, &sk_ij, sk, source_xa, source_xe, scratch_5);
            });
        })
    }
}
