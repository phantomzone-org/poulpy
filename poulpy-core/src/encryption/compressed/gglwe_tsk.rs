use poulpy_hal::{
    api::{
        ScratchAvailable, SvpApplyDftToDft, SvpApplyDftToDftInplace, SvpPPolAlloc, SvpPPolAllocBytes, SvpPrepare, TakeScalarZnx,
        TakeVecZnx, TakeVecZnxBig, TakeVecZnxDft, VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxBigAllocBytes,
        VecZnxBigNormalize, VecZnxDftAllocBytes, VecZnxDftApply, VecZnxFillUniform, VecZnxIdftApplyConsume, VecZnxIdftApplyTmpA,
        VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxSub, VecZnxSubABInplace, VecZnxSwitchRing,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
    source::Source,
};

use crate::{
    TakeGLWESecret, TakeGLWESecretPrepared,
    layouts::{
        GGLWELayoutInfos, GGLWETensorKey, GLWEInfos, GLWESecret, LWEInfos, Rank, compressed::GGLWETensorKeyCompressed,
        prepared::Prepare,
    },
};

impl GGLWETensorKeyCompressed<Vec<u8>> {
    pub fn encrypt_sk_scratch_space<B: Backend, A>(module: &Module<B>, infos: &A) -> usize
    where
        A: GGLWELayoutInfos,
        Module<B>:
            SvpPPolAllocBytes + VecZnxNormalizeTmpBytes + VecZnxDftAllocBytes + VecZnxNormalizeTmpBytes + VecZnxBigAllocBytes,
    {
        GGLWETensorKey::encrypt_sk_scratch_space(module, infos)
    }
}

impl<DataSelf: DataMut> GGLWETensorKeyCompressed<DataSelf> {
    pub fn encrypt_sk<DataSk: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        sk: &GLWESecret<DataSk>,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: SvpApplyDftToDft<B>
            + VecZnxIdftApplyTmpA<B>
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
            + VecZnxSwitchRing
            + VecZnxAddScalarInplace
            + SvpPrepare<B>
            + SvpPPolAllocBytes
            + SvpPPolAlloc<B>,
        Scratch<B>: ScratchAvailable
            + TakeScalarZnx
            + TakeVecZnxDft<B>
            + TakeGLWESecretPrepared<B>
            + ScratchAvailable
            + TakeVecZnx
            + TakeVecZnxBig<B>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.rank_out(), sk.rank());
            assert_eq!(self.n(), sk.n());
        }

        let n: usize = sk.n().into();
        let rank: usize = self.rank_out().into();

        let (mut sk_dft_prep, scratch_1) = scratch.take_glwe_secret_prepared(sk.n(), self.rank_out());
        sk_dft_prep.prepare(module, sk, scratch_1);

        let (mut sk_dft, scratch_2) = scratch_1.take_vec_znx_dft(n, rank, 1);

        for i in 0..rank {
            module.vec_znx_dft_apply(1, 0, &mut sk_dft, i, &sk.data.as_vec_znx(), i);
        }

        let (mut sk_ij_big, scratch_3) = scratch_2.take_vec_znx_big(n, 1, 1);
        let (mut sk_ij, scratch_4) = scratch_3.take_glwe_secret(sk.n(), Rank(1));
        let (mut sk_ij_dft, scratch_5) = scratch_4.take_vec_znx_dft(n, 1, 1);

        let mut source_xa: Source = Source::new(seed_xa);

        for i in 0..rank {
            for j in i..rank {
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

                let (seed_xa_tmp, _) = source_xa.branch();

                self.at_mut(i, j)
                    .encrypt_sk(module, &sk_ij, sk, seed_xa_tmp, source_xe, scratch_5);
            }
        }
    }
}
