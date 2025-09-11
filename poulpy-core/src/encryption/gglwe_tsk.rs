use poulpy_hal::{
    api::{
        ScratchAvailable, SvpApplyDftToDft, SvpApplyDftToDftInplace, SvpPPolAllocBytes, SvpPrepare, TakeScalarZnx, TakeVecZnx,
        TakeVecZnxBig, TakeVecZnxDft, VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxBigAllocBytes,
        VecZnxBigNormalize, VecZnxDftAllocBytes, VecZnxDftApply, VecZnxFillUniform, VecZnxIdftApplyConsume, VecZnxIdftApplyTmpA,
        VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxSub, VecZnxSubABInplace, VecZnxSwitchRing,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
    source::Source,
};

use crate::{
    TakeGLWESecret, TakeGLWESecretPrepared,
    layouts::{
        GGLWESwitchingKey, GGLWETensorKey, GLWESecret, Infos,
        prepared::{GLWESecretPrepared, Prepare},
    },
};

impl GGLWETensorKey<Vec<u8>> {
    pub fn encrypt_sk_scratch_space<B: Backend>(module: &Module<B>, basek: usize, k: usize, rank: usize) -> usize
    where
        Module<B>:
            SvpPPolAllocBytes + VecZnxNormalizeTmpBytes + VecZnxDftAllocBytes + VecZnxNormalizeTmpBytes + VecZnxBigAllocBytes,
    {
        GLWESecretPrepared::bytes_of(module, rank)
            + module.vec_znx_dft_alloc_bytes(rank, 1)
            + module.vec_znx_big_alloc_bytes(1, 1)
            + module.vec_znx_dft_alloc_bytes(1, 1)
            + GLWESecret::bytes_of(module.n(), 1)
            + GGLWESwitchingKey::encrypt_sk_scratch_space(module, basek, k, rank, rank)
    }
}

impl<DataSelf: DataMut> GGLWETensorKey<DataSelf> {
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
        Scratch<B>:
            TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx + TakeScalarZnx + TakeGLWESecretPrepared<B> + TakeVecZnxBig<B>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.rank(), sk.rank());
            assert_eq!(self.n(), sk.n());
        }

        let n: usize = sk.n();

        let rank: usize = self.rank();

        let (mut sk_dft_prep, scratch_1) = scratch.take_glwe_secret_prepared(n, rank);
        sk_dft_prep.prepare(module, sk, scratch_1);

        let (mut sk_dft, scratch_2) = scratch_1.take_vec_znx_dft(n, rank, 1);

        (0..rank).for_each(|i| {
            module.vec_znx_dft_apply(1, 0, &mut sk_dft, i, &sk.data.as_vec_znx(), i);
        });

        let (mut sk_ij_big, scratch_3) = scratch_2.take_vec_znx_big(n, 1, 1);
        let (mut sk_ij, scratch_4) = scratch_3.take_glwe_secret(n, 1);
        let (mut sk_ij_dft, scratch_5) = scratch_4.take_vec_znx_dft(n, 1, 1);

        (0..rank).for_each(|i| {
            (i..rank).for_each(|j| {
                module.svp_apply_dft_to_dft(&mut sk_ij_dft, 0, &sk_dft_prep.data, j, &sk_dft, i);

                module.vec_znx_idft_apply_tmpa(&mut sk_ij_big, 0, &mut sk_ij_dft, 0);
                module.vec_znx_big_normalize(
                    self.basek(),
                    &mut sk_ij.data.as_vec_znx_mut(),
                    0,
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
