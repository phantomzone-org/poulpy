use poulpy_hal::{
    api::{
        DFT, IDFTConsume, IDFTTmpA, ScratchAvailable, SvpApply, SvpApplyInplace, SvpPPolAlloc, SvpPPolAllocBytes, SvpPrepare,
        TakeScalarZnx, TakeVecZnx, TakeVecZnxBig, TakeVecZnxDft, VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace,
        VecZnxBigAllocBytes, VecZnxBigNormalize, VecZnxDftAllocBytes, VecZnxFillUniform, VecZnxNormalize, VecZnxNormalizeInplace,
        VecZnxNormalizeTmpBytes, VecZnxSub, VecZnxSubABInplace, VecZnxSwithcDegree,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
    source::Source,
};

use crate::{
    TakeGLWESecret, TakeGLWESecretPrepared,
    layouts::{GGLWETensorKey, GLWESecret, Infos, compressed::GGLWETensorKeyCompressed, prepared::Prepare},
};

impl GGLWETensorKeyCompressed<Vec<u8>> {
    pub fn encrypt_sk_scratch_space<B: Backend>(module: &Module<B>, basek: usize, k: usize, rank: usize) -> usize
    where
        Module<B>:
            SvpPPolAllocBytes + VecZnxNormalizeTmpBytes + VecZnxDftAllocBytes + VecZnxNormalizeTmpBytes + VecZnxBigAllocBytes,
    {
        GGLWETensorKey::encrypt_sk_scratch_space(module, basek, k, rank)
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
        Module<B>: SvpApply<B>
            + IDFTTmpA<B>
            + VecZnxDftAllocBytes
            + VecZnxBigNormalize<B>
            + DFT<B>
            + SvpApplyInplace<B>
            + IDFTConsume<B>
            + VecZnxNormalizeTmpBytes
            + VecZnxFillUniform
            + VecZnxSubABInplace
            + VecZnxAddInplace
            + VecZnxNormalizeInplace<B>
            + VecZnxAddNormal
            + VecZnxNormalize<B>
            + VecZnxSub
            + VecZnxSwithcDegree<B>
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
            assert_eq!(self.rank(), sk.rank());
            assert_eq!(self.n(), sk.n());
        }

        let n: usize = sk.n();
        let rank: usize = self.rank();

        let (mut sk_dft_prep, scratch_1) = scratch.take_glwe_secret_prepared(n, rank);
        sk_dft_prep.prepare(module, sk, scratch_1);

        let (mut sk_dft, scratch_2) = scratch_1.take_vec_znx_dft(n, rank, 1);

        (0..rank).for_each(|i| {
            module.dft(1, 0, &mut sk_dft, i, &sk.data.as_vec_znx(), i);
        });

        let (mut sk_ij_big, scratch_3) = scratch_2.take_vec_znx_big(n, 1, 1);
        let (mut sk_ij, scratch_4) = scratch_3.take_glwe_secret(n, 1);
        let (mut sk_ij_dft, scratch_5) = scratch_4.take_vec_znx_dft(n, 1, 1);

        let mut source_xa: Source = Source::new(seed_xa);

        (0..rank).for_each(|i| {
            (i..rank).for_each(|j| {
                module.svp_apply(&mut sk_ij_dft, 0, &sk_dft_prep.data, j, &sk_dft, i);

                module.idft_tmp_a(&mut sk_ij_big, 0, &mut sk_ij_dft, 0);
                module.vec_znx_big_normalize(
                    self.basek(),
                    &mut sk_ij.data.as_vec_znx_mut(),
                    0,
                    &sk_ij_big,
                    0,
                    scratch_5,
                );

                let (seed_xa_tmp, _) = source_xa.branch();

                self.at_mut(i, j)
                    .encrypt_sk(module, &sk_ij, sk, seed_xa_tmp, source_xe, scratch_5);
            });
        })
    }
}
