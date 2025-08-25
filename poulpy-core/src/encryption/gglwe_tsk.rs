use poulpy_hal::{
    api::{
        DFT, IDFTConsume, IDFTTmpA, ScratchAvailable, SvpApply, SvpApplyInplace, SvpPPolAllocBytes, SvpPrepare, TakeScalarZnx,
        TakeVecZnx, TakeVecZnxBig, TakeVecZnxDft, VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxBigAllocBytes,
        VecZnxBigNormalize, VecZnxDftAllocBytes, VecZnxFillUniform, VecZnxNormalize, VecZnxNormalizeInplace,
        VecZnxNormalizeTmpBytes, VecZnxSub, VecZnxSubABInplace, VecZnxSwithcDegree,
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
        Module<B>: SvpApply<B>
            + IDFTTmpA<B>
            + VecZnxAddScalarInplace
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
            + SvpPrepare<B>
            + VecZnxSwithcDegree
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

        let (mut sk_dft_prep, scratch1) = scratch.take_glwe_secret_prepared(n, rank);
        sk_dft_prep.prepare(module, sk, scratch1);

        let (mut sk_dft, scratch2) = scratch1.take_vec_znx_dft(n, rank, 1);

        (0..rank).for_each(|i| {
            module.dft(1, 0, &mut sk_dft, i, &sk.data.as_vec_znx(), i);
        });

        let (mut sk_ij_big, scratch3) = scratch2.take_vec_znx_big(n, 1, 1);
        let (mut sk_ij, scratch4) = scratch3.take_glwe_secret(n, 1);
        let (mut sk_ij_dft, scratch5) = scratch4.take_vec_znx_dft(n, 1, 1);

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
                    scratch5,
                );

                self.at_mut(i, j)
                    .encrypt_sk(module, &sk_ij, sk, source_xa, source_xe, scratch5);
            });
        })
    }
}
