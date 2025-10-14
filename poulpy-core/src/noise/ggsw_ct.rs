use poulpy_hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyDftToDftInplace, VecZnxAddScalarInplace, VecZnxBigAddInplace,
        VecZnxBigAddSmallInplace, VecZnxBigAlloc, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxDftAlloc, VecZnxDftApply, VecZnxDftBytesOf, VecZnxIdftApplyConsume, VecZnxIdftApplyTmpA, VecZnxNormalizeTmpBytes,
        VecZnxSubInplace,
    },
    layouts::{Backend, DataRef, Module, ScalarZnx, ScratchOwned, VecZnxBig, VecZnxDft, ZnxZero},
    oep::{ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeVecZnxBigImpl, TakeVecZnxDftImpl},
};

use crate::layouts::{GGSW, GGSWInfos, GLWE, GLWEInfos, GLWEPlaintext, LWEInfos, prepared::GLWESecretPrepared};

impl<D: DataRef> GGSW<D> {
    pub fn assert_noise<B, DataSk, DataScalar, F>(
        &self,
        module: &Module<B>,
        sk_prepared: &GLWESecretPrepared<DataSk, B>,
        pt_want: &ScalarZnx<DataScalar>,
        max_noise: F,
    ) where
        DataSk: DataRef,
        DataScalar: DataRef,
        Module<B>: VecZnxDftBytesOf
            + VecZnxBigBytesOf
            + VecZnxDftApply<B>
            + SvpApplyDftToDftInplace<B>
            + VecZnxIdftApplyConsume<B>
            + VecZnxBigAddInplace<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxNormalizeTmpBytes
            + VecZnxBigAlloc<B>
            + VecZnxDftAlloc<B>
            + VecZnxBigNormalizeTmpBytes
            + VecZnxIdftApplyTmpA<B>
            + VecZnxAddScalarInplace
            + VecZnxSubInplace,
        B: Backend + TakeVecZnxDftImpl<B> + TakeVecZnxBigImpl<B> + ScratchOwnedAllocImpl<B> + ScratchOwnedBorrowImpl<B>,
        F: Fn(usize) -> f64,
    {
        let base2k: usize = self.base2k().into();
        let dsize: usize = self.dsize().into();

        let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(self);
        let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(self);
        let mut pt_dft: VecZnxDft<Vec<u8>, B> = module.vec_znx_dft_alloc(1, self.size());
        let mut pt_big: VecZnxBig<Vec<u8>, B> = module.vec_znx_big_alloc(1, self.size());

        let mut scratch: ScratchOwned<B> =
            ScratchOwned::alloc(GLWE::decrypt_scratch_space(module, self) | module.vec_znx_normalize_tmp_bytes());

        (0..(self.rank() + 1).into()).for_each(|col_j| {
            (0..self.dnum().into()).for_each(|row_i| {
                module.vec_znx_add_scalar_inplace(&mut pt.data, 0, (dsize - 1) + row_i * dsize, pt_want, 0);

                // mul with sk[col_j-1]
                if col_j > 0 {
                    module.vec_znx_dft_apply(1, 0, &mut pt_dft, 0, &pt.data, 0);
                    module.svp_apply_dft_to_dft_inplace(&mut pt_dft, 0, &sk_prepared.data, col_j - 1);
                    module.vec_znx_idft_apply_tmpa(&mut pt_big, 0, &mut pt_dft, 0);
                    module.vec_znx_big_normalize(
                        base2k,
                        &mut pt.data,
                        0,
                        base2k,
                        &pt_big,
                        0,
                        scratch.borrow(),
                    );
                }

                self.at(row_i, col_j)
                    .decrypt(module, &mut pt_have, sk_prepared, scratch.borrow());

                module.vec_znx_sub_inplace(&mut pt_have.data, 0, &pt.data, 0);

                let std_pt: f64 = pt_have.data.std(base2k, 0).log2();
                let noise: f64 = max_noise(col_j);
                assert!(std_pt <= noise, "{std_pt} > {noise}");

                pt.data.zero();
            });
        });
    }
}

impl<D: DataRef> GGSW<D> {
    pub fn print_noise<B, DataSk, DataScalar>(
        &self,
        module: &Module<B>,
        sk_prepared: &GLWESecretPrepared<DataSk, B>,
        pt_want: &ScalarZnx<DataScalar>,
    ) where
        DataSk: DataRef,
        DataScalar: DataRef,
        Module<B>: VecZnxDftBytesOf
            + VecZnxBigBytesOf
            + VecZnxDftApply<B>
            + SvpApplyDftToDftInplace<B>
            + VecZnxIdftApplyConsume<B>
            + VecZnxBigAddInplace<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxNormalizeTmpBytes
            + VecZnxBigAlloc<B>
            + VecZnxDftAlloc<B>
            + VecZnxBigNormalizeTmpBytes
            + VecZnxIdftApplyTmpA<B>
            + VecZnxAddScalarInplace
            + VecZnxSubInplace,
        B: Backend + TakeVecZnxDftImpl<B> + TakeVecZnxBigImpl<B> + ScratchOwnedAllocImpl<B> + ScratchOwnedBorrowImpl<B>,
    {
        let base2k: usize = self.base2k().into();
        let dsize: usize = self.dsize().into();

        let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(self);
        let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(self);
        let mut pt_dft: VecZnxDft<Vec<u8>, B> = module.vec_znx_dft_alloc(1, self.size());
        let mut pt_big: VecZnxBig<Vec<u8>, B> = module.vec_znx_big_alloc(1, self.size());

        let mut scratch: ScratchOwned<B> =
            ScratchOwned::alloc(GLWE::decrypt_scratch_space(module, self) | module.vec_znx_normalize_tmp_bytes());

        (0..(self.rank() + 1).into()).for_each(|col_j| {
            (0..self.dnum().into()).for_each(|row_i| {
                module.vec_znx_add_scalar_inplace(&mut pt.data, 0, (dsize - 1) + row_i * dsize, pt_want, 0);

                // mul with sk[col_j-1]
                if col_j > 0 {
                    module.vec_znx_dft_apply(1, 0, &mut pt_dft, 0, &pt.data, 0);
                    module.svp_apply_dft_to_dft_inplace(&mut pt_dft, 0, &sk_prepared.data, col_j - 1);
                    module.vec_znx_idft_apply_tmpa(&mut pt_big, 0, &mut pt_dft, 0);
                    module.vec_znx_big_normalize(
                        base2k,
                        &mut pt.data,
                        0,
                        base2k,
                        &pt_big,
                        0,
                        scratch.borrow(),
                    );
                }

                self.at(row_i, col_j)
                    .decrypt(module, &mut pt_have, sk_prepared, scratch.borrow());
                module.vec_znx_sub_inplace(&mut pt_have.data, 0, &pt.data, 0);

                let std_pt: f64 = pt_have.data.std(base2k, 0).log2();
                println!("col: {col_j} row: {row_i}: {std_pt}");
                pt.data.zero();
                // println!(">>>>>>>>>>>>>>>>");
            });
        });
    }
}
