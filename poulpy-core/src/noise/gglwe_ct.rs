use poulpy_hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyDftToDftInplace, VecZnxBigAddInplace, VecZnxBigAddSmallInplace,
        VecZnxBigAllocBytes, VecZnxBigNormalize, VecZnxDftAllocBytes, VecZnxDftApply, VecZnxIdftApplyConsume,
        VecZnxNormalizeTmpBytes, VecZnxSubScalarInplace,
    },
    layouts::{Backend, DataRef, Module, ScalarZnx, ScratchOwned, ZnxZero},
    oep::{ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeVecZnxBigImpl, TakeVecZnxDftImpl},
};

use crate::layouts::{GGLWE, GGLWEInfos, GLWE, GLWEPlaintext, LWEInfos, prepared::GLWESecretPrepared};

impl<D: DataRef> GGLWE<D> {
    pub fn assert_noise<B, DataSk, DataWant>(
        &self,
        module: &Module<B>,
        sk: &GLWESecretPrepared<DataSk, B>,
        pt_want: &ScalarZnx<DataWant>,
        max_noise: f64,
    ) where
        DataSk: DataRef,
        DataWant: DataRef,
        Module<B>: VecZnxDftAllocBytes
            + VecZnxBigAllocBytes
            + VecZnxDftApply<B>
            + SvpApplyDftToDftInplace<B>
            + VecZnxIdftApplyConsume<B>
            + VecZnxBigAddInplace<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxNormalizeTmpBytes
            + VecZnxSubScalarInplace,
        B: Backend + TakeVecZnxDftImpl<B> + TakeVecZnxBigImpl<B> + ScratchOwnedAllocImpl<B> + ScratchOwnedBorrowImpl<B>,
    {
        let dsize: usize = self.dsize().into();
        let base2k: usize = self.base2k().into();

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(GLWE::decrypt_scratch_space(module, self));
        let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(self);

        (0..self.rank_in().into()).for_each(|col_i| {
            (0..self.dnum().into()).for_each(|row_i| {
                self.at(row_i, col_i)
                    .decrypt(module, &mut pt, sk, scratch.borrow());

                module.vec_znx_sub_scalar_inplace(&mut pt.data, 0, (dsize - 1) + row_i * dsize, pt_want, col_i);

                let noise_have: f64 = pt.data.std(base2k, 0).log2();

                println!("noise_have: {noise_have}");

                assert!(
                    noise_have <= max_noise,
                    "noise_have: {noise_have} > max_noise: {max_noise}"
                );

                pt.data.zero();
            });
        });
    }
}
