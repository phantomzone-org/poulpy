use poulpy_hal::{
    api::{
        DFT, IDFTConsume, ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyInplace, VecZnxBigAddInplace, VecZnxBigAddSmallInplace,
        VecZnxBigAllocBytes, VecZnxBigNormalize, VecZnxDftAllocBytes, VecZnxNormalizeTmpBytes, VecZnxSubScalarInplace,
    },
    layouts::{Backend, DataRef, Module, ScalarZnx, ScratchOwned, ZnxZero},
    oep::{ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeVecZnxBigImpl, TakeVecZnxDftImpl},
};

use crate::layouts::{GGLWECiphertext, GLWECiphertext, GLWEPlaintext, Infos, prepared::GLWESecretPrepared};

impl<D: DataRef> GGLWECiphertext<D> {
    pub fn assert_noise<B, DataSk, DataWant>(
        self,
        module: &Module<B>,
        sk: &GLWESecretPrepared<DataSk, B>,
        pt_want: &ScalarZnx<DataWant>,
        max_noise: f64,
    ) where
        DataSk: DataRef,
        DataWant: DataRef,
        Module<B>: VecZnxDftAllocBytes
            + VecZnxBigAllocBytes
            + DFT<B>
            + SvpApplyInplace<B>
            + IDFTConsume<B>
            + VecZnxBigAddInplace<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxNormalizeTmpBytes
            + VecZnxSubScalarInplace,
        B: Backend + TakeVecZnxDftImpl<B> + TakeVecZnxBigImpl<B> + ScratchOwnedAllocImpl<B> + ScratchOwnedBorrowImpl<B>,
    {
        let digits: usize = self.digits();
        let basek: usize = self.basek();
        let k: usize = self.k();

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(GLWECiphertext::decrypt_scratch_space(module, basek, k));
        let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(self.n(), basek, k);

        (0..self.rank_in()).for_each(|col_i| {
            (0..self.rows()).for_each(|row_i| {
                self.at(row_i, col_i)
                    .decrypt(module, &mut pt, sk, scratch.borrow());

                module.vec_znx_sub_scalar_inplace(
                    &mut pt.data,
                    0,
                    (digits - 1) + row_i * digits,
                    pt_want,
                    col_i,
                );

                let noise_have: f64 = pt.data.std(basek, 0).log2();

                assert!(
                    noise_have <= max_noise,
                    "noise_have: {} > max_noise: {}",
                    noise_have,
                    max_noise
                );

                pt.data.zero();
            });
        });
    }
}
