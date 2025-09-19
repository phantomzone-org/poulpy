use poulpy_hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyDftToDftInplace, VecZnxBigAddInplace, VecZnxBigAddSmallInplace,
        VecZnxBigAllocBytes, VecZnxBigNormalize, VecZnxDftAllocBytes, VecZnxDftApply, VecZnxIdftApplyConsume,
        VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxSubABInplace,
    },
    layouts::{Backend, DataRef, Module, ScratchOwned},
    oep::{ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeVecZnxBigImpl, TakeVecZnxDftImpl},
};

use crate::layouts::{GLWECiphertext, GLWEPlaintext, LWEInfos, prepared::GLWESecretPrepared};

impl<D: DataRef> GLWECiphertext<D> {
    pub fn assert_noise<B, DataSk, DataPt>(
        &self,
        module: &Module<B>,
        sk_prepared: &GLWESecretPrepared<DataSk, B>,
        pt_want: &GLWEPlaintext<DataPt>,
        max_noise: f64,
    ) where
        DataSk: DataRef,
        DataPt: DataRef,
        Module<B>: VecZnxDftAllocBytes
            + VecZnxBigAllocBytes
            + VecZnxDftApply<B>
            + SvpApplyDftToDftInplace<B>
            + VecZnxIdftApplyConsume<B>
            + VecZnxBigAddInplace<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxNormalizeTmpBytes
            + VecZnxSubABInplace
            + VecZnxNormalizeInplace<B>,
        B: Backend + TakeVecZnxDftImpl<B> + TakeVecZnxBigImpl<B> + ScratchOwnedAllocImpl<B> + ScratchOwnedBorrowImpl<B>,
    {
        let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(self);

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(GLWECiphertext::decrypt_scratch_space(module, self));

        self.decrypt(module, &mut pt_have, sk_prepared, scratch.borrow());

        module.vec_znx_sub_ab_inplace(&mut pt_have.data, 0, &pt_want.data, 0);
        module.vec_znx_normalize_inplace(self.base2k().into(), &mut pt_have.data, 0, scratch.borrow());

        let noise_have: f64 = pt_have.data.std(self.base2k().into(), 0).log2();
        assert!(noise_have <= max_noise, "{noise_have} {max_noise}");
    }
}
