use poulpy_hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyDftToDftInplace, VecZnxBigAddInplace, VecZnxBigAddSmallInplace,
        VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxDftApply, VecZnxDftBytesOf, VecZnxIdftApplyConsume, VecZnxNormalizeInplace,
        VecZnxNormalizeTmpBytes, VecZnxSubInplace,
    },
    layouts::{Backend, DataRef, Module, Scratch, ScratchOwned},
    oep::{ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl},
};

use crate::layouts::{GLWE, GLWEPlaintext, LWEInfos, prepared::GLWESecretPrepared};

impl<D: DataRef> GLWE<D> {
    pub fn noise<B, DataSk, DataPt>(
        &self,
        module: &Module<B>,
        sk_prepared: &GLWESecretPrepared<DataSk, B>,
        pt_want: &GLWEPlaintext<DataPt>,
        scratch: &mut Scratch<B>,
    ) -> f64
    where
        DataSk: DataRef,
        DataPt: DataRef,
        B: Backend,
        Module<B>: VecZnxDftApply<B>
            + VecZnxSubInplace
            + VecZnxNormalizeInplace<B>
            + SvpApplyDftToDftInplace<B>
            + VecZnxIdftApplyConsume<B>
            + VecZnxBigAddInplace<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>,
        Scratch<B>:,
    {
        let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(module, self);
        self.decrypt(module, &mut pt_have, sk_prepared, scratch);
        module.vec_znx_sub_inplace(&mut pt_have.data, 0, &pt_want.data, 0);
        module.vec_znx_normalize_inplace(self.base2k().into(), &mut pt_have.data, 0, scratch);
        pt_have.data.std(self.base2k().into(), 0).log2()
    }

    pub fn assert_noise<B, DataSk, DataPt>(
        &self,
        module: &Module<B>,
        sk_prepared: &GLWESecretPrepared<DataSk, B>,
        pt_want: &GLWEPlaintext<DataPt>,
        max_noise: f64,
    ) where
        DataSk: DataRef,
        DataPt: DataRef,
        Module<B>: VecZnxDftBytesOf
            + VecZnxBigBytesOf
            + VecZnxDftApply<B>
            + SvpApplyDftToDftInplace<B>
            + VecZnxIdftApplyConsume<B>
            + VecZnxBigAddInplace<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxNormalizeTmpBytes
            + VecZnxSubInplace
            + VecZnxNormalizeInplace<B>,
        B: Backend + ScratchOwnedAllocImpl<B> + ScratchOwnedBorrowImpl<B>,
    {
        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(GLWE::decrypt_tmp_bytes(module, self));
        let noise_have: f64 = self.noise(module, sk_prepared, pt_want, scratch.borrow());
        assert!(noise_have <= max_noise, "{noise_have} {max_noise}");
    }
}
