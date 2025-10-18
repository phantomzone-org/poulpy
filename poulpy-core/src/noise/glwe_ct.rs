use poulpy_hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, ScratchTakeBasic, SvpApplyDftToDftInplace, VecZnxBigAddInplace,
        VecZnxBigAddSmallInplace, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxDftApply, VecZnxDftBytesOf, VecZnxIdftApplyConsume,
        VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxSubInplace,
    },
    layouts::{Backend, DataRef, Module, Scratch, ScratchOwned},
    oep::{ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl},
};

use crate::{
    decryption::GLWEDecryption,
    layouts::{
        GLWE, GLWEPlaintext, GLWEPlaintextToRef, GLWEToRef, LWEInfos,
        prepared::{GLWESecretPrepared, GLWESecretPreparedToRef},
    },
};

impl<D: DataRef> GLWE<D> {
    pub fn noise<M, S, P, BE: Backend>(&self, module: &M, sk_prepared: &S, pt_want: &P, scratch: &mut Scratch<BE>) -> f64
    where
        M: GLWENoise<BE>,
        S: GLWESecretPreparedToRef<BE>,
        P: GLWEPlaintextToRef,
    {
        module.glwe_noise(self, sk_prepared, pt_want, scratch)
    }
    // pub fn noise<B, DataSk, DataPt>(
    //     &self,
    //     module: &Module<B>,
    //     sk_prepared: &GLWESecretPrepared<DataSk, B>,
    //     pt_want: &GLWEPlaintext<DataPt>,
    //     scratch: &mut Scratch<B>,
    // ) -> f64
    // where
    //     DataSk: DataRef,
    //     DataPt: DataRef,
    //     B: Backend,
    //     Module<B>: VecZnxDftApply<B>
    //         + VecZnxSubInplace
    //         + VecZnxNormalizeInplace<B>
    //         + SvpApplyDftToDftInplace<B>
    //         + VecZnxIdftApplyConsume<B>
    //         + VecZnxBigAddInplace<B>
    //         + VecZnxBigAddSmallInplace<B>
    //         + VecZnxBigNormalize<B>,
    //     Scratch<B>:,
    // {
    //     let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(module, self);
    //     self.decrypt(module, &mut pt_have, sk_prepared, scratch);
    //     module.vec_znx_sub_inplace(&mut pt_have.data, 0, &pt_want.data, 0);
    //     module.vec_znx_normalize_inplace(self.base2k().into(), &mut pt_have.data, 0, scratch);
    //     pt_have.data.std(self.base2k().into(), 0).log2()
    // }

    pub fn assert_noise<M, BE, DataSk, DataPt>(
        &self,
        module: &M,
        sk_prepared: &GLWESecretPrepared<DataSk, BE>,
        pt_want: &GLWEPlaintext<DataPt>,
        max_noise: f64,
    ) where
        DataSk: DataRef,
        DataPt: DataRef,
        M: GLWENoise<BE>,
        BE: Backend + ScratchOwnedAllocImpl<BE> + ScratchOwnedBorrowImpl<BE> + ScratchOwnedBorrow<BE>,
    {
        module.glwe_assert_noise(self, sk_prepared, pt_want, max_noise);
    }

    // pub fn assert_noise<B, DataSk, DataPt>(
    //     &self,
    //     module: &Module<B>,
    //     sk_prepared: &GLWESecretPrepared<DataSk, B>,
    //     pt_want: &GLWEPlaintext<DataPt>,
    //     max_noise: f64,
    // ) where
    //     DataSk: DataRef,
    //     DataPt: DataRef,
    //     Module<B>: VecZnxDftBytesOf
    //         + VecZnxBigBytesOf
    //         + VecZnxDftApply<B>
    //         + SvpApplyDftToDftInplace<B>
    //         + VecZnxIdftApplyConsume<B>
    //         + VecZnxBigAddInplace<B>
    //         + VecZnxBigAddSmallInplace<B>
    //         + VecZnxBigNormalize<B>
    //         + VecZnxNormalizeTmpBytes
    //         + VecZnxSubInplace
    //         + VecZnxNormalizeInplace<B>,
    //     B: Backend + ScratchOwnedAllocImpl<B> + ScratchOwnedBorrowImpl<B>,
    // {
    //     let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(GLWE::decrypt_tmp_bytes(module, self));
    //     let noise_have: f64 = self.noise(module, sk_prepared, pt_want, scratch.borrow());
    //     assert!(noise_have <= max_noise, "{noise_have} {max_noise}");
    // }
}

pub trait GLWENoise<BE: Backend> {
    fn glwe_noise<R, S, P>(&self, res: &R, sk_prepared: &S, pt_want: &P, scratch: &mut Scratch<BE>) -> f64
    where
        R: GLWEToRef,
        S: GLWESecretPreparedToRef<BE>,
        P: GLWEPlaintextToRef;

    fn glwe_assert_noise<R, S, P>(&self, res: &R, sk_prepared: &S, pt_want: &P, max_noise: f64)
    where
        R: GLWEToRef,
        S: GLWESecretPreparedToRef<BE>,
        P: GLWEPlaintextToRef,
        BE: ScratchOwnedAllocImpl<BE> + ScratchOwnedBorrowImpl<BE> + ScratchOwnedBorrow<BE>;
}

impl<BE: Backend> GLWENoise<BE> for Module<BE>
where
    Module<BE>: VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VecZnxDftApply<BE>
        + SvpApplyDftToDftInplace<BE>
        + VecZnxIdftApplyConsume<BE>
        + VecZnxBigAddInplace<BE>
        + VecZnxBigAddSmallInplace<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxSubInplace
        + VecZnxNormalizeInplace<BE>
        + GLWEDecryption<BE>,
    Scratch<BE>: ScratchTakeBasic
        + ScratchOwnedAllocImpl<BE>
        + ScratchOwnedBorrowImpl<BE>
        + ScratchOwnedBorrowImpl<BE>
        + ScratchOwnedBorrow<BE>,
{
    fn glwe_noise<R, S, P>(&self, res: &R, sk_prepared: &S, pt_want: &P, scratch: &mut Scratch<BE>) -> f64
    where
        R: GLWEToRef,
        S: GLWESecretPreparedToRef<BE>,
        P: GLWEPlaintextToRef,
    {
        let res_ref: &GLWE<&[u8]> = &res.to_ref();

        let pt_want: &GLWEPlaintext<&[u8]> = &pt_want.to_ref();

        let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(self, res_ref);
        self.glwe_decrypt(res, &mut pt_have, sk_prepared, scratch);
        self.vec_znx_sub_inplace(&mut pt_have.data, 0, &pt_want.data, 0);
        self.vec_znx_normalize_inplace(res_ref.base2k().into(), &mut pt_have.data, 0, scratch);
        pt_have.data.std(res_ref.base2k().into(), 0).log2()
    }

    fn glwe_assert_noise<R, S, P>(&self, res: &R, sk_prepared: &S, pt_want: &P, max_noise: f64)
    where
        R: GLWEToRef,
        S: GLWESecretPreparedToRef<BE>,
        P: GLWEPlaintextToRef,
        BE: ScratchOwnedAllocImpl<BE> + ScratchOwnedBorrowImpl<BE> + ScratchOwnedBorrow<BE>,
    {
        let res: &GLWE<&[u8]> = &res.to_ref();
        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(self.glwe_decrypt_tmp_bytes(res));
        let noise_have: f64 = self.glwe_noise(res, sk_prepared, pt_want, scratch.borrow());
        assert!(noise_have <= max_noise, "{noise_have} {max_noise}");
    }
}
