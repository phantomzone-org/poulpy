use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, ScratchTakeBasic, VecZnxSubScalarInplace},
    layouts::{Backend, DataRef, Module, ScalarZnx, ScalarZnxToRef, Scratch, ScratchOwned, ZnxZero},
    oep::{ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, VecZnxSubScalarInplaceImpl},
};

use crate::decryption::GLWEDecrypt;
use crate::layouts::{
    GGLWE, GGLWEInfos, GGLWEToRef, GLWEPlaintext, LWEInfos,
    prepared::{GLWESecretPrepared, GLWESecretPreparedToRef},
};

impl<D: DataRef> GGLWE<D> {
    pub fn assert_noise<M, BE, DataSk, DataWant>(
        &self,
        module: &M,
        sk_prepared: &GLWESecretPrepared<DataSk, BE>,
        pt_want: &ScalarZnx<DataWant>,
        max_noise: f64,
    ) where
        DataSk: DataRef,
        DataWant: DataRef,
        M: GGLWENoise<BE>,
        BE: Backend
            + ScratchOwnedAllocImpl<BE>
            + ScratchOwnedBorrowImpl<BE>
            + ScratchOwnedBorrow<BE>
            + VecZnxSubScalarInplaceImpl<BE>,
    {
        module.gglwe_assert_noise(self, sk_prepared, pt_want, max_noise);
    }

    // pub fn assert_noise<B, DataSk, DataWant>(
    //     &self,
    //     module: &Module<B>,
    //     sk: &GLWESecretPrepared<DataSk, B>,
    //     pt_want: &ScalarZnx<DataWant>,
    //     max_noise: f64,
    // ) where
    //     DataSk: DataRef,
    //     DataWant: DataRef,
    //     Module<B>: VecZnxDftBytesOf
    //         + VecZnxBigBytesOf
    //         + VecZnxDftApply<B>
    //         + SvpApplyDftToDftInplace<B>
    //         + VecZnxIdftApplyConsume<B>
    //         + VecZnxBigAddInplace<B>
    //         + VecZnxBigAddSmallInplace<B>
    //         + VecZnxBigNormalize<B>
    //         + VecZnxNormalizeTmpBytes
    //         + VecZnxSubScalarInplace,
    //     B: Backend + ScratchOwnedAllocImpl<B> + ScratchOwnedBorrowImpl<B>,
    // {
    //     let dsize: usize = self.dsize().into();
    //     let base2k: usize = self.base2k().into();

    //     let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(GLWE::decrypt_tmp_bytes(module, self));
    //     let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(module, self);

    //     (0..self.rank_in().into()).for_each(|col_i| {
    //         (0..self.dnum().into()).for_each(|row_i| {
    //             self.at(row_i, col_i)
    //                 .decrypt(module, &mut pt, sk, scratch.borrow());

    //             module.vec_znx_sub_scalar_inplace(&mut pt.data, 0, (dsize - 1) + row_i * dsize, pt_want, col_i);

    //             let noise_have: f64 = pt.data.std(base2k, 0).log2();

    //             println!("noise_have: {noise_have}");

    //             assert!(
    //                 noise_have <= max_noise,
    //                 "noise_have: {noise_have} > max_noise: {max_noise}"
    //             );

    //             pt.data.zero();
    //         });
    //     });
    // }
}

pub trait GGLWENoise<BE: Backend> {
    fn gglwe_assert_noise<R, S, P>(&self, res: &R, sk_prepared: &S, pt_want: &P, max_noise: f64)
    where
        R: GGLWEToRef,
        S: GLWESecretPreparedToRef<BE>,
        P: ScalarZnxToRef,
        BE: ScratchOwnedAllocImpl<BE> + ScratchOwnedBorrowImpl<BE> + ScratchOwnedBorrow<BE> + VecZnxSubScalarInplaceImpl<BE>;
}

impl<BE: Backend> GGLWENoise<BE> for Module<BE>
where
    Module<BE>: GLWEDecrypt<BE>,
    Scratch<BE>: ScratchTakeBasic
        + ScratchOwnedAllocImpl<BE>
        + ScratchOwnedBorrowImpl<BE>
        + ScratchOwnedBorrowImpl<BE>
        + ScratchOwnedBorrow<BE>,
{
    fn gglwe_assert_noise<R, S, P>(&self, res: &R, sk_prepared: &S, pt_want: &P, max_noise: f64)
    where
        R: GGLWEToRef,
        S: GLWESecretPreparedToRef<BE>,
        P: ScalarZnxToRef,
        BE: ScratchOwnedAllocImpl<BE> + ScratchOwnedBorrowImpl<BE> + ScratchOwnedBorrow<BE> + VecZnxSubScalarInplaceImpl<BE>,
    {
        let res: &GGLWE<&[u8]> = &res.to_ref();

        let dsize: usize = res.dsize().into();
        let base2k: usize = res.base2k().into();

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(self.glwe_decrypt_tmp_bytes(res));
        let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(self, res);

        (0..res.rank_in().into()).for_each(|col_i| {
            (0..res.dnum().into()).for_each(|row_i| {
                self.glwe_decrypt(
                    &res.at(row_i, col_i),
                    &mut pt,
                    sk_prepared,
                    scratch.borrow(),
                );

                self.vec_znx_sub_scalar_inplace(&mut pt.data, 0, (dsize - 1) + row_i * dsize, pt_want, col_i);

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
