use backend::hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAlloc, VecZnxStd, VecZnxSubScalarInplace, ZnxZero},
    layouts::{Backend, Module, ScalarZnx, ScratchOwned},
    oep::{ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeVecZnxBigImpl, TakeVecZnxDftImpl},
};

use crate::{GGLWECiphertext, GLWECiphertext, GLWEDecryptFamily, GLWEPlaintext, GLWESecretExec, Infos};

impl<DataIn: AsRef<[u8]>> GGLWECiphertext<DataIn> {
    pub fn assert_noise<B: Backend, DataSk, DataWant>(
        self,
        module: &Module<B>,
        sk: &GLWESecretExec<DataSk, B>,
        pt_want: &ScalarZnx<DataWant>,
        max_noise: f64,
    ) where
        DataSk: AsRef<[u8]>,
        DataWant: AsRef<[u8]>,
        Module<B>: GLWEDecryptFamily<B> + VecZnxStd + VecZnxAlloc + VecZnxSubScalarInplace,
        B: TakeVecZnxDftImpl<B> + TakeVecZnxBigImpl<B> + ScratchOwnedAllocImpl<B> + ScratchOwnedBorrowImpl<B>,
    {
        let digits: usize = self.digits();
        let basek: usize = self.basek();
        let k: usize = self.k();

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(GLWECiphertext::decrypt_scratch_space(module, basek, k));
        let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(module, basek, k);

        (0..self.rank_in()).for_each(|col_i| {
            (0..self.rows()).for_each(|row_i| {
                self.at(row_i, col_i)
                    .decrypt(&module, &mut pt, &sk, scratch.borrow());

                module.vec_znx_sub_scalar_inplace(
                    &mut pt.data,
                    0,
                    (digits - 1) + row_i * digits,
                    pt_want,
                    col_i,
                );

                let noise_have: f64 = module.vec_znx_std(basek, &pt.data, 0).log2();

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
