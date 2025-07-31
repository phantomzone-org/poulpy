use backend::{Backend, Module, ScalarZnx, ScratchOwned, VecZnxStd, VecZnxSubScalarInplace, ZnxZero};

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
        Module<B>: GLWEDecryptFamily<B>,
    {
        let digits: usize = self.digits();
        let basek: usize = self.basek();
        let k: usize = self.k();

        let mut scratch: ScratchOwned = ScratchOwned::new(GLWECiphertext::decrypt_scratch_space(module, basek, k));
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
