use backend::{
    Backend, Module, ScalarZnx, ScratchOwned, Stats, VecZnxBig, VecZnxBigAlloc, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
    VecZnxDft, VecZnxDftAlloc, VecZnxDftToVecZnxBigTmpA, VecZnxOps, VecZnxScratch, ZnxZero,
};

use crate::{GGSWCiphertext, GLWECiphertext, GLWEDecryptFamily, GLWEPlaintext, GLWESecretExec, Infos};

pub trait GGSWAssertNoiseFamily<B: Backend> = GLWEDecryptFamily<B>
    + VecZnxBigAlloc<B>
    + VecZnxDftAlloc<B>
    + VecZnxBigNormalizeTmpBytes
    + VecZnxBigNormalize<B>
    + VecZnxDftToVecZnxBigTmpA<B>;

impl<D: AsRef<[u8]>> GGSWCiphertext<D> {
    pub fn assert_noise<B: Backend, DataSk, DataScalar, F>(
        &self,
        module: &Module<B>,
        sk_exec: &GLWESecretExec<DataSk, B>,
        pt_want: &ScalarZnx<DataScalar>,
        max_noise: F,
    ) where
        DataSk: AsRef<[u8]>,
        DataScalar: AsRef<[u8]>,
        Module<B>: GGSWAssertNoiseFamily<B>,
        F: Fn(usize) -> f64,
    {
        let basek: usize = self.basek();
        let k: usize = self.k();
        let digits: usize = self.digits();

        let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(module, basek, k);
        let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(module, basek, k);
        let mut pt_dft: VecZnxDft<Vec<u8>, B> = module.vec_znx_dft_alloc(1, self.size());
        let mut pt_big: VecZnxBig<Vec<u8>, B> = module.vec_znx_big_alloc(1, self.size());

        let mut scratch: ScratchOwned =
            ScratchOwned::new(GLWECiphertext::decrypt_scratch_space(module, basek, k) | module.vec_znx_normalize_tmp_bytes());

        (0..self.rank() + 1).for_each(|col_j| {
            (0..self.rows()).for_each(|row_i| {
                module.vec_znx_add_scalar_inplace(&mut pt.data, 0, (digits - 1) + row_i * digits, pt_want, 0);

                // mul with sk[col_j-1]
                if col_j > 0 {
                    module.vec_znx_dft_from_vec_znx(1, 0, &mut pt_dft, 0, &pt.data, 0);
                    module.svp_apply_inplace(&mut pt_dft, 0, &sk_exec.data, col_j - 1);
                    module.vec_znx_dft_to_vec_znx_big_tmp_a(&mut pt_big, 0, &mut pt_dft, 0);
                    module.vec_znx_big_normalize(basek, &mut pt.data, 0, &pt_big, 0, scratch.borrow());
                }

                self.at(row_i, col_j)
                    .decrypt(module, &mut pt_have, &sk_exec, scratch.borrow());

                module.vec_znx_sub_ab_inplace(&mut pt_have.data, 0, &pt.data, 0);

                let std_pt: f64 = pt_have.data.std(0, basek) * (k as f64).exp2();
                let noise: f64 = max_noise(col_j);
                assert!(std_pt <= noise, "{} > {}", std_pt, noise);

                pt.data.zero();
            });
        });
    }
}
