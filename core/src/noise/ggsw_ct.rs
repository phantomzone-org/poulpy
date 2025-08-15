use backend::hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAddScalarInplace, VecZnxBigAlloc, VecZnxBigNormalize, VecZnxDftAlloc,
        VecZnxDftToVecZnxBigTmpA, VecZnxNormalizeTmpBytes, VecZnxSubABInplace, ZnxZero,
    },
    layouts::{Backend, DataRef, Module, ScalarZnx, ScratchOwned, VecZnxBig, VecZnxDft},
    oep::{ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeVecZnxBigImpl, TakeVecZnxDftImpl},
};

use crate::{
    layouts::{GGSWCiphertext, GLWECiphertext, GLWEPlaintext, Infos, prepared::GLWESecretPrepared},
    trait_families::GGSWAssertNoiseFamily,
};

impl<D: DataRef> GGSWCiphertext<D> {
    pub fn assert_noise<B: Backend, DataSk, DataScalar, F>(
        &self,
        module: &Module<B>,
        sk_exec: &GLWESecretPrepared<DataSk, B>,
        pt_want: &ScalarZnx<DataScalar>,
        max_noise: F,
    ) where
        DataSk: DataRef,
        DataScalar: DataRef,
        Module<B>: GGSWAssertNoiseFamily<B> + VecZnxAddScalarInplace + VecZnxSubABInplace,
        B: TakeVecZnxDftImpl<B> + TakeVecZnxBigImpl<B> + ScratchOwnedAllocImpl<B> + ScratchOwnedBorrowImpl<B>,
        F: Fn(usize) -> f64,
    {
        let basek: usize = self.basek();
        let k: usize = self.k();
        let digits: usize = self.digits();

        let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(self.n(), basek, k);
        let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(self.n(), basek, k);
        let mut pt_dft: VecZnxDft<Vec<u8>, B> = module.vec_znx_dft_alloc(self.n(), 1, self.size());
        let mut pt_big: VecZnxBig<Vec<u8>, B> = module.vec_znx_big_alloc(self.n(), 1, self.size());

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
            GLWECiphertext::decrypt_scratch_space(module, self.n(), basek, k) | module.vec_znx_normalize_tmp_bytes(self.n()),
        );

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

                let std_pt: f64 = pt_have.data.std(basek, 0).log2();
                let noise: f64 = max_noise(col_j);
                println!("{} {}", std_pt, noise);
                assert!(std_pt <= noise, "{} > {}", std_pt, noise);

                pt.data.zero();
            });
        });
    }
}

impl<D: DataRef> GGSWCiphertext<D> {
    pub fn print_noise<B: Backend, DataSk, DataScalar>(
        &self,
        module: &Module<B>,
        sk_exec: &GLWESecretPrepared<DataSk, B>,
        pt_want: &ScalarZnx<DataScalar>,
    ) where
        DataSk: DataRef,
        DataScalar: DataRef,
        Module<B>: GGSWAssertNoiseFamily<B> + VecZnxAddScalarInplace + VecZnxSubABInplace,
        B: TakeVecZnxDftImpl<B> + TakeVecZnxBigImpl<B> + ScratchOwnedAllocImpl<B> + ScratchOwnedBorrowImpl<B>,
    {
        let basek: usize = self.basek();
        let k: usize = self.k();
        let digits: usize = self.digits();

        let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(self.n(), basek, k);
        let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(self.n(), basek, k);
        let mut pt_dft: VecZnxDft<Vec<u8>, B> = module.vec_znx_dft_alloc(self.n(), 1, self.size());
        let mut pt_big: VecZnxBig<Vec<u8>, B> = module.vec_znx_big_alloc(self.n(), 1, self.size());

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
            GLWECiphertext::decrypt_scratch_space(module, self.n(), basek, k) | module.vec_znx_normalize_tmp_bytes(module.n()),
        );

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

                let std_pt: f64 = pt_have.data.std(basek, 0).log2();
                println!("{}", std_pt);
                pt.data.zero();
            });
        });
    }
}
