use backend::hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxNormalizeInplace, VecZnxSubABInplace},
    layouts::{Backend, DataRef, Module, ScratchOwned},
    oep::{ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeVecZnxBigImpl, TakeVecZnxDftImpl},
};

use crate::{
    layouts::GLWEPlaintext,
    layouts::prepared::GLWESecretPrepared,
    layouts::{GLWECiphertext, Infos},
    trait_families::GLWEDecryptFamily,
};

impl<D: DataRef> GLWECiphertext<D> {
    pub fn assert_noise<B: Backend, DataSk, DataPt>(
        &self,
        module: &Module<B>,
        sk_exec: &GLWESecretPrepared<DataSk, B>,
        pt_want: &GLWEPlaintext<DataPt>,
        max_noise: f64,
    ) where
        DataSk: DataRef,
        DataPt: DataRef,
        Module<B>: GLWEDecryptFamily<B> + VecZnxSubABInplace + VecZnxNormalizeInplace<B>,
        B: TakeVecZnxDftImpl<B> + TakeVecZnxBigImpl<B> + ScratchOwnedAllocImpl<B> + ScratchOwnedBorrowImpl<B>,
    {
        let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(self.n(), self.basek(), self.k());

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(GLWECiphertext::decrypt_scratch_space(
            module,
            self.n(),
            self.basek(),
            self.k(),
        ));

        self.decrypt(module, &mut pt_have, &sk_exec, scratch.borrow());

        module.vec_znx_sub_ab_inplace(&mut pt_have.data, 0, &pt_want.data, 0);
        module.vec_znx_normalize_inplace(self.basek(), &mut pt_have.data, 0, scratch.borrow());

        let noise_have: f64 = pt_have.data.std(self.basek(), 0).log2();
        assert!(noise_have <= max_noise, "{} {}", noise_have, max_noise);
    }
}
