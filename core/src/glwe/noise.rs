use backend::{Backend, Module, ScratchOwned, Stats, VecZnxOps};

use crate::{GLWECiphertext, GLWEDecryptFamily, GLWEPlaintext, GLWESecretExec, Infos};

impl<D: AsRef<[u8]>> GLWECiphertext<D> {
    pub fn assert_noise<B: Backend, DataSk, DataPt>(
        &self,
        module: &Module<B>,
        sk_exec: &GLWESecretExec<DataSk, B>,
        pt_want: &GLWEPlaintext<DataPt>,
        max_noise: f64,
    ) where
        DataSk: AsRef<[u8]>,
        DataPt: AsRef<[u8]>,
        Module<B>: GLWEDecryptFamily<B>,
    {
        let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(module, self.basek(), self.k());

        let mut scratch: ScratchOwned = ScratchOwned::new(GLWECiphertext::decrypt_scratch_space(
            module,
            self.basek(),
            self.k(),
        ));

        self.decrypt(module, &mut pt_have, &sk_exec, scratch.borrow());

        module.vec_znx_sub_ab_inplace(&mut pt_have.data, 0, &pt_want.data, 0);
        module.vec_znx_normalize_inplace(self.basek(), &mut pt_have.data, 0, scratch.borrow());

        let noise_have: f64 = pt_have.data.std(0, self.basek()).log2();
        assert!(noise_have <= max_noise, "{} {}", noise_have, max_noise);
    }
}
