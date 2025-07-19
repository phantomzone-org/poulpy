use backend::{
    Backend, DataViewMut, Module, ScalarZnxDftPrepOps, Scratch, VecZnxBig, VecZnxBigAlloc, VecZnxBigOps, VecZnxBigScratch, VecZnxDftAlloc, VecZnxDftOps
};

use crate::{FourierGLWECiphertext, FourierGLWESecret, GLWEPlaintext, Infos};

impl<B: Backend> FourierGLWECiphertext<Vec<u8>, B> {
    pub fn decrypt_scratch_space(module: &Module<B>, basek: usize, k: usize) -> usize where Module<B>: VecZnxDftAlloc<B> + VecZnxDftOps<B>{
        let size: usize = k.div_ceil(basek);
        (module.vec_znx_big_normalize_tmp_bytes()
            | module.bytes_of_vec_znx_dft(1, size)
            | (module.bytes_of_vec_znx_big(1, size) + module.vec_znx_idft_tmp_bytes()))
            + module.bytes_of_vec_znx_big(1, size)
    }
}

impl<DataSelf: AsRef<[u8]>, B: Backend> FourierGLWECiphertext<DataSelf, B> {
    pub fn decrypt<DataPt: AsRef<[u8]> + AsMut<[u8]>, DataSk: AsRef<[u8]>>(
        &self,
        module: &Module<B>,
        pt: &mut GLWEPlaintext<DataPt>,
        sk: &FourierGLWESecret<DataSk, B>,
        scratch: &mut Scratch,
    ) where Module<B>: ScalarZnxDftPrepOps<B> + VecZnxDftAlloc<B> + VecZnxDftOps<B> + VecZnxBigOps<B> {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.rank(), sk.rank());
            assert_eq!(self.n(), module.n());
            assert_eq!(pt.n(), module.n());
            assert_eq!(sk.n(), module.n());
        }

        let cols = self.rank() + 1;

        let (mut pt_big, scratch_1) = scratch.tmp_vec_znx_big(module, 1, self.size()); // TODO optimize size when pt << ct

        pt_big.data_mut().fill(0);

        {
            (1..cols).for_each(|i| {
                let (mut ci_dft, _) = scratch_1.tmp_vec_znx_dft(module, 1, self.size()); // TODO optimize size when pt << ct
                module.svp_apply(&mut ci_dft, 0, &sk.data, i - 1, &self.data, i);
                let ci_big: VecZnxBig<&mut [u8], B> = module.vec_znx_idft_consume(ci_dft);
                module.vec_znx_big_add_inplace(&mut pt_big, 0, &ci_big, 0);
            });
        }

        {
            let (mut c0_big, scratch_2) = scratch_1.tmp_vec_znx_big(module, 1, self.size());
            // c0_big = (a * s) + (-a * s + m + e) = BIG(m + e)
            module.vec_znx_idft(&mut c0_big, 0, &self.data, 0, scratch_2);
            module.vec_znx_big_add_inplace(&mut pt_big, 0, &c0_big, 0);
        }

        // pt = norm(BIG(m + e))
        module.vec_znx_big_normalize(self.basek(), &mut pt.data, 0, &mut pt_big, 0, scratch_1);

        pt.basek = self.basek();
        pt.k = pt.k().min(self.k());
    }
}
