use backend::{
    FFT64, Module, ScalarZnxDftOps, Scratch, VecZnxBig, VecZnxBigAlloc, VecZnxBigOps, VecZnxBigScratch, VecZnxDftAlloc,
    VecZnxDftOps, ZnxZero,
};

use crate::{FourierGLWECiphertext, FourierGLWESecret, GLWECiphertext, GLWEPlaintext, Infos, div_ceil};

impl FourierGLWECiphertext<Vec<u8>, FFT64> {
    pub fn decrypt_scratch_space(module: &Module<FFT64>, basek: usize, k: usize) -> usize {
        let size: usize = div_ceil(k, basek);
        (module.vec_znx_big_normalize_tmp_bytes()
            | module.bytes_of_vec_znx_dft(1, size)
            | (module.bytes_of_vec_znx_big(1, size) + module.vec_znx_idft_tmp_bytes()))
            + module.bytes_of_vec_znx_big(1, size)
    }
}

impl<DataSelf: AsRef<[u8]>> FourierGLWECiphertext<DataSelf, FFT64> {
    pub fn decrypt<DataPt: AsRef<[u8]> + AsMut<[u8]>, DataSk: AsRef<[u8]>>(
        &self,
        module: &Module<FFT64>,
        pt: &mut GLWEPlaintext<DataPt>,
        sk: &FourierGLWESecret<DataSk, FFT64>,
        scratch: &mut Scratch,
    ) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.rank(), sk.rank());
            assert_eq!(self.n(), module.n());
            assert_eq!(pt.n(), module.n());
            assert_eq!(sk.n(), module.n());
        }

        let cols = self.rank() + 1;

        let (mut pt_big, scratch_1) = scratch.tmp_vec_znx_big(module, 1, self.size()); // TODO optimize size when pt << ct
        pt_big.zero();

        {
            (1..cols).for_each(|i| {
                let (mut ci_dft, _) = scratch_1.tmp_vec_znx_dft(module, 1, self.size()); // TODO optimize size when pt << ct
                module.svp_apply(&mut ci_dft, 0, &sk.data, i - 1, &self.data, i);
                let ci_big: VecZnxBig<&mut [u8], FFT64> = module.vec_znx_idft_consume(ci_dft);
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

    #[allow(dead_code)]
    pub(crate) fn idft<DataRes: AsRef<[u8]> + AsMut<[u8]>>(
        &self,
        module: &Module<FFT64>,
        res: &mut GLWECiphertext<DataRes>,
        scratch: &mut Scratch,
    ) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.rank(), res.rank());
            assert_eq!(self.basek(), res.basek())
        }

        let min_size: usize = self.size().min(res.size());

        let (mut res_big, scratch1) = scratch.tmp_vec_znx_big(module, 1, min_size);

        (0..self.rank() + 1).for_each(|i| {
            module.vec_znx_idft(&mut res_big, 0, &self.data, i, scratch1);
            module.vec_znx_big_normalize(self.basek(), &mut res.data, i, &res_big, 0, scratch1);
        });
    }
}
