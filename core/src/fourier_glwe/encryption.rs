use backend::{FFT64, Module, Scratch, VecZnxAlloc, VecZnxBigScratch, VecZnxDftOps};
use sampling::source::Source;

use crate::{FourierGLWECiphertext, FourierGLWESecret, GLWECiphertext, Infos, ScratchCore};

impl FourierGLWECiphertext<Vec<u8>, FFT64> {
    #[allow(dead_code)]
    pub(crate) fn idft_scratch_space(module: &Module<FFT64>, basek: usize, k: usize) -> usize {
        module.bytes_of_vec_znx(1, k.div_ceil(basek))
            + (module.vec_znx_big_normalize_tmp_bytes() | module.vec_znx_idft_tmp_bytes())
    }

    pub fn encrypt_sk_scratch_space(module: &Module<FFT64>, basek: usize, k: usize, rank: usize) -> usize {
        module.bytes_of_vec_znx(rank + 1, k.div_ceil(basek)) + GLWECiphertext::encrypt_sk_scratch_space(module, basek, k)
    }
}

impl<DataSelf: AsMut<[u8]> + AsRef<[u8]>> FourierGLWECiphertext<DataSelf, FFT64> {
    pub fn encrypt_zero_sk<DataSk: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        sk: &FourierGLWESecret<DataSk, FFT64>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch,
    ) {
        let (mut tmp_ct, scratch1) = scratch.tmp_glwe_ct(module, self.basek(), self.k(), self.rank());
        tmp_ct.encrypt_zero_sk(module, sk, source_xa, source_xe, sigma, scratch1);
        tmp_ct.dft(module, self);
    }
}
