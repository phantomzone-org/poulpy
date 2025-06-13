use backend::{FFT64, Module, Scratch};

use crate::{FourierGLWECiphertext, GLWECiphertext, GLWESwitchingKey, Infos, ScratchCore};

impl FourierGLWECiphertext<Vec<u8>, FFT64> {
    pub fn keyswitch_scratch_space(
        module: &Module<FFT64>,
        basek: usize,
        k_out: usize,
        k_in: usize,
        k_ksk: usize,
        digits: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> usize {
        GLWECiphertext::bytes_of(module, basek, k_out, rank_out)
            + GLWECiphertext::keyswitch_from_fourier_scratch_space(module, basek, k_out, k_in, k_ksk, digits, rank_in, rank_out)
    }

    pub fn keyswitch_inplace_scratch_space(
        module: &Module<FFT64>,
        basek: usize,
        k_out: usize,
        k_ksk: usize,
        digits: usize,
        rank: usize,
    ) -> usize {
        Self::keyswitch_scratch_space(module, basek, k_out, k_out, k_ksk, digits, rank, rank)
    }
}

impl<DataSelf: AsMut<[u8]> + AsRef<[u8]>> FourierGLWECiphertext<DataSelf, FFT64> {
    pub fn keyswitch<DataLhs: AsRef<[u8]>, DataRhs: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        lhs: &FourierGLWECiphertext<DataLhs, FFT64>,
        rhs: &GLWESwitchingKey<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) {
        let (mut tmp_ct, scratch1) = scratch.tmp_glwe_ct(module, self.basek(), self.k(), self.rank());
        tmp_ct.keyswitch_from_fourier(module, lhs, rhs, scratch1);
        tmp_ct.dft(module, self);
    }

    pub fn keyswitch_inplace<DataRhs: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        rhs: &GLWESwitchingKey<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) {
        unsafe {
            let self_ptr: *mut FourierGLWECiphertext<DataSelf, FFT64> = self as *mut FourierGLWECiphertext<DataSelf, FFT64>;
            self.keyswitch(&module, &*self_ptr, rhs, scratch);
        }
    }
}
