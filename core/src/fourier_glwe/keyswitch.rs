use backend::{Backend, Module, Scratch, VecZnxDftAlloc};

use crate::{FourierGLWECiphertext, GLWECiphertext, GLWESwitchingKeyPrep, Infos, ScratchCore};

impl<B: Backend> FourierGLWECiphertext<Vec<u8>, B> {
    pub fn keyswitch_scratch_space(
        module: &Module<B>,
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
        module: &Module<B>,
        basek: usize,
        k_out: usize,
        k_ksk: usize,
        digits: usize,
        rank: usize,
    ) -> usize {
        Self::keyswitch_scratch_space(module, basek, k_out, k_out, k_ksk, digits, rank, rank)
    }
}

impl<DataSelf: AsMut<[u8]> + AsRef<[u8]>, B: Backend> FourierGLWECiphertext<DataSelf, B> {
    pub fn keyswitch<DataLhs: AsRef<[u8]>, DataRhs: AsRef<[u8]>>(
        &mut self,
        module: &Module<B>,
        lhs: &FourierGLWECiphertext<DataLhs, B>,
        rhs: &GLWESwitchingKeyPrep<DataRhs, B>,
        scratch: &mut Scratch,
    ) where
        Module<B>: VecZnxDftAlloc<B>,
    {
        let (mut tmp_ct, scratch1) = scratch.tmp_glwe_ct(module, self.basek(), self.k(), self.rank());
        tmp_ct.keyswitch_from_fourier(module, lhs, rhs, scratch1);
        tmp_ct.dft(module, self);
    }

    pub fn keyswitch_inplace<DataRhs: AsRef<[u8]>>(
        &mut self,
        module: &Module<B>,
        rhs: &GLWESwitchingKeyPrep<DataRhs, B>,
        scratch: &mut Scratch,
    ) {
        unsafe {
            let self_ptr: *mut FourierGLWECiphertext<DataSelf, B> = self as *mut FourierGLWECiphertext<DataSelf, B>;
            self.keyswitch(&module, &*self_ptr, rhs, scratch);
        }
    }
}
