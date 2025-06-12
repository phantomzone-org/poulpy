use backend::{FFT64, Module, Scratch, VecZnxOps};

use crate::{AutomorphismKey, GLWECiphertext};

impl GLWECiphertext<Vec<u8>> {
    pub fn automorphism_scratch_space(
        module: &Module<FFT64>,
        basek: usize,
        k_out: usize,
        k_in: usize,
        k_ksk: usize,
        digits: usize,
        rank: usize,
    ) -> usize {
        Self::keyswitch_scratch_space(module, basek, k_out, k_in, k_ksk, digits, rank, rank)
    }

    pub fn automorphism_inplace_scratch_space(
        module: &Module<FFT64>,
        basek: usize,
        k_out: usize,
        k_ksk: usize,
        digits: usize,
        rank: usize,
    ) -> usize {
        Self::keyswitch_inplace_scratch_space(module, basek, k_out, k_ksk, digits, rank)
    }
}

impl<DataSelf: AsRef<[u8]> + AsMut<[u8]>> GLWECiphertext<DataSelf> {
    pub fn automorphism<DataLhs: AsRef<[u8]>, DataRhs: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        lhs: &GLWECiphertext<DataLhs>,
        rhs: &AutomorphismKey<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) {
        self.keyswitch(module, lhs, &rhs.key, scratch);
        (0..self.rank() + 1).for_each(|i| {
            module.vec_znx_automorphism_inplace(rhs.p(), &mut self.data, i);
        })
    }

    pub fn automorphism_inplace<DataRhs: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        rhs: &AutomorphismKey<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) {
        self.keyswitch_inplace(module, &rhs.key, scratch);
        (0..self.rank() + 1).for_each(|i| {
            module.vec_znx_automorphism_inplace(rhs.p(), &mut self.data, i);
        })
    }

    pub fn automorphism_add<DataLhs: AsRef<[u8]>, DataRhs: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        lhs: &GLWECiphertext<DataLhs>,
        rhs: &AutomorphismKey<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) {
        Self::keyswitch_private::<_, _, 1>(self, rhs.p(), module, lhs, &rhs.key, scratch);
    }

    pub fn automorphism_add_inplace<DataRhs: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        rhs: &AutomorphismKey<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) {
        unsafe {
            let self_ptr: *mut GLWECiphertext<DataSelf> = self as *mut GLWECiphertext<DataSelf>;
            Self::keyswitch_private::<_, _, 1>(self, rhs.p(), module, &*self_ptr, &rhs.key, scratch);
        }
    }

    pub fn automorphism_sub_ab<DataLhs: AsRef<[u8]>, DataRhs: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        lhs: &GLWECiphertext<DataLhs>,
        rhs: &AutomorphismKey<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) {
        Self::keyswitch_private::<_, _, 2>(self, rhs.p(), module, lhs, &rhs.key, scratch);
    }

    pub fn automorphism_sub_ab_inplace<DataRhs: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        rhs: &AutomorphismKey<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) {
        unsafe {
            let self_ptr: *mut GLWECiphertext<DataSelf> = self as *mut GLWECiphertext<DataSelf>;
            Self::keyswitch_private::<_, _, 2>(self, rhs.p(), module, &*self_ptr, &rhs.key, scratch);
        }
    }

    pub fn automorphism_sub_ba<DataLhs: AsRef<[u8]>, DataRhs: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        lhs: &GLWECiphertext<DataLhs>,
        rhs: &AutomorphismKey<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) {
        Self::keyswitch_private::<_, _, 3>(self, rhs.p(), module, lhs, &rhs.key, scratch);
    }

    pub fn automorphism_sub_ba_inplace<DataRhs: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        rhs: &AutomorphismKey<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) {
        unsafe {
            let self_ptr: *mut GLWECiphertext<DataSelf> = self as *mut GLWECiphertext<DataSelf>;
            Self::keyswitch_private::<_, _, 3>(self, rhs.p(), module, &*self_ptr, &rhs.key, scratch);
        }
    }
}
