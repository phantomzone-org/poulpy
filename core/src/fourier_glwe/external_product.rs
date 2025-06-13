use backend::{
    FFT64, MatZnxDftOps, MatZnxDftScratch, Module, Scratch, VecZnxAlloc, VecZnxBig, VecZnxBigOps, VecZnxBigScratch,
    VecZnxDftAlloc, VecZnxDftOps,
};

use crate::{FourierGLWECiphertext, GGSWCiphertext, Infos};

impl FourierGLWECiphertext<Vec<u8>, FFT64> {
    // WARNING TODO: UPDATE
    pub fn external_product_scratch_space(
        module: &Module<FFT64>,
        basek: usize,
        _k_out: usize,
        k_in: usize,
        k_ggsw: usize,
        digits: usize,
        rank: usize,
    ) -> usize {
        let ggsw_size: usize = k_ggsw.div_ceil(basek);
        let res_dft: usize = module.bytes_of_vec_znx_dft(rank + 1, ggsw_size);
        let in_size: usize = k_in.div_ceil(basek).div_ceil(digits);
        let ggsw_size: usize = k_ggsw.div_ceil(basek);
        let vmp: usize = module.bytes_of_vec_znx_dft(rank + 1, in_size)
            + module.vmp_apply_tmp_bytes(ggsw_size, in_size, in_size, rank + 1, rank + 1, ggsw_size);
        let res_small: usize = module.bytes_of_vec_znx(rank + 1, ggsw_size);
        let normalize: usize = module.vec_znx_big_normalize_tmp_bytes();
        res_dft + (vmp | (res_small + normalize))
    }

    pub fn external_product_inplace_scratch_space(
        module: &Module<FFT64>,
        basek: usize,
        k_out: usize,
        k_ggsw: usize,
        digits: usize,
        rank: usize,
    ) -> usize {
        Self::external_product_scratch_space(module, basek, k_out, k_out, k_ggsw, digits, rank)
    }
}

impl<DataSelf: AsMut<[u8]> + AsRef<[u8]>> FourierGLWECiphertext<DataSelf, FFT64> {
    pub fn external_product<DataLhs: AsRef<[u8]>, DataRhs: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        lhs: &FourierGLWECiphertext<DataLhs, FFT64>,
        rhs: &GGSWCiphertext<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) {
        let basek: usize = self.basek();

        #[cfg(debug_assertions)]
        {
            assert_eq!(rhs.rank(), lhs.rank());
            assert_eq!(rhs.rank(), self.rank());
            assert_eq!(self.basek(), basek);
            assert_eq!(lhs.basek(), basek);
            assert_eq!(rhs.n(), module.n());
            assert_eq!(self.n(), module.n());
            assert_eq!(lhs.n(), module.n());
            assert!(
                scratch.available()
                    >= FourierGLWECiphertext::external_product_scratch_space(
                        module,
                        self.basek(),
                        self.k(),
                        lhs.k(),
                        rhs.k(),
                        rhs.digits(),
                        rhs.rank(),
                    )
            );
        }

        let cols: usize = rhs.rank() + 1;
        let digits = rhs.digits();

        // Space for VMP result in DFT domain and high precision
        let (mut res_dft, scratch1) = scratch.tmp_vec_znx_dft(module, cols, rhs.size());
        let (mut a_dft, scratch2) = scratch1.tmp_vec_znx_dft(module, cols, (lhs.size() + digits - 1) / digits);

        {
            (0..digits).for_each(|di| {
                a_dft.set_size((lhs.size() + di) / digits);

                // Small optimization for digits > 2
                // VMP produce some error e, and since we aggregate vmp * 2^{di * B}, then
                // we also aggregate ei * 2^{di * B}, with the largest error being ei * 2^{(digits-1) * B}.
                // As such we can ignore the last digits-2 limbs safely of the sum of vmp products.
                // It is possible to further ignore the last digits-1 limbs, but this introduce
                // ~0.5 to 1 bit of additional noise, and thus not chosen here to ensure that the same
                // noise is kept with respect to the ideal functionality.
                res_dft.set_size(rhs.size() - ((digits - di) as isize - 2).max(0) as usize);

                (0..cols).for_each(|col_i| {
                    module.vec_znx_dft_copy(digits, digits - 1 - di, &mut a_dft, col_i, &lhs.data, col_i);
                });

                if di == 0 {
                    module.vmp_apply(&mut res_dft, &a_dft, &rhs.data, scratch2);
                } else {
                    module.vmp_apply_add(&mut res_dft, &a_dft, &rhs.data, di, scratch2);
                }
            });
        }

        // VMP result in high precision
        let res_big: VecZnxBig<&mut [u8], FFT64> = module.vec_znx_idft_consume::<&mut [u8]>(res_dft);

        // Space for VMP result normalized
        let (mut res_small, scratch2) = scratch1.tmp_vec_znx(module, cols, rhs.size());
        (0..cols).for_each(|i| {
            module.vec_znx_big_normalize(basek, &mut res_small, i, &res_big, i, scratch2);
            module.vec_znx_dft(1, 0, &mut self.data, i, &res_small, i);
        });
    }

    pub fn external_product_inplace<DataRhs: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        rhs: &GGSWCiphertext<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) {
        unsafe {
            let self_ptr: *mut FourierGLWECiphertext<DataSelf, FFT64> = self as *mut FourierGLWECiphertext<DataSelf, FFT64>;
            self.external_product(&module, &*self_ptr, rhs, scratch);
        }
    }
}
