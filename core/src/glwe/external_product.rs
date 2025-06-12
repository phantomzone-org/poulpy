use backend::{
    FFT64, MatZnxDftOps, MatZnxDftScratch, Module, Scratch, VecZnxBig, VecZnxBigOps, VecZnxDftAlloc, VecZnxDftOps, VecZnxScratch,
};

use crate::{FourierGLWECiphertext, GGSWCiphertext, GLWECiphertext, Infos, div_ceil};

impl GLWECiphertext<Vec<u8>> {
    pub fn external_product_scratch_space(
        module: &Module<FFT64>,
        basek: usize,
        k_out: usize,
        k_in: usize,
        ggsw_k: usize,
        digits: usize,
        rank: usize,
    ) -> usize {
        let res_dft: usize = FourierGLWECiphertext::bytes_of(module, basek, k_out, rank);
        let in_size: usize = div_ceil(div_ceil(k_in, basek), digits);
        let out_size: usize = div_ceil(k_out, basek);
        let ggsw_size: usize = div_ceil(ggsw_k, basek);
        let vmp: usize = module.bytes_of_vec_znx_dft(rank + 1, in_size)
            + module.vmp_apply_tmp_bytes(
                out_size,
                in_size,
                in_size,  // rows
                rank + 1, // cols in
                rank + 1, // cols out
                ggsw_size,
            );
        let normalize: usize = module.vec_znx_normalize_tmp_bytes();
        res_dft + (vmp | normalize)
    }

    pub fn external_product_inplace_scratch_space(
        module: &Module<FFT64>,
        basek: usize,
        k_out: usize,
        ggsw_k: usize,
        digits: usize,
        rank: usize,
    ) -> usize {
        Self::external_product_scratch_space(module, basek, k_out, k_out, ggsw_k, digits, rank)
    }
}

impl<DataSelf: AsRef<[u8]> + AsMut<[u8]>> GLWECiphertext<DataSelf> {
    pub fn external_product<DataLhs: AsRef<[u8]>, DataRhs: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        lhs: &GLWECiphertext<DataLhs>,
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
                    >= GLWECiphertext::external_product_scratch_space(
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
        let digits: usize = rhs.digits();

        let (mut res_dft, scratch1) = scratch.tmp_vec_znx_dft(module, cols, rhs.size()); // Todo optimise
        let (mut a_dft, scratch2) = scratch1.tmp_vec_znx_dft(module, cols, (lhs.size() + digits - 1) / digits);

        {
            (0..digits).for_each(|di| {
                // (lhs.size() + di) / digits = (a - (digit - di - 1) + digit - 1) / digits
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
                    module.vec_znx_dft(digits, digits - 1 - di, &mut a_dft, col_i, &lhs.data, col_i);
                });

                if di == 0 {
                    module.vmp_apply(&mut res_dft, &a_dft, &rhs.data, scratch2);
                } else {
                    module.vmp_apply_add(&mut res_dft, &a_dft, &rhs.data, di, scratch2);
                }
            });
        }

        let res_big: VecZnxBig<&mut [u8], FFT64> = module.vec_znx_idft_consume(res_dft);

        (0..cols).for_each(|i| {
            module.vec_znx_big_normalize(basek, &mut self.data, i, &res_big, i, scratch1);
        });
    }

    pub fn external_product_inplace<DataRhs: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        rhs: &GGSWCiphertext<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) {
        unsafe {
            let self_ptr: *mut GLWECiphertext<DataSelf> = self as *mut GLWECiphertext<DataSelf>;
            self.external_product(&module, &*self_ptr, rhs, scratch);
        }
    }
}
