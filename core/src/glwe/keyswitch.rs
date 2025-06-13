use backend::{
    FFT64, MatZnxDftOps, MatZnxDftScratch, Module, Scratch, VecZnxBig, VecZnxBigOps, VecZnxBigScratch, VecZnxDftAlloc,
    VecZnxDftOps, ZnxZero,
};

use crate::{FourierGLWECiphertext, GLWECiphertext, GLWESwitchingKey, Infos};

impl GLWECiphertext<Vec<u8>> {
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
        let res_dft: usize = FourierGLWECiphertext::bytes_of(module, basek, k_out, rank_out + 1);
        let in_size: usize = k_in.div_ceil(basek).div_ceil(digits);
        let out_size: usize = k_out.div_ceil(basek);
        let ksk_size: usize = k_ksk.div_ceil(basek);
        let ai_dft: usize = module.bytes_of_vec_znx_dft(rank_in, in_size);
        let vmp: usize = module.vmp_apply_tmp_bytes(out_size, in_size, in_size, rank_in, rank_out + 1, ksk_size)
            + module.bytes_of_vec_znx_dft(rank_in, in_size);
        let normalize: usize = module.vec_znx_big_normalize_tmp_bytes();
        return res_dft + ((ai_dft + vmp) | normalize);
    }

    pub fn keyswitch_from_fourier_scratch_space(
        module: &Module<FFT64>,
        basek: usize,
        k_out: usize,
        k_in: usize,
        k_ksk: usize,
        digits: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> usize {
        Self::keyswitch_scratch_space(module, basek, k_out, k_in, k_ksk, digits, rank_in, rank_out)
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

impl<DataSelf: AsRef<[u8]> + AsMut<[u8]>> GLWECiphertext<DataSelf> {
    pub fn keyswitch<DataLhs: AsRef<[u8]>, DataRhs: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        lhs: &GLWECiphertext<DataLhs>,
        rhs: &GLWESwitchingKey<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) {
        Self::keyswitch_private::<_, _, 0>(self, 0, module, lhs, rhs, scratch);
    }

    pub fn keyswitch_inplace<DataRhs: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        rhs: &GLWESwitchingKey<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) {
        unsafe {
            let self_ptr: *mut GLWECiphertext<DataSelf> = self as *mut GLWECiphertext<DataSelf>;
            self.keyswitch(&module, &*self_ptr, rhs, scratch);
        }
    }

    pub(crate) fn keyswitch_private<DataLhs: AsRef<[u8]>, DataRhs: AsRef<[u8]>, const OP: u8>(
        &mut self,
        apply_auto: i64,
        module: &Module<FFT64>,
        lhs: &GLWECiphertext<DataLhs>,
        rhs: &GLWESwitchingKey<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) {
        let basek: usize = self.basek();

        #[cfg(debug_assertions)]
        {
            assert_eq!(lhs.rank(), rhs.rank_in());
            assert_eq!(self.rank(), rhs.rank_out());
            assert_eq!(self.basek(), basek);
            assert_eq!(lhs.basek(), basek);
            assert_eq!(rhs.n(), module.n());
            assert_eq!(self.n(), module.n());
            assert_eq!(lhs.n(), module.n());
            assert!(
                scratch.available()
                    >= GLWECiphertext::keyswitch_scratch_space(
                        module,
                        self.basek(),
                        self.k(),
                        lhs.k(),
                        rhs.k(),
                        rhs.digits(),
                        rhs.rank_in(),
                        rhs.rank_out(),
                    )
            );
        }

        let cols_in: usize = rhs.rank_in();
        let cols_out: usize = rhs.rank_out() + 1;
        let digits: usize = rhs.digits();

        let (mut res_dft, scratch1) = scratch.tmp_vec_znx_dft(module, cols_out, rhs.size()); // Todo optimise
        let (mut ai_dft, scratch2) = scratch1.tmp_vec_znx_dft(module, cols_in, (lhs.size() + digits - 1) / digits);
        ai_dft.zero();
        {
            (0..digits).for_each(|di| {
                ai_dft.set_size((lhs.size() + di) / digits);

                // Small optimization for digits > 2
                // VMP produce some error e, and since we aggregate vmp * 2^{di * B}, then
                // we also aggregate ei * 2^{di * B}, with the largest error being ei * 2^{(digits-1) * B}.
                // As such we can ignore the last digits-2 limbs safely of the sum of vmp products.
                // It is possible to further ignore the last digits-1 limbs, but this introduce
                // ~0.5 to 1 bit of additional noise, and thus not chosen here to ensure that the same
                // noise is kept with respect to the ideal functionality.
                res_dft.set_size(rhs.size() - ((digits - di) as isize - 2).max(0) as usize);

                (0..cols_in).for_each(|col_i| {
                    module.vec_znx_dft(
                        digits,
                        digits - di - 1,
                        &mut ai_dft,
                        col_i,
                        &lhs.data,
                        col_i + 1,
                    );
                });

                if di == 0 {
                    module.vmp_apply(&mut res_dft, &ai_dft, &rhs.0.data, scratch2);
                } else {
                    module.vmp_apply_add(&mut res_dft, &ai_dft, &rhs.0.data, di, scratch2);
                }
            });
        }

        let mut res_big: VecZnxBig<&mut [u8], FFT64> = module.vec_znx_idft_consume(res_dft);

        module.vec_znx_big_add_small_inplace(&mut res_big, 0, &lhs.data, 0);

        (0..cols_out).for_each(|i| {
            if apply_auto != 0 {
                module.vec_znx_big_automorphism_inplace(apply_auto, &mut res_big, i);
            }

            match OP {
                1 => module.vec_znx_big_add_small_inplace(&mut res_big, i, &lhs.data, i),
                2 => module.vec_znx_big_sub_small_a_inplace(&mut res_big, i, &lhs.data, i),
                3 => module.vec_znx_big_sub_small_b_inplace(&mut res_big, i, &lhs.data, i),
                _ => {}
            }
            module.vec_znx_big_normalize(basek, &mut self.data, i, &res_big, i, scratch1);
        });
    }

    pub(crate) fn keyswitch_from_fourier<DataLhs: AsRef<[u8]>, DataRhs: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        lhs: &FourierGLWECiphertext<DataLhs, FFT64>,
        rhs: &GLWESwitchingKey<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) {
        let basek: usize = self.basek();

        #[cfg(debug_assertions)]
        {
            assert_eq!(lhs.rank(), rhs.rank_in());
            assert_eq!(self.rank(), rhs.rank_out());
            assert_eq!(self.basek(), basek);
            assert_eq!(lhs.basek(), basek);
            assert_eq!(rhs.n(), module.n());
            assert_eq!(self.n(), module.n());
            assert_eq!(lhs.n(), module.n());
            assert!(
                scratch.available()
                    >= GLWECiphertext::keyswitch_from_fourier_scratch_space(
                        module,
                        self.basek(),
                        self.k(),
                        lhs.k(),
                        rhs.k(),
                        rhs.digits(),
                        rhs.rank_in(),
                        rhs.rank_out(),
                    )
            );
        }

        let cols_in: usize = rhs.rank_in();
        let cols_out: usize = rhs.rank_out() + 1;

        // Buffer of the result of VMP in DFT
        let (mut res_dft, scratch1) = scratch.tmp_vec_znx_dft(module, cols_out, rhs.size()); // Todo optimise

        {
            let digits = rhs.digits();

            (0..digits).for_each(|di| {
                // (lhs.size() + di) / digits = (a - (digit - di - 1) + digit - 1) / digits
                let (mut ai_dft, scratch2) = scratch1.tmp_vec_znx_dft(module, cols_in, (lhs.size() + di) / digits);

                (0..cols_in).for_each(|col_i| {
                    module.vec_znx_dft_copy(
                        digits,
                        digits - 1 - di,
                        &mut ai_dft,
                        col_i,
                        &lhs.data,
                        col_i + 1,
                    );
                });

                if di == 0 {
                    module.vmp_apply(&mut res_dft, &ai_dft, &rhs.0.data, scratch2);
                } else {
                    module.vmp_apply_add(&mut res_dft, &ai_dft, &rhs.0.data, di, scratch2);
                }
            });
        }

        module.vec_znx_dft_add_inplace(&mut res_dft, 0, &lhs.data, 0);

        // Switches result of VMP outside of DFT
        let res_big: VecZnxBig<&mut [u8], FFT64> = module.vec_znx_idft_consume::<&mut [u8]>(res_dft);

        (0..cols_out).for_each(|i| {
            module.vec_znx_big_normalize(basek, &mut self.data, i, &res_big, i, scratch1);
        });
    }
}
