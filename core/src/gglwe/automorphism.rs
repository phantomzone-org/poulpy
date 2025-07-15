use backend::{FFT64, Module, Scratch, VecZnx, VecZnxDftOps, VecZnxOps, ZnxZero};

use crate::{FourierGLWECiphertext, GLWEAutomorphismKey, GLWECiphertext, GetRow, Infos, ScratchCore, SetRow};

impl GLWEAutomorphismKey<Vec<u8>, FFT64> {
    pub fn automorphism_scratch_space(
        module: &Module<FFT64>,
        basek: usize,
        k_out: usize,
        k_in: usize,
        k_ksk: usize,
        digits: usize,
        rank: usize,
    ) -> usize {
        let tmp_dft: usize = FourierGLWECiphertext::bytes_of(module, basek, k_in, rank);
        let tmp_idft: usize = FourierGLWECiphertext::bytes_of(module, basek, k_out, rank);
        let idft: usize = module.vec_znx_idft_tmp_bytes();
        let keyswitch: usize = GLWECiphertext::keyswitch_inplace_scratch_space(module, basek, k_out, k_ksk, digits, rank);
        tmp_dft + tmp_idft + idft + keyswitch
    }

    pub fn automorphism_inplace_scratch_space(
        module: &Module<FFT64>,
        basek: usize,
        k_out: usize,
        k_ksk: usize,
        digits: usize,
        rank: usize,
    ) -> usize {
        GLWEAutomorphismKey::automorphism_scratch_space(module, basek, k_out, k_out, k_ksk, digits, rank)
    }
}

impl<DataSelf: AsMut<[u8]> + AsRef<[u8]>> GLWEAutomorphismKey<DataSelf, FFT64> {
    pub fn automorphism<DataLhs: AsRef<[u8]>, DataRhs: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        lhs: &GLWEAutomorphismKey<DataLhs, FFT64>,
        rhs: &GLWEAutomorphismKey<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(
                self.rank_in(),
                lhs.rank_in(),
                "ksk_out input rank: {} != ksk_in input rank: {}",
                self.rank_in(),
                lhs.rank_in()
            );
            assert_eq!(
                lhs.rank_out(),
                rhs.rank_in(),
                "ksk_in output rank: {} != ksk_apply input rank: {}",
                self.rank_out(),
                rhs.rank_in()
            );
            assert_eq!(
                self.rank_out(),
                rhs.rank_out(),
                "ksk_out output rank: {} != ksk_apply output rank: {}",
                self.rank_out(),
                rhs.rank_out()
            );
            assert!(
                self.k() <= lhs.k(),
                "output k={} cannot be greater than input k={}",
                self.k(),
                lhs.k()
            )
        }

        let cols_out: usize = rhs.rank_out() + 1;

        (0..self.rank_in()).for_each(|col_i| {
            (0..self.rows()).for_each(|row_j| {
                let (mut tmp_idft_data, scratct1) = scratch.tmp_vec_znx_big(module, cols_out, self.size());

                {
                    let (mut tmp_dft, scratch2) = scratct1.tmp_fourier_glwe_ct(module, lhs.basek(), lhs.k(), lhs.rank());

                    // Extracts relevant row
                    lhs.get_row(module, row_j, col_i, &mut tmp_dft);

                    // Get a VecZnxBig from scratch space

                    // Switches input outside of DFT
                    (0..cols_out).for_each(|i| {
                        module.vec_znx_idft(&mut tmp_idft_data, i, &tmp_dft.data, i, scratch2);
                    });
                }

                // Consumes to small vec znx
                let mut tmp_idft_small_data: VecZnx<&mut [u8]> = tmp_idft_data.to_vec_znx_small();

                // Reverts the automorphis key from (-pi^{-1}_{k}(s)a + s, a) to (-sa + pi_{k}(s), a)
                (0..cols_out).for_each(|i| {
                    module.vec_znx_automorphism_inplace(lhs.p(), &mut tmp_idft_small_data, i);
                });

                // Wraps into ciphertext
                let mut tmp_idft: GLWECiphertext<&mut [u8]> = GLWECiphertext::<&mut [u8]> {
                    data: tmp_idft_small_data,
                    basek: self.basek(),
                    k: self.k(),
                };

                // Key-switch (-sa + pi_{k}(s), a) to (-pi^{-1}_{k'}(s)a + pi_{k}(s), a)
                tmp_idft.keyswitch_inplace(module, &rhs.key, scratct1);

                {
                    let (mut tmp_dft, _) = scratct1.tmp_fourier_glwe_ct(module, self.basek(), self.k(), self.rank());

                    // Applies back the automorphism X^{k}: (-pi^{-1}_{k'}(s)a + pi_{k}(s), a) -> (-pi^{-1}_{k'+k}(s)a + s, a)
                    // and switches back to DFT domain
                    (0..self.rank_out() + 1).for_each(|i| {
                        module.vec_znx_automorphism_inplace(lhs.p(), &mut tmp_idft.data, i);
                        module.vec_znx_dft(1, 0, &mut tmp_dft.data, i, &tmp_idft.data, i);
                    });

                    // Sets back the relevant row
                    self.set_row(module, row_j, col_i, &tmp_dft);
                }
            });
        });

        let (mut tmp_dft, _) = scratch.tmp_fourier_glwe_ct(module, self.basek(), self.k(), self.rank());
        tmp_dft.data.zero();

        (self.rows().min(lhs.rows())..self.rows()).for_each(|row_i| {
            (0..self.rank_in()).for_each(|col_j| {
                self.set_row(module, row_i, col_j, &tmp_dft);
            });
        });

        self.p = (lhs.p * rhs.p) % (module.cyclotomic_order() as i64);
    }

    pub fn automorphism_inplace<DataRhs: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        rhs: &GLWEAutomorphismKey<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) {
        unsafe {
            let self_ptr: *mut GLWEAutomorphismKey<DataSelf, FFT64> = self as *mut GLWEAutomorphismKey<DataSelf, FFT64>;
            self.automorphism(&module, &*self_ptr, rhs, scratch);
        }
    }
}
