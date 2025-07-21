use backend::{Backend, Module, Scratch, VecZnxDftAllocBytes, VecZnxDftToVecZnxBig, VecZnxOps, ZnxZero};

use crate::{FourierGLWECiphertext, GLWEAutomorphismKey, GLWEAutomorphismKeyPrep, GLWECiphertext, Infos};

impl GLWEAutomorphismKey<Vec<u8>> {
    pub fn automorphism_scratch_space<B: Backend>(
        module: &Module<B>,
        basek: usize,
        k_out: usize,
        k_in: usize,
        k_ksk: usize,
        digits: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: VecZnxDftAllocBytes + VecZnxDftToVecZnxBig<B>,
    {
        let tmp_dft: usize = FourierGLWECiphertext::bytes_of(module, basek, k_in, rank);
        let tmp_idft: usize = FourierGLWECiphertext::bytes_of(module, basek, k_out, rank);
        let idft: usize = module.vec_znx_dft_to_vec_znx_big_scratch_space();
        let keyswitch: usize = GLWECiphertext::keyswitch_inplace_scratch_space(module, basek, k_out, k_ksk, digits, rank);
        tmp_dft + tmp_idft + idft + keyswitch
    }

    pub fn automorphism_inplace_scratch_space<B: Backend>(
        module: &Module<B>,
        basek: usize,
        k_out: usize,
        k_ksk: usize,
        digits: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: VecZnxDftAllocBytes + VecZnxDftToVecZnxBig<B>,
    {
        GLWEAutomorphismKey::automorphism_scratch_space(module, basek, k_out, k_out, k_ksk, digits, rank)
    }
}

impl<DataSelf: AsMut<[u8]> + AsRef<[u8]>> GLWEAutomorphismKey<DataSelf> {
    pub fn automorphism<DataLhs: AsRef<[u8]>, DataRhs: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        lhs: &GLWEAutomorphismKey<DataLhs>,
        rhs: &GLWEAutomorphismKeyPrep<DataRhs, B>,
        scratch: &mut Scratch,
    ) where
        Module<B>: VecZnxDftAllocBytes,
    {
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

        let p: i64 = lhs.p();
        let p_inv = module.galois_element_inv(p);

        (0..self.rank_in()).for_each(|col_i| {
            (0..self.rows()).for_each(|row_j| {
                let mut res_ct: GLWECiphertext<&mut [u8]> = self.at_mut(col_i, row_j);
                let lhs_ct: GLWECiphertext<&[u8]> = lhs.at(col_i, row_j);

                // Reverts the automorphism X^{-k}: (-pi^{-1}_{k}(s)a + s, a) to (-sa + pi_{k}(s), a)
                (0..cols_out).for_each(|i| {
                    module.vec_znx_automorphism(lhs.p(), &mut res_ct.data, i, &lhs_ct.data, i);
                });

                // Key-switch (-sa + pi_{k}(s), a) to (-pi^{-1}_{k'}(s)a + pi_{k}(s), a)
                res_ct.keyswitch_inplace(module, &rhs.key, scratch);

                // Applies back the automorphism X^{-k}: (-pi^{-1}_{k'}(s)a + pi_{k}(s), a) to (-pi^{-1}_{k'+k}(s)a + s, a)
                (0..cols_out).for_each(|i| {
                    module.vec_znx_automorphism_inplace(p_inv, &mut res_ct.data, i);
                });
            });
        });

        (self.rows().min(lhs.rows())..self.rows()).for_each(|row_i| {
            (0..self.rank_in()).for_each(|col_j| {
                self.at_mut(row_i, col_j).data.zero();
            });
        });

        self.p = (lhs.p * rhs.p) % (module.cyclotomic_order() as i64);
    }

    pub fn automorphism_inplace<DataRhs: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        rhs: &GLWEAutomorphismKeyPrep<DataRhs, B>,
        scratch: &mut Scratch,
    ) where
        Module<B>: VecZnxDftAllocBytes,
    {
        unsafe {
            let self_ptr: *mut GLWEAutomorphismKey<DataSelf> = self as *mut GLWEAutomorphismKey<DataSelf>;
            self.automorphism(&module, &*self_ptr, rhs, scratch);
        }
    }
}
